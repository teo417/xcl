//! `XCL_GPU_TIMING=1`: per-submission GPU timestamp queries for DIRECT queues. COMPUTE queues
//! are left alone: with them timed too, a title stopped submitting as soon as they started to wait
//! for its direct queue (RTX 4080 SUPER, two runs of two).

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

const SLOTS: usize = 256;
const CAT_SLOTS: usize = 32768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cat {
    Chain = 0,
    TwoPass = 1,
    Draw = 2,
    Dispatch = 3,
    Copy = 4,
    IndirectDraw = 5,
    IndirectDispatch = 6,
    /// UAV and depth clears, resolves
    Clear = 7,
}

/// the names of [`Cat`] in its order
const CAT_NAMES: [&str; 8] = ["chain", "two_pass", "draw", "dispatch", "copy", "indirect_draw", "indirect_dispatch", "clear"];
const CATS: usize = CAT_NAMES.len();

static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static LISTS_TIMED: AtomicU64 = AtomicU64::new(0);
static GPU_TIME_US: AtomicU64 = AtomicU64::new(0);
static MAX_SUBMIT_US: AtomicU64 = AtomicU64::new(0);
static UNTIMED: AtomicU64 = AtomicU64::new(0);
static CAT_US: [AtomicU64; CATS] = [const { AtomicU64::new(0) }; CATS];
static CAT_COUNT: [AtomicU64; CATS] = [const { AtomicU64::new(0) }; CATS];
/// scopes not timed because no query pair was free
static CAT_DROPPED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// a scope is open on this thread: what is recorded inside it counts for it alone
    static CAT_OPEN: Cell<bool> = const { Cell::new(false) };
}

/// The pipeline a scope ran: its console shaders' label (`dred::label`), the category, and whether
/// it is a split program (its pieces or its whole form).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PipeKey {
    label: String,
    cat: usize,
    split: bool,
}

/// pipeline → (microseconds, scopes) since the last `take_pipeline_lines`
static PIPES: Mutex<Option<HashMap<PipeKey, (u64, u64)>>> = Mutex::new(None);

/// The `top` costliest pipelines since the last call, then what the others took together.
pub fn take_pipeline_lines(top: usize) -> Vec<String> {
    let map = PIPES.lock().unwrap().take().unwrap_or_default();
    pipeline_lines(map, top)
}

fn pipeline_lines(map: HashMap<PipeKey, (u64, u64)>, top: usize) -> Vec<String> {
    let mut all: Vec<_> = map.into_iter().collect();
    all.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.0.label.cmp(&b.0.label)));
    let rest = all.split_off(top.min(all.len()));
    let mut out: Vec<String> = all
        .iter()
        .map(|(k, (us, n))| format!("us={us} n={n} cat={} split={} {}", CAT_NAMES[k.cat], u8::from(k.split), k.label))
        .collect();
    if !rest.is_empty() {
        let (us, n) = rest.iter().fold((0, 0), |t, (_, v)| (t.0 + v.0, t.1 + v.1));
        out.push(format!("us={us} n={n} others={}", rest.len()));
    }
    out
}

/// `name=ms (scopes)` per category, and the scopes left untimed
pub fn cat_line() -> String {
    let mut out = String::new();
    for (i, name) in CAT_NAMES.iter().enumerate() {
        let _ = write!(out, "{name}={} ({}), ", CAT_US[i].load(Ordering::Relaxed) / 1000, CAT_COUNT[i].load(Ordering::Relaxed));
    }
    let _ = write!(out, "{} untimed", CAT_DROPPED.load(Ordering::Relaxed));
    out
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_TIMING").is_ok_and(|v| v.trim() == "1"))
}

/// `[submissions, lists_timed, gpu_time_us, max_submit_us, untimed_submissions]`; `gpu_time_us`
/// sums the direct queues, which may overlap on the GPU.
pub fn activity() -> [u64; 5] {
    [
        SUBMISSIONS.load(Ordering::Relaxed),
        LISTS_TIMED.load(Ordering::Relaxed),
        GPU_TIME_US.load(Ordering::Relaxed),
        MAX_SUBMIT_US.load(Ordering::Relaxed),
        UNTIMED.load(Ordering::Relaxed),
    ]
}

fn ticks_to_micros(t0: u64, t1: u64, freq: u64) -> u64 {
    if freq == 0 {
        return 0;
    }
    ((t1.saturating_sub(t0) as u128 * 1_000_000) / freq as u128) as u64
}

struct Slot {
    alloc: ID3D12CommandAllocator,
    start: ID3D12GraphicsCommandList,
    end: ID3D12GraphicsCommandList,
    val: u64,
    lists: usize,
    cat_slots: Vec<Timed>,
    in_flight: bool,
}

struct CatPool {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    free: Vec<usize>,
    list_pending: HashMap<usize, Vec<Timed>>,
}

/// A recorded scope: its query pair, category, and pipeline (0: none, as for copies and clears)
/// with whether that is a split program
#[derive(Clone, Copy)]
struct Timed {
    slot: usize,
    cat: Cat,
    pso: usize,
    split: bool,
}

static CAT_POOL: Mutex<Option<CatPool>> = Mutex::new(None);

impl CatPool {
    unsafe fn new(dev: &ID3D12Device) -> Option<Self> {
        let mut heap: Option<ID3D12QueryHeap> = None;
        dev.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: (CAT_SLOTS * 2) as u32,
                NodeMask: 0,
            },
            &mut heap,
        )
        .ok()?;
        let heap = heap?;
        let rb_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: (CAT_SLOTS * 2 * 8) as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &rb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut readback,
        )
        .ok()?;
        Some(Self {
            heap,
            readback: readback?,
            free: (0..CAT_SLOTS).collect(),
            list_pending: HashMap::new(),
        })
    }
}

pub struct CatScope {
    list: usize,
    timed: Timed,
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
}

impl CatScope {
    pub unsafe fn begin_raw(list_raw: *mut c_void, cat: Cat) -> Option<Self> {
        if !enabled() || list_raw.is_null() {
            return None;
        }
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&list_raw)?;
        Self::begin(list, cat)
    }

    pub unsafe fn begin(list: &ID3D12GraphicsCommandList, cat: Cat) -> Option<Self> {
        if !enabled() || CAT_OPEN.get() || list.GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT {
            return None;
        }
        let pso = match cat {
            Cat::Copy | Cat::Clear => 0,
            _ => super::bind::bound_pso(list.as_raw()),
        };
        let split = pso != 0 && (super::real::dispatch_chain::is_chain(pso) || super::real::dispatch_chain::is_chain(super::pso_async::resolve(pso)));
        let mut guard = CAT_POOL.lock().unwrap();
        if guard.is_none() {
            let mut dev: Option<ID3D12Device> = None;
            list.GetDevice(&mut dev).ok()?;
            *guard = CatPool::new(&dev?);
        }
        let pool = guard.as_mut()?;
        let Some(slot) = pool.free.pop() else {
            CAT_DROPPED.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let heap = pool.heap.clone();
        let readback = pool.readback.clone();
        drop(guard);
        list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, (slot * 2) as u32);
        CAT_OPEN.set(true);
        Some(Self {
            list: list.as_raw() as usize,
            timed: Timed { slot, cat, pso, split },
            heap,
            readback,
        })
    }
}

impl Drop for CatScope {
    fn drop(&mut self) {
        // (cleared last: the resolve below goes through the list's hooks too)
        struct Close;
        impl Drop for Close {
            fn drop(&mut self) {
                CAT_OPEN.set(false);
            }
        }
        let _close = Close;
        unsafe {
            let raw = self.list as *mut c_void;
            if let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&raw) {
                let q0 = (self.timed.slot * 2) as u32;
                list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, q0 + 1);
                list.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, q0, 2, &self.readback, (self.timed.slot * 16) as u64);
                if let Some(pool) = CAT_POOL.lock().unwrap().as_mut() {
                    pool.list_pending.entry(self.list).or_default().push(self.timed);
                    return;
                }
            }
            if let Some(pool) = CAT_POOL.lock().unwrap().as_mut() {
                pool.free.push(self.timed.slot);
            }
        }
    }
}

pub fn cat_drop_list(list_raw: usize) {
    if !enabled() {
        return;
    }
    if let Some(pool) = CAT_POOL.lock().unwrap().as_mut() {
        if let Some(gone) = pool.list_pending.remove(&list_raw) {
            pool.free.extend(gone.into_iter().map(|t| t.slot));
        }
    }
}

fn cat_take_lists(chunk: &[*mut c_void]) -> Vec<Timed> {
    let mut guard = CAT_POOL.lock().unwrap();
    let Some(pool) = guard.as_mut() else { return Vec::new() };
    let mut out = Vec::new();
    for &l in chunk {
        if let Some(v) = pool.list_pending.remove(&(l as usize)) {
            out.extend(v);
        }
    }
    out
}

unsafe fn cat_resolve_and_free(slots: &[Timed], freq: u64) {
    if slots.is_empty() {
        return;
    }
    let rb = {
        let guard = CAT_POOL.lock().unwrap();
        guard.as_ref().map(|p| p.readback.clone())
    };
    if let Some(rb) = rb {
        let mut ptr = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: CAT_SLOTS * 2 * 8,
        };
        if rb.Map(0, Some(&range), Some(&mut ptr)).is_ok() {
            let samples = std::slice::from_raw_parts(ptr as *const u64, CAT_SLOTS * 2);
            let mut pipes = PIPES.lock().unwrap();
            let pipes = pipes.get_or_insert_with(HashMap::new);
            for &Timed { slot: s, cat, pso, split } in slots {
                let us = ticks_to_micros(samples[s * 2], samples[s * 2 + 1], freq);
                CAT_US[cat as usize].fetch_add(us, Ordering::Relaxed);
                CAT_COUNT[cat as usize].fetch_add(1, Ordering::Relaxed);
                if pso != 0 {
                    let label = super::dred::label_of(pso)
                        .or_else(|| super::dred::label_of(super::pso_async::resolve(pso)))
                        .unwrap_or_else(|| format!("pso {pso:#x}"));
                    let total = pipes.entry(PipeKey { label, cat: cat as usize, split }).or_default();
                    total.0 += us;
                    total.1 += 1;
                }
            }
            rb.Unmap(0, Some(&D3D12_RANGE::default()));
        }
    }
    if let Some(pool) = CAT_POOL.lock().unwrap().as_mut() {
        pool.free.extend(slots.iter().map(|t| t.slot));
    }
}

struct QueueTimer {
    freq: u64,
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    fence: ID3D12Fence,
    next_val: u64,
    slots: Vec<Slot>,
}

/// one lock per queue: a submission the driver blocks (behind a queue wait) must not stop the
/// other queues, which may be the ones to signal it
type Timers = HashMap<usize, Arc<Mutex<QueueTimer>>>;

static TIMERS: Mutex<Option<Timers>> = Mutex::new(None);

impl QueueTimer {
    unsafe fn new(queue: &ID3D12CommandQueue) -> Option<Self> {
        let desc = queue.GetDesc();
        if desc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT {
            return None;
        }
        let freq = queue.GetTimestampFrequency().ok()?;
        if freq == 0 {
            return None;
        }
        let mut dev: Option<ID3D12Device> = None;
        queue.GetDevice(&mut dev).ok()?;
        let dev = dev?;
        let mut heap: Option<ID3D12QueryHeap> = None;
        dev.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: (SLOTS * 2) as u32,
                NodeMask: 0,
            },
            &mut heap,
        )
        .ok()?;
        let heap = heap?;
        let rb_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: (SLOTS * 2 * 8) as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &rb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut readback,
        )
        .ok()?;
        let readback = readback?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(desc.Type).ok()?;
            let start: ID3D12GraphicsCommandList = dev.CreateCommandList(0, desc.Type, &alloc, None).ok()?;
            let _ = start.Close();
            let end: ID3D12GraphicsCommandList = dev.CreateCommandList(0, desc.Type, &alloc, None).ok()?;
            let _ = end.Close();
            slots.push(Slot {
                alloc,
                start,
                end,
                val: 0,
                lists: 0,
                cat_slots: Vec::new(),
                in_flight: false,
            });
        }
        Some(Self {
            freq,
            heap,
            readback,
            fence,
            next_val: 1,
            slots,
        })
    }

    /// Record the two timestamp lists of a free slot: `(start, end, fence, value)`.
    unsafe fn prepare(&mut self, chunk: &[*mut c_void]) -> Option<(*mut c_void, *mut c_void, ID3D12Fence, u64)> {
        self.drain();
        let idx = self.slots.iter().position(|s| !s.in_flight)?;
        let q0 = (idx * 2) as u32;
        let val = self.next_val;
        let slot = &mut self.slots[idx];
        slot.alloc.Reset().ok()?;
        slot.start.Reset(&slot.alloc, None).ok()?;
        slot.start.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, q0);
        slot.start.Close().ok()?;
        slot.end.Reset(&slot.alloc, None).ok()?;
        slot.end.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, q0 + 1);
        slot.end.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, q0, 2, &self.readback, (idx * 2 * 8) as u64);
        slot.end.Close().ok()?;
        slot.val = val;
        slot.lists = chunk.len();
        slot.cat_slots = cat_take_lists(chunk);
        slot.in_flight = true;
        self.next_val += 1;
        Some((slot.start.as_raw(), slot.end.as_raw(), self.fence.clone(), val))
    }

    unsafe fn drain(&mut self) {
        let done = self.fence.GetCompletedValue();
        let any_ready = self.slots.iter().any(|s| s.in_flight && s.val <= done);
        if !any_ready {
            return;
        }
        let mut ptr = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: SLOTS * 2 * 8,
        };
        if self.readback.Map(0, Some(&range), Some(&mut ptr)).is_err() {
            return;
        }
        let samples = std::slice::from_raw_parts(ptr as *const u64, SLOTS * 2);
        for (i, s) in self.slots.iter_mut().enumerate() {
            if s.in_flight && s.val <= done {
                let us = ticks_to_micros(samples[i * 2], samples[i * 2 + 1], self.freq);
                SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
                LISTS_TIMED.fetch_add(s.lists as u64, Ordering::Relaxed);
                GPU_TIME_US.fetch_add(us, Ordering::Relaxed);
                MAX_SUBMIT_US.fetch_max(us, Ordering::Relaxed);
                if us >= 50_000 {
                    crate::xlog_info!("D3D12.X: slow GPU submit: {} ms ({} lists)", us / 1000, s.lists);
                }
                let cats = std::mem::take(&mut s.cat_slots);
                cat_resolve_and_free(&cats, self.freq);
                s.in_flight = false;
            }
        }
        self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
    }
}

/// Wrap `submit(chunk)` with start/end timestamp queries when `XCL_GPU_TIMING=1`.
pub unsafe fn wrap_submit(queue_raw: *mut c_void, chunk: &[*mut c_void], submit: impl FnOnce(&[*mut c_void])) {
    if !enabled() || queue_raw.is_null() || chunk.is_empty() {
        submit(chunk);
        return;
    }
    let Some(queue) = ID3D12CommandQueue::from_raw_borrowed(&queue_raw) else {
        submit(chunk);
        return;
    };
    let timer = {
        let mut guard = TIMERS.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if !map.contains_key(&(queue_raw as usize)) {
            if let Some(t) = QueueTimer::new(queue) {
                map.insert(queue_raw as usize, Arc::new(Mutex::new(t)));
            }
        }
        map.get(&(queue_raw as usize)).cloned()
    };
    let Some(timer) = timer else {
        UNTIMED.fetch_add(1, Ordering::Relaxed);
        submit(chunk);
        return;
    };
    // held until the fence is signalled: two threads submitting to one queue must signal the
    // values in order, or a slot is reused while the GPU still runs it
    let mut timer = timer.lock().unwrap();
    let prepared = timer.prepare(chunk);
    let Some((start_raw, end_raw, fence, val)) = prepared else {
        drop(timer);
        UNTIMED.fetch_add(1, Ordering::Relaxed);
        submit(chunk);
        return;
    };
    let mut wrapped = Vec::with_capacity(chunk.len() + 2);
    wrapped.push(start_raw);
    wrapped.extend_from_slice(chunk);
    wrapped.push(end_raw);
    submit(&wrapped);
    let _ = queue.Signal(&fence, val);
    drop(timer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_lines_are_the_costliest_first_and_the_rest_summed() {
        let key = |label: &str, cat: Cat, split| PipeKey { label: label.into(), cat: cat as usize, split };
        let map = HashMap::from([
            (key("cs b", Cat::Dispatch, false), (10, 1)),
            (key("cs a", Cat::Chain, true), (500, 3)),
            (key("ms c ps d", Cat::TwoPass, false), (40, 2)),
            (key("cs e", Cat::IndirectDispatch, false), (5, 4)),
        ]);
        assert_eq!(
            pipeline_lines(map, 2),
            ["us=500 n=3 cat=chain split=1 cs a", "us=40 n=2 cat=two_pass split=0 ms c ps d", "us=15 n=5 others=2"]
        );
        assert!(pipeline_lines(HashMap::new(), 2).is_empty());
    }

    #[test]
    fn ticks_to_micros_computes_exact_duration() {
        assert_eq!(ticks_to_micros(1_000_000, 2_500_000, 10_000_000), 150_000);
        assert_eq!(ticks_to_micros(500, 100, 10_000_000), 0);
        assert_eq!(ticks_to_micros(0, 100, 0), 0);
    }
}
