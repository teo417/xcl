//! PSOs for a split compute entry. The first piece is the real PSO returned
//! to the caller. Other pieces and the lazily built whole form belong to it.
use super::*;
use crate::gpu::{dxil::kernel::DispatchChain, recomp::pso};
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, OnceLock};

static PROGRAMS: AtomicU64 = AtomicU64::new(0);
static PIECES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// slabs dispatched (`pso::dispatch_slabs`)
static SLABS: AtomicU64 = AtomicU64::new(0);
/// scratch alive now, the most there was, and the most a dispatch would have needed at once (bytes)
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static WANTED: AtomicU64 = AtomicU64::new(0);
/// ExecuteIndirects run as chains, those of them that ran the whole form instead (more slabs than
/// rounds), and the slabs lost when there was no whole form yet
static INDIRECT: AtomicU64 = AtomicU64::new(0);
static INDIRECT_WHOLE: AtomicU64 = AtomicU64::new(0);
static INDIRECT_LOST: AtomicU64 = AtomicU64::new(0);
/// ExecuteIndirects that ran the whole form (not chainable)
static WHOLE_INDIRECT: AtomicU64 = AtomicU64::new(0);
/// recordings made (`RECORDINGS`)
static RECORDED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn log_counts() {
    let mib = |a: &AtomicU64| a.load(Ordering::Relaxed) >> 20;
    let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
    crate::xlog_info!("compute split: {} programs split, {} pieces built, {} chained dispatches issued in {} slabs; scratch {} MiB (most {}, largest dispatch {} in one slab); ExecuteIndirect: {} chained ({} of them whole, {} slabs lost), {} whole",
        n(&PROGRAMS), n(&PIECES), n(&DISPATCHES), n(&SLABS), mib(&LIVE), mib(&PEAK), mib(&WANTED),
        n(&INDIRECT), n(&INDIRECT_WHOLE), n(&INDIRECT_LOST), n(&WHOLE_INDIRECT));
    // what holds the recordings still kept: they go when their PC allocator is reset or, put aside, is free
    let (keys, held) = {
        let r = RECORDINGS.lock().unwrap();
        let keys: Vec<(usize, usize)> = r.iter().map(|(&a, v)| (a, v.len())).collect();
        let held = keys.iter().map(|&(_, n)| n).sum::<usize>();
        (keys, held)
    };
    let [users, gpu, free, busy, idle, unknown] = super::super::alloc_pool::holds(&keys);
    crate::xlog_info!("compute split: {held} of {} recordings held, under {} PC allocators: put aside with a list on it {users}, put aside with submits on the GPU {gpu}, put aside and free {free}; in use and busy {busy}, in use and free {idle}; not the pool's {unknown}",
        n(&RECORDED), keys.len());
}

/// The scratch a dispatch uses at once, `XCL_SPLIT_SCRATCH_MB` (256): more groups than fit run in
/// slabs. A retail title's split programs need 190 KB for each group (median), 3 GB for 960x540
/// pixels in groups of 8x4.
fn scratch_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(256u64).min(4096) << 20)
}

struct PipelineBuilder {
    device: ID3D12Device,
    create: PsoFn,
    signature: ID3D12RootSignature,
    flags: D3D12_PIPELINE_STATE_FLAGS,
    node: u32,
    record: u64,
}

impl PipelineBuilder {
    unsafe fn pipeline(&self, dxil: &[u8], iid: &GUID) -> Option<ID3D12PipelineState> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { NodeMask: self.node, Flags: self.flags, ..Default::default() };
        let mut raw = std::ptr::null_mut();
        let (hr, _, _) = compute_container(self.device.as_raw(), self.create, &desc, self.signature.as_raw(),
            dxil, self.record, iid, &mut raw, ComputeBuild::Chain);
        (hr >= 0 && !raw.is_null()).then(|| ID3D12PipelineState::from_raw(raw))
    }
}

struct Chain {
    builder: Arc<PipelineBuilder>,
    tail: Vec<ID3D12PipelineState>,
    words: u32,
    whole_code: Vec<u8>,
    whole: OnceLock<ID3D12PipelineState>,
    build_whole: Mutex<()>,
    /// threads other than `want_whole`'s waiting for the whole form
    waiting: std::sync::atomic::AtomicUsize,
    /// the slabs of the last of its ExecuteIndirects read back, and whether one was read back yet
    last: std::sync::atomic::AtomicU32,
    measured: std::sync::atomic::AtomicBool,
    /// its whole form was asked for in the background / a lost slab was reported
    wanted: std::sync::atomic::AtomicBool,
    warned: std::sync::atomic::AtomicBool,
}

impl Chain {
    unsafe fn whole(&self) -> Result<ID3D12PipelineState, String> {
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        // `want_whole`'s build may hold `build_whole` while it waits for its turn
        // (`big_compile_turn`): while this thread waits, that build is one the title waits for
        self.waiting.fetch_add(1, Ordering::AcqRel);
        let built = self.build_whole_form();
        self.waiting.fetch_sub(1, Ordering::AcqRel);
        built
    }

    unsafe fn build_whole_form(&self) -> Result<ID3D12PipelineState, String> {
        let _guard = self.build_whole.lock().unwrap();
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        let p = self.builder.pipeline(&self.whole_code, &pc_iid(Kind::PipelineState)).ok_or("whole compute pipeline")?;
        let _ = self.whole.set(p.clone());
        Ok(p)
    }

    /// The whole form built on another thread, for a later ExecuteIndirect with more slabs than rounds
    fn want_whole(self: &Arc<Self>) {
        if !self.wanted.swap(true, Ordering::Relaxed) {
            let len = self.whole_code.len();
            let lane = if len >= GIANT_DRIVER_COMPILE { 2 } else if len >= huge_pso_bytes() { 1 } else { 0 };
            WHOLE_BUILDS[lane].push(Arc::downgrade(self), |chain| unsafe {
                // (a program released before its turn is not built)
                let Some(chain) = chain.upgrade() else { return };
                let waited = chain.clone();
                let built = super::super::pso_async::build_waited_on(move || waited.waiting.load(Ordering::Acquire) > 0, || {
                    if chain.whole.get().is_some() { Ok(()) } else { chain.build_whole_form().map(drop) }
                });
                if let Err(e) = built { crate::xlog_warn!("D3D12.X: whole form of a split program: {e}"); }
            });
        }
    }
}

/// The thread of `want_whole`: an `xcl-pso*` one, so a huge whole form waits for its turn
/// (`driver_compile`) as the workers' builds do
fn whole_thread() -> std::thread::Builder {
    std::thread::Builder::new().name("xcl-pso-chain-whole".into())
}

/// Work done one item after another on one thread of `whole_thread`'s, started when there is
/// some and none is running.
struct InTurn<T> {
    queue: Mutex<(std::collections::VecDeque<T>, bool)>,
    /// threads started so far
    started: AtomicU64,
}

impl<T: Send + 'static> InTurn<T> {
    const fn new() -> Self {
        Self { queue: Mutex::new((std::collections::VecDeque::new(), false)), started: AtomicU64::new(0) }
    }

    fn push(&'static self, item: T, run: fn(T)) {
        let mut q = self.queue.lock().unwrap();
        q.0.push_back(item);
        if std::mem::replace(&mut q.1, true) { return; }
        drop(q);
        self.started.fetch_add(1, Ordering::Relaxed);
        let spawned = whole_thread().spawn(move || loop {
            let next = {
                let mut q = self.queue.lock().unwrap();
                let next = q.0.pop_front();
                q.1 = next.is_some();
                next
            };
            match next { Some(item) => run(item), None => return }
        });
        // (the next push tries again)
        if spawned.is_err() { self.queue.lock().unwrap().1 = false; }
    }
}

/// The whole forms `want_whole` asked for, one lane for each slot set of `driver_compile` (small
/// ones are never gated, huge and giant ones take their turns): a thread of their own for each
/// would wait for its turn as well, and a title had 200 such threads, 199 of them waiting (#2404).
static WHOLE_BUILDS: [InTurn<std::sync::Weak<Chain>>; 3] = [InTurn::new(), InTurn::new(), InTurn::new()];

static CHAINS: LazyLock<Mutex<HashMap<usize, Arc<Chain>>>> = LazyLock::new(Default::default);

pub(super) unsafe fn build(this: *mut c_void, create: PsoFn, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    iid: *const GUID, result: *mut *mut c_void, record: u64, whole: &[u8], samplers: &[pso::StaticSampler],
    plan: Arc<DispatchChain>) -> Option<usize> {
    if result.is_null() || iid.is_null() || plan.pieces.len() < 2 { return None; }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let signature = pso::dispatch_chain_root_signature(dev, samplers).ok()?;
    let rs = signature.as_raw() as usize;
    let builder = Arc::new(PipelineBuilder { device: dev.clone(), create, signature,
        flags: desc.Flags, node: desc.NodeMask, record });
    let jobs = (0..plan.pieces.len()).map(|i| {
        let plan = plan.clone();
        let builder = builder.clone();
        let iid = if i == 0 { *iid } else { pc_iid(Kind::PipelineState) };
        Box::new(move || builder.pipeline(&plan.pieces[i], &iid).map_or(0, |p| p.into_raw() as usize)) as super::super::pso_async::Job
    }).collect();
    // Take ownership of every success, including when a different piece failed.
    let built: Vec<_> = super::super::pso_async::build_pieces(jobs).into_iter()
        .map(|p| (p != 0).then(|| ID3D12PipelineState::from_raw(p as *mut c_void))).collect();
    let mut pieces = built.into_iter().collect::<Option<Vec<_>>>()?.into_iter();
    let first = pieces.next()?;
    let chain = Arc::new(Chain { builder, tail: pieces.collect(), words: plan.dwords_per_group,
        whole_code: whole.to_vec(), whole: OnceLock::new(), build_whole: Mutex::new(()), waiting: Default::default(),
        last: Default::default(), measured: Default::default(), wanted: Default::default(), warned: Default::default() });
    PROGRAMS.fetch_add(1, Ordering::Relaxed);
    PIECES.fetch_add(plan.pieces.len() as u64, Ordering::Relaxed);
    *result = first.into_raw();
    CHAINS.lock().unwrap().insert(*result as usize, chain);
    Some(rs)
}

pub(super) fn forget(pso: usize) {
    let gone = CHAINS.lock().unwrap().remove(&pso);
    drop(gone); // releasing a tail PSO also goes through forget_pso
}

pub(in crate::hle::d3d12x) unsafe fn whole(pso: usize) -> Result<Option<ID3D12PipelineState>, String> {
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    if chain.is_some() { WHOLE_INDIRECT.fetch_add(1, Ordering::Relaxed); }
    chain.map(|c| c.whole()).transpose()
}

struct Scratch {
    buffer: ID3D12Resource,
    bytes: u64,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

struct Recording {
    scratch: Arc<Scratch>,
    first: ID3D12PipelineState,
    chain: Arc<Chain>,
    indirect: Option<Indirect>,
}

/// The commands of a chained ExecuteIndirect (`pso::record_slicer`), read when the GPU is done
/// with them (the allocator is reset): with more slabs than rounds, the next ExecuteIndirect of
/// the program records more, and the whole form is built for it
struct Indirect {
    commands: ID3D12Resource,
    bytes: u64,
    at: u64,
    rounds: u32,
    /// the whole form ran instead (the slicer gave it the arguments); `checked`: the chain was
    /// recorded and the title's predication was not set (it may skip the slicer, whose words are
    /// then no count)
    whole: bool,
    checked: bool,
}

impl Recording {
    fn read_back(&self) {
        let Some(i) = self.indirect.as_ref().filter(|i| i.checked) else { return };
        let mut p = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: i.at as usize, End: i.at as usize + 16 };
        if unsafe { i.commands.Map(0, Some(&range), Some(&mut p)) }.is_err() { return; }
        let [partial, _, slabs, _] = unsafe { std::ptr::read_volatile((p as *const u8).add(i.at as usize) as *const [u32; 4]) };
        unsafe { i.commands.Unmap(0, Some(&D3D12_RANGE::default())) };
        self.chain.last.store(slabs, Ordering::Relaxed);
        self.chain.measured.store(true, Ordering::Relaxed);
        if partial == 0 { return; }
        if i.whole {
            INDIRECT_WHOLE.fetch_add(1, Ordering::Relaxed);
            return;
        }
        INDIRECT_LOST.fetch_add(u64::from(slabs.saturating_sub(i.rounds)), Ordering::Relaxed);
        if !self.chain.warned.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect of a split program ({:#x}) had {slabs} slabs for {} rounds: {} lost; its whole form is built for the next ones",
                self.first.as_raw() as usize, i.rounds, slabs - i.rounds);
        }
        self.chain.want_whole();
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.read_back();
    }
}

static RECORDINGS: LazyLock<Mutex<HashMap<usize, Vec<Arc<Recording>>>>> = LazyLock::new(Default::default);
/// Recordings of dispatches on a list with no title allocator (a bundle, or a list whose
/// allocator XCL cannot see): their scratch lives until the title resets that list, which D3D12
/// only allows after the GPU has finished the commands recorded in it.
static LIST_RECORDINGS: LazyLock<Mutex<HashMap<usize, Vec<Arc<Recording>>>>> = LazyLock::new(Default::default);
/// list → the allocator it records on and the scratch its chains share: they are dispatched one
/// after another, with a barrier between them (the recordings keep it alive)
static SHARED: LazyLock<Mutex<HashMap<usize, (usize, std::sync::Weak<Scratch>)>>> = LazyLock::new(Default::default);

/// `list` begins another recording, which the GPU may run beside the last one: a Reset, or a
/// list created at the address of a released one (#2588). Also when it is released.
pub(in crate::hle::d3d12x) fn list_reset(list: usize) {
    SHARED.lock().unwrap().remove(&list);
    LIST_RECORDINGS.lock().unwrap().remove(&list);
    release_constants(list);
}

/// `to` records right after `from` on the same allocator: its chains go on in `from`'s scratch.
pub(in crate::hle::d3d12x) fn list_follows(from: usize, to: usize) {
    let mut shared = SHARED.lock().unwrap();
    if let Some(e) = shared.get(&from).cloned() {
        shared.insert(to, e);
    }
}

#[cfg(test)]
pub(in crate::hle::d3d12x) fn note_shared_for_test(list: usize, allocator: usize) {
    SHARED.lock().unwrap().insert(list, (allocator, std::sync::Weak::new()));
}

#[cfg(test)]
pub(in crate::hle::d3d12x) fn shares_scratch_for_test(list: usize) -> bool {
    SHARED.lock().unwrap().contains_key(&list)
}

/// The scratch of all recordings alive together, `XCL_SPLIT_SCRATCH_TOTAL_MB` (384). A recording
/// keeps its scratch until its allocator is reset, and a title that records ahead of the GPU has
/// many: with a bound on one dispatch alone a retail title's world took 1.2 GB and went on growing.
fn scratch_total() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_TOTAL_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(384u64) << 20)
}

/// A recording gets at least this much (or one group's worth) however much the others hold.
const LEAST: u64 = 2 << 20;

/// How much scratch a dispatch of `needed` bytes (`group` for one group) runs with at once: the
/// list's own buffer (`had` bytes) when that is enough or no more may be made, else a new one of
/// the bytes given (second: make one).
fn scratch_bytes(needed: u64, group: u64, budget: u64, had: Option<u64>, live: u64, total: u64) -> (u64, bool) {
    let want = needed.min(budget).max(group);
    let room = total.saturating_sub(live);
    match had {
        Some(h) if h >= want || (room < want && h >= group) => (h, false),
        _ => (want.min(room.max(LEAST)).max(group), true),
    }
}

/// A scratch for a chain of `needed` bytes recorded on `list` with `allocator`, and whether a
/// chain before it on the list used it.
unsafe fn scratch_for(dev: &ID3D12Device, list: usize, allocator: usize, needed: u64, group: u64) -> Option<(Arc<Scratch>, bool)> {
    let had = SHARED.lock().unwrap().get(&list).filter(|s| s.0 == allocator).and_then(|s| s.1.upgrade());
    let (bytes, make) = scratch_bytes(needed, group, scratch_budget(), had.as_ref().map(|s| s.bytes), LIVE.load(Ordering::Relaxed), scratch_total());
    if !make {
        return had.map(|s| (s, true));
    }
    let s = Arc::new(Scratch { buffer: scratch_buffer(dev, bytes)?, bytes });
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
    SHARED.lock().unwrap().insert(list, (allocator, Arc::downgrade(&s)));
    Some((s, false))
}

/// Command buffers of chains whose allocator was reset, by size, for the next chains: a retail
/// title's world records hundreds a second, and creating them took its recording threads 0.7 ms
/// each (48 s for 65,000 in a 240 s run). One pool for all allocators: the GPU is done with a
/// buffer once its allocator is reset, and the next to record is mostly another one (a pool per
/// allocator reused 15%, this one 90%).
static CMD_POOL: LazyLock<Mutex<BTreeMap<u64, Vec<ID3D12Resource>>>> = LazyLock::new(Default::default);
/// bytes the pool keeps at most (system memory)
const CMD_POOL_MAX_BYTES: u64 = 512 << 20;
static CMD_POOL_BYTES: AtomicU64 = AtomicU64::new(0);
static CMD_ALLOCS: AtomicU64 = AtomicU64::new(0);
static CMD_REUSES: AtomicU64 = AtomicU64::new(0);
/// time the recording threads spent creating new command buffers
static CMD_ALLOC_MICROS: AtomicU64 = AtomicU64::new(0);

/// `XCL_CHAIN_CMD_POOL=0`: every chained ExecuteIndirect gets a new buffer, as before the pool
fn cmd_pool_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_CHAIN_CMD_POOL").map_or(true, |v| v.trim() != "0"))
}

/// A committed buffer takes whole 64 KiB pages, so it is made that big and fits more chains.
fn pooled_size(needed: u64) -> u64 {
    needed.max(1).next_multiple_of(64 << 10)
}

/// The smallest pooled buffer of at least `needed` bytes, and its size.
fn take_cmd_buffer(needed: u64) -> Option<(u64, ID3D12Resource)> {
    let mut pool = CMD_POOL.lock().unwrap();
    let (&bytes, buffers) = pool.range_mut(needed..).next()?;
    let buffer = buffers.pop()?;
    if buffers.is_empty() {
        pool.remove(&bytes);
    }
    CMD_POOL_BYTES.fetch_sub(bytes, Ordering::Relaxed);
    Some((bytes, buffer))
}

fn return_cmd_buffer(bytes: u64, buffer: ID3D12Resource) {
    let mut pool = CMD_POOL.lock().unwrap();
    if !cmd_pool_enabled() || CMD_POOL_BYTES.load(Ordering::Relaxed) + bytes > CMD_POOL_MAX_BYTES {
        return;
    }
    CMD_POOL_BYTES.fetch_add(bytes, Ordering::Relaxed);
    pool.entry(bytes).or_default().push(buffer);
}

/// Zero the words `read_back` takes: a pooled buffer holds an earlier chain's, and a list that is
/// recorded but never run would have them read as its own.
unsafe fn clear_read_back(buffer: &ID3D12Resource, at: u64) -> Result<(), String> {
    let mut p = std::ptr::null_mut();
    buffer.Map(0, Some(&D3D12_RANGE::default()), Some(&mut p)).map_err(|e| format!("mapping a chain's command buffer: {e}"))?;
    std::ptr::write_bytes((p as *mut u8).add(at as usize), 0, 16);
    buffer.Unmap(0, Some(&D3D12_RANGE { Begin: at as usize, End: at as usize + 16 }));
    Ok(())
}

pub(in crate::hle::d3d12x) fn cmd_buffer_stats() -> (u64, u64, u64, u64, u64) {
    (
        LIVE.load(Ordering::Relaxed),
        CMD_POOL_BYTES.load(Ordering::Relaxed),
        CMD_ALLOCS.load(Ordering::Relaxed),
        CMD_REUSES.load(Ordering::Relaxed),
        CMD_ALLOC_MICROS.load(Ordering::Relaxed),
    )
}

pub(in crate::hle::d3d12x) fn allocator_reset(allocator: usize) {
    if let Some(recordings) = RECORDINGS.lock().unwrap().remove(&allocator) {
        for r in recordings {
            match Arc::try_unwrap(r) {
                Ok(mut rec) => {
                    rec.read_back();
                    if let Some(ind) = rec.indirect.take() {
                        return_cmd_buffer(ind.bytes, ind.commands);
                    }
                }
                Err(r) => drop(r),
            }
        }
    }
    release_constants(allocator);
}

/// The slab constants of an allocator's (or an XCL-owned list's) recordings are free to reuse.
fn release_constants(key: usize) {
    let constants = CONSTANTS.lock().unwrap().remove(&key);
    if let Some((blocks, _)) = constants {
        let gone = keep_free_constants(&mut FREE_CONSTANTS.lock().unwrap(), blocks);
        drop(gone); // (outside the lock)
    }
}

/// The blocks of `Constants` kept for use again: those of `CONSTANTS_BLOCK` bytes, up to
/// `FREE_CONSTANTS_BYTES` of them; the others are returned to be released. Without a bound a
/// burst of big slab constants stayed mapped for the rest of the process.
fn keep_free_constants(free: &mut Vec<Constants>, blocks: Vec<Constants>) -> Vec<Constants> {
    let mut kept: u64 = free.iter().map(|b| b.bytes).sum();
    let mut gone = Vec::new();
    for b in blocks {
        if b.bytes == CONSTANTS_BLOCK && kept + b.bytes <= FREE_CONSTANTS_BYTES {
            kept += b.bytes;
            free.push(b);
        } else {
            gone.push(b);
        }
    }
    gone
}

/// Memory the CPU writes the slabs' constants of direct dispatches in (`pso::record_dispatch_chain`):
/// an allocator's recordings use a block of it until the allocator is reset, then other ones do.
struct Constants {
    buffer: ID3D12Resource,
    device: usize,
    cpu: usize,
    bytes: u64,
}

/// allocator → its blocks and the bytes used of the last one
static CONSTANTS: LazyLock<Mutex<HashMap<usize, (Vec<Constants>, u64)>>> = LazyLock::new(Default::default);
static FREE_CONSTANTS: Mutex<Vec<Constants>> = Mutex::new(Vec::new());
const CONSTANTS_BLOCK: u64 = 64 << 10;
/// the most bytes of blocks kept for use again (`keep_free_constants`)
const FREE_CONSTANTS_BYTES: u64 = 8 << 20;

/// `bytes` of `Constants` for a dispatch recorded with `allocator`: their CPU and GPU addresses.
unsafe fn constants_for(dev: &ID3D12Device, allocator: usize, bytes: u64) -> Option<(*mut u8, u64)> {
    let mut all = CONSTANTS.lock().unwrap();
    let (blocks, used) = all.entry(allocator).or_default();
    if blocks.last().is_none_or(|b| *used + bytes > b.bytes) {
        let device = dev.as_raw() as usize;
        let free = {
            let mut f = FREE_CONSTANTS.lock().unwrap();
            f.iter().position(|b| b.device == device && b.bytes >= bytes).map(|k| f.swap_remove(k))
        };
        blocks.push(match free { Some(b) => b, None => constants_block(dev, bytes.max(CONSTANTS_BLOCK))? });
        *used = 0;
    }
    let b = blocks.last()?;
    let at = *used;
    *used += bytes;
    Some(((b.cpu as *mut u8).add(at as usize), b.buffer.GetGPUVirtualAddress() + at))
}

unsafe fn constants_block(dev: &ID3D12Device, bytes: u64) -> Option<Constants> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut resource).ok()?;
    let buffer = resource?;
    let mut cpu = std::ptr::null_mut();
    buffer.Map(0, Some(&D3D12_RANGE::default()), Some(&mut cpu)).ok()?;
    Some(Constants { buffer, device: dev.as_raw() as usize, cpu: cpu as usize, bytes })
}

#[cfg(test)]
pub(in crate::hle::d3d12x) fn note_recordings_for_test(allocator: usize) {
    RECORDINGS.lock().unwrap().entry(allocator).or_default();
}

#[cfg(test)]
pub(in crate::hle::d3d12x) fn holds_recordings_for_test(allocator: usize) -> bool {
    RECORDINGS.lock().unwrap().contains_key(&allocator)
}

unsafe fn scratch_buffer(dev: &ID3D12Device, bytes: u64) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut resource).ok()?;
    resource
}

/// Called with the original six ABI views bound. The extended signature is
/// common to every piece and the whole fallback. Choose before recording any
/// part of the dispatch, including when scratch allocation fails.
pub(in crate::hle::d3d12x) unsafe fn dispatch(list: &ID3D12GraphicsCommandList, pso: usize, allocator: usize, grid: [u32; 3]) -> Result<(), String> {
    let [x, y, z] = grid;
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { list.Dispatch(x, y, z); return Ok(()) };
    if grid.contains(&0) { return Ok(()); }
    // A title Reset can switch to a different PC allocator while this
    // recording is still submitted on the old one. Retain by backing.
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    let group = u64::from(chain.words) * 4;
    let needed = grid.into_iter().fold(group, |s, n| s.saturating_mul(n as u64));
    WANTED.fetch_max(needed, Ordering::Relaxed);
    let list_key = list.as_raw() as usize;
    // No title allocator (a bundle, or a list whose allocator XCL cannot see): XCL owns the
    // scratch and keeps it until the title resets the list (D3D12 allows a reset only after the
    // GPU has finished the commands recorded in it), so the pieces run instead of the whole form.
    let owned = allocator == 0 || list.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE;
    let scratch = scratch_for(&chain.builder.device, list_key, if owned { 0 } else { allocator }, needed, group);
    let most = scratch.as_ref().map_or(1, |(s, _)| (s.bytes / group).max(1));
    let slabs = pso::dispatch_slabs(grid, most);
    let constants = scratch.as_ref()
        .and_then(|_| constants_for(&chain.builder.device, if owned { list_key } else { allocator }, slabs.len() as u64 * u64::from(pso::CHAIN_COMMAND_BYTES)));
    let (Some((scratch, used)), Some(constants)) = (scratch, constants) else {
        list.SetPipelineState(&chain.whole()?);
        list.Dispatch(x, y, z);
        return Ok(());
    };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    let held = Arc::new(Recording { scratch, first, chain, indirect: None });
    RECORDED.fetch_add(1, Ordering::Relaxed);
    if owned {
        LIST_RECORDINGS.lock().unwrap().entry(list_key).or_default().push(held.clone());
    } else {
        RECORDINGS.lock().unwrap().entry(allocator).or_default().push(held.clone());
    }
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    SLABS.fetch_add(slabs.len() as u64, Ordering::Relaxed);
    pso::record_dispatch_chain(list, grid, most, &held.scratch.buffer, used, constants, std::iter::once(&held.first).chain(&held.chain.tail));
    Ok(())
}

/// The rounds a chained ExecuteIndirect records at most, `XCL_SPLIT_INDIRECT_ROUNDS` (32): each
/// is one ExecuteIndirect and one barrier per piece, also when the title's arguments leave it empty.
fn most_rounds() -> u32 {
    static B: OnceLock<u32> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_INDIRECT_ROUNDS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(32u32).min(1024))
}

/// The rounds for slabs of `most` groups: as many as the arguments there now (`seen`, the GPU's
/// last ones) or an earlier ExecuteIndirect of the program (`before`) had, a quarter more, at
/// least two and at most `cap`. Before one was read back (`before` None), `cap`: the arguments
/// seen when recording can be 0 while the GPU's are not (a title's first ones had 17 slabs).
fn indirect_rounds(seen: &[[u32; 3]], most: u64, before: Option<u32>, cap: u32) -> u32 {
    let Some(before) = before else { return cap.max(2) };
    let now: u64 = seen.iter().map(|&g| pso::slab_count(g, most)).sum();
    let want = now.max(u64::from(before));
    (want + want.div_ceil(4)).clamp(2, u64::from(cap.max(2))) as u32
}

/// The slabs of an earlier ExecuteIndirect of the program for `indirect_rounds`: None (the most
/// rounds) until one was read back and its whole form is built, as slabs past the rounds are
/// lost until then (one program's first ExecuteIndirects had a slab, a later one 22, #2404). Then
/// the last one's, as more run the whole form: every round costs its pieces' ExecuteIndirects,
/// also when empty (each ~0.4 us of GPU after its barrier on an RTX 5070 Ti, #2404).
fn slabs_before(measured: bool, whole: bool, last: u32) -> Option<u32> {
    (measured && whole).then_some(last)
}

/// An ExecuteIndirect of split program `pso` whose command signature is one Dispatch of `stride`
/// bytes: `commands` of them at GPU address `args` (their count at `count`, when given), run as
/// the chain. `seen`: the arguments the CPU sees there now. `bind` binds the ABI after the
/// slicer. The whole form (built already) runs instead when the arguments have more slabs than
/// the rounds, also under the title's predication (`predicated`, which then holds for the slicer
/// too).
/// Ok(false): nothing recorded (no chain, a bundle, no scratch) – the caller runs the whole form.
#[allow(clippy::too_many_arguments)]
pub(in crate::hle::d3d12x) unsafe fn execute_indirect(list: &ID3D12GraphicsCommandList, pso: usize, allocator: usize, args: u64, stride: u32,
    commands: u32, count: Option<u64>, seen: &[[u32; 3]], predicated: bool, bind: impl FnOnce() -> Result<(), String>) -> Result<bool, String> {
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { return Ok(false) };
    if list.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE { return Ok(false); }
    // kept with the PC allocator the list records on, also for a list the pool has not seen (#2333
    // keeps such a list's with the title's allocator: the pool knew nothing of the list, and could
    // take that allocator for free while the list still recorded or ran on the GPU)
    let allocator = super::super::alloc_pool::recording_backing_or_take_on(list.as_raw() as usize, allocator, list.GetType().0);
    if allocator == 0 { return Ok(false); }
    let dev = &chain.builder.device;
    let group = u64::from(chain.words) * 4;
    let Some((scratch, _)) = scratch_for(dev, list.as_raw() as usize, allocator, scratch_budget(), group) else { return Ok(false) };
    let most = (scratch.bytes / group).max(1);
    let sig = pso::dispatch_command_signature(dev)?;
    let full = chain.whole.get().cloned();
    let before = slabs_before(chain.measured.load(Ordering::Relaxed), full.is_some(), chain.last.load(Ordering::Relaxed));
    let rounds = indirect_rounds(seen, most, before, most_rounds());
    let at = pso::chain_tail_offset(rounds);
    let cmd_bytes = pooled_size(pso::chain_commands_bytes(rounds, commands));
    // a pooled buffer is in COMMON again (buffers decay when their command lists have run), so
    // the slicer's first write promotes it as it does a new one
    let (cmd_bytes, buffer) = match take_cmd_buffer(cmd_bytes) {
        Some((bytes, b)) => {
            clear_read_back(&b, at)?;
            CMD_REUSES.fetch_add(1, Ordering::Relaxed);
            (bytes, b)
        }
        None => {
            CMD_ALLOCS.fetch_add(1, Ordering::Relaxed);
            let start = std::time::Instant::now();
            let b = pso::chain_commands_buffer_of(dev, cmd_bytes);
            CMD_ALLOC_MICROS.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            (cmd_bytes, b.ok_or("no buffer for the commands of a chained ExecuteIndirect")?)
        }
    };
    // not built: asked for now, so that it is there before an ExecuteIndirect of the program
    // needs more slabs than its rounds (until then those are lost)
    if full.is_none() {
        chain.want_whole();
    }
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    pso::record_slicer(list, args, stride, commands, count, rounds, most, full.is_some(), &buffer)?;
    // the list uses `buffer` from here: the recording keeps it until the allocator is reset, also
    // when binding fails and nothing more is recorded
    let bound = bind();
    if bound.is_ok() {
        // the slicer left either these arguments or the rounds empty. The whole form goes first
        // (each round's pieces start after a UAV barrier), with nothing of the chain before it:
        // after the pieces' ExecuteIndirects it did not run right on an RTX 3080 Ti (#2404)
        if let Some(full) = &full {
            list.SetPipelineState(full);
            list.ExecuteIndirect(&sig, commands, &buffer, at + 16, None::<&ID3D12Resource>, 0);
        }
        pso::record_dispatch_chain_indirect(list, &sig, &buffer, rounds, &scratch.buffer, std::iter::once(&first).chain(&chain.tail));
    }
    let indirect = Indirect { commands: buffer, bytes: cmd_bytes, at, rounds, whole: full.is_some(), checked: bound.is_ok() && !predicated };
    RECORDED.fetch_add(1, Ordering::Relaxed);
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(Arc::new(Recording { scratch, first, chain, indirect: Some(indirect) }));
    bound?;
    INDIRECT.fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

#[cfg(test)]
mod scratch_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    unsafe extern "system" fn no_create(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT {
        E_INVALIDARG
    }

    /// A bundle (or a list with no title allocator) runs the already-compiled pieces with an
    /// XCL-owned scratch, not the whole form; the scratch is released when the bundle is reset.
    #[test]
    fn a_bundle_dispatch_runs_the_pieces_with_an_xcl_scratch_not_the_whole_form() {
        unsafe {
            if crate::gpu::recomp::dxc::location().is_err() {
                return crate::gpu::recomp::runner::skip_test("no DXC");
            }
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>());
                if let Ok(warp) = warp {
                    let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
                }
            }
            let Some(dev) = dev else { return crate::gpu::recomp::runner::skip_test("no D3D12 device") };
            let dxil = crate::gpu::recomp::dxc::compile(
                "RWByteAddressBuffer mem : register(u2); [numthreads(1,1,1)] void main() { mem.Store(0, 1); }",
                "main", "cs_6_6", &[],
            ).expect("DXC compile");
            let rs = pso::dispatch_chain_root_signature(&dev, &[]).expect("dispatch chain RS");
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
                ..Default::default()
            };
            let first: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
            std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
            let builder = Arc::new(PipelineBuilder { device: dev.clone(), create: no_create, signature: rs.clone(), flags: Default::default(), node: 0, record: 0 });
            let chain = Arc::new(Chain {
                builder, tail: Vec::new(), words: 4, whole_code: Vec::new(), whole: OnceLock::new(), build_whole: Mutex::new(()), waiting: Default::default(),
                last: Default::default(), measured: Default::default(),
                wanted: AtomicBool::new(false), warned: AtomicBool::new(false),
            });
            let pso = first.clone().into_raw() as usize;
            CHAINS.lock().unwrap().insert(pso, chain.clone());

            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE).expect("bundle allocator");
            let bundle: ID3D12GraphicsCommandList = dev.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_BUNDLE,
                &alloc,
                Option::<&ID3D12PipelineState>::None,
            ).expect("bundle");
            bundle.SetComputeRootSignature(&rs);
            assert!(dispatch(&bundle, pso, 0, [1, 1, 1]).is_ok(), "the bundle records the pieces");
            assert!(LIST_RECORDINGS.lock().unwrap().get(&(bundle.as_raw() as usize)).is_some_and(|v| !v.is_empty()), "the scratch is kept with the bundle");
            list_reset(bundle.as_raw() as usize);
            assert!(!LIST_RECORDINGS.lock().unwrap().contains_key(&(bundle.as_raw() as usize)), "resetting the bundle releases its scratch");
            drop(bundle);
            drop(alloc);
            forget(pso);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{scratch_bytes, LEAST};

    const MB: u64 = 1 << 20;

    #[test]
    fn a_list_recorded_after_another_goes_on_in_its_scratch() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            let warp = unsafe { CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) };
            if warp.and_then(|w| unsafe { D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev) }).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
        }
        let dev = dev.unwrap();
        let (a, b, c, alloc) = (0x7E57_5C01usize, 0x7E57_5C02usize, 0x7E57_5C03usize, 0x7E57_5CA0usize);
        let (first, used) = unsafe { super::scratch_for(&dev, a, alloc, MB, 4096) }.unwrap();
        assert!(!used);
        super::list_follows(a, b);
        let (second, used) = unsafe { super::scratch_for(&dev, b, alloc, MB, 4096) }.unwrap();
        assert!(used && std::sync::Arc::ptr_eq(&first, &second), "the list after it made a scratch of its own");
        let (third, _) = unsafe { super::scratch_for(&dev, c, alloc, MB, 4096) }.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &third));
        for list in [a, b, c] {
            super::list_reset(list);
        }
    }

    #[test]
    fn a_chains_scratch_is_bounded_by_what_all_recordings_hold() {
        let group = 187 << 10;
        // the first of a recording: as much as it needs, up to the budget of one dispatch
        assert_eq!(scratch_bytes(15 * MB, group, 64 * MB, None, 0, 384 * MB), (15 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 0, 384 * MB), (64 * MB, true));
        // the others hold nearly all: what is left, and at least LEAST
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 364 * MB, 384 * MB), (20 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 1200 * MB, 384 * MB), (LEAST, true));
        // the list's own buffer: kept when big enough, or when no more may be made
        assert_eq!(scratch_bytes(8 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (16 * MB, false));
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 380 * MB, 384 * MB), (16 * MB, false));
        // … but made again, larger, while there is room
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (60 * MB, true));
        // one group always fits, whatever the bounds say
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, None, 400 * MB, 384 * MB), (9 * MB, true));
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, Some(MB), 400 * MB, 384 * MB), (9 * MB, true));
    }

    #[test]
    fn a_programs_first_execute_indirect_gets_the_most_rounds() {
        use super::indirect_rounds;
        // none read back yet: the arguments seen when recording (0 while a title loads) don't count
        assert_eq!(indirect_rounds(&[[0, 0, 0]], 8, None, 32), 32);
        assert_eq!(indirect_rounds(&[[2, 3, 2]], 1, None, 32), 32);
        assert_eq!(indirect_rounds(&[], 1, None, 1), 2);
        // then the slabs seen or read back, a quarter more, 2 to the cap
        assert_eq!(indirect_rounds(&[[0, 0, 0]], 8, Some(0), 32), 2);
        assert_eq!(indirect_rounds(&[[2, 3, 2]], 1, Some(0), 32), 15);
        assert_eq!(indirect_rounds(&[[0, 0, 0]], 1, Some(17), 32), 22);
        assert_eq!(indirect_rounds(&[[0, 0, 0]], 1, Some(40), 32), 32);
    }

    #[test]
    fn once_the_whole_form_is_built_the_last_execute_indirect_sizes_the_rounds() {
        use super::{indirect_rounds, slabs_before};
        assert_eq!(slabs_before(false, true, 1), None);
        // not built: the most rounds, also after one of a slab was read back (a later one can
        // have 22, whose slabs past the rounds would be lost)
        assert_eq!(slabs_before(true, false, 1), None);
        assert_eq!(indirect_rounds(&[[0, 0, 0]], 1, slabs_before(true, false, 1), 32), 32);
        // built: the last one's, an overflow runs the whole form
        assert_eq!(slabs_before(true, true, 1), Some(1));
        assert_eq!(indirect_rounds(&[[0, 0, 0]], 1, slabs_before(true, true, 1), 32), 2);
        assert_eq!(indirect_rounds(&[[3, 1, 1]], 1, slabs_before(true, true, 1), 32), 4);
    }

    #[test]
    fn the_whole_form_built_in_the_background_waits_for_a_huge_compiles_turn() {
        use super::super::{big_compile_turn, driver_compile, huge_pso_bytes, huge_pso_concurrency, BIG_COMPILES, GIANT_COMPILES};
        // every slot taken: a huge compile on `want_whole`'s thread waits for one
        let mut held: Vec<_> = (0..huge_pso_concurrency()).map(|_| big_compile_turn(&BIG_COMPILES, || true, || false)).collect();
        held.push(big_compile_turn(&GIANT_COMPILES, || true, || false));
        let t = super::whole_thread().spawn(|| driver_compile("test", huge_pso_bytes(), true, || 7)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(!t.is_finished(), "a huge compile on the whole form's thread did not wait for a slot");
        drop(held);
        assert_eq!(t.join().unwrap(), 7);
    }

    #[test]
    fn direct_dispatches_write_their_constants_in_blocks_used_again_after_the_allocator_reset() {
        use super::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let dev = dev.unwrap();
            // allocators no list records with
            let (a, b, c) = (usize::MAX - 2, usize::MAX - 1, usize::MAX);
            let slab = u64::from(pso::CHAIN_COMMAND_BYTES);
            let (cpu1, gpu1) = constants_for(&dev, a, 2 * slab).unwrap();
            let (cpu2, gpu2) = constants_for(&dev, a, slab).unwrap();
            assert_eq!((cpu2 as u64 - cpu1 as u64, gpu2 - gpu1), (2 * slab, 2 * slab));
            assert_eq!(gpu1 % slab, 0);
            // another allocator's dispatch gets other memory
            let (_, gpu3) = constants_for(&dev, b, slab).unwrap();
            assert!(gpu3 + slab <= gpu1 || gpu3 >= gpu1 + CONSTANTS_BLOCK);
            // more than the rest of a block: another one, of its size
            let (_, gpu4) = constants_for(&dev, a, CONSTANTS_BLOCK + slab).unwrap();
            assert!(gpu4 + CONSTANTS_BLOCK + slab <= gpu1 || gpu4 >= gpu1 + CONSTANTS_BLOCK);
            std::ptr::write_bytes(cpu1, 0xab, (2 * slab) as usize);
            // reset, its blocks are used again
            allocator_reset(a);
            let mine = |g: u64| (gpu1..gpu1 + CONSTANTS_BLOCK).contains(&g) || (gpu4..gpu4 + CONSTANTS_BLOCK + slab).contains(&g);
            let (_, again) = constants_for(&dev, c, CONSTANTS_BLOCK).unwrap();
            assert!(mine(again), "{again:#x}");
            // but not the bigger one, and no more than the budget
            {
                let free = FREE_CONSTANTS.lock().unwrap();
                assert!(free.iter().all(|f| f.bytes == CONSTANTS_BLOCK), "a block bigger than {CONSTANTS_BLOCK} kept for use again");
                assert!(free.iter().map(|f| f.bytes).sum::<u64>() <= FREE_CONSTANTS_BYTES);
            }
            allocator_reset(b);
            allocator_reset(c);
        }
    }

    #[test]
    fn a_chained_execute_indirect_keeps_its_commands_when_binding_fails() {
        use super::*;
        unsafe extern "system" fn never(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT { E_INVALIDARG }
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let dev = dev.unwrap();
            let Ok(dxil) = crate::gpu::recomp::dxc::compile("[numthreads(1, 1, 1)] void main() {}", "main", "cs_6_0", &[]) else {
                return crate::gpu::recomp::runner::skip_test("no DXC");
            };
            let rs = pso::dispatch_chain_root_signature(&dev, &[]).unwrap();
            let first: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() }, ..Default::default() }).unwrap();
            let builder = Arc::new(PipelineBuilder { device: dev.clone(), create: never, signature: rs.clone(), flags: D3D12_PIPELINE_STATE_FLAG_NONE, node: 0, record: 0 });
            let pso = first.as_raw() as usize;
            CHAINS.lock().unwrap().insert(pso, Arc::new(Chain { builder, tail: vec![first.clone()], words: 4, whole_code: Vec::new(), whole: OnceLock::new(),
                build_whole: Mutex::new(()), waiting: Default::default(), last: Default::default(), measured: Default::default(), wanted: Default::default(), warned: Default::default() }));
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let (raw_list, allocator) = (list.as_raw() as usize, alloc.as_raw() as usize);
            super::super::super::alloc_pool::created_on(raw_list, allocator, ty.0);
            let args = pso::chain_commands_buffer(&dev, 1, 1).unwrap();
            let r = execute_indirect(&list, pso, allocator, args.GetGPUVirtualAddress(), 12, 1, None, &[[1, 1, 1]], false,
                || Err("unified memory is full".to_string()));
            assert_eq!(r, Err("unified memory is full".to_string()));
            // the program's first ExecuteIndirect asks for its whole form, before any slab is lost
            assert!(CHAINS.lock().unwrap()[&pso].wanted.load(Ordering::Relaxed));
            // the slicer's dispatch is recorded: its command buffer lives until the allocator is
            // reset, and the read-back does not take the rounds that were not recorded
            // under the title's predication the slicer may not run, so its words are not read back
            // (reported by @thehollowedone, #2404); without it they are
            // (`bind` sets the chain's root signature, as `bind_abi` does, on a list with descriptor
            // heaps, as a title's list has: the signature indexes them directly)
            let heap = |ty| -> ID3D12DescriptorHeap { dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).unwrap() };
            let heaps = [Some(heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)), Some(heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER))];
            for predicated in [true, false] {
                let bind = || {
                    list.SetDescriptorHeaps(&heaps);
                    list.SetComputeRootSignature(&rs);
                    Ok(())
                };
                assert_eq!(execute_indirect(&list, pso, allocator, args.GetGPUVirtualAddress(), 12, 1, None, &[[1, 1, 1]], predicated, bind), Ok(true));
            }
            let kept = RECORDINGS.lock().unwrap().get(&allocator).map(|v| v.iter().filter_map(|r| r.indirect.as_ref().map(|i| i.checked)).collect::<Vec<_>>());
            assert_eq!(kept, Some(vec![false, false, true]));
            list.Close().unwrap();
            let commands_of = || RECORDINGS.lock().unwrap().get(&allocator).map(|v| v.iter().filter_map(|r| r.indirect.as_ref().map(|i| i.commands.as_raw() as usize)).collect::<Vec<_>>()).unwrap_or_default();
            let first_buffers = commands_of();
            assert_eq!(first_buffers.len(), 3);
            allocator_reset(allocator);
            alloc.Reset().unwrap();
            list.Reset(&alloc, None).unwrap();
            let bind = || {
                list.SetComputeRootSignature(&rs);
                Ok(())
            };
            assert_eq!(execute_indirect(&list, pso, allocator, args.GetGPUVirtualAddress(), 12, 1, None, &[[1, 1, 1]], false, bind), Ok(true));
            assert!(first_buffers.contains(&commands_of()[0]), "the reset allocator's buffer is used again");
            list.Close().unwrap();
            allocator_reset(allocator);
            list_reset(raw_list);
            forget(pso);
            super::super::super::alloc_pool::forget_list(raw_list);
        }
    }

    #[test]
    fn command_buffers_are_pooled_by_whole_pages() {
        assert_eq!(super::pooled_size(1), 64 << 10);
        assert_eq!(super::pooled_size(64 << 10), 64 << 10);
        assert_eq!(super::pooled_size((64 << 10) + 1), 128 << 10);
    }

    /// A title thread that needs the whole form while `want_whole`'s build of it waits for its
    /// turn (`big_compile_turn`, every background slot taken) does not wait behind that turn: the
    /// build is then one the title waits for, with the wanted cap and no commit wait (#2404)
    #[test]
    fn whole_forms_wanted_together_are_built_one_after_another_on_one_thread() {
        use super::InTurn;
        use std::sync::{atomic::Ordering, Mutex};
        static TURNS: InTurn<u32> = InTurn::new();
        static GATE: Mutex<()> = Mutex::new(());
        static RAN: Mutex<Vec<(u32, Option<String>)>> = Mutex::new(Vec::new());
        let gate = GATE.lock().unwrap();
        for i in 0..5 {
            TURNS.push(i, |i| {
                drop(GATE.lock().unwrap());
                RAN.lock().unwrap().push((i, std::thread::current().name().map(String::from)));
            });
        }
        let started = TURNS.started.load(Ordering::Relaxed);
        drop(gate);
        let t = std::time::Instant::now();
        while RAN.lock().unwrap().len() < 5 && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let ran = RAN.lock().unwrap().clone();
        assert_eq!(started, 1, "a thread for each of 5 whole forms");
        assert_eq!(ran, (0..5).map(|i| (i, Some("xcl-pso-chain-whole".to_string()))).collect::<Vec<_>>());
        // the thread ends with the queue; the next one starts another
        let t = std::time::Instant::now();
        while TURNS.queue.lock().unwrap().1 && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        TURNS.push(5, |i| RAN.lock().unwrap().push((i, None)));
        assert_eq!(TURNS.started.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_title_thread_does_not_wait_behind_a_background_whole_build_waiting_for_its_turn() {
        use super::*;
        use std::time::{Duration, Instant};
        unsafe extern "system" fn refused(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT { E_INVALIDARG }
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let dev = dev.unwrap();
            let rs = pso::dispatch_chain_root_signature(&dev, &[]).unwrap();
            let builder = Arc::new(PipelineBuilder { device: dev, create: refused, signature: rs, flags: D3D12_PIPELINE_STATE_FLAG_NONE, node: 0, record: 0 });
            // a huge whole form no cache has: its build on `want_whole`'s thread is gated
            let mut code = vec![0u8; huge_pso_bytes()];
            let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            code[..16].copy_from_slice(&nonce.to_le_bytes());
            let chain = Arc::new(Chain { builder, tail: Vec::new(), words: 4, whole_code: code, whole: OnceLock::new(),
                build_whole: Mutex::new(()), waiting: Default::default(), last: Default::default(), measured: Default::default(), wanted: Default::default(), warned: Default::default() });
            let held: Vec<_> = (0..huge_pso_concurrency()).map(|_| big_compile_turn(&BIG_COMPILES, || true, || false)).collect();
            chain.want_whole();
            let started = Instant::now();
            while chain.build_whole.try_lock().is_ok() {
                assert!(started.elapsed() < Duration::from_secs(10), "the background build did not start");
                std::thread::sleep(Duration::from_millis(10));
            }
            let (done, waited) = std::sync::mpsc::channel();
            let title = chain.clone();
            std::thread::spawn(move || {
                let _ = title.whole();
                let _ = done.send(());
            });
            let waited = waited.recv_timeout(Duration::from_secs(5));
            drop(held);
            assert!(waited.is_ok(), "the title thread waited for the background build's turn");
        }
    }
}
