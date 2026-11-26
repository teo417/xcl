//! `XCL_GPU_TIMING=1`: per-submission GPU timestamp queries for DIRECT and COMPUTE queues.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

const SLOTS: usize = 32;

static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static LISTS_TIMED: AtomicU64 = AtomicU64::new(0);
static GPU_TIME_US: AtomicU64 = AtomicU64::new(0);
static MAX_SUBMIT_US: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_TIMING").is_ok_and(|v| v.trim() == "1"))
}

/// `[submissions, lists_timed, gpu_time_us, max_submit_us]`
pub fn activity() -> [u64; 4] {
    [
        SUBMISSIONS.load(Ordering::Relaxed),
        LISTS_TIMED.load(Ordering::Relaxed),
        GPU_TIME_US.load(Ordering::Relaxed),
        MAX_SUBMIT_US.load(Ordering::Relaxed),
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
    in_flight: bool,
}

struct QueueTimer {
    freq: u64,
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    fence: ID3D12Fence,
    next_val: u64,
    slots: Vec<Slot>,
}

static TIMERS: Mutex<Option<HashMap<usize, QueueTimer>>> = Mutex::new(None);

impl QueueTimer {
    unsafe fn new(queue: &ID3D12CommandQueue) -> Option<Self> {
        let desc = queue.GetDesc();
        if desc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT && desc.Type != D3D12_COMMAND_LIST_TYPE_COMPUTE {
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
    let prepared = {
        let mut guard = TIMERS.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if !map.contains_key(&(queue_raw as usize)) {
            if let Some(t) = QueueTimer::new(queue) {
                map.insert(queue_raw as usize, t);
            }
        }
        if let Some(timer) = map.get_mut(&(queue_raw as usize)) {
            timer.drain();
            if let Some(idx) = timer.slots.iter().position(|s| !s.in_flight) {
                let q0 = (idx * 2) as u32;
                let q1 = q0 + 1;
                let heap = timer.heap.clone();
                let readback = timer.readback.clone();
                let fence = timer.fence.clone();
                let val = timer.next_val;
                timer.next_val += 1;
                let slot = &mut timer.slots[idx];
                if slot.alloc.Reset().is_ok()
                    && slot.start.Reset(&slot.alloc, None).is_ok()
                {
                    slot.start.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, q0);
                    let ok_start = slot.start.Close().is_ok();
                    let ok_end = ok_start && slot.end.Reset(&slot.alloc, None).is_ok();
                    if ok_end {
                        slot.end.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, q1);
                        slot.end.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, q0, 2, &readback, (idx * 2 * 8) as u64);
                        if slot.end.Close().is_ok() {
                            slot.val = val;
                            slot.lists = chunk.len();
                            slot.in_flight = true;
                            Some((slot.start.as_raw(), slot.end.as_raw(), fence, val))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    };
    let Some((start_raw, end_raw, fence, val)) = prepared else {
        submit(chunk);
        return;
    };
    let mut wrapped = Vec::with_capacity(chunk.len() + 2);
    wrapped.push(start_raw);
    wrapped.extend_from_slice(chunk);
    wrapped.push(end_raw);
    submit(&wrapped);
    let _ = queue.Signal(&fence, val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_to_micros_computes_exact_duration() {
        assert_eq!(ticks_to_micros(1_000_000, 2_500_000, 10_000_000), 150_000);
        assert_eq!(ticks_to_micros(500, 100, 10_000_000), 0);
        assert_eq!(ticks_to_micros(0, 100, 0), 0);
    }
}
