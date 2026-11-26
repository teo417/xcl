//! `XCL_COMPUTE_CRUMBS=1` (with `XCL_DRED=1` for the pipeline labels): which translated compute
//! dispatches the GPU started and has not finished.
//!
//! A compute shader that never ends raises no TDR while the GPU can preempt it, and a forced
//! removal left DRED without breadcrumbs (RTX 4080 SUPER, driver 621.40). So every dispatch XCL
//! records on a COMPUTE list writes its number into a readback buffer before it
//! (`MARKER_IN`) and after it (`MARKER_OUT`); the GPU wait watchdog names those with only the first.

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// Dispatches remembered; a stalled GPU is a few frames (some thousand dispatches) behind.
const ENTRIES: usize = 1 << 18;
/// In-flight dispatches named in one report.
const REPORTED: usize = 24;

struct Crumbs {
    buffer: ID3D12Resource,
    /// the buffer's CPU view: (started, finished) per entry
    at: usize,
    next: u32,
    /// (number, list, pipeline) per entry
    meta: Vec<(u32, usize, usize)>,
}

static CRUMBS: Mutex<Option<Crumbs>> = Mutex::new(None);

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_COMPUTE_CRUMBS").is_some())
}

/// Writes the finishing mark when dropped, after what was recorded meanwhile.
pub struct Crumb {
    list: ID3D12GraphicsCommandList2,
    address: u64,
    number: u32,
}

unsafe fn mark(list: &ID3D12GraphicsCommandList2, address: u64, value: u32, mode: D3D12_WRITEBUFFERIMMEDIATE_MODE) {
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: address, Value: value };
    list.WriteBufferImmediate(1, &p, Some(&mode));
}

unsafe fn make(list: &ID3D12GraphicsCommandList) -> Option<Crumbs> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (ENTRIES * 8) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() };
    let mut buffer: Option<ID3D12Resource> = None;
    dev?.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()?;
    let buffer = buffer?;
    let mut at: *mut c_void = std::ptr::null_mut();
    buffer.Map(0, None, Some(&mut at)).ok()?;
    Some(Crumbs { buffer, at: at as usize, next: 0, meta: vec![(0, 0, 0); ENTRIES] })
}

/// Before a dispatch of `pso` is recorded on `list`; the guard marks its end.
pub unsafe fn begin(list: &ID3D12GraphicsCommandList, pso: usize) -> Option<Crumb> {
    if !on() || list.GetType() != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return None;
    }
    let list2: ID3D12GraphicsCommandList2 = list.cast().ok()?;
    let (address, number) = {
        let mut g = CRUMBS.lock().unwrap();
        if g.is_none() {
            *g = make(list);
        }
        let c = g.as_mut()?;
        c.next = c.next.wrapping_add(1).max(1);
        let number = c.next;
        let k = number as usize % ENTRIES;
        c.meta[k] = (number, list.as_raw() as usize, pso);
        (c.buffer.GetGPUVirtualAddress() + (k * 8) as u64, number)
    };
    mark(&list2, address, number, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
    Some(Crumb { list: list2, address: address + 4, number })
}

impl Drop for Crumb {
    fn drop(&mut self) {
        unsafe { mark(&self.list, self.address, self.number, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT) }
    }
}

/// The entries whose dispatch the GPU started and did not finish, oldest first. `marks` gives
/// an entry's (started, finished) words; `meta` the number each entry was last given.
fn unfinished(meta: &[(u32, usize, usize)], marks: impl Fn(usize) -> (u32, u32)) -> Vec<(u32, usize, usize)> {
    let mut out: Vec<_> = meta
        .iter()
        .enumerate()
        .filter(|&(k, &(number, ..))| {
            let (started, finished) = marks(k);
            number != 0 && started == number && finished != number
        })
        .map(|(_, &m)| m)
        .collect();
    out.sort_by_key(|m| m.0);
    out
}

/// Lines for the log: the unfinished dispatches, one per list and pipeline.
pub fn report() -> Vec<String> {
    let g = CRUMBS.lock().unwrap();
    let Some(c) = g.as_ref() else { return Vec::new() };
    let words = c.at as *const u32;
    let left = unfinished(&c.meta, |k| unsafe { (std::ptr::read_volatile(words.add(2 * k)), std::ptr::read_volatile(words.add(2 * k + 1))) });
    let mut seen = HashSet::new();
    let mut out = vec![format!("{} compute dispatch(es) started and not finished (of {} recorded)", left.len(), c.next)];
    for &(number, list, pso) in &left {
        if out.len() > REPORTED || !seen.insert((list, pso)) {
            continue;
        }
        let label = super::dred::label_of(pso)
            .or_else(|| super::dred::label_of(super::pso_async::resolve(pso)))
            .unwrap_or_else(|| "unlabelled (XCL_DRED=1 names them)".into());
        let split = super::real::dispatch_chain::is_chain(pso) || super::real::dispatch_chain::is_chain(super::pso_async::resolve(pso));
        out.push(format!("dispatch #{number} on list {list:#x}, pipeline {pso:#x}{}: {label}", if split { " (split program)" } else { "" }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_started_and_unfinished_dispatches_are_named() {
        // finished, running, recorded but not started, never used, an entry reused (old marks)
        let meta = [(1, 10, 100), (2, 10, 101), (3, 11, 102), (0, 0, 0), (9, 12, 103)];
        let marks = [(1, 1), (2, 0), (0, 0), (0, 0), (4, 4)];
        assert_eq!(unfinished(&meta, |k| marks[k]), vec![(2, 10, 101)]);
    }
}
