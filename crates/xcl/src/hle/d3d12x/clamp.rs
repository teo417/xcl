//! Indirect compute dispatches with group counts D3D12 takes (`XCL_CLAMP_INDIRECT=0`: off).
//!
//! The console runs up to 2^32 − 1 groups a dimension; D3D12 takes 65 535, and past that the
//! dispatch is undefined – on an NVIDIA GPU it ran on until the driver reset the device
//! (0x887A0006). A retail game's world loading dispatches a compute program indirectly with
//! 72 512 × 1 × 1 and 75 072 × 1 × 1 groups, counts another dispatch wrote, so the CPU never sees
//! them.
//!
//! Before such a dispatch a one-thread pass reads its three counts and writes the ones to
//! dispatch into a slot of XCL's own buffer: more than 65 535 in X with 1 in Y folded into Y,
//! with the console's count written into the dispatch's io (`recomp::IO_GROUP_LIMIT`: the
//! program takes its group in X from both and leaves out the groups past the count), any other
//! count past 65 535 clamped. The dispatch then reads the slot. How many were folded and clamped,
//! and the latest counts, are kept in the buffer's first 32 bytes and logged now and then (the
//! buffer is CPU-visible memory, a custom heap).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// Slots of 16 bytes after the header; reused in turn (a slot is read by the dispatch recorded
/// right after it is written, long before 65 536 more indirect dispatches come).
const SLOTS: u64 = 1 << 16;
/// Counts clamped (count, the last x, y, z) and folded (count, the last x) – 32 bytes.
const HEADER: u64 = 32;

const HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer C : register(b0) { uint slot; };
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
RWByteAddressBuffer io : register(u2);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint3 g = src.Load3(0);
    uint3 c;
    uint folded = 0u;
    // rows of 65 535 groups (a ceiling without an addition that could wrap near 2^32)
    uint rows = g.x / 65535u + (g.x % 65535u != 0u ? 1u : 0u);
    if (g.x > 65535u && g.y == 1u && g.z <= 65535u && rows <= 65535u) {
        // more groups in X than D3D12 takes: folded into Y, the program told the count
        // (as `bind::fold` does on the CPU)
        folded = g.x;
        c = uint3(65535u, rows, g.z);
    } else {
        c = min(g, 65535u);
    }
    io.Store(0, folded);
    if (folded != 0u) {
        uint n;
        dst.InterlockedAdd(16, 1, n);
        dst.Store(20, g.x);
    } else if (any(c != g)) {
        uint n;
        dst.InterlockedAdd(0, 1, n);
        dst.Store3(4, g);
    }
    dst.Store3(32 + slot * 16, c);
}
"#;

struct Clamp {
    rs: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    buf: ID3D12Resource,
    gpu: u64,
    cpu: *const u32,
}
// SAFETY: D3D12 objects are free-threaded; `cpu` is the persistent map of `buf`, only read
unsafe impl Send for Clamp {}
unsafe impl Sync for Clamp {}

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("XCL_CLAMP_INDIRECT").map_or(true, |v| v.trim() != "0"))
}

fn get(list: &ID3D12GraphicsCommandList) -> Option<&'static Clamp> {
    static C: OnceLock<Option<Clamp>> = OnceLock::new();
    C.get_or_init(|| match unsafe { make(list) } {
        Ok(c) => {
            crate::xlog_info!("D3D12.X: indirect dispatches past 65535 groups a dimension are folded or clamped on the GPU");
            Some(c)
        }
        Err(e) => {
            crate::xlog_warn!("D3D12.X: indirect dispatches are not clamped: {e}");
            None
        }
    })
    .as_ref()
}

unsafe fn make(list: &ID3D12GraphicsCommandList) -> Result<Clamp, String> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).map_err(|e| e.to_string())?;
    let dev = dev.ok_or("no device")?;
    let dxil = crate::gpu::recomp::dxc::compile(HLSL, "main", "cs_6_0", &[])?;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).map_err(|e| format!("root signature: {e}"))?;
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&desc);
    drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
    let pso = pso.map_err(|e| format!("pipeline: {e}"))?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let bdesc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: HEADER + SLOTS * 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).map_err(|e| format!("buffer: {e}"))?;
    let buf = buf.ok_or("buffer")?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).map_err(|e| format!("map: {e}"))?;
    std::ptr::write_bytes(p as *mut u8, 0, HEADER as usize);
    Ok(Clamp { rs, pso, gpu: buf.GetGPUVirtualAddress(), buf, cpu: p as *const u32 })
}

fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(res) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Before one indirect dispatch whose counts are at `args` + `off` (a buffer that takes UAV
/// access): the counts as D3D12 takes them – more than 65 535 in X (1 in Y) folded into Y with
/// the console's count written to `io_limit` (the dispatch's `recomp::IO_GROUP_LIMIT` word, 0
/// otherwise), any other count past 65 535 clamped – in a slot of XCL's buffer, which is
/// returned with the slot's offset, in INDIRECT_ARGUMENT state; `done` returns it to UAV after
/// the dispatch. None: left as it was (the caller dispatches from `args`). The list's compute
/// root signature, root arguments and pipeline are changed: the caller sets its own again.
pub unsafe fn before(list: &ID3D12GraphicsCommandList, args: usize, off: u64, io_limit: u64) -> Option<(usize, u64)> {
    if !enabled() {
        return None;
    }
    let raw = args as *mut c_void;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let d = res.GetDesc();
    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 || off % 4 != 0 || off + 12 > d.Width {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: an indirect dispatch's arguments ({:?}, flags {:#x}, offset {off:#x}) cannot be clamped", d.Dimension, d.Flags.0);
        }
        return None;
    }
    let c = get(list)?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let slot = NEXT.fetch_add(1, Ordering::Relaxed) % SLOTS;
    // COMMON at the list's start: promoted to UAV by its first use
    list.SetComputeRootSignature(&c.rs);
    list.SetPipelineState(&c.pso);
    list.SetComputeRoot32BitConstant(0, slot as u32, 0);
    list.SetComputeRootUnorderedAccessView(1, res.GetGPUVirtualAddress() + off);
    list.SetComputeRootUnorderedAccessView(2, c.gpu);
    list.SetComputeRootUnorderedAccessView(3, io_limit);
    list.Dispatch(1, 1, 1);
    // the slot for the dispatch's arguments; the io word for its program (a UAV barrier on all)
    let uav = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE, Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) } };
    list.ResourceBarrier(&[transition(&c.buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT), uav]);
    report(c);
    Some((c.buf.as_raw() as usize, HEADER + slot * 16))
}

/// After the dispatch `before` clamped: XCL's buffer back in UAV state.
pub unsafe fn done(list: &ID3D12GraphicsCommandList) {
    if let Some(c) = get(list) {
        list.ResourceBarrier(&[transition(&c.buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    }
}

/// The counts the GPU had to change so far: logged when more were changed (at most every 2 s).
fn report(c: &Clamp) {
    static SEEN: AtomicU32 = AtomicU32::new(0);
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let mut l = LAST.lock().unwrap();
    if l.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(2)) {
        return;
    }
    *l = Some(std::time::Instant::now());
    // SAFETY: the mapped header of `buf` (16 bytes)
    let w = |k: usize| unsafe { std::ptr::read_volatile(c.cpu.add(k)) };
    let n = w(0);
    if n != SEEN.swap(n, Ordering::Relaxed) {
        crate::xlog_warn!("D3D12.X: {n} indirect dispatches had group counts clamped so far (the latest asked for {}x{}x{})", w(1), w(2), w(3));
    }
    static FOLDED: AtomicU32 = AtomicU32::new(0);
    let f = w(4);
    if f != FOLDED.swap(f, Ordering::Relaxed) {
        crate::xlog_info!("D3D12.X: {f} indirect dispatches folded so far (the latest of {} groups in X)", w(5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CPU-visible buffer that takes UAV access (a custom heap), mapped.
    unsafe fn visible(dev: &ID3D12Device, bytes: u64) -> (ID3D12Resource, *mut u32) {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).unwrap();
        let buf = buf.unwrap();
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut p)).unwrap();
        (buf, p as *mut u32)
    }

    /// The pass writes the counts D3D12 takes into its slot: 72 512 × 1 × 1 folded into
    /// 65 535 × 2 × 1 with 72 512 in io; 70 000 × 2 × 1 clamped with 0 in io; 300 × 4 × 2 as
    /// it was.
    #[test]
    fn indirect_group_counts_are_folded_or_clamped_on_the_gpu() {
        use crate::gpu::recomp::runner;
        unsafe {
            let dev = match (runner::device(), crate::gpu::recomp::dxc::location()) {
                (Ok(d), Ok(_)) => d,
                _ => {
                    runner::skip_test("no GPU or DXC");
                    return;
                }
            };
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            // (the last three at the top of the range: 65 535² folds into exactly 65 535 rows,
            // one more or u32::MAX would need 65 536 rows – clamped, as `bind::fold` does)
            for (k, (args, want, io_want)) in [
                ([72_512u32, 1, 1], [65_535u32, 2, 1], 72_512u32),
                ([70_000, 2, 1], [65_535, 2, 1], 0),
                ([300, 4, 2], [300, 4, 2], 0),
                ([65_535 * 65_535, 1, 1], [65_535, 65_535, 1], 65_535 * 65_535),
                ([65_535 * 65_535 + 1, 1, 1], [65_535, 1, 1], 0),
                ([u32::MAX, 1, 1], [65_535, 1, 1], 0),
            ]
            .into_iter()
            .enumerate()
            {
                let (limit, groups) = super::super::bind::fold(args[0], args[1], args[2]);
                assert_eq!((groups, limit), (want, io_want), "{args:?}: the CPU's fold, which the GPU's must match");
                let (a, ap) = visible(&dev, 256);
                let (io, iop) = visible(&dev, 256);
                std::ptr::copy_nonoverlapping(args.as_ptr(), ap.add(16), 3);
                *iop = 0xDEAD;
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
                let (buf, off) = before(&list, a.as_raw() as usize, 64, io.GetGPUVirtualAddress()).expect("clamped");
                done(&list);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                queue.Signal(&fence, k as u64 + 1).unwrap();
                let t = std::time::Instant::now();
                while fence.GetCompletedValue() < k as u64 + 1 {
                    assert!(t.elapsed().as_secs() < 10, "the pass did not finish");
                    std::thread::yield_now();
                }
                let c = get(&list).unwrap();
                assert_eq!(buf, c.buf.as_raw() as usize);
                let slot = c.cpu.add(off as usize / 4);
                let got = [*slot, *slot.add(1), *slot.add(2)];
                assert_eq!(got, want, "{args:?}");
                assert_eq!(*iop, io_want, "{args:?}: io");
            }
        }
    }
}
