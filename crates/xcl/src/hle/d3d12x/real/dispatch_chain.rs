//! PSOs for a split compute entry. The first piece is the real PSO returned
//! to the caller. Other pieces and the lazily built whole form belong to it.
use super::*;
use crate::gpu::{dxil::kernel::DispatchChain, recomp::pso};
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

pub(crate) fn log_counts() {
    let mib = |a: &AtomicU64| a.load(Ordering::Relaxed) >> 20;
    let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
    crate::xlog_info!("compute split: {} programs split, {} pieces built, {} chained dispatches issued in {} slabs; scratch {} MiB (most {}, largest dispatch {} in one slab); ExecuteIndirect: {} chained ({} of them whole, {} slabs lost), {} whole",
        n(&PROGRAMS), n(&PIECES), n(&DISPATCHES), n(&SLABS), mib(&LIVE), mib(&PEAK), mib(&WANTED),
        n(&INDIRECT), n(&INDIRECT_WHOLE), n(&INDIRECT_LOST), n(&WHOLE_INDIRECT));
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
    /// its command signature for ExecuteIndirect (`pso::chain_command_signature`)
    signature: OnceLock<ID3D12CommandSignature>,
    /// the most slabs one of its ExecuteIndirects had, and whether one was read back yet
    slabs: std::sync::atomic::AtomicU32,
    measured: std::sync::atomic::AtomicBool,
    /// its whole form was asked for in the background / a lost slab was reported
    wanted: std::sync::atomic::AtomicBool,
    warned: std::sync::atomic::AtomicBool,
}

impl Chain {
    unsafe fn whole(&self) -> Result<ID3D12PipelineState, String> {
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        let _guard = self.build_whole.lock().unwrap();
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        let p = self.builder.pipeline(&self.whole_code, &pc_iid(Kind::PipelineState)).ok_or("whole compute pipeline")?;
        let _ = self.whole.set(p.clone());
        Ok(p)
    }

    /// The whole form built on another thread, for a later ExecuteIndirect with more slabs than rounds
    fn want_whole(self: &Arc<Self>) {
        if !self.wanted.swap(true, Ordering::Relaxed) {
            let chain = self.clone();
            whole_thread().spawn(move || unsafe {
                if let Err(e) = chain.whole() { crate::xlog_warn!("D3D12.X: whole form of a split program: {e}"); }
            }).ok();
        }
    }
}

/// The thread of `want_whole`: an `xcl-pso*` one, so a huge whole form waits for its turn
/// (`driver_compile`) as the workers' builds do
fn whole_thread() -> std::thread::Builder {
    std::thread::Builder::new().name("xcl-pso-chain-whole".into())
}

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
        whole_code: whole.to_vec(), whole: OnceLock::new(), build_whole: Mutex::new(()), signature: OnceLock::new(),
        slabs: Default::default(), measured: Default::default(), wanted: Default::default(), warned: Default::default() });
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
        self.chain.slabs.fetch_max(slabs, Ordering::Relaxed);
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
/// list → the allocator it records on and the scratch its chains share: they are dispatched one
/// after another, with a barrier between them (the recordings keep it alive)
static SHARED: LazyLock<Mutex<HashMap<usize, (usize, std::sync::Weak<Scratch>)>>> = LazyLock::new(Default::default);

/// `list` begins another recording, which the GPU may run beside the last one.
pub(in crate::hle::d3d12x) fn list_reset(list: usize) {
    SHARED.lock().unwrap().remove(&list);
}

/// `to` records right after `from` on the same allocator: its chains go on in `from`'s scratch.
pub(in crate::hle::d3d12x) fn list_follows(from: usize, to: usize) {
    let mut shared = SHARED.lock().unwrap();
    if let Some(e) = shared.get(&from).cloned() {
        shared.insert(to, e);
    }
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

pub(in crate::hle::d3d12x) fn allocator_reset(allocator: usize) {
    let recordings = RECORDINGS.lock().unwrap().remove(&allocator);
    drop(recordings);
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
    let scratch = (allocator != 0 && list.GetType() != D3D12_COMMAND_LIST_TYPE_BUNDLE)
        .then(|| scratch_for(&chain.builder.device, list.as_raw() as usize, allocator, needed, group)).flatten();
    let most = scratch.as_ref().map_or(1, |(s, _)| (s.bytes / group).max(1));
    let slabs = pso::dispatch_slabs(grid, most);
    let Some((scratch, used)) = scratch else {
        list.SetPipelineState(&chain.whole()?);
        list.Dispatch(x, y, z);
        return Ok(());
    };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    let held = Arc::new(Recording { scratch, first, chain, indirect: None });
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(held.clone());
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    SLABS.fetch_add(slabs.len() as u64, Ordering::Relaxed);
    pso::record_dispatch_chain(list, grid, most, &held.scratch.buffer, used, std::iter::once(&held.first).chain(&held.chain.tail));
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
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    if allocator == 0 || list.GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE { return Ok(false); }
    let dev = &chain.builder.device;
    let group = u64::from(chain.words) * 4;
    let Some((scratch, _)) = scratch_for(dev, list.as_raw() as usize, allocator, scratch_budget(), group) else { return Ok(false) };
    let most = (scratch.bytes / group).max(1);
    let sig = match chain.signature.get() {
        Some(s) => s.clone(),
        None => { let s = pso::chain_command_signature(dev, &chain.builder.signature)?; chain.signature.get_or_init(|| s).clone() }
    };
    let before = chain.measured.load(Ordering::Relaxed).then(|| chain.slabs.load(Ordering::Relaxed));
    let rounds = indirect_rounds(seen, most, before, most_rounds());
    let pieces = chain.tail.len() as u32 + 1;
    let buffer = pso::chain_commands_buffer(dev, rounds, pieces, commands).ok_or("no buffer for the commands of a chained ExecuteIndirect")?;
    let full = chain.whole.get().cloned().map(|p| pso::dispatch_command_signature(dev).map(|s| (p, s))).transpose()?;
    // not built: asked for now, so that it is there before an ExecuteIndirect of the program
    // needs more slabs than its rounds (until then those are lost)
    if full.is_none() {
        chain.want_whole();
    }
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    pso::record_slicer(list, args, stride, commands, count, rounds, most, pieces, full.is_some(), &buffer)?;
    // the list uses `buffer` from here: the recording keeps it until the allocator is reset, also
    // when binding fails and nothing more is recorded
    let bound = bind();
    let at = pso::chain_tail_offset(rounds, pieces);
    if bound.is_ok() {
        // the slicer left either these arguments or the rounds empty. The whole form goes first
        // (each round's pieces start after a UAV barrier), with nothing of the chain before it:
        // after the pieces' ExecuteIndirects it did not run right on an RTX 3080 Ti (#2404)
        if let Some((full, sig)) = &full {
            list.SetPipelineState(full);
            list.ExecuteIndirect(sig, commands, &buffer, at + 16, None::<&ID3D12Resource>, 0);
        }
        pso::record_dispatch_chain_indirect(list, &sig, &buffer, rounds, &scratch.buffer, std::iter::once(&first).chain(&chain.tail));
    }
    let indirect = Indirect { commands: buffer, at, rounds, whole: full.is_some(), checked: bound.is_ok() && !predicated };
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(Arc::new(Recording { scratch, first, chain, indirect: Some(indirect) }));
    bound?;
    INDIRECT.fetch_add(1, Ordering::Relaxed);
    Ok(true)
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
            let builder = Arc::new(PipelineBuilder { device: dev.clone(), create: never, signature: rs, flags: D3D12_PIPELINE_STATE_FLAG_NONE, node: 0, record: 0 });
            let pso = first.as_raw() as usize;
            CHAINS.lock().unwrap().insert(pso, Arc::new(Chain { builder, tail: vec![first.clone()], words: 4, whole_code: Vec::new(), whole: OnceLock::new(),
                build_whole: Mutex::new(()), signature: OnceLock::new(), slabs: Default::default(), measured: Default::default(), wanted: Default::default(), warned: Default::default() }));
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let (raw_list, allocator) = (list.as_raw() as usize, alloc.as_raw() as usize);
            super::super::super::alloc_pool::created_on(raw_list, allocator, ty.0);
            let args = pso::chain_commands_buffer(&dev, 1, 1, 1).unwrap();
            let r = execute_indirect(&list, pso, allocator, args.GetGPUVirtualAddress(), 12, 1, None, &[[1, 1, 1]], false,
                || Err("unified memory is full".to_string()));
            assert_eq!(r, Err("unified memory is full".to_string()));
            // the program's first ExecuteIndirect asks for its whole form, before any slab is lost
            assert!(CHAINS.lock().unwrap()[&pso].wanted.load(Ordering::Relaxed));
            // the slicer's dispatch is recorded: its command buffer lives until the allocator is
            // reset, and the read-back does not take the rounds that were not recorded
            // under the title's predication the slicer may not run, so its words are not read back
            // (reported by @thehollowedone, #2404); without it they are
            for predicated in [true, false] {
                assert_eq!(execute_indirect(&list, pso, allocator, args.GetGPUVirtualAddress(), 12, 1, None, &[[1, 1, 1]], predicated, || Ok(())), Ok(true));
            }
            let kept = RECORDINGS.lock().unwrap().get(&allocator).map(|v| v.iter().filter_map(|r| r.indirect.as_ref().map(|i| i.checked)).collect::<Vec<_>>());
            assert_eq!(kept, Some(vec![false, false, true]));
            list.Close().unwrap();
            allocator_reset(allocator);
            list_reset(raw_list);
            forget(pso);
            super::super::super::alloc_pool::forget_list(raw_list);
        }
    }
}
