//! `XCL_DUMP_TARGETS=<work>[,<frames>]`: what a busy frame draws into. In the first `frames`
//! frames (default 1) with at least `work` draws and dispatches, every render target the frame
//! drew into (and the depth buffer bound most) is logged with how its draws went – ran, drew
//! nothing (a stand-in while the pipeline is built), went to the PC untranslated, skipped – and
//! the eight drawn into most are written next to the presented frame (`target<k>_<n>.bmp`) with
//! how many of their texels are not zero and how many differ from the first one. A black picture
//! over a scene that was drawn tells a lost composite from a scene that never was.
//! `XCL_DUMP_AFTER_S=<s>`: no frame is dumped until `s` seconds after the first present.
//!
//! A draw counts for the frame whose PresentX follows its recording (bundles, and lists executed
//! twice or never, are not exact); an ExecuteIndirect counts once, whatever its count.
//! Multisampled and non-2D targets are logged as not dumped; an array dumps its subresource 0.
//! The depth buffer is not copied: XCL does not know the state the title left it in, and a copy
//! from DEPTH_WRITE (possibly compressed) is invalid – the present list would not close.
//! Targets are kept by address only (not AddRef'd): `present::view_resource` hands one out only
//! while a view still names it, and one released meanwhile is logged as released.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// targets dumped of one frame
const DUMPED: usize = 8;

/// How a draw went: ran on its translated pipeline, drew nothing (a stand-in for a pipeline still
/// being built), went to the PC untranslated, or was skipped.
pub const RAN: usize = 0;
pub const STANDIN: usize = 1;
pub const PC: usize = 2;
pub const SKIPPED: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Config {
    /// draws and dispatches a frame needs
    work: u64,
    /// frames to dump
    frames: u64,
    /// no dump before this long after the first present
    after: Duration,
}

fn config() -> Option<Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    *C.get_or_init(|| {
        let v = std::env::var("XCL_DUMP_TARGETS").ok()?;
        let Some((work, frames)) = parse(&v) else {
            crate::xlog_warn!("D3D12.X: XCL_DUMP_TARGETS={v:?} is not <draws and dispatches>[,<frames>]: no targets dumped");
            return None;
        };
        let after = match std::env::var("XCL_DUMP_AFTER_S") {
            Ok(s) => parse_after(&s).unwrap_or_else(|| {
                crate::xlog_warn!("D3D12.X: XCL_DUMP_AFTER_S={s:?} is not a number of seconds: dumps start at once");
                Duration::ZERO
            }),
            Err(_) => Duration::ZERO,
        };
        Some(Config { work, frames, after })
    })
}

fn parse(v: &str) -> Option<(u64, u64)> {
    let mut it = v.split(',').map(|s| s.trim().parse::<u64>());
    let work = it.next()?.ok()?;
    let frames = match it.next() {
        Some(n) => n.ok()?,
        None => 1,
    };
    (work > 0 && frames > 0).then_some((work, frames))
}

/// `XCL_DUMP_AFTER_S`: seconds, fractions too (None: not a number of seconds)
fn parse_after(v: &str) -> Option<Duration> {
    v.trim().parse::<f64>().ok().and_then(|s| Duration::try_from_secs_f64(s).ok())
}

pub fn enabled() -> bool {
    config().is_some()
}

/// What a frame drew: its draws and dispatches, and the draws into each target by outcome.
#[derive(Default, Debug, PartialEq)]
pub struct Frame {
    work: u64,
    targets: HashMap<usize, [u64; 4]>,
    depth: HashMap<usize, [u64; 4]>,
}

impl Frame {
    fn draw(&mut self, targets: &[usize], depth: Option<usize>, outcome: usize) {
        self.work += 1;
        for &t in targets {
            self.targets.entry(t).or_default()[outcome] += 1;
        }
        if let Some(d) = depth {
            self.depth.entry(d).or_default()[outcome] += 1;
        }
    }

    /// targets by draws into them, most first (ties: by address, for a stable order)
    fn ranked(&self) -> Vec<(usize, [u64; 4])> {
        let mut v: Vec<(usize, [u64; 4])> = self.targets.iter().map(|(&t, &n)| (t, n)).collect();
        v.sort_by(|a, b| b.1.iter().sum::<u64>().cmp(&a.1.iter().sum()).then(a.0.cmp(&b.0)));
        v
    }

    /// the depth buffer drawn with most
    fn depth(&self) -> Option<(usize, [u64; 4])> {
        self.depth.iter().map(|(&d, &n)| (d, n)).max_by(|a, b| a.1.iter().sum::<u64>().cmp(&b.1.iter().sum()).then(b.0.cmp(&a.0)))
    }
}

/// the frame being recorded
static FRAME: Mutex<Option<Frame>> = Mutex::new(None);
/// frames dumped so far (their copies submitted)
static DONE: AtomicU64 = AtomicU64::new(0);
/// the frame to dump at the next present that gets as far as copying
static PENDING: Mutex<Option<Frame>> = Mutex::new(None);

/// A draw on `list` with `outcome`: it counts for each render target and the depth buffer bound.
pub unsafe fn note_draw(list: *mut c_void, outcome: usize) {
    if !enabled() {
        return;
    }
    let mut targets: Vec<usize> = super::list_x::bound_render_targets(list).into_iter().filter_map(super::present::rtv_resource).collect();
    targets.sort_unstable();
    targets.dedup();
    let depth = super::list_x::bound_dsv(list).and_then(super::present::rtv_resource);
    FRAME.lock().unwrap().get_or_insert_with(Frame::default).draw(&targets, depth, outcome);
}

/// A dispatch: work of the frame.
pub fn note_dispatch() {
    if enabled() {
        FRAME.lock().unwrap().get_or_insert_with(Frame::default).work += 1;
    }
}

/// Whether a frame of `work` is dumped, `done` frames dumped before it, `since` the first present.
fn due(c: &Config, work: u64, done: u64, since: Duration) -> bool {
    work >= c.work && done < c.frames && since >= c.after
}

/// The title presents: a frame that did enough work is dumped at the next present that copies
/// (one still waiting for that is kept, and this one dropped).
pub fn frame_end() {
    let Some(c) = config() else { return };
    static T0: OnceLock<Instant> = OnceLock::new();
    let t0 = *T0.get_or_init(Instant::now);
    let f = FRAME.lock().unwrap().take().unwrap_or_default();
    if !due(&c, f.work, DONE.load(Ordering::Relaxed), t0.elapsed()) {
        return;
    }
    PENDING.lock().unwrap().get_or_insert(f);
}

/// The frame to dump at this present, if any (`submitted` once its copies are on the GPU).
pub fn take_pending() -> Option<Frame> {
    if !enabled() {
        return None;
    }
    PENDING.lock().unwrap().take()
}

/// The copies of a frame taken by `take_pending` were submitted: it counts as dumped.
pub fn submitted() {
    DONE.fetch_add(1, Ordering::Relaxed);
}

/// Logs what frame `n` drew into; the targets to copy for its dump: (tag, resource, desc).
pub unsafe fn to_dump(n: u64, f: &Frame) -> Vec<(String, ID3D12Resource, D3D12_RESOURCE_DESC)> {
    let ranked = f.ranked();
    crate::xlog_info!("D3D12.X: target dump of frame {n}: {} draws and dispatches, {} render targets drawn into", f.work, ranked.len());
    let mut out = Vec::new();
    let depth = f.depth().map(|(r, c)| (usize::MAX, r, c));
    for (k, r, c) in ranked.into_iter().enumerate().map(|(k, (r, c))| (k, r, c)).chain(depth) {
        let res = super::present::view_resource(r);
        let d = res.as_ref().map(|x| x.GetDesc());
        let what = d.map_or("released".into(), |d| format!("{}x{} format {}", d.Width, d.Height, d.Format.0));
        let copyable = d.is_some_and(|d| d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count <= 1);
        let tag = match (k, copyable) {
            (usize::MAX, _) => " (not dumped: its state is not known)".into(),
            (k, true) if k < DUMPED => format!(" → target{k}"),
            (_, false) if d.is_some() => " (not dumped)".into(),
            _ => String::new(),
        };
        crate::xlog_info!(
            "D3D12.X: target dump:   {} {r:#x} {what}: {} draws (ran {}, stand-in {}, PC {}, skipped {}){tag}",
            if k == usize::MAX { "depth buffer" } else { "render target" },
            c.iter().sum::<u64>(),
            c[RAN],
            c[STANDIN],
            c[PC],
            c[SKIPPED]
        );
        if let (true, Some(res), Some(d)) = (k < DUMPED && copyable, res, d) {
            out.push((format!("target{k}"), res, d));
        }
    }
    out
}

/// Of the texels sampled in `data` (every 4th of every 4th row, as the dumps do): how many are not
/// all zero, how many differ from the first one sampled (a cleared target has none), and how many
/// were sampled; any format, by its texel size; texels past the end of `data` are not sampled.
fn census(data: &[u8], width: usize, height: usize, pitch: usize, format: DXGI_FORMAT) -> (u64, u64, u64) {
    let bytes = texel_bytes(format);
    let (mut lit, mut differ, mut n) = (0, 0, 0);
    let mut first: Option<&[u8]> = None;
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let at = y * pitch + x * bytes;
            let Some(t) = data.get(at..at + bytes) else { continue };
            n += 1;
            if t.iter().any(|&b| b != 0) {
                lit += 1;
            }
            if *first.get_or_insert(t) != t {
                differ += 1;
            }
        }
    }
    (lit, differ, n)
}

/// bytes of a texel of a render-target format (or of the plane of a planar format a footprint
/// names: a copied depth-stencil's depth plane is R32 / R24X8 typeless, 4 bytes)
fn texel_bytes(f: DXGI_FORMAT) -> usize {
    match f {
        DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT | DXGI_FORMAT_R32G32B32A32_TYPELESS => 16,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_UNORM | DXGI_FORMAT_R16G16B16A16_UINT | DXGI_FORMAT_R16G16B16A16_SNORM
        | DXGI_FORMAT_R16G16B16A16_SINT | DXGI_FORMAT_R16G16B16A16_TYPELESS | DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R32G32_UINT
        | DXGI_FORMAT_R32G32_SINT | DXGI_FORMAT_R32G32_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT | DXGI_FORMAT_R32G8X24_TYPELESS => 8,
        DXGI_FORMAT_R8G8_UNORM | DXGI_FORMAT_R8G8_UINT | DXGI_FORMAT_R8G8_SNORM | DXGI_FORMAT_R8G8_SINT | DXGI_FORMAT_R8G8_TYPELESS
        | DXGI_FORMAT_R16_FLOAT | DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_SNORM | DXGI_FORMAT_R16_SINT
        | DXGI_FORMAT_R16_TYPELESS | DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_B5G6R5_UNORM | DXGI_FORMAT_B5G5R5A1_UNORM | DXGI_FORMAT_B4G4R4A4_UNORM => 2,
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_SNORM | DXGI_FORMAT_R8_SINT | DXGI_FORMAT_A8_UNORM | DXGI_FORMAT_R8_TYPELESS => 1,
        // also D32 / R32 typeless, D24S8 / R24G8 typeless, R11G11B10
        _ => 4,
    }
}

/// A dump `tag` read back: how many of its texels are not all zero and differ from the first.
/// The texel size is that of the footprint's format – the copied plane's for a planar format,
/// not the resource's – and the bytes read stop at the buffer's end.
pub unsafe fn log_census(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, tag: &str) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let f = &fp.Footprint;
    let len = (rb.GetDesc().Width as usize).saturating_sub(fp.Offset as usize);
    let data = std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize), len);
    let (lit, differ, n) = census(data, f.Width as usize, f.Height as usize, f.RowPitch as usize, f.Format);
    rb.Unmap(0, None);
    crate::xlog_info!("D3D12.X: target dump: {tag}: {lit} of {n} texels sampled not zero, {differ} differ from the first texel (format {})", f.Format.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_a_work_threshold_and_a_frame_count() {
        assert_eq!(parse("200"), Some((200, 1)));
        assert_eq!(parse("200,3"), Some((200, 3)));
        assert_eq!(parse(" 50 , 2 "), Some((50, 2)));
        assert_eq!(parse("0"), None);
        assert_eq!(parse("200,0"), None);
        assert_eq!(parse("x"), None);
        assert_eq!(parse("200,x"), None);
    }

    #[test]
    fn dumps_wait_the_seconds_after_the_first_present() {
        assert_eq!(parse_after(" 30 "), Some(Duration::from_secs(30)));
        assert_eq!(parse_after("2.5"), Some(Duration::from_millis(2500)));
        assert_eq!(parse_after("soon"), None);
        assert_eq!(parse_after("-1"), None);
        assert_eq!(parse_after("inf"), None);
        let c = Config { work: 100, frames: 2, after: Duration::from_secs(30) };
        assert!(!due(&c, 500, 0, Duration::from_secs(29)), "too early");
        assert!(due(&c, 500, 0, Duration::from_secs(30)));
        assert!(!due(&c, 99, 0, Duration::from_secs(40)), "too little work");
        assert!(due(&c, 100, 1, Duration::from_secs(40)));
        assert!(!due(&c, 100, 2, Duration::from_secs(40)), "frames all dumped");
    }

    #[test]
    fn draws_count_per_target_by_outcome() {
        let mut f = Frame::default();
        f.draw(&[0x10, 0x20], Some(0x90), RAN);
        f.draw(&[0x10], Some(0x90), STANDIN);
        f.draw(&[0x10], None, PC);
        f.draw(&[0x20], Some(0x80), SKIPPED);
        f.draw(&[], None, RAN);
        assert_eq!(f.work, 5);
        assert_eq!(f.targets[&0x10], [1, 1, 1, 0]);
        assert_eq!(f.targets[&0x20], [1, 0, 0, 1]);
        assert_eq!(f.depth(), Some((0x90, [1, 1, 0, 0])));
        assert_eq!(Frame::default().depth(), None);
    }

    #[test]
    fn targets_rank_by_draws_and_texels_count_by_format() {
        let mut f = Frame::default();
        for (t, n) in [(0x30, 5), (0x10, 9), (0x20, 5)] {
            for _ in 0..n {
                f.draw(&[t], None, if n == 9 { SKIPPED } else { RAN });
            }
        }
        let order: Vec<(usize, u64)> = f.ranked().into_iter().map(|(t, c)| (t, c.iter().sum())).collect();
        assert_eq!(order, vec![(0x10, 9), (0x20, 5), (0x30, 5)]);
        // 8 x 8 texels of 8 bytes (R16G16B16A16_FLOAT), every 4th of every 4th row sampled:
        // (0,0) (4,0) (0,4) (4,4); one of them has a non-zero byte in its last half
        let (w, h, bytes) = (8usize, 8usize, 8usize);
        let pitch = w * bytes;
        let mut data = vec![0u8; pitch * h];
        data[4 * pitch + 4 * bytes + 7] = 1;
        data[pitch + bytes] = 9; // a texel not sampled
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R16G16B16A16_FLOAT), (1, 1, 4));
        // read as 4-byte texels the same bytes land elsewhere
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R8G8B8A8_UNORM), (0, 0, 4));
        assert_eq!(texel_bytes(DXGI_FORMAT_R11G11B10_FLOAT), 4);
        assert_eq!(texel_bytes(DXGI_FORMAT_R8_UNORM), 1);
        for f in [DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_B5G6R5_UNORM, DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_B4G4R4A4_UNORM] {
            assert_eq!(texel_bytes(f), 2, "{f:?}");
        }
    }

    #[test]
    fn a_cleared_target_is_not_zero_but_has_no_texel_that_differs() {
        // 8 x 8 of R8G8B8A8 cleared to (0, 0, 0, 1): every texel not zero, none differs
        let (w, h, pitch) = (8usize, 8usize, 32usize);
        let mut data: Vec<u8> = [0, 0, 0, 255].repeat(w * h);
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R8G8B8A8_UNORM), (4, 0, 4));
        // one draw over texel (4, 4)
        data[4 * pitch + 4 * 4] = 200;
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R8G8B8A8_UNORM), (4, 1, 4));
    }

    #[test]
    fn depth_formats_have_their_texel_sizes() {
        for f in [DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24G8_TYPELESS] {
            assert_eq!(texel_bytes(f), 4, "{f:?}");
        }
        assert_eq!(texel_bytes(DXGI_FORMAT_D32_FLOAT_S8X24_UINT), 8);
        assert_eq!(texel_bytes(DXGI_FORMAT_R32G8X24_TYPELESS), 8);
        assert_eq!(texel_bytes(DXGI_FORMAT_D16_UNORM), 2);
    }

    #[test]
    fn the_census_stops_at_the_end_of_the_buffer() {
        // 8 x 8 of D32S8X24 (8 bytes), a 256-byte pitch, but the buffer ends in the last row
        // (the footprint's last row is not padded to its pitch): its texels are not read
        let (w, h, pitch) = (8usize, 8usize, 256usize);
        let mut data = vec![1u8; pitch * 4 + 4 * 8 + 4];
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_D32_FLOAT_S8X24_UINT), (3, 0, 3));
        data.extend_from_slice(&[0; 4]);
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_D32_FLOAT_S8X24_UINT), (4, 1, 4));
    }
}
