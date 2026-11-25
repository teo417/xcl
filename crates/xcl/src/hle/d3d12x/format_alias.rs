//! Placed UAV/SRV texture views (`Create{ShaderResource,UnorderedAccess}ViewAtX`) and
//! format-aliased GPU copies across incompatible DXGI format families.
//!
//! Console titles write packed lighting data (sky, light probes, volumetric fog) through a placed
//! UAV texture view in an integer format (such as `R32_UINT`) and sample the same title address
//! through a placed SRV view in an incompatible format of the same texel width (such as
//! `R9G9B9E5_SHAREDEXP`). On PC D3D12:
//! * `CreateUnorderedAccessViewAtX` often passes a resource desc without `ALLOW_UNORDERED_ACCESS`
//!   (or with `R9G9B9E5_SHAREDEXP` / `SRGB`), which caused XCL to produce a null UAV view or
//!   promote `R9G9B9E5` to `R11G11B10_FLOAT` (breaking `R32_UINT` views and RGB9E5 decoding).
//! * `CreateShaderResourceViewAtX` with `ALLOW_RENDER_TARGET | ALLOW_UNORDERED_ACCESS` bits still
//!   set caused `R9G9B9E5_SHAREDEXP` to be promoted to `R11G11B10_FLOAT` and could register a
//!   GPU-written texture in `title_textures` (overwriting GPU data with zeros from CPU memory).
//! * Direct `CopyResource` / `CopyTextureRegion` between different DXGI format families is invalid
//!   on PC D3D12; texels of equal byte width must be transferred through an intermediate buffer
//!   (`CopyTextureRegion` across all mips and array/depth slices).
//!
//! Set `XCL_FORMAT_ALIAS=0` to disable format-alias syncing.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::list_x;
use super::real;

const STATE_COMMON: u32 = 0;
const STATE_RENDER_TARGET: u32 = 0x4;
const STATE_UAV: u32 = 0x8;
const STATE_SRV_NON_PIXEL: u32 = 0x40;
const STATE_SRV_ALL: u32 = 0x40 | 0x80;

const FLAG_RENDER_TARGET: u32 = 0x1;
const FLAG_DEPTH_STENCIL: u32 = 0x2;
const FLAG_UAV: u32 = 0x4;

/// Geometry & format summary of a texture for alias / buffer-through copy compatibility checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TexShape {
    pub dimension: i32,
    pub width: u64,
    pub height: u32,
    pub depth_or_array: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub samples: u32,
}

impl TexShape {
    pub(super) fn from_pc_desc(d: &D3D12_RESOURCE_DESC) -> Self {
        Self {
            dimension: d.Dimension.0,
            width: d.Width,
            height: d.Height,
            depth_or_array: d.DepthOrArraySize.max(1),
            mip_levels: d.MipLevels.max(1),
            format: d.Format.0 as u32,
            samples: d.SampleDesc.Count.max(1),
        }
    }

    pub(super) fn subresource_count(&self) -> u32 {
        let slices = if self.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D.0 {
            1
        } else {
            self.depth_or_array as u32
        };
        (self.mip_levels as u32).saturating_mul(slices).max(1)
    }
}

/// `(bytes_per_element, is_4x4_block_compressed)` for common DXGI formats.
pub(super) fn format_element_bytes(fmt: u32) -> Option<(u32, bool)> {
    match fmt {
        // 128-bit uncompressed (16 bytes/texel)
        1..=4 => Some((16, false)),
        // 96-bit uncompressed (12 bytes/texel)
        5..=8 => Some((12, false)),
        // 64-bit uncompressed (8 bytes/texel): R16G16B16A16_*, R32G32_*
        9..=18 => Some((8, false)),
        // 32-bit uncompressed (4 bytes/texel): R10G10B10A2_*, R11G11B10_FLOAT, R8G8B8A8_*, R16G16_*, R32_*, R9G9B9E5_SHAREDEXP, B8G8R8A8/X8_*
        23..=43 | 67 | 87..=93 => Some((4, false)),
        // 16-bit uncompressed (2 bytes/texel): R8G8_*, R16_*, B5G6R5, B5G5R5A1, B4G4R4A4
        48..=59 | 85 | 86 | 115 => Some((2, false)),
        // 8-bit uncompressed (1 byte/texel): R8_*, A8_UNORM
        60..=65 => Some((1, false)),
        // 4x4 block-compressed, 8 bytes/block: BC1, BC4
        70..=72 | 79..=81 => Some((8, true)),
        // 4x4 block-compressed, 16 bytes/block: BC2, BC3, BC5, BC6H, BC7
        73..=78 | 82..=84 | 94..=99 => Some((16, true)),
        _ => None,
    }
}

/// Canonical typeless family representative for a DXGI format.
pub(super) fn typeless_family(fmt: u32) -> u32 {
    match fmt {
        1..=4 => 1,   // R32G32B32A32_TYPELESS
        5..=8 => 5,   // R32G32B32_TYPELESS
        9..=14 => 9,  // R16G16B16A16_TYPELESS
        15..=18 => 15, // R32G32_TYPELESS
        19..=22 => 19, // R32G8X24_TYPELESS
        23..=25 => 23, // R10G10B10A2_TYPELESS
        27..=32 => 27, // R8G8B8A8_TYPELESS
        33..=38 => 33, // R16G16_TYPELESS
        39..=43 => 39, // R32_TYPELESS
        44..=47 => 44, // R24G8_TYPELESS
        48..=52 => 48, // R8G8_TYPELESS
        53..=59 => 53, // R16_TYPELESS
        60..=64 => 60, // R8_TYPELESS
        70..=72 => 70, // BC1_TYPELESS
        73..=75 => 73, // BC2_TYPELESS
        76..=78 => 76, // BC3_TYPELESS
        79..=81 => 79, // BC4_TYPELESS
        82..=84 => 82, // BC5_TYPELESS
        87 | 90 | 91 => 90, // B8G8R8A8_TYPELESS
        88 | 92 | 93 => 92, // B8G8R8X8_TYPELESS
        94..=96 => 94, // BC6H_TYPELESS
        97..=99 => 97, // BC7_TYPELESS
        other => other,
    }
}

/// Whether `dst_fmt` and `src_fmt` belong to different DXGI typeless families (so direct
/// `CopyResource` is illegal on PC D3D12 and a buffer-through `CopyTextureRegion` is needed).
pub(super) fn needs_buffer_through(dst_fmt: u32, src_fmt: u32) -> bool {
    dst_fmt != 0 && src_fmt != 0 && typeless_family(dst_fmt) != typeless_family(src_fmt)
}

/// Whether two textures can exchange texels via an intermediate buffer (`CopyTextureRegion`).
pub(super) fn can_copy_through_buffer(dst: &TexShape, src: &TexShape) -> bool {
    if dst.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0
        || src.dimension == D3D12_RESOURCE_DIMENSION_BUFFER.0
        || dst.dimension != src.dimension
        || dst.samples > 1
        || src.samples > 1
        || dst.mip_levels != src.mip_levels
        || dst.depth_or_array != src.depth_or_array
    {
        return false;
    }
    let (Some((db, dbc)), Some((sb, sbc))) = (format_element_bytes(dst.format), format_element_bytes(src.format)) else {
        return false;
    };
    if db != sb {
        return false;
    }
    match (dbc, sbc) {
        (false, false) | (true, true) => dst.width == src.width && dst.height == src.height,
        (true, false) => src.width == dst.width.div_ceil(4) && src.height == dst.height.div_ceil(4),
        (false, true) => dst.width == src.width.div_ceil(4) && dst.height == src.height.div_ceil(4),
    }
}

/// Choose a UAV-capable PC resource format that accepts `view_fmt` (or `rdesc_fmt` if `view_fmt == 0`).
pub(super) fn uav_resource_format(rdesc_fmt: u32, view_fmt: u32) -> u32 {
    let target = if view_fmt != 0 { view_fmt } else { rdesc_fmt };
    // R9G9B9E5_SHAREDEXP cannot have ALLOW_UNORDERED_ACCESS on PC; use R32_TYPELESS when the
    // view wants 32-bit integer/float UAV access, otherwise let device_x handle it.
    if rdesc_fmt == 67 && view_fmt != 0 && view_fmt != 67 {
        return typeless_family(view_fmt);
    }
    // sRGB formats cannot have ALLOW_UNORDERED_ACCESS on PC; use their typeless family.
    if matches!(target, 29 | 91 | 93) {
        return typeless_family(target);
    }
    // If the view requests a different family than the resource desc (e.g. R32_UINT view over
    // an R9G9B9E5 or R11G11B10 resource), make the UAV texture typeless in the view's family.
    if view_fmt != 0 && typeless_family(rdesc_fmt) != typeless_family(view_fmt) {
        return typeless_family(view_fmt);
    }
    // Use typeless when available so both typed UAV and SRV views match cleanly.
    let fam = typeless_family(target);
    match fam {
        1 | 9 | 15 | 23 | 27 | 33 | 39 | 48 | 53 | 60 | 90 => fam,
        _ => target,
    }
}

/// Sanitize a console resource desc for `CreateUnorderedAccessViewAtX`: guarantee
/// `ALLOW_UNORDERED_ACCESS` and a UAV-compatible format so PC D3D12 never drops it to a null view.
pub(super) fn prepare_placed_uav_desc(rdesc: &[u8; 0x38], view_fmt: u32) -> [u8; 0x38] {
    let mut x = *rdesc;
    x[0x34..].fill(0);
    let rdesc_fmt = u32::from_le_bytes(x[0x20..0x24].try_into().unwrap());
    let mut flags = u32::from_le_bytes(x[0x30..0x34].try_into().unwrap());
    let is_depth = flags & FLAG_DEPTH_STENCIL != 0 || matches!(rdesc_fmt, 19 | 20 | 40 | 44 | 45 | 55 | 118..=120);
    if !is_depth {
        // Strip RTV bit on placed UAV textures so integer/typeless UAV formats are never rejected
        // by render-target format restrictions.
        flags = (flags & !FLAG_RENDER_TARGET) | FLAG_UAV;
        let fmt = uav_resource_format(rdesc_fmt, view_fmt);
        x[0x20..0x24].copy_from_slice(&fmt.to_le_bytes());
    } else {
        flags |= FLAG_UAV;
    }
    x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    x
}

/// Sanitize a console resource desc for `CreateShaderResourceViewAtX`: strip `ALLOW_RENDER_TARGET`
/// and `ALLOW_UNORDERED_ACCESS` so `R9G9B9E5_SHAREDEXP` stays format 67 and placed SRVs never
/// retire or overwrite GPU-owned target tracking at `va`.
pub(super) fn prepare_placed_srv_desc(rdesc: &[u8; 0x38], view_fmt: u32) -> [u8; 0x38] {
    let mut x = *rdesc;
    x[0x34..].fill(0);
    let rdesc_fmt = u32::from_le_bytes(x[0x20..0x24].try_into().unwrap());
    let mut flags = u32::from_le_bytes(x[0x30..0x34].try_into().unwrap());
    flags &= !(FLAG_RENDER_TARGET | FLAG_UAV);
    x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    if view_fmt != 0 && typeless_family(rdesc_fmt) != typeless_family(view_fmt) && format_element_bytes(view_fmt).is_some() {
        // Keep R9G9B9E5_SHAREDEXP (67) and other single-member formats verbatim; use typeless
        // only when it belongs to a multi-format typeless family.
        let fmt = match typeless_family(view_fmt) {
            f if f == view_fmt => view_fmt,
            f if matches!(f, 1 | 9 | 15 | 23 | 27 | 33 | 39 | 48 | 53 | 60 | 70 | 73 | 76 | 79 | 82 | 90 | 92 | 94 | 97) => f,
            _ => view_fmt,
        };
        x[0x20..0x24].copy_from_slice(&fmt.to_le_bytes());
    }
    x
}

/// Ownership of a resource participating in format-alias tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Owner {
    /// Internal texture created by `dev_create_uav_at_x` (resting state: `UNORDERED_ACCESS`).
    PlacedUav,
    /// Internal texture created by `dev_create_srv_at_x` (resting state outside open list: `COMMON`).
    PlacedSrv,
    /// Title-created texture from `CreatePlacedResourceX` (tracked PC state).
    TitlePlaced,
}

struct TrackedTex {
    device: usize,
    va: usize,
    shape: TexShape,
    owner: Owner,
    /// Per-subresource PC states (only tracked for `Owner::TitlePlaced`).
    pc_states: Vec<u32>,
    is_writer: bool,
}

impl TrackedTex {
    fn new(device: usize, va: usize, shape: TexShape, owner: Owner, initial_state: u32, is_writer: bool) -> Self {
        let n = if owner == Owner::TitlePlaced { shape.subresource_count().max(1) as usize } else { 0 };
        Self { device, va, shape, owner, pc_states: vec![initial_state; n], is_writer }
    }

    fn uniform_state(&self) -> Option<u32> {
        let &first = self.pc_states.first()?;
        self.pc_states.iter().all(|&s| s == first).then_some(first)
    }

    fn update_state(&mut self, subresource: u32, after: u32) {
        if subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES {
            self.pc_states.fill(after);
        } else if let Some(slot) = self.pc_states.get_mut(subresource as usize) {
            *slot = after;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AliasPair {
    writer: usize,
    reader: usize,
    device: usize,
    buffer_through: bool,
    writer_owner: Owner,
    reader_owner: Owner,
}

struct AliasTable {
    /// Cached internal placed-UAV textures: `(va, prepared_desc)` -> owned `ID3D12Resource`.
    placed_uavs: HashMap<(u64, [u8; 0x38]), ID3D12Resource>,
    /// Cached internal placed-SRV textures: `(va, prepared_desc)` -> owned `ID3D12Resource`.
    placed_srvs: HashMap<(u64, [u8; 0x38]), ID3D12Resource>,
    /// All tracked textures by raw resource pointer.
    by_res: HashMap<usize, TrackedTex>,
    /// Title VA -> tracked resource pointers at that exact VA.
    by_va: HashMap<usize, Vec<usize>>,
    /// Active `(writer -> reader)` sync pairs.
    pairs: Vec<AliasPair>,
    /// Pairs whose writer is an internal `PlacedUav` (must be synced on UAV barriers).
    placed_uav_pairs: usize,
}

unsafe impl Send for AliasTable {}

impl AliasTable {
    fn new() -> Self {
        Self {
            placed_uavs: HashMap::new(),
            placed_srvs: HashMap::new(),
            by_res: HashMap::new(),
            by_va: HashMap::new(),
            pairs: Vec::new(),
            placed_uav_pairs: 0,
        }
    }

    fn has_writer_at(&self, va: usize, shape: &TexShape) -> bool {
        self.by_va.get(&va).is_some_and(|list| {
            list.iter().any(|r| {
                self.by_res.get(r).is_some_and(|t| t.is_writer && can_copy_through_buffer(shape, &t.shape))
            })
        })
    }

    /// Register `res` at `va` and link any compatible `(writer -> reader)` pairs at the same VA.
    /// Returns any existing reader resources that should be marked `gpu_written` in `title_textures`.
    fn insert(&mut self, res: usize, tex: TrackedTex) -> Vec<usize> {
        self.remove(res);
        let mut mark_gpu_written = Vec::new();
        let mut new_pairs = Vec::new();
        if let Some(peers) = self.by_va.get(&tex.va) {
            for &peer in peers {
                let Some(p) = self.by_res.get(&peer) else {
                    continue;
                };
                if p.device != tex.device {
                    continue;
                }
                // Link writer -> reader when one is a PlacedUav (always needs sync to its SRV alias)
                // OR when their formats are in different typeless families (format-aliased GPU textures).
                if tex.is_writer && !p.is_writer && can_copy_through_buffer(&p.shape, &tex.shape) {
                    let bt = needs_buffer_through(p.shape.format, tex.shape.format);
                    if tex.owner == Owner::PlacedUav || bt {
                        new_pairs.push(AliasPair {
                            writer: res,
                            reader: peer,
                            device: tex.device,
                            buffer_through: bt,
                            writer_owner: tex.owner,
                            reader_owner: p.owner,
                        });
                        mark_gpu_written.push(peer);
                    }
                } else if !tex.is_writer && p.is_writer && can_copy_through_buffer(&tex.shape, &p.shape) {
                    let bt = needs_buffer_through(tex.shape.format, p.shape.format);
                    if p.owner == Owner::PlacedUav || bt {
                        new_pairs.push(AliasPair {
                            writer: peer,
                            reader: res,
                            device: tex.device,
                            buffer_through: bt,
                            writer_owner: p.owner,
                            reader_owner: tex.owner,
                        });
                        mark_gpu_written.push(res);
                    }
                }
            }
        }
        for pair in new_pairs {
            self.add_pair(pair);
        }
        self.by_va.entry(tex.va).or_default().push(res);
        if tex.owner == Owner::TitlePlaced {
            TRACKED_TITLE_TEX.fetch_add(1, Ordering::Relaxed);
        }
        self.by_res.insert(res, tex);
        mark_gpu_written
    }

    /// Evict cached internal `PlacedUav`/`PlacedSrv` resources at `va` whose shape no longer
    /// matches `keep_shape` (or all internal resources at `va` when `keep_shape` is `None` and
    /// no `TitlePlaced` peers remain at `va`), returning the evicted `ID3D12Resource`s to be
    /// deferred-released outside the lock.
    fn evict_stale_internal_at(&mut self, va: usize, keep_shape: Option<&TexShape>) -> Vec<ID3D12Resource> {
        let mut evicted_ptrs = Vec::new();
        if let Some(list) = self.by_va.get(&va) {
            let has_title = list.iter().any(|r| self.by_res.get(r).is_some_and(|t| t.owner == Owner::TitlePlaced));
            if keep_shape.is_none() && has_title {
                return Vec::new();
            }
            for &r in list {
                let Some(t) = self.by_res.get(&r) else { continue };
                if t.owner == Owner::TitlePlaced {
                    continue;
                }
                let stale = match keep_shape {
                    Some(s) => !can_copy_through_buffer(s, &t.shape),
                    None => true,
                };
                if stale {
                    evicted_ptrs.push(r);
                }
            }
        }
        let mut dropped = Vec::new();
        for r in evicted_ptrs {
            self.remove(r);
            let uav_keys: Vec<_> = self.placed_uavs.iter().filter(|(_, v)| v.as_raw() as usize == r).map(|(k, _)| *k).collect();
            for k in uav_keys {
                if let Some(res) = self.placed_uavs.remove(&k) {
                    dropped.push(res);
                }
            }
            let srv_keys: Vec<_> = self.placed_srvs.iter().filter(|(_, v)| v.as_raw() as usize == r).map(|(k, _)| *k).collect();
            for k in srv_keys {
                if let Some(res) = self.placed_srvs.remove(&k) {
                    dropped.push(res);
                }
            }
        }
        dropped
    }

    fn add_pair(&mut self, pair: AliasPair) {
        if self.pairs.iter().any(|p| p.writer == pair.writer && p.reader == pair.reader) {
            return;
        }
        if pair.writer_owner == Owner::PlacedUav {
            self.placed_uav_pairs += 1;
            ACTIVE_PLACED_UAV_PAIRS.store(self.placed_uav_pairs, Ordering::Relaxed);
        }
        ACTIVE_PAIRS.store(self.pairs.len() + 1, Ordering::Relaxed);
        self.pairs.push(pair);
    }

    fn remove(&mut self, res: usize) -> Option<usize> {
        if let Some(old) = self.by_res.remove(&res) {
            if old.owner == Owner::TitlePlaced {
                TRACKED_TITLE_TEX.fetch_sub(1, Ordering::Relaxed);
            }
            if let Some(list) = self.by_va.get_mut(&old.va) {
                list.retain(|&r| r != res);
                if list.is_empty() {
                    self.by_va.remove(&old.va);
                }
            }
            let before = self.pairs.len();
            self.pairs.retain(|p| p.writer != res && p.reader != res);
            if self.pairs.len() != before {
                self.placed_uav_pairs = self.pairs.iter().filter(|p| p.writer_owner == Owner::PlacedUav).count();
                ACTIVE_PLACED_UAV_PAIRS.store(self.placed_uav_pairs, Ordering::Relaxed);
                ACTIVE_PAIRS.store(self.pairs.len(), Ordering::Relaxed);
            }
            return Some(old.va);
        }
        None
    }
}

static TABLE: Mutex<Option<AliasTable>> = Mutex::new(None);
static TRACKED_TITLE_TEX: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_PAIRS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_PLACED_UAV_PAIRS: AtomicUsize = AtomicUsize::new(0);

/// Defer releasing internal resources evicted from `placed_uavs` / `placed_srvs` until in-flight
/// GPU work has completed.
unsafe fn defer_drop_resources(evicted: Vec<ID3D12Resource>) {
    for r in evicted {
        super::late_release::hold(r.into_raw() as usize);
    }
}

/// D3D12 resource states legal in a `ResourceBarrier` on a `COMPUTE` command list for textures
/// (`COMMON`, `UNORDERED_ACCESS`, `NON_PIXEL_SHADER_RESOURCE`, `COPY_DEST`, `COPY_SOURCE`).
const COMPUTE_VALID_STATES: u32 = 0x8 | 0x40 | 0x400 | 0x800;

fn state_valid_on_list(ty: D3D12_COMMAND_LIST_TYPE, st: u32) -> bool {
    ty == D3D12_COMMAND_LIST_TYPE_DIRECT || (st & !COMPUTE_VALID_STATES == 0)
}

/// Per-open-list state for `PlacedSrv` resources transitioned into read state on that list
/// (returned to `COMMON` before `Close()`), plus last synced `bind::activity()`.
#[derive(Default)]
struct ListSyncState {
    /// `PlacedSrv` resource -> current state on this open list (`SRV_ALL` or `SRV_NON_PIXEL`).
    srv_states: HashMap<usize, u32>,
    last_activity: (u64, u64),
    list_type: Option<D3D12_COMMAND_LIST_TYPE>,
}

static LIST_STATES: Mutex<Option<HashMap<usize, ListSyncState>>> = Mutex::new(None);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_FORMAT_ALIAS").is_ok_and(|v| v.trim() == "0"))
}

unsafe fn create_internal_texture(dev: *mut c_void, pc_desc: &[u8; 0x38], initial_state: u32) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut out: *mut c_void = std::ptr::null_mut();
    let hr = real::dev_create_committed_resource(
        dev,
        &heap as *const _ as usize,
        0,
        pc_desc.as_ptr() as usize,
        initial_state as usize,
        0,
        &ID3D12Resource::IID as *const _ as *const windows_sys::core::GUID,
        &mut out,
    );
    if hr < 0 || out.is_null() {
        return None;
    }
    Some(ID3D12Resource::from_raw(out))
}

/// Obtain or create the host UAV texture backing `CreateUnorderedAccessViewAtX` at `va`.
pub(super) unsafe fn placed_uav_resource(dev: *mut c_void, rdesc: *const u8, va: u64, view_fmt: u32) -> *mut c_void {
    if rdesc.is_null() || va == 0 || !crate::gpu::shader::readable(rdesc as usize, 0x38) {
        return std::ptr::null_mut();
    }
    let raw_x: [u8; 0x38] = (rdesc as *const [u8; 0x38]).read_unaligned();
    let prepared = prepare_placed_uav_desc(&raw_x, view_fmt);
    let key = (va, prepared);

    {
        let mut g = TABLE.lock().unwrap();
        let t = g.get_or_insert_with(AliasTable::new);
        if let Some(res) = t.placed_uavs.get(&key) {
            return res.as_raw();
        }
    }

    let pc = super::device_x::pc_resource_desc(&prepared);
    let flags = u32::from_le_bytes(prepared[0x30..0x34].try_into().unwrap());
    let is_depth_uav = flags & (FLAG_DEPTH_STENCIL | FLAG_UAV) == (FLAG_DEPTH_STENCIL | FLAG_UAV);
    let init_state = if is_depth_uav {
        super::device_x::pc_initial_state(flags, STATE_UAV)
    } else {
        STATE_UAV
    };
    let Some(res) = create_internal_texture(dev, &pc, init_state) else {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_warn!("D3D12.X: placed UAV texture at {va:#x} (view fmt {view_fmt}) could not be created");
        }
        return std::ptr::null_mut();
    };
    let raw = res.as_raw();
    if is_depth_uav {
        super::depth_uav::register_if_depth_uav(dev, raw, &prepared, init_state, STATE_UAV);
    }
    let pc_desc = res.GetDesc();
    let shape = TexShape::from_pc_desc(&pc_desc);
    let mark = {
        let mut g = TABLE.lock().unwrap();
        let t = g.get_or_insert_with(AliasTable::new);
        let entry = t.placed_uavs.entry(key).or_insert(res);
        let raw_in = entry.as_raw();
        if raw_in == raw && !is_depth_uav && enabled() {
            t.insert(
                raw as usize,
                TrackedTex::new(dev as usize, va as usize, shape, Owner::PlacedUav, STATE_UAV, true),
            )
        } else {
            Vec::new()
        }
    };
    for reader in mark {
        super::title_textures::note_gpu_write(reader);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let n = LOGGED.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: placed UAV texture at {va:#x} ({}x{}x{}, {} mips, pc fmt {}, view fmt {view_fmt}) -> {raw:p} ({} created)",
            pc_desc.Width,
            pc_desc.Height,
            pc_desc.DepthOrArraySize,
            pc_desc.MipLevels,
            pc_desc.Format.0,
            n + 1
        );
    }
    raw
}

/// Obtain or create the host SRV texture backing `CreateShaderResourceViewAtX` at `va`.
/// If a GPU writer already occupies `va` with a compatible shape, creates a dedicated GPU-synced
/// SRV texture (skipping `title_textures` so CPU zeros never overwrite GPU writes); otherwise
/// delegates to `create_placed_resource_x` and registers it as a reader in case a placed UAV
/// appears at `va` later.
pub(super) unsafe fn placed_srv_resource(dev: *mut c_void, rdesc: *const u8, va: u64, view_fmt: u32) -> *mut c_void {
    if rdesc.is_null() || va == 0 || !crate::gpu::shader::readable(rdesc as usize, 0x38) {
        return std::ptr::null_mut();
    }
    let raw_x: [u8; 0x38] = (rdesc as *const [u8; 0x38]).read_unaligned();
    let prepared = prepare_placed_srv_desc(&raw_x, view_fmt);
    let key = (va, prepared);
    let pc = super::device_x::pc_resource_desc(&prepared);
    let pc_desc = std::ptr::read_unaligned(pc.as_ptr() as *const D3D12_RESOURCE_DESC);
    let shape = TexShape::from_pc_desc(&pc_desc);

    let has_writer = {
        let mut g = TABLE.lock().unwrap();
        let t = g.get_or_insert_with(AliasTable::new);
        if let Some(res) = t.placed_srvs.get(&key) {
            return res.as_raw();
        }
        enabled() && t.has_writer_at(va as usize, &shape)
    };

    let res = if has_writer {
        // Dedicated GPU-synced SRV texture in COMMON state (not registered in title_textures).
        match create_internal_texture(dev, &pc, STATE_COMMON) {
            Some(r) => r,
            None => return std::ptr::null_mut(),
        }
    } else {
        let mut r = std::ptr::null_mut();
        let hr = super::device_x::create_placed_resource_x(
            dev,
            va,
            prepared.as_ptr(),
            0,
            std::ptr::null(),
            &ID3D12Resource::IID as *const _ as *const windows_sys::core::GUID,
            &mut r,
        );
        if hr < 0 || r.is_null() {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::xlog_warn!("D3D12.X: placed SRV texture at {va:#x}: resource creation failed ({hr:#x})");
            }
            return std::ptr::null_mut();
        }
        ID3D12Resource::from_raw(r)
    };

    let raw = res.as_raw();
    let owner = if has_writer { Owner::PlacedSrv } else { Owner::TitlePlaced };
    let mark = {
        let mut g = TABLE.lock().unwrap();
        let t = g.get_or_insert_with(AliasTable::new);
        let entry = t.placed_srvs.entry(key).or_insert(res);
        let raw_in = entry.as_raw();
        if raw_in == raw && enabled() {
            t.insert(
                raw as usize,
                TrackedTex::new(dev as usize, va as usize, shape, owner, STATE_COMMON, false),
            )
        } else {
            Vec::new()
        }
    };
    for reader in mark {
        super::title_textures::note_gpu_write(reader);
    }
    raw
}

/// Register a title-created placed texture from `device_x::create_placed_resource_x`.
/// Returns `true` if the texture is already aliased to a GPU writer at `va` (so callers should
/// call `title_textures::note_gpu_write` and avoid CPU uploads over GPU-written texels).
pub(super) fn register_title_texture(dev: usize, res: usize, va: usize, pc_desc: &D3D12_RESOURCE_DESC, flags: u32, initial_state: u32) -> bool {
    if !enabled() || res == 0 || va == 0 || pc_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return false;
    }
    let shape = TexShape::from_pc_desc(pc_desc);
    if format_element_bytes(shape.format).is_none() || shape.samples > 1 {
        return false;
    }
    let is_writer = flags & (FLAG_RENDER_TARGET | FLAG_UAV) != 0;
    let (mark, evicted) = {
        let mut g = TABLE.lock().unwrap();
        let t = g.get_or_insert_with(AliasTable::new);
        let evicted = t.evict_stale_internal_at(va, Some(&shape));
        let mark = t.insert(
            res,
            TrackedTex::new(dev, va, shape, Owner::TitlePlaced, initial_state, is_writer),
        );
        (mark, evicted)
    };
    if !evicted.is_empty() {
        unsafe { defer_drop_resources(evicted) };
    }
    let mut self_gpu_written = false;
    for r in mark {
        if r == res {
            self_gpu_written = true;
        }
        super::title_textures::note_gpu_write(r);
    }
    self_gpu_written
}

/// Remove a released title resource from format-alias tracking, evicting any orphaned internal
/// `PlacedUav`/`PlacedSrv` textures at the same VA via deferred GPU release.
pub(super) fn forget(res: usize) {
    let evicted = {
        let mut g = TABLE.lock().unwrap();
        if let Some(t) = g.as_mut() {
            let mut dropped = Vec::new();
            let va = t.remove(res);
            let uav_keys: Vec<_> = t.placed_uavs.iter().filter(|(_, v)| v.as_raw() as usize == res).map(|(k, _)| *k).collect();
            for k in uav_keys {
                if let Some(r) = t.placed_uavs.remove(&k) {
                    dropped.push(r);
                }
            }
            let srv_keys: Vec<_> = t.placed_srvs.iter().filter(|(_, v)| v.as_raw() as usize == res).map(|(k, _)| *k).collect();
            for k in srv_keys {
                if let Some(r) = t.placed_srvs.remove(&k) {
                    dropped.push(r);
                }
            }
            if let Some(va) = va {
                dropped.extend(t.evict_stale_internal_at(va, None));
            }
            dropped
        } else {
            Vec::new()
        }
    };
    if !evicted.is_empty() {
        unsafe { defer_drop_resources(evicted) };
    }
    if let Some(m) = LIST_STATES.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            s.srv_states.remove(&res);
        }
    }
}

/// Track state transitions of title-owned textures (even before an alias peer is created at the
/// same VA) and collect any writers leaving `UNORDERED_ACCESS` / `RENDER_TARGET` whose
/// format-aliased readers need a copy. Ignores split-barrier `BEGIN_ONLY` flags.
pub(super) fn note_title_transition(res: usize, subresource: u32, before: u32, after: u32, flags: u32, writers_to_sync: &mut Vec<(usize, u32)>) {
    const BARRIER_BEGIN_ONLY: u32 = 1;
    if TRACKED_TITLE_TEX.load(Ordering::Relaxed) == 0 || flags == BARRIER_BEGIN_ONLY {
        return;
    }
    let mut g = TABLE.lock().unwrap();
    let Some(t) = g.as_mut() else {
        return;
    };
    let Some(tex) = t.by_res.get_mut(&res) else {
        return;
    };
    if tex.owner != Owner::TitlePlaced {
        return;
    }
    tex.update_state(subresource, after);
    let uniform = tex.uniform_state();
    let is_writer = tex.is_writer;
    let leaving_write = (before & (STATE_UAV | STATE_RENDER_TARGET) != 0) && (after & (STATE_UAV | STATE_RENDER_TARGET) == 0);
    if is_writer && leaving_write && ACTIVE_PAIRS.load(Ordering::Relaxed) != 0 && t.pairs.iter().any(|p| p.writer == res) {
        if let Some(st) = uniform {
            writers_to_sync.push((res, st));
        }
    }
}

/// Copy all subresources of `src` into `dst` through an intermediate buffer (`CopyTextureRegion`),
/// preserving raw texel bits across incompatible DXGI format families (e.g. `R32_UINT` ->
/// `R9G9B9E5_SHAREDEXP` or `BCn` <-> `R32G32B32A32_UINT`).
/// Callers must ensure `src` is in `COPY_SOURCE` and `dst` is in `COPY_DEST`.
pub(super) unsafe fn copy_all_subresources_through_buffer(
    list: *mut c_void,
    dev: &ID3D12Device,
    dst: &ID3D12Resource,
    src: &ID3D12Resource,
) -> bool {
    let dst_desc = dst.GetDesc();
    let src_desc = src.GetDesc();
    let dst_shape = TexShape::from_pc_desc(&dst_desc);
    let src_shape = TexShape::from_pc_desc(&src_desc);
    if !can_copy_through_buffer(&dst_shape, &src_shape) {
        return false;
    }
    let count = dst_shape.subresource_count().min(src_shape.subresource_count());
    for sub in 0..count {
        if !copy_one_subresource_through_buffer(list, dev, dst, &dst_desc, sub, src, &src_desc, sub) {
            return false;
        }
    }
    true
}

struct OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION);

impl Drop for OwnedCopyLoc {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.0.pResource) };
    }
}

/// Verify two single-subresource footprints have identical element byte widths and compatible
/// block/texel dimensions (preventing accidental aliasing of incompatible formats that merely
/// share a 256-byte-aligned `RowPitch`).
fn subresource_footprints_compatible(dst_fmt: u32, dst_fp: &D3D12_SUBRESOURCE_FOOTPRINT, src_fmt: u32, src_fp: &D3D12_SUBRESOURCE_FOOTPRINT) -> bool {
    let (Some((db, dbc)), Some((sb, sbc))) = (format_element_bytes(dst_fmt), format_element_bytes(src_fmt)) else {
        return false;
    };
    if db != sb || dst_fp.Depth != src_fp.Depth {
        return false;
    }
    let eff = |bc: bool, w: u32, h: u32| -> (u32, u32) {
        if bc { (w.div_ceil(4), h.div_ceil(4)) } else { (w, h) }
    };
    eff(dbc, dst_fp.Width, dst_fp.Height) == eff(sbc, src_fp.Width, src_fp.Height)
}

unsafe fn copy_one_subresource_through_buffer(
    list: *mut c_void,
    dev: &ID3D12Device,
    dst: &ID3D12Resource,
    dst_desc: &D3D12_RESOURCE_DESC,
    dst_sub: u32,
    src: &ID3D12Resource,
    src_desc: &D3D12_RESOURCE_DESC,
    src_sub: u32,
) -> bool {
    let mut src_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut dst_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut src_rows = 0u32;
    let mut dst_rows = 0u32;
    let mut src_total = 0u64;
    let mut dst_total = 0u64;
    dev.GetCopyableFootprints(src_desc, src_sub, 1, 0, Some(&mut src_fp), Some(&mut src_rows), None, Some(&mut src_total));
    dev.GetCopyableFootprints(dst_desc, dst_sub, 1, 0, Some(&mut dst_fp), Some(&mut dst_rows), None, Some(&mut dst_total));
    if src_total == 0
        || src_total == u64::MAX
        || dst_total == 0
        || dst_total == u64::MAX
        || src_rows != dst_rows
        || !subresource_footprints_compatible(dst_desc.Format.0 as u32, &dst_fp.Footprint, src_desc.Format.0 as u32, &src_fp.Footprint)
    {
        return false;
    }
    let row_pitch = src_fp.Footprint.RowPitch.max(dst_fp.Footprint.RowPitch);
    src_fp.Footprint.RowPitch = row_pitch;
    dst_fp.Footprint.RowPitch = row_pitch;
    let slices = (src_fp.Footprint.Depth as u64).max(1);
    let needed = (row_pitch as u64).saturating_mul(src_rows as u64).saturating_mul(slices).max(src_total).max(dst_total);
    let Some(scratch_raw) = list_x::acquire_scratch_for_copy(list, needed) else {
        return false;
    };
    let scratch_ptr = scratch_raw as *mut c_void;
    let Some(scratch) = ID3D12Resource::from_raw_borrowed(&scratch_ptr) else {
        return false;
    };

    list_x::transition(list, scratch_raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    {
        let src_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(src.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: src_sub },
        });
        let buf_dst_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(scratch.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: src_fp },
        });
        list_x::pc_copy_texture_region(list, &buf_dst_loc.0, 0, 0, 0, &src_loc.0, std::ptr::null());
    }

    list_x::transition(list, scratch_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    {
        let buf_src_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(scratch.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: dst_fp },
        });
        let dst_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(dst.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: dst_sub },
        });
        list_x::pc_copy_texture_region(list, &dst_loc.0, 0, 0, 0, &buf_src_loc.0, std::ptr::null());
    }

    list_x::transition(list, scratch_raw, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    true
}

unsafe fn sync_pair(list: *mut c_void, dev: &ID3D12Device, ty: D3D12_COMMAND_LIST_TYPE, pair: AliasPair, writer_state: u32) {
    let w_ptr = pair.writer as *mut c_void;
    let r_ptr = pair.reader as *mut c_void;
    let (Some(writer_res), Some(reader_res)) = (ID3D12Resource::from_raw_borrowed(&w_ptr), ID3D12Resource::from_raw_borrowed(&r_ptr)) else {
        return;
    };
    let target_srv = if ty == D3D12_COMMAND_LIST_TYPE_DIRECT { STATE_SRV_ALL } else { STATE_SRV_NON_PIXEL };

    // Determine reader's incoming state on this list.
    let (reader_before, reader_after) = match pair.reader_owner {
        Owner::PlacedSrv => {
            let cur = LIST_STATES
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|m| m.get(&(list as usize)))
                .and_then(|s| s.srv_states.get(&pair.reader).copied())
                .unwrap_or(STATE_COMMON);
            (cur, target_srv)
        }
        Owner::TitlePlaced => {
            let Some(st) = TABLE
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|t| t.by_res.get(&pair.reader))
                .and_then(TrackedTex::uniform_state)
            else {
                return;
            };
            // On a compute list, never emit transitions referencing DIRECT-only states
            // (`RENDER_TARGET`, `INDEX_BUFFER`, `VERTEX_AND_CONSTANT_BUFFER`, `DEPTH_*`, `PIXEL_SHADER_RESOURCE`).
            if !state_valid_on_list(ty, st) {
                return;
            }
            (st, st)
        }
        Owner::PlacedUav => return,
    };

    if writer_state != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
        list_x::transition(list, pair.writer, D3D12_RESOURCE_STATES(writer_state as i32), D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    if reader_before != D3D12_RESOURCE_STATE_COPY_DEST.0 as u32 {
        list_x::transition(list, pair.reader, D3D12_RESOURCE_STATES(reader_before as i32), D3D12_RESOURCE_STATE_COPY_DEST);
    }

    let ok = if pair.buffer_through {
        copy_all_subresources_through_buffer(list, dev, reader_res, writer_res)
    } else {
        list_x::pc_copy_resource(list, pair.reader, pair.writer);
        true
    };

    if writer_state != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
        list_x::transition(list, pair.writer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATES(writer_state as i32));
    }
    if reader_after != D3D12_RESOURCE_STATE_COPY_DEST.0 as u32 {
        list_x::transition(list, pair.reader, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATES(reader_after as i32));
    }

    if pair.reader_owner == Owner::PlacedSrv {
        let mut g = LIST_STATES.lock().unwrap();
        let ls = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        ls.list_type = Some(ty);
        ls.srv_states.insert(pair.reader, reader_after);
    }

    if ok {
        static SYNCED: AtomicU64 = AtomicU64::new(0);
        let n = SYNCED.fetch_add(1, Ordering::Relaxed);
        if n < 8 || n.is_power_of_two() {
            crate::xlog_info!(
                "D3D12.X: synced format-aliased texture {:#x} -> {:#x} ({}, {} syncs so far)",
                pair.writer,
                pair.reader,
                if pair.buffer_through { "buffer-through" } else { "direct copy" },
                n + 1
            );
        }
    }
}

/// Sync active placed-UAV -> format-aliased SRV pairs after a UAV barrier on `list`.
/// When `specific_targets` is `Some(targets)`, only sync pairs whose `writer` is in `targets`;
/// when `None` (global UAV barrier), deduplicate against `bind::activity()`.
pub(super) unsafe fn on_uav_barrier(list: *mut c_void, specific_targets: Option<&[usize]>) {
    if ACTIVE_PLACED_UAV_PAIRS.load(Ordering::Relaxed) == 0 || list.is_null() {
        return;
    }
    let Some(ty) = list_x::tracked_list_type(list) else {
        return;
    };
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return;
    }
    // Skip duplicate back-to-back global UAV barriers when no draw or dispatch executed in between.
    if specific_targets.is_none() {
        let act = super::bind::activity();
        let mut g = LIST_STATES.lock().unwrap();
        let ls = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        if ls.last_activity == act && act != (0, 0) {
            return;
        }
        ls.last_activity = act;
        ls.list_type = Some(ty);
    }
    let Some(dev) = list_x::device_of_list(list) else {
        return;
    };
    let dev_raw = dev.as_raw() as usize;
    let pairs: Vec<AliasPair> = {
        let g = TABLE.lock().unwrap();
        match g.as_ref() {
            Some(t) => t
                .pairs
                .iter()
                .copied()
                .filter(|p| {
                    p.writer_owner == Owner::PlacedUav
                        && p.device == dev_raw
                        && specific_targets.is_none_or(|targets| targets.contains(&p.writer))
                })
                .collect(),
            None => return,
        }
    };
    for pair in pairs {
        sync_pair(list, &dev, ty, pair, STATE_UAV);
    }
}

/// Sync format-aliased readers for title-owned writers that just transitioned out of write state.
pub(super) unsafe fn sync_title_writers(list: *mut c_void, writers: &[(usize, u32)]) {
    if writers.is_empty() || ACTIVE_PAIRS.load(Ordering::Relaxed) == 0 || list.is_null() {
        return;
    }
    let Some(ty) = list_x::tracked_list_type(list) else {
        return;
    };
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return;
    }
    let Some(dev) = list_x::device_of_list(list) else {
        return;
    };
    let dev_raw = dev.as_raw() as usize;
    let mut seen = HashSet::new();
    for &(writer, state) in writers {
        if !seen.insert(writer) {
            continue;
        }
        let pairs: Vec<AliasPair> = {
            let g = TABLE.lock().unwrap();
            match g.as_ref() {
                Some(t) => t.pairs.iter().copied().filter(|p| p.writer == writer && p.device == dev_raw).collect(),
                None => continue,
            }
        };
        for pair in pairs {
            sync_pair(list, &dev, ty, pair, state);
        }
    }
}

/// Return any `PlacedSrv` resources transitioned into read state on `list` back to `COMMON`
/// before `Close()`, keeping cross-queue state invariants clean.
pub(super) unsafe fn before_close(list: *mut c_void) {
    if list.is_null() {
        return;
    }
    let pending: Vec<(usize, u32)> = {
        let mut g = LIST_STATES.lock().unwrap();
        match g.as_mut().and_then(|m| m.get_mut(&(list as usize))) {
            Some(ls) if !ls.srv_states.is_empty() => ls.srv_states.drain().collect(),
            _ => return,
        }
    };
    for (res, state) in pending {
        if state != STATE_COMMON {
            list_x::transition(list, res, D3D12_RESOURCE_STATES(state as i32), D3D12_RESOURCE_STATE_COMMON);
        }
    }
}

/// Clear per-list sync state on `Reset` / `Create` / `Release`.
pub(super) fn list_reset(list: usize) {
    if let Some(m) = LIST_STATES.lock().unwrap().as_mut() {
        m.remove(&list);
    }
}

/// Intercept `CopyResource(dst, src)` between incompatible DXGI format families of equal texel
/// width (e.g. `R32_UINT` <-> `R9G9B9E5_SHAREDEXP`) and run it through an intermediate buffer.
/// Returns `true` if handled.
pub(super) unsafe fn try_copy_resource(list: *mut c_void, dst: usize, src: usize) -> bool {
    if !enabled() || dst == 0 || src == 0 || dst == src {
        return false;
    }
    let (dp, sp) = (dst as *mut c_void, src as *mut c_void);
    let (Some(d), Some(s)) = (ID3D12Resource::from_raw_borrowed(&dp), ID3D12Resource::from_raw_borrowed(&sp)) else {
        return false;
    };
    let (dd, sd) = (d.GetDesc(), s.GetDesc());
    if dd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || sd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return false;
    }
    if !needs_buffer_through(dd.Format.0 as u32, sd.Format.0 as u32) {
        return false;
    }
    let Some(dev) = list_x::device_of_list(list) else {
        return false;
    };
    let ok = copy_all_subresources_through_buffer(list, &dev, d, s);
    if ok {
        static N: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        if k < 8 || k.is_power_of_two() {
            crate::xlog_info!(
                "D3D12.X: format-aliased CopyResource fmt {} -> fmt {} ({}x{}) copied via intermediate buffer ({} so far)",
                sd.Format.0,
                dd.Format.0,
                dd.Width,
                dd.Height,
                k + 1
            );
        }
    }
    ok
}

/// Intercept full-subresource texture-to-texture `CopyTextureRegion` between incompatible DXGI
/// format families of equal texel width and run it through an intermediate buffer.
/// Returns `true` if handled.
pub(super) unsafe fn try_copy_texture_region(
    list: *mut c_void,
    dst: &D3D12_TEXTURE_COPY_LOCATION,
    x: u32,
    y: u32,
    z: u32,
    src: &D3D12_TEXTURE_COPY_LOCATION,
    bx: *const D3D12_BOX,
) -> bool {
    if !enabled()
        || dst.Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
        || src.Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
        || x != 0
        || y != 0
        || z != 0
        || !bx.is_null()
    {
        return false;
    }
    let (Some(d), Some(s)) = (dst.pResource.as_ref(), src.pResource.as_ref()) else {
        return false;
    };
    let (dd, sd) = (d.GetDesc(), s.GetDesc());
    if dd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || sd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return false;
    }
    if !needs_buffer_through(dd.Format.0 as u32, sd.Format.0 as u32) {
        return false;
    }
    let Some(dev) = list_x::device_of_list(list) else {
        return false;
    };
    copy_one_subresource_through_buffer(
        list,
        &dev,
        d,
        &dd,
        dst.Anonymous.SubresourceIndex,
        s,
        &sd,
        src.Anonymous.SubresourceIndex,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(fmt: u32, w: u64, h: u32, d: u16, mips: u16) -> TexShape {
        TexShape {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D.0,
            width: w,
            height: h,
            depth_or_array: d,
            mip_levels: mips,
            format: fmt,
            samples: 1,
        }
    }

    #[test]
    fn r32_uint_and_r9g9b9e5_alias_needs_buffer_through() {
        assert!(needs_buffer_through(67, 42)); // R9G9B9E5_SHAREDEXP vs R32_UINT
        assert!(needs_buffer_through(67, 39)); // R9G9B9E5_SHAREDEXP vs R32_TYPELESS
        assert!(!needs_buffer_through(42, 39)); // R32_UINT vs R32_TYPELESS (same family)
        assert!(!needs_buffer_through(28, 27)); // R8G8B8A8_UNORM vs R8G8B8A8_TYPELESS

        let s_uav = shape(39, 128, 128, 6, 4);
        let s_srv = shape(67, 128, 128, 6, 4);
        assert!(can_copy_through_buffer(&s_srv, &s_uav));

        // 3D volumetric fog froxel texture: R32_UINT UAV -> R9G9B9E5_SHAREDEXP SRV
        let fog_uav = TexShape { dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D.0, ..shape(42, 160, 90, 64, 1) };
        let fog_srv = TexShape { dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D.0, ..shape(67, 160, 90, 64, 1) };
        assert!(can_copy_through_buffer(&fog_srv, &fog_uav));
        assert_eq!(fog_srv.subresource_count(), 1);

        // BC7 (16 B per 4x4 block) <-> R32G32B32A32_UINT (16 B per texel, 1/4 width & height)
        let bc7 = shape(98, 256, 128, 1, 1);
        let u32x4 = shape(3, 64, 32, 1, 1);
        assert!(can_copy_through_buffer(&bc7, &u32x4));
        assert!(can_copy_through_buffer(&u32x4, &bc7));

        // Mismatched dimensions or bpp rejected
        assert!(!can_copy_through_buffer(&shape(67, 64, 64, 1, 1), &shape(42, 128, 128, 1, 1)));
        assert!(!can_copy_through_buffer(&shape(10, 128, 128, 1, 1), &shape(42, 128, 128, 1, 1)));
    }

    #[test]
    fn placed_uav_and_srv_descs_are_sanitized() {
        let mut x = [0u8; 0x38];
        x[0..4].copy_from_slice(&3u32.to_le_bytes()); // Texture2D
        x[0x20..0x24].copy_from_slice(&67u32.to_le_bytes()); // R9G9B9E5_SHAREDEXP
        x[0x30..0x34].copy_from_slice(&0x30_0005u32.to_le_bytes()); // RTV | UAV + console bits

        // Placed UAV with R32_UINT view over R9G9B9E5 desc -> R32_TYPELESS + ALLOW_UNORDERED_ACCESS (no RTV)
        let uav = prepare_placed_uav_desc(&x, 42);
        let uav_fmt = u32::from_le_bytes(uav[0x20..0x24].try_into().unwrap());
        let uav_flags = u32::from_le_bytes(uav[0x30..0x34].try_into().unwrap());
        assert_eq!(uav_fmt, 39, "R32_UINT UAV over R9G9B9E5 must become R32_TYPELESS, not R11G11B10_FLOAT");
        assert!(uav_flags & FLAG_UAV != 0 && uav_flags & FLAG_RENDER_TARGET == 0);

        // Placed SRV with R9G9B9E5 view -> strips RTV|UAV so pc_resource_desc keeps format 67!
        let srv = prepare_placed_srv_desc(&x, 67);
        let srv_pc = super::super::device_x::pc_resource_desc(&srv);
        let srv_fmt = u32::from_le_bytes(srv_pc[0x20..0x24].try_into().unwrap());
        let srv_flags = u32::from_le_bytes(srv_pc[0x30..0x34].try_into().unwrap());
        assert_eq!(srv_fmt, 67, "placed SRV must stay R9G9B9E5_SHAREDEXP");
        assert_eq!(srv_flags & (FLAG_RENDER_TARGET | FLAG_UAV), 0);
    }

    #[test]
    fn alias_table_links_in_both_creation_orders() {
        let mut t = AliasTable::new();
        let va = 0x2_0000_0000usize;
        let uav_shape = shape(39, 64, 64, 1, 1);
        let srv_shape = shape(67, 64, 64, 1, 1);

        // Order 1: SRV created first, placed UAV created second -> links and marks SRV gpu_written
        let m1 = t.insert(0x100, TrackedTex::new(1, va, srv_shape, Owner::TitlePlaced, 0, false));
        assert!(m1.is_empty());
        assert!(t.pairs.is_empty());

        let m2 = t.insert(0x200, TrackedTex::new(1, va, uav_shape, Owner::PlacedUav, STATE_UAV, true));
        assert_eq!(m2, vec![0x100]);
        assert_eq!(t.pairs.len(), 1);
        assert!(t.pairs[0].buffer_through);
        assert_eq!(t.placed_uav_pairs, 1);

        // Order 2: second SRV created after UAV -> links immediately and marks itself gpu_written
        let m3 = t.insert(0x300, TrackedTex::new(1, va, srv_shape, Owner::PlacedSrv, 0, false));
        assert_eq!(m3, vec![0x300]);
        assert_eq!(t.pairs.len(), 2);

        // Removing the reader cleans up its pair
        t.remove(0x100);
        assert_eq!(t.pairs.len(), 1);
        assert_eq!(t.pairs[0].reader, 0x300);

        // Removing the writer cleans up remaining pairs
        t.remove(0x200);
        assert!(t.pairs.is_empty());
        assert_eq!(t.placed_uav_pairs, 0);

        // Compute-list state validation rejects RENDER_TARGET (0x4), INDEX_BUFFER (0x2), VCB (0x1),
        // DEPTH_WRITE (0x10), PIXEL_SHADER_RESOURCE (0x80), while allowing COMMON/UAV/NON_PIXEL_SRV/COPY.
        assert!(state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, STATE_COMMON));
        assert!(state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, STATE_UAV));
        assert!(state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, STATE_SRV_NON_PIXEL));
        assert!(!state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, STATE_RENDER_TARGET));
        assert!(!state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, 0x1));
        assert!(!state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, 0x2));
        assert!(!state_valid_on_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, STATE_SRV_ALL));
        assert!(state_valid_on_list(D3D12_COMMAND_LIST_TYPE_DIRECT, STATE_RENDER_TARGET));

        // Subresource footprint compatibility rejects mismatched element byte widths even with equal RowPitch
        let fp = |w, h, d| D3D12_SUBRESOURCE_FOOTPRINT {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(0),
            Width: w,
            Height: h,
            Depth: d,
            RowPitch: 256,
        };
        assert!(subresource_footprints_compatible(67, &fp(32, 32, 1), 42, &fp(32, 32, 1)));
        assert!(!subresource_footprints_compatible(10, &fp(32, 32, 1), 42, &fp(32, 32, 1)), "8 Bpp vs 4 Bpp must be rejected");
        assert!(subresource_footprints_compatible(98, &fp(64, 64, 1), 3, &fp(16, 16, 1)), "BC7 64x64 == R32G32B32A32 16x16");
    }
}
