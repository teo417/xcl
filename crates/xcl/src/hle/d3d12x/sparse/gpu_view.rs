//! Buffers in a GPU view of the title's memory, readable by shaders (`XCL_GPU_VIEW_BUFFERS=1`).
//!
//! A title reserves 64 KiB pages twice, once for the CPU and once with no CPU access (`kernelx::
//! mirrors`), and gives the GPU addresses of the second. Those lie outside the memory windows, so
//! a shader reads zeros at them: a retail title's culling reads its object lists, and its mesh
//! shaders their geometry, from there.
//!
//! The tiles a buffer view covers are mapped in free address space of a sparse window (video
//! memory only: nothing is committed for the CPU) and hold what the title's pages hold: their
//! bytes go up when the view is made and again when the title writes them. The view's address is
//! replaced by the window's. Only what is read is mirrored: a write through it is lost, so UAVs
//! keep their address.

use std::collections::HashMap;

use super::*;

/// The window space a chunk of the title's view takes. Views inside one chunk share it; a view
/// across chunks gets a space of its own (its bytes have to be contiguous).
const EXTENT: usize = 1 << 20;
/// Views longer than this keep their address.
const MOST: usize = 64 << 20;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_VIEW_BUFFERS").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_GPU_VIEW_MB` (512): the video memory the mirrors may take.
fn budget() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_GPU_VIEW_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(512usize) << 20)
}

#[derive(Default)]
pub(super) struct Mirrors {
    /// (first byte of the title's chunks, chunks) → the window space of them
    extents: HashMap<(usize, usize), usize>,
    /// window tile → title tile (an address in the GPU view)
    title_of: FxHashMap<usize, usize>,
    /// where the bytes of a title tile are read from (its own address, or the CPU view's)
    read: FxHashMap<usize, usize>,
    /// read tile → window tiles whose bytes are there
    by_read: FxHashMap<usize, Vec<usize>>,
    /// window tiles none of whose bytes have gone up yet (or whose pages changed place)
    fresh: Vec<usize>,
    /// the next title tile the slow check looks at
    turn: usize,
    /// GPU-view mappings seen (`kernelx::gpu_view_map_gen`)
    gen: u64,
}

static MIRRORS: Mutex<Option<Mirrors>> = Mutex::new(None);

impl Mirrors {
    /// Title tile `title` is shown at window tile `win`, its bytes read from `read`.
    pub(super) fn show(&mut self, win: usize, title: usize, read: usize) {
        self.title_of.insert(win, title);
        self.read.insert(title, read);
        self.by_read.entry(read).or_default().push(win);
        self.fresh.push(win);
    }

    fn set_read(&mut self, title: usize, read: usize) {
        let old = self.read.insert(title, read);
        if old == Some(read) {
            return;
        }
        let wins: Vec<usize> = self.title_of.iter().filter(|&(_, &t)| t == title).map(|(&w, _)| w).collect();
        if let Some(o) = old.and_then(|o| self.by_read.get_mut(&o)) {
            o.retain(|w| !wins.contains(w));
        }
        self.by_read.entry(read).or_default().extend(&wins);
        self.fresh.extend(wins);
    }

    /// The window tiles and pages a set of written pages (`dirty`, title or CPU addresses) is
    /// shown at: (window page, the read address of its bytes).
    pub(super) fn shown_at(&self, dirty: &[usize]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &d in dirty {
            let tile = d & !(TILE - 1);
            for &w in self.by_read.get(&tile).map(Vec::as_slice).unwrap_or(&[]) {
                out.push((w + (d - tile), d));
            }
        }
        out
    }

    pub(super) fn tiles(&self) -> usize {
        self.title_of.len()
    }
}

/// Copies `n` bytes from the title's memory at `at`; false when the page went away (nothing read).
unsafe fn read_title(at: usize, out: *mut u8, n: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    let mut got = 0usize;
    ReadProcessMemory(windows_sys::Win32::System::Threading::GetCurrentProcess(), at as *const c_void, out as *mut c_void, n, &mut got) != 0 && got == n
}

impl State {
    /// Brings the mirrored tiles in step with the title's memory: whole tiles that are new, and
    /// the pages the title wrote (`dirty`). Before the title's lists; `m` is held by the caller.
    pub(super) unsafe fn mirror_up(&mut self, m: &mut Mirrors, dirty: &[usize]) -> usize {
        let mut pages: Vec<(usize, usize)> = m.shown_at(dirty);
        for win in std::mem::take(&mut m.fresh) {
            let Some(&title) = m.title_of.get(&win) else { continue };
            let read = m.read[&title];
            pages.extend((0..PAGES_PER_TILE).map(|i| (win + i * PAGE, read + i * PAGE)));
        }
        pages.sort_unstable();
        pages.dedup();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut from: FxHashMap<usize, usize> = FxHashMap::default();
        let mut bytes = [0u8; PAGE];
        for (win, read) in pages {
            let tile = win & !(TILE - 1);
            let Some(entry) = self.mapped.get_mut(&tile) else { continue };
            if !read_title(read, bytes.as_mut_ptr(), PAGE) {
                continue;
            }
            let i = (win - tile) / PAGE;
            entry.holes &= !(1u16 << i);
            scan_page(&bytes, win, &mut entry.sent[i], &mut runs);
            from.insert(win, read);
        }
        let n = runs.iter().map(|r| r.1).sum();
        // (the title tile a run lies in never changes within it: a run is at most a page here, but
        // `push_run` joins neighbours, and two neighbouring pages may come from two places)
        let split: Vec<(usize, usize)> = runs.iter().flat_map(|&(a, l)| {
            let mut out = Vec::new();
            let (mut a, end) = (a, a + l);
            while a < end {
                let e = ((a & !(PAGE - 1)) + PAGE).min(end);
                out.push((a, e - a));
                a = e;
            }
            out
        }).collect();
        self.upload_runs(&split, &|dst, p, len| {
            let page = dst & !(PAGE - 1);
            from.get(&page).is_some_and(|r| read_title(r + (dst - page), p, len))
        });
        n
    }
}

/// The address a view of `len` bytes at `va` gets when it lies in a GPU view of the title's and
/// in no memory window: where the same bytes can be read by a shader. None: it keeps `va`.
pub fn base_for(va: usize, len: usize) -> Option<usize> {
    if !on() || len == 0 || len > MOST {
        return None;
    }
    let u = crate::gpu::uma::get()?;
    if u.contains(va) || !u.has_sparse() {
        return None;
    }
    let last = va.checked_add(len)?.checked_sub(1)?;
    crate::hle::kernelx::mirrors::source(va)?;
    crate::hle::kernelx::mirrors::source(last)?;
    let (t0, t1) = (va & !(TILE - 1), (last & !(TILE - 1)) + TILE);
    let first = t0 & !(EXTENT - 1);
    let n = (t1 - first).div_ceil(EXTENT);
    let known = MIRRORS.lock().unwrap().get_or_insert_with(Mirrors::default).extents.get(&(first, n)).copied();
    let win = match known {
        Some(w) => w,
        None => {
            let (w, _) = u.sparse_alloc(n * EXTENT, EXTENT)?;
            *MIRRORS.lock().unwrap().get_or_insert_with(Mirrors::default).extents.entry((first, n)).or_insert(w)
        }
    };
    // (held: STATE, then MIRRORS, as in `sync`)
    let mut g = lock_state();
    let s = g.as_mut()?;
    let mut mg = MIRRORS.lock().unwrap();
    let m = mg.get_or_insert_with(Mirrors::default);
    let new: Vec<usize> = (t0..t1).step_by(TILE).filter(|t| !m.title_of.contains_key(&(win + (t - first)))).collect();
    if !new.is_empty() {
        if m.tiles() + new.len() > budget() / TILE {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: the buffers of the GPU view take more than XCL_GPU_VIEW_MB: the rest keeps its address"));
            return None;
        }
        // whole runs of new tiles at once (a mapping call and a signal for each run)
        let mut i = 0;
        while i < new.len() {
            let mut j = i + 1;
            while j < new.len() && new[j] == new[j - 1] + TILE {
                j += 1;
            }
            let (a, b) = (new[i], win + (new[i] - first));
            // SAFETY: the tiles are address space of a window, with no CPU memory behind them
            unsafe { s.ensure_mapped(b, (j - i) * TILE, |w| query_tile_holes(super::super::title_textures::mirror_source(a + (w - b)))) };
            for &t in &new[i..j] {
                let w = win + (t - first);
                if s.mapped.contains_key(&w) {
                    m.show(w, t, super::super::title_textures::mirror_source(t) & !(TILE - 1));
                }
            }
            i = j;
        }
        static N: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed) + 1;
        if k.is_power_of_two() {
            crate::xlog_info!("D3D12.X: buffers of the GPU view: {} tiles mirrored ({} MB), latest view {va:#x} ({len} B)", m.tiles(), (m.tiles() * TILE) >> 20);
        }
    }
    if (t0..t1).step_by(TILE).any(|t| !m.title_of.contains_key(&(win + (t - first)))) {
        return None;
    }
    Some(win + (va - first))
}

/// The pages of the CPU (and pool) side whose writes have to be seen: for the write watch.
pub fn sources() -> Vec<(usize, usize)> {
    MIRRORS.lock().unwrap().as_ref().map_or(Vec::new(), |m| m.by_read.keys().map(|&r| (r, r + TILE)).collect())
}

/// `State::mirror_up` for a submit (`before_execute`; `s` is the locked state). A page the title
/// mapped elsewhere or now has no mapping at all is looked at again when the mappings of a GPU
/// view change, and a few tiles in turn otherwise.
pub(super) unsafe fn sync(s: &mut State, dirty: &[usize]) {
    let mut g = MIRRORS.lock().unwrap();
    let Some(m) = g.as_mut().filter(|m| m.tiles() != 0) else { return };
    let gen = crate::hle::kernelx::gpu_view_map_gen();
    let titles: Vec<usize> = m.read.keys().copied().collect();
    let look: Vec<usize> = if std::mem::replace(&mut m.gen, gen) != gen {
        titles
    } else {
        let n = titles.len().max(1);
        (0..8.min(titles.len())).map(|i| titles[(m.turn + i) % n]).collect()
    };
    m.turn = m.turn.wrapping_add(8);
    for t in look {
        let r = super::super::title_textures::mirror_source(t) & !(TILE - 1);
        m.set_read(t, r);
    }
    let n = s.mirror_up(m, dirty);
    static SENT: AtomicU64 = AtomicU64::new(0);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let total = SENT.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
    let k = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if k.is_power_of_two() && k >= 64 {
        crate::xlog_info!("D3D12.X: buffers of the GPU view: {} tiles, {} MB sent in {k} submits", m.tiles(), total >> 20);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_page_is_shown_at_every_window_tile_it_is_read_for() {
        let mut m = Mirrors::default();
        m.show(0x10_0000, 0xAA_0000, 0xAA_0000);
        m.show(0x20_0000, 0xAA_0000, 0xAA_0000);
        m.show(0x30_0000, 0xBB_0000, 0xCC_0000);
        assert_eq!(m.tiles(), 3);
        let mut a = m.shown_at(&[0xAA_3000]);
        a.sort();
        assert_eq!(a, vec![(0x10_3000, 0xAA_3000), (0x20_3000, 0xAA_3000)]);
        assert_eq!(m.shown_at(&[0xCC_1000]), vec![(0x30_1000, 0xCC_1000)], "by the read address, not the title's");
        assert!(m.shown_at(&[0xBB_1000]).is_empty());
        m.fresh.clear();
        // the page behind a title tile moved: its window tiles are sent whole again
        m.set_read(0xBB_0000, 0xDD_0000);
        assert_eq!(m.fresh, vec![0x30_0000]);
        assert!(m.shown_at(&[0xCC_1000]).is_empty());
        assert_eq!(m.shown_at(&[0xDD_2000]), vec![(0x30_2000, 0xDD_2000)]);
        m.set_read(0xBB_0000, 0xDD_0000);
        assert_eq!(m.fresh, vec![0x30_0000], "the same place again is nothing new");
    }

    /// The bytes of the title's pages reach the window tiles mapped for them, and again when the
    /// title writes a page; a page that went away is left as it is.
    #[test]
    fn gpu_the_titles_bytes_reach_the_window_tiles_shown_for_them() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let window = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let source = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert!(window % TILE == 0 && source % TILE == 0);
            for i in 0..TILE / 4 {
                *((source + 4 * i) as *mut u32) = i as u32 * 7 + 1;
            }
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("list");
            rb_list.Close().expect("close");
            let mut slots = Vec::new();
            for _ in 0..UPLOAD_SLOTS {
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
                list.Close().expect("close");
                slots.push(UploadSlot { alloc, list, fence_val: 0 });
            }
            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                windows: vec![(window, reserved.expect("reserved buffer"))],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots,
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
            };
            let mut m = Mirrors::default();
            s.ensure_mapped(window, TILE, |_| query_tile_holes(source));
            m.show(window, 0x7FFF_0000, source);
            let at = |i: usize| *((window + 4 * i) as *const u32);
            let shown = |s: &mut State, m: &mut Mirrors, dirty: &[usize]| {
                s.mirror_up(m, dirty);
                s.readback_ranges(&[(window, TILE)]);
            };
            shown(&mut s, &mut m, &[]);
            assert!((0..TILE / 4).all(|i| at(i) == i as u32 * 7 + 1), "the whole tile at first");
            // a page the title wrote (and one it did not)
            *((source + 0x3010) as *mut u32) = 0xDEAD_BEEF;
            shown(&mut s, &mut m, &[source + 0x3000]);
            assert_eq!((at(0x3010 / 4), at(0x3014 / 4)), (0xDEAD_BEEF, (0x3014 / 4) as u32 * 7 + 1));
            // not reported as written: not sent
            *((source + 0x4010) as *mut u32) = 0x1234_5678;
            shown(&mut s, &mut m, &[]);
            assert_eq!(at(0x4010 / 4), (0x4010 / 4) as u32 * 7 + 1);
            // the page is gone: what the GPU has stays, and nothing faults
            assert_ne!(VirtualFree(source as *mut c_void, 0, MEM_RELEASE), 0);
            m.fresh.push(window);
            shown(&mut s, &mut m, &[source + 0x5000]);
            assert_eq!(at(0x3010 / 4), 0xDEAD_BEEF);
            s.wait_for(s.value);
            drop(s);
            VirtualFree(window as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
