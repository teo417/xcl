//! `XCL_WATCH_TRAP=1`: the looks before each submit skip watched memory nobody writes.
//!
//! A 2 MiB block of a watched range that no look has found written for `COLD_LOOKS` looks is
//! made read-only (its committed read-write parts) and left out of the looks. The first write
//! there faults into `handler`, which gives the block its protection back and returns it to the
//! looks; the write itself happens after that, so write-watch sees it as before. When a block
//! is made read-only, a last look at it reports what was written since the previous one.
//!
//! What the handler cannot catch: a kernel write into a read-only page fails (ERROR_NOACCESS)
//! instead of faulting. The title's ReadFile buffers are therefore given their protection back
//! and pinned (never made read-only) until the read is over: when ReadFile returns, or for a
//! pending read when a look sees its OVERLAPPED completed (or the title uses that OVERLAPPED
//! for its next read, which only a finished one allows). Either way the buffer's pages are
//! then reported as written: an unbuffered read flags its pages when it starts, not when the
//! data lands, so a look in between would reset them. Commits, decommits and protection changes
//! through kernelx and lazy pages (`guard`) and sparse commits (`changed`) give the blocks they
//! meet their protection back first and keep them from going read-only meanwhile. XCL's own
//! checks whether memory is writable (DirectStorage destinations, CPU-side writes for the GPU,
//! emulated stores) give it back too: their writes come from user mode and would be let
//! through, but the check reads the protection.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_RESERVE};

const BLOCK: usize = 2 << 20;
/// looks a block must be found clean before it is made read-only: protecting 2 MiB costs about
/// 36 us, looking at it ~1 us, so it pays once a block stays cold ~40 looks (one to two frames)
const COLD_LOOKS: u32 = 64;
/// blocks made read-only per look at most (~36 us each), so a cold world goes over in steps
const PROTECTS_PER_LOOK: usize = 32;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_READONLY: u32 = 0x02;
/// protection modifiers kept as they are (PAGE_NOCACHE, PAGE_WRITECOMBINE)
const MODIFIERS: u32 = 0x200 | 0x400;
const STATUS_PENDING: usize = 0x103;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// A read-only span: its end and the committed parts with the protection they had.
struct Cold {
    end: usize,
    parts: Vec<(usize, usize, u32)>,
}

/// read-only spans by start (one block or the part of it in one watched range)
static COLD: Mutex<BTreeMap<usize, Cold>> = Mutex::new(BTreeMap::new());
/// blocks with a kernel read into them in flight -> how many
static PINS: Mutex<BTreeMap<usize, u32>> = Mutex::new(BTreeMap::new());
/// pending title reads: (OVERLAPPED, buffer, length, byte count output)
static PENDING: Mutex<Vec<(usize, usize, usize, usize)>> = Mutex::new(Vec::new());
/// finished reads whose pages the next look reports as written
static REPORT: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
/// an OVERLAPPED's size (Internal, InternalHigh, Offset/Pointer, hEvent)
const OVERLAPPED_LEN: usize = 32;
/// the looks' own state (one look at a time: `title_textures::REFRESH`)
static LOOKS: Mutex<Looks> = Mutex::new(Looks { clean: BTreeMap::new(), n: 0 });
/// blocks ever made read-only (for `was_ours`)
static EVER: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// faults taken, blocks made read-only, spans given back by `changed`, bytes left out of looks,
/// committed regions a thaw could not give back
static STATS: [AtomicU64; 6] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

struct Looks {
    /// block -> (span looked at in it this look, looks found clean in a row)
    clean: BTreeMap<usize, ((usize, usize), u32)>,
    n: u64,
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_WATCH_TRAP").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("watch trap: on (blocks of {} MiB clean for {COLD_LOOKS} looks are made read-only)", BLOCK >> 20);
        }
        on
    })
}

fn install() {
    if !INSTALLED.swap(true, Ordering::AcqRel) {
        unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
    }
}

unsafe fn protect(a: usize, n: usize, p: u32) -> bool {
    let mut old = 0u32;
    VirtualProtect(a as *const c_void, n, p, &mut old) != 0
}

/// Give a read-only span its protection back. A part with pages decommitted since then cannot
/// be protected as a whole (ERROR_INVALID_ADDRESS): its committed regions are, one by one.
unsafe fn thaw(c: &Cold) {
    for &(a, n, p) in &c.parts {
        if protect(a, n, p) {
            continue;
        }
        let end = a + n;
        let mut at = a;
        while at < end {
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
                break;
            }
            let next = (m.BaseAddress as usize + m.RegionSize).min(end);
            if m.State == MEM_COMMIT && m.Protect & !MODIFIERS == PAGE_READONLY && !protect(at, next - at, p) {
                STATS[4].fetch_add(1, Ordering::Relaxed);
            }
            at = next;
        }
    }
}

/// The read-only spans meeting `[a, a + n)` given their protection back and dropped.
unsafe fn thaw_range(cold: &mut BTreeMap<usize, Cold>, a: usize, n: usize) -> usize {
    let end = a.saturating_add(n);
    let from = cold.range(..=a).next_back().filter(|(_, c)| c.end > a).map_or(a, |(&s, _)| s);
    let hit: Vec<usize> = cold.range(from..end).map(|(&s, _)| s).collect();
    for s in &hit {
        if let Some(c) = cold.remove(s) {
            thaw(&c);
        }
    }
    hit.len()
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 || r.ExceptionInformation[0] != 1 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let at = r.ExceptionInformation[1];
    let mut cold = COLD.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = cold.range(..=at).next_back().filter(|(_, c)| at < c.end).map(|(&s, _)| s) else {
        // another thread gave the span back after this write faulted: go on if it may write now
        drop(cold);
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        let now = VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0
            && m.State == MEM_COMMIT
            && matches!(m.Protect & !MODIFIERS, PAGE_READWRITE | 0x40);
        return if now && was_ours(at) { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH };
    };
    let c = cold.remove(&s).unwrap();
    thaw(&c);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    EXCEPTION_CONTINUE_EXECUTION
}

/// Whether `at` lies in a block this trap has made read-only at some point (a write fault there
/// that finds the page writable again lost a race with another thread's `thaw`).
fn was_ours(at: usize) -> bool {
    EVER.lock().unwrap_or_else(|e| e.into_inner()).contains(&(at & !(BLOCK - 1)))
}

/// `[a, a + n)` is about to be committed, decommitted, freed or given another protection (or a
/// kernel write goes there): the read-only spans it meets get their protection back first.
pub fn changed(a: usize, n: usize) {
    if on() {
        changed_now(a, n);
    }
}

/// `guard` for a VirtualFree of `[a, a + n)`; `n == 0` (a release) means the whole allocation.
pub fn guard_free(a: usize, n: usize) -> Guard {
    if !on() || a == 0 {
        return Guard(0, 0);
    }
    if n != 0 {
        return guard(a, n);
    }
    let mut at = a;
    loop {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.AllocationBase as usize != a {
            break;
        }
        at = m.BaseAddress as usize + m.RegionSize;
    }
    guard(a, at.saturating_sub(a))
}

fn changed_now(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut cold = COLD.lock().unwrap_or_else(|e| e.into_inner());
    if cold.is_empty() {
        return;
    }
    let k = unsafe { thaw_range(&mut cold, a, n) };
    STATS[2].fetch_add(k as u64, Ordering::Relaxed);
}

fn blocks(a: usize, n: usize) -> impl Iterator<Item = usize> {
    let first = a & !(BLOCK - 1);
    let last = (a + n.max(1) - 1) & !(BLOCK - 1);
    (first..=last).step_by(BLOCK)
}

fn pin(a: usize, n: usize, by: i32) {
    let mut pins = PINS.lock().unwrap_or_else(|e| e.into_inner());
    for b in blocks(a, n) {
        let e = pins.entry(b).or_insert(0);
        *e = e.saturating_add_signed(by);
        if *e == 0 {
            pins.remove(&b);
        }
    }
}

/// Pinned and writable while it lives: memory a call is about to write, commit, decommit or
/// give another protection.
pub struct Guard(usize, usize);

impl Drop for Guard {
    fn drop(&mut self) {
        release(self.0, self.1);
    }
}

fn hold(a: usize, n: usize) {
    if a != 0 && n != 0 {
        pin(a, n, 1);
        changed_now(a, n);
    }
}

fn release(a: usize, n: usize) {
    if a != 0 && n != 0 {
        pin(a, n, -1);
    }
}

/// `[a, a + n)` is about to be committed, decommitted or given another protection by the call
/// made while the guard lives.
pub fn guard(a: usize, n: usize) -> Guard {
    if !on() || a == 0 || n == 0 {
        return Guard(0, 0);
    }
    hold(a, n);
    Guard(a, n)
}

/// Before the title's ReadFile into `[buf, buf + n)`: the buffer, its byte count output `read`
/// and the OVERLAPPED `ov` (both written by the kernel too) pinned and writable.
pub fn read_start(buf: usize, n: usize, read: usize, ov: usize) {
    if !on() || n == 0 {
        return;
    }
    retire_reused(ov);
    hold(buf, n);
    hold(read, 4);
    hold(ov, OVERLAPPED_LEN);
}

/// A pending read whose OVERLAPPED the title is about to use for another call: that read is
/// over (an OVERLAPPED may be used again only then), and the new call replaces the status the
/// next look would have read, which kept the earlier buffer pinned until the new read ended.
/// It is unpinned and its buffer reported as written here. One whose status still says pending
/// (a title that breaks the rule) is left to the looks.
fn retire_reused(ov: usize) {
    if ov == 0 {
        return;
    }
    let over: Vec<(usize, usize, usize, usize)> = {
        let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if !p.iter().any(|&(o, ..)| o == ov) || unsafe { overlapped_status(ov) } == Some(STATUS_PENDING) {
            return;
        }
        let (over, still) = p.drain(..).partition(|&(o, ..)| o == ov);
        *p = still;
        over
    };
    for (o, buf, n, read) in over {
        REPORT.lock().unwrap_or_else(|e| e.into_inner()).push((buf, n));
        release(buf, n);
        release(read, 4);
        release(o, OVERLAPPED_LEN);
    }
}

/// After it: a pending read stays pinned until a look sees `ov` completed; a finished one is
/// unpinned and, if it succeeded, its buffer reported as written.
pub fn read_end(buf: usize, n: usize, read: usize, ov: usize, pending: bool, ok: bool) {
    if !on() || n == 0 {
        return;
    }
    if pending && ov != 0 {
        PENDING.lock().unwrap_or_else(|e| e.into_inner()).push((ov, buf, n, read));
        return;
    }
    if ok {
        REPORT.lock().unwrap_or_else(|e| e.into_inner()).push((buf, n));
    }
    release(buf, n);
    release(read, 4);
    release(ov, OVERLAPPED_LEN);
}

/// `OVERLAPPED::Internal` without faulting if the title has freed it.
unsafe fn overlapped_status(ov: usize) -> Option<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let (mut v, mut done) = (0usize, 0usize);
    (ReadProcessMemory(GetCurrentProcess(), ov as *const c_void, &mut v as *mut usize as *mut c_void, 8, &mut done) != 0 && done == 8).then_some(v)
}

/// The committed parts of `[a, a + n)` if all of them are plain read-write (modifiers kept),
/// with their protection. None parts (a hole: all of it reserved, nothing committed) is a cold
/// span too: nothing to protect, out of the looks until a commit in it (every commit path holds
/// a guard, which gives it back; one that did not would only delay the look, since the write
/// watch keeps its pages until they are read).
unsafe fn writable_parts(a: usize, n: usize) -> Option<Vec<(usize, usize, u32)>> {
    let end = a + n;
    let mut at = a;
    let mut parts = Vec::new();
    while at < end {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            return None;
        }
        let next = (m.BaseAddress as usize + m.RegionSize).min(end);
        if m.State == MEM_COMMIT {
            if m.Protect & !MODIFIERS != PAGE_READWRITE {
                return None;
            }
            parts.push((at, next - at, m.Protect));
        } else if m.State != MEM_RESERVE {
            return None;
        }
        at = next;
    }
    Some(parts)
}

/// Make `[a, a + n)` read-only unless a block of it is pinned; false if it stays as it is.
unsafe fn freeze(a: usize, n: usize) -> bool {
    let mut cold = COLD.lock().unwrap_or_else(|e| e.into_inner());
    {
        let pins = PINS.lock().unwrap_or_else(|e| e.into_inner());
        if blocks(a, n).any(|b| pins.contains_key(&b)) {
            return false;
        }
    }
    let Some(parts) = writable_parts(a, n) else { return false };
    for (i, &(p, len, prot)) in parts.iter().enumerate() {
        if !protect(p, len, PAGE_READONLY | (prot & MODIFIERS)) {
            for &(p, len, prot) in &parts[..i] {
                protect(p, len, prot);
            }
            return false;
        }
    }
    cold.insert(a, Cold { end: a + n, parts });
    EVER.lock().unwrap_or_else(|e| e.into_inner()).extend(blocks(a, n));
    true
}

/// `ranges` minus the read-only spans, as pieces within one block each.
fn hot_pieces(ranges: &[(usize, usize)], cold: &BTreeMap<usize, usize>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &(base, size) in ranges {
        let end = base + size;
        let mut at = base;
        while at < end {
            let block_end = ((at & !(BLOCK - 1)) + BLOCK).min(end);
            // the read-only span covering `at`, else the next one starting before `block_end`
            if let Some((_, &e)) = cold.range(..=at).next_back().filter(|(_, &e)| e > at) {
                at = e.min(end);
                continue;
            }
            let stop = cold.range(at..block_end).next().map_or(block_end, |(&s, _)| s);
            out.push((at, stop - at));
            at = stop;
        }
    }
    out
}

/// One look: `take(base, len, dirty)` is called on the watched pieces that are not read-only
/// (neighbours in one call), completed pending reads report their pages, clean blocks are made
/// read-only.
pub unsafe fn look(ranges: &[(usize, usize)], dirty: &mut Vec<usize>, page: usize, mut take: impl FnMut(usize, usize, &mut Vec<usize>)) {
    install();
    // pending reads that are over: unpinned, their pages written
    let done: Vec<(usize, usize, usize, usize)> = {
        let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        let (over, still): (Vec<_>, Vec<_>) = p.drain(..).partition(|&(ov, ..)| overlapped_status(ov) != Some(STATUS_PENDING));
        *p = still;
        over
    };
    let mut report = std::mem::take(&mut *REPORT.lock().unwrap_or_else(|e| e.into_inner()));
    for &(ov, buf, n, read) in &done {
        release(buf, n);
        release(read, 4);
        release(ov, OVERLAPPED_LEN);
        report.push((buf, n));
    }
    for (buf, n) in report {
        dirty.extend((buf & !(page - 1)..buf + n).step_by(page));
    }
    let cold: BTreeMap<usize, usize> = COLD.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(&s, c)| (s, c.end)).collect();
    let pieces = hot_pieces(ranges, &cold);
    let watched: usize = ranges.iter().map(|r| r.1).sum();
    let hot: usize = pieces.iter().map(|p| p.1).sum();
    STATS[3].fetch_add((watched - hot) as u64, Ordering::Relaxed);
    // pieces that touch go in one call (a range lies in one allocation; pieces of two ranges
    // never merge)
    let mut written = std::collections::BTreeSet::new();
    let mut i = 0;
    let range_of = |a: usize| ranges.partition_point(|&(b, _)| b <= a).saturating_sub(1);
    while i < pieces.len() {
        let (a, mut n) = pieces[i];
        let r = range_of(a);
        let mut j = i + 1;
        while j < pieces.len() && pieces[j].0 == a + n && range_of(pieces[j].0) == r {
            n += pieces[j].1;
            j += 1;
        }
        let from = dirty.len();
        take(a, n, dirty);
        written.extend(dirty[from..].iter().map(|&p| p & !(BLOCK - 1)));
        i = j;
    }
    let mut looks = LOOKS.lock().unwrap_or_else(|e| e.into_inner());
    looks.n += 1;
    // the clean count of each block looked at (a block with two pieces is never made read-only)
    let mut clean: BTreeMap<usize, ((usize, usize), u32)> = BTreeMap::new();
    let mut shared = std::collections::BTreeSet::new();
    for &(a, n) in &pieces {
        let b = a & !(BLOCK - 1);
        if clean.contains_key(&b) {
            shared.insert(b);
            continue;
        }
        let was = looks.clean.get(&b).filter(|(span, _)| *span == (a, n)).map_or(0, |&(_, k)| k);
        clean.insert(b, ((a, n), if written.contains(&b) { 0 } else { was + 1 }));
    }
    let mut made = 0;
    for (&b, ((a, n), k)) in clean.iter_mut() {
        if made == PROTECTS_PER_LOOK {
            break;
        }
        if *k < COLD_LOOKS || shared.contains(&b) || *n < 64 << 10 {
            continue;
        }
        if freeze(*a, *n) {
            made += 1;
            // what was written between this look and the protection
            take(*a, *n, dirty);
        } else {
            // pinned, nothing committed or not all read-write: tried again after another
            // COLD_LOOKS looks, not at every look
            *k = 0;
            STATS[5].fetch_add(1, Ordering::Relaxed);
        }
    }
    for b in shared {
        clean.remove(&b);
    }
    looks.clean = clean;
    STATS[1].fetch_add(made as u64, Ordering::Relaxed);
    if looks.n.is_power_of_two() || looks.n % 4096 == 0 {
        let c = COLD.lock().unwrap_or_else(|e| e.into_inner());
        let mb: usize = c.values().flat_map(|c| c.parts.iter().map(|p| p.1)).sum::<usize>() >> 20;
        let holes = c.values().filter(|c| c.parts.is_empty()).count();
        let s: Vec<u64> = STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
        crate::xlog_info!(
            "watch trap: look {}, {} spans read-only ({mb} MB) or holes ({holes}), this look {} of {} MB looked at in {} pieces; so far {} faults, {} made read-only, {} given back by changes, {} pending reads, {} regions not given back, {} refused",
            looks.n, c.len(), hot >> 20, watched >> 20, pieces.len(), s[0], s[1], s[2], PENDING.lock().unwrap_or_else(|e| e.into_inner()).len(), s[4], s[5]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{GetWriteWatch, VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE};
    const WRITE_WATCH_FLAG_RESET: u32 = 1;

    #[test]
    fn hot_pieces_skip_read_only_spans_and_split_at_blocks() {
        let b = 0x4000_0000usize;
        let ranges = [(b, 3 * BLOCK)];
        let cold = BTreeMap::from([(b + BLOCK, b + 2 * BLOCK)]);
        assert_eq!(hot_pieces(&ranges, &cold), vec![(b, BLOCK), (b + 2 * BLOCK, BLOCK)]);
        let cold = BTreeMap::from([(b + 0x1000, b + 0x3000)]);
        assert_eq!(hot_pieces(&[(b, 0x5000)], &cold), vec![(b, 0x1000), (b + 0x3000, 0x2000)]);
    }

    /// (from the top of the address space: tests that free an address and expect it to stay free
    /// sit low)
    const MEM_TOP_DOWN: u32 = 0x10_0000;

    /// The looks keep one state per process (one scanner in a title): tests that look take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// A test's turn, with the clean counts of an earlier test's blocks gone: `MEM_TOP_DOWN` can
    /// give the next test the same base, which then started with a count already part way up.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        LOOKS.lock().unwrap_or_else(|e| e.into_inner()).clean.clear();
        g
    }

    unsafe fn written(base: usize, n: usize) -> Vec<usize> {
        let mut buf = vec![std::ptr::null_mut::<c_void>(); 4096];
        let (mut k, mut g) = (buf.len(), 0u32);
        assert_eq!(GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, n, buf.as_mut_ptr(), &mut k, &mut g), 0);
        buf[..k].iter().map(|&p| p as usize).collect()
    }

    /// A block left clean goes read-only and out of the looks; a write to it faults once, is
    /// seen by the next look, and the block is looked at again. A read into it (pinned) works.
    #[test]
    fn a_cold_block_traps_its_first_write_and_the_look_sees_it() {
        let _serial = serial();
        unsafe {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            let raw = VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE | MEM_WRITE_WATCH | MEM_TOP_DOWN, PAGE_READWRITE) as usize;
            assert_ne!(raw, 0);
            VirtualFree(raw as *mut c_void, 0, MEM_RELEASE);
            let base = (raw + BLOCK - 1) & !(BLOCK - 1);
            let at = VirtualAlloc(base as *const c_void, BLOCK, MEM_RESERVE | windows_sys::Win32::System::Memory::MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_eq!(at, base);
            let ranges = [(base, BLOCK)];
            let mut dirty = Vec::new();
            let take = |a: usize, n: usize, d: &mut Vec<usize>| d.extend(written(a, n));
            for _ in 0..COLD_LOOKS + 1 {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert!(dirty.is_empty());
            assert!(COLD.lock().unwrap().contains_key(&base), "clean block made read-only");
            std::ptr::write_volatile((base + 5 * 0x1000) as *mut u8, 9);
            assert!(!COLD.lock().unwrap().contains_key(&base), "the write gave it back");
            look(&ranges, &mut dirty, 0x1000, take);
            assert_eq!(dirty, vec![base + 5 * 0x1000]);
            // a pinned read target is writable for the kernel and never made read-only
            dirty.clear();
            for _ in 0..COLD_LOOKS + 1 {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert!(COLD.lock().unwrap().contains_key(&base));
            pin(base + 0x8000, 0x1000, 1);
            changed_now(base + 0x8000, 0x1000);
            assert!(!COLD.lock().unwrap().contains_key(&base));
            for _ in 0..COLD_LOOKS + 1 {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert!(!COLD.lock().unwrap().contains_key(&base), "pinned");
            pin(base + 0x8000, 0x1000, -1);
            changed_now(base, BLOCK);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A clean block that cannot be made read-only (here: committed read-only by the title) is
    /// tried once, then again only after another COLD_LOOKS looks, not at every look.
    #[test]
    fn a_block_that_cannot_be_made_read_only_waits_before_the_next_try() {
        let _serial = serial();
        unsafe {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            use windows_sys::Win32::System::Memory::MEM_COMMIT;
            let raw = VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE | MEM_WRITE_WATCH | MEM_TOP_DOWN, PAGE_READWRITE) as usize;
            assert_ne!(raw, 0);
            let base = (raw + BLOCK - 1) & !(BLOCK - 1);
            assert_eq!(VirtualAlloc(base as *const c_void, BLOCK, MEM_COMMIT, PAGE_READONLY) as usize, base);
            let ranges = [(base, BLOCK)];
            let mut dirty = Vec::new();
            let take = |_: usize, _: usize, _: &mut Vec<usize>| {};
            let refused = || STATS[5].load(Ordering::Relaxed);
            let before = refused();
            for _ in 0..COLD_LOOKS {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert_eq!(refused() - before, 1, "tried once it was clean for COLD_LOOKS looks");
            for _ in 1..COLD_LOOKS {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert_eq!(refused() - before, 1, "not tried again at every look");
            look(&ranges, &mut dirty, 0x1000, take);
            assert_eq!(refused() - before, 2, "tried again after COLD_LOOKS more looks");
            assert!(!COLD.lock().unwrap().contains_key(&base));
            VirtualFree(raw as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A clean block with nothing committed is a hole: out of the looks with nothing protected,
    /// until a commit in it (through a guard) gives it back; the write after the commit is seen.
    #[test]
    fn a_reserved_block_leaves_the_looks_until_a_commit() {
        let _serial = serial();
        unsafe {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            use windows_sys::Win32::System::Memory::MEM_COMMIT;
            let raw = VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE | MEM_WRITE_WATCH | MEM_TOP_DOWN, PAGE_READWRITE) as usize;
            assert_ne!(raw, 0);
            let base = (raw + BLOCK - 1) & !(BLOCK - 1);
            let ranges = [(base, BLOCK)];
            let mut dirty = Vec::new();
            let looked = std::cell::Cell::new(0usize);
            let take = |a: usize, n: usize, d: &mut Vec<usize>| {
                looked.set(looked.get() + n);
                d.extend(written(a, n));
            };
            for _ in 0..COLD_LOOKS {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert!(COLD.lock().unwrap().get(&base).is_some_and(|c| c.parts.is_empty()), "a hole");
            looked.set(0);
            look(&ranges, &mut dirty, 0x1000, take);
            assert_eq!(looked.get(), 0, "out of the looks");
            // what guard() does around a commit (it is a no-op here: the switch is off in tests)
            hold(base + 0x3000, 0x1000);
            assert_eq!(VirtualAlloc((base + 0x3000) as *const c_void, 0x1000, MEM_COMMIT, PAGE_READWRITE) as usize, base + 0x3000);
            pin(base + 0x3000, 0x1000, -1);
            assert!(!COLD.lock().unwrap().contains_key(&base), "the commit gave it back");
            std::ptr::write_volatile((base + 0x3000) as *mut u8, 7);
            look(&ranges, &mut dirty, 0x1000, take);
            assert_eq!(dirty, vec![base + 0x3000]);
            changed_now(base, BLOCK);
            VirtualFree(raw as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A read-only block with 64 KiB decommitted inside it by a path that did not tell the trap
    /// (VirtualProtect of the whole span then fails): giving it back still makes the pages that
    /// are left writable, not read-only with no span to catch their writes.
    #[test]
    fn a_partly_decommitted_cold_block_is_given_back_where_committed() {
        let _serial = serial();
        unsafe {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_DECOMMIT};
            let raw = VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE | MEM_WRITE_WATCH | MEM_TOP_DOWN, PAGE_READWRITE) as usize;
            assert_ne!(raw, 0);
            VirtualFree(raw as *mut c_void, 0, MEM_RELEASE);
            let base = (raw + BLOCK - 1) & !(BLOCK - 1);
            assert_eq!(VirtualAlloc(base as *const c_void, BLOCK, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize, base);
            let ranges = [(base, BLOCK)];
            let mut dirty = Vec::new();
            let take = |a: usize, n: usize, d: &mut Vec<usize>| d.extend(written(a, n));
            for _ in 0..COLD_LOOKS + 1 {
                look(&ranges, &mut dirty, 0x1000, take);
            }
            assert!(COLD.lock().unwrap().contains_key(&base));
            assert_ne!(VirtualFree((base + 0x10000) as *mut c_void, 0x10000, MEM_DECOMMIT), 0);
            changed_now(base, BLOCK);
            for at in [base, base + 0x20000, base + BLOCK - 0x1000] {
                let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)), 0);
                assert_eq!(m.Protect, PAGE_READWRITE, "{at:#x} given back");
            }
            std::ptr::write_volatile((base + 0x20000) as *mut u8, 1);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A pending read whose OVERLAPPED is used for the next read is over: its buffer is reported
    /// and unpinned then, not when the second read ends. One still pending stays with the looks.
    #[test]
    fn an_overlapped_used_again_retires_its_earlier_read() {
        let _serial = serial();
        let pinned = |a: usize| PINS.lock().unwrap().contains_key(&(a & !(BLOCK - 1)));
        let pending = || PENDING.lock().unwrap().len();
        unsafe {
            // two buffers in blocks of their own, an OVERLAPPED and a byte count in a third
            let mem = VirtualAlloc(std::ptr::null(), 4 * BLOCK, MEM_RESERVE | windows_sys::Win32::System::Memory::MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(mem, 0);
            let base = (mem + BLOCK - 1) & !(BLOCK - 1);
            let (first, second, ov) = (base + 0x3000, base + BLOCK + 0x5000, base + 2 * BLOCK);
            let read = ov + 0x100;
            let start = |buf: usize| {
                retire_reused(ov);
                hold(buf, 0x2000);
                hold(read, 4);
                hold(ov, OVERLAPPED_LEN);
                *(ov as *mut usize) = STATUS_PENDING;
                PENDING.lock().unwrap().push((ov, buf, 0x2000, read));
            };
            start(first);
            let mut dirty = Vec::new();
            look(&[], &mut dirty, 0x1000, |_, _, _| {});
            assert!(dirty.is_empty() && pinned(first) && pending() == 1, "a pending read stays pinned");
            // used again while its status says pending: nothing is retired
            retire_reused(ov);
            assert!(pinned(first) && pending() == 1);
            // the first read ends, the title starts the second with the same OVERLAPPED
            *(ov as *mut usize) = 0;
            start(second);
            assert!(!pinned(first) && pinned(second) && pending() == 1, "the first read is retired at the second's start");
            look(&[], &mut dirty, 0x1000, |_, _, _| {});
            assert_eq!(dirty, vec![first, first + 0x1000], "the first buffer is reported once, the second not yet");
            dirty.clear();
            *(ov as *mut usize) = 0;
            look(&[], &mut dirty, 0x1000, |_, _, _| {});
            assert_eq!(dirty, vec![second, second + 0x1000]);
            assert!(!pinned(second) && !pinned(ov) && pending() == 0);
            VirtualFree(mem as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
