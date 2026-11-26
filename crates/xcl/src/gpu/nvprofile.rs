//! An NVIDIA driver profile (DRS) for an executable, written through NVAPI (#2403).
//!
//! NVIDIA's D3D12 driver reserves 512 MB of VA for the machine code of the pipelines in use and
//! removes the device when a submission needs more. Setting `0x001ABC0F` ("the amount of VA space
//! in bytes reserved for the shader heap") in a profile for the executable raises it; NVIDIA's own
//! profiles use 1 or 2 GiB. This writes that one setting into a profile named [`PROFILE`] that
//! holds only XCL's executables, so [`remove`] takes it out again without touching anything else.

use std::ffi::c_void;

/// The setting: VA space in bytes reserved for the shader heap (a multiple of 64 KiB).
pub const SHADER_HEAP_VA: u32 = 0x001A_BC0F;
/// The profile XCL owns.
pub const PROFILE: &str = "XCL (xcl-dev)";

const UNICODE_LEN: usize = 2048;
type Wide = [u16; UNICODE_LEN];
type Handle = *mut c_void;

#[repr(C)]
struct Profile {
    version: u32,
    name: Wide,
    gpu_support: u32,
    is_predefined: u32,
    num_apps: u32,
    num_settings: u32,
}

#[repr(C)]
struct Application {
    version: u32,
    is_predefined: u32,
    app_name: Wide,
    friendly_name: Wide,
    launcher: Wide,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Binary {
    len: u32,
    data: [u8; 4096],
}

#[repr(C)]
#[derive(Clone, Copy)]
union Value {
    u32: u32,
    binary: Binary,
    wide: Wide,
}

#[repr(C)]
struct Setting {
    version: u32,
    name: Wide,
    id: u32,
    kind: u32,
    location: u32,
    is_current_predefined: u32,
    is_predefined_valid: u32,
    predefined: Value,
    current: Value,
}

const fn ver<T>(v: u32) -> u32 {
    std::mem::size_of::<T>() as u32 | (v << 16)
}

fn wide(s: &str) -> Wide {
    let mut w = [0u16; UNICODE_LEN];
    for (d, c) in w.iter_mut().zip(s.encode_utf16().take(UNICODE_LEN - 1)) {
        *d = c;
    }
    w
}

fn zeroed<T>() -> Box<T> {
    // SAFETY: every type here is plain integers / arrays, for which all-zero is valid.
    unsafe { Box::new(std::mem::zeroed()) }
}

type Status = i32;
const OK: Status = 0;
const INVALID_USER_PRIVILEGE: Status = -137;

struct Api {
    create_session: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy_session: unsafe extern "C" fn(Handle) -> Status,
    load: unsafe extern "C" fn(Handle) -> Status,
    save: unsafe extern "C" fn(Handle) -> Status,
    find_app: unsafe extern "C" fn(Handle, *const u16, *mut Handle, *mut Application) -> Status,
    find_profile: unsafe extern "C" fn(Handle, *const u16, *mut Handle) -> Status,
    create_profile: unsafe extern "C" fn(Handle, *mut Profile, *mut Handle) -> Status,
    delete_profile: unsafe extern "C" fn(Handle, Handle) -> Status,
    create_app: unsafe extern "C" fn(Handle, Handle, *mut Application) -> Status,
    set_setting: unsafe extern "C" fn(Handle, Handle, *mut Setting) -> Status,
    get_setting: unsafe extern "C" fn(Handle, Handle, u32, *mut Setting) -> Status,
}

fn api() -> Result<Api, String> {
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    unsafe {
        let lib = LoadLibraryW(windows::core::w!("nvapi64.dll")).map_err(|e| format!("no nvapi64.dll: {e}"))?;
        let qi = GetProcAddress(lib, windows::core::s!("nvapi_QueryInterface")).ok_or("no nvapi_QueryInterface")?;
        let qi: unsafe extern "C" fn(u32) -> *const c_void = std::mem::transmute(qi);
        let get = |id: u32| {
            let p = qi(id);
            if p.is_null() { Err(format!("NVAPI has no function {id:#010x}")) } else { Ok(p) }
        };
        let init: unsafe extern "C" fn() -> Status = std::mem::transmute(get(0x0150_E828)?);
        let s = init();
        if s != OK {
            return Err(format!("NvAPI_Initialize: {s}"));
        }
        Ok(Api {
            create_session: std::mem::transmute(get(0x0694_D52E)?),
            destroy_session: std::mem::transmute(get(0xDAD9_CFF8)?),
            load: std::mem::transmute(get(0x375D_BD6B)?),
            save: std::mem::transmute(get(0xFCBC_7E14)?),
            find_app: std::mem::transmute(get(0xEEE5_66B2)?),
            find_profile: std::mem::transmute(get(0x7E4A_9A0B)?),
            create_profile: std::mem::transmute(get(0xCC17_6068)?),
            delete_profile: std::mem::transmute(get(0x1709_3206)?),
            create_app: std::mem::transmute(get(0x4347_A9DE)?),
            set_setting: std::mem::transmute(get(0x577D_D202)?),
            get_setting: std::mem::transmute(get(0x73BF_8338)?),
        })
    }
}

struct Session<'a> {
    api: &'a Api,
    h: Handle,
}

impl<'a> Session<'a> {
    fn open(api: &'a Api) -> Result<Self, String> {
        let mut h = std::ptr::null_mut();
        unsafe {
            check("CreateSession", (api.create_session)(&mut h))?;
            let s = Session { api, h };
            check("LoadSettings", (api.load)(h))?;
            Ok(s)
        }
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.destroy_session)(self.h) };
    }
}

fn check(what: &str, s: Status) -> Result<(), String> {
    if s == OK { Ok(()) } else { Err(format!("NvAPI_DRS_{what}: {s}")) }
}

/// The shader heap the driver will reserve for `exe` (a file name such as `xcl.exe`): `None` where
/// the exe is in no profile or its profile does not set it (then the driver's 512 MB applies).
pub fn status(exe: &str) -> Result<Option<u32>, String> {
    let api = api()?;
    let s = Session::open(&api)?;
    let name = wide(exe);
    let mut profile = std::ptr::null_mut();
    let mut app = zeroed::<Application>();
    app.version = ver::<Application>(1);
    unsafe {
        if (api.find_app)(s.h, name.as_ptr(), &mut profile, &mut *app) != OK {
            return Ok(None);
        }
        let mut set = zeroed::<Setting>();
        set.version = ver::<Setting>(1);
        if (api.get_setting)(s.h, profile, SHADER_HEAP_VA, &mut *set) != OK {
            return Ok(None);
        }
        Ok(Some(set.current.u32))
    }
}

/// Puts `exes` into [`PROFILE`] (made if missing) with the shader heap at `bytes`, and saves.
/// Writing the driver's settings may need an elevated process.
pub fn install(exes: &[&str], bytes: u32) -> Result<(), String> {
    if bytes % 0x1_0000 != 0 {
        return Err("the shader heap must be a multiple of 64 KiB".into());
    }
    let api = api()?;
    let s = Session::open(&api)?;
    let pname = wide(PROFILE);
    let mut profile = std::ptr::null_mut();
    unsafe {
        if (api.find_profile)(s.h, pname.as_ptr(), &mut profile) != OK {
            let mut p = zeroed::<Profile>();
            p.version = ver::<Profile>(1);
            p.name = pname;
            check("CreateProfile", (api.create_profile)(s.h, &mut *p, &mut profile))?;
        }
        for exe in exes {
            let mut found = std::ptr::null_mut();
            let mut a = zeroed::<Application>();
            a.version = ver::<Application>(1);
            let name = wide(exe);
            if (api.find_app)(s.h, name.as_ptr(), &mut found, &mut *a) == OK && found == profile {
                continue;
            }
            let mut a = zeroed::<Application>();
            a.version = ver::<Application>(1);
            a.app_name = name;
            a.friendly_name = wide(exe);
            check("CreateApplication", (api.create_app)(s.h, profile, &mut *a))?;
        }
        let mut set = zeroed::<Setting>();
        set.version = ver::<Setting>(1);
        set.id = SHADER_HEAP_VA;
        set.kind = 0; // NVDRS_DWORD_TYPE
        set.current.u32 = bytes;
        let r = (api.set_setting)(s.h, profile, &mut *set);
        if r == INVALID_USER_PRIVILEGE {
            return Err("writing the driver's settings needs an elevated prompt (run as administrator)".into());
        }
        check("SetSetting", r)?;
        check("SaveSettings", (api.save)(s.h))
    }
}

/// Deletes [`PROFILE`], if there is one.
pub fn remove() -> Result<bool, String> {
    let api = api()?;
    let s = Session::open(&api)?;
    let pname = wide(PROFILE);
    let mut profile = std::ptr::null_mut();
    unsafe {
        if (api.find_profile)(s.h, pname.as_ptr(), &mut profile) != OK {
            return Ok(false);
        }
        check("DeleteProfile", (api.delete_profile)(s.h, profile))?;
        check("SaveSettings", (api.save)(s.h))?;
    }
    Ok(true)
}

/// What `xcl nvprofile` is asked to do.
#[derive(Debug, PartialEq)]
enum Command {
    Status,
    Install(u32),
    Remove,
}

fn parse(args: &[String]) -> Result<Command, String> {
    let mut p = crate::cli::Parser::new(args);
    let cmd = match p.next_arg() {
        None | Some("status") => Command::Status,
        Some("remove") => Command::Remove,
        Some("install") => {
            let mut mb = 1024;
            while let Some(a) = p.next_arg() {
                match a {
                    "--mb" => mb = p.parse_usize("--mb")?,
                    _ => return Err(crate::cli::Parser::unexpected(a)),
                }
            }
            if !(512..=4095).contains(&mb) {
                return Err(format!("--mb {mb}: expected 512 - 4096"));
            }
            return Ok(Command::Install((mb as u32) << 20));
        }
        Some(a) => return Err(crate::cli::Parser::unexpected(a)),
    };
    match p.next_arg() {
        Some(a) => Err(crate::cli::Parser::unexpected(a)),
        None => Ok(cmd),
    }
}

/// `xcl nvprofile [status | install [--mb N] | remove]`.
pub fn run(args: &[String]) -> i32 {
    let cmd = match parse(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("xcl nvprofile: {e}");
            return 2;
        }
    };
    let done = match cmd {
        Command::Status => Ok(()),
        Command::Install(bytes) => install(&["xcl.exe"], bytes).map(|()| println!("xcl.exe is in the profile \"{PROFILE}\"")),
        Command::Remove => remove().map(|had| println!("{}", if had { "profile removed" } else { "no XCL profile" })),
    };
    let shown = done.and_then(|()| status("xcl.exe")).map(|v| match v {
        Some(b) => println!("xcl.exe: shader heap {} MB", b >> 20),
        None => println!("xcl.exe: the driver's default shader heap (512 MB on D3D12)"),
    });
    match shown {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("xcl nvprofile: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_nvapi() {
        assert_eq!(std::mem::size_of::<Profile>(), 4 + 4096 + 16);
        assert_eq!(std::mem::size_of::<Application>(), 8 + 3 * 4096);
        assert_eq!(std::mem::size_of::<Setting>(), 4 + 4096 + 20 + 2 * 4100);
    }

    #[test]
    fn the_command_line_is_checked() {
        let a = |s: &[&str]| parse(&s.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        assert_eq!(a(&[]), Ok(Command::Status));
        assert_eq!(a(&["remove"]), Ok(Command::Remove));
        assert_eq!(a(&["install"]), Ok(Command::Install(1 << 30)));
        assert_eq!(a(&["install", "--mb", "2048"]), Ok(Command::Install(1 << 31)));
        assert_eq!(a(&["install", "--mb", "4095"]), Ok(Command::Install(4095u32 << 20)));
        assert!(a(&["install", "--mb", "4096"]).is_err(), "4096 MB overflows u32 when shifted");
        assert!(a(&["install", "--mb", "100"]).is_err());
        assert!(a(&["install", "--mb"]).is_err());
        assert!(a(&["status", "x"]).is_err());
        assert!(a(&["frobnicate"]).is_err());
    }
}
