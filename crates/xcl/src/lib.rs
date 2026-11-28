//! XCL — Xbox Series X|S → Windows compatibility layer.
//!
//! See `docs/specs/2026-11-20-xcl-design.md` for the design.

// Function → address casts are the whole point of an import resolver.
#![allow(function_casts_as_integer)]

#[macro_use]
pub mod log;
pub mod alias;
pub mod manifest;
pub mod memmap;
pub mod memdiag;
pub mod dxc_heap;

// The counting allocator costs two contended updates per allocation; a production build does not
// pay them (`memdiag`'s lines are off there too, and its counters stay readable).
#[cfg(not(feature = "production"))]
#[global_allocator]
static ALLOCATOR: memdiag::Counting = memdiag::Counting;
#[cfg(feature = "production")]
#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;
pub mod config;
pub mod control;
pub mod disasm;
pub mod eh;
pub mod gpu;
pub mod hle;
pub mod hooks;
pub mod host;
pub mod hosting;
pub mod insn;
pub mod inspect;
pub mod launcher;
pub mod loader;
pub mod modules;
pub mod pe;
pub mod preflight;
pub mod progress;
pub mod progress_window;
mod live_stall;
pub mod report;
pub mod resolver;
pub mod rpf;
mod rpf_probe;
pub mod shaders;
pub mod state;
pub mod stubs;
pub mod tls;
pub mod util;
pub mod vfs;
pub mod vfs_hooks;
pub mod wait_handles;
pub mod warm_up_json;
