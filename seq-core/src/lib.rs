//! seq_manip's domain code: `state`, `thumbs` and `commands` are the Tauri
//! app's own files, vendored unchanged; `commands.rs` names Tauri, and
//! `../tauri-shim` gives it just enough Tauri to build. Left out: `lib.rs`
//! (the Tauri builder) and `sentry_init.rs`.
//!
//! `backend` is ours: the managed state and one plain call per command.

#![allow(dead_code, unused_imports)]

pub mod state;
pub mod thumbs;
pub mod commands;

pub mod backend;

pub use sequitur;
