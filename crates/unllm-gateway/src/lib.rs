//! Deployable multi-protocol gateway built on the unllm conversion crates.

#![allow(missing_docs)]

mod config;
mod media;
mod server;
mod sse;

pub use config::*;
pub use server::run;
