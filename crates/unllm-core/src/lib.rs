//! Protocol-neutral data model and conversion contracts used by unllm.

mod adapter;
mod assets;
mod conversion;
mod error;
mod model;
mod stream;

pub use adapter::*;
pub use assets::*;
pub use conversion::*;
pub use error::*;
pub use model::*;
pub use stream::*;

/// The version of the canonical JSON envelope emitted by this crate.
pub const SCHEMA_VERSION: u32 = 1;
