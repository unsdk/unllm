//! OpenAI Chat Completions, Responses, Embeddings, and Images codecs.

#![allow(missing_docs)]

mod codec;
mod types;

pub use codec::{OpenAiChatAdapter, OpenAiResponsesAdapter};
pub use types::*;
