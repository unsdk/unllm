use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{Operation, Protocol};

/// Stable high-level error categories used by the gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorKind {
    Authentication,
    RateLimit,
    InvalidRequest,
    Unsupported,
    Upstream,
    Timeout,
    Cancelled,
    Internal,
}

/// A normalized error safe to encode for another protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Error)]
#[error("{code}: {message}")]
pub struct UnllmError {
    /// Stable machine-readable code.
    pub code: String,
    /// High-level category.
    pub kind: ErrorKind,
    /// Safe English message.
    pub message: String,
    /// Optional source protocol.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_protocol: Option<Protocol>,
    /// Optional target protocol.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Protocol>,
    /// Optional operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<Operation>,
    /// Safe structured metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl UnllmError {
    /// Creates an invalid-request error.
    #[must_use]
    pub fn invalid(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            kind: ErrorKind::InvalidRequest,
            message: message.into(),
            source_protocol: None,
            target: None,
            operation: None,
            metadata: None,
        }
    }

    /// Creates an unsupported-capability error.
    #[must_use]
    pub fn unsupported(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            kind: ErrorKind::Unsupported,
            message: message.into(),
            source_protocol: None,
            target: None,
            operation: None,
            metadata: None,
        }
    }
}
