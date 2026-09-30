use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Capabilities, Extensions, Protocol, UnllmError};

/// Conversion behavior when the target cannot represent source semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConversionMode {
    /// Reject every known semantic loss.
    #[default]
    Strict,
    /// Produce the best supported target and report every loss.
    Lenient,
}

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
}

/// A stable diagnostic emitted during conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Diagnostic {
    /// Stable diagnostic code.
    pub code: String,
    /// Diagnostic severity.
    pub severity: DiagnosticSeverity,
    /// Canonical JSON path associated with the issue.
    pub path: String,
    /// Source protocol.
    pub source: Protocol,
    /// Target protocol.
    pub target: Protocol,
    /// Short English message.
    pub message: String,
    /// Optional machine-readable metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A successfully converted value and its non-fatal diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConversionOutcome<T> {
    /// Converted value.
    pub value: T,
    /// Informational and lossy-conversion diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Diagnostic>,
}

impl<T> ConversionOutcome<T> {
    /// Creates an outcome without diagnostics.
    #[must_use]
    pub const fn clean(value: T) -> Self {
        Self {
            value,
            diagnostics: Vec::new(),
        }
    }

    /// Maps the value while retaining diagnostics.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> ConversionOutcome<U> {
        ConversionOutcome {
            value: map(self.value),
            diagnostics: self.diagnostics,
        }
    }
}

/// Context applied to a single conversion.
#[derive(Debug, Clone)]
pub struct ConversionContext {
    /// Source protocol.
    pub source: Protocol,
    /// Target protocol.
    pub target: Protocol,
    /// Strict or lenient behavior.
    pub mode: ConversionMode,
    /// Route-level target capabilities.
    pub capabilities: Capabilities,
}

impl ConversionContext {
    /// Handles a lossy field according to the selected mode.
    pub fn lossy(
        &self,
        code: impl Into<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<Diagnostic, UnllmError> {
        let code = code.into();
        let message = message.into();
        if self.mode == ConversionMode::Strict {
            let mut error = UnllmError::unsupported(code, message);
            error.source_protocol = Some(self.source);
            error.target = Some(self.target);
            return Err(error);
        }
        Ok(Diagnostic {
            code,
            severity: DiagnosticSeverity::Warning,
            path: path.into(),
            source: self.source,
            target: self.target,
            message,
            metadata: None,
        })
    }

    /// Creates an informational diagnostic.
    #[must_use]
    pub fn info(
        &self,
        code: impl Into<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Diagnostic {
        Diagnostic {
            code: code.into(),
            severity: DiagnosticSeverity::Info,
            path: path.into(),
            source: self.source,
            target: self.target,
            message: message.into(),
            metadata: None,
        }
    }

    /// Validates namespaced extensions before encoding a target protocol.
    pub fn check_extensions(
        &self,
        extensions: &Extensions,
        target_namespace: &str,
        path: &str,
    ) -> Result<Vec<Diagnostic>, UnllmError> {
        let mut diagnostics = Vec::new();
        for namespace in extensions.keys() {
            if namespace != target_namespace {
                diagnostics.push(self.lossy(
                    "foreign_provider_extension",
                    format!("{path}/extensions/{namespace}"),
                    format!(
                        "Extension namespace {namespace} is not understood by {target_namespace}"
                    ),
                )?);
            }
        }
        Ok(diagnostics)
    }
}
