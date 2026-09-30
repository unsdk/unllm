use std::fmt::Debug;

use serde_json::Value;

use crate::{
    Capabilities, ConversionContext, ConversionOutcome, Operation, Protocol, StreamEvent,
    UnifiedRequest, UnifiedResponse, UnllmError,
};

/// An object-safe protocol codec used for runtime dispatch.
pub trait DynAdapter: Debug + Send + Sync {
    /// Protocol represented by this adapter.
    fn protocol(&self) -> Protocol;

    /// Protocol-level maximum capabilities.
    fn capabilities(&self) -> Capabilities;

    /// Decodes a protocol request body into the canonical model.
    fn decode_request(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError>;

    /// Encodes a canonical request into a protocol request body.
    fn encode_request(
        &self,
        request: &UnifiedRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError>;

    /// Decodes a complete protocol response.
    fn decode_response(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError>;

    /// Encodes a complete canonical response.
    fn encode_response(
        &self,
        response: &UnifiedResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError>;

    /// Decodes one protocol stream event.
    fn decode_stream_event(
        &self,
        event_type: Option<&str>,
        data: &[u8],
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError>;

    /// Encodes one canonical stream event into zero or more SSE frames.
    fn encode_stream_event(
        &self,
        event: &StreamEvent,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError>;
}

/// Converts a complete request between two protocol adapters through the canonical model.
pub fn convert_request(
    source: &dyn DynAdapter,
    target: &dyn DynAdapter,
    operation: Operation,
    body: Value,
    mode: crate::ConversionMode,
    target_capabilities: Capabilities,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let decode_context = ConversionContext {
        source: source.protocol(),
        target: target.protocol(),
        mode,
        capabilities: target_capabilities.clone(),
    };
    let decoded = source.decode_request(operation, body, &decode_context)?;
    decoded.value.validate()?;
    let mut encoded = target.encode_request(&decoded.value, &decode_context)?;
    let mut diagnostics = decoded.diagnostics;
    diagnostics.append(&mut encoded.diagnostics);
    Ok(ConversionOutcome {
        value: encoded.value,
        diagnostics,
    })
}

/// Converts a complete response between two protocol adapters through the canonical model.
pub fn convert_response(
    source: &dyn DynAdapter,
    target: &dyn DynAdapter,
    operation: Operation,
    body: Value,
    mode: crate::ConversionMode,
    target_capabilities: Capabilities,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let context = ConversionContext {
        source: source.protocol(),
        target: target.protocol(),
        mode,
        capabilities: target_capabilities,
    };
    let decoded = source.decode_response(operation, body, &context)?;
    let mut encoded = target.encode_response(&decoded.value, &context)?;
    let mut diagnostics = decoded.diagnostics;
    diagnostics.append(&mut encoded.diagnostics);
    Ok(ConversionOutcome {
        value: encoded.value,
        diagnostics,
    })
}

/// Converts one stream event between two protocol adapters.
pub fn convert_stream_event(
    source: &dyn DynAdapter,
    target: &dyn DynAdapter,
    event_type: Option<&str>,
    data: &[u8],
    mode: crate::ConversionMode,
    target_capabilities: Capabilities,
) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
    let context = ConversionContext {
        source: source.protocol(),
        target: target.protocol(),
        mode,
        capabilities: target_capabilities,
    };
    let decoded = source.decode_stream_event(event_type, data, &context)?;
    let mut frames = Vec::new();
    let mut diagnostics = decoded.diagnostics;
    for event in decoded.value {
        let mut encoded = target.encode_stream_event(&event, &context)?;
        frames.append(&mut encoded.value);
        diagnostics.append(&mut encoded.diagnostics);
    }
    Ok(ConversionOutcome {
        value: frames,
        diagnostics,
    })
}
