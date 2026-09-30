use std::collections::{BTreeMap, BTreeSet};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use serde_json::{Map, Value, json};
use unllm_core::{
    Candidate, Capabilities, Capability, ContentDelta, ContentPart, ConversionContext,
    ConversionOutcome, DynAdapter, Extensions, FinishReason, GenerateRequest, GenerateResponse,
    Instruction, InstructionKind, MediaAsset, MediaSource, Message, Operation, Protocol,
    ResponseFormat, Role, StreamEvent, ToolCall, ToolChoice, ToolDefinition, ToolResult,
    UnifiedRequest, UnifiedResponse, UnllmError, Usage,
};

use crate::types::{MessagesRequest, MessagesResponse};

/// Anthropic Messages codec.
#[derive(Debug, Default)]
pub struct AnthropicAdapter;

impl AnthropicAdapter {
    /// Decodes a typed Anthropic Messages request.
    pub fn decode_generate(
        &self,
        request: MessagesRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateRequest>, UnllmError> {
        let decoded = decode_request(serde_json::to_value(request).map_err(json_error)?, context)?;
        let UnifiedRequest::Generate(value) = decoded.value else {
            return Err(UnllmError::invalid(
                "operation_mismatch",
                "Expected a generation request",
            ));
        };
        Ok(ConversionOutcome {
            value,
            diagnostics: decoded.diagnostics,
        })
    }

    /// Encodes a canonical request as a typed Anthropic Messages request.
    pub fn encode_generate(
        &self,
        request: &GenerateRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<MessagesRequest>, UnllmError> {
        let encoded = encode_request(request, context)?;
        Ok(ConversionOutcome {
            value: serde_json::from_value(encoded.value).map_err(json_error)?,
            diagnostics: encoded.diagnostics,
        })
    }

    /// Decodes a typed Anthropic Messages response.
    pub fn decode_generate_response(
        &self,
        response: MessagesResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateResponse>, UnllmError> {
        let decoded =
            decode_response(serde_json::to_value(response).map_err(json_error)?, context)?;
        let UnifiedResponse::Generate(value) = decoded.value else {
            return Err(UnllmError::invalid(
                "operation_mismatch",
                "Expected a generation response",
            ));
        };
        Ok(ConversionOutcome {
            value,
            diagnostics: decoded.diagnostics,
        })
    }
}

impl DynAdapter for AnthropicAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::AnthropicMessages
    }

    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn decode_request(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
        if operation != Operation::Generate {
            return Err(unsupported_operation(operation));
        }
        decode_request(body, context)
    }

    fn encode_request(
        &self,
        request: &UnifiedRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        let UnifiedRequest::Generate(request) = request else {
            return Err(unsupported_operation(request.operation()));
        };
        let mut outcome = encode_request(request, context)?;
        outcome.diagnostics.extend(context.check_extensions(
            &request.extensions,
            "anthropic",
            "/request",
        )?);
        Ok(outcome)
    }

    fn decode_response(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
        if operation != Operation::Generate {
            return Err(unsupported_operation(operation));
        }
        decode_response(body, context)
    }

    fn encode_response(
        &self,
        response: &UnifiedResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        let UnifiedResponse::Generate(response) = response else {
            return Err(UnllmError::unsupported(
                "anthropic_operation",
                "Anthropic Messages only supports generation responses",
            ));
        };
        let mut outcome = encode_response(response, context)?;
        outcome.diagnostics.extend(context.check_extensions(
            &response.extensions,
            "anthropic",
            "/response",
        )?);
        Ok(outcome)
    }

    fn decode_stream_event(
        &self,
        event_type: Option<&str>,
        data: &[u8],
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
        decode_stream(event_type, data, context)
    }

    fn encode_stream_event(
        &self,
        event: &StreamEvent,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
        encode_stream(event, context)
    }
}

fn capabilities() -> Capabilities {
    Capabilities {
        operations: BTreeSet::from([Operation::Generate]),
        features: BTreeSet::from([
            Capability::Streaming,
            Capability::Text,
            Capability::Image,
            Capability::File,
            Capability::FunctionTools,
            Capability::StructuredOutput,
            Capability::Reasoning,
            Capability::FileUpload,
        ]),
    }
}

fn decode_request(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let request: MessagesRequest = serde_json::from_value(body)
        .map_err(|error| UnllmError::invalid("invalid_anthropic_request", error.to_string()))?;
    let instructions = request
        .system
        .as_ref()
        .map(|system| {
            Ok(Instruction {
                kind: InstructionKind::System,
                priority: 0,
                content: parse_content(system, 0)?,
                extensions: Extensions::new(),
            })
        })
        .transpose()?
        .into_iter()
        .collect();
    let mut messages = Vec::new();
    for (index, message) in request.messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        messages.push(Message {
            role: if role == "assistant" {
                Role::Assistant
            } else {
                Role::User
            },
            content: parse_content(message.get("content").unwrap_or(&Value::Null), index)?,
            id: None,
            extensions: Extensions::new(),
        });
    }
    let tools = request
        .tools
        .iter()
        .filter_map(|tool| {
            Some(ToolDefinition {
                name: tool.get("name")?.as_str()?.to_owned(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                input_schema: tool
                    .get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
                extensions: Extensions::new(),
            })
        })
        .collect();
    let parameters = unllm_core::GenerationParameters {
        max_output_tokens: Some(request.max_tokens),
        temperature: request
            .extra
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
        top_p: request
            .extra
            .get("top_p")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
        top_k: request.extra.get("top_k").and_then(Value::as_u64),
        stop_sequences: request
            .extra
            .get("stop_sequences")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        ..Default::default()
    };
    let reasoning = request
        .extra
        .get("thinking")
        .map(|thinking| unllm_core::ReasoningConfig {
            effort: None,
            budget_tokens: thinking.get("budget_tokens").and_then(Value::as_u64),
            include_summary: None,
        });
    let mut extra = request.extra;
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "thinking",
    ] {
        extra.remove(key);
    }
    let extensions = if extra.is_empty() {
        Extensions::new()
    } else {
        BTreeMap::from([(
            "anthropic".into(),
            Value::Object(extra.into_iter().collect()),
        )])
    };
    Ok(ConversionOutcome::clean(UnifiedRequest::Generate(
        GenerateRequest {
            model: request.model,
            instructions,
            messages,
            tools,
            tool_choice: request.tool_choice.as_ref().and_then(parse_tool_choice),
            response_format: None,
            reasoning,
            parameters,
            stream: request.stream,
            extensions,
        },
    )))
}

fn encode_request(
    request: &GenerateRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    if request.parameters.candidate_count.unwrap_or(1) > 1 {
        context.lossy(
            "anthropic_multiple_candidates",
            "/request/parameters/candidate_count",
            "Anthropic Messages does not support multiple candidates",
        )?;
    }
    let system_parts: Vec<_> = request
        .instructions
        .iter()
        .flat_map(|instruction| instruction.content.iter())
        .cloned()
        .collect();
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(encode_content(&system_parts, context)?)
    };
    let mut messages = Vec::new();
    for message in &request.messages {
        let role = match message.role {
            Role::Assistant => "assistant",
            Role::User | Role::Tool => "user",
        };
        messages.push(json!({"role": role, "content": encode_content(&message.content, context)?}));
    }
    let mut value = serde_json::to_value(MessagesRequest {
        model: request.model.clone(),
        max_tokens: request.parameters.max_output_tokens.unwrap_or(1024),
        system,
        messages,
        tools: request.tools.iter().map(|tool| json!({"name": tool.name, "description": tool.description, "input_schema": tool.input_schema})).collect(),
        tool_choice: request.tool_choice.as_ref().map(encode_tool_choice),
        stream: request.stream,
        extra: BTreeMap::new(),
    })
    .map_err(json_error)?;
    let object = value.as_object_mut().expect("object");
    macro_rules! optional {
        ($name:literal, $value:expr) => {
            if let Some(value) = $value {
                object.insert($name.into(), json!(value));
            }
        };
    }
    optional!("temperature", request.parameters.temperature);
    optional!("top_p", request.parameters.top_p);
    optional!("top_k", request.parameters.top_k);
    if !request.parameters.stop_sequences.is_empty() {
        object.insert(
            "stop_sequences".into(),
            json!(request.parameters.stop_sequences),
        );
    }
    if let Some(reasoning) = &request.reasoning {
        if let Some(budget) = reasoning.budget_tokens {
            object.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
    }
    if let Some(format) = &request.response_format {
        match format {
            ResponseFormat::Text => {}
            ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. } => {
                context.lossy(
                    "anthropic_structured_output",
                    "/request/response_format",
                    "Structured output requires an Anthropic beta feature or tool strategy",
                )?;
            }
        }
    }
    merge_extensions(object, &request.extensions);
    Ok(ConversionOutcome::clean(value))
}

fn decode_response(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let response: MessagesResponse = serde_json::from_value(body)
        .map_err(|error| UnllmError::invalid("invalid_anthropic_response", error.to_string()))?;
    let content = parse_content(&Value::Array(response.content), 0)?;
    let finish_reason = response.stop_reason.as_deref().map(parse_finish_reason);
    let usage = response.usage.as_ref().map(parse_usage);
    let extensions = if response.extra.is_empty() {
        Extensions::new()
    } else {
        BTreeMap::from([(
            "anthropic".into(),
            Value::Object(response.extra.into_iter().collect()),
        )])
    };
    Ok(ConversionOutcome::clean(UnifiedResponse::Generate(
        GenerateResponse {
            id: response.id,
            model: response.model,
            candidates: vec![Candidate {
                index: 0,
                id: None,
                content,
                finish_reason,
                extensions: Extensions::new(),
            }],
            usage,
            extensions,
        },
    )))
}

fn encode_response(
    response: &GenerateResponse,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    if response.candidates.len() > 1 {
        context.lossy(
            "anthropic_multiple_candidates",
            "/response/candidates",
            "Anthropic Messages can encode only one candidate",
        )?;
    }
    let candidate = response.candidates.first();
    Ok(ConversionOutcome::clean(json!({
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": candidate.map(|candidate| encode_content(&candidate.content, context)).transpose()?.unwrap_or_else(|| Value::Array(Vec::new())),
        "stop_reason": candidate.and_then(|candidate| candidate.finish_reason.as_ref()).map(encode_finish_reason),
        "usage": response.usage.as_ref().map(encode_usage),
    })))
}

fn parse_content(value: &Value, message_index: usize) -> Result<Vec<ContentPart>, UnllmError> {
    if let Some(text) = value.as_str() {
        return Ok(vec![ContentPart::text(text)]);
    }
    let mut output = Vec::new();
    for (block_index, block) in value.as_array().into_iter().flatten().enumerate() {
        match block.get("type").and_then(Value::as_str).unwrap_or("text") {
            "text" => output.push(ContentPart::text(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )),
            "image" => output.push(ContentPart::Image {
                asset: parse_source(block.get("source"), "image")?,
            }),
            "document" => output.push(ContentPart::File {
                asset: parse_source(block.get("source"), "application/octet-stream")?,
                filename: block
                    .get("title")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            }),
            "tool_use" => output.push(ContentPart::ToolCall {
                call: ToolCall {
                    id: block
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("call-{message_index}-{block_index}")),
                    name: block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                    extensions: Extensions::new(),
                },
            }),
            "tool_result" => output.push(ContentPart::ToolResult {
                result: ToolResult {
                    call_id: block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    content: parse_content(
                        block.get("content").unwrap_or(&Value::Null),
                        message_index,
                    )?,
                    is_error: block
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    extensions: Extensions::new(),
                },
            }),
            "thinking" | "redacted_thinking" => output.push(ContentPart::Reasoning {
                text: block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                extensions: BTreeMap::from([("anthropic".into(), block.clone())]),
            }),
            _ => {}
        }
    }
    Ok(output)
}

fn encode_content(parts: &[ContentPart], context: &ConversionContext) -> Result<Value, UnllmError> {
    let mut output = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text, .. } => output.push(json!({"type": "text", "text": text})),
            ContentPart::Image { asset } => output.push(json!({"type": "image", "source": encode_source(asset, "anthropic")?})),
            ContentPart::File { asset, filename } => output.push(json!({"type": "document", "source": encode_source(asset, "anthropic")?, "title": filename})),
            ContentPart::ToolCall { call } => output.push(json!({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})),
            ContentPart::ToolResult { result } => output.push(json!({"type": "tool_result", "tool_use_id": result.call_id, "content": encode_content(&result.content, context)?, "is_error": result.is_error})),
            ContentPart::Reasoning { text, extensions } => {
                if let Some(raw) = extensions.get("anthropic") {
                    output.push(raw.clone());
                } else if let Some(text) = text {
                    output.push(json!({"type": "thinking", "thinking": text}));
                }
            }
            ContentPart::Refusal { reason } => output.push(json!({"type": "text", "text": reason})),
            ContentPart::Audio { .. } | ContentPart::Video { .. } => {
                context.lossy(
                    "anthropic_media",
                    "/content",
                    "Anthropic Messages cannot encode this media type without a beta extension",
                )?;
            }
        }
    }
    Ok(Value::Array(output))
}

fn parse_source(value: Option<&Value>, default_mime: &str) -> Result<MediaAsset, UnllmError> {
    let value = value.unwrap_or(&Value::Null);
    match value.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let mime_type = value
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or(default_mime);
            let bytes = value
                .get("data")
                .and_then(Value::as_str)
                .map(|data| STANDARD.decode(data))
                .transpose()
                .map_err(|error| {
                    UnllmError::invalid("invalid_anthropic_base64", error.to_string())
                })?
                .unwrap_or_default();
            Ok(MediaAsset::inline(mime_type, Bytes::from(bytes)))
        }
        Some("url") => Ok(MediaAsset::url(
            value.get("url").and_then(Value::as_str).unwrap_or_default(),
            value
                .get("media_type")
                .and_then(Value::as_str)
                .map(str::to_owned),
        )),
        Some("file") => Ok(MediaAsset {
            source: MediaSource::ProviderFile {
                provider: "anthropic".into(),
                id: value
                    .get("file_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
                mime_type: value
                    .get("media_type")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
            width: None,
            height: None,
            sha256: None,
            extensions: Extensions::new(),
        }),
        _ => Err(UnllmError::invalid(
            "invalid_anthropic_source",
            "Unsupported Anthropic media source",
        )),
    }
}

fn encode_source(asset: &MediaAsset, provider: &str) -> Result<Value, UnllmError> {
    match &asset.source {
        MediaSource::Inline { mime_type, data } => {
            Ok(json!({"type": "base64", "media_type": mime_type, "data": STANDARD.encode(data)}))
        }
        MediaSource::Url { url, .. } => Ok(json!({"type": "url", "url": url})),
        MediaSource::ProviderFile {
            provider: owner,
            id,
            ..
        } if owner == provider => Ok(json!({"type": "file", "file_id": id})),
        MediaSource::ProviderFile { .. } => Err(UnllmError::unsupported(
            "foreign_file_id",
            "A provider file identifier cannot be sent to Anthropic",
        )),
    }
}

fn parse_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value.get("type").and_then(Value::as_str) {
        Some("auto") => Some(ToolChoice::Auto),
        Some("none") => Some(ToolChoice::None),
        Some("any") => Some(ToolChoice::Required),
        Some("tool") => value
            .get("name")
            .and_then(Value::as_str)
            .map(|name| ToolChoice::Function { name: name.into() }),
        _ => None,
    }
}

fn encode_tool_choice(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Required => json!({"type": "any"}),
        ToolChoice::Function { name } => json!({"type": "tool", "name": name}),
    }
}

fn decode_stream(
    event_type: Option<&str>,
    data: &[u8],
    _context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
    let value: Value = serde_json::from_slice(data).map_err(json_error)?;
    let kind = event_type
        .or_else(|| value.get("type").and_then(Value::as_str))
        .unwrap_or("unknown");
    let index = value
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    let event = match kind {
        "message_start" => StreamEvent::ResponseStart {
            id: value
                .pointer("/message/id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            model: value
                .pointer("/message/model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            extensions: Extensions::new(),
        },
        "content_block_start" => {
            let block = value.get("content_block").cloned().unwrap_or(Value::Null);
            let part = parse_content(&Value::Array(vec![block]), 0)?
                .into_iter()
                .next();
            StreamEvent::ContentBlockStart {
                candidate_index: 0,
                block_index: index,
                part,
            }
        }
        "content_block_delta" => match value.pointer("/delta/type").and_then(Value::as_str) {
            Some("text_delta") => StreamEvent::ContentDelta {
                candidate_index: 0,
                block_index: index,
                delta: ContentDelta::Text {
                    text: value
                        .pointer("/delta/text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                },
            },
            Some("thinking_delta") => StreamEvent::ContentDelta {
                candidate_index: 0,
                block_index: index,
                delta: ContentDelta::Reasoning {
                    text: value
                        .pointer("/delta/thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                },
            },
            Some("input_json_delta") => StreamEvent::ContentDelta {
                candidate_index: 0,
                block_index: index,
                delta: ContentDelta::ToolArguments {
                    call_id: format!("call-0-{index}"),
                    name: None,
                    fragment: value
                        .pointer("/delta/partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                },
            },
            _ => StreamEvent::Raw {
                provider: "anthropic".into(),
                event_type: Some(kind.into()),
                payload: value,
            },
        },
        "content_block_stop" => StreamEvent::ContentBlockStop {
            candidate_index: 0,
            block_index: index,
        },
        "message_delta" => {
            if let Some(usage) = value.get("usage") {
                StreamEvent::Usage {
                    usage: parse_usage(usage),
                }
            } else {
                StreamEvent::Finish {
                    candidate_index: 0,
                    reason: value
                        .pointer("/delta/stop_reason")
                        .and_then(Value::as_str)
                        .map(parse_finish_reason),
                }
            }
        }
        "message_stop" => StreamEvent::Finish {
            candidate_index: 0,
            reason: Some(FinishReason::Stop),
        },
        "error" => StreamEvent::Error {
            error: UnllmError {
                code: value
                    .pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("anthropic_stream_error")
                    .into(),
                kind: unllm_core::ErrorKind::Upstream,
                message: value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("Anthropic stream failed")
                    .into(),
                source_protocol: Some(Protocol::AnthropicMessages),
                target: None,
                operation: Some(Operation::Generate),
                metadata: None,
            },
        },
        _ => StreamEvent::Raw {
            provider: "anthropic".into(),
            event_type: Some(kind.into()),
            payload: value,
        },
    };
    Ok(ConversionOutcome::clean(vec![event]))
}

fn encode_stream(
    event: &StreamEvent,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
    let (kind, value) = match event {
        StreamEvent::ResponseStart { id, model, .. } => (
            "message_start",
            json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": null, "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        ),
        StreamEvent::ContentBlockStart {
            block_index, part, ..
        } => (
            "content_block_start",
            json!({"type": "content_block_start", "index": block_index, "content_block": part.as_ref().map(|part| encode_content(std::slice::from_ref(part), context)).transpose()?.and_then(|value| value.as_array().and_then(|values| values.first()).cloned()).unwrap_or_else(|| json!({"type": "text", "text": ""}))}),
        ),
        StreamEvent::ContentDelta {
            block_index, delta, ..
        } => match delta {
            ContentDelta::Text { text } => (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "text_delta", "text": text}}),
            ),
            ContentDelta::Reasoning { text } => (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "thinking_delta", "thinking": text}}),
            ),
            ContentDelta::ToolArguments { fragment, .. } => (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "input_json_delta", "partial_json": fragment}}),
            ),
            ContentDelta::Media { .. } => {
                return Err(UnllmError::unsupported(
                    "anthropic_media_delta",
                    "Anthropic Messages cannot encode the media delta",
                ));
            }
        },
        StreamEvent::ContentBlockStop { block_index, .. } => (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": block_index}),
        ),
        StreamEvent::Usage { usage } => (
            "message_delta",
            json!({"type": "message_delta", "delta": {}, "usage": encode_usage(usage)}),
        ),
        StreamEvent::Finish { reason, .. } => (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": reason.as_ref().map(encode_finish_reason), "stop_sequence": null}, "usage": {"output_tokens": 0}}),
        ),
        StreamEvent::Error { error } => (
            "error",
            json!({"type": "error", "error": {"type": error.code, "message": error.message}}),
        ),
        StreamEvent::Raw {
            provider,
            event_type,
            payload,
        } if provider == "anthropic" => {
            (event_type.as_deref().unwrap_or("unknown"), payload.clone())
        }
        StreamEvent::Raw { .. } => {
            let warning = context.lossy(
                "raw_stream_event",
                "/",
                "A provider-specific stream event cannot be represented by Anthropic Messages",
            )?;
            return Ok(ConversionOutcome {
                value: Vec::new(),
                diagnostics: vec![warning],
            });
        }
    };
    Ok(ConversionOutcome::clean(vec![sse(kind, &value)]))
}

fn parse_usage(value: &Value) -> Usage {
    Usage {
        input_tokens: value.get("input_tokens").and_then(Value::as_u64),
        output_tokens: value.get("output_tokens").and_then(Value::as_u64),
        total_tokens: None,
        cached_tokens: value.get("cache_read_input_tokens").and_then(Value::as_u64),
        reasoning_tokens: None,
        details: BTreeMap::new(),
    }
}

fn encode_usage(usage: &Usage) -> Value {
    json!({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cache_read_input_tokens": usage.cached_tokens})
}

fn parse_finish_reason(value: &str) -> FinishReason {
    match value {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCall,
        "refusal" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.into()),
    }
}

fn encode_finish_reason(value: &FinishReason) -> &str {
    match value {
        FinishReason::Stop => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolCall => "tool_use",
        FinishReason::ContentFilter => "refusal",
        FinishReason::Error => "error",
        FinishReason::Other(raw) => raw,
    }
}

fn merge_extensions(object: &mut Map<String, Value>, extensions: &Extensions) {
    if let Some(Value::Object(extra)) = extensions.get("anthropic") {
        for (key, value) in extra {
            object.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
}

fn unsupported_operation(operation: Operation) -> UnllmError {
    UnllmError::unsupported(
        "anthropic_operation",
        format!("Anthropic Messages does not support {operation:?}"),
    )
}

fn json_error(error: serde_json::Error) -> UnllmError {
    UnllmError::invalid("invalid_json", error.to_string())
}

fn sse(event_type: &str, value: &Value) -> Vec<u8> {
    format!("event: {event_type}\ndata: {value}\n\n").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unllm_core::ConversionMode;

    #[test]
    fn decodes_tool_use() {
        let context = ConversionContext {
            source: Protocol::AnthropicMessages,
            target: Protocol::AnthropicMessages,
            mode: ConversionMode::Strict,
            capabilities: capabilities(),
        };
        let body = json!({"model": "claude-test", "max_tokens": 64, "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "Paris"}}]}]});
        let decoded = AnthropicAdapter
            .decode_request(Operation::Generate, body, &context)
            .unwrap();
        let UnifiedRequest::Generate(request) = decoded.value else {
            panic!("expected generation")
        };
        assert!(matches!(
            request.messages[0].content[0],
            ContentPart::ToolCall { .. }
        ));
    }
}
