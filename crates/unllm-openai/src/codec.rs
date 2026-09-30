use std::collections::{BTreeMap, BTreeSet};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use serde_json::{Map, Value, json};
use unllm_core::{
    Annotation, Candidate, Capabilities, Capability, ContentDelta, ContentPart, ConversionContext,
    ConversionOutcome, DynAdapter, EmbedRequest, EmbedResponse, Embedding, EmbeddingInput,
    Extensions, FinishReason, GenerateRequest, GenerateResponse, ImageRequest, ImageResponse,
    ImageTask, Instruction, InstructionKind, MediaAsset, MediaSource, Message, Operation, Protocol,
    ResponseFormat, Role, StreamEvent, ToolCall, ToolChoice, ToolDefinition, ToolResult,
    UnifiedRequest, UnifiedResponse, UnllmError, Usage,
};

use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingsRequest, ImagesRequest,
    ResponsesRequest, ResponsesResponse,
};

/// OpenAI Chat Completions codec.
#[derive(Debug, Default)]
pub struct OpenAiChatAdapter;

/// OpenAI Responses codec.
#[derive(Debug, Default)]
pub struct OpenAiResponsesAdapter;

impl OpenAiChatAdapter {
    /// Decodes a typed Chat Completions request.
    pub fn decode_generate(
        &self,
        request: ChatCompletionRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateRequest>, UnllmError> {
        decode_chat_request(serde_json::to_value(request).map_err(json_error)?, context)?
            .map_result()
    }

    /// Encodes a canonical request as a typed Chat Completions request.
    pub fn encode_generate(
        &self,
        request: &GenerateRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<ChatCompletionRequest>, UnllmError> {
        encode_chat_request(request, context)?.try_map("invalid_openai_chat_request")
    }

    /// Decodes a typed Chat Completions response.
    pub fn decode_generate_response(
        &self,
        response: ChatCompletionResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateResponse>, UnllmError> {
        decode_chat_response(serde_json::to_value(response).map_err(json_error)?, context)?
            .map_response()
    }
}

impl OpenAiResponsesAdapter {
    /// Decodes a typed Responses request.
    pub fn decode_generate(
        &self,
        request: ResponsesRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateRequest>, UnllmError> {
        decode_responses_request(serde_json::to_value(request).map_err(json_error)?, context)?
            .map_result()
    }

    /// Encodes a canonical request as a typed Responses request.
    pub fn encode_generate(
        &self,
        request: &GenerateRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<ResponsesRequest>, UnllmError> {
        encode_responses_request(request, context)?.try_map("invalid_openai_responses_request")
    }

    /// Decodes a typed Responses response.
    pub fn decode_generate_response(
        &self,
        response: ResponsesResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateResponse>, UnllmError> {
        decode_responses_response(serde_json::to_value(response).map_err(json_error)?, context)?
            .map_response()
    }
}

trait TypedOutcomeExt<T> {
    fn try_map<U: serde::de::DeserializeOwned>(
        self,
        code: &str,
    ) -> Result<ConversionOutcome<U>, UnllmError>;
}

impl TypedOutcomeExt<Value> for ConversionOutcome<Value> {
    fn try_map<U: serde::de::DeserializeOwned>(
        self,
        code: &str,
    ) -> Result<ConversionOutcome<U>, UnllmError> {
        Ok(ConversionOutcome {
            value: from_value(self.value, code)?,
            diagnostics: self.diagnostics,
        })
    }
}

trait UnifiedRequestOutcomeExt {
    fn map_result(self) -> Result<ConversionOutcome<GenerateRequest>, UnllmError>;
}

impl UnifiedRequestOutcomeExt for ConversionOutcome<UnifiedRequest> {
    fn map_result(self) -> Result<ConversionOutcome<GenerateRequest>, UnllmError> {
        let UnifiedRequest::Generate(value) = self.value else {
            return Err(UnllmError::invalid(
                "operation_mismatch",
                "Expected a generation request",
            ));
        };
        Ok(ConversionOutcome {
            value,
            diagnostics: self.diagnostics,
        })
    }
}

trait UnifiedResponseOutcomeExt {
    fn map_response(self) -> Result<ConversionOutcome<GenerateResponse>, UnllmError>;
}

impl UnifiedResponseOutcomeExt for ConversionOutcome<UnifiedResponse> {
    fn map_response(self) -> Result<ConversionOutcome<GenerateResponse>, UnllmError> {
        let UnifiedResponse::Generate(value) = self.value else {
            return Err(UnllmError::invalid(
                "operation_mismatch",
                "Expected a generation response",
            ));
        };
        Ok(ConversionOutcome {
            value,
            diagnostics: self.diagnostics,
        })
    }
}

fn capabilities() -> Capabilities {
    Capabilities {
        operations: BTreeSet::from([Operation::Generate, Operation::Embed, Operation::Image]),
        features: BTreeSet::from([
            Capability::Streaming,
            Capability::Text,
            Capability::Image,
            Capability::Audio,
            Capability::File,
            Capability::FunctionTools,
            Capability::StructuredOutput,
            Capability::Reasoning,
            Capability::MultipleCandidates,
            Capability::Embeddings,
            Capability::ImageEdit,
            Capability::ImageVariation,
            Capability::FileUpload,
        ]),
    }
}

impl DynAdapter for OpenAiChatAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::OpenAiChatCompletions
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
        decode_request(operation, body, context, false)
    }

    fn encode_request(
        &self,
        request: &UnifiedRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        encode_request(request, context, false)
    }

    fn decode_response(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
        decode_response(operation, body, context, false)
    }

    fn encode_response(
        &self,
        response: &UnifiedResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        encode_response(response, context, false)
    }

    fn decode_stream_event(
        &self,
        event_type: Option<&str>,
        data: &[u8],
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
        decode_chat_stream(event_type, data, context)
    }

    fn encode_stream_event(
        &self,
        event: &StreamEvent,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
        encode_chat_stream(event, context)
    }
}

impl DynAdapter for OpenAiResponsesAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::OpenAiResponses
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
        decode_request(operation, body, context, true)
    }

    fn encode_request(
        &self,
        request: &UnifiedRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        encode_request(request, context, true)
    }

    fn decode_response(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
        decode_response(operation, body, context, true)
    }

    fn encode_response(
        &self,
        response: &UnifiedResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        encode_response(response, context, true)
    }

    fn decode_stream_event(
        &self,
        event_type: Option<&str>,
        data: &[u8],
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
        decode_responses_stream(event_type, data, context)
    }

    fn encode_stream_event(
        &self,
        event: &StreamEvent,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
        encode_responses_stream(event, context)
    }
}

fn decode_request(
    operation: Operation,
    body: Value,
    context: &ConversionContext,
    responses: bool,
) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    match operation {
        Operation::Generate if responses => decode_responses_request(body, context),
        Operation::Generate => decode_chat_request(body, context),
        Operation::Embed => decode_embedding_request(body),
        Operation::Image => decode_image_request(body),
    }
}

fn encode_request(
    request: &UnifiedRequest,
    context: &ConversionContext,
    responses: bool,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let mut outcome = match request {
        UnifiedRequest::Generate(request) if responses => {
            encode_responses_request(request, context)
        }
        UnifiedRequest::Generate(request) => encode_chat_request(request, context),
        UnifiedRequest::Embed(request) => encode_embedding_request(request),
        UnifiedRequest::Image(request) => encode_image_request(request, context),
    }?;
    let extensions = match request {
        UnifiedRequest::Generate(value) => &value.extensions,
        UnifiedRequest::Embed(value) => &value.extensions,
        UnifiedRequest::Image(value) => &value.extensions,
    };
    outcome
        .diagnostics
        .extend(context.check_extensions(extensions, "openai", "/request")?);
    Ok(outcome)
}

fn decode_response(
    operation: Operation,
    body: Value,
    context: &ConversionContext,
    responses: bool,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    match operation {
        Operation::Generate if responses => decode_responses_response(body, context),
        Operation::Generate => decode_chat_response(body, context),
        Operation::Embed => decode_embedding_response(body),
        Operation::Image => decode_image_response(body),
    }
}

fn encode_response(
    response: &UnifiedResponse,
    context: &ConversionContext,
    responses: bool,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let mut outcome = match response {
        UnifiedResponse::Generate(response) if responses => {
            encode_responses_response(response, context)
        }
        UnifiedResponse::Generate(response) => encode_chat_response(response, context),
        UnifiedResponse::Embed(response) => encode_embedding_response(response),
        UnifiedResponse::Image(response) => encode_image_response(response),
    }?;
    let extensions = match response {
        UnifiedResponse::Generate(value) => &value.extensions,
        UnifiedResponse::Embed(value) => &value.extensions,
        UnifiedResponse::Image(value) => &value.extensions,
    };
    outcome
        .diagnostics
        .extend(context.check_extensions(extensions, "openai", "/response")?);
    Ok(outcome)
}

fn decode_chat_request(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let request: ChatCompletionRequest = from_value(body, "invalid_openai_chat_request")?;
    let mut instructions = Vec::new();
    let mut messages = Vec::new();
    for (message_index, message) in request.messages.into_iter().enumerate() {
        let role = required_str(&message, "role")?;
        let content = parse_openai_content(message.get("content"), message_index)?;
        match role {
            "system" | "developer" => instructions.push(Instruction {
                kind: if role == "developer" {
                    InstructionKind::Developer
                } else {
                    InstructionKind::System
                },
                priority: i32::try_from(message_index).unwrap_or(i32::MAX),
                content,
                extensions: Extensions::new(),
            }),
            "user" | "assistant" | "tool" => {
                let mut content = content;
                if role == "assistant" {
                    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                        for (index, call) in calls.iter().enumerate() {
                            content.push(ContentPart::ToolCall {
                                call: parse_openai_tool_call(call, index),
                            });
                        }
                    }
                } else if role == "tool" {
                    content = vec![ContentPart::ToolResult {
                        result: ToolResult {
                            call_id: message
                                .get("tool_call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            content,
                            is_error: false,
                            extensions: Extensions::new(),
                        },
                    }];
                }
                messages.push(Message {
                    role: match role {
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::User,
                    },
                    content,
                    id: None,
                    extensions: Extensions::new(),
                });
            }
            _ => {
                return Err(UnllmError::invalid(
                    "unsupported_openai_role",
                    format!("Unsupported OpenAI message role: {role}"),
                ));
            }
        }
    }
    let tools = request.tools.iter().filter_map(parse_openai_tool).collect();
    let response_format = request
        .response_format
        .as_ref()
        .and_then(parse_response_format);
    let parameters = parameters_from_extra(&request.extra);
    let extensions = namespace_extra("openai", request.extra, &parameter_keys());
    Ok(ConversionOutcome::clean(UnifiedRequest::Generate(
        GenerateRequest {
            model: request.model,
            instructions,
            messages,
            tools,
            tool_choice: request.tool_choice.as_ref().and_then(parse_tool_choice),
            response_format,
            reasoning: None,
            parameters,
            stream: request.stream,
            extensions,
        },
    )))
}

fn encode_chat_request(
    request: &GenerateRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let mut messages = Vec::new();
    let mut instructions = request.instructions.clone();
    instructions.sort_by_key(|instruction| instruction.priority);
    for instruction in instructions {
        messages.push(json!({
            "role": match instruction.kind {
                InstructionKind::System => "system",
                InstructionKind::Developer => "developer",
            },
            "content": encode_openai_content(&instruction.content, context)?,
        }));
    }
    for message in &request.messages {
        let mut object = Map::new();
        object.insert(
            "role".into(),
            Value::String(
                match message.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                }
                .into(),
            ),
        );
        if message.role == Role::Tool {
            if let Some(ContentPart::ToolResult { result }) = message.content.first() {
                object.insert("tool_call_id".into(), Value::String(result.call_id.clone()));
                object.insert(
                    "content".into(),
                    encode_openai_content(&result.content, context)?,
                );
            }
        } else {
            let normal: Vec<_> = message
                .content
                .iter()
                .filter(|part| !matches!(part, ContentPart::ToolCall { .. }))
                .cloned()
                .collect();
            object.insert("content".into(), encode_openai_content(&normal, context)?);
            let calls: Vec<_> = message
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolCall { call } => Some(json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments.to_string()}
                    })),
                    _ => None,
                })
                .collect();
            if !calls.is_empty() {
                object.insert("tool_calls".into(), Value::Array(calls));
            }
        }
        messages.push(Value::Object(object));
    }
    let mut value = json!({
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
    });
    let object = value.as_object_mut().expect("JSON object");
    insert_generation_parameters(object, &request.parameters);
    if !request.tools.is_empty() {
        object.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(encode_openai_tool).collect()),
        );
    }
    if let Some(choice) = &request.tool_choice {
        object.insert("tool_choice".into(), encode_tool_choice(choice));
    }
    if let Some(format) = &request.response_format {
        object.insert("response_format".into(), encode_response_format(format));
    }
    merge_provider_extensions(object, &request.extensions, "openai");
    Ok(ConversionOutcome::clean(value))
}

fn decode_responses_request(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let request: ResponsesRequest = from_value(body, "invalid_openai_responses_request")?;
    let instructions = request
        .instructions
        .map(|value| Instruction {
            kind: InstructionKind::Developer,
            priority: 0,
            content: parse_openai_content(Some(&value), 0)
                .unwrap_or_else(|_| vec![ContentPart::text(value.to_string())]),
            extensions: Extensions::new(),
        })
        .into_iter()
        .collect();
    let input_items = match request.input {
        Value::String(text) => vec![json!({"role": "user", "content": text})],
        Value::Array(items) => items,
        value => vec![value],
    };
    let mut messages = Vec::new();
    for (index, item) in input_items.iter().enumerate() {
        if item.get("type").and_then(Value::as_str) == Some("function_call_output") {
            messages.push(Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult {
                    result: ToolResult {
                        call_id: item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .into(),
                        content: vec![ContentPart::text(
                            item.get("output")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        )],
                        is_error: false,
                        extensions: Extensions::new(),
                    },
                }],
                id: None,
                extensions: Extensions::new(),
            });
            continue;
        }
        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
        messages.push(Message {
            role: if role == "assistant" {
                Role::Assistant
            } else {
                Role::User
            },
            content: parse_openai_content(item.get("content"), index)?,
            id: item.get("id").and_then(Value::as_str).map(str::to_owned),
            extensions: Extensions::new(),
        });
    }
    let tools = request.tools.iter().filter_map(parse_openai_tool).collect();
    let parameters = parameters_from_extra(&request.extra);
    let reasoning = request
        .extra
        .get("reasoning")
        .map(|value| unllm_core::ReasoningConfig {
            effort: value
                .get("effort")
                .and_then(Value::as_str)
                .map(str::to_owned),
            budget_tokens: None,
            include_summary: value.get("summary").map(|summary| !summary.is_null()),
        });
    let extensions = namespace_extra("openai", request.extra, &parameter_keys());
    Ok(ConversionOutcome::clean(UnifiedRequest::Generate(
        GenerateRequest {
            model: request.model,
            instructions,
            messages,
            tools,
            tool_choice: None,
            response_format: None,
            reasoning,
            parameters,
            stream: request.stream,
            extensions,
        },
    )))
}

fn encode_responses_request(
    request: &GenerateRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let instructions = request
        .instructions
        .iter()
        .flat_map(|instruction| instruction.content.iter())
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut input = Vec::new();
    for message in &request.messages {
        for part in &message.content {
            if let ContentPart::ToolResult { result } = part {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": content_as_text(&result.content),
                }));
            }
        }
        let normal: Vec<_> = message
            .content
            .iter()
            .filter(|part| !matches!(part, ContentPart::ToolResult { .. }))
            .cloned()
            .collect();
        if !normal.is_empty() {
            input.push(json!({
                "role": match message.role { Role::Assistant => "assistant", _ => "user" },
                "content": encode_responses_content(&normal, context)?,
            }));
        }
    }
    let mut value = json!({"model": request.model, "input": input, "stream": request.stream});
    let object = value.as_object_mut().expect("JSON object");
    if !instructions.is_empty() {
        object.insert("instructions".into(), Value::String(instructions));
    }
    if !request.tools.is_empty() {
        object.insert(
            "tools".into(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        })
                    })
                    .collect(),
            ),
        );
    }
    insert_generation_parameters(object, &request.parameters);
    if let Some(reasoning) = &request.reasoning {
        object.insert("reasoning".into(), json!({
            "effort": reasoning.effort,
            "summary": reasoning.include_summary.map(|enabled| if enabled { "auto" } else { "none" }),
        }));
    }
    merge_provider_extensions(object, &request.extensions, "openai");
    Ok(ConversionOutcome::clean(value))
}

fn decode_chat_response(
    body: Value,
    context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let response: ChatCompletionResponse = from_value(body, "invalid_openai_chat_response")?;
    let mut candidates = Vec::new();
    for (fallback_index, choice) in response.choices.iter().enumerate() {
        let index = choice
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or_else(|| u32::try_from(fallback_index).unwrap_or(u32::MAX));
        let message = choice.get("message").cloned().unwrap_or(Value::Null);
        let mut content = parse_openai_content(message.get("content"), fallback_index)?;
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for (call_index, call) in calls.iter().enumerate() {
                content.push(ContentPart::ToolCall {
                    call: parse_openai_tool_call(call, call_index),
                });
            }
        }
        if let Some(refusal) = message.get("refusal").and_then(Value::as_str) {
            content.push(ContentPart::Refusal {
                reason: refusal.into(),
            });
        }
        candidates.push(Candidate {
            index,
            id: None,
            content,
            finish_reason: choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .map(parse_finish_reason),
            extensions: Extensions::new(),
        });
    }
    let usage = response.usage.as_ref().map(parse_openai_usage);
    let extensions = namespace_extra(
        "openai",
        response.extra,
        &BTreeSet::from(["object".into(), "created".into()]),
    );
    let result = GenerateResponse {
        id: response.id,
        model: response.model,
        candidates,
        usage,
        extensions,
    };
    let _ = context;
    Ok(ConversionOutcome::clean(UnifiedResponse::Generate(result)))
}

fn encode_chat_response(
    response: &GenerateResponse,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let choices = response
        .candidates
        .iter()
        .map(|candidate| {
            let calls: Vec<_> = candidate
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolCall { call } => Some(json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments.to_string()},
                    })),
                    _ => None,
                })
                .collect();
            let normal: Vec<_> = candidate
                .content
                .iter()
                .filter(|part| !matches!(part, ContentPart::ToolCall { .. }))
                .cloned()
                .collect();
            Ok(json!({
                "index": candidate.index,
                "message": {
                    "role": "assistant",
                    "content": encode_openai_content(&normal, context)?,
                    "tool_calls": if calls.is_empty() { Value::Null } else { Value::Array(calls) },
                },
                "finish_reason": candidate.finish_reason.as_ref().map(encode_finish_reason),
            }))
        })
        .collect::<Result<Vec<Value>, UnllmError>>()?;
    Ok(ConversionOutcome::clean(json!({
        "id": response.id,
        "object": "chat.completion",
        "model": response.model,
        "choices": choices,
        "usage": response.usage.as_ref().map(encode_openai_usage),
    })))
}

fn decode_responses_response(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let response: ResponsesResponse = from_value(body, "invalid_openai_responses_response")?;
    let mut content = Vec::new();
    for (index, item) in response.output.iter().enumerate() {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => content.extend(parse_openai_content(item.get("content"), index)?),
            Some("function_call") => content.push(ContentPart::ToolCall {
                call: ToolCall {
                    id: item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("call-{index}")),
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    arguments: parse_json_string(item.get("arguments")),
                    extensions: Extensions::new(),
                },
            }),
            Some("reasoning") => content.push(ContentPart::Reasoning {
                text: item.get("summary").and_then(Value::as_array).map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
                extensions: Extensions::new(),
            }),
            _ => {}
        }
    }
    let extensions = namespace_extra(
        "openai",
        response.extra,
        &BTreeSet::from(["object".into(), "status".into(), "created_at".into()]),
    );
    let finish_reason = if content
        .iter()
        .any(|part| matches!(part, ContentPart::ToolCall { .. }))
    {
        FinishReason::ToolCall
    } else {
        FinishReason::Stop
    };
    Ok(ConversionOutcome::clean(UnifiedResponse::Generate(
        GenerateResponse {
            id: response.id,
            model: response.model,
            candidates: vec![Candidate {
                index: 0,
                id: None,
                content,
                finish_reason: Some(finish_reason),
                extensions: Extensions::new(),
            }],
            usage: response.usage.as_ref().map(parse_openai_usage),
            extensions,
        },
    )))
}

fn encode_responses_response(
    response: &GenerateResponse,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let mut output = Vec::new();
    for candidate in &response.candidates {
        let mut message_content = Vec::new();
        for part in &candidate.content {
            match part {
                ContentPart::ToolCall { call } => output.push(json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                })),
                ContentPart::Reasoning { text, .. } => output.push(json!({
                    "type": "reasoning",
                    "summary": text.as_ref().map(|text| vec![json!({"type": "summary_text", "text": text})]).unwrap_or_default(),
                })),
                _ => message_content.extend(encode_responses_output_content(part, context)?),
            }
        }
        if !message_content.is_empty() {
            output
                .push(json!({"type": "message", "role": "assistant", "content": message_content}));
        }
    }
    Ok(ConversionOutcome::clean(json!({
        "id": response.id,
        "object": "response",
        "model": response.model,
        "status": "completed",
        "output": output,
        "usage": response.usage.as_ref().map(encode_openai_usage),
    })))
}

fn encode_responses_output_content(
    part: &ContentPart,
    context: &ConversionContext,
) -> Result<Vec<Value>, UnllmError> {
    match part {
        ContentPart::Text { text, annotations } => Ok(vec![json!({
            "type": "output_text",
            "text": text,
            "annotations": annotations,
        })]),
        ContentPart::Refusal { reason } => Ok(vec![json!({"type": "refusal", "refusal": reason})]),
        ContentPart::Image { asset } => Ok(vec![json!({
            "type": "output_image",
            "image_url": encode_media_ref(asset, "openai")?,
        })]),
        _ => {
            context.lossy(
                "responses_output_content",
                "/response/candidates/content",
                "The content part cannot be represented as OpenAI Responses message output",
            )?;
            Ok(Vec::new())
        }
    }
}

fn decode_embedding_request(body: Value) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let request: EmbeddingsRequest = from_value(body, "invalid_openai_embedding_request")?;
    let values = match request.input {
        Value::Array(values) => values,
        value => vec![value],
    };
    let inputs = values
        .into_iter()
        .map(|value| EmbeddingInput::Text {
            text: value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string()),
        })
        .collect();
    Ok(ConversionOutcome::clean(UnifiedRequest::Embed(
        EmbedRequest {
            model: request.model,
            inputs,
            dimensions: request.dimensions,
            task_type: None,
            extensions: namespace_extra("openai", request.extra, &BTreeSet::new()),
        },
    )))
}

fn encode_embedding_request(
    request: &EmbedRequest,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let input = request
        .inputs
        .iter()
        .map(|input| match input {
            EmbeddingInput::Text { text } => Ok(Value::String(text.clone())),
            EmbeddingInput::Content { content } => Ok(Value::String(content_as_text(content))),
        })
        .collect::<Result<Vec<_>, UnllmError>>()?;
    let mut value = serde_json::to_value(EmbeddingsRequest {
        model: request.model.clone(),
        input: Value::Array(input),
        dimensions: request.dimensions,
        extra: BTreeMap::new(),
    })
    .map_err(json_error)?;
    merge_provider_extensions(
        value.as_object_mut().expect("object"),
        &request.extensions,
        "openai",
    );
    Ok(ConversionOutcome::clean(value))
}

fn decode_embedding_response(
    body: Value,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let embeddings = body
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(fallback_index, item)| Embedding {
            index: item
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| u32::try_from(index).ok())
                .unwrap_or_else(|| u32::try_from(fallback_index).unwrap_or(u32::MAX)),
            values: item
                .get("embedding")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_f64)
                .map(|value| value as f32)
                .collect(),
            extensions: Extensions::new(),
        })
        .collect();
    Ok(ConversionOutcome::clean(UnifiedResponse::Embed(
        EmbedResponse {
            model,
            embeddings,
            usage: body.get("usage").map(parse_openai_usage),
            extensions: Extensions::new(),
        },
    )))
}

fn encode_embedding_response(
    response: &EmbedResponse,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    Ok(ConversionOutcome::clean(json!({
        "object": "list",
        "model": response.model,
        "data": response.embeddings.iter().map(|embedding| json!({
            "object": "embedding", "index": embedding.index, "embedding": embedding.values,
        })).collect::<Vec<_>>(),
        "usage": response.usage.as_ref().map(encode_openai_usage),
    })))
}

fn decode_image_request(body: Value) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let request: ImagesRequest = from_value(body, "invalid_openai_image_request")?;
    let task = match request
        .extra
        .get("_unllm_image_task")
        .and_then(Value::as_str)
    {
        Some("edit") => ImageTask::Edit {
            image: parse_multipart_asset(request.extra.get("_unllm_image"))?,
            mask: request
                .extra
                .get("_unllm_mask")
                .map(|value| parse_multipart_asset(Some(value)))
                .transpose()?,
        },
        Some("variation") => ImageTask::Variation {
            image: parse_multipart_asset(request.extra.get("_unllm_image"))?,
        },
        _ => ImageTask::Generate,
    };
    let mut extra = request.extra;
    let negative_prompt = extra
        .get("negative_prompt")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let quality = extra
        .get("quality")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let style = extra
        .get("style")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let output_format = extra
        .get("output_format")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let stream = extra
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    for key in [
        "_unllm_image_task",
        "_unllm_image",
        "_unllm_mask",
        "negative_prompt",
        "quality",
        "style",
        "output_format",
        "stream",
    ] {
        extra.remove(key);
    }
    Ok(ConversionOutcome::clean(UnifiedRequest::Image(
        ImageRequest {
            model: request.model,
            task,
            prompt: request.prompt,
            negative_prompt,
            count: request.n,
            size: request.size,
            quality,
            style,
            output_format,
            stream,
            extensions: namespace_extra("openai", extra, &BTreeSet::new()),
        },
    )))
}

fn encode_image_request(
    request: &ImageRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let mut value = serde_json::to_value(ImagesRequest {
        model: request.model.clone(),
        prompt: request.prompt.clone(),
        n: request.count,
        size: request.size.clone(),
        extra: BTreeMap::new(),
    })
    .map_err(json_error)?;
    let object = value.as_object_mut().expect("object");
    for (key, value) in [
        ("quality", request.quality.as_ref()),
        ("style", request.style.as_ref()),
        ("output_format", request.output_format.as_ref()),
    ] {
        if let Some(value) = value {
            object.insert(key.into(), Value::String(value.clone()));
        }
    }
    if request.stream {
        object.insert("stream".into(), Value::Bool(true));
    }
    match &request.task {
        ImageTask::Generate => {}
        ImageTask::Edit { image, mask } => {
            object.insert(
                "_unllm_multipart".into(),
                json!({
                    "task": "edit",
                    "image": encode_multipart_asset(image)?,
                    "mask": mask.as_ref().map(encode_multipart_asset).transpose()?,
                }),
            );
        }
        ImageTask::Variation { image } => {
            object.insert(
                "_unllm_multipart".into(),
                json!({"task": "variation", "image": encode_multipart_asset(image)?}),
            );
        }
    }
    merge_provider_extensions(object, &request.extensions, "openai");
    let _ = context;
    Ok(ConversionOutcome::clean(value))
}

fn parse_multipart_asset(value: Option<&Value>) -> Result<MediaAsset, UnllmError> {
    let value = value.ok_or_else(|| {
        UnllmError::invalid("openai_image_file", "The multipart image field is required")
    })?;
    let mime_type = value
        .get("mime_type")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    let data = value
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| UnllmError::invalid("openai_image_file", "Image data is missing"))?;
    let data = STANDARD
        .decode(data)
        .map_err(|error| UnllmError::invalid("openai_image_file", error.to_string()))?;
    Ok(MediaAsset::inline(mime_type, Bytes::from(data)))
}

fn encode_multipart_asset(asset: &MediaAsset) -> Result<Value, UnllmError> {
    match &asset.source {
        MediaSource::Inline { mime_type, data } => {
            Ok(json!({"mime_type": mime_type, "data": STANDARD.encode(data)}))
        }
        _ => Err(UnllmError::unsupported(
            "openai_image_file",
            "OpenAI image edits and variations require inline image bytes",
        )),
    }
}

fn decode_image_response(body: Value) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let images = body
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            if let Some(url) = item.get("url").and_then(Value::as_str) {
                Some(MediaAsset::url(url, Some("image/png".into())))
            } else {
                item.get("b64_json")
                    .and_then(Value::as_str)
                    .and_then(|data| STANDARD.decode(data).ok())
                    .map(|data| MediaAsset::inline("image/png", Bytes::from(data)))
            }
        })
        .collect();
    Ok(ConversionOutcome::clean(UnifiedResponse::Image(
        ImageResponse {
            model: body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            images,
            usage: body.get("usage").map(parse_openai_usage),
            extensions: Extensions::new(),
        },
    )))
}

fn encode_image_response(response: &ImageResponse) -> Result<ConversionOutcome<Value>, UnllmError> {
    let data = response
        .images
        .iter()
        .map(|image| match &image.source {
            MediaSource::Url { url, .. } => json!({"url": url}),
            MediaSource::Inline { data, .. } => json!({"b64_json": STANDARD.encode(data)}),
            MediaSource::ProviderFile { id, .. } => json!({"file_id": id}),
        })
        .collect::<Vec<_>>();
    Ok(ConversionOutcome::clean(
        json!({"data": data, "model": response.model}),
    ))
}

fn decode_chat_stream(
    event_type: Option<&str>,
    data: &[u8],
    _context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
    if data == b"[DONE]" {
        return Ok(ConversionOutcome::clean(Vec::new()));
    }
    let value: Value = serde_json::from_slice(data).map_err(json_error)?;
    if value.get("error").is_some() {
        return Ok(ConversionOutcome::clean(vec![StreamEvent::Error {
            error: UnllmError {
                code: value
                    .pointer("/error/code")
                    .and_then(Value::as_str)
                    .unwrap_or("upstream_error")
                    .into(),
                kind: unllm_core::ErrorKind::Upstream,
                message: value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("OpenAI stream failed")
                    .into(),
                source_protocol: Some(Protocol::OpenAiChatCompletions),
                target: None,
                operation: Some(Operation::Generate),
                metadata: None,
            },
        }]));
    }
    let mut events = Vec::new();
    let id = value.get("id").and_then(Value::as_str).map(str::to_owned);
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    for choice in value
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let index = choice
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0);
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if delta.get("role").is_some() {
            events.push(StreamEvent::ResponseStart {
                id: id.clone(),
                model: model.clone(),
                extensions: Extensions::new(),
            });
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            events.push(StreamEvent::ContentDelta {
                candidate_index: index,
                block_index: 0,
                delta: ContentDelta::Text { text: text.into() },
            });
        }
        for tool in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let block_index = tool
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0)
                + 1;
            events.push(StreamEvent::ContentDelta {
                candidate_index: index,
                block_index,
                delta: ContentDelta::ToolArguments {
                    call_id: tool
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("call-{index}-{block_index}")),
                    name: tool
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    fragment: tool
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                },
            });
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            events.push(StreamEvent::Finish {
                candidate_index: index,
                reason: Some(parse_finish_reason(reason)),
            });
        }
    }
    if let Some(usage) = value.get("usage").filter(|value| !value.is_null()) {
        events.push(StreamEvent::Usage {
            usage: parse_openai_usage(usage),
        });
    }
    if events.is_empty() {
        events.push(StreamEvent::Raw {
            provider: "openai".into(),
            event_type: event_type.map(str::to_owned),
            payload: value,
        });
    }
    Ok(ConversionOutcome::clean(events))
}

fn encode_chat_stream(
    event: &StreamEvent,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
    let value = match event {
        StreamEvent::ResponseStart { id, model, .. } => json!({
            "id": id, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }),
        StreamEvent::ContentDelta {
            candidate_index,
            block_index,
            delta,
        } => match delta {
            ContentDelta::Text { text } => {
                json!({"object": "chat.completion.chunk", "choices": [{"index": candidate_index, "delta": {"content": text}, "finish_reason": null}]})
            }
            ContentDelta::Reasoning { text } => {
                json!({"object": "chat.completion.chunk", "choices": [{"index": candidate_index, "delta": {"reasoning_content": text}, "finish_reason": null}]})
            }
            ContentDelta::ToolArguments {
                call_id,
                name,
                fragment,
            } => {
                json!({"object": "chat.completion.chunk", "choices": [{"index": candidate_index, "delta": {"tool_calls": [{"index": block_index, "id": call_id, "type": "function", "function": {"name": name, "arguments": fragment}}]}, "finish_reason": null}]})
            }
            ContentDelta::Media { .. } => {
                return Err(UnllmError::unsupported(
                    "openai_chat_media_delta",
                    "OpenAI Chat Completions cannot encode this media delta",
                ));
            }
        },
        StreamEvent::Usage { usage } => {
            json!({"object": "chat.completion.chunk", "choices": [], "usage": encode_openai_usage(usage)})
        }
        StreamEvent::Finish {
            candidate_index,
            reason,
        } => {
            json!({"object": "chat.completion.chunk", "choices": [{"index": candidate_index, "delta": {}, "finish_reason": reason.as_ref().map(encode_finish_reason)}]})
        }
        StreamEvent::Error { error } => {
            json!({"error": {"code": error.code, "message": error.message, "type": format!("{:?}", error.kind).to_lowercase()}})
        }
        StreamEvent::Raw {
            provider, payload, ..
        } if provider == "openai" => payload.clone(),
        StreamEvent::ContentBlockStart { .. } | StreamEvent::ContentBlockStop { .. } => {
            return Ok(ConversionOutcome::clean(Vec::new()));
        }
        StreamEvent::Raw { .. } => {
            let warning = context.lossy(
                "raw_stream_event",
                "/",
                "A provider-specific stream event cannot be represented by OpenAI Chat Completions",
            )?;
            return Ok(ConversionOutcome {
                value: Vec::new(),
                diagnostics: vec![warning],
            });
        }
    };
    Ok(ConversionOutcome::clean(vec![sse(None, &value)]))
}

fn decode_responses_stream(
    event_type: Option<&str>,
    data: &[u8],
    _context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
    let value: Value = serde_json::from_slice(data).map_err(json_error)?;
    let kind = event_type
        .or_else(|| value.get("type").and_then(Value::as_str))
        .unwrap_or("unknown");
    let sequence = value
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    let event = match kind {
        "response.created" | "response.in_progress" => StreamEvent::ResponseStart {
            id: value
                .pointer("/response/id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            model: value
                .pointer("/response/model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            extensions: Extensions::new(),
        },
        "response.output_text.delta" => StreamEvent::ContentDelta {
            candidate_index: 0,
            block_index: sequence,
            delta: ContentDelta::Text {
                text: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
            },
        },
        "response.reasoning_summary_text.delta" => StreamEvent::ContentDelta {
            candidate_index: 0,
            block_index: sequence,
            delta: ContentDelta::Reasoning {
                text: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
            },
        },
        "response.function_call_arguments.delta" => StreamEvent::ContentDelta {
            candidate_index: 0,
            block_index: sequence,
            delta: ContentDelta::ToolArguments {
                call_id: value
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or("call-0")
                    .into(),
                name: value.get("name").and_then(Value::as_str).map(str::to_owned),
                fragment: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
            },
        },
        "response.completed" => StreamEvent::Finish {
            candidate_index: 0,
            reason: Some(FinishReason::Stop),
        },
        "error" | "response.failed" => StreamEvent::Error {
            error: UnllmError {
                code: "openai_stream_error".into(),
                kind: unllm_core::ErrorKind::Upstream,
                message: value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("OpenAI response stream failed")
                    .into(),
                source_protocol: Some(Protocol::OpenAiResponses),
                target: None,
                operation: Some(Operation::Generate),
                metadata: None,
            },
        },
        _ => StreamEvent::Raw {
            provider: "openai".into(),
            event_type: Some(kind.into()),
            payload: value,
        },
    };
    Ok(ConversionOutcome::clean(vec![event]))
}

fn encode_responses_stream(
    event: &StreamEvent,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
    let (kind, value) = match event {
        StreamEvent::ResponseStart { id, model, .. } => (
            "response.created",
            json!({"type": "response.created", "response": {"id": id, "model": model, "status": "in_progress"}}),
        ),
        StreamEvent::ContentDelta {
            candidate_index,
            block_index,
            delta: ContentDelta::Text { text },
        } => (
            "response.output_text.delta",
            json!({"type": "response.output_text.delta", "output_index": candidate_index, "content_index": block_index, "delta": text}),
        ),
        StreamEvent::ContentDelta {
            candidate_index,
            block_index,
            delta: ContentDelta::Reasoning { text },
        } => (
            "response.reasoning_summary_text.delta",
            json!({"type": "response.reasoning_summary_text.delta", "output_index": candidate_index, "summary_index": block_index, "delta": text}),
        ),
        StreamEvent::ContentDelta {
            candidate_index,
            block_index,
            delta:
                ContentDelta::ToolArguments {
                    call_id,
                    name,
                    fragment,
                },
        } => (
            "response.function_call_arguments.delta",
            json!({"type": "response.function_call_arguments.delta", "output_index": candidate_index, "content_index": block_index, "item_id": call_id, "name": name, "delta": fragment}),
        ),
        StreamEvent::Finish { .. } => (
            "response.completed",
            json!({"type": "response.completed", "response": {"status": "completed"}}),
        ),
        StreamEvent::Error { error } => (
            "error",
            json!({"type": "error", "error": {"code": error.code, "message": error.message}}),
        ),
        StreamEvent::Raw {
            provider,
            event_type,
            payload,
        } if provider == "openai" => (event_type.as_deref().unwrap_or("unknown"), payload.clone()),
        StreamEvent::Usage { .. }
        | StreamEvent::ContentBlockStart { .. }
        | StreamEvent::ContentBlockStop { .. } => return Ok(ConversionOutcome::clean(Vec::new())),
        _ => {
            let warning = context.lossy(
                "responses_stream_event",
                "/",
                "The stream event cannot be represented by OpenAI Responses",
            )?;
            return Ok(ConversionOutcome {
                value: Vec::new(),
                diagnostics: vec![warning],
            });
        }
    };
    Ok(ConversionOutcome::clean(vec![sse(Some(kind), &value)]))
}

fn parse_openai_content(
    value: Option<&Value>,
    index: usize,
) -> Result<Vec<ContentPart>, UnllmError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if let Some(text) = value.as_str() {
        return Ok(vec![ContentPart::text(text)]);
    }
    let Some(items) = value.as_array() else {
        return Ok(vec![ContentPart::text(value.to_string())]);
    };
    let mut content = Vec::new();
    for (part_index, item) in items.iter().enumerate() {
        match item.get("type").and_then(Value::as_str).unwrap_or("text") {
            "text" | "input_text" | "output_text" => content.push(ContentPart::Text {
                text: item
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
                annotations: item
                    .get("annotations")
                    .and_then(Value::as_array)
                    .map(|values| values.iter().filter_map(parse_openai_annotation).collect())
                    .unwrap_or_default(),
            }),
            "image_url" | "input_image" => {
                let url = item
                    .pointer("/image_url/url")
                    .or_else(|| item.get("image_url"))
                    .and_then(Value::as_str)
                    .or_else(|| item.get("file_id").and_then(Value::as_str));
                if let Some(url) = url {
                    content.push(ContentPart::Image {
                        asset: media_from_openai_ref(url, "image", index, part_index)?,
                    });
                }
            }
            "input_audio" => {
                let data = item
                    .pointer("/input_audio/data")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let format = item
                    .pointer("/input_audio/format")
                    .and_then(Value::as_str)
                    .unwrap_or("wav");
                let bytes = STANDARD.decode(data).map_err(|error| {
                    UnllmError::invalid("invalid_audio_base64", error.to_string())
                })?;
                content.push(ContentPart::Audio {
                    asset: MediaAsset::inline(format!("audio/{format}"), Bytes::from(bytes)),
                });
            }
            "input_file" => {
                if let Some(file_id) = item.get("file_id").and_then(Value::as_str) {
                    content.push(ContentPart::File {
                        asset: MediaAsset {
                            source: MediaSource::ProviderFile {
                                provider: "openai".into(),
                                id: file_id.into(),
                                mime_type: None,
                            },
                            width: None,
                            height: None,
                            sha256: None,
                            extensions: Extensions::new(),
                        },
                        filename: item
                            .get("filename")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                    });
                }
            }
            "refusal" => content.push(ContentPart::Refusal {
                reason: item
                    .get("refusal")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
            }),
            _ => {}
        }
    }
    Ok(content)
}

fn encode_openai_content(
    parts: &[ContentPart],
    context: &ConversionContext,
) -> Result<Value, UnllmError> {
    if parts.len() == 1 {
        if let ContentPart::Text { text, annotations } = &parts[0] {
            if annotations.is_empty() {
                return Ok(Value::String(text.clone()));
            }
        }
    }
    let mut encoded = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text, .. } => encoded.push(json!({"type": "text", "text": text})),
            ContentPart::Image { asset } => encoded.push(json!({"type": "image_url", "image_url": {"url": encode_media_ref(asset, "openai")?}})),
            ContentPart::Audio { asset } => match &asset.source {
                MediaSource::Inline { data, mime_type } => encoded.push(json!({"type": "input_audio", "input_audio": {"data": STANDARD.encode(data), "format": mime_type.split('/').next_back().unwrap_or("wav")}})),
                _ => return Err(UnllmError::unsupported("openai_audio_source", "OpenAI Chat audio input requires inline bytes")),
            },
            ContentPart::File { asset, filename } => match &asset.source {
                MediaSource::ProviderFile { provider, id, .. } if provider == "openai" => encoded.push(json!({"type": "input_file", "file_id": id, "filename": filename})),
                _ => return Err(UnllmError::unsupported("openai_file_source", "OpenAI file input requires an OpenAI file identifier")),
            },
            ContentPart::Reasoning { .. } => {
                context.lossy("reasoning_content", "/content", "OpenAI Chat Completions cannot encode a reasoning content block")?;
            }
            ContentPart::Refusal { reason } => encoded.push(json!({"type": "refusal", "refusal": reason})),
            ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } | ContentPart::Video { .. } => {
                return Err(UnllmError::unsupported("openai_content_part", "The content part cannot be encoded inside an OpenAI Chat content array"));
            }
        }
    }
    Ok(Value::Array(encoded))
}

fn encode_responses_content(
    parts: &[ContentPart],
    context: &ConversionContext,
) -> Result<Value, UnllmError> {
    let mut encoded = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text, .. } => encoded.push(json!({"type": "input_text", "text": text})),
            ContentPart::Image { asset } => match &asset.source {
                MediaSource::ProviderFile { provider, id, .. } if provider == "openai" => encoded.push(json!({"type": "input_image", "file_id": id})),
                _ => encoded.push(json!({"type": "input_image", "image_url": encode_media_ref(asset, "openai")?})),
            },
            ContentPart::File { asset, filename } => match &asset.source {
                MediaSource::ProviderFile { provider, id, .. } if provider == "openai" => encoded.push(json!({"type": "input_file", "file_id": id, "filename": filename})),
                MediaSource::Inline { data, mime_type } => encoded.push(json!({"type": "input_file", "file_data": format!("data:{mime_type};base64,{}", STANDARD.encode(data)), "filename": filename})),
                _ => return Err(UnllmError::unsupported("responses_file_source", "OpenAI Responses file input requires inline bytes or an OpenAI file identifier")),
            },
            ContentPart::Audio { .. } | ContentPart::Video { .. } => return Err(UnllmError::unsupported("responses_media", "This media content requires an OpenAI beta extension")),
            ContentPart::Reasoning { .. } => { context.lossy("reasoning_input", "/content", "Reasoning content is output-only in OpenAI Responses")?; }
            ContentPart::Refusal { reason } => encoded.push(json!({"type": "refusal", "refusal": reason})),
            ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } => {}
        }
    }
    Ok(Value::Array(encoded))
}

fn media_from_openai_ref(
    value: &str,
    media: &str,
    _message: usize,
    _part: usize,
) -> Result<MediaAsset, UnllmError> {
    if let Some(data) = value.strip_prefix("data:") {
        let (mime_and_encoding, payload) = data.split_once(',').ok_or_else(|| {
            UnllmError::invalid("invalid_data_url", "Data URL is missing a comma")
        })?;
        let mime_type = mime_and_encoding
            .strip_suffix(";base64")
            .unwrap_or(mime_and_encoding);
        let bytes = STANDARD
            .decode(payload)
            .map_err(|error| UnllmError::invalid("invalid_data_url", error.to_string()))?;
        Ok(MediaAsset::inline(mime_type, Bytes::from(bytes)))
    } else if value.starts_with("http://") || value.starts_with("https://") {
        Ok(MediaAsset::url(value, Some(format!("{media}/*"))))
    } else {
        Ok(MediaAsset {
            source: MediaSource::ProviderFile {
                provider: "openai".into(),
                id: value.into(),
                mime_type: None,
            },
            width: None,
            height: None,
            sha256: None,
            extensions: Extensions::new(),
        })
    }
}

fn encode_media_ref(asset: &MediaAsset, provider: &str) -> Result<String, UnllmError> {
    match &asset.source {
        MediaSource::Url { url, .. } => Ok(url.clone()),
        MediaSource::Inline { mime_type, data } => {
            Ok(format!("data:{mime_type};base64,{}", STANDARD.encode(data)))
        }
        MediaSource::ProviderFile {
            provider: owner,
            id,
            ..
        } if owner == provider => Ok(id.clone()),
        MediaSource::ProviderFile { .. } => Err(UnllmError::unsupported(
            "foreign_file_id",
            "A provider file identifier cannot be sent to another provider",
        )),
    }
}

fn parse_openai_tool(value: &Value) -> Option<ToolDefinition> {
    let function = if value.get("type").and_then(Value::as_str) == Some("function") {
        value.get("function").unwrap_or(value)
    } else {
        return None;
    };
    Some(ToolDefinition {
        name: function.get("name")?.as_str()?.to_owned(),
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        input_schema: function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
        extensions: Extensions::new(),
    })
}

fn encode_openai_tool(tool: &ToolDefinition) -> Value {
    json!({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}})
}

fn parse_openai_tool_call(value: &Value, index: usize) -> ToolCall {
    ToolCall {
        id: value
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("call-{index}")),
        name: value
            .pointer("/function/name")
            .or_else(|| value.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
        arguments: parse_json_string(
            value
                .pointer("/function/arguments")
                .or_else(|| value.get("arguments")),
        ),
        extensions: Extensions::new(),
    }
}

fn parse_json_string(value: Option<&Value>) -> Value {
    value
        .and_then(Value::as_str)
        .and_then(|text| serde_json::from_str(text).ok())
        .or_else(|| value.cloned())
        .unwrap_or_else(|| json!({}))
}

fn parse_tool_choice(value: &Value) -> Option<ToolChoice> {
    if let Some(choice) = value.as_str() {
        return match choice {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        };
    }
    value
        .pointer("/function/name")
        .and_then(Value::as_str)
        .map(|name| ToolChoice::Function { name: name.into() })
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".into()),
        ToolChoice::None => Value::String("none".into()),
        ToolChoice::Required => Value::String("required".into()),
        ToolChoice::Function { name } => json!({"type": "function", "function": {"name": name}}),
    }
}

fn parse_response_format(value: &Value) -> Option<ResponseFormat> {
    match value.get("type").and_then(Value::as_str) {
        Some("text") => Some(ResponseFormat::Text),
        Some("json_object") => Some(ResponseFormat::JsonObject),
        Some("json_schema") => Some(ResponseFormat::JsonSchema {
            name: value
                .pointer("/json_schema/name")
                .and_then(Value::as_str)
                .unwrap_or("response")
                .into(),
            schema: value
                .pointer("/json_schema/schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"})),
            strict: value
                .pointer("/json_schema/strict")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }),
        _ => None,
    }
}

fn encode_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({"type": "text"}),
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": strict}})
        }
    }
}

fn parameters_from_extra(extra: &BTreeMap<String, Value>) -> unllm_core::GenerationParameters {
    unllm_core::GenerationParameters {
        max_output_tokens: extra
            .get("max_completion_tokens")
            .or_else(|| extra.get("max_tokens"))
            .and_then(Value::as_u64),
        temperature: extra
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        top_p: extra.get("top_p").and_then(Value::as_f64).map(|v| v as f32),
        top_k: None,
        presence_penalty: extra
            .get("presence_penalty")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        frequency_penalty: extra
            .get("frequency_penalty")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        seed: extra.get("seed").and_then(Value::as_i64),
        logprobs: extra.get("logprobs").and_then(Value::as_bool),
        stop_sequences: extra
            .get("stop")
            .map(|value| match value {
                Value::String(text) => vec![text.clone()],
                Value::Array(items) => items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect(),
                _ => Vec::new(),
            })
            .unwrap_or_default(),
        candidate_count: extra
            .get("n")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok()),
    }
}

fn insert_generation_parameters(
    object: &mut Map<String, Value>,
    parameters: &unllm_core::GenerationParameters,
) {
    macro_rules! insert {
        ($name:literal, $value:expr) => {
            if let Some(value) = $value {
                object.insert($name.into(), json!(value));
            }
        };
    }
    insert!("max_completion_tokens", parameters.max_output_tokens);
    insert!("temperature", parameters.temperature);
    insert!("top_p", parameters.top_p);
    insert!("presence_penalty", parameters.presence_penalty);
    insert!("frequency_penalty", parameters.frequency_penalty);
    insert!("seed", parameters.seed);
    insert!("logprobs", parameters.logprobs);
    insert!("n", parameters.candidate_count);
    if !parameters.stop_sequences.is_empty() {
        object.insert("stop".into(), json!(parameters.stop_sequences));
    }
}

fn parameter_keys() -> BTreeSet<String> {
    [
        "max_completion_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "logprobs",
        "stop",
        "n",
        "reasoning",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn namespace_extra(
    namespace: &str,
    mut extra: BTreeMap<String, Value>,
    consumed: &BTreeSet<String>,
) -> Extensions {
    for key in consumed {
        extra.remove(key);
    }
    if extra.is_empty() {
        Extensions::new()
    } else {
        BTreeMap::from([(namespace.into(), Value::Object(extra.into_iter().collect()))])
    }
}

fn merge_provider_extensions(
    object: &mut Map<String, Value>,
    extensions: &Extensions,
    namespace: &str,
) {
    if let Some(Value::Object(extra)) = extensions.get(namespace) {
        for (key, value) in extra {
            object.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
}

fn parse_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCall,
        "content_filter" => FinishReason::ContentFilter,
        "error" => FinishReason::Error,
        other => FinishReason::Other(other.into()),
    }
}

fn encode_finish_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCall => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Error => "error",
        FinishReason::Other(raw) => raw,
    }
}

fn parse_openai_usage(value: &Value) -> Usage {
    Usage {
        input_tokens: value
            .get("prompt_tokens")
            .or_else(|| value.get("input_tokens"))
            .and_then(Value::as_u64),
        output_tokens: value
            .get("completion_tokens")
            .or_else(|| value.get("output_tokens"))
            .and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
        cached_tokens: value
            .pointer("/prompt_tokens_details/cached_tokens")
            .or_else(|| value.pointer("/input_tokens_details/cached_tokens"))
            .and_then(Value::as_u64),
        reasoning_tokens: value
            .pointer("/completion_tokens_details/reasoning_tokens")
            .or_else(|| value.pointer("/output_tokens_details/reasoning_tokens"))
            .and_then(Value::as_u64),
        details: BTreeMap::new(),
    }
}

fn encode_openai_usage(usage: &Usage) -> Value {
    json!({"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens, "total_tokens": usage.total_tokens, "prompt_tokens_details": {"cached_tokens": usage.cached_tokens}, "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens}})
}

fn parse_openai_annotation(value: &Value) -> Option<Annotation> {
    if value.get("type").and_then(Value::as_str) == Some("url_citation") {
        Some(Annotation::Citation {
            source: value
                .pointer("/url_citation/url")
                .and_then(Value::as_str)?
                .into(),
            start: value
                .pointer("/url_citation/start_index")
                .and_then(Value::as_u64),
            end: value
                .pointer("/url_citation/end_index")
                .and_then(Value::as_u64),
            title: value
                .pointer("/url_citation/title")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    } else {
        None
    }
}

fn content_as_text(content: &[ContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            ContentPart::Refusal { reason } => Some(reason.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn required_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, UnllmError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| UnllmError::invalid("missing_field", format!("Missing string field: {key}")))
}

fn from_value<T: serde::de::DeserializeOwned>(value: Value, code: &str) -> Result<T, UnllmError> {
    serde_json::from_value(value).map_err(|error| UnllmError::invalid(code, error.to_string()))
}

fn json_error(error: serde_json::Error) -> UnllmError {
    UnllmError::invalid("invalid_json", error.to_string())
}

fn sse(event_type: Option<&str>, value: &Value) -> Vec<u8> {
    let mut frame = String::new();
    if let Some(event_type) = event_type {
        frame.push_str("event: ");
        frame.push_str(event_type);
        frame.push('\n');
    }
    frame.push_str("data: ");
    frame.push_str(&value.to_string());
    frame.push_str("\n\n");
    frame.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unllm_core::{ConversionMode, Protocol};

    fn context() -> ConversionContext {
        ConversionContext {
            source: Protocol::OpenAiChatCompletions,
            target: Protocol::OpenAiChatCompletions,
            mode: ConversionMode::Strict,
            capabilities: capabilities(),
        }
    }

    #[test]
    fn chat_request_round_trip_preserves_semantics() {
        let body = json!({
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "temperature": 0.2,
            "stream": true
        });
        let decoded = OpenAiChatAdapter
            .decode_request(Operation::Generate, body, &context())
            .unwrap();
        let encoded = OpenAiChatAdapter
            .encode_request(&decoded.value, &context())
            .unwrap();
        assert_eq!(encoded.value["model"], "gpt-test");
        assert_eq!(encoded.value["messages"][0]["role"], "system");
        assert_eq!(encoded.value["stream"], true);
    }

    #[test]
    fn image_edit_uses_transport_neutral_multipart_metadata() {
        let body = json!({
            "model": "image-test",
            "prompt": "Add a hat",
            "_unllm_image_task": "edit",
            "_unllm_image": {"mime_type": "image/png", "data": "aW1hZ2U="}
        });
        let decoded = OpenAiChatAdapter
            .decode_request(Operation::Image, body, &context())
            .unwrap();
        let encoded = OpenAiChatAdapter
            .encode_request(&decoded.value, &context())
            .unwrap();
        assert_eq!(encoded.value["_unllm_multipart"]["task"], "edit");
    }
}
