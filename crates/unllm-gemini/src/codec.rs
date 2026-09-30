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

use crate::types::{EmbedContentRequest, GenerateContentRequest, GenerateContentResponse};

/// Gemini Developer API codec.
#[derive(Debug, Default)]
pub struct GeminiAdapter;

impl GeminiAdapter {
    /// Decodes a typed Gemini generateContent request.
    pub fn decode_generate(
        &self,
        model: impl Into<String>,
        request: GenerateContentRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateRequest>, UnllmError> {
        let decoded =
            decode_generate_request(serde_json::to_value(request).map_err(json_error)?, context)?;
        let UnifiedRequest::Generate(mut value) = decoded.value else {
            return Err(UnllmError::invalid(
                "operation_mismatch",
                "Expected a generation request",
            ));
        };
        value.model = model.into();
        Ok(ConversionOutcome {
            value,
            diagnostics: decoded.diagnostics,
        })
    }

    /// Encodes a canonical request as a typed Gemini generateContent request.
    pub fn encode_generate(
        &self,
        request: &GenerateRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateContentRequest>, UnllmError> {
        let encoded = encode_generate_request(request, context)?;
        Ok(ConversionOutcome {
            value: serde_json::from_value(encoded.value).map_err(json_error)?,
            diagnostics: encoded.diagnostics,
        })
    }

    /// Decodes a typed Gemini generateContent response.
    pub fn decode_generate_response(
        &self,
        response: GenerateContentResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<GenerateResponse>, UnllmError> {
        let decoded =
            decode_generate_response(serde_json::to_value(response).map_err(json_error)?, context)?;
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

impl DynAdapter for GeminiAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::Gemini
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
        match operation {
            Operation::Generate => decode_generate_request(body, context),
            Operation::Embed => decode_embed_request(body),
            Operation::Image => decode_image_request(body, context),
        }
    }

    fn encode_request(
        &self,
        request: &UnifiedRequest,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        let mut outcome = match request {
            UnifiedRequest::Generate(request) => encode_generate_request(request, context),
            UnifiedRequest::Embed(request) => encode_embed_request(request, context),
            UnifiedRequest::Image(request) => encode_image_request(request, context),
        }?;
        let extensions = match request {
            UnifiedRequest::Generate(value) => &value.extensions,
            UnifiedRequest::Embed(value) => &value.extensions,
            UnifiedRequest::Image(value) => &value.extensions,
        };
        outcome
            .diagnostics
            .extend(context.check_extensions(extensions, "gemini", "/request")?);
        Ok(outcome)
    }

    fn decode_response(
        &self,
        operation: Operation,
        body: Value,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
        match operation {
            Operation::Generate => decode_generate_response(body, context),
            Operation::Embed => decode_embed_response(body),
            Operation::Image => decode_image_response(body, context),
        }
    }

    fn encode_response(
        &self,
        response: &UnifiedResponse,
        context: &ConversionContext,
    ) -> Result<ConversionOutcome<Value>, UnllmError> {
        let mut outcome = match response {
            UnifiedResponse::Generate(response) => encode_generate_response(response, context),
            UnifiedResponse::Embed(response) => encode_embed_response(response),
            UnifiedResponse::Image(response) => encode_image_response(response, context),
        }?;
        let extensions = match response {
            UnifiedResponse::Generate(value) => &value.extensions,
            UnifiedResponse::Embed(value) => &value.extensions,
            UnifiedResponse::Image(value) => &value.extensions,
        };
        outcome
            .diagnostics
            .extend(context.check_extensions(extensions, "gemini", "/response")?);
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
        operations: BTreeSet::from([Operation::Generate, Operation::Embed, Operation::Image]),
        features: BTreeSet::from([
            Capability::Streaming,
            Capability::Text,
            Capability::Image,
            Capability::Audio,
            Capability::Video,
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

fn decode_generate_request(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let request: GenerateContentRequest = serde_json::from_value(body)
        .map_err(|error| UnllmError::invalid("invalid_gemini_request", error.to_string()))?;
    let instructions = request
        .system_instruction
        .as_ref()
        .map(|content| Instruction {
            kind: InstructionKind::System,
            priority: 0,
            content: parse_parts(content.get("parts").unwrap_or(content), 0),
            extensions: Extensions::new(),
        })
        .into_iter()
        .collect();
    let messages = request
        .contents
        .iter()
        .enumerate()
        .map(|(index, content)| Message {
            role: if content.get("role").and_then(Value::as_str) == Some("model") {
                Role::Assistant
            } else {
                Role::User
            },
            content: parse_parts(content.get("parts").unwrap_or(content), index),
            id: None,
            extensions: Extensions::new(),
        })
        .collect();
    let tools = request
        .tools
        .iter()
        .flat_map(|container| {
            container
                .get("functionDeclarations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|tool| {
            Some(ToolDefinition {
                name: tool.get("name")?.as_str()?.to_owned(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                input_schema: tool
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
                extensions: Extensions::new(),
            })
        })
        .collect();
    let config = request.generation_config.unwrap_or_else(|| json!({}));
    let response_format = match config.get("responseMimeType").and_then(Value::as_str) {
        Some("application/json") => config
            .get("responseJsonSchema")
            .or_else(|| config.get("responseSchema"))
            .map_or(Some(ResponseFormat::JsonObject), |schema| {
                Some(ResponseFormat::JsonSchema {
                    name: "response".into(),
                    schema: schema.clone(),
                    strict: false,
                })
            }),
        _ => None,
    };
    let parameters = unllm_core::GenerationParameters {
        max_output_tokens: config.get("maxOutputTokens").and_then(Value::as_u64),
        temperature: config
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
        top_p: config
            .get("topP")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
        top_k: config.get("topK").and_then(Value::as_u64),
        stop_sequences: config
            .get("stopSequences")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        candidate_count: config
            .get("candidateCount")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        ..Default::default()
    };
    let tool_choice = request.tool_config.as_ref().and_then(parse_tool_choice);
    let extensions = if request.extra.is_empty() {
        Extensions::new()
    } else {
        BTreeMap::from([(
            "gemini".into(),
            Value::Object(request.extra.into_iter().collect()),
        )])
    };
    Ok(ConversionOutcome::clean(UnifiedRequest::Generate(
        GenerateRequest {
            model: String::new(),
            instructions,
            messages,
            tools,
            tool_choice,
            response_format,
            reasoning: None,
            parameters,
            stream: false,
            extensions,
        },
    )))
}

fn encode_generate_request(
    request: &GenerateRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let system_parts: Vec<_> = request
        .instructions
        .iter()
        .flat_map(|instruction| instruction.content.iter())
        .cloned()
        .collect();
    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(json!({"parts": encode_parts(&system_parts, context)?}))
    };
    let contents = request
        .messages
        .iter()
        .map(|message| {
            Ok(json!({
                "role": if message.role == Role::Assistant { "model" } else { "user" },
                "parts": encode_parts(&message.content, context)?,
            }))
        })
        .collect::<Result<Vec<Value>, UnllmError>>()?;
    let tools = if request.tools.is_empty() {
        Vec::new()
    } else {
        vec![
            json!({"functionDeclarations": request.tools.iter().map(|tool| json!({"name": tool.name, "description": tool.description, "parameters": tool.input_schema})).collect::<Vec<_>>()}),
        ]
    };
    let mut generation_config = Map::new();
    macro_rules! optional {
        ($name:literal, $value:expr) => {
            if let Some(value) = $value {
                generation_config.insert($name.into(), json!(value));
            }
        };
    }
    optional!("maxOutputTokens", request.parameters.max_output_tokens);
    optional!("temperature", request.parameters.temperature);
    optional!("topP", request.parameters.top_p);
    optional!("topK", request.parameters.top_k);
    optional!("candidateCount", request.parameters.candidate_count);
    if !request.parameters.stop_sequences.is_empty() {
        generation_config.insert(
            "stopSequences".into(),
            json!(request.parameters.stop_sequences),
        );
    }
    if let Some(format) = &request.response_format {
        match format {
            ResponseFormat::Text => {}
            ResponseFormat::JsonObject => {
                generation_config.insert(
                    "responseMimeType".into(),
                    Value::String("application/json".into()),
                );
            }
            ResponseFormat::JsonSchema { schema, .. } => {
                generation_config.insert(
                    "responseMimeType".into(),
                    Value::String("application/json".into()),
                );
                generation_config.insert("responseJsonSchema".into(), schema.clone());
            }
        }
    }
    if let Some(reasoning) = &request.reasoning {
        if let Some(budget) = reasoning.budget_tokens {
            generation_config.insert(
                "thinkingConfig".into(),
                json!({"thinkingBudget": budget, "includeThoughts": reasoning.include_summary}),
            );
        }
    }
    let mut value = serde_json::to_value(GenerateContentRequest {
        system_instruction,
        contents,
        tools,
        tool_config: request.tool_choice.as_ref().map(encode_tool_choice),
        generation_config: (!generation_config.is_empty())
            .then_some(Value::Object(generation_config)),
        extra: BTreeMap::new(),
    })
    .map_err(json_error)?;
    merge_extensions(value.as_object_mut().expect("object"), &request.extensions);
    Ok(ConversionOutcome::clean(value))
}

fn decode_generate_response(
    body: Value,
    _context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let response: GenerateContentResponse = serde_json::from_value(body)
        .map_err(|error| UnllmError::invalid("invalid_gemini_response", error.to_string()))?;
    let candidates = response
        .candidates
        .iter()
        .enumerate()
        .map(|(fallback_index, candidate)| Candidate {
            index: candidate
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or_else(|| u32::try_from(fallback_index).unwrap_or(u32::MAX)),
            id: None,
            content: parse_parts(
                candidate.pointer("/content/parts").unwrap_or(&Value::Null),
                fallback_index,
            ),
            finish_reason: candidate
                .get("finishReason")
                .and_then(Value::as_str)
                .map(parse_finish_reason),
            extensions: candidate
                .get("safetyRatings")
                .cloned()
                .map(|value| BTreeMap::from([("gemini".into(), json!({"safetyRatings": value}))]))
                .unwrap_or_default(),
        })
        .collect();
    let usage = response.usage_metadata.as_ref().map(parse_usage);
    let extensions = if response.extra.is_empty() {
        Extensions::new()
    } else {
        BTreeMap::from([(
            "gemini".into(),
            Value::Object(response.extra.into_iter().collect()),
        )])
    };
    Ok(ConversionOutcome::clean(UnifiedResponse::Generate(
        GenerateResponse {
            id: response.response_id,
            model: response.model_version.unwrap_or_default(),
            candidates,
            usage,
            extensions,
        },
    )))
}

fn encode_generate_response(
    response: &GenerateResponse,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let candidates = response
        .candidates
        .iter()
        .map(|candidate| {
            Ok(json!({
                "index": candidate.index,
                "content": {"role": "model", "parts": encode_parts(&candidate.content, context)?},
                "finishReason": candidate.finish_reason.as_ref().map(encode_finish_reason),
            }))
        })
        .collect::<Result<Vec<Value>, UnllmError>>()?;
    Ok(ConversionOutcome::clean(json!({
        "responseId": response.id,
        "modelVersion": response.model,
        "candidates": candidates,
        "usageMetadata": response.usage.as_ref().map(encode_usage),
    })))
}

fn decode_embed_request(body: Value) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    if let Some(requests) = body.get("requests").and_then(Value::as_array) {
        let model = requests
            .first()
            .and_then(|value| value.get("model"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim_start_matches("models/")
            .to_owned();
        let inputs = requests
            .iter()
            .map(|request| EmbeddingInput::Content {
                content: parse_parts(request.pointer("/content/parts").unwrap_or(&Value::Null), 0),
            })
            .collect();
        return Ok(ConversionOutcome::clean(UnifiedRequest::Embed(
            EmbedRequest {
                model,
                inputs,
                dimensions: requests
                    .first()
                    .and_then(|value| value.get("outputDimensionality"))
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok()),
                task_type: requests
                    .first()
                    .and_then(|value| value.get("taskType"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                extensions: Extensions::new(),
            },
        )));
    }
    let request: EmbedContentRequest = serde_json::from_value(body).map_err(json_error)?;
    Ok(ConversionOutcome::clean(UnifiedRequest::Embed(
        EmbedRequest {
            model: String::new(),
            inputs: vec![EmbeddingInput::Content {
                content: parse_parts(request.content.get("parts").unwrap_or(&request.content), 0),
            }],
            dimensions: request.output_dimensionality,
            task_type: request.task_type,
            extensions: if request.extra.is_empty() {
                Extensions::new()
            } else {
                BTreeMap::from([(
                    "gemini".into(),
                    Value::Object(request.extra.into_iter().collect()),
                )])
            },
        },
    )))
}

fn encode_embed_request(
    request: &EmbedRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let requests = request.inputs.iter().map(|input| {
        let parts = match input {
            EmbeddingInput::Text { text } => vec![json!({"text": text})],
            EmbeddingInput::Content { content } => encode_parts(content, context)?,
        };
        Ok(json!({"model": format!("models/{}", request.model), "content": {"parts": parts}, "taskType": request.task_type, "outputDimensionality": request.dimensions}))
    }).collect::<Result<Vec<Value>, UnllmError>>()?;
    let value = if requests.len() == 1 {
        let mut request = requests.into_iter().next().expect("one item");
        request.as_object_mut().expect("object").remove("model");
        request
    } else {
        json!({"requests": requests})
    };
    Ok(ConversionOutcome::clean(value))
}

fn decode_embed_response(body: Value) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let values: Vec<Value> = body
        .get("embeddings")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| body.get("embedding").cloned().map(|value| vec![value]))
        .unwrap_or_default();
    let embeddings = values
        .iter()
        .enumerate()
        .map(|(index, value)| Embedding {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            values: value
                .get("values")
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
            model: body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim_start_matches("models/")
                .into(),
            embeddings,
            usage: None,
            extensions: Extensions::new(),
        },
    )))
}

fn encode_embed_response(response: &EmbedResponse) -> Result<ConversionOutcome<Value>, UnllmError> {
    let embeddings = response
        .embeddings
        .iter()
        .map(|embedding| json!({"values": embedding.values}))
        .collect::<Vec<_>>();
    Ok(ConversionOutcome::clean(if embeddings.len() == 1 {
        json!({"embedding": embeddings[0]})
    } else {
        json!({"embeddings": embeddings})
    }))
}

fn decode_image_request(
    body: Value,
    context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedRequest>, UnllmError> {
    let generate = decode_generate_request(body, context)?.value;
    let UnifiedRequest::Generate(generate) = generate else {
        unreachable!()
    };
    let mut input_media = generate
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            ContentPart::Image { asset } => Some(asset.clone()),
            _ => None,
        });
    let first = input_media.next();
    let second = input_media.next();
    let task = match (first, second) {
        (None, _) => ImageTask::Generate,
        (Some(image), None) => ImageTask::Variation { image },
        (Some(image), Some(mask)) => ImageTask::Edit {
            image,
            mask: Some(mask),
        },
    };
    let prompt = generate
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(ConversionOutcome::clean(UnifiedRequest::Image(
        ImageRequest {
            model: generate.model,
            task,
            prompt,
            negative_prompt: None,
            count: generate.parameters.candidate_count,
            size: None,
            quality: None,
            style: None,
            output_format: Some("image/png".into()),
            stream: generate.stream,
            extensions: generate.extensions,
        },
    )))
}

fn encode_image_request(
    request: &ImageRequest,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    let mut parts = vec![ContentPart::text(&request.prompt)];
    match &request.task {
        ImageTask::Generate => {}
        ImageTask::Variation { image } => parts.push(ContentPart::Image {
            asset: image.clone(),
        }),
        ImageTask::Edit { image, mask } => {
            parts.push(ContentPart::Image {
                asset: image.clone(),
            });
            if let Some(mask) = mask {
                parts.push(ContentPart::Image {
                    asset: mask.clone(),
                });
            }
        }
    }
    let generate = GenerateRequest {
        model: request.model.clone(),
        instructions: Vec::new(),
        messages: vec![Message {
            role: Role::User,
            content: parts,
            id: None,
            extensions: Extensions::new(),
        }],
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        reasoning: None,
        parameters: unllm_core::GenerationParameters {
            candidate_count: request.count,
            ..Default::default()
        },
        stream: request.stream,
        extensions: request.extensions.clone(),
    };
    let mut result = encode_generate_request(&generate, context)?;
    let config = result
        .value
        .as_object_mut()
        .expect("object")
        .entry("generationConfig")
        .or_insert_with(|| json!({}));
    config
        .as_object_mut()
        .expect("object")
        .insert("responseModalities".into(), json!(["IMAGE"]));
    Ok(result)
}

fn decode_image_response(
    body: Value,
    context: &ConversionContext,
) -> Result<ConversionOutcome<UnifiedResponse>, UnllmError> {
    let generated = decode_generate_response(body, context)?.value;
    let UnifiedResponse::Generate(generated) = generated else {
        unreachable!()
    };
    let images = generated
        .candidates
        .iter()
        .flat_map(|candidate| candidate.content.iter())
        .filter_map(|part| match part {
            ContentPart::Image { asset } => Some(asset.clone()),
            _ => None,
        })
        .collect();
    Ok(ConversionOutcome::clean(UnifiedResponse::Image(
        ImageResponse {
            model: generated.model,
            images,
            usage: generated.usage,
            extensions: generated.extensions,
        },
    )))
}

fn encode_image_response(
    response: &ImageResponse,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Value>, UnllmError> {
    encode_generate_response(
        &GenerateResponse {
            id: None,
            model: response.model.clone(),
            candidates: vec![Candidate {
                index: 0,
                id: None,
                content: response
                    .images
                    .iter()
                    .cloned()
                    .map(|asset| ContentPart::Image { asset })
                    .collect(),
                finish_reason: Some(FinishReason::Stop),
                extensions: Extensions::new(),
            }],
            usage: response.usage.clone(),
            extensions: response.extensions.clone(),
        },
        context,
    )
}

fn parse_parts(value: &Value, message_index: usize) -> Vec<ContentPart> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(part_index, part)| {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                let annotations = if part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    vec![Annotation::Safety {
                        category: "reasoning".into(),
                        probability: None,
                        blocked: false,
                    }]
                } else {
                    Vec::new()
                };
                if part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    return Some(ContentPart::Reasoning {
                        text: Some(text.into()),
                        extensions: Extensions::new(),
                    });
                }
                return Some(ContentPart::Text {
                    text: text.into(),
                    annotations,
                });
            }
            if let Some(data) = part.get("inlineData") {
                let mime_type = data
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or("application/octet-stream");
                let bytes = data
                    .get("data")
                    .and_then(Value::as_str)
                    .and_then(|data| STANDARD.decode(data).ok())
                    .unwrap_or_default();
                let asset = MediaAsset::inline(mime_type, Bytes::from(bytes));
                return Some(match mime_type.split('/').next().unwrap_or_default() {
                    "image" => ContentPart::Image { asset },
                    "audio" => ContentPart::Audio { asset },
                    "video" => ContentPart::Video { asset },
                    _ => ContentPart::File {
                        asset,
                        filename: None,
                    },
                });
            }
            if let Some(file) = part.get("fileData") {
                let asset = MediaAsset::url(
                    file.get("fileUri")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    file.get("mimeType")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                );
                return Some(ContentPart::File {
                    asset,
                    filename: None,
                });
            }
            if let Some(call) = part.get("functionCall") {
                return Some(ContentPart::ToolCall {
                    call: ToolCall {
                        id: part
                            .get("thoughtSignature")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("call-{message_index}-{part_index}")),
                        name: call
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .into(),
                        arguments: call.get("args").cloned().unwrap_or_else(|| json!({})),
                        extensions: Extensions::new(),
                    },
                });
            }
            if let Some(result) = part.get("functionResponse") {
                return Some(ContentPart::ToolResult {
                    result: ToolResult {
                        call_id: result
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("call-{message_index}-{part_index}")),
                        content: vec![ContentPart::text(
                            result
                                .get("response")
                                .cloned()
                                .unwrap_or(Value::Null)
                                .to_string(),
                        )],
                        is_error: false,
                        extensions: Extensions::new(),
                    },
                });
            }
            None
        })
        .collect()
}

fn encode_parts(
    parts: &[ContentPart],
    context: &ConversionContext,
) -> Result<Vec<Value>, UnllmError> {
    let mut output = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text, .. } => output.push(json!({"text": text})),
            ContentPart::Reasoning { text, extensions } => output.push(json!({"text": text, "thought": true, "thoughtSignature": extensions.get("gemini").and_then(|value| value.get("thoughtSignature"))})),
            ContentPart::Image { asset } | ContentPart::Audio { asset } | ContentPart::Video { asset } => output.push(encode_media(asset, "gemini")?),
            ContentPart::File { asset, .. } => output.push(encode_media(asset, "gemini")?),
            ContentPart::ToolCall { call } => output.push(json!({"functionCall": {"name": call.name, "args": call.arguments}, "thoughtSignature": call.extensions.get("gemini").and_then(|value| value.get("thoughtSignature"))})),
            ContentPart::ToolResult { result } => output.push(json!({"functionResponse": {"id": result.call_id, "name": result.extensions.get("gemini").and_then(|value| value.get("name")), "response": {"output": content_as_text(&result.content)}}})),
            ContentPart::Refusal { reason } => output.push(json!({"text": reason})),
        }
    }
    let _ = context;
    Ok(output)
}

fn encode_media(asset: &MediaAsset, provider: &str) -> Result<Value, UnllmError> {
    match &asset.source {
        MediaSource::Inline { mime_type, data } => {
            Ok(json!({"inlineData": {"mimeType": mime_type, "data": STANDARD.encode(data)}}))
        }
        MediaSource::Url { url, mime_type } => {
            Ok(json!({"fileData": {"mimeType": mime_type, "fileUri": url}}))
        }
        MediaSource::ProviderFile {
            provider: owner,
            id,
            mime_type,
        } if owner == provider => Ok(json!({"fileData": {"mimeType": mime_type, "fileUri": id}})),
        MediaSource::ProviderFile { .. } => Err(UnllmError::unsupported(
            "foreign_file_id",
            "A provider file identifier cannot be sent to Gemini",
        )),
    }
}

fn parse_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value
        .pointer("/functionCallingConfig/mode")
        .and_then(Value::as_str)
    {
        Some("AUTO") => Some(ToolChoice::Auto),
        Some("NONE") => Some(ToolChoice::None),
        Some("ANY") => value
            .pointer("/functionCallingConfig/allowedFunctionNames")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
            .and_then(Value::as_str)
            .map_or(Some(ToolChoice::Required), |name| {
                Some(ToolChoice::Function { name: name.into() })
            }),
        _ => None,
    }
}

fn encode_tool_choice(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!({"functionCallingConfig": {"mode": "AUTO"}}),
        ToolChoice::None => json!({"functionCallingConfig": {"mode": "NONE"}}),
        ToolChoice::Required => json!({"functionCallingConfig": {"mode": "ANY"}}),
        ToolChoice::Function { name } => {
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}})
        }
    }
}

fn decode_stream(
    event_type: Option<&str>,
    data: &[u8],
    _context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<StreamEvent>>, UnllmError> {
    let value: Value = serde_json::from_slice(data).map_err(json_error)?;
    if value.get("error").is_some() {
        return Ok(ConversionOutcome::clean(vec![StreamEvent::Error {
            error: UnllmError {
                code: value
                    .pointer("/error/status")
                    .and_then(Value::as_str)
                    .unwrap_or("gemini_stream_error")
                    .into(),
                kind: unllm_core::ErrorKind::Upstream,
                message: value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("Gemini stream failed")
                    .into(),
                source_protocol: Some(Protocol::Gemini),
                target: None,
                operation: Some(Operation::Generate),
                metadata: None,
            },
        }]));
    }
    let mut events = Vec::new();
    if let Some(id) = value.get("responseId").and_then(Value::as_str) {
        events.push(StreamEvent::ResponseStart {
            id: Some(id.into()),
            model: value
                .get("modelVersion")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            extensions: Extensions::new(),
        });
    }
    for (candidate_position, candidate) in value
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let candidate_index = candidate
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or_else(|| u32::try_from(candidate_position).unwrap_or(u32::MAX));
        for (block_position, part) in candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let block_index = u32::try_from(block_position).unwrap_or(u32::MAX);
            let delta = if let Some(text) = part.get("text").and_then(Value::as_str) {
                if part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    ContentDelta::Reasoning { text: text.into() }
                } else {
                    ContentDelta::Text { text: text.into() }
                }
            } else if let Some(call) = part.get("functionCall") {
                ContentDelta::ToolArguments {
                    call_id: format!("call-{candidate_index}-{block_index}"),
                    name: call.get("name").and_then(Value::as_str).map(str::to_owned),
                    fragment: call
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| json!({}))
                        .to_string(),
                }
            } else {
                let parsed = parse_parts(&Value::Array(vec![part.clone()]), candidate_position)
                    .into_iter()
                    .next();
                if let Some(part) = parsed {
                    ContentDelta::Media { part }
                } else {
                    continue;
                }
            };
            events.push(StreamEvent::ContentDelta {
                candidate_index,
                block_index,
                delta,
            });
        }
        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
            events.push(StreamEvent::Finish {
                candidate_index,
                reason: Some(parse_finish_reason(reason)),
            });
        }
    }
    if let Some(usage) = value.get("usageMetadata") {
        events.push(StreamEvent::Usage {
            usage: parse_usage(usage),
        });
    }
    if events.is_empty() {
        events.push(StreamEvent::Raw {
            provider: "gemini".into(),
            event_type: event_type.map(str::to_owned),
            payload: value,
        });
    }
    Ok(ConversionOutcome::clean(events))
}

fn encode_stream(
    event: &StreamEvent,
    context: &ConversionContext,
) -> Result<ConversionOutcome<Vec<Vec<u8>>>, UnllmError> {
    let value = match event {
        StreamEvent::ResponseStart { id, model, .. } => {
            json!({"responseId": id, "modelVersion": model, "candidates": []})
        }
        StreamEvent::ContentDelta {
            candidate_index,
            delta,
            ..
        } => {
            let part = match delta {
                ContentDelta::Text { text } => json!({"text": text}),
                ContentDelta::Reasoning { text } => json!({"text": text, "thought": true}),
                ContentDelta::ToolArguments { name, fragment, .. } => {
                    json!({"functionCall": {"name": name, "args": serde_json::from_str::<Value>(fragment).unwrap_or_else(|_| Value::String(fragment.clone()))}})
                }
                ContentDelta::Media { part } => encode_parts(std::slice::from_ref(part), context)?
                    .into_iter()
                    .next()
                    .unwrap_or(Value::Null),
            };
            json!({"candidates": [{"index": candidate_index, "content": {"role": "model", "parts": [part]}}]})
        }
        StreamEvent::Finish {
            candidate_index,
            reason,
        } => {
            json!({"candidates": [{"index": candidate_index, "finishReason": reason.as_ref().map(encode_finish_reason)}]})
        }
        StreamEvent::Usage { usage } => json!({"usageMetadata": encode_usage(usage)}),
        StreamEvent::Error { error } => {
            json!({"error": {"status": error.code, "message": error.message}})
        }
        StreamEvent::Raw {
            provider, payload, ..
        } if provider == "gemini" => payload.clone(),
        StreamEvent::ContentBlockStart { .. } | StreamEvent::ContentBlockStop { .. } => {
            return Ok(ConversionOutcome::clean(Vec::new()));
        }
        StreamEvent::Raw { .. } => {
            let warning = context.lossy(
                "raw_stream_event",
                "/",
                "A provider-specific stream event cannot be represented by Gemini",
            )?;
            return Ok(ConversionOutcome {
                value: Vec::new(),
                diagnostics: vec![warning],
            });
        }
    };
    Ok(ConversionOutcome::clean(vec![
        format!("data: {value}\n\n").into_bytes(),
    ]))
}

fn parse_usage(value: &Value) -> Usage {
    Usage {
        input_tokens: value.get("promptTokenCount").and_then(Value::as_u64),
        output_tokens: value.get("candidatesTokenCount").and_then(Value::as_u64),
        total_tokens: value.get("totalTokenCount").and_then(Value::as_u64),
        cached_tokens: value.get("cachedContentTokenCount").and_then(Value::as_u64),
        reasoning_tokens: value.get("thoughtsTokenCount").and_then(Value::as_u64),
        details: BTreeMap::new(),
    }
}

fn encode_usage(value: &Usage) -> Value {
    json!({"promptTokenCount": value.input_tokens, "candidatesTokenCount": value.output_tokens, "totalTokenCount": value.total_tokens, "cachedContentTokenCount": value.cached_tokens, "thoughtsTokenCount": value.reasoning_tokens})
}

fn parse_finish_reason(value: &str) -> FinishReason {
    match value {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" => FinishReason::ContentFilter,
        "MALFORMED_FUNCTION_CALL" => FinishReason::Error,
        other => FinishReason::Other(other.into()),
    }
}

fn encode_finish_reason(value: &FinishReason) -> &str {
    match value {
        FinishReason::Stop => "STOP",
        FinishReason::Length => "MAX_TOKENS",
        FinishReason::ToolCall => "STOP",
        FinishReason::ContentFilter => "SAFETY",
        FinishReason::Error => "MALFORMED_FUNCTION_CALL",
        FinishReason::Other(raw) => raw,
    }
}

fn content_as_text(content: &[ContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } | ContentPart::Refusal { reason: text } => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn merge_extensions(object: &mut Map<String, Value>, extensions: &Extensions) {
    if let Some(Value::Object(extra)) = extensions.get("gemini") {
        for (key, value) in extra {
            object.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
}

fn json_error(error: serde_json::Error) -> UnllmError {
    UnllmError::invalid("invalid_json", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use unllm_core::ConversionMode;

    #[test]
    fn decodes_generate_content() {
        let context = ConversionContext {
            source: Protocol::Gemini,
            target: Protocol::Gemini,
            mode: ConversionMode::Strict,
            capabilities: capabilities(),
        };
        let body = json!({"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]});
        let decoded = GeminiAdapter
            .decode_request(Operation::Generate, body, &context)
            .unwrap();
        let UnifiedRequest::Generate(request) = decoded.value else {
            panic!("expected generation")
        };
        assert_eq!(request.messages.len(), 1);
    }
}
