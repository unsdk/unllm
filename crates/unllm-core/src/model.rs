use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Arbitrary JSON values keyed by provider namespace.
pub type Extensions = BTreeMap<String, Value>;

/// A protocol understood by unllm.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// OpenAI Chat Completions.
    OpenAiChatCompletions,
    /// OpenAI Responses.
    OpenAiResponses,
    /// Anthropic Messages.
    AnthropicMessages,
    /// Google Gemini Developer API.
    Gemini,
}

impl Protocol {
    /// Returns the stable provider namespace used by extensions.
    #[must_use]
    pub const fn namespace(self) -> &'static str {
        match self {
            Self::OpenAiChatCompletions | Self::OpenAiResponses => "openai",
            Self::AnthropicMessages => "anthropic",
            Self::Gemini => "gemini",
        }
    }
}

/// A top-level inference operation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Conversational or multimodal generation.
    Generate,
    /// Embedding generation.
    Embed,
    /// Image generation, editing, or variation.
    Image,
}

/// A capability that may be advertised by an adapter or route.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Streaming generation.
    Streaming,
    /// Text input or output.
    Text,
    /// Image input or output.
    Image,
    /// Audio input or output.
    Audio,
    /// Video input or output.
    Video,
    /// File input.
    File,
    /// Function tool calling.
    FunctionTools,
    /// Structured JSON output.
    StructuredOutput,
    /// Reasoning controls or content.
    Reasoning,
    /// Multiple candidates.
    MultipleCandidates,
    /// Embeddings.
    Embeddings,
    /// Image editing.
    ImageEdit,
    /// Image variations.
    ImageVariation,
    /// Provider-managed file upload.
    FileUpload,
}

/// Capabilities supported by an adapter or configured route.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Capabilities {
    /// Supported top-level operations.
    pub operations: BTreeSet<Operation>,
    /// Supported operation features.
    pub features: BTreeSet<Capability>,
}

impl Capabilities {
    /// Returns true when this set includes every requested capability.
    #[must_use]
    pub fn contains_all(&self, requested: &BTreeSet<Capability>) -> bool {
        requested.is_subset(&self.features)
    }

    /// Intersects this capability set with a route-level restriction.
    #[must_use]
    pub fn restricted_to(&self, allowed: &Self) -> Self {
        Self {
            operations: self
                .operations
                .intersection(&allowed.operations)
                .copied()
                .collect(),
            features: self
                .features
                .intersection(&allowed.features)
                .cloned()
                .collect(),
        }
    }
}

/// A media value used by multimodal inputs and outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MediaAsset {
    /// The media payload or reference.
    pub source: MediaSource,
    /// Optional pixel width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Optional pixel height.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Optional SHA-256 digest in lowercase hexadecimal form.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Provider-specific media metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

impl MediaAsset {
    /// Creates an inline media value.
    #[must_use]
    pub fn inline(mime_type: impl Into<String>, data: Bytes) -> Self {
        Self {
            source: MediaSource::Inline {
                mime_type: mime_type.into(),
                data,
            },
            width: None,
            height: None,
            sha256: None,
            extensions: Extensions::new(),
        }
    }

    /// Creates a remote media reference.
    #[must_use]
    pub fn url(url: impl Into<String>, mime_type: Option<String>) -> Self {
        Self {
            source: MediaSource::Url {
                url: url.into(),
                mime_type,
            },
            width: None,
            height: None,
            sha256: None,
            extensions: Extensions::new(),
        }
    }
}

/// The backing source of a media asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaSource {
    /// Bytes held directly by the request or response.
    Inline {
        /// IANA media type.
        mime_type: String,
        /// Raw decoded bytes. Canonical JSON uses base64.
        #[serde(with = "base64_bytes")]
        #[schemars(with = "String")]
        data: Bytes,
    },
    /// A remote URL.
    Url {
        /// Absolute media URL.
        url: String,
        /// Optional IANA media type.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    /// A provider-managed file reference.
    ProviderFile {
        /// Provider namespace that owns the file.
        provider: String,
        /// Provider file identifier.
        id: String,
        /// Optional IANA media type.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
}

/// A message author role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// End-user content.
    User,
    /// Model-generated content.
    Assistant,
    /// A tool result message.
    Tool,
}

/// The kind of high-priority instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InstructionKind {
    /// A system-level instruction.
    System,
    /// A developer-level instruction.
    Developer,
}

/// A high-priority instruction kept separate from conversation messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Instruction {
    /// Instruction class.
    pub kind: InstructionKind,
    /// Lower values indicate higher priority.
    #[serde(default)]
    pub priority: i32,
    /// Ordered multimodal content.
    pub content: Vec<ContentPart>,
    /// Provider-specific instruction data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

impl Instruction {
    /// Creates a system instruction from text.
    #[must_use]
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            kind: InstructionKind::System,
            priority: 0,
            content: vec![ContentPart::text(text)],
            extensions: Extensions::new(),
        }
    }
}

/// A conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Message {
    /// Message author.
    pub role: Role,
    /// Ordered message content.
    pub content: Vec<ContentPart>,
    /// Optional stable provider message identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-specific message data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

impl Message {
    /// Creates a user message from text.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    /// Creates an assistant message from text.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::text(text)],
            id: None,
            extensions: Extensions::new(),
        }
    }
}

/// A typed content item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text.
    Text {
        /// Text content.
        text: String,
        /// Structured annotations associated with the text.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Annotation>,
    },
    /// An image.
    Image { asset: MediaAsset },
    /// Audio.
    Audio { asset: MediaAsset },
    /// Video.
    Video { asset: MediaAsset },
    /// A general file or document.
    File {
        asset: MediaAsset,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// A model-requested function invocation.
    ToolCall { call: ToolCall },
    /// The result of a previous tool invocation.
    ToolResult { result: ToolResult },
    /// Model reasoning or thinking content.
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        extensions: Extensions,
    },
    /// A structured refusal.
    Refusal { reason: String },
}

impl ContentPart {
    /// Creates a plain text content part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            annotations: Vec::new(),
        }
    }
}

/// A portable response annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// A citation into a source.
    Citation {
        source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        start: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// A safety classification.
    Safety {
        category: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        probability: Option<f32>,
        blocked: bool,
    },
    /// A refusal marker.
    Refusal { reason: String },
}

/// A portable function declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolDefinition {
    /// Function name.
    pub name: String,
    /// Optional function description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema accepted by the function.
    pub input_schema: Value,
    /// Provider-specific tool data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// Tool selection policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// The provider chooses whether to call a tool.
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call a named function.
    Function { name: String },
}

/// A completed tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCall {
    /// Stable call identifier.
    pub id: String,
    /// Function name.
    pub name: String,
    /// Completed JSON arguments.
    pub arguments: Value,
    /// Provider-specific call data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// A completed tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolResult {
    /// Identifier of the matching tool call.
    pub call_id: String,
    /// Ordered tool result content.
    pub content: Vec<ContentPart>,
    /// True when the tool invocation failed.
    #[serde(default)]
    pub is_error: bool,
    /// Provider-specific result data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// Structured output preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Unconstrained text.
    Text,
    /// Any JSON object.
    JsonObject,
    /// JSON constrained by a named JSON Schema.
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(default)]
        strict: bool,
    },
}

/// Common reasoning controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningConfig {
    /// Provider-neutral effort label such as low, medium, or high.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Requested reasoning token budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
    /// Whether a human-readable summary is requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_summary: Option<bool>,
}

/// Common generation controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerationParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
}

/// A normalized generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerateRequest {
    /// Route-facing model name.
    pub model: String,
    /// High-priority instructions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instructions: Vec<Instruction>,
    /// Conversation history and current user input.
    pub messages: Vec<Message>,
    /// Available function tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Tool selection policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Structured output preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Reasoning controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Sampling and output controls.
    #[serde(default)]
    pub parameters: GenerationParameters,
    /// Whether the caller requested a stream.
    #[serde(default)]
    pub stream: bool,
    /// Provider-specific request fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

impl GenerateRequest {
    /// Creates a generation request with default optional controls.
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            instructions: Vec::new(),
            messages,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            reasoning: None,
            parameters: GenerationParameters::default(),
            stream: false,
            extensions: Extensions::new(),
        }
    }
}

/// A model candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Candidate {
    /// Stable candidate position.
    pub index: u32,
    /// Optional provider candidate identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Ordered output content.
    pub content: Vec<ContentPart>,
    /// Why generation ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Provider-specific candidate data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// A normalized finish reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "raw", rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
    Other(String),
}

/// Token usage reported by an upstream provider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, u64>,
}

/// A normalized generation response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerateResponse {
    /// Optional provider response identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Model reported by the provider.
    pub model: String,
    /// Ordered response candidates.
    pub candidates: Vec<Candidate>,
    /// Provider-reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Provider-specific response fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// Input accepted by the embedding operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingInput {
    /// Plain text.
    Text { text: String },
    /// Ordered multimodal content.
    Content { content: Vec<ContentPart> },
}

/// A normalized embedding request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EmbedRequest {
    pub model: String,
    /// A non-empty ordered list of inputs.
    pub inputs: Vec<EmbeddingInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

impl EmbedRequest {
    /// Creates an embedding request.
    #[must_use]
    pub fn new(model: impl Into<String>, inputs: Vec<EmbeddingInput>) -> Self {
        Self {
            model: model.into(),
            inputs,
            dimensions: None,
            task_type: None,
            extensions: Extensions::new(),
        }
    }
}

/// A single embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Embedding {
    pub index: u32,
    pub values: Vec<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// A normalized embedding response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Embedding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// Image operation details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageTask {
    /// Generate images from a prompt.
    Generate,
    /// Edit an input image, optionally under a mask.
    Edit {
        image: MediaAsset,
        #[serde(skip_serializing_if = "Option::is_none")]
        mask: Option<MediaAsset>,
    },
    /// Produce variations of an input image.
    Variation { image: MediaAsset },
}

/// A normalized image request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ImageRequest {
    pub model: String,
    pub task: ImageTask,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

impl ImageRequest {
    /// Creates a text-to-image generation request.
    #[must_use]
    pub fn generate(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            task: ImageTask::Generate,
            prompt: prompt.into(),
            negative_prompt: None,
            count: None,
            size: None,
            quality: None,
            style: None,
            output_format: None,
            stream: false,
            extensions: Extensions::new(),
        }
    }
}

/// A normalized image response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ImageResponse {
    pub model: String,
    pub images: Vec<MediaAsset>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: Extensions,
}

/// Any normalized request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", content = "request", rename_all = "snake_case")]
pub enum UnifiedRequest {
    Generate(GenerateRequest),
    Embed(EmbedRequest),
    Image(ImageRequest),
}

impl UnifiedRequest {
    /// Returns the operation represented by this request.
    #[must_use]
    pub const fn operation(&self) -> Operation {
        match self {
            Self::Generate(_) => Operation::Generate,
            Self::Embed(_) => Operation::Embed,
            Self::Image(_) => Operation::Image,
        }
    }

    /// Returns the route-facing model name.
    #[must_use]
    pub fn model(&self) -> &str {
        match self {
            Self::Generate(request) => &request.model,
            Self::Embed(request) => &request.model,
            Self::Image(request) => &request.model,
        }
    }

    /// Replaces the model name after route resolution.
    pub fn set_model(&mut self, model: impl Into<String>) {
        let model = model.into();
        match self {
            Self::Generate(request) => request.model = model,
            Self::Embed(request) => request.model = model,
            Self::Image(request) => request.model = model,
        }
    }

    /// Returns whether this request asks for a streaming response.
    #[must_use]
    pub const fn is_streaming(&self) -> bool {
        match self {
            Self::Generate(request) => request.stream,
            Self::Image(request) => request.stream,
            Self::Embed(_) => false,
        }
    }

    /// Validates invariants that are independent of a provider.
    pub fn validate(&self) -> Result<(), crate::UnllmError> {
        if self.model().trim().is_empty() {
            return Err(crate::UnllmError::invalid(
                "empty_model",
                "A model name is required",
            ));
        }
        match self {
            Self::Generate(request) => {
                if request.messages.is_empty() && request.instructions.is_empty() {
                    return Err(crate::UnllmError::invalid(
                        "empty_generation_input",
                        "A generation request requires a message or instruction",
                    ));
                }
                if request.parameters.candidate_count == Some(0) {
                    return Err(crate::UnllmError::invalid(
                        "invalid_candidate_count",
                        "candidate_count must be greater than zero",
                    ));
                }
                for (name, value) in [
                    ("temperature", request.parameters.temperature),
                    ("top_p", request.parameters.top_p),
                ] {
                    if value.is_some_and(|value| !value.is_finite()) {
                        return Err(crate::UnllmError::invalid(
                            "invalid_generation_parameter",
                            format!("{name} must be finite"),
                        ));
                    }
                }
            }
            Self::Embed(request) if request.inputs.is_empty() => {
                return Err(crate::UnllmError::invalid(
                    "empty_embedding_input",
                    "An embedding request requires at least one input",
                ));
            }
            Self::Image(request)
                if !matches!(request.task, ImageTask::Variation { .. })
                    && request.prompt.trim().is_empty() =>
            {
                return Err(crate::UnllmError::invalid(
                    "empty_image_prompt",
                    "An image request requires a prompt",
                ));
            }
            Self::Embed(_) | Self::Image(_) => {}
        }
        Ok(())
    }
}

/// Any normalized response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", content = "response", rename_all = "snake_case")]
pub enum UnifiedResponse {
    Generate(GenerateResponse),
    Embed(EmbedResponse),
    Image(ImageResponse),
}

/// Versioned serialized request contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RequestEnvelope {
    /// Canonical schema version. Version 1 is currently supported.
    pub schema_version: u32,
    /// Canonical request.
    pub value: UnifiedRequest,
}

impl RequestEnvelope {
    /// Wraps a request in the current schema version.
    #[must_use]
    pub const fn new(value: UnifiedRequest) -> Self {
        Self {
            schema_version: crate::SCHEMA_VERSION,
            value,
        }
    }
}

/// Versioned serialized response contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResponseEnvelope {
    /// Canonical schema version. Version 1 is currently supported.
    pub schema_version: u32,
    /// Canonical response.
    pub value: UnifiedResponse,
}

impl ResponseEnvelope {
    /// Wraps a response in the current schema version.
    #[must_use]
    pub const fn new(value: UnifiedResponse) -> Self {
        Self {
            schema_version: crate::SCHEMA_VERSION,
            value,
        }
    }
}

mod base64_bytes {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        STANDARD
            .decode(encoded)
            .map(Bytes::from)
            .map_err(serde::de::Error::custom)
    }
}
