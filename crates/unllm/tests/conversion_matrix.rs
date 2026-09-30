use std::collections::BTreeSet;

use serde_json::json;
use unllm::{
    Candidate, Capabilities, Capability, ContentPart, ConversionContext, ConversionMode,
    DynAdapter, Extensions, FinishReason, GenerateResponse, Operation, Protocol, UnifiedRequest,
    UnifiedResponse, Usage,
    anthropic::AnthropicAdapter,
    convert_request,
    gemini::GeminiAdapter,
    openai::{OpenAiChatAdapter, OpenAiResponsesAdapter},
};

fn all_capabilities() -> Capabilities {
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

fn context(source: Protocol, target: Protocol, mode: ConversionMode) -> ConversionContext {
    ConversionContext {
        source,
        target,
        mode,
        capabilities: all_capabilities(),
    }
}

#[test]
fn chat_request_converts_to_every_generation_protocol() {
    let source = OpenAiChatAdapter;
    let decoded = source
        .decode_request(
            Operation::Generate,
            json!({
                "model": "route",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Echo a value",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"]
                        }
                    }
                }],
                "stream": true
            }),
            &context(
                Protocol::OpenAiChatCompletions,
                Protocol::OpenAiChatCompletions,
                ConversionMode::Strict,
            ),
        )
        .unwrap();

    let targets: Vec<(Protocol, Box<dyn DynAdapter>)> = vec![
        (Protocol::OpenAiChatCompletions, Box::new(OpenAiChatAdapter)),
        (Protocol::OpenAiResponses, Box::new(OpenAiResponsesAdapter)),
        (Protocol::AnthropicMessages, Box::new(AnthropicAdapter)),
        (Protocol::Gemini, Box::new(GeminiAdapter)),
    ];
    for (protocol, adapter) in targets {
        adapter
            .encode_request(
                &decoded.value,
                &context(
                    Protocol::OpenAiChatCompletions,
                    protocol,
                    ConversionMode::Strict,
                ),
            )
            .unwrap();
    }
}

#[test]
fn every_generation_request_decodes_and_reencodes_across_the_matrix() {
    let sources: Vec<(Protocol, Box<dyn DynAdapter>, serde_json::Value)> = vec![
        (
            Protocol::OpenAiResponses,
            Box::new(OpenAiResponsesAdapter),
            json!({"model": "route", "input": "Hello"}),
        ),
        (
            Protocol::AnthropicMessages,
            Box::new(AnthropicAdapter),
            json!({"model": "route", "max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}]}),
        ),
        (
            Protocol::Gemini,
            Box::new(GeminiAdapter),
            json!({"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}),
        ),
    ];
    for (source_protocol, source, body) in sources {
        let mut request = source
            .decode_request(
                Operation::Generate,
                body,
                &context(source_protocol, source_protocol, ConversionMode::Strict),
            )
            .unwrap()
            .value;
        request.set_model("route");
        for (target_protocol, target) in adapters() {
            target
                .encode_request(
                    &request,
                    &context(source_protocol, target_protocol, ConversionMode::Strict),
                )
                .unwrap();
        }
    }
}

#[test]
fn canonical_response_encodes_to_every_generation_protocol() {
    let response = UnifiedResponse::Generate(GenerateResponse {
        id: Some("response-1".into()),
        model: "route".into(),
        candidates: vec![Candidate {
            index: 0,
            id: None,
            content: vec![ContentPart::text("Hello")],
            finish_reason: Some(FinishReason::Stop),
            extensions: Extensions::new(),
        }],
        usage: Some(Usage {
            input_tokens: Some(2),
            output_tokens: Some(1),
            total_tokens: Some(3),
            ..Default::default()
        }),
        extensions: Extensions::new(),
    });
    for (protocol, adapter) in adapters() {
        adapter
            .encode_response(
                &response,
                &context(Protocol::OpenAiResponses, protocol, ConversionMode::Strict),
            )
            .unwrap();
    }
}

fn adapters() -> Vec<(Protocol, Box<dyn DynAdapter>)> {
    vec![
        (Protocol::OpenAiChatCompletions, Box::new(OpenAiChatAdapter)),
        (Protocol::OpenAiResponses, Box::new(OpenAiResponsesAdapter)),
        (Protocol::AnthropicMessages, Box::new(AnthropicAdapter)),
        (Protocol::Gemini, Box::new(GeminiAdapter)),
    ]
}

#[test]
fn convenience_api_converts_wire_values_directly() {
    let outcome = convert_request(
        &OpenAiChatAdapter,
        &GeminiAdapter,
        Operation::Generate,
        json!({"model": "route", "messages": [{"role": "user", "content": "Hello"}]}),
        ConversionMode::Strict,
        all_capabilities(),
    )
    .unwrap();
    assert_eq!(outcome.value["contents"][0]["role"], "user");
}

#[test]
fn strict_mode_rejects_foreign_extensions() {
    let source = OpenAiChatAdapter;
    let mut request = source
        .decode_request(
            Operation::Generate,
            json!({
                "model": "route",
                "messages": [{"role": "user", "content": "Hello"}],
                "service_tier": "priority"
            }),
            &context(
                Protocol::OpenAiChatCompletions,
                Protocol::OpenAiChatCompletions,
                ConversionMode::Strict,
            ),
        )
        .unwrap()
        .value;
    let UnifiedRequest::Generate(generate) = &mut request else {
        panic!("expected a generation request");
    };
    assert!(generate.extensions.contains_key("openai"));

    let error = AnthropicAdapter
        .encode_request(
            &request,
            &context(
                Protocol::OpenAiChatCompletions,
                Protocol::AnthropicMessages,
                ConversionMode::Strict,
            ),
        )
        .unwrap_err();
    assert_eq!(error.code, "foreign_provider_extension");
}

#[test]
fn lenient_mode_reports_foreign_extensions() {
    let source = OpenAiChatAdapter;
    let request = source
        .decode_request(
            Operation::Generate,
            json!({
                "model": "route",
                "messages": [{"role": "user", "content": "Hello"}],
                "service_tier": "priority"
            }),
            &context(
                Protocol::OpenAiChatCompletions,
                Protocol::OpenAiChatCompletions,
                ConversionMode::Strict,
            ),
        )
        .unwrap()
        .value;
    let outcome = AnthropicAdapter
        .encode_request(
            &request,
            &context(
                Protocol::OpenAiChatCompletions,
                Protocol::AnthropicMessages,
                ConversionMode::Lenient,
            ),
        )
        .unwrap();
    assert_eq!(outcome.diagnostics[0].code, "foreign_provider_extension");
}
