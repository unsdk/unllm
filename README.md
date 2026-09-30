# unllm

unllm is a Rust workspace for translating large-model requests, responses, and streams through a protocol-neutral intermediate representation.

Supported API families:

- OpenAI Chat Completions and Responses
- Anthropic Messages
- Google Gemini Developer API
- OpenAI and Gemini embeddings and image operations

The unllm-gateway binary exposes the supported HTTP protocols at the same time and routes every configured model to exactly one upstream. It does not perform load balancing or failover.

## Workspace

| Crate | Purpose |
| --- | --- |
| unllm-core | Canonical requests, responses, diagnostics, capabilities, streaming events, and asset traits |
| unllm-openai | OpenAI Chat, Responses, Embeddings, and Images codecs |
| unllm-anthropic | Anthropic Messages codec |
| unllm-gemini | Gemini generateContent and embedding codecs |
| unllm | Feature-gated facade; all protocols are enabled by default |
| unllm-gateway | Installable Axum-based HTTP gateway |

All public crates support Rust 1.85 or newer and use the Apache-2.0 license.

## Library usage

    use serde_json::json;
    use unllm::{
        Capabilities, ConversionContext, ConversionMode, DynAdapter, Operation, Protocol,
    };
    use unllm::openai::OpenAiChatAdapter;

    let adapter = OpenAiChatAdapter;
    let context = ConversionContext {
        source: Protocol::OpenAiChatCompletions,
        target: Protocol::OpenAiChatCompletions,
        mode: ConversionMode::Strict,
        capabilities: adapter.capabilities(),
    };

    let canonical = adapter.decode_request(
        Operation::Generate,
        json!({
            "model": "example",
            "messages": [{"role": "user", "content": "Hello"}]
        }),
        &context,
    )?;

Canonical values have a stable, versioned JSON representation. Generated schemas are committed under schema/v1.

## Gateway quick start

Install and start a single-upstream gateway:

    cargo install unllm-gateway
    export UPSTREAM_API_KEY=...
    unllm-gateway serve \
      --protocol openai-responses \
      --base-url https://api.openai.com/ \
      --model gpt-5 \
      --token-env UPSTREAM_API_KEY

Or use the multi-upstream configuration:

    unllm-gateway generate-config > unllm.toml
    unllm-gateway check-config unllm.toml
    unllm-gateway serve --config unllm.toml

The complete example is available at examples/unllm.toml.

Model resolution is deterministic: exact alias, then provider/model, then a globally unique remote model. There is no automatic fallback or request retry.

## Conversion behavior

Strict conversion is the default. A field that the target cannot represent returns a structured Unsupported error. A route may explicitly enable lenient conversion; the library then returns stable warning codes and canonical paths in ConversionOutcome.

Provider-specific fields are stored under provider namespaces. They can be restored during a same-protocol semantic round trip. Cross-provider conversion only consumes fields explicitly understood by the destination adapter.

Streams are translated incrementally. The canonical stream describes response and content-block lifecycle events, text/reasoning/tool/media deltas, usage, finish reasons, in-stream errors, and unknown namespaced events. StreamAccumulator can reconstruct a complete response and validates fragmented tool JSON.

## Security model

- The server binds to 127.0.0.1 by default.
- Optional client keys are loaded from environment variables.
- Upstream secrets are never copied from inbound authentication headers.
- Arbitrary headers are not forwarded unless explicitly allowlisted.
- Media fetching is disabled by default. Enabling it requires an HTTPS host allowlist and rejects private or loopback addresses after DNS resolution and after every redirect.
- The explicit --log-bodies flag may log prompts and tool arguments; media payloads are replaced with hashes and size metadata.
- TLS, request-size limits, stream-size limits, and concurrency limits are intentionally delegated to a reverse proxy. Do not bind the gateway to an untrusted network without those external controls.

## Development

    cargo fmt --all -- --check
    cargo clippy --workspace --all-features --all-targets -- -D warnings
    cargo test --workspace --all-features
    cargo run -p unllm-core --example generate-schema -- --check
    pnpm install --frozen-lockfile
    pnpm test:sdk

Live provider tests are opt-in through pnpm test:live and require the corresponding provider API key environment variables.

## Release

All crates use one version. Pushing vX.Y.Z, or approving the manual release workflow for an optional commit on the default branch, validates the workspace, publishes crates in dependency order, builds gateway archives for five targets, and creates a GitHub Release with changelogithub. An omitted manual commit defaults to the latest commit on the remote default branch.

Before the first release:

1. Confirm ownership or availability of every unllm-* crate name.
2. Configure a protected GitHub release environment.
3. Register each crate's GitHub Trusted Publisher on crates.io, or configure the CARGO_REGISTRY_TOKEN repository secret. A configured API token takes precedence over Trusted Publishing.
4. Configure a GitHub remote for this repository.
