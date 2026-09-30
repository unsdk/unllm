# Architecture

## Data flow

Every request follows one explicit translation path:

1. The inbound adapter decodes an official wire body into a canonical request.
2. The router resolves the public model name to one upstream and remote model.
3. Capability preflight rejects unsupported operations or modalities.
4. The optional preparation stage resolves remote media and stages files.
5. The outbound adapter encodes the canonical request.
6. The gateway sends one upstream HTTP request without automatic retry.
7. The upstream response or stream is decoded to canonical form and encoded with the inbound protocol.
8. Temporary file leases are cleaned up on completion, cancellation, or stream drop.

The pure codec path is synchronous. Network-backed media preparation is asynchronous and injected through core traits, so unllm-core has no Tokio dependency.

## Protocol snapshots

The initial wire contracts are intentionally pinned:

| Family | Snapshot |
| --- | --- |
| OpenAI | Chat Completions, Responses, Embeddings, and Images REST/SSE shapes included in the 0.1 fixtures |
| Anthropic | Messages with anthropic-version 2023-06-01; beta features require route opt-in |
| Gemini | Developer API v1beta generateContent, streamGenerateContent, embedContent, and batchEmbedContents |

New provider fields are captured in a provider namespace. A new official API version or beta feature must add fixtures and ship in a new crate version.

## Conversion guarantees

- Same-protocol round trips preserve known semantics and captured unknown fields, not JSON whitespace or object-key order.
- Strict mode rejects known semantic loss.
- Lenient mode drops unsupported fields only after emitting stable diagnostics.
- Provider file identifiers never cross provider namespaces.
- IDs missing from a source protocol are generated deterministically within a request.
- Token usage is never estimated.

## Streaming

SSE input is decoded incrementally across arbitrary byte and UTF-8 boundaries. Canonical events retain candidate and content-block indexes. Tool argument fragments remain strings until the accumulator receives the complete call.

Backpressure is inherited from the HTTP body stream. The gateway does not buffer a completed stream for logging or conversion. Once response headers are sent, failures use an inbound-native error event when one exists and otherwise terminate the stream.

## Out of scope

The first version does not include agents, RAG, vector stores, speech APIs, transcription, moderation, reranking, batch jobs, provider file-management endpoints, failover, load balancing, usage quotas, a database, configuration hot reload, or TLS termination.
