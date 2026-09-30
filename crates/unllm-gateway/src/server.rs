use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    future::IntoFuture,
    sync::Arc,
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Multipart, Path, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, header},
    response::Response,
    routing::{get, post},
};
use base64::Engine as _;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{Value, json};
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, time::timeout};
use tracing::{Instrument, info, info_span, warn};
use tracing_subscriber::EnvFilter;
use unllm::{
    AssetStager, Capabilities, Capability, ContentPart, ConversionContext, DynAdapter, ErrorKind,
    ImageTask, MediaResolver, Operation, Protocol, StreamEvent, UnifiedRequest, UnllmError,
};
use uuid::Uuid;

use crate::{
    Cli, Command, EXAMPLE_CONFIG, GatewayConfig, RouteConfig, ServeArgs, UpstreamConfig,
    config::is_forbidden_forward_header,
    media::{
        HttpMediaResolver, LeaseCleanupGuard, ProviderAssetStager, cleanup_leases, prepare_request,
        summarize_json,
    },
    sse::SseDecoder,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
struct AppState {
    config: Arc<GatewayConfig>,
    client: Client,
    adapters: Arc<BTreeMap<Protocol, Arc<dyn DynAdapter>>>,
    client_keys: Arc<Vec<String>>,
    media_resolver: Option<Arc<dyn MediaResolver>>,
    log_bodies: bool,
}

pub async fn run(cli: Cli) -> Result<(), BoxError> {
    match cli.command {
        Command::CheckConfig { path } => {
            let config = GatewayConfig::from_path(&path)?;
            println!(
                "Configuration is valid: {} upstream(s), {} route(s)",
                config.upstreams.len(),
                config.routes.len()
            );
            Ok(())
        }
        Command::GenerateConfig => {
            print!("{EXAMPLE_CONFIG}");
            Ok(())
        }
        Command::Serve(args) => serve(args).await,
    }
}

async fn serve(args: ServeArgs) -> Result<(), BoxError> {
    let mut config = GatewayConfig::quick(&args)?;
    if let Some(listen) = args.listen {
        config.server.listen = listen;
    }
    init_tracing(&config.server.log);
    if args.token.is_some() {
        warn!("A literal --token may be visible in shell history and the process list");
    }
    if args.log_bodies {
        warn!("Semantic body logging is enabled and may record sensitive prompt content");
    }
    let state = AppState::new(config, args.log_bodies)?;
    let listen = state.config.server.listen;
    let grace = Duration::from_millis(state.config.server.shutdown_grace_ms);
    let app = router(state.clone()).layer(DefaultBodyLimit::disable());
    let listener = TcpListener::bind(listen).await?;
    info!(%listen, "unllm gateway is listening");
    let notification = Arc::new(tokio::sync::Notify::new());
    let server_notification = notification.clone();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            server_notification.notified().await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result?,
        () = shutdown_signal() => {
            notification.notify_waiters();
            info!(grace_ms = grace.as_millis(), "Draining active requests");
            if timeout(grace, &mut server).await.is_err() {
                warn!("Graceful shutdown deadline expired; cancelling remaining requests");
            }
        }
    }
    Ok(())
}

impl AppState {
    fn new(config: GatewayConfig, log_bodies: bool) -> Result<Self, BoxError> {
        config.validate()?;
        let client = Client::builder()
            .connect_timeout(config.timeout_connect())
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let adapters: BTreeMap<Protocol, Arc<dyn DynAdapter>> = BTreeMap::from([
            (
                Protocol::OpenAiChatCompletions,
                Arc::new(unllm::openai::OpenAiChatAdapter) as Arc<dyn DynAdapter>,
            ),
            (
                Protocol::OpenAiResponses,
                Arc::new(unllm::openai::OpenAiResponsesAdapter) as Arc<dyn DynAdapter>,
            ),
            (
                Protocol::AnthropicMessages,
                Arc::new(unllm::anthropic::AnthropicAdapter) as Arc<dyn DynAdapter>,
            ),
            (
                Protocol::Gemini,
                Arc::new(unllm::gemini::GeminiAdapter) as Arc<dyn DynAdapter>,
            ),
        ]);
        for route in &config.routes {
            let upstream = config
                .upstream(&route.upstream)
                .ok_or_else(|| format!("Missing upstream {}", route.upstream))?;
            for operation in &route.operations {
                let protocol = if *operation == Operation::Generate {
                    route.generate_protocol.unwrap_or(upstream.protocol)
                } else {
                    upstream.protocol
                };
                let adapter = adapters
                    .get(&protocol)
                    .ok_or_else(|| format!("No adapter for {protocol:?}"))?;
                if !adapter.capabilities().operations.contains(operation) {
                    return Err(format!(
                        "Route {} declares {operation:?}, which is unsupported by {protocol:?}",
                        route.alias
                    )
                    .into());
                }
            }
        }
        let media_resolver = if config.media.enabled {
            Some(Arc::new(HttpMediaResolver::new(&config.media)?) as Arc<dyn MediaResolver>)
        } else {
            None
        };
        let client_keys = Arc::new(config.client_keys());
        Ok(Self {
            config: Arc::new(config),
            client,
            adapters: Arc::new(adapters),
            client_keys,
            media_resolver,
            log_bodies,
        })
    }

    fn adapter(&self, protocol: Protocol) -> Arc<dyn DynAdapter> {
        self.adapters
            .get(&protocol)
            .expect("all protocols are registered")
            .clone()
    }
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/images/generations", post(openai_images))
        .route("/v1/images/edits", post(openai_image_edits))
        .route("/v1/images/variations", post(openai_image_variations))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1beta/models", get(gemini_models))
        .route("/v1beta/models/{*method}", post(gemini_dispatch))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn ready() -> &'static str {
    "ready"
}

macro_rules! proxy_handler {
    ($name:ident, $protocol:expr, $operation:expr) => {
        async fn $name(
            State(state): State<AppState>,
            headers: HeaderMap,
            uri: Uri,
            body: Bytes,
        ) -> Response {
            proxy(state, headers, uri, body, $protocol, $operation, None, None).await
        }
    };
}

proxy_handler!(
    openai_chat,
    Protocol::OpenAiChatCompletions,
    Operation::Generate
);

async fn openai_image_edits(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    multipart: Multipart,
) -> Response {
    openai_multipart_image(state, headers, uri, multipart, "edit").await
}

async fn openai_image_variations(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    multipart: Multipart,
) -> Response {
    openai_multipart_image(state, headers, uri, multipart, "variation").await
}

async fn openai_multipart_image(
    state: AppState,
    headers: HeaderMap,
    uri: Uri,
    mut multipart: Multipart,
    task: &'static str,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let mut object = serde_json::Map::new();
    let mut image = None;
    let mut mask = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => {
                return error_response(
                    Protocol::OpenAiChatCompletions,
                    StatusCode::BAD_REQUEST,
                    UnllmError::invalid("invalid_multipart", error.to_string()),
                    &request_id,
                );
            }
        };
        let name = field.name().unwrap_or_default().to_owned();
        let mime_type = field
            .content_type()
            .map(str::to_owned)
            .unwrap_or_else(|| "application/octet-stream".into());
        if name == "image" || name == "mask" {
            match field.bytes().await {
                Ok(bytes) => {
                    let value = json!({
                        "mime_type": mime_type,
                        "data": base64::engine::general_purpose::STANDARD.encode(bytes),
                    });
                    if name == "image" {
                        image = Some(value);
                    } else {
                        mask = Some(value);
                    }
                }
                Err(error) => {
                    return error_response(
                        Protocol::OpenAiChatCompletions,
                        StatusCode::BAD_REQUEST,
                        UnllmError::invalid("invalid_multipart", error.to_string()),
                        &request_id,
                    );
                }
            }
        } else if let Ok(text) = field.text().await {
            let value = match name.as_str() {
                "n" => text.parse::<u64>().map_or(Value::String(text), Value::from),
                "stream" => text
                    .parse::<bool>()
                    .map_or(Value::String(text), Value::Bool),
                _ => Value::String(text),
            };
            object.insert(name, value);
        }
    }
    object
        .entry("model")
        .or_insert_with(|| Value::String("dall-e-2".into()));
    object
        .entry("prompt")
        .or_insert_with(|| Value::String(String::new()));
    object.insert("_unllm_image_task".into(), Value::String(task.into()));
    if let Some(image) = image {
        object.insert("_unllm_image".into(), image);
    }
    if let Some(mask) = mask {
        object.insert("_unllm_mask".into(), mask);
    }
    proxy(
        state,
        headers,
        uri,
        Bytes::from(Value::Object(object).to_string()),
        Protocol::OpenAiChatCompletions,
        Operation::Image,
        None,
        Some(false),
    )
    .await
}
proxy_handler!(
    openai_responses,
    Protocol::OpenAiResponses,
    Operation::Generate
);
proxy_handler!(
    openai_embeddings,
    Protocol::OpenAiChatCompletions,
    Operation::Embed
);
proxy_handler!(
    openai_images,
    Protocol::OpenAiChatCompletions,
    Operation::Image
);
proxy_handler!(
    anthropic_messages,
    Protocol::AnthropicMessages,
    Operation::Generate
);

async fn gemini_dispatch(
    State(state): State<AppState>,
    Path(method): Path<String>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Response {
    let Some((model, method)) = method.rsplit_once(':') else {
        return error_response(
            Protocol::Gemini,
            StatusCode::NOT_FOUND,
            UnllmError::invalid("gemini_method", "Gemini model method is missing"),
            &Uuid::new_v4().to_string(),
        );
    };
    let (operation, stream) = match method {
        "generateContent" => (Operation::Generate, false),
        "streamGenerateContent" => (Operation::Generate, true),
        "embedContent" | "batchEmbedContents" => (Operation::Embed, false),
        _ => {
            return error_response(
                Protocol::Gemini,
                StatusCode::NOT_FOUND,
                UnllmError::invalid(
                    "gemini_method",
                    format!("Unsupported Gemini model method: {method}"),
                ),
                &Uuid::new_v4().to_string(),
            );
        }
    };
    proxy(
        state,
        headers,
        uri,
        body,
        Protocol::Gemini,
        operation,
        Some(model.to_owned()),
        Some(stream),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn proxy(
    state: AppState,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
    source_protocol: Protocol,
    operation: Operation,
    path_model: Option<String>,
    forced_stream: Option<bool>,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let span = info_span!(
        "gateway_request",
        request_id = %request_id,
        client_request_id = tracing::field::Empty,
        source_protocol = ?source_protocol,
        operation = ?operation,
        model = tracing::field::Empty,
        route = tracing::field::Empty,
        upstream = tracing::field::Empty,
    );
    async move {
        if let Some(client_id) = safe_client_request_id(&headers) {
            tracing::Span::current().record("client_request_id", client_id);
        }
        if !authorized(&state.client_keys, &headers, &uri) {
            return error_response(
                source_protocol,
                StatusCode::UNAUTHORIZED,
                UnllmError {
                    code: "authentication_error".into(),
                    kind: ErrorKind::Authentication,
                    message: "Invalid gateway API key".into(),
                    source_protocol: Some(source_protocol),
                    target: None,
                    operation: Some(operation),
                    metadata: None,
                },
                &request_id,
            );
        }
        let body_value: Value = match serde_json::from_slice(&body) {
            Ok(value) => value,
            Err(error) => {
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    UnllmError::invalid("invalid_json", error.to_string()),
                    &request_id,
                );
            }
        };
        if state.log_bodies {
            info!(body = %summarize_json(&body_value), "Decoded inbound request body");
        }
        let source_adapter = state.adapter(source_protocol);
        let source_context = ConversionContext {
            source: source_protocol,
            target: source_protocol,
            mode: unllm::ConversionMode::Strict,
            capabilities: source_adapter.capabilities(),
        };
        let (mut canonical, mut request_warnings) =
            match source_adapter.decode_request(operation, body_value, &source_context) {
                Ok(outcome) => (
                    outcome.value,
                    outcome
                        .diagnostics
                        .into_iter()
                        .map(|diagnostic| diagnostic.code)
                        .collect::<Vec<_>>(),
                ),
                Err(error) => {
                    return error_response(
                        source_protocol,
                        StatusCode::BAD_REQUEST,
                        error,
                        &request_id,
                    );
                }
            };
        if let Some(model) = path_model {
            canonical.set_model(model);
        }
        set_stream(&mut canonical, forced_stream);
        if let Err(error) = canonical.validate() {
            return error_response(source_protocol, StatusCode::BAD_REQUEST, error, &request_id);
        }
        let requested_model = canonical.model().to_owned();
        tracing::Span::current().record("model", requested_model.as_str());
        let route = match state.config.resolve_route(&requested_model, operation) {
            Ok(route) => route.clone(),
            Err(message) => {
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    UnllmError::invalid("route_not_found", message),
                    &request_id,
                );
            }
        };
        tracing::Span::current().record("route", route.alias.as_str());
        let upstream = state
            .config
            .upstream(&route.upstream)
            .expect("configuration was validated")
            .clone();
        tracing::Span::current().record("upstream", upstream.name.as_str());
        let target_protocol = if operation == Operation::Generate {
            route.generate_protocol.unwrap_or(upstream.protocol)
        } else {
            upstream.protocol
        };
        let target_adapter = state.adapter(target_protocol);
        let capabilities = route.capabilities(&target_adapter.capabilities());
        if let Err(error) = preflight(&canonical, &capabilities, target_protocol) {
            return error_response(source_protocol, StatusCode::BAD_REQUEST, error, &request_id);
        }
        canonical.set_model(route.remote_model.clone());
        let stager: Option<Arc<dyn AssetStager>> = route.stage_files.then(|| {
            Arc::new(ProviderAssetStager::new(
                state.client.clone(),
                upstream.clone(),
            )) as Arc<dyn AssetStager>
        });
        let leases = match prepare_request(
            &mut canonical,
            state.media_resolver.as_deref(),
            stager.clone(),
            target_protocol.namespace(),
            route.resolve_media,
            route.stage_files,
        )
        .await
        {
            Ok(leases) => leases,
            Err(error) => {
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    error,
                    &request_id,
                );
            }
        };
        let conversion_context = ConversionContext {
            source: source_protocol,
            target: target_protocol,
            mode: route.mode,
            capabilities,
        };
        let encoded = match target_adapter.encode_request(&canonical, &conversion_context) {
            Ok(outcome) => outcome,
            Err(error) => {
                cleanup_if_needed(stager, leases).await;
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    error,
                    &request_id,
                );
            }
        };
        let mut upstream_body = encoded.value;
        if let Some(additional) = route.additional_body.get(&operation) {
            if let Err(error) = merge_additional_body(&mut upstream_body, additional) {
                cleanup_if_needed(stager, leases).await;
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    error,
                    &request_id,
                );
            }
        }
        if state.log_bodies {
            info!(body = %summarize_json(&upstream_body), "Encoded upstream request body");
        }
        let upstream_url = match upstream_url(&upstream, &route, &canonical, target_protocol) {
            Ok(url) => url,
            Err(error) => {
                cleanup_if_needed(stager, leases).await;
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    error,
                    &request_id,
                );
            }
        };
        let builder = match upstream_request(
            &state,
            &upstream,
            &route,
            target_protocol,
            upstream_url,
            &headers,
            upstream_body,
        ) {
            Ok(builder) => builder,
            Err(error) => {
                cleanup_if_needed(stager, leases).await;
                return error_response(
                    source_protocol,
                    StatusCode::BAD_REQUEST,
                    error,
                    &request_id,
                );
            }
        };
        let response = match timeout(
            Duration::from_millis(state.config.server.headers_timeout_ms),
            builder.send(),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                cleanup_if_needed(stager, leases).await;
                return error_response(
                    source_protocol,
                    StatusCode::BAD_GATEWAY,
                    upstream_transport_error(error),
                    &request_id,
                );
            }
            Err(_) => {
                cleanup_if_needed(stager, leases).await;
                return error_response(
                    source_protocol,
                    StatusCode::GATEWAY_TIMEOUT,
                    timeout_error("upstream_headers_timeout"),
                    &request_id,
                );
            }
        };
        if !response.status().is_success() {
            let status = response.status();
            let error = normalize_upstream_error(status, response.bytes().await.ok());
            cleanup_if_needed(stager, leases).await;
            return error_response(source_protocol, map_status(&error), error, &request_id);
        }
        request_warnings.extend(
            encoded
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.code.clone()),
        );
        let warnings = request_warnings;
        if canonical.is_streaming() {
            stream_response(
                state,
                response,
                source_protocol,
                target_protocol,
                source_adapter,
                target_adapter,
                conversion_context,
                request_id,
                stager,
                leases,
                warnings,
            )
        } else {
            let response = non_stream_response(
                &state,
                response,
                operation,
                source_protocol,
                target_protocol,
                source_adapter,
                target_adapter,
                route,
                request_id.clone(),
                warnings,
            )
            .await;
            cleanup_if_needed(stager, leases).await;
            response
        }
    }
    .instrument(span)
    .await
}

#[allow(clippy::too_many_arguments)]
async fn non_stream_response(
    state: &AppState,
    response: reqwest::Response,
    operation: Operation,
    source_protocol: Protocol,
    target_protocol: Protocol,
    source_adapter: Arc<dyn DynAdapter>,
    target_adapter: Arc<dyn DynAdapter>,
    route: RouteConfig,
    request_id: String,
    mut warning_codes: Vec<String>,
) -> Response {
    let bytes = match timeout(
        Duration::from_millis(state.config.server.non_stream_timeout_ms),
        response.bytes(),
    )
    .await
    {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(error)) => {
            return error_response(
                source_protocol,
                StatusCode::BAD_GATEWAY,
                upstream_transport_error(error),
                &request_id,
            );
        }
        Err(_) => {
            return error_response(
                source_protocol,
                StatusCode::GATEWAY_TIMEOUT,
                timeout_error("upstream_body_timeout"),
                &request_id,
            );
        }
    };
    let upstream_body: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return error_response(
                source_protocol,
                StatusCode::BAD_GATEWAY,
                UnllmError {
                    code: "upstream_invalid_json".into(),
                    kind: ErrorKind::Upstream,
                    message: format!("Upstream returned invalid JSON: {error}"),
                    source_protocol: Some(target_protocol),
                    target: Some(source_protocol),
                    operation: Some(operation),
                    metadata: None,
                },
                &request_id,
            );
        }
    };
    if state.log_bodies {
        info!(body = %summarize_json(&upstream_body), "Decoded upstream response body");
    }
    let context = ConversionContext {
        source: target_protocol,
        target: source_protocol,
        mode: route.mode,
        capabilities: source_adapter.capabilities(),
    };
    let decoded = match target_adapter.decode_response(operation, upstream_body, &context) {
        Ok(outcome) => outcome,
        Err(error) => {
            return error_response(source_protocol, StatusCode::BAD_GATEWAY, error, &request_id);
        }
    };
    warning_codes.extend(decoded.diagnostics.iter().map(|item| item.code.clone()));
    let encoded = match source_adapter.encode_response(&decoded.value, &context) {
        Ok(outcome) => outcome,
        Err(error) => {
            return error_response(source_protocol, StatusCode::BAD_GATEWAY, error, &request_id);
        }
    };
    warning_codes.extend(encoded.diagnostics.iter().map(|item| item.code.clone()));
    if state.log_bodies {
        info!(body = %summarize_json(&encoded.value), "Encoded outbound response body");
    }
    json_response(StatusCode::OK, encoded.value, &request_id, &warning_codes)
}

#[allow(clippy::too_many_arguments)]
fn stream_response(
    state: AppState,
    response: reqwest::Response,
    source_protocol: Protocol,
    target_protocol: Protocol,
    source_adapter: Arc<dyn DynAdapter>,
    target_adapter: Arc<dyn DynAdapter>,
    mut context: ConversionContext,
    request_id: String,
    stager: Option<Arc<dyn AssetStager>>,
    leases: Vec<unllm::AssetLease>,
    warning_codes: Vec<String>,
) -> Response {
    context.source = target_protocol;
    context.target = source_protocol;
    context.capabilities = source_adapter.capabilities();
    let idle = Duration::from_millis(state.config.server.stream_idle_timeout_ms);
    let log_bodies = state.log_bodies;
    let stream_request_id = request_id.clone();
    let output = async_stream::stream! {
        let _cleanup_guard = stager.map(|stager| LeaseCleanupGuard::new(stager, leases));
        let mut decoder = SseDecoder::default();
        let mut upstream_stream = response.bytes_stream();
        'outer: loop {
            let chunk = match timeout(idle, upstream_stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(error))) => {
                    let event = StreamEvent::Error { error: upstream_transport_error(error) };
                    if let Ok(encoded) = source_adapter.encode_stream_event(&event, &context) {
                        for bytes in encoded.value {
                            yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                        }
                    }
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    let event = StreamEvent::Error { error: timeout_error("upstream_stream_idle_timeout") };
                    if let Ok(encoded) = source_adapter.encode_stream_event(&event, &context) {
                        for bytes in encoded.value {
                            yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                        }
                    }
                    break;
                }
            };
            for message in decoder.push(&chunk) {
                let decoded = match target_adapter.decode_stream_event(
                    message.event_type.as_deref(),
                    &message.data,
                    &context,
                ) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        let event = StreamEvent::Error { error };
                        if let Ok(encoded) = source_adapter.encode_stream_event(&event, &context) {
                            for bytes in encoded.value {
                                yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                            }
                        }
                        break 'outer;
                    }
                };
                for diagnostic in &decoded.diagnostics {
                    warn!(
                        request_id = %stream_request_id,
                        code = %diagnostic.code,
                        path = %diagnostic.path,
                        "Lossy stream conversion"
                    );
                }
                for event in decoded.value {
                    if log_bodies {
                        info!(
                            event = %summarize_json(&serde_json::to_value(&event).unwrap_or(serde_json::Value::Null)),
                            "Streaming canonical event"
                        );
                    }
                    match source_adapter.encode_stream_event(&event, &context) {
                        Ok(encoded) => {
                            for diagnostic in &encoded.diagnostics {
                                warn!(
                                    request_id = %stream_request_id,
                                    code = %diagnostic.code,
                                    path = %diagnostic.path,
                                    "Lossy stream conversion"
                                );
                            }
                            for bytes in encoded.value {
                                yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                            }
                        }
                        Err(error) => {
                            let event = StreamEvent::Error { error };
                            if let Ok(encoded) = source_adapter.encode_stream_event(&event, &context) {
                                for bytes in encoded.value {
                                    yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                                }
                            }
                            break 'outer;
                        }
                    }
                }
            }
        }
        if let Some(message) = decoder.finish() {
            if let Ok(decoded) = target_adapter.decode_stream_event(
                message.event_type.as_deref(),
                &message.data,
                &context,
            ) {
                for event in decoded.value {
                    if let Ok(encoded) = source_adapter.encode_stream_event(&event, &context) {
                        for bytes in encoded.value {
                            yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                        }
                    }
                }
            }
        }
        if let Some(terminator) = stream_terminator(source_protocol) {
            yield Ok::<Bytes, Infallible>(Bytes::from_static(terminator));
        }
    };
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-request-id", &request_id);
    if !warning_codes.is_empty() {
        builder = builder.header("x-unllm-warnings", warning_codes.join(","));
    }
    builder
        .body(Body::from_stream(output))
        .expect("valid streaming response")
}

async fn openai_models(State(state): State<AppState>, headers: HeaderMap, uri: Uri) -> Response {
    let protocol =
        if headers.contains_key("x-api-key") && !headers.contains_key(header::AUTHORIZATION) {
            Protocol::AnthropicMessages
        } else {
            Protocol::OpenAiChatCompletions
        };
    models(state, headers, uri, protocol).await
}

async fn gemini_models(State(state): State<AppState>, headers: HeaderMap, uri: Uri) -> Response {
    models(state, headers, uri, Protocol::Gemini).await
}

async fn models(state: AppState, headers: HeaderMap, uri: Uri, protocol: Protocol) -> Response {
    let request_id = Uuid::new_v4().to_string();
    if !authorized(&state.client_keys, &headers, &uri) {
        return error_response(
            protocol,
            StatusCode::UNAUTHORIZED,
            UnllmError::invalid("authentication_error", "Invalid gateway API key"),
            &request_id,
        );
    }
    let value = if protocol == Protocol::Gemini {
        json!({
            "models": state.config.routes.iter().map(|route| json!({
                "name": format!("models/{}", route.alias),
                "displayName": route.alias,
                "supportedGenerationMethods": route.operations.iter().map(|operation| match operation {
                    Operation::Generate | Operation::Image => "generateContent",
                    Operation::Embed => "embedContent",
                }).collect::<Vec<_>>()
            })).collect::<Vec<_>>()
        })
    } else if protocol == Protocol::AnthropicMessages {
        let data = state
            .config
            .routes
            .iter()
            .filter(|route| route.operations.contains(&Operation::Generate))
            .map(|route| {
                json!({
                    "id": route.alias,
                    "type": "model",
                    "display_name": route.alias,
                    "created_at": "1970-01-01T00:00:00Z"
                })
            })
            .collect::<Vec<_>>();
        json!({
            "data": data,
            "has_more": false,
            "first_id": state.config.routes.first().map(|route| &route.alias),
            "last_id": state.config.routes.last().map(|route| &route.alias),
        })
    } else {
        json!({
            "object": "list",
            "data": state.config.routes.iter().map(|route| json!({
                "id": route.alias,
                "object": "model",
                "owned_by": "unllm",
                "capabilities": route.capabilities,
            })).collect::<Vec<_>>()
        })
    };
    json_response(StatusCode::OK, value, &request_id, &[])
}

fn upstream_request(
    state: &AppState,
    upstream: &UpstreamConfig,
    route: &RouteConfig,
    protocol: Protocol,
    mut url: url::Url,
    inbound_headers: &HeaderMap,
    body: Value,
) -> Result<reqwest::RequestBuilder, UnllmError> {
    for (key, value) in upstream.query.iter().chain(route.query.iter()) {
        url.query_pairs_mut()
            .append_pair(key, &value.resolve().map_err(config_error)?);
    }
    let token = upstream.token().map_err(config_error)?;
    let mut body = body;
    let multipart = if matches!(
        protocol,
        Protocol::OpenAiChatCompletions | Protocol::OpenAiResponses
    ) && body.get("_unllm_multipart").is_some()
    {
        Some(openai_multipart_form(&mut body)?)
    } else {
        None
    };
    let mut builder = if let Some(multipart) = multipart {
        state.client.post(url).multipart(multipart)
    } else {
        state.client.post(url).json(&body)
    };
    match protocol {
        Protocol::OpenAiChatCompletions | Protocol::OpenAiResponses => {
            if let Some(token) = token {
                builder = builder.bearer_auth(token);
            }
            if !route.beta_features.is_empty() {
                builder = builder.header("openai-beta", route.beta_features.join(","));
            }
        }
        Protocol::AnthropicMessages => {
            if let Some(token) = token {
                builder = builder.header("x-api-key", token);
            }
            builder = builder.header("anthropic-version", "2023-06-01");
            if !route.beta_features.is_empty() {
                builder = builder.header("anthropic-beta", route.beta_features.join(","));
            }
        }
        Protocol::Gemini => {
            if let Some(token) = token {
                builder = builder.header("x-goog-api-key", token);
            }
        }
    }
    for (key, value) in upstream.headers.iter().chain(route.headers.iter()) {
        if is_protected_upstream_header(key) {
            return Err(UnllmError::invalid(
                "protected_header",
                format!("Configuration cannot override protected header {key}"),
            ));
        }
        let name = HeaderName::try_from(key.as_str())
            .map_err(|error| UnllmError::invalid("invalid_header", error.to_string()))?;
        let value = HeaderValue::try_from(value.resolve().map_err(config_error)?)
            .map_err(|error| UnllmError::invalid("invalid_header", error.to_string()))?;
        builder = builder.header(name, value);
    }
    for name in &route.forward_headers {
        if is_forbidden_forward_header(name) {
            continue;
        }
        if let Some(value) = inbound_headers.get(name) {
            builder = builder.header(name, value);
        }
    }
    Ok(builder)
}

fn openai_multipart_form(body: &mut Value) -> Result<reqwest::multipart::Form, UnllmError> {
    use base64::Engine as _;

    let object = body.as_object_mut().ok_or_else(|| {
        UnllmError::invalid(
            "openai_multipart",
            "Multipart request body must be an object",
        )
    })?;
    let multipart = object
        .remove("_unllm_multipart")
        .and_then(|value| value.as_object().cloned())
        .ok_or_else(|| UnllmError::invalid("openai_multipart", "Multipart metadata is missing"))?;
    let mut form = reqwest::multipart::Form::new();
    for (name, value) in object.iter() {
        if value.is_null() {
            continue;
        }
        let value = value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string());
        form = form.text(name.clone(), value);
    }
    for name in ["image", "mask"] {
        let Some(asset) = multipart.get(name) else {
            continue;
        };
        if asset.is_null() {
            continue;
        }
        let mime_type = asset
            .get("mime_type")
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream");
        let data = asset.get("data").and_then(Value::as_str).ok_or_else(|| {
            UnllmError::invalid("openai_multipart", format!("{name} data is missing"))
        })?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|error| UnllmError::invalid("openai_multipart", error.to_string()))?;
        let part = reqwest::multipart::Part::bytes(data)
            .file_name(format!("{name}.bin"))
            .mime_str(mime_type)
            .map_err(|error| UnllmError::invalid("openai_multipart", error.to_string()))?;
        form = form.part(name.to_owned(), part);
    }
    Ok(form)
}

fn upstream_url(
    upstream: &UpstreamConfig,
    route: &RouteConfig,
    request: &UnifiedRequest,
    protocol: Protocol,
) -> Result<url::Url, UnllmError> {
    let operation = request.operation();
    let path = route
        .paths
        .get(&operation)
        .cloned()
        .unwrap_or_else(|| default_path(protocol, request));
    let mut base = upstream.base_url.clone();
    if !base.path().ends_with('/') {
        base.set_path(&format!("{}/", base.path()));
    }
    let mut url = base
        .join(path.trim_start_matches('/'))
        .map_err(|error| UnllmError::invalid("upstream_url", error.to_string()))?;
    if protocol == Protocol::Gemini && request.is_streaming() {
        url.query_pairs_mut().append_pair("alt", "sse");
    }
    Ok(url)
}

fn default_path(protocol: Protocol, request: &UnifiedRequest) -> String {
    match (protocol, request) {
        (Protocol::OpenAiResponses, UnifiedRequest::Generate(_)) => "v1/responses".into(),
        (Protocol::OpenAiChatCompletions, UnifiedRequest::Generate(_)) => {
            "v1/chat/completions".into()
        }
        (Protocol::OpenAiChatCompletions | Protocol::OpenAiResponses, UnifiedRequest::Embed(_)) => {
            "v1/embeddings".into()
        }
        (
            Protocol::OpenAiChatCompletions | Protocol::OpenAiResponses,
            UnifiedRequest::Image(request),
        ) => match request.task {
            ImageTask::Generate => "v1/images/generations".into(),
            ImageTask::Edit { .. } => "v1/images/edits".into(),
            ImageTask::Variation { .. } => "v1/images/variations".into(),
        },
        (Protocol::AnthropicMessages, _) => "v1/messages".into(),
        (Protocol::Gemini, UnifiedRequest::Generate(request)) => format!(
            "v1beta/models/{}:{}",
            request.model,
            if request.stream {
                "streamGenerateContent"
            } else {
                "generateContent"
            }
        ),
        (Protocol::Gemini, UnifiedRequest::Embed(request)) => format!(
            "v1beta/models/{}:{}",
            request.model,
            if request.inputs.len() > 1 {
                "batchEmbedContents"
            } else {
                "embedContent"
            }
        ),
        (Protocol::Gemini, UnifiedRequest::Image(request)) => {
            format!("v1beta/models/{}:generateContent", request.model)
        }
    }
}

fn preflight(
    request: &UnifiedRequest,
    capabilities: &Capabilities,
    target: Protocol,
) -> Result<(), UnllmError> {
    if !capabilities.operations.contains(&request.operation()) {
        return Err(UnllmError::unsupported(
            "operation_not_supported",
            format!("The route does not support {:?}", request.operation()),
        ));
    }
    let required = required_capabilities(request);
    if let Some(missing) = required
        .iter()
        .find(|capability| !capabilities.features.contains(capability))
    {
        let mut error = UnllmError::unsupported(
            "capability_not_supported",
            format!("The route does not support {missing:?}"),
        );
        error.target = Some(target);
        error.operation = Some(request.operation());
        return Err(error);
    }
    Ok(())
}

fn required_capabilities(request: &UnifiedRequest) -> BTreeSet<Capability> {
    let mut output = BTreeSet::new();
    match request {
        UnifiedRequest::Generate(request) => {
            if request.stream {
                output.insert(Capability::Streaming);
            }
            if !request.tools.is_empty() {
                output.insert(Capability::FunctionTools);
            }
            if request.response_format.is_some() {
                output.insert(Capability::StructuredOutput);
            }
            if request.reasoning.is_some() {
                output.insert(Capability::Reasoning);
            }
            if request.parameters.candidate_count.unwrap_or(1) > 1 {
                output.insert(Capability::MultipleCandidates);
            }
            for part in request
                .instructions
                .iter()
                .flat_map(|instruction| instruction.content.iter())
                .chain(
                    request
                        .messages
                        .iter()
                        .flat_map(|message| message.content.iter()),
                )
            {
                collect_part_capabilities(part, &mut output);
            }
        }
        UnifiedRequest::Embed(_) => {
            output.insert(Capability::Embeddings);
        }
        UnifiedRequest::Image(request) => {
            output.insert(Capability::Image);
            match request.task {
                ImageTask::Generate => {}
                ImageTask::Edit { .. } => {
                    output.insert(Capability::ImageEdit);
                }
                ImageTask::Variation { .. } => {
                    output.insert(Capability::ImageVariation);
                }
            }
        }
    }
    output
}

fn collect_part_capabilities(part: &ContentPart, output: &mut BTreeSet<Capability>) {
    match part {
        ContentPart::Text { .. } => {
            output.insert(Capability::Text);
        }
        ContentPart::Image { .. } => {
            output.insert(Capability::Image);
        }
        ContentPart::Audio { .. } => {
            output.insert(Capability::Audio);
        }
        ContentPart::Video { .. } => {
            output.insert(Capability::Video);
        }
        ContentPart::File { .. } => {
            output.insert(Capability::File);
        }
        ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } => {
            output.insert(Capability::FunctionTools);
        }
        ContentPart::Reasoning { .. } => {
            output.insert(Capability::Reasoning);
        }
        ContentPart::Refusal { .. } => {}
    }
}

fn merge_additional_body(target: &mut Value, additional: &Value) -> Result<(), UnllmError> {
    const PROTECTED: &[&str] = &[
        "model",
        "messages",
        "input",
        "contents",
        "stream",
        "tools",
        "system",
        "systemInstruction",
    ];
    let target = target.as_object_mut().ok_or_else(|| {
        UnllmError::invalid(
            "upstream_body",
            "Encoded upstream body is not a JSON object",
        )
    })?;
    let additional = additional.as_object().ok_or_else(|| {
        UnllmError::invalid("additional_body", "additional_body must be a JSON object")
    })?;
    for (key, value) in additional {
        if PROTECTED.contains(&key.as_str()) && target.contains_key(key) {
            return Err(UnllmError::invalid(
                "protected_body_field",
                format!("additional_body cannot override protected field {key}"),
            ));
        }
        match target.get_mut(key) {
            Some(existing) => fill_missing(existing, value),
            None => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(())
}

fn fill_missing(target: &mut Value, additional: &Value) {
    if let (Value::Object(target), Value::Object(additional)) = (target, additional) {
        for (key, value) in additional {
            match target.get_mut(key) {
                Some(existing) => fill_missing(existing, value),
                None => {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
    }
}

fn set_stream(request: &mut UnifiedRequest, forced: Option<bool>) {
    let Some(stream) = forced else {
        return;
    };
    match request {
        UnifiedRequest::Generate(request) => request.stream = stream,
        UnifiedRequest::Image(request) => request.stream = stream,
        UnifiedRequest::Embed(_) => {}
    }
}

fn safe_client_request_id(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            value.len() <= 128 && value.chars().all(|character| !character.is_control())
        })
}

fn authorized(keys: &[String], headers: &HeaderMap, uri: &Uri) -> bool {
    if keys.is_empty() {
        return true;
    }
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .or_else(|| {
            headers
                .get("x-goog-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .map(str::to_owned)
        .or_else(|| {
            uri.query().and_then(|query| {
                url::form_urlencoded::parse(query.as_bytes())
                    .find(|(key, _)| key == "key")
                    .map(|(_, value)| value.into_owned())
            })
        });
    presented.is_some_and(|presented| {
        keys.iter()
            .any(|expected| constant_time_equal(expected.as_bytes(), presented.as_bytes()))
    })
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && bool::from(left.ct_eq(right))
}

fn normalize_upstream_error(status: reqwest::StatusCode, body: Option<Bytes>) -> UnllmError {
    let body = body
        .as_ref()
        .and_then(|body| serde_json::from_slice::<Value>(body).ok());
    let message = body
        .as_ref()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
        })
        .unwrap_or("Upstream request failed")
        .to_owned();
    let kind = match status.as_u16() {
        401 | 403 => ErrorKind::Authentication,
        400 | 404 | 409 | 422 => ErrorKind::InvalidRequest,
        429 => ErrorKind::RateLimit,
        408 | 504 => ErrorKind::Timeout,
        _ => ErrorKind::Upstream,
    };
    UnllmError {
        code: "upstream_error".into(),
        kind,
        message,
        source_protocol: None,
        target: None,
        operation: None,
        metadata: None,
    }
}

fn upstream_transport_error(error: reqwest::Error) -> UnllmError {
    UnllmError {
        code: "upstream_transport".into(),
        kind: if error.is_timeout() {
            ErrorKind::Timeout
        } else {
            ErrorKind::Upstream
        },
        message: "The upstream transport failed".into(),
        source_protocol: None,
        target: None,
        operation: None,
        metadata: None,
    }
}

fn timeout_error(code: &str) -> UnllmError {
    UnllmError {
        code: code.into(),
        kind: ErrorKind::Timeout,
        message: "The upstream request timed out".into(),
        source_protocol: None,
        target: None,
        operation: None,
        metadata: None,
    }
}

fn error_response(
    protocol: Protocol,
    status: StatusCode,
    error: UnllmError,
    request_id: &str,
) -> Response {
    warn!(request_id, code = %error.code, kind = ?error.kind, "Gateway request failed");
    let value = match protocol {
        Protocol::AnthropicMessages => json!({
            "type": "error",
            "error": {"type": error.code, "message": error.message},
            "request_id": request_id,
        }),
        Protocol::Gemini => json!({
            "error": {"code": status.as_u16(), "message": error.message, "status": error.code},
        }),
        Protocol::OpenAiChatCompletions | Protocol::OpenAiResponses => json!({
            "error": {"message": error.message, "type": format!("{:?}", error.kind).to_lowercase(), "code": error.code},
        }),
    };
    json_response(status, value, request_id, &[])
}

fn json_response(
    status: StatusCode,
    value: Value,
    request_id: &str,
    warnings: &[String],
) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-request-id", request_id);
    if !warnings.is_empty() {
        builder = builder.header("x-unllm-warnings", warnings.join(","));
    }
    builder
        .body(Body::from(value.to_string()))
        .expect("valid JSON response")
}

fn map_status(error: &UnllmError) -> StatusCode {
    match error.kind {
        ErrorKind::Authentication => StatusCode::UNAUTHORIZED,
        ErrorKind::RateLimit => StatusCode::TOO_MANY_REQUESTS,
        ErrorKind::InvalidRequest | ErrorKind::Unsupported => StatusCode::BAD_REQUEST,
        ErrorKind::Timeout => StatusCode::GATEWAY_TIMEOUT,
        ErrorKind::Cancelled => StatusCode::REQUEST_TIMEOUT,
        ErrorKind::Upstream | ErrorKind::Internal => StatusCode::BAD_GATEWAY,
        _ => StatusCode::BAD_GATEWAY,
    }
}

fn is_protected_upstream_header(header: &str) -> bool {
    matches!(
        header.to_ascii_lowercase().as_str(),
        "authorization"
            | "x-api-key"
            | "x-goog-api-key"
            | "host"
            | "content-length"
            | "content-type"
            | "connection"
            | "transfer-encoding"
    )
}

fn stream_terminator(protocol: Protocol) -> Option<&'static [u8]> {
    match protocol {
        Protocol::OpenAiChatCompletions => Some(b"data: [DONE]\n\n"),
        Protocol::AnthropicMessages => {
            Some(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
        }
        Protocol::OpenAiResponses | Protocol::Gemini => None,
    }
}

async fn cleanup_if_needed(stager: Option<Arc<dyn AssetStager>>, leases: Vec<unllm::AssetLease>) {
    if let Some(stager) = stager {
        cleanup_leases(stager, leases).await;
    }
}

fn config_error(message: String) -> UnllmError {
    UnllmError::invalid("gateway_configuration", message)
}

fn init_tracing(default_filter: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .try_init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(%error, "Failed to install Ctrl-C handler");
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => warn!(%error, "Failed to install SIGTERM handler"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    info!("Shutdown requested; draining active requests");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protected_body_fields_cannot_be_overridden() {
        let mut body = json!({"model": "m", "messages": []});
        let error = merge_additional_body(&mut body, &json!({"model": "other"})).unwrap_err();
        assert_eq!(error.code, "protected_body_field");
    }

    #[test]
    fn constant_time_key_comparison_checks_length() {
        assert!(constant_time_equal(b"secret", b"secret"));
        assert!(!constant_time_equal(b"secret", b"other"));
        assert!(!constant_time_equal(b"secret", b"secret-long"));
    }
}
