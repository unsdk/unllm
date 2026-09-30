use std::{collections::BTreeSet, net::IpAddr, sync::Arc};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::future::BoxFuture;
use reqwest::{Client, StatusCode, header};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::net::lookup_host;
use unllm::{
    AssetLease, AssetStager, ContentPart, EmbeddingInput, ErrorKind, ImageTask, MediaAsset,
    MediaResolver, MediaSource, StagingRequest, UnifiedRequest, UnllmError,
};
use url::Url;

use crate::{MediaConfig, UpstreamConfig};

pub struct HttpMediaResolver {
    allowed_hosts: BTreeSet<String>,
}

impl HttpMediaResolver {
    pub fn new(config: &MediaConfig) -> Result<Self, String> {
        Ok(Self {
            allowed_hosts: config
                .allowed_hosts
                .iter()
                .map(|host| host.to_ascii_lowercase())
                .collect(),
        })
    }

    async fn validate_url(&self, url: &Url) -> Result<(String, std::net::SocketAddr), UnllmError> {
        if url.scheme() != "https" {
            return Err(UnllmError::invalid(
                "media_scheme",
                "The built-in media resolver only accepts HTTPS URLs",
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| UnllmError::invalid("media_host", "Media URL has no host"))?
            .to_ascii_lowercase();
        let allowed = self.allowed_hosts.iter().any(|pattern| {
            host == *pattern
                || pattern
                    .strip_prefix("*.")
                    .is_some_and(|suffix| host.ends_with(&format!(".{suffix}")))
        });
        if !allowed {
            return Err(UnllmError::invalid(
                "media_host_not_allowed",
                format!("Media host {host} is not in the allowlist"),
            ));
        }
        let port = url.port_or_known_default().unwrap_or(443);
        let addresses: Vec<_> = lookup_host((host.as_str(), port))
            .await
            .map_err(|error| {
                UnllmError::invalid("media_dns", format!("Media host lookup failed: {error}"))
            })?
            .collect();
        for address in &addresses {
            if is_private(address.ip()) {
                return Err(UnllmError::invalid(
                    "media_private_address",
                    "Media host resolves to a private or local address",
                ));
            }
        }
        let address = addresses.first().copied().ok_or_else(|| {
            UnllmError::invalid("media_dns", "Media host did not resolve to an address")
        })?;
        Ok((host, address))
    }
}

#[async_trait]
impl MediaResolver for HttpMediaResolver {
    async fn resolve(&self, asset: &MediaAsset) -> Result<MediaAsset, UnllmError> {
        let MediaSource::Url { url, mime_type } = &asset.source else {
            return Ok(asset.clone());
        };
        let mut current =
            Url::parse(url).map_err(|error| UnllmError::invalid("media_url", error.to_string()))?;
        for _ in 0..10 {
            let (host, address) = self.validate_url(&current).await?;
            let client = Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .resolve(&host, address)
                .build()
                .map_err(|error| UnllmError::invalid("media_client", error.to_string()))?;
            let response =
                client
                    .get(current.clone())
                    .send()
                    .await
                    .map_err(|error| UnllmError {
                        code: "media_fetch".into(),
                        kind: ErrorKind::Upstream,
                        message: format!("Media request failed: {error}"),
                        source_protocol: None,
                        target: None,
                        operation: None,
                        metadata: None,
                    })?;
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get(header::LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| {
                        UnllmError::invalid(
                            "media_redirect",
                            "Media redirect is missing a Location header",
                        )
                    })?;
                current = current
                    .join(location)
                    .map_err(|error| UnllmError::invalid("media_redirect", error.to_string()))?;
                continue;
            }
            if !response.status().is_success() {
                return Err(UnllmError {
                    code: "media_fetch_status".into(),
                    kind: ErrorKind::Upstream,
                    message: format!("Media server returned HTTP {}", response.status()),
                    source_protocol: None,
                    target: None,
                    operation: None,
                    metadata: None,
                });
            }
            let response_mime = response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .map(str::to_owned)
                .or_else(|| mime_type.clone())
                .unwrap_or_else(|| "application/octet-stream".into());
            let data = response.bytes().await.map_err(|error| UnllmError {
                code: "media_fetch_body".into(),
                kind: ErrorKind::Upstream,
                message: format!("Failed to read media response: {error}"),
                source_protocol: None,
                target: None,
                operation: None,
                metadata: None,
            })?;
            let mut resolved = MediaAsset::inline(response_mime, data.clone());
            resolved.sha256 = Some(format!("{:x}", Sha256::digest(&data)));
            return Ok(resolved);
        }
        Err(UnllmError::invalid(
            "media_redirect_loop",
            "Media request exceeded the redirect safety limit",
        ))
    }
}

pub struct ProviderAssetStager {
    client: Client,
    upstream: UpstreamConfig,
}

impl ProviderAssetStager {
    pub fn new(client: Client, upstream: UpstreamConfig) -> Self {
        Self { client, upstream }
    }

    fn token(&self) -> Result<Option<String>, UnllmError> {
        self.upstream
            .token()
            .map_err(|message| UnllmError::invalid("upstream_token", message))
    }

    fn url(&self, path: &str) -> Result<Url, UnllmError> {
        self.upstream
            .base_url
            .join(path.trim_start_matches('/'))
            .map_err(|error| UnllmError::invalid("asset_upload_url", error.to_string()))
    }
}

#[async_trait]
impl AssetStager for ProviderAssetStager {
    async fn stage(
        &self,
        request: &StagingRequest,
        asset: &MediaAsset,
    ) -> Result<AssetLease, UnllmError> {
        let MediaSource::Inline { mime_type, data } = &asset.source else {
            return Err(UnllmError::unsupported(
                "asset_stage_source",
                "Asset staging requires inline bytes",
            ));
        };
        match request.provider.as_str() {
            "openai" | "anthropic" => {
                let url = self.url("v1/files")?;
                let filename = format!("unllm-{}.bin", uuid::Uuid::new_v4());
                let part = reqwest::multipart::Part::bytes(data.to_vec())
                    .file_name(filename)
                    .mime_str(mime_type)
                    .map_err(|error| UnllmError::invalid("asset_mime", error.to_string()))?;
                let mut form = reqwest::multipart::Form::new().part("file", part);
                if request.provider == "openai" {
                    form = form.text(
                        "purpose",
                        request
                            .purpose
                            .clone()
                            .unwrap_or_else(|| "assistants".into()),
                    );
                }
                let mut builder = self.client.post(url).multipart(form);
                if let Some(token) = self.token()? {
                    if request.provider == "anthropic" {
                        builder = builder.header("x-api-key", token);
                    } else {
                        builder = builder.bearer_auth(token);
                    }
                }
                if request.provider == "anthropic" {
                    builder = builder
                        .header("anthropic-version", "2023-06-01")
                        .header("anthropic-beta", "files-api-2025-04-14");
                }
                let response = builder.send().await.map_err(upstream_error)?;
                let status = response.status();
                let body: Value = response.json().await.map_err(upstream_error)?;
                if !status.is_success() {
                    return Err(upload_status(status, &body));
                }
                let id = body
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        UnllmError::invalid(
                            "asset_upload_response",
                            "File upload response did not include an id",
                        )
                    })?
                    .to_owned();
                Ok(AssetLease {
                    asset: MediaAsset {
                        source: MediaSource::ProviderFile {
                            provider: request.provider.clone(),
                            id: id.clone(),
                            mime_type: Some(mime_type.clone()),
                        },
                        width: asset.width,
                        height: asset.height,
                        sha256: asset.sha256.clone(),
                        extensions: asset.extensions.clone(),
                    },
                    cleanup_token: format!("{}:{id}", request.provider),
                })
            }
            "gemini" => {
                let mut url = self.url("upload/v1beta/files")?;
                if let Some(token) = self.token()? {
                    url.query_pairs_mut().append_pair("key", &token);
                }
                let start = self
                    .client
                    .post(url)
                    .header("X-Goog-Upload-Protocol", "resumable")
                    .header("X-Goog-Upload-Command", "start")
                    .header("X-Goog-Upload-Header-Content-Length", data.len())
                    .header("X-Goog-Upload-Header-Content-Type", mime_type)
                    .json(&json!({"file": {"display_name": "unllm-temporary"}}))
                    .send()
                    .await
                    .map_err(upstream_error)?;
                let upload_url = start
                    .headers()
                    .get("X-Goog-Upload-URL")
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| {
                        UnllmError::invalid(
                            "asset_upload_response",
                            "Gemini upload start did not return an upload URL",
                        )
                    })?;
                let response = self
                    .client
                    .post(upload_url)
                    .header("X-Goog-Upload-Offset", "0")
                    .header("X-Goog-Upload-Command", "upload, finalize")
                    .header(header::CONTENT_TYPE, mime_type)
                    .body(data.clone())
                    .send()
                    .await
                    .map_err(upstream_error)?;
                let status = response.status();
                let body: Value = response.json().await.map_err(upstream_error)?;
                if !status.is_success() {
                    return Err(upload_status(status, &body));
                }
                let uri = body
                    .pointer("/file/uri")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        UnllmError::invalid(
                            "asset_upload_response",
                            "Gemini upload response did not include a file URI",
                        )
                    })?
                    .to_owned();
                let name = body
                    .pointer("/file/name")
                    .and_then(Value::as_str)
                    .unwrap_or(&uri)
                    .to_owned();
                Ok(AssetLease {
                    asset: MediaAsset {
                        source: MediaSource::ProviderFile {
                            provider: "gemini".into(),
                            id: uri,
                            mime_type: Some(mime_type.clone()),
                        },
                        width: asset.width,
                        height: asset.height,
                        sha256: asset.sha256.clone(),
                        extensions: asset.extensions.clone(),
                    },
                    cleanup_token: format!("gemini:{name}"),
                })
            }
            provider => Err(UnllmError::unsupported(
                "asset_stager_provider",
                format!("No asset stager is available for provider {provider}"),
            )),
        }
    }

    async fn cleanup(&self, lease: AssetLease) -> Result<(), UnllmError> {
        let (provider, id) = lease.cleanup_token.split_once(':').ok_or_else(|| {
            UnllmError::invalid("asset_cleanup_token", "Invalid asset cleanup token")
        })?;
        let path = if provider == "gemini" {
            format!("v1beta/{id}")
        } else {
            format!("v1/files/{id}")
        };
        let mut url = self.url(&path)?;
        let mut builder = self.client.delete(url.clone());
        if let Some(token) = self.token()? {
            if provider == "anthropic" {
                builder = builder
                    .header("x-api-key", token)
                    .header("anthropic-version", "2023-06-01")
                    .header("anthropic-beta", "files-api-2025-04-14");
            } else if provider == "gemini" {
                url.query_pairs_mut().append_pair("key", &token);
                builder = self.client.delete(url);
            } else {
                builder = builder.bearer_auth(token);
            }
        }
        let response = builder.send().await.map_err(upstream_error)?;
        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            Err(UnllmError {
                code: "asset_cleanup_status".into(),
                kind: ErrorKind::Upstream,
                message: format!("Asset cleanup returned HTTP {}", response.status()),
                source_protocol: None,
                target: None,
                operation: None,
                metadata: None,
            })
        }
    }
}

pub async fn prepare_request(
    request: &mut UnifiedRequest,
    resolver: Option<&dyn MediaResolver>,
    stager: Option<Arc<dyn AssetStager>>,
    provider: &str,
    resolve_media: bool,
    stage_files: bool,
) -> Result<Vec<AssetLease>, UnllmError> {
    let mut leases = Vec::new();
    let result = prepare_request_inner(
        request,
        resolver,
        stager.as_deref(),
        provider,
        resolve_media,
        stage_files,
        &mut leases,
    )
    .await;
    if let Err(error) = result {
        if let Some(stager) = stager {
            cleanup_leases(stager, leases).await;
        }
        return Err(error);
    }
    Ok(leases)
}

async fn prepare_request_inner(
    request: &mut UnifiedRequest,
    resolver: Option<&dyn MediaResolver>,
    stager: Option<&dyn AssetStager>,
    provider: &str,
    resolve_media: bool,
    stage_files: bool,
    leases: &mut Vec<AssetLease>,
) -> Result<(), UnllmError> {
    match request {
        UnifiedRequest::Generate(request) => {
            for instruction in &mut request.instructions {
                prepare_parts(
                    &mut instruction.content,
                    resolver,
                    stager,
                    provider,
                    resolve_media,
                    stage_files,
                    leases,
                )
                .await?;
            }
            for message in &mut request.messages {
                prepare_parts(
                    &mut message.content,
                    resolver,
                    stager,
                    provider,
                    resolve_media,
                    stage_files,
                    leases,
                )
                .await?;
            }
        }
        UnifiedRequest::Embed(request) => {
            for input in &mut request.inputs {
                if let EmbeddingInput::Content { content } = input {
                    prepare_parts(
                        content,
                        resolver,
                        stager,
                        provider,
                        resolve_media,
                        stage_files,
                        leases,
                    )
                    .await?;
                }
            }
        }
        UnifiedRequest::Image(request) => match &mut request.task {
            ImageTask::Generate => {}
            ImageTask::Edit { image, mask } => {
                prepare_asset(image, resolver, resolve_media).await?;
                if let Some(mask) = mask {
                    prepare_asset(mask, resolver, resolve_media).await?;
                }
            }
            ImageTask::Variation { image } => {
                prepare_asset(image, resolver, resolve_media).await?;
            }
        },
    }
    Ok(())
}

fn prepare_parts<'a>(
    parts: &'a mut [ContentPart],
    resolver: Option<&'a dyn MediaResolver>,
    stager: Option<&'a dyn AssetStager>,
    provider: &'a str,
    resolve_media: bool,
    stage_files: bool,
    leases: &'a mut Vec<AssetLease>,
) -> BoxFuture<'a, Result<(), UnllmError>> {
    Box::pin(async move {
        for part in parts {
            match part {
                ContentPart::Image { asset }
                | ContentPart::Audio { asset }
                | ContentPart::Video { asset } => {
                    prepare_asset(asset, resolver, resolve_media).await?;
                }
                ContentPart::File { asset, .. } => {
                    prepare_asset(asset, resolver, resolve_media).await?;
                    if stage_files
                        && !matches!(&asset.source, MediaSource::ProviderFile { provider: owner, .. } if owner == provider)
                    {
                        let stager = stager.ok_or_else(|| {
                            UnllmError::unsupported(
                                "asset_stager_disabled",
                                "The route requires file staging but no stager is configured",
                            )
                        })?;
                        let lease = stager
                            .stage(
                                &StagingRequest {
                                    provider: provider.into(),
                                    purpose: None,
                                },
                                asset,
                            )
                            .await?;
                        *asset = lease.asset.clone();
                        leases.push(lease);
                    }
                }
                ContentPart::ToolResult { result } => {
                    prepare_parts(
                        &mut result.content,
                        resolver,
                        stager,
                        provider,
                        resolve_media,
                        stage_files,
                        leases,
                    )
                    .await?;
                }
                ContentPart::Text { .. }
                | ContentPart::ToolCall { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::Refusal { .. } => {}
            }
        }
        Ok(())
    })
}

async fn prepare_asset(
    asset: &mut MediaAsset,
    resolver: Option<&dyn MediaResolver>,
    resolve_media: bool,
) -> Result<(), UnllmError> {
    if resolve_media && matches!(asset.source, MediaSource::Url { .. }) {
        let resolver = resolver.ok_or_else(|| {
            UnllmError::unsupported(
                "media_resolver_disabled",
                "The route requires media resolution but the resolver is disabled",
            )
        })?;
        *asset = resolver.resolve(asset).await?;
    }
    Ok(())
}

pub async fn cleanup_leases(stager: Arc<dyn AssetStager>, leases: Vec<AssetLease>) {
    for lease in leases {
        let mut last_error = None;
        for attempt in 0..3 {
            match stager.cleanup(lease.clone()).await {
                Ok(()) => {
                    last_error = None;
                    break;
                }
                Err(error) => {
                    last_error = Some(error);
                    tokio::time::sleep(std::time::Duration::from_millis(100 * (attempt + 1))).await;
                }
            }
        }
        if let Some(error) = last_error {
            tracing::warn!(code = %error.code, "Temporary provider file cleanup failed");
        }
    }
}

pub struct LeaseCleanupGuard {
    stager: Arc<dyn AssetStager>,
    leases: Option<Vec<AssetLease>>,
}

impl LeaseCleanupGuard {
    pub fn new(stager: Arc<dyn AssetStager>, leases: Vec<AssetLease>) -> Self {
        Self {
            stager,
            leases: Some(leases),
        }
    }
}

impl Drop for LeaseCleanupGuard {
    fn drop(&mut self) {
        if let Some(leases) = self.leases.take() {
            let stager = self.stager.clone();
            tokio::spawn(async move {
                cleanup_leases(stager, leases).await;
            });
        }
    }
}

pub fn summarize_json(value: &Value) -> Value {
    match value {
        Value::String(text) if text.len() > 4096 && looks_like_base64(text) => json!({
            "media_summary": true,
            "encoded_bytes": text.len(),
            "sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
        }),
        Value::Array(values) => Value::Array(values.iter().map(summarize_json).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), summarize_json(value)))
                .collect(),
        ),
        value => value.clone(),
    }
}

fn looks_like_base64(value: &str) -> bool {
    let candidate = value
        .strip_prefix("data:")
        .and_then(|value| value.split_once(',').map(|(_, data)| data))
        .unwrap_or(value);
    candidate.len() % 4 == 0
        && candidate.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=' | b'\n' | b'\r')
        })
        && STANDARD.decode(candidate.as_bytes()).is_ok()
}

fn is_private(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            address.is_private()
                || address.is_loopback()
                || address.is_link_local()
                || address.is_broadcast()
                || address.is_documentation()
                || address.is_unspecified()
        }
        IpAddr::V6(address) => {
            address.is_loopback()
                || address.is_unspecified()
                || address.is_unique_local()
                || address.is_unicast_link_local()
        }
    }
}

fn upstream_error(error: reqwest::Error) -> UnllmError {
    UnllmError {
        code: "asset_upstream".into(),
        kind: ErrorKind::Upstream,
        message: format!("Provider file operation failed: {error}"),
        source_protocol: None,
        target: None,
        operation: None,
        metadata: None,
    }
}

fn upload_status(status: StatusCode, body: &Value) -> UnllmError {
    UnllmError {
        code: "asset_upload_status".into(),
        kind: ErrorKind::Upstream,
        message: body
            .pointer("/error/message")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Provider file upload returned HTTP {status}")),
        source_protocol: None,
        target: None,
        operation: None,
        metadata: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolver_rejects_non_https_urls() {
        let resolver = HttpMediaResolver::new(&MediaConfig {
            enabled: true,
            allowed_hosts: vec!["example.com".into()],
        })
        .unwrap();
        let error = resolver
            .resolve(&MediaAsset::url("http://example.com/image.png", None))
            .await
            .unwrap_err();
        assert_eq!(error.code, "media_scheme");
    }
}
