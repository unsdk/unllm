use std::{collections::BTreeMap, fs, net::SocketAddr, path::PathBuf, time::Duration};

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use unllm::{Capabilities, Capability, ConversionMode, Operation, Protocol};
use url::Url;

#[derive(Debug, Parser)]
#[command(name = "unllm-gateway", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Starts the HTTP gateway.
    Serve(ServeArgs),
    /// Parses and validates a configuration file.
    CheckConfig { path: PathBuf },
    /// Prints a complete example configuration.
    GenerateConfig,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// TOML configuration file. Mutually exclusive with quick-mode upstream arguments.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Override the server listen address.
    #[arg(long)]
    pub listen: Option<SocketAddr>,
    /// Records semantic request and response bodies. Media is summarized.
    #[arg(long, default_value_t = false)]
    pub log_bodies: bool,
    /// Quick-mode upstream protocol.
    #[arg(long, value_enum)]
    pub protocol: Option<ProtocolArg>,
    /// Quick-mode upstream base URL.
    #[arg(long)]
    pub base_url: Option<Url>,
    /// Quick-mode model exposed under the same name.
    #[arg(long)]
    pub model: Option<String>,
    /// Quick-mode upstream token. Prefer --token-env.
    #[arg(long, conflicts_with = "token_env")]
    pub token: Option<String>,
    /// Quick-mode environment variable containing the upstream token.
    #[arg(long)]
    pub token_env: Option<String>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ProtocolArg {
    OpenaiChat,
    OpenaiResponses,
    Anthropic,
    Gemini,
}

impl From<ProtocolArg> for Protocol {
    fn from(value: ProtocolArg) -> Self {
        match value {
            ProtocolArg::OpenaiChat => Self::OpenAiChatCompletions,
            ProtocolArg::OpenaiResponses => Self::OpenAiResponses,
            ProtocolArg::Anthropic => Self::AnthropicMessages,
            ProtocolArg::Gemini => Self::Gemini,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    pub schema_version: u32,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub media: MediaConfig,
    pub upstreams: Vec<UpstreamConfig>,
    pub routes: Vec<RouteConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub client_key_env: Vec<String>,
    pub connect_timeout_ms: u64,
    pub headers_timeout_ms: u64,
    pub stream_idle_timeout_ms: u64,
    pub non_stream_timeout_ms: u64,
    pub shutdown_grace_ms: u64,
    pub log: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8080".parse().expect("valid default address"),
            client_key_env: Vec::new(),
            connect_timeout_ms: 10_000,
            headers_timeout_ms: 30_000,
            stream_idle_timeout_ms: 60_000,
            non_stream_timeout_ms: 300_000,
            shutdown_grace_ms: 30_000,
            log: "info".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaConfig {
    pub enabled: bool,
    pub allowed_hosts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamConfig {
    pub name: String,
    pub protocol: Protocol,
    pub base_url: Url,
    #[serde(default)]
    pub token_env: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, SecretValue>,
    #[serde(default)]
    pub query: BTreeMap<String, SecretValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SecretValue {
    Literal(String),
    Environment { env: String },
}

impl SecretValue {
    pub fn resolve(&self) -> Result<String, String> {
        match self {
            Self::Literal(value) => Ok(value.clone()),
            Self::Environment { env } => std::env::var(env)
                .map_err(|_| format!("Required environment variable {env} is not set")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteConfig {
    pub alias: String,
    pub upstream: String,
    pub remote_model: String,
    pub operations: Vec<Operation>,
    #[serde(default)]
    pub generate_protocol: Option<Protocol>,
    #[serde(default)]
    pub mode: ConversionMode,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    #[serde(default)]
    pub beta_features: Vec<String>,
    #[serde(default)]
    pub paths: BTreeMap<Operation, String>,
    #[serde(default)]
    pub headers: BTreeMap<String, SecretValue>,
    #[serde(default)]
    pub query: BTreeMap<String, SecretValue>,
    #[serde(default)]
    pub additional_body: BTreeMap<Operation, Value>,
    #[serde(default)]
    pub forward_headers: Vec<String>,
    #[serde(default)]
    pub resolve_media: bool,
    #[serde(default)]
    pub stage_files: bool,
}

impl GatewayConfig {
    pub fn from_path(path: &PathBuf) -> Result<Self, String> {
        let source = fs::read_to_string(path)
            .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
        let config: Self = toml::from_str(&source)
            .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn quick(args: &ServeArgs) -> Result<Self, String> {
        if let Some(path) = &args.config {
            if args.protocol.is_some()
                || args.base_url.is_some()
                || args.model.is_some()
                || args.token.is_some()
                || args.token_env.is_some()
            {
                return Err(
                    "--config cannot be combined with quick-mode upstream arguments".into(),
                );
            }
            return Self::from_path(path);
        }
        let protocol: Protocol = args
            .protocol
            .ok_or("Quick mode requires --protocol")?
            .into();
        let base_url = args
            .base_url
            .clone()
            .ok_or("Quick mode requires --base-url")?;
        let model = args.model.clone().ok_or("Quick mode requires --model")?;
        let mut config = Self {
            schema_version: 1,
            server: ServerConfig::default(),
            media: MediaConfig::default(),
            upstreams: vec![UpstreamConfig {
                name: "default".into(),
                protocol,
                base_url,
                token_env: args.token_env.clone(),
                token: args.token.clone(),
                headers: BTreeMap::new(),
                query: BTreeMap::new(),
            }],
            routes: vec![RouteConfig {
                alias: model.clone(),
                upstream: "default".into(),
                remote_model: model,
                operations: if protocol == Protocol::AnthropicMessages {
                    vec![Operation::Generate]
                } else {
                    vec![Operation::Generate, Operation::Embed, Operation::Image]
                },
                generate_protocol: Some(protocol),
                mode: ConversionMode::Strict,
                capabilities: Vec::new(),
                beta_features: Vec::new(),
                paths: BTreeMap::new(),
                headers: BTreeMap::new(),
                query: BTreeMap::new(),
                additional_body: BTreeMap::new(),
                forward_headers: Vec::new(),
                resolve_media: false,
                stage_files: false,
            }],
        };
        if let Some(listen) = args.listen {
            config.server.listen = listen;
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "Unsupported configuration schema version: {}",
                self.schema_version
            ));
        }
        if self.upstreams.is_empty() {
            return Err("At least one upstream is required".into());
        }
        if self.routes.is_empty() {
            return Err("At least one route is required".into());
        }
        if self.media.enabled && self.media.allowed_hosts.is_empty() {
            return Err(
                "media.allowed_hosts must not be empty when media fetching is enabled".into(),
            );
        }
        let mut upstream_names = std::collections::BTreeSet::new();
        for upstream in &self.upstreams {
            if !upstream_names.insert(upstream.name.clone()) {
                return Err(format!("Duplicate upstream name: {}", upstream.name));
            }
            if upstream.token.is_some() && upstream.token_env.is_some() {
                return Err(format!(
                    "Upstream {} cannot set both token and token_env",
                    upstream.name
                ));
            }
            if upstream.base_url.scheme() != "http" && upstream.base_url.scheme() != "https" {
                return Err(format!("Upstream {} must use HTTP or HTTPS", upstream.name));
            }
        }
        let mut aliases = std::collections::BTreeSet::new();
        let mut remote_models: BTreeMap<&str, &str> = BTreeMap::new();
        for route in &self.routes {
            if !aliases.insert(route.alias.clone()) {
                return Err(format!("Duplicate route alias: {}", route.alias));
            }
            if !upstream_names.contains(&route.upstream) {
                return Err(format!(
                    "Route {} references unknown upstream {}",
                    route.alias, route.upstream
                ));
            }
            if route.operations.is_empty() {
                return Err(format!("Route {} has no operations", route.alias));
            }
            if let Some(existing) = remote_models.insert(&route.remote_model, &route.upstream) {
                if existing != route.upstream {
                    return Err(format!(
                        "Remote model {} is assigned to more than one upstream",
                        route.remote_model
                    ));
                }
            }
            for path in route.paths.values() {
                if !path.starts_with('/') || path.starts_with("//") || path.contains("://") {
                    return Err(format!(
                        "Route {} contains a non-relative override path",
                        route.alias
                    ));
                }
            }
            for header in &route.forward_headers {
                if is_forbidden_forward_header(header) {
                    return Err(format!(
                        "Route {} attempts to forward forbidden header {header}",
                        route.alias
                    ));
                }
            }
        }
        for env in &self.server.client_key_env {
            std::env::var(env).map_err(|_| {
                format!("Required client key environment variable {env} is not set")
            })?;
        }
        for upstream in &self.upstreams {
            if let Some(env) = &upstream.token_env {
                std::env::var(env).map_err(|_| {
                    format!(
                        "Required token environment variable {env} for upstream {} is not set",
                        upstream.name
                    )
                })?;
            }
            for value in upstream.headers.values().chain(upstream.query.values()) {
                value.resolve()?;
            }
        }
        for route in &self.routes {
            for value in route.headers.values().chain(route.query.values()) {
                value.resolve()?;
            }
        }
        Ok(())
    }

    pub fn upstream(&self, name: &str) -> Option<&UpstreamConfig> {
        self.upstreams.iter().find(|upstream| upstream.name == name)
    }

    pub fn resolve_route(&self, model: &str, operation: Operation) -> Result<&RouteConfig, String> {
        if let Some(route) = self
            .routes
            .iter()
            .find(|route| route.alias == model && route.operations.contains(&operation))
        {
            return Ok(route);
        }
        if let Some((provider, remote_model)) = model.split_once('/') {
            let matches: Vec<_> = self
                .routes
                .iter()
                .filter(|route| {
                    route.operations.contains(&operation)
                        && route.remote_model == remote_model
                        && self
                            .upstream(&route.upstream)
                            .is_some_and(|upstream| upstream.protocol.namespace() == provider)
                })
                .collect();
            if matches.len() == 1 {
                return Ok(matches[0]);
            }
        }
        let matches: Vec<_> = self
            .routes
            .iter()
            .filter(|route| route.remote_model == model && route.operations.contains(&operation))
            .collect();
        match matches.as_slice() {
            [route] => Ok(route),
            [] => Err(format!("No route is configured for model {model}")),
            _ => Err(format!("Model name {model} is ambiguous")),
        }
    }

    pub fn client_keys(&self) -> Vec<String> {
        self.server
            .client_key_env
            .iter()
            .filter_map(|name| std::env::var(name).ok())
            .collect()
    }

    pub fn timeout_connect(&self) -> Duration {
        Duration::from_millis(self.server.connect_timeout_ms)
    }
}

impl UpstreamConfig {
    pub fn token(&self) -> Result<Option<String>, String> {
        if let Some(env) = &self.token_env {
            return std::env::var(env)
                .map(Some)
                .map_err(|_| format!("Required token environment variable {env} is not set"));
        }
        Ok(self.token.clone())
    }
}

impl RouteConfig {
    pub fn capabilities(&self, adapter: &Capabilities) -> Capabilities {
        if self.capabilities.is_empty() {
            return adapter.clone();
        }
        adapter.restricted_to(&Capabilities {
            operations: self.operations.iter().copied().collect(),
            features: self.capabilities.iter().cloned().collect(),
        })
    }
}

pub fn is_forbidden_forward_header(header: &str) -> bool {
    matches!(
        header.to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "set-cookie"
            | "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
            | "content-length"
    )
}

pub const EXAMPLE_CONFIG: &str = r#"schema_version = 1

[server]
listen = "127.0.0.1:8080"
client_key_env = ["UNLLM_CLIENT_KEY"]
connect_timeout_ms = 10000
headers_timeout_ms = 30000
stream_idle_timeout_ms = 60000
non_stream_timeout_ms = 300000
shutdown_grace_ms = 30000
log = "info"

[media]
enabled = false
allowed_hosts = []

[[upstreams]]
name = "openai"
protocol = "open_ai_responses"
base_url = "https://api.openai.com"
token_env = "OPENAI_API_KEY"

[[routes]]
alias = "default"
upstream = "openai"
remote_model = "gpt-5"
operations = ["generate", "embed", "image"]
generate_protocol = "open_ai_responses"
mode = "strict"
capabilities = ["streaming", "text", "image", "function_tools", "structured_output"]
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example() {
        let config: GatewayConfig = toml::from_str(EXAMPLE_CONFIG).unwrap();
        assert_eq!(config.schema_version, 1);
        assert_eq!(config.routes[0].alias, "default");
    }

    #[test]
    fn route_resolution_has_deterministic_precedence() {
        let config: GatewayConfig = toml::from_str(EXAMPLE_CONFIG).unwrap();
        assert_eq!(
            config
                .resolve_route("default", Operation::Generate)
                .unwrap()
                .alias,
            "default"
        );
        assert_eq!(
            config
                .resolve_route("openai/gpt-5", Operation::Generate)
                .unwrap()
                .alias,
            "default"
        );
        assert_eq!(
            config
                .resolve_route("gpt-5", Operation::Generate)
                .unwrap()
                .alias,
            "default"
        );
    }

    #[test]
    fn unknown_configuration_keys_are_rejected() {
        let invalid = format!("{EXAMPLE_CONFIG}\nunknown = true\n");
        assert!(toml::from_str::<GatewayConfig>(&invalid).is_err());
    }
}
