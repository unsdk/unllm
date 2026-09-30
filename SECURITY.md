# Security

## Reporting

Please report security issues privately to the repository maintainers. Do not include credentials, prompts, generated content, or provider responses in a public issue.

## Gateway deployment

The gateway is an application-layer protocol translator, not an internet edge proxy.

- It binds to loopback by default.
- Put a production deployment behind a TLS-terminating reverse proxy.
- Configure request-size, connection, stream-size, and concurrency limits at that proxy. The gateway intentionally has no application-level limits.
- Configure client-key authentication before exposing a listener outside the host.
- Keep provider tokens in environment variables. Literal command-line tokens can be exposed through process inspection and shell history.
- Do not allow untrusted static headers, query parameters, or additional request bodies in route configuration.

## Remote media

Remote media resolution is disabled by default. When enabled, the gateway requires an HTTPS host allowlist, validates each redirect, resolves DNS before fetching, and rejects private, loopback, link-local, unspecified, and documentation addresses.

The configured design intentionally imposes no media byte or media timeout limit. Operators must enforce both externally and should use a narrow host allowlist.

## Logging

Request and response bodies are not logged by default. The --log-bodies option records prompts, tool arguments, and other semantic data, while replacing large base64 media with a hash and size summary. Authentication values are never logged. Treat body logs as sensitive data.
