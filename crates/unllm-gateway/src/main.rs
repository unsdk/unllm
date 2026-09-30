//! Command-line entry point for the unllm gateway.

use clap::Parser;

#[tokio::main]
async fn main() {
    if let Err(error) = unllm_gateway::run(unllm_gateway::Cli::parse()).await {
        eprintln!("unllm-gateway: {error}");
        std::process::exit(1);
    }
}
