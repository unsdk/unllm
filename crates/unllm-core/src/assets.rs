use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{MediaAsset, UnllmError};

/// Describes why an asset needs to be staged at a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct StagingRequest {
    /// Target provider namespace.
    pub provider: String,
    /// Optional provider file purpose.
    pub purpose: Option<String>,
}

/// A temporary provider-managed asset and its cleanup token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AssetLease {
    /// Asset reference to place into the normalized request.
    pub asset: MediaAsset,
    /// Opaque token required by the stager during cleanup.
    pub cleanup_token: String,
}

/// Resolves remote media into another supported representation.
#[async_trait]
pub trait MediaResolver: Send + Sync {
    /// Resolves one media value without changing its semantic content.
    async fn resolve(&self, asset: &MediaAsset) -> Result<MediaAsset, UnllmError>;
}

/// Stages media in a provider-managed file service.
#[async_trait]
pub trait AssetStager: Send + Sync {
    /// Uploads an asset and returns a lease that must later be cleaned up.
    async fn stage(
        &self,
        request: &StagingRequest,
        asset: &MediaAsset,
    ) -> Result<AssetLease, UnllmError>;

    /// Performs best-effort cleanup of one lease.
    async fn cleanup(&self, lease: AssetLease) -> Result<(), UnllmError>;
}

/// Resolver used when network media fetching is disabled.
#[derive(Debug, Default)]
pub struct DisabledMediaResolver;

#[async_trait]
impl MediaResolver for DisabledMediaResolver {
    async fn resolve(&self, _asset: &MediaAsset) -> Result<MediaAsset, UnllmError> {
        Err(UnllmError::unsupported(
            "media_resolution_disabled",
            "Remote media resolution is disabled",
        ))
    }
}
