mod checkpoint;
mod edit_session;
#[cfg(feature = "mlx")]
mod mlx;
mod session;
mod vae;

pub(crate) use edit_session::Flux2EditFamily;
pub(crate) use session::Flux2Family;

use crate::common::ops::empirical_mu_sigmas;
use crate::weights::WeightError;
use std::path::Path;

/// Both registry names are `FlowMatchEulerScheduler`. The card does not set
/// `use_empirical_mu` false, and this config requires sigma shift, so the
/// trajectory is empirical mu on the VAE grid (`width/16 * height/16`).
pub(super) fn schedule(steps: u32, grid_w: i64, grid_h: i64) -> Vec<f32> {
    let seq = (grid_w.max(0) as u32).saturating_mul(grid_h.max(0) as u32);
    empirical_mu_sigmas(steps, seq)
}

/// Optional edge cap. A finished prequantized install may have no manifest;
/// that is no cap. A present file that is not JSON still fails.
pub(super) fn manifest_max_edge(bundle: &Path) -> Result<Option<u32>, WeightError> {
    let path = bundle.join("bundle.manifest.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: err.to_string(),
            });
        }
    };
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| WeightError::NotBuilt(format!("bundle.manifest.json is not json: {err}")))?;
    Ok(value
        .get("max_edge")
        .and_then(|item| item.as_u64())
        .map(|edge| edge as u32))
}

#[cfg(test)]
mod tests {
    use super::manifest_max_edge;

    #[test]
    fn missing_manifest_is_no_edge_cap_and_bad_json_fails() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-flux2-manifest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(manifest_max_edge(&dir).unwrap(), None);
        std::fs::write(dir.join("bundle.manifest.json"), b"{").unwrap();
        let err = manifest_max_edge(&dir).unwrap_err().to_string();
        assert!(err.contains("not json"), "{err}");
        std::fs::write(dir.join("bundle.manifest.json"), br#"{"max_edge":512}"#).unwrap();
        assert_eq!(manifest_max_edge(&dir).unwrap(), Some(512));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
