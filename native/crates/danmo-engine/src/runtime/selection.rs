use super::{probe_attention, AttentionProbeReport, BackendKind, GgmlContext, RuntimeError};

#[derive(Debug, Clone)]
pub struct BackendSelection {
    pub backend: BackendKind,
    pub reason: String,
    pub probe: Option<AttentionProbeReport>,
}

impl BackendSelection {
    fn explicit(backend: BackendKind) -> Self {
        Self {
            backend,
            reason: format!("explicit backend {}", backend.name()),
            probe: None,
        }
    }
}

pub fn select_from_env() -> Result<BackendSelection, RuntimeError> {
    match BackendKind::from_env()? {
        BackendKind::Auto => select_auto(),
        backend => Ok(BackendSelection::explicit(backend)),
    }
}

fn select_auto() -> Result<BackendSelection, RuntimeError> {
    let candidates = auto_candidates();
    let priority = candidates
        .iter()
        .map(|backend| backend.name())
        .collect::<Vec<_>>()
        .join(" > ");
    let mut failures = Vec::new();
    for backend in candidates {
        match GgmlContext::new(backend) {
            Ok(ctx) => match probe_attention(&ctx) {
                Ok(probe) => {
                    return Ok(BackendSelection {
                        backend,
                        reason: format!("auto: platform priority {priority}"),
                        probe: Some(probe),
                    })
                }
                Err(error) => failures.push(format!("{backend:?} probe: {error}")),
            },
            Err(error) => failures.push(format!("{backend:?} init: {error}")),
        }
    }
    Err(RuntimeError::Message(format!(
        "auto backend selection failed: {}",
        failures.join("; ")
    )))
}

fn auto_candidates() -> Vec<BackendKind> {
    let mut candidates = Vec::new();
    #[cfg(target_os = "macos")]
    candidates.push(BackendKind::Metal);
    #[cfg(not(target_os = "macos"))]
    candidates.push(BackendKind::Cuda);
    #[cfg(target_os = "macos")]
    candidates.push(BackendKind::Cuda);
    candidates.push(BackendKind::Vulkan);
    #[cfg(not(target_os = "macos"))]
    candidates.push(BackendKind::Metal);
    candidates.push(BackendKind::Cpu);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_candidates_keep_cpu_as_the_last_fallback() {
        let candidates = auto_candidates();
        assert_eq!(candidates.last(), Some(&BackendKind::Cpu));
        #[cfg(target_os = "macos")]
        assert_eq!(candidates.first(), Some(&BackendKind::Metal));
        #[cfg(not(target_os = "macos"))]
        assert_eq!(candidates.first(), Some(&BackendKind::Cuda));
    }

    #[test]
    fn explicit_backend_does_not_run_auto_selection() {
        let selection = BackendSelection::explicit(BackendKind::Cpu);
        assert_eq!(selection.backend, BackendKind::Cpu);
        assert!(selection.probe.is_none());
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn auto_falls_back_to_cpu_when_metal_is_not_compiled() {
        let selection = select_auto().unwrap();
        assert_eq!(selection.backend, BackendKind::Cpu);
        assert!(selection.probe.is_some());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn auto_selects_metal_when_it_is_compiled_and_validated() {
        let selection = select_auto().unwrap();
        assert_eq!(selection.backend, BackendKind::Metal);
        assert!(selection
            .probe
            .as_ref()
            .is_some_and(|probe| probe.validated));
    }
}
