use crate::contract::{EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::FacadeFamily;
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;

pub fn run(
    ctx: &GgmlContext,
    run: &ResolvedRun,
    family: &dyn FacadeFamily,
    request: &EngineRequest,
    resolver: &dyn ResolvePort,
    exec: &ExecutionContext,
) -> Result<EngineOutput, EngineError> {
    exec.phase("prepare")?;
    let mut session = family.prepare(ctx, run, request, resolver, exec)?;
    exec.phase("infer")?;
    let mut output = session.infer(ctx, exec)?;
    output.family = run.family.clone();
    output.action = run.action.as_str().into();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Action, RunObserver};
    use crate::families::FacadeSession;
    use crate::runtime::BackendKind;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    struct SecondFamily;
    struct SecondSession;

    impl FacadeFamily for SecondFamily {
        fn prepare(
            &self,
            _ctx: &GgmlContext,
            _run: &ResolvedRun,
            _request: &EngineRequest,
            _resolver: &dyn ResolvePort,
            _exec: &ExecutionContext,
        ) -> Result<Box<dyn FacadeSession>, EngineError> {
            Ok(Box::new(SecondSession))
        }
    }

    impl FacadeSession for SecondSession {
        fn infer(
            &mut self,
            _ctx: &GgmlContext,
            _exec: &ExecutionContext,
        ) -> Result<EngineOutput, EngineError> {
            Ok(EngineOutput {
                metadata: json!({"implementation": "second"}),
                ..EngineOutput::default()
            })
        }
    }

    struct Resolver;
    impl ResolvePort for Resolver {
        fn resolve(&self, _model: &str, _action: Action) -> Result<ResolvedRun, EngineError> {
            unreachable!()
        }

        fn resolve_component(
            &self,
            _run: &ResolvedRun,
            _role: &str,
            _model: &str,
        ) -> Result<PathBuf, EngineError> {
            unreachable!()
        }

        fn read_source_audio(
            &self,
            _run: &ResolvedRun,
            _source: &str,
        ) -> Result<Vec<u8>, EngineError> {
            unreachable!()
        }
    }

    #[derive(Default)]
    struct Observer(Mutex<Vec<String>>);
    impl RunObserver for Observer {
        fn progress(&self, _phase: &str, _step: u32, _total: u32) -> Result<(), EngineError> {
            Ok(())
        }

        fn log(&self, _level: &str, _message: &str) -> Result<(), EngineError> {
            Ok(())
        }

        fn phase(&self, phase: &str) -> Result<(), EngineError> {
            self.0.lock().unwrap().push(phase.into());
            Ok(())
        }
    }

    #[test]
    fn second_family_uses_the_same_facade_without_pipeline_changes() {
        let run = ResolvedRun {
            model: "second".into(),
            version: None,
            family: "second_audio".into(),
            action: Action::AudioCreate,
            bundle: PathBuf::new(),
            backend: BackendKind::Cpu,
        };
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let observer = std::sync::Arc::new(Observer::default());
        let output = super::run(
            &ctx,
            &run,
            &SecondFamily,
            &EngineRequest::Unsupported {
                model: "second".into(),
                action: Action::AudioCreate,
            },
            &Resolver,
            &ExecutionContext::new("task", observer.clone()),
        )
        .unwrap();
        assert_eq!(output.family, "second_audio");
        assert_eq!(output.metadata["implementation"], "second");
        assert_eq!(*observer.0.lock().unwrap(), ["prepare", "infer"]);
    }
}
