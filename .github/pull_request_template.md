## What / Why

- What changed:
- Why this change is needed:

## Scope

- [ ] Image
- [ ] Video
- [ ] Audio
- [ ] Frontend
- [ ] Packaging / Build
- [ ] Docs only

## New Model Integration Checklist (if applicable)

Reference: `docs/engine_architecture.md` §6

- [ ] Updated `default_config/models_registry.json` (real GGUF version rows only; no invented URL)
- [ ] Added a `FamilySpec` row and wired `execute` only from `native/crates/danmo-engine/src/families/mod.rs`
- [ ] Added the family under `native/crates/danmo-engine/src/families/<family>/` without importing another family
- [ ] Reused `common/qwen3`, `common/vae`, `common/png`, or `common/ops` with parameters, or kept a divergent denoise / VAE in the family
- [ ] Shared modules still have no family id
- [ ] Fail-loud behavior preserved (no silent fallback on load or inference errors)

## Reuse / Architecture Notes

- Reused `native/crates/danmo-engine/src/common/*`:
- Family-specific logic intentionally kept local:
- Device backend (`cpu` / `cuda` / `vulkan` / `metal`), if the change touches `GgmlContext`:

## Validation

- [ ] `cargo test --manifest-path native/Cargo.toml --offline -p danmo-engine --lib`
- [ ] Server tests when the API contract changed
- [ ] Minimal inference path validated (engine test or API smoke)

Commands/results:

```text
# paste relevant commands and key outputs
```

## Risks / Follow-ups

- Risks:
- Follow-up tasks:
