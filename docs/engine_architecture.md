# Danmo Make Engine Architecture

The engine is Rust on ggml, under `native/`. Product code is `danqing-api`. Weights are GGUF. Tensor names in the file are graph parameter names.

Operational rules: [`AGENTS.md`](../AGENTS.md).

---

## 1. Principles

1. **Fail loud.** A missing tensor, a wrong shape, an unknown scheduler, or a graph that is not built returns an error. Nothing is filled in, truncated, or swapped for a smaller path.
2. **One device for the whole graph.** `GgmlContext` is that device (`cpu`, `cuda`, `vulkan`, or `metal`). The binary must be built with the matching cargo feature. Choosing CPU as the device is valid. A missing operator is an error. It is not run on a second CPU backend beside the chosen device.
3. **Families add models.** A new chip backend is another `GgmlContext` backend. It is not a new family and not a second engine tree.
4. **Families are isolated.** A family module does not import another family. The factory in `families/mod.rs` is the only match on a family id.
5. **Shared code does not know family ids.** It reads weight-file tensor names and the parameters the caller passes. `layers.*` and `decoder.*` are file formats, not family names.
6. **Keep shared code small.** A real difference is a neutral parameter (a layer list, a `bool`, a `VaeSpec`). When the difference needs a pile of callbacks or mode switches, each family implements that piece itself.
7. **Phases stay in order.** resolve → load → encode → schedule → infer → decode → persist. Families do not reorder them.
8. **No key remap.** A missing GGUF tensor fails the task and asks for a re-download. Safetensors and MLX packs are not loaded.

---

## 2. Dependency direction

```
Vue
  → danqing-api
  → pipeline          phases, no family match
  → families/mod.rs   the one factory match
  → one family        DiT, weight table, readout policy, denoise loop
  → common            Qwen3, AutoencoderKL, PNG, schedules
  → runtime           GgmlContext, GraphBuilder, ggml operators
  → danmo-ggml-sys
```

`common` and `runtime` do not import `families`. A family does not import another family.

---

## 3. Layout

| Path | Role |
|------|------|
| `native/crates/danmo-server` | Axum API, `danqing-api` |
| `native/crates/danmo-engine` | Engine |
| `native/crates/danmo-engine/src/pipeline/` | Phase order. `denoise.rs` is the Flux2 step loop |
| `native/crates/danmo-engine/src/families/mod.rs` | `FamilySpec` rows and `run_dit_image` |
| `native/crates/danmo-engine/src/families/<id>/` | That model's graph, weights, and policies |
| `native/crates/danmo-engine/src/common/qwen3/` | Qwen3 tokenizer and tower |
| `native/crates/danmo-engine/src/common/vae.rs` | AutoencoderKL decoder |
| `native/crates/danmo-engine/src/common/png.rs` | Uncompressed RGB PNG |
| `native/crates/danmo-engine/src/common/ops.rs` | Sigma schedules, normal noise |
| `native/crates/danmo-engine/src/runtime/` | `GgmlContext`, `GraphBuilder`, flash attention, RoPE, group norm, nearest ×2, preview conv |
| `native/crates/danmo-ggml-sys` | ggml bindings. The only place that calls ggml |

---

## 4. Shared code and family code

Shared modules take parameters. They do not branch on `flux2` or `z_image`.

| Piece | Where | What the caller passes |
|-------|--------|------------------------|
| Qwen3 tokenizer and tower | `common/qwen3` | 1-based layer indices, `thinking: bool`, output width |
| AutoencoderKL decode | `common/vae` | `VaeSpec` (channels, block widths, groups, scaling, shift, mid attention) |
| PNG | `common/png` | width, height, RGB bytes |
| Linear, static-shift, and empirical-mu sigmas; Flux2 timestep; N(0, 1) noise | `common/ops` | steps, shift, image sequence length, sigma, seed, length |
| ggml operators | `runtime` | tensor shapes |

The caller owns the policy:

| Policy | Who decides |
|--------|-------------|
| Which Qwen layers, and whether thinking is on | The family. Flux2 passes layers 9, 18, and 27 with thinking off. A short tower whose `hidden * layers` equals the joint width passes every layer. Z-Image passes the penultimate layer with thinking on. |
| VAE numbers | The family. Z-Image passes 16 channels, blocks 128/256/512/512, 32 groups, scaling 0.3611, shift 0.1159. Flux2 passes the same blocks after its own unpack, with 32 channels, scaling 1, shift 0. |
| Scheduler name | The family accepts the names its registry lists and calls the matching function in `common/ops`. Flux2 runs empirical mu for both `flow_match_euler` and `flow_match_euler_discrete`, and embeds `sigma * 1000` as cosine then sine. Z-Image uses those two names as one scheduler: static shift unless `use_empirical_mu` is set. `linear` keeps the resolution shift. |
| Negative prompt | The family. Z-Image requires the same token count and a guidance other than 0. Flux2 rejects a non-empty negative. |

These stay in the family because one shared function would be a pile of switches:

- **Denoise.** Flux2 uses `pipeline/denoise.rs`: velocity is `[channels, seq]`, the step is `x + dt * (u + g * (c - u))`, and guidance near 1 skips the second pass. Z-Image steps `x + dt * (c + g * (c - u))` on `[W, H, C]`, and guidance 0 is one pass. The Euler child graph is built by the family that owns that rank.
- **1×1 preview.** Both families can decode a file whose only tensor is `decoder.conv.weight`. The latent layout differs, so each family keeps its own preview. Both call `common/png`.
- **Flux2 packed latents.** The DiT latent is 128 channels. The family applies `bn.running_mean` / `bn.running_var`, unpatches groups of 4 channels into a 2×2 block, applies scaling 1 and shift 0, then `post_quant_conv`. The shared decoder then sees 32 channels. That front end stays in `families/flux2` because it is not an AutoencoderKL parameter.
- **DiT graph and weight table.** Each family.

A later model that needs the Qwen3 tower calls `common/qwen3` with its own layer list. It does not import `families/flux2`. A later model that needs this AutoencoderKL calls `common/vae` with its own `VaeSpec`. A different VAE topology stays in that family and still uses the runtime conv, group norm, and nearest-upsample operators.

---

## 5. What runs today

Shape on `FamilySpec`:

| Shape | Entry | Built graphs |
|-------|--------|----------------|
| **A — `DitPipeline`** | `pipeline::image_create` → `run_dit_image` | `flux2` (MM-DiT), `z_image` (DiT) |
| **B — `JobPipeline`** | `pipeline::upscale` | none; fails after the bundle check |
| **C — `GenerationFacade`** | video / audio routes | none; fails after the bundle check |

Any other registered family passes the bundle check and then fails with `has no ggml graph yet`. Edit, upscale, ControlNet, LoRA inference, and LoRA training are not built.

`flux2` text is the Qwen3 tower. `flux2` pixels use the packed-latent front end and then the shared AutoencoderKL graph when the file contains `decoder.conv_in.weight`. `z_image` pixels use that same decoder when `decoder.conv_in.weight` is present. Either family keeps the 1×1 preview when the VAE file is only `decoder.conv.weight`. Any other VAE file fails.

One graph is reserved and replayed. Steps replace inputs. The host reads the latent after the last step.

---

## 6. New model

1. Add the registry row in `default_config/models_registry.json` (`family`, `engine`, `actions`, `backends`, bilingual `name` / `description`, GGUF version keys). Do not invent a download URL.
2. Add a `FamilySpec` row. Wire `execute` only from `families/mod.rs`.
3. Put the DiT, the weight-name table, and the readout policy in `families/<id>/`. Do not import another family.
4. Call `common/qwen3`, `common/vae`, `common/png`, or `common/ops` when the algorithm is the same and the difference is a parameter list. Do not add a family name inside those modules.
5. Keep the denoise loop in the family when the tensor rank or the CFG formula differs from the loop that already exists.
6. Missing tensors, unknown schedulers, and actions with no graph fail the task.
7. `cargo test --manifest-path native/Cargo.toml --offline -p danmo-engine --lib`

---

## 7. Weights and backends

A bundle directory holds `transformer.gguf`, `text_encoder.gguf`, `vae.gguf`, `tokenizer.json`, and `bundle.manifest.json`. Shards (`role-00001-of-000NN.gguf`) are mmap'd per file. Duplicate tensor names across shards fail. FLUX.2 Klein 4B and Z-Image version installs also download the published text encoder, VAE, and tokenizer into that version directory. The loader still reads the GGUF roles.

Version keys are `bf16`, `q8_0`, and `q4_k`. Inside a quantized file, norm, bias, and modulation stay F32 or F16. The VAE decoder reads F32.

`DANQING_GGML_BACKEND=cpu|cuda|vulkan|metal`. The cargo feature has to be on. Context init fails when the feature is missing.
