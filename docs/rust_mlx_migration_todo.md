# Rust MLX 迁移待办

架构在 [`engine_architecture.md`](engine_architecture.md)。数值证据 JSON 在 `tests/parity/evidence/`

执行分支：`feat/rust-mlx-c-migration`。

本文只记还要做什么、做到哪、怎样才算完成。不在这里重写运行时设计。

## 基线

- ggml Rust 基线：`67495d3353d64f1ffd605e3586d0eea74837aebe`。
- 行为基线 `main`：`ab4456cb1a712b9e8a56a0629862170c967bef52`，本地路径 `/Users/nil.luo/Workspace/coding/danmo-make`。
- Registry 已从该 `main` 恢复：92 个 model card，22 个 runtime family。52 个 card 有 action，展开成矩阵里 216 个 model/version/action。40 个无 action 的组件 card 还没进矩阵。
- 量化源已定稿为魔塔 `OsaurusAI/Z-Image-Turbo-mflux-4bit` 与 `-8bit`。不要改回小型 HF 镜像。
- `main` 是行为基线，不是不可改的数学。和上游冲突时记下差异、补回归，再决定兼容还是修正。换基线 commit 必须写下 registry diff、contract diff 和受影响的条目。
- 有意偏差：生产权重保持 checkpoint 原 dtype。`main` 把 BF16/FP16 切进 F32 参数是错误设计，不迁。官方本身就是 F32 的 tensor 保持 F32。

## 更新规则

阶段状态只用 `pending` / `in_progress` / `done`。矩阵状态只用矩阵里的 `status_values`。两者不要混用。

- 开始一项，把对应阶段改成 `in_progress`。
- 完成时写下 commit、命令、平台、MLX 或 CUDA 版本、fixture。失败或没跑过的项保持原状态。
- 没有 CUDA 机器时，CUDA 列保持 pending。macOS 构建不能代替。
- 一个 family 通过不等于它的每个 model、version、action 通过。
- 不因为工期删掉 model、action、平台或训练。偏差要写下原因和决定。
- 某个 family 的 MLX 路径点亮时，同一提交删掉它的 ggml 实现。不要长期留三套。

## 阶段

| 阶段 | 状态 | 还剩什么 |
|---|---|---|
| P0 mlx-c 与 Z-Image 切片 | in_progress | 量化 encode 热性能、生产路由 E2E、executor 字节预算与心跳、custom kernel、干净打包、CUDA runtime |
| P0H main 收割 | pending | 治理与缓存 revision、submit gate、fixture 转录、`main` 侧停止把低精度权重放大成 F32 |
| P1 控制面 | pending | 真实 DB fixture、实时 SSE、鉴权、EngineRouter、核心路由 E2E、registry 启动校验（server adapter 家族分支已消除） |
| P2 MLX runtime | pending | pipeline/session 去掉 ggml 类型、attention 分派、字节预算、TeaCache 原语、panic/OOM/cancel；DANQING_MLX_DEVICE 或删文档；mlx-cuda smoke（ltx25 已全部上 executor） |
| P3 common | pending | 第二个使用方出现后再抽的 tokenizer、VAE、媒体和 scheduler fixture（guard 扩覆盖、ltx 孤儿删除已完成） |
| P4 图像核心 | in_progress | **Flux2 已收口**：16/16 `e2e`+画质 `metal_parity`+`route`；fp16 create 热门 4/4；fp16 rewrite / `mlx-q`* 热门 `not_applicable`（路径不对称 / AGENTS 量化政策）。下一刀 Qwen-Image 或 Flux1 |
| P5 图像长尾 | in_progress | Flux1 各结构变体，以及 fibo、ernie、cogview4、hidream、step1x、boogu、longcat；esrgan 已落地 Metal，待性能/CUDA 验收 |
| P6 音频 | in_progress | ACE 与 `main` 的成歌 parity、DiffRhythm。FSQ 翻唱编码已接到 `task=cover` |
| P7 超分 | in_progress | SeedVR2 视频 7B MLX 已打通并有数值/E2E 证据；仍待性能门、长任务、3B/图像和 CUDA |
| P8 视频 | in_progress | LTX25 数值与配置基础切片已进树；仍待 Wan、LTX25 模型/两阶段会话、Hunyuan、MiniMax H3、LongCat Avatar |
| P9 外围、assistant、训练 | in_progress | 安装与画廊、本地 assistant。**训练骨架已进树**（假 latent probe）；缺 VAE/TE cache、可微 DiT+LoRA、真图 E2E |
| P10 退役 | in_progress | 产品 Python 树 / thin uvicorn / parity 脚本已删。仍待：默认切满 Rust MLX、ggml 计算栈退役、干净环境安装测试 |

P0 已落地、不要重做的部分：源码 Metal 构建；Z-Image fp16 create 在 Metal 上的 parity、热性能和峰值显存；mlx-q4/mlx-q8 的 parity 和峰值显存；会话生命周期；safetensors create/rewrite 进进程内 executor。量化 `metal_performance` 没有点亮。证据文件名在 `tests/parity/evidence/zimage-*2026-11-18.json`。

P6 已落地、不要重做的部分：ACE XL 的 MLX 生成与 cover / cover-nofsq / repaint / lego / extract / complete 会话；5Hz LM 的格式化采样和时长规划音频码；FSQ tokenizer 把原曲潜变量编成码再 detokenize 成 DiT 条件；歌词 cross-attention 时间对齐；LoRA 走共享 `LinearWeight` 残差。还没有和 `main` 做逐条 parity，所以阶段不能标 done。

## 验收门

每一条 model/version/action 都要单独过，Metal 和 CUDA 分开记。

- 数值：token、scheduler、conditioning、噪声、关键 step、最终 latent、decode 分段比。固定注入噪声和固定 seed 各测一次。音频看波形、时长和非静音。视频看帧、fps、时长和音轨。不要求 Metal 和 CUDA 逐 bit 相同。
- 性能：同一硬件、同一 MLX、同一权重和缓存状态。热运行 p50 延迟不超过 Python 的 105%，或吞吐不低于 95%。峰值内存原则上不超过 Python 的 105%，对照的是 checkpoint 原 dtype，不是被放大成 F32 的那份。冷启动和首次 compile 单独报。跑失败之后不许放宽阈值。
- E2E：提交、进度、取消、成功和预期失败、产物、lineage、重启、缓存命中、换模型和 unload。
- 训练另计：loss 和每个可训练 tensor 的梯度、冻结参数无梯度、checkpoint 往返、输出的 LoRA 能被 Rust 推理加载。

## 矩阵还要补

- `python_baseline_status` 的取值不在受控枚举里。
- 227 条 `architecture_variant` 仍是 `unclassified`。
- 38 个组件 card（LoRA、ControlNet、Redux、depth-pro、VEDA 等）没有条目。
- 峰值显存字段还没建。Z-Image 已点亮的行要和仍 pending 的行拆开，不能整卡标通过。
- `out/parity-zimage-*` 在 gitignore 的 `/out/` 下。摘要写进矩阵，或把 fixture 放到能提交的路径。

## 各阶段工作

### P0 剩余

- Linux CUDA Release 与真实 CUDA runtime。
- 最小 macOS 分发：dylib、metallib、rpath、deployment target、codesign。
- 算子探针补齐：超过 32K packed rows 的 QMM 已有探针；conv3d、conv transpose、FFT/STFT、sliding mask、一个 Metal custom kernel 和对应 CUDA kernel 还没有。
- 量化 encode 的热性能门。不要为此重编 Qwen3 塔。

### P0H

- ModelCache / TextEmbeddingCache 的 revision 规则扩到全部 shard、config、tokenizer、manifest 和 LoRA。
- 把 `main` 的 `_assign_param_tensor` 改成按 checkpoint dtype 赋值，并重做因此失效的 fixture。

### P1

- HTTP DTO、领域 contract、`EngineRequest` 的转换收成机械 adapter。
- 用一份能力数据驱动路由、registry projection 和 engine gate。
- 用真实 main v3 DB fixture 对字段、时间、状态、priority、分页、索引和 lineage。
- scheduler 补 EWMA、ETA、队列带、孤儿恢复、预览、取消和优先级。
- SSE 改成实时事件，保留顺序和重连。
- loopback / remote / MCP 的鉴权与 main 对齐。设置里有 access key 不等于中间件已经完成。
- ~~Python 纯推理 worker~~：**不做**。产品 Python 已删除；缺能力显式失败，禁止回落 Python。
- **架构差距（2026-11-19 评估，当日部分落地）**：~~消除 server adapter 家族分支~~ **已做**——视频 kind 门改 `ensure_available`；`validate_structural_guide` 走 `FamilySpec.structural_guide`；训练提交 / register 的 family 从 catalog + `FamilySpec.lora_training` 带入；训练入口中立化为 `danmo_engine::run_family_lora_training`（dispatch 在 `families/mod.rs`）；`engine_request` 映射下沉 `danmo-engine/adapter.rs` 供 server 与 CLI 共用。仍待：启动时校验 registry JSON；落地或明确砍掉 `EngineRouter`。

### P2

- `DitImageSession` / `FacadeSession` 去掉对 `GgmlContext` 的依赖，改成后端中立。
- attention 统一成原生、通用、custom、unavailable 四条，缺的显式失败。
- 缓存除条目数外有字节预算。health 暴露 entries、hits、misses、bytes。
- 取消、OOM 和 worker 崩溃的回归。
- **架构差距（2026-11-19 评估，当日部分落地）**：~~ltx25 接入~~ `mlx_executor` **已做**——生成路径与 install hook（ingest）都在 executor 上，生产代码无 `Stream::gpu()` 私有流；`run_install_hook` 改收 owned 闭包。msr/bundle 绕过权重层的 `SafeTensors::load` / `quantize_affine` 已收进共享层（`weights::probe_affine_quantization` / `read_component` / `pack_affine`），guard 转绿。仍待：实现或删除文档中的 `DANQING_MLX_DEVICE`；MLX active / cache / peak / device limit 进 health（现 `mlx_memory_limit: 0` 桩）；`mlx-cuda` 至少加编译期 smoke gate。

### P3

按真实的第二个使用方再抽，不预建空目录：

- schedulers、norm、embedding、CFG。
- Qwen3、Qwen2.5-VL、Qwen3-VL 的参数化 trunk。
- umT5、Gemma、CLIP、T5、BERT、byT5、whisper 里真正共享的部分。
- VaeSpec、2D/3D tiling 和 stream cache（拓扑相同仅配置不同才进 `common/vae`；因果 / 3D 产品 VAE 留在 family）。
- PNG/JPEG/WebP、WAV/FLAC/MP3/OPUS/AAC、ffmpeg。
- DiffRhythm 中文 G2P 走 ONNX Runtime C API，去掉 bundle 里的 Python。
- **架构差距（2026-11-19 评估，当日落地）**：~~guard 扩展~~ **已做**——`architecture.rs` 覆盖 `weights/`、跨家族 import、`danmo-mlx-sys` 直连（7/7 绿）；`families/ltx/` ~~孤儿树已删~~；`lib.rs` ~~根部家族训练导出收敛~~为中立 `run_family_lora_training`。`weights/bundle.rs` 的 gguf 布局 helper（`assert_hunyuan_bundle` 等）按发布格式命名保留，随 P10 ggml 退役一起清。

### P4

- `z_image`：
  - **已落地**：create/rewrite safetensors 会话；共享 `attach_lora` + `remap_lora_module_prefix`；user LoRA 注册写入完整 `inference`（含 `extra_adapters` / steps / guidance / weight）；推理侧中立展开 `inference_stack`（客户端不注入配方 companion）。
  - **真实图 LoRA E2E（2026-11-19，Metal）**：用 lyt 数据集身份 LoRA `user-lora-a97835981f18`（李雪01 / trigger `Lyt`，来自既有真训 `tsk_aa0ce54848f9e4dd060317f7`）在 `z-image-turbo` fp16 上 create：`768x1280`、9 步、guidance 0、seed 42、weight 0.9；任务 `8afd0b5f-…` 完成，产物 `tests/parity/evidence/zimage-lora-lyt-p4-2026-11-19.{json,png}`。`bind_installed_loras` 正确解析目录型 `local_path`。**仍不算矩阵勾选**（无 parity / 热性能门；本条 LoRA 为 turbo scheme，`extra_adapters` 为空）。
  - **DistillPatch companion 叠载 E2E（2026-11-19，Metal）**：HF `timeout: connect` 后把 `z-image-turbo-distillpatch-lora:bf16` 的 registry `source` 改为 **modelscope**（同 repo `DiffSynth-Studio/Z-Image-Turbo-DistillPatch`）并装入；用元数据行 `user-lora-lyt-distill-e2e`（同 李雪01 权重 + `extra_adapters: DistillPatch`）只提交用户 LoRA，服务端展开双 adapter；任务 `3e534cf4-…` 完成（8 步 / seed 42），产物 `tests/parity/evidence/zimage-lora-lyt-distillpatch-p4-2026-11-19.{json,png}`。**仍不算矩阵勾选**。
  - **rewrite Metal parity + E2E（2026-11-19）**：`z-image-turbo` fp16；main fixture `out/parity-zimage-rewrite-{16x16,256x256}.json`；cargo `real_zimage_rewrite_matches_main_latents_and_pixels`（mlx-metal）对 blend/velocity/latent/RGB **逐位一致**；API `17481890-…` rewrite completed（256² / 8 步 / fidelity 0.6 / init=4）；证据 `tests/parity/evidence/zimage-rewrite-turbo-fp16-p4-2026-11-19.{json,png}`。矩阵该行 `metal_parity`/`e2e` 已 passed。
  - **rewrite 热性能（2026-11-19，Metal）**：256² / 8 步 / init=4；main warm fixture `out/parity-zimage-rewrite-256x256-warm-main.json`；cargo `real_zimage_rewrite_warm_passes_match_main_and_report_performance`；p50 rust/main **denoise 0.995 / vae 0.852 / total 0.991**（≤105%）；证据 `tests/parity/evidence/zimage-rewrite-turbo-fp16-warm-gate-2026-11-19.json`。矩阵 `metal_performance` passed（256²，非 768×1280）。
  - **ControlNet union create E2E（2026-11-19，Metal）**：`mlx/control.rs` forward（33ch、hint、refiner/layers、`before_proj`/`after_proj`）；`remap_control_name` 加载；`structural_guide.weight>0` 走 eager DiT + hint。任务 `a1699df0-…`（256² / 4 步 / canny）completed；证据 `tests/parity/evidence/controlnet-union-e2e.json`。矩阵 `e2e`/`route` passed；**parity / 热性能 pending**。仅 **control-only ≤512²** 可辨（768² control-only 仍差）；显式 canny 资产 vs 引擎 Sobel 边图在 inpaint 上无本质差别。
  - **ControlNet 配置驱动 + Union inpaint（2026-11-19）**：registry/UI 配置驱动；`build_control_context_inpaint_nchw` + preserve 蒙版；去掉 decode 硬合成。任务 `538284ce-…`（256² / 4 步）**路由完成**；证据 `controlnet-union-inpaint-e2e.json`。**不等于画质验收**： companion 768×1280（如 `ast_94de97aa…`）全噪点；512² inpaint 马赛克（8 步 / 25 步 / 显式 canny 均如此）。
  - **ControlNet forward 修复（2026-11-19 续，已验证）**：根因是对 `ZImageControlTransformerBlock` hint 栈的误读——`all_c.pop(-1)` 会**移除**上一个原始输出，栈实为 `[after_proj(c0)…after_proj(ck), ck]`，注入 hint = 每块一个 after_proj 输出，控制流 = 原始块输出。修复：hint 语义回正；`refined_control` 改为原始 `c1`（原误用 after_proj 输出）；去掉错误的「共享 noise_refiner 跑 control 流」；结构采用 VideoX 单遍式（refiner hint 注入主 noise_refiner，与 8steps 蒸馏训练框架一致）；preserve mask 下采样 box→nearest。**Diffusers parity 门**：`real_control_branch_matches_diffusers_fixture`（torch bf16 参考，真权重固定输入）——embed/refiner/caption/15 层 hint 全部 rel_l2<0.9%、cos≥0.99996，证据 `tests/parity/evidence/controlnet-union-forward-fix-2026-11-19.json` + fixture `studio-workspace/parity-fixtures/zimage-control-diffusers-slim.safetensors`。**E2E**：512² control（`5eaac06a-…`）、512² inpaint 红→蓝（`e81d5092-…`）、768×1280 control（`5158ef94-…`）、512² companion 自动边图（`d0dda1ae-…`）、768×1280 inpaint（`dc9a493f-…`）均出图正常（修复前 512²+ 全为马赛克/曝白）。**另**：未蒸馏 2.1 权重 A/B（25 步）不能绕过旧 bug，已留存 `models/ControlNet/z-image-union-2.1-hold/`。
  - **局部重绘保色搁置（2026-11-19）**：去掉同源 Sobel、权重 0.9→0.65 后重跑 `ast_6709f3d6635d43d29be3`，chroma 仍约 7，与改前黑白图相同。该实验已回滚。companion 重绘不再作为当前产品主线。
  - **Flux2 Klein（2026-11-19，Metal，已出图，矩阵不勾）**：create 和 rewrite 走 `families/flux2/mlx/`，对照 `danmo-make` main（性能用 `bin/danqing-generate` CLI，避免占 7800）。DiT 宽度、层数和 Qwen3 文本塔从 checkpoint 读（4B 是 Qwen3-4B，9B 是 Qwen3-8B：8 double / 24 single，joint 12288）。9B mlx-q4 在 `models/Image/flux2-klein-9b-mlx-q4`，没有 `guidance_linear_`*。create：4B `a0811625-…`、9B `8e164544-…`（512² / 4 步 / guidance 1 / seed 1）都出了红苹果。rewrite：编码器每层是 2 个 ResNet，解码器仍是 3；9B `7e2486f5-…`、4B `027f21f4-…`（fidelity 0.35，init 1）把红苹果改成绿苹果。guidance 2 的任务 `1395e5f8-…` 失败，原因是这包没有 guidance embedder。缺 `bundle.manifest.json` 不再当成边长上限；安装结束时若目录里还没有清单、且下到了 transformer safetensors，会补写。object-remove 只留 Comfy 那份 safetensors（具名文件头越过文件尾）。BFL 键 `img_in` / `txt_in` / `time_in.*` / `final_layer.linear` / `*.lin` 映射到 Klein 图参数。4B create `631d805f-…`（weight 0.8）出了红苹果 `ast_c35a1d6e…`。同一张图加品红块后 rewrite `6dde0b4c-…`（prompt 为卡片原文，weight 1.1，fidelity 0.35）出了红杯子 `ast_fd1e118c…`，没有去掉高亮物体。界面默认去噪（strength 0.3，fidelity 0.7，init 2）`5acf44ac-…` 品红块几乎还在 `ast_8faf76b9…`。默认尺寸：4B 1024² `3774af4d-…`、4B 768×1280 `fa2b5372-…`、9B 768×1280 `c90daa0d-…` 都是正常红苹果。负向词、retouch、未知 scheduler、9B 挂 4B LoRA 都显式失败。base fp16 从魔塔 `black-forest-labs/FLUX.2-klein-base-4B` 与 `…-base-9B` 装入。Diffusers 键 `timestep_embedder` 和 `to_out.0` 在加载时摊平；`bn.num_batches_tracked` 的 int64 计数不进权重。4B create `42021d95-…`（512² / 50 步 / guidance 1 / seed 1）红苹果 `ast_27a2b02b…`。9B create `14f8cf7e-…` 红苹果 `ast_85dc413d…`。4B rewrite `6d752099-…`（fidelity 0.35）绿苹果 `ast_92016b74…`。两包都没有 `guidance_linear_*`，guidance 4 的 `f587df0a-…` 失败。注册表默认 guidance 改为 1。**parity 推进（续）**：create 噪声改为 MLX `RandomKey` NCHW→pack（与 main `seeded_randn` 逐位一致）；Qwen3 `added_tokens` 编码不再只收 `special:true`，修复官方包里 `special:false` 的 `<think>`/`</think>` 被 BPE 拆碎（token ids 与 transformers 对齐）；latent/temb 对齐 bf16。4B mlx-q4 512²/4 步/seed1 vs main CLI：像素 MSE **10.35**（此前 LCG≈4402、仅 RNG≈443）；caption maxabs≈15.8 / mse≈0.0014（仍非逐位）；warm p50 Rust debug API **2.55s** vs main resident **2.24s**（≈114%）。证据 `tests/parity/evidence/flux2-klein-4b-mlx-q4-create-progress-2026-11-19.json`。rewrite 绿苹果 `ast_61fdf82c…`。**矩阵仍不勾**。
- **Flux2 验收口径调整（2026-11-19）**：与 `danmo-make` 像素 MSE 不作硬门；以 BFL/fal 权威用法与画质为准。object-remove LoRA 的 `base_model` 保持蒸馏 `flux2-klein-4b`（HF 卡片），默认 scale **1.1**；fal 固定提示词 `Remove the highlighted object from the scene`。误挂到 base-4b 时出现灰阶拖影产物。矩阵仍不勾。
- **Flux2 Klein edit = reference conditioning（2026-11-19 续）**：根因是 rewrite 曾按 fidelity 噪声混合做 img2img，而 BFL/diffusers `Flux2KleinPipeline` 是把干净 source 潜变量拼到序列上（`T=10,20,…`），目标从全噪声起步，每步只对目标 token Euler。改完后：品红高亮苹果 + object-remove LoRA（4 步 / scale 1.1 / seed 42）`02273e20-…` → 空桌 `ast_dc576284…`；无 LoRA 同提示仍保留品红苹果；无 LoRA「make the apple green」`541310c8-…` 绿苹果。`source_fidelity` 仍收但显式记日志忽略。
- **Flux2 已装行矩阵推进（2026-11-19 release）**：证据 `tests/parity/evidence/flux2-installed-e2e-warm-2026-11-19.json`。4b mlx-q4 create warm p50 **1.62s / main 2.24s = 72%**（≤105%）；rewrite warm ~3.01s。9b mlx-q4 与 base 4b/9b fp16 create/rewrite E2E 出图。矩阵 8 行 `e2e`+`metal_parity`(画质/权威口径)+`route` → passed；量化行 `metal_performance` 按 AGENTS 不勾。未装版本与 base 相对 main 的正式性能门仍缺。
- **Flux2 并行收敛（2026-11-19 晚）**：蒸馏 4b fp16 / base4b fp16 create warm 已过 `metal_performance`（rust/main CLI 35.7% / 43.8%）。1024² q4 create OK。object-remove 在 4b fp16 可去掉物体但比 q4 软。证据 `tests/parity/evidence/flux2-parallel-converge-2026-11-19.json`。
- **Flux2 满血验收（2026-11-19，9b fp16 已装）**：蒸馏 9b fp16 create `88e836c4-…` / rewrite `39384771-…`（reference_conditioning）出图；create warm **2.48s / main 3.98s = 62%**；base9b create warm **4.77s / main 8.45s = 56%**。矩阵 16 行 `e2e`+`metal_parity`+`route` 全过；fp16 create 热门 4/4。证据 `tests/parity/evidence/flux2-9b-fp16-fullblood-2026-11-19.{json,png}`。
- **Flux2 收口（2026-11-19）**：修 main `session_persist` 误传 ctx 后采到 rewrite warm；因 main 仍走 fidelity img2img、Rust 走 BFL refcond，rewrite 热门标 `not_applicable`；`mlx-q`* 热门按 AGENTS 标 `not_applicable`。矩阵无 pending。证据 `tests/parity/evidence/flux2-fullblood-closeout-2026-11-19.json`。
- `flux2`：**已收口**。Klein 4B/9B/base create/rewrite E2E+画质+路由已勾；fp16 create 热门已过；rewrite/量化热门 N/A。蒸馏包 guidance 必须是 1。像素相对 main 仍为观测项（9b MSE~5.5）。GGUF 图已删除。
- `qwen_image`：create、edit、edit-2511、lightning、多参考、LoRA。Qwen2.5-VL 进 common；108 token crop 和条件拼接留在 family。

### P5

`flux1` 的 Fill、Redux、Depth、Kontext 按 model card 分结构变体。`fibo`、`ernie_image`、`cogview4`、`hidream_o1`、`step1x_edit`、`boogu_image`、`longcat`、`esrgan`。LongCat BSA 要分别证明 Metal 和 CUDA。没有权重的条目保持 unavailable。

- **Real-ESRGAN x4+（2026-11-20，Metal，实现与 E2E 已落地）**：`families/esrgan/` 按本机 `config.json` 严格加载 702 个 MLX OHWI FP16 张量，完成 23 个 RRDB、最近邻 x4、官方右/下 10px 反射填充、10px tile context、2× Lanczos 输出、取消/进度和 PNG lineage。独立 PyTorch FP32 参考前向 `max_abs=0.005783975`、`MSE=8.743e-7`；合成权重 6 项测试通过。真实 API 4×/tile16 任务 `aff95691-…` 输出 128×96（9 tiles），2×/tile0 任务 `c9ebfce6-…` 输出 64×48，均完成且 lineage 正确。热性能、峰值内存、CUDA 和矩阵勾选仍待做，不能据此把 family 标为全部验收完成。

### P6

- ACE：`task=cover` 会把原曲 VAE 潜变量按 5 帧一窗做 FSQ，再 detokenize 成 25 Hz 条件。create、cover-nofsq、repaint、lego、extract、complete、5Hz 两阶段、歌词对齐、共享 LoRA 已接上，缺的是和 `main` 的 parity，以及 PMI 的正式门。
- 复用边界：LM 的 FSM、歌词和 DTW 留在 family；WAV 和 ffmpeg 的通用部分进媒体层；采样器和 KV cache 只有 assistant 也要用时才抽出去。
- DiffRhythm：BERT 可共享，G2P 走 ONNX，mulan 按 `main` 转录。

### P7

SeedVR2 图像和视频超分。tiling、因果时间缓存、wavelet/LAB、ffmpeg 分块可复用，但 common 里不写 SeedVR2 分支。conv3d 在两个后端上都要有性能和长任务内存数字。

- **架构差距（2026-11-19 评估，当日落地；2026-11-20 更新）**：`seedvr2` ~~shape 元数据~~ 已改为 `GenerationFacade`（与 Facade driver 一致）；`esrgan` 也已接入 `GenerationFacade`，`JobPipeline` 当前没有 family 使用。
- **SeedVR2 视频 7B MLX（2026-11-20，Metal，进行中）**：视频会话不再走 ggml，原生 safetensors FP16 DiT（36 层）与因果 3D VAE 在进程级 `mlx_executor` 上完成分块编码、单步去噪、解码和 MP4 回封装。已修正 ResNet 跨时间 GroupNorm 与 mid-attention 单帧归一化、7B `rope_on_text=false` 的单视频 RoPE、temporal/spatial pixel-shuffle 轴序、带音轨 mux 丢尾帧；不足 5 帧的输入补到 5 帧后只输出原帧数。真权重固定输入下 latent/noise/velocity/restored/decoded 相对 L2 分别为 `0.002889/0/0.004767/0.005433/0.005248`（断言均 `<0.01`），DiT/VAE 分层探针同门。API 任务 `69a70d5c-…` 输出 32×32、24fps、5 帧、0.208333s、H.264+AAC；`df1c4b70-…` 输出 4 帧、0.166667s，lineage 正确。证据 `tests/parity/evidence/seedvr2-video-7b-mlx-2026-11-20.json`（含命令与 fixture）。**尚未点亮完成门**：热 p50/峰值内存、空间大 tiling 与多分块、取消/重启/缓存/unload、3B 和图像 MLX、CUDA；时间流缓存尚未迁移。

### P8

- Wan：t2v / i2v / 5B / 14B / MoE / distill / LoRA 分开验。umT5 的 Viterbi 分词优先复用。
- LTX25：两阶段音视频、per-token timestep、mask、IC-LoRA。只共享真正相同的组件，family 之间不互相引用。
  - **LTX25 进行中（q4 T2V 已打通真实浏览器链路）**：独立 family 按 `GenerationFacade` 定界；中立音视频几何/RoPE/数值算子在 `common`，checkpoint 拓扑、固定 8+3 调度、联合 A/V denoise、upsampler、VAE 与 vocoder 留在 `ltx25`，未引入 Python 产品路径。真实 q4 Gemma、48 层 DiT、双阶段 T2V、A/V 解码和 MP4 持久化已运行；已修复 Gemma 左 padding、DiT 最终 LayerNorm，以及 RoPE 的全局频率网格/head 布局和大相位角 MLX 三角函数计算。同一 Python fixture 下 RoPE 最大误差约 `0.00195`，首层 video/audio velocity 平均误差约 `0.00291/0.00377`；补齐 keyframe 绝对位置嵌入后，2026-11-20 同权重参考 fixture 的 48 层 video/audio 平均误差约 `0.0290/0.0305`。
  - **q4 T2V 验收与剩余项**：早期浏览器任务 `5950feec-2340-4c31-a599-5eefd6a80e11` 暴露了网格伪影和音频漂移。2026-11-20 已修复 keyframe 绝对位置嵌入、二阶段 ancestral 采样与噪声种子、音频 VAE 布局、WAV 声道交织、上采样/视频 VAE 时间维 padding 及 vocoder 插零滤波。同权重参考对照中 upsampler 平均误差约 `0.0000023`、视频 VAE 为 0、vocoder 约 `0.0245`；原图生视频任务 `98a73c0e-f7ad-4e03-9e61-e6e1a48c8acf` 的同输入 121 帧复验已无网格，音轨 RMS 约 `-38.8dB`。仍不能宣称逐值 bit-parity 完成；性能/峰值内存、真实 70GB checkpoint 转换、CUDA 等保持原有门槛。
  - **满血推进中**：`VideoEncoder`（含 `SpaceToDepthDownsample`）已进 `video_vae`；`conditioning` 模块落地 prepend / target-suffix / MSR 负时间偏移；`VideoGenerationRequest` 增加 `media_references` + `adapters`；vocoder 双布局加载。
  - **本轮**：DiT `forward_tokens_timed`；MSR 接线；MSR LoRA 模型库单独安装（魔搭）。**animate E2E**：`0bbfce6b-…`。**Ref2VA 静帧**：`7744fddc-…`；**Ref2VA 音频**：`caa0ab18-…`。**cancel E2E**：`51fa2be3-…`。**infer 已改走进程级** `mlx_executor`。**q8 满血 E2E（2026-11-19）**：create CLI cold/warm；animate CLI（q8 **23.95s** / q4 **16.12s**）；Ref2VA API `179a44df-…`（2 stills + `ltx25-msr-lora` → `ast_8e34c621…`）。ingest hook 补齐 channel-last（upsampler/audio_vae/`ups.`*）与 vocoder `{vocoder,bwe}` 摊平。证据 `tests/parity/evidence/ltx25-acceptance-2026-11-19.json`。矩阵 q8 create/animate `e2e=passed`（量化 `metal_performance=N/A`）。**收口（2026-11-19）**：4/4 行 `e2e`+`route`+画质 `metal_parity`；量化 `metal_performance=N/A`；vs main create warm **15.41/17.7s=87%**（观测）。证据 `ltx25-vs-main-q4-create-2026-11-19.json` / `ltx25-acceptance-2026-11-19.json`。仍待（非收口门）：峰值内存正式门、resident 缓存/卸载、cuda、逐值 bit-parity。
- Hunyuan：animate、byT5 字形、1080p SR。Qwen2.5-VL 走 common。SigLIP 等第二个视觉用户出现再抽。
- MiniMax H3：Qwen3-VL conditioner 不在 family 里复制。大 QMM 和长序列内存两个后端都要测。
- LongCat Avatar：whisper 前端抽成中立音频特征，数字人条件留在 family。

### P9

- 下载、components、manifest、workspace registry、user LoRA、画廊、metadata、MCP、CLI。
  - **CLI 已落地（2026-11-19）**：新 crate `danmo-cli` 出 `danmo-generate` / `danmo-edit` / `danmo-audio-generate` / `danmo-audio-edit` / `danmo-video-generate` / `danmo-video-edit`，命令面参考 danmo-make 基线（前缀按产品命名为 danmo）；bundle/LoRA 解析下沉新 crate `danmo-resolve` 与 server 共用；任务映射走 `danmo_engine::engine_request`。Metal 真机 smoke：z-image-turbo 512²/4 步出图、flux2-klein-4b rewrite 红苹果→绿苹果、z-image + `user-lora-a97835981f18` LoRA 出图。bin crate 经 `DEP_MLXC_LIB_DIR` 自写 LC_RPATH。仍待：音频/视频 CLI 真机验收、install hook 的 CLI 入口（`danmo-runtime-setup` 类）。
- 本地 assistant：qwen3 / qwen3_vl、chat template、视觉预处理、采样、KV cache、SSE、卸载。它用自己的 worker，不和分钟级 DiT 串在同一个 executor 上。
- **训练（进行中，**`3f2be72` **起）**：不恢复 Python trainer。
  - **已落地**：`danmo-mlx` `value_and_grad` + AdamW + `mx.checkpoint` 绑定；DiffSynth 对齐的 Flow-Match schedule/`training_weight`；hybrid 数据集 `train.jsonl`+`meta.json`（upload/list/health/import）；Z-Image/Turbo 命名 presets（`turbo-sft` / `turbo-diff` / `base-face`）；共享 train loop（假 latent smoke 保留为单测）；API `datasets` / `requirements` / `trainings` / `artifacts` / `register`（MLX feature 下 `ready:true`）；训练 UI 四步向导 + 完成态摘要/注册；注册校验 companion 已安装；推理不按配方名分支。
  - **满血训练管线（2026-11-21 落地，Metal）**：真 VAE latent cache + Qwen3 TE + caption refiner cache（逐图 + 空 caption dropout 变体）；完整 30 块 DiT 上可微 LoRA（F32 A/B 挂推理 `LinearWeight` residual，`add_loras` dtype 感知；每块 `mx.checkpoint` 闭包，反向重算控显存）；AdamW 全量 210 模块；checkpoint 定时导出 + resume（共享 LoRA loader 回读）；dense bundle 门禁（量化 tier 提交即 400）；`face_crop=on` 显式报错（无人脸检测），`auto` 记日志后中心裁剪；空 caption 走 ChatML scaffold 分词（生成侧空 prompt 门禁不变）。单测：checkpoint 梯度与朴素 `value_and_grad` 等价；toy DiT（1 块）CPU+Metal 训练 loss 下降、14 个因子梯度齐全。
  - **真图短训 E2E（已过，2026-11-21）**：9 图数据集 1024²、20 步（`metadata.iterations` dev override，日志明示）、rank 32 turbo-sft，encode cache 5s、约 9.5s/step；adapter（630 张量）经 `danmo-generate --lora` 中立推理出图，与无 LoRA 基线对比构图明显偏向数据集风格。证据 `tests/parity/evidence/zimage-lora-training-e2e-2026-11-21.json` + 对照 PNG ×2。
  - **仍缺（训练验收门）**：人脸检测/门禁真链路（`face_crop=on` 当前 fail loud）；loss/梯度按可训练 tensor 的证据文件；checkpoint resume 往返 E2E；2500 步全长跑（1024² 约 6.6h 预估）；CUDA 平台门；矩阵勾选。在此之前不要把训练行标 done。

### P10

- **已做**：删除产品 Python（空 `backend/`、parity 脚本、桌面 thin uvicorn / venv bootstrap）。
- **已过时表述**：旧「训练提交一律 501」仅适用于无 MLX feature 的构建；`mlx-metal` / `mlx-cuda` 构建下训练 API 可提交（当前仍是 probe，非产品完备）。
- 仍待：22 个 runtime family 达到发布要求后默认走满 Rust MLX；退役 `danmo-ggml-sys` / `native/third_party/ggml` / GGUF 安装逻辑（按矩阵，勿提前）。
- 干净的 macOS Metal 和 Linux CUDA 安装、生成、升级各跑一遍（训练真图 E2E 过门后再把训练纳入安装验收）。

## 未决

写进实现之前先确认：

1. MLX 推理是否拆成独立 worker 进程。驱动 abort 和一部分 OOM 会直接杀掉进程。若和 SQLite 写者、SSE、鉴权同进程，一次推理崩溃会带走控制面。

**已拍板**：不保留 Python 训练 worker；训练只走未来的 Rust MLX trainer。

## 明确不迁

- ggml 的计算图、GGUF 解析、多后端 auto、mmap 权重缓存本体。
- 把 MLX safetensors 转成 GGUF，或按 ggml 布局 repack。
- 生产路径上把稠密权重放大成 F32。
- 用 ggml 的近似结果充当 MLX 的数值实现。
