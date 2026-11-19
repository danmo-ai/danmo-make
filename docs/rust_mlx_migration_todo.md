# Rust MLX 迁移待办

架构在 [`engine_architecture.md`](engine_architecture.md)。能力矩阵在 [`rust_mlx_migration_matrix.json`](rust_mlx_migration_matrix.json)。数值证据 JSON 在 `tests/parity/evidence/`（Python baseline 脚本已删除）。

执行分支：`feat/rust-mlx-c-migration`。

本文只记还要做什么、做到哪、怎样才算完成。不在这里重写运行时设计。

## 基线

- ggml Rust 基线：`67495d3353d64f1ffd605e3586d0eea74837aebe`。
- 行为基线 `main`：`ab4456cb1a712b9e8a56a0629862170c967bef52`，本地路径 `/Users/nil.luo/Workspace/coding/danmo-make`。
- Registry 已从该 `main` 恢复：92 个 model card，23 个 runtime family。54 个 card 有 action，展开成矩阵里 227 个 model/version/action。38 个无 action 的组件 card 还没进矩阵。
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
| P1 控制面 | pending | 真实 DB fixture、实时 SSE、鉴权、EngineRouter、核心路由 E2E（不再保留 Python 推理 worker） |
| P2 MLX runtime | pending | pipeline/session 去掉 ggml 类型、attention 分派、字节预算、TeaCache 原语、panic/OOM/cancel |
| P3 common | pending | 第二个使用方出现后再抽的 tokenizer、VAE、媒体和 scheduler fixture |
| P4 图像核心 | in_progress | rewrite + ControlNet union create E2E 已过（256²）；下一刀：ControlNet parity/热性能 → LEMICA/latent refine；其后 Flux2 / Qwen-Image |
| P5 图像长尾 | pending | Flux1 各结构变体，以及 fibo、ernie、cogview4、hidream、step1x、boogu、longcat、esrgan |
| P6 音频 | in_progress | ACE 与 `main` 的成歌 parity、DiffRhythm。FSQ 翻唱编码已接到 `task=cover` |
| P7 超分 | pending | SeedVR2 图像和视频 |
| P8 视频 | pending | Wan、LTX、LTX25、Hunyuan、MiniMax H3、LongCat Avatar |
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

### P2

- `DitImageSession` / `FacadeSession` 去掉对 `GgmlContext` 的依赖，改成后端中立。
- attention 统一成原生、通用、custom、unavailable 四条，缺的显式失败。
- 缓存除条目数外有字节预算。health 暴露 entries、hits、misses、bytes。
- 取消、OOM 和 worker 崩溃的回归。

### P3

按真实的第二个使用方再抽，不预建空目录：

- schedulers、norm、embedding、CFG。
- Qwen3、Qwen2.5-VL、Qwen3-VL 的参数化 trunk。
- umT5、Gemma、CLIP、T5、BERT、byT5、whisper 里真正共享的部分。
- VaeSpec、2D/3D tiling 和 stream cache。
- PNG/JPEG/WebP、WAV/FLAC/MP3/OPUS/AAC、ffmpeg。
- DiffRhythm 中文 G2P 走 ONNX Runtime C API，去掉 bundle 里的 Python。

### P4

- `z_image`：
  - **已落地**：create/rewrite safetensors 会话；共享 `attach_lora` + `remap_lora_module_prefix`；user LoRA 注册写入完整 `inference`（含 `extra_adapters` / steps / guidance / weight）；推理侧中立展开 `inference_stack`（客户端不注入配方 companion）。
  - **真实图 LoRA E2E（2026-11-19，Metal）**：用 lyt 数据集身份 LoRA `user-lora-a97835981f18`（李雪01 / trigger `Lyt`，来自既有真训 `tsk_aa0ce54848f9e4dd060317f7`）在 `z-image-turbo` fp16 上 create：`768x1280`、9 步、guidance 0、seed 42、weight 0.9；任务 `8afd0b5f-…` 完成，产物 `tests/parity/evidence/zimage-lora-lyt-p4-2026-11-19.{json,png}`。`bind_installed_loras` 正确解析目录型 `local_path`。**仍不算矩阵勾选**（无 parity / 热性能门；本条 LoRA 为 turbo scheme，`extra_adapters` 为空）。
  - **DistillPatch companion 叠载 E2E（2026-11-19，Metal）**：HF `timeout: connect` 后把 `z-image-turbo-distillpatch-lora:bf16` 的 registry `source` 改为 **modelscope**（同 repo `DiffSynth-Studio/Z-Image-Turbo-DistillPatch`）并装入；用元数据行 `user-lora-lyt-distill-e2e`（同 李雪01 权重 + `extra_adapters: DistillPatch`）只提交用户 LoRA，服务端展开双 adapter；任务 `3e534cf4-…` 完成（8 步 / seed 42），产物 `tests/parity/evidence/zimage-lora-lyt-distillpatch-p4-2026-11-19.{json,png}`。**仍不算矩阵勾选**。
  - **rewrite Metal parity + E2E（2026-11-19）**：`z-image-turbo` fp16；main fixture `out/parity-zimage-rewrite-{16x16,256x256}.json`；cargo `real_zimage_rewrite_matches_main_latents_and_pixels`（mlx-metal）对 blend/velocity/latent/RGB **逐位一致**；API `17481890-…` rewrite completed（256² / 8 步 / fidelity 0.6 / init=4）；证据 `tests/parity/evidence/zimage-rewrite-turbo-fp16-p4-2026-11-19.{json,png}`。矩阵该行 `metal_parity`/`e2e` 已 passed。
  - **rewrite 热性能（2026-11-19，Metal）**：256² / 8 步 / init=4；main warm fixture `out/parity-zimage-rewrite-256x256-warm-main.json`；cargo `real_zimage_rewrite_warm_passes_match_main_and_report_performance`；p50 rust/main **denoise 0.995 / vae 0.852 / total 0.991**（≤105%）；证据 `tests/parity/evidence/zimage-rewrite-turbo-fp16-warm-gate-2026-11-19.json`。矩阵 `metal_performance` passed（256²，非 768×1280）。
  - **ControlNet union create E2E（2026-11-19，Metal）**：`mlx/control.rs` 完整 forward（33ch、hint、refiner/layers、`before_proj`/`after_proj`）；`remap_control_name` 加载；session 在 `structural_guide.weight>0` 时走 eager DiT + hint；`controlnet_runtime_available` 随 mlx feature 打开。任务 `a1699df0-…`（`z-image-turbo:fp16` + `z-image-turbo-fun-controlnet-union:8steps`，256² / 4 步 / canny / weight 0.75）completed；证据 `tests/parity/evidence/controlnet-union-e2e.json` + `out/controlnet-e2e.png`。矩阵该行 `e2e`/`route` passed；**parity / 热性能仍 pending**。控制图按「已预处理 RGB」resize 后 VAE encode（引擎内未做 OpenCV canny/depth）。
  - **ControlNet 配置驱动 + Union inpaint（2026-11-19）**：registry `catalog.binding` / `guide_type` / `supports_inpaint` / `allows_with_reference`；`compatible_controlnets` create scope 保留同时声明 `create+retouch` 的网；Union `actions.retouch`；前端去掉名字启发式；画布 companion → create+`structural_guide` inpaint，standalone Fill → edits。引擎 `build_control_context_inpaint_nchw` + session 源图/蒙版（白=重绘→preserve 取反）。任务 `538284ce-…`（256² / 4 步 / inpaint）completed；证据 `tests/parity/evidence/controlnet-union-inpaint-e2e.json`。**Fill 引擎仍 unstarted（P5）**；parity / 主机预处理仍 pending。
  - **下一刀（顺序）**：
    1. ControlNet metal_parity / metal_performance；主机端预处理。
    2. latent refine / LEMICA（现 501）；量化 rewrite；`z-image`（非 turbo）行；可选 768×1280 rewrite 热性能加严。
  - **仍缺**：上表 1–2；Flux2 / Qwen-Image；flux1 Fill retouch/extend。
- `flux2`：Klein 4B/9B/base 的 create、edit、LoRA。变体不能用一次 family 级 parity 代替。
- `qwen_image`：create、edit、edit-2511、lightning、多参考、LoRA。Qwen2.5-VL 进 common；108 token crop 和条件拼接留在 family。

### P5

`flux1` 的 Fill、Redux、Depth、Kontext 按 model card 分结构变体。`fibo`、`ernie_image`、`cogview4`、`hidream_o1`、`step1x_edit`、`boogu_image`、`longcat`、`esrgan`。LongCat BSA 要分别证明 Metal 和 CUDA。没有权重的条目保持 unavailable。

### P6

- ACE：`task=cover` 会把原曲 VAE 潜变量按 5 帧一窗做 FSQ，再 detokenize 成 25 Hz 条件。create、cover-nofsq、repaint、lego、extract、complete、5Hz 两阶段、歌词对齐、共享 LoRA 已接上，缺的是和 `main` 的 parity，以及 PMI 的正式门。
- 复用边界：LM 的 FSM、歌词和 DTW 留在 family；WAV 和 ffmpeg 的通用部分进媒体层；采样器和 KV cache 只有 assistant 也要用时才抽出去。
- DiffRhythm：BERT 可共享，G2P 走 ONNX，mulan 按 `main` 转录。

### P7

SeedVR2 图像和视频超分。tiling、因果时间缓存、wavelet/LAB、ffmpeg 分块可复用，但 common 里不写 SeedVR2 分支。conv3d 在两个后端上都要有性能和长任务内存数字。

### P8

- Wan：t2v / i2v / 5B / 14B / MoE / distill / LoRA 分开验。umT5 的 Viterbi 分词优先复用。
- LTX 与 LTX25：两阶段音视频、per-token timestep、mask、IC-LoRA。只共享真正相同的组件，family 之间不互相引用。
- Hunyuan：animate、byT5 字形、1080p SR。Qwen2.5-VL 走 common。SigLIP 等第二个视觉用户出现再抽。
- MiniMax H3：Qwen3-VL conditioner 不在 family 里复制。大 QMM 和长序列内存两个后端都要测。
- LongCat Avatar：whisper 前端抽成中立音频特征，数字人条件留在 family。

### P9

- 下载、components、manifest、workspace registry、user LoRA、画廊、metadata、MCP、CLI。
- 本地 assistant：qwen3 / qwen3_vl、chat template、视觉预处理、采样、KV cache、SSE、卸载。它用自己的 worker，不和分钟级 DiT 串在同一个 executor 上。
- **训练（进行中，`3f2be72` 起）**：不恢复 Python trainer。
  - **已落地**：`danmo-mlx` `value_and_grad` + AdamW；DiffSynth 对齐的 Flow-Match schedule/`training_weight`；hybrid 数据集 `train.jsonl`+`meta.json`（upload/list/health/import）；Z-Image/Turbo 命名 presets（`turbo-sft` / `turbo-diff` / `base-face`）；共享 train loop（**假 latent** loss↓ + 导出 adapter，`attach_lora` 可加载）；API `datasets` / `requirements` / `trainings` / `artifacts` / `register`（MLX feature 下 `ready:true`）；训练 UI 瘦身为四步向导 + 完成态摘要/注册；注册校验 companion 已安装；推理不按配方名分支。
  - **仍缺（训练验收门）**：VAE encode cache、TE encode、完整 DiT forward 上可微 LoRA（替换 probe latent）；人脸检测/门禁真链路；loss/梯度按可训练 tensor 的证据；checkpoint 往返；**真图短训 E2E**（训出的 adapter 经中立推理加载出图）；矩阵勾选。在此之前不要把训练行标 done。

### P10

- **已做**：删除产品 Python（空 `backend/`、parity 脚本、桌面 thin uvicorn / venv bootstrap）。
- **已过时表述**：旧「训练提交一律 501」仅适用于无 MLX feature 的构建；`mlx-metal` / `mlx-cuda` 构建下训练 API 可提交（当前仍是 probe，非产品完备）。
- 仍待：23 个 runtime family 达到发布要求后默认走满 Rust MLX；退役 `danmo-ggml-sys` / `native/third_party/ggml` / GGUF 安装逻辑（按矩阵，勿提前）。
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
