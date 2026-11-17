# Rust MLX 迁移共享资产与算法归属盘点

> 配套计划：`docs/rust_mlx_migration_plan.md`。  
> 目的：在迁移 family 前确定复用边界，避免 main 迁移完成后再做一次架构重整。  
> 原则：复用接口、组织和已验证的运行时无关算法；MLX 数值以固定 main 基线为准，不把 ggml 近似直接当作 MLX 实现。

## 1. 资产分类

- **A — 直接复用**：运行时无关，现有测试可继续作为参考。
- **B — 保留接口/组织，替换实现**：架构正确，ggml 计算实现需要换成 MLX。
- **C — 转录算法/fixture**：算法或测试有参考价值，但必须与 main Python MLX 对照。
- **D — 不迁移**：ggml/GGUF 专用或会造成长期架构负担。

每个 common 组件在进入共享层前必须记录：使用方、参数化维度、独立 fixture、后端要求和不应知道的 family 信息。

## 2. server / contracts / ports

| 现有资产 | 分类 | 迁移处理 | 前置架构优化 |
|---|---|---|---|
| `danmo-server/app.rs` generation/health/queue/SSE | B | 保留 Axum 路由和任务流，替换 engine 配置与能力 gate | DTO 转换集中；静态 EngineRouter；实时 SSE；统一错误与日志 |
| `danmo-server/product.rs` | B | 复用广泛 REST 覆盖，逐项与 main 语义对照 | 清除空实现造成的假 parity；按能力矩阵开放功能 |
| `danmo-server/download.rs` | B | 保留下载任务框架，替换 GGUF key/layout | components/manifest 与 MLX quant key；安装不可变发布 |
| `danmo-store` | B | 复用 SQLite 访问层 | 真实 main DB fixture；时间/状态/索引/分页/lineage 一致 |
| `danmo-schedule` | B | 保留单 worker 默认执行模型 | EWMA/ETA/orphan/preview/priority；实时事件；唯一队列写者 |
| `danmo-contracts` | B | 以 main contracts 为基线补齐 | 避免 HTTP DTO、内部 contract、family options 三处漂移 |
| `danmo-catalog` | B | 复用加载和 projection | registry/driver/routing 同源；MLX version key |
| `ResolvePort` | A/B | 保留端口，扩展输入媒体和组件 revision | server 负责工作区解析；engine 不访问正式 DB |
| `ArtifactSink` | A/B | 保留统一持久化 | bytes 与受控 file-backed payload；原子、幂等、路径边界 |
| `RunObserver` / `ExecutionContext` | A | 保留 | 统一 phase/progress/log/cancel/timing/capacity |

### 不再延后的 server 改造

1. HTTP 鉴权中间件与 main 对齐，不能把 access-key 设置 DTO 当作鉴权完成。
2. 任务、资产、canvas 只能由活跃控制面写；worker 只发事件和 manifest。
3. 大媒体不走 JSON base64。
4. registry projection、迁移矩阵和 submit gate 同步。
5. family 类型不得泄漏到 server adapter。

## 3. pipeline / session / family driver

| 现有资产 | 分类 | 处理 |
|---|---|---|
| `pipeline/dit.rs` | A/B | 保留 resolve/load/encode/schedule/infer/decode/persist 形状；将 ggml session trait 换成 MLX 中立 trait |
| `pipeline/facade.rs` | A/B | 保留 prepare/infer/persist；用于 audio/video 和非标准流程 |
| Job shape | A/B | 保留独立 job 流程，不强套 DiT |
| `pipeline/denoise.rs` | C | 只复用真正通用的 Euler/CFG 组织；布局或公式不同则归 family。MLX Euler 目前在 `families/z_image/mlx/denoise.rs`（家族私有），出现第二个使用方时按 §7 模板迁入 common |
| `pipeline/legacy.rs` | D/C | ggml 生产路径不保留；数值测试有选择的转录，不做永久双实现 |
| `families/driver.rs` | A/B | 保留 family trait/session 抽象，去除 ggml 类型；P2 改为后端中立 trait，MLX 路径接入 driver（目前 MLX 实现仅在测试中使用） |
| `families/mod.rs` | A | 保留唯一 family/action match 和 FamilySpec 组织，增加细粒度能力关联 |

### 形状使用规则

- 标准 DiT 只有 phase 真正一致时才使用 DitPipeline。
- 不同 CFG 公式、token layout、内部多阶段和 companion model 不被大量 callback 塞入 DitPipeline。
- Facade 内部仍需明确子阶段，不能成为无结构黑洞。
- 新 family 不得要求修改其他 family，也不得要求 common 识别其 id。

## 4. common 现有组件

| 组件 | 分类 | 去向与共享边界 | 必要验证 |
|---|---|---|---|
| `common/bpe.rs` | A | 运行时无关 BPE；供 Gemma 等 tokenizer 复用 | Unicode、special/added token、byte fallback、截断和 padding fixture |
| `common/qwen3/tokenizer.rs` | A/C | 保留实现并对照 main tokenizer 行为 | token IDs、chat template、thinking、special tokens |
| `common/qwen3/graph.rs` | B/C | 保留 tower 组织和层索引参数化，计算改 MLX | AfterLayer/最终 norm、attention、KV/输出宽度分阶段 fixture |
| `common/qwen3/session.rs` | B | 保留 encode 接口和输出策略 | 缓存 revision、executor 所有权、dtype |
| `common/qwen25vl.rs` | B/C | 保留共享 trunk/AfterLayer 思路；服务 qwen、hunyuan 等 | vision merge、position/RoPE、层输出、mask、多参考 |
| `common/ops.rs` | 拆分 | 纯 host scheduler/数学转 common；ggml operator 改 `danmo-mlx`/runtime | 每个 scheduler/embedding 独立 host fixture |
| `common/vae.rs` | B/C | 保留 AutoencoderKL + VaeSpec；conv/norm/readout 改 MLX | spec 推断、scaling/shift、tiling、输出范围 |
| `common/png.rs` | A/C | 保留 PNG 编解码（非 ggml 部分） | 压缩 PNG/JPEG/WebP 输入、尺寸和 RGB 顺序 |
| `common/image.rs` | A/C | 通用 resize/crop/color 可复用；与 main PIL 语义对照 | LANCZOS/BICUBIC/NEAREST、边界、偶数对齐 |
| `common/ffmpeg.rs` | A/B | 保留进程媒体和音轨组织，统一安全参数与错误 | 帧数、fps、尾块、音轨、原子输出、取消 |
| `common/conditioning.rs` | B | 保留 revision/LRU/stats 思路 | 不再全局传 MLX Array；归 executor；增加字节预算和完整 key |

### common 目标模块

P3 按真实使用方建立，而不是预先创建空目录：

- `common::ops`：schedulers、activations、norm、embedding、CFG 和家族中立算法。
- `common::attention`：布局转换、mask、runtime attention 请求，不包含具体后端 FFI。
- `common::tokenizers`：BPE/unigram/CLIP/T5/BERT 等真正共享代码。
- `common::qwen3` / `qwen25vl` / `qwen3vl`：参数化 trunk。
- `common::vae`：VaeSpec、AutoencoderKL、tiling/stream cache 的中立部分。
- `common::media`：图片、音频、ffmpeg、色彩与容器。
- `common::cache`：只放 family-neutral key/revision 算法；MLX 值的所有权仍归 runtime executor。

### 稠密权重 dtype（共享层，纠正 main 的错误设计）

**`main` 把稠密 BF16/FP16 权重加载成 F32 是错误设计，不迁移。** 原因是共享加载器 `common/model/base.py::_assign_param_tensor` 的 `param[:] = tensor`：参数初始化为 F32，切片赋值不改变目标 dtype，发布的 BF16 权重被静默放大。Z-Image DiT 因此从约 12 GB 变成约 24 GB，Qwen3 文本编码器与 VAE 同样放大；带宽受限的 DiT 推理也因此变慢，且精度无收益。`main` 内部也不一致（H3、LTX VAE 按参数 dtype 处理）。

Rust 落点：`weights/mlx.rs` 提供共享 `WeightDtype::Native | F32`。

- `Native` 为生产默认，保留 checkpoint 原生 dtype；精度敏感的小张量在算子内按需转 F32。
- `F32` 只在 parity 测试中复现 `main` 现状。P0 中 `common::vae::mlx::load_dense` 与 `families/z_image/mlx` DiT 的 `astype(F32)` 必须迁入该策略，不得作为生产默认。
- family 内禁止写 dtype 分支；显存门以原生 dtype 为准。
- `main` 侧在 P0H 修复后，BF16 输出成为生产层 parity 基线。详见计划 §1.5.1。

### 量化权重加载（共享层，非家族分支）

main 的 `common/model/base.py::load_weights` **不区分家族/版本处理量化**：检测 `*.scales` 键即自动识别 MLX affine 打包（uint32 codes + scales + biases，4/8-bit 按形状与 shard metadata `quantization_level` 推断）；`inference_mode=quantized` 走 `QuantizedLinear` 不解量化（低显存），否则默认解量化回稠密 Linear（legacy/debug）。家族代码零量化分支。Rust 侧对应落点是 `weights/mlx.rs` 共享加载层 + `danmo-mlx` QMM（已有 >32K 行探针），**禁止**在 family 内写 `quant == ...` 分支。执行顺序按用户指示：**fp16 完整验证（真实分辨率/热性能/会话生命周期）收口后再做量化切片**；量化 parity 以 main 同一 checkpoint 的 QuantizedLinear 路径为基线。

## 5. family 资产与算法归属

### 5.1 z_image

| 资产 | 处理 |
|---|---|
| family/session/edit_session/geom 组织 | 保留职责划分，以 main `transformer_mlx.py`、control、latent refine、LoRA 为数值基线 |
| transformer/graph/weights | MLX 重写；remap/sanitize 对齐 main safetensors |
| Qwen3 text | 复用 common trunk，不在 z_image 复制 |
| AutoencoderKL | 复用 common VaeSpec |
| controlnet union | 家族内实现 control refiner/layer hints；通用 attention/VAE 进入 common |
| edit/latent refine/LEMICA/TeaCache | scheduler/cache 原语中立化，策略和系数留 family config |

验收不能只做 turbo create：base/turbo、所有发布 quant、rewrite、LoRA、control、inpaint、latent refine 分开记录。

- turbo 默认路径是 9 步、768×1280、`linear` scheduler，步数大于 8 时 `main` 在 Apple Silicon 自动开 compile；8 步只对应加载 Base LoRA 时自动附加 `z-image-turbo-distillpatch-lora`（Scheme 4）的场景。两条路径分别验收。
- DistillPatch、starface、awportrait 等 LoRA card 无 action，目前不在能力矩阵中，需以组件条目登记。

### 5.2 flux2

- 保留 create/edit session 和 weights 组织。
- double/single block、RoPE、guidance 和 denoise 以 main MLX 为准。
- Qwen3、VAE、scheduler 使用 common。
- Klein 4B/9B/base 和 distilled/base 的行为不能由 family 级一次 parity 代替。
- LoRA 绑定与量化 Linear 是 runtime/weights 共享能力，不在 family 复制。

### 5.3 qwen_image

- Qwen2.5-VL trunk 优先进入 common；family 保留 prompt、108/特定 crop、条件拼接等业务。
- 3D causal VAE 只有在拓扑相同且有第二个使用方时才抽 common；否则留 family 并使用 common tiling/cache 原语。
- create/edit-2511/lightning/多参考分别建 fixture。
- vision preprocess 和 position ids 必须独立验证。

### 5.4 ace_step

可重点复用的运行时无关算法：

- LM FSM、format、codes、token decode。
- lyrics/DTW/vocal prompt。
- APG、Haar DCW、FSQ/quantization 的纯算法。
- cover/repaint/lego/extract/complete 的 mask、region、blend 和 track instruction。
- quality/PMI 的组织。

必须 MLX 重写或对照：

- DiT、VAE、audio codec、LM tower 和采样循环。
- ggml 权重名和 graph 不迁移。
- main 当前未开放的编辑模式先作为独立 main 特性实现并验证，再进入 Rust parity。

共享边界：

- LM sampler/KV cache 若服务 assistant，抽 family-neutral 组件；ACE constrained FSM 留 family。
- WAV/ffmpeg/denoise 的通用部分进 media；歌词、track、FSQ 语义留 ACE。

### 5.5 seedvr2

- tiling、causal temporal cache、wavelet/LAB 和 ffmpeg chunk 组织具有复用价值。
- VAE/DiT MLX 化，conv3d 和 attention 走 runtime。
- color correction 和 PIL bicubic 语义用独立 host fixture。
- image/video upscale 共用 preprocess 与 chunk 原语，但不在 common 写 SeedVR2 family 分支。

### 5.6 wan

- `umt5.rs` tokenizer/Viterbi 是重点直接复用资产，同时回植 main 去依赖。
- scheduler/UniPC、causal VAE cache、patchify/RoPE 算法可转录 fixture。
- DiT/VAE/text encoder 计算改 MLX。
- t2v/i2v/5B/14B/MoE/distill/LoRA 按 model/action 分别验收。

### 5.7 ltx / ltx25

- Gemma tokenizer/BPE 直接复用。
- connector、dual A/V tower、audio VAE、BigVGAN/BWE 的组织可保留，计算改 MLX。
- LTX schedule、stage sigmas、upscaler、pixel shuffle/unpatchify 建独立算法 fixture。
- per-token timestep、mask、IC-LoRA 不得因 t2v 基础版通过而省略。
- LTX25 只共享真实相同的组件，不用 family 继承或 cross-family import。

### 5.8 hunyuan

- Qwen2.5-VL trunk 复用 common。
- SigLIP preprocess/tower可在出现第二个真实使用方后抽 common vision。
- byT5 tokenizer/glyph、DCAE/causal VAE、joint DiT 以 main 为基线。
- animate、SR、t2v 按 registry 实际条目开放；未发布权重明确 unavailable。

### 5.9 minimax_h3

- Qwen3-VL conditioner 是共享 trunk，不在 family 内复制 assistant 实现。
- packed QMM >32K rows 是 P0 runtime 门。
- turbo LoRA、Ref2VA、media roles、scheduler 和帧规则独立 fixture。
- Metal/CUDA 都需验证大序列内存与性能。

### 5.10 其他图像 family

- flux1/fibo/ernie/cogview4/hidream/step1x/boogu/longcat/esrgan 先复用 common scheduler/text/VAE/attention，再实现 family 独有拓扑。
- Flux1 的 Fill/Redux/Depth/Kontext 等按 model card 识别结构变体。
- LongCat BSA 是 runtime 稀疏 attention 能力，不污染 common pipeline。
- ESRGAN 若没有可支持权重或后端实现，保持明确 unavailable，不伪造迁移成功。

### 5.11 diffrhythm / longcat_avatar

- DiffRhythm 的 BERT tokenizer 可共享；G2P 使用 ONNX Runtime C API，去除 bundle Python 动态执行。
- mulan 和音频前端按 main 转录。
- LongCat Avatar whisper 前端应抽通用 audio feature 组件，数字人条件与生成流程留 family。

## 6. runtime / weights 去留

| 现有资产 | 处理 |
|---|---|
| ggml backend selection/context/graph | 不迁移；替换为 MLX executor/device/stream |
| ggml graph builder/replay | 不迁移；MLX lazy graph/compile 语义重新验证 |
| GGUF parser/bundle/quant folding | 不迁移 |
| ggml attention probe | 保留独立 host-reference 思路，改 MLX Metal/CUDA probe |
| capacity | 保留接口思路，改 MLX active/cache/peak/limit/device bytes |
| mmap weight cache | 不直接复用；MLX Array/model cache 归 executor（executor 目前无缓存位置，P2 前定状态模型） |
| cache revision | 复用并扩展到所有 shard/config/tokenizer/manifest/LoRA，以及 `WeightDtype` 策略 |
| safetensors 辅助代码 | 可复用严格解析思路，但与 `danmo-mlx` IO 统一，避免两套加载器；`weights/mlx.rs::load_directory` 的 `entry.unwrap()` 改为返回错误 |
| 稠密权重 F32 放大（main 行为） | **不迁移**：生产保持原生 dtype，F32 仅限 parity 模式（见 §4「稠密权重 dtype」） |
| `danmo-ggml-sys`、`third_party/ggml`、各 family ggml 实现 | 不迁移；family 的 MLX 版点亮时同一 commit 删除其 ggml 实现，P10 清点并删除 ggml 栈 |

## 7. 共享算法分析模板

每次准备把代码移入 common 前，在对应 PR/提交说明中回答：

```text
算法名称：
行为基线（main 文件/函数/commit）：
现有 ggml 参考：
真实使用方：
完全相同的部分：
参数化差异：
必须留 family 的差异：
输入/输出 shape 与 dtype：
Metal 算子路径：
CUDA 算子路径：
独立 fixture：
性能/内存风险：
取消和缓存语义：
```

缺少真实第二个使用方时，可以先定义中立接口，但不要为假想的未来 family 建复杂抽象。

## 8. 架构完成定义

架构优化不是另留一个收尾阶段，而是每个 family 的完成条件：

- 没有 cross-family import。
- 没有 common/runtime family 字面量分派。
- 没有 server family 输出类型。
- 没有第二套权重加载、attention、VAE 或 tokenizer 私有复制。
- 差异化数学有明确 family 归属，共享数学有参数和 fixture。
- registry、能力矩阵和唯一 driver 同步。
- 新增后端只改 runtime/build，不改 family 数学。
- 新增同拓扑 model 通常只需 registry/config 和 fixture；新增 family 不需要改其他 family。
- 生产路径权重保持原生 dtype，峰值显存已记录进矩阵；没有 F32 放大。
- 该 family 的 ggml 实现已删除。
