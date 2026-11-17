# Rust + MLX-C 全量迁移执行方案

> 状态：执行中  
> 执行分支：`feat/rust-mlx-c-migration`  
> 本文是唯一阶段计划与验收索引。执行过程中同步更新状态、证据、偏差和能力矩阵，不凭记忆缩减范围。

## 0. 基线与目标

### 0.1 固定基线

- ggml Rust 基线 commit：`67495d3353d64f1ffd605e3586d0eea74837aebe`。
- `main` 行为基线 commit：`ab4456cb1a712b9e8a56a0629862170c967bef52`。
- 创建迁移分支时工作树已有未提交的 ggml Rust 修改（Hunyuan/LTX/Wan/SeedVR2、server 等），已与 MLX P0 切片一并提交在 `dbcfaff`。该 commit 内 ggml 在途工作与 MLX 切片混合：MLX 新增成果限于 `danmo-mlx*`、`common/qwen3/mlx.rs`、`common/vae/mlx.rs`、`weights/mlx.rs`、`families/z_image/mlx/`、`tests/parity/` 与本组文档；其余不得记为本迁移成果。
- `main` registry 基线：92 个 model card，23 个 runtime family，其中 21 个生成 family，另有 `qwen3`、`qwen3_vl` 两个 assistant family。92 个 card 中 **54 个有 action**（展开为矩阵 227 个 model/version/action 条目），**38 个无 action 的组件 card**（各家族 LoRA、`z-image-turbo-distillpatch-lora`、`flux-canny/depth-controlnet`、`flux-redux`、`depth-pro`、`minimax-h3-veda`、`minimax-h3-turbo-lora`、Wan/Qwen lightning LoRA 等）**尚未进入能力矩阵**，见 §4。
- **已知偏差（2026-11-18 已修复）**：分支 registry 曾是 ggml 时期的 GGUF 布局（93 个 model，多出 `ace-step-5hz-lm`）。现已从固定 main 基线 `ab4456c` 复制恢复 `default_config/models_registry.json`（92 个 model、23 个 family、MLX safetensors 版本键 `fp16`/`int4`/`int8`/`mlx-q4`/`mlx-q8`）；`ace-step-5hz-lm` 按 main 基线移除（如后续需要须单独决策）。架构守卫 `registry_and_family_factory_stay_in_sync` 通过，全量 engine 测试仅剩既有 ggml Hunyuan smoke 失败（与 registry 无关）。repo 来源语义与 main 一致：`source` 未设置默认 ModelScope。**有意偏差（2026-11-18，用户决策，两次迭代）**：main 的 `mlx-q4`/`mlx-q8` 指向 mflux 社区量化（`filipstrand/Z-Image-Turbo-mflux-4bit`、`carsenk/z-image-turbo-mflux-8bit`）。实测 mflux bundle 是魔改布局：VAE conv 键包一层（`decoder.conv_in.conv.weight`）、norm 键 `conv_norm_out.norm.*`、conv kernel 已 NHWC，main 的 `remap_vae_weights` 不处理，加载 fail loud；魔塔镜像（`OsaurusAI/...`）同样布局。最终改为 hf-mirror 可用的**标准 diffusers 布局 MLX affine 量化**仓库：`mlx-q4 → zimageapp/z-image-turbo-q4`（4-bit，group_size 32）、`mlx-q8 → mzbac/Z-Image-Turbo-8bit`（8-bit，group_size 32），`source: huggingface`（main 下载器 HF→hf-mirror 兜底）。键名与官方 fp16 一致，Rust 现有 remap 直接适用；group_size 32 由加载器按 shape 推断（不硬编码 64）。mlx-community 官方仅有 bf16（`mlx-community/Z-Image-Turbo-bf16`），无量化版。
- `main` 是迁移行为基线，不是不可质疑的数学真相。与上游参考实现冲突时记录差异、增加回归，再决定兼容或修复。
- 后续更新行为基线必须显式记录新 commit、registry diff、contract diff 和受影响能力，禁止无声跟随移动分支。

### 0.2 最终目标

形成：

```text
Vue / Desktop / CLI / MCP
    → Rust danmo-server
    → Rust scheduler/store/contracts/catalog
    → Rust danmo-engine
    → Rust common/families/runtime
    → danmo-mlx 安全封装
    → danmo-mlx-sys
    → 官方 mlx-c
    → MLX C++
    → Metal / CUDA / CPU
```

最终包含：

- Rust API、鉴权、SSE、任务调度、资产与 canvas 持久化。
- Rust MLX 服务侧推理。
- macOS Apple Silicon Metal 与 Linux x86_64 CUDA 双后端。
- Rust 本地 assistant 和 OpenAI-compatible 云代理。
- Rust LoRA 训练、checkpoint/resume、候选评估。
- Python serving、Python inference、Python training、`backend_llm` 最终退役。

CPU 是显式真实设备，不是失败后的静默降级路径。Windows 保持暂不支持。

### 0.3 全程不变量

1. Fail loud：缺失算子、权重、后端、action 或实现时明确失败。
2. 入队前静态决定 `model × version × action` 的执行引擎。
3. Rust 失败后不得自动重试到 Python，Metal/CUDA 失败后不得自动换设备。
4. 家族隔离；`families/mod.rs` 是唯一 family/action 工厂分派点。
5. 共享组件不知道 family id，不包含家族业务分支。
6. 每个迁移条目通过 parity、性能、E2E 后才点亮。
7. 活跃控制面是 SQLite 唯一正式写者；推理/训练 worker 不直接写正式任务或资产表。
8. 不把构建成功当作后端支持，不把单个 family 成功当作整个 family 的所有 model/action 成功。
9. 每阶段结束更新本文、能力矩阵、架构文档和测试证据。
10. 未经明确复核，不因为工期或局部困难删除 model、action、平台、训练或安全能力。

## 1. 已确认技术决策

### 1.1 直接绑定官方 mlx-c

正式依赖方向：

```text
families/common/runtime → danmo-mlx → danmo-mlx-sys → mlx-c → MLX C++
```

- `danmo-mlx-sys`：bindgen、CMake 原生构建、链接和原始类型，唯一 unsafe FFI 入口。
- `danmo-mlx`：安全 RAII、typed array、stream/device、dtype、error、compile、quant、IO 和 transform 封装。
- 只有 `danmo-mlx` 可以依赖 `danmo-mlx-sys`。
- families/common 禁止直接使用 C handle、原始返回码和 unsafe FFI。
- `mlx-c` 未覆盖的通用 C++ 能力只能进入唯一 `danmo-mlx-native` shim；shim 不得包含 family 数学。
- `mlx-rs` 可用于阅读和参考，不作为正式运行时依赖。

起始版本基线为 `mlx-c 0.8.0`（commit `a26f2d584a14ec619ae8db9a09be03becd47598a`）与 MLX `0.32.4`（commit `36969c094354086e15d2a327a9c50feb20b9cf11`）。两者均已作为 submodule 固定。MLX 构建依赖 fmt `12.1.0` 与 nlohmann-json `3.11.3` 也已本地化，来源和 SHA-256 分别记录在 `native/third_party/fmt/DANMO_SOURCE.json`、`native/third_party/json/DANMO_SOURCE.json`；构建通过 `FETCHCONTENT_SOURCE_DIR_*` 禁止临时拉取这些源码。P0 继续记录 Rust MSRV、构建参数和依赖校验。Python parity 环境使用相同 MLX core。

### 1.2 Metal/CUDA 是一等后端

| 平台 | 架构 | 构建后端 | 发布形态 |
|---|---|---|---|
| macOS | arm64 | Metal + CPU | desktop/server |
| Linux | x86_64 | CUDA + CPU | desktop/server |

- Metal 和 CUDA 是不同构建产物，不做同进程热切换。
- Linux CUDA 使用 MLX 原生 `MLX_BUILD_CUDA=ON`，并固定 CUDA Toolkit、cuDNN、driver 和支持矩阵。
- `DANQING_MLX_DEVICE=auto|gpu|cpu` 只在进程启动时选择一次。
- `gpu` 不可用即失败；`cpu` 必须显式选择；`auto` 记录选择原因。
- 家族数学不写 `if cuda` / `if metal`。后端分派仅存在 runtime 的 probe、kernel、内存和能力层。
- 自定义算子优先使用通用 MLX 算子；否则分别实现 Metal/CUDA kernel；没有实现的后端显式 unavailable。
- `mlx-mfa` 是 Metal 参考资产，不作为跨后端运行时依赖。

### 1.3 长期 MLX executor

不得每任务新建线程后共享跨任务 MLX Array。每个进程/设备一个长期 executor：

- executor 创建并拥有 Array、模型、stream、compiled function、权重和条件缓存。
- `Array` 和 compiled closure 保持 `!Send`，默认不实现 `Sync`。
- 跨线程只传普通请求数据、路径、字节、事件和取消标记。
- HTTP/Tokio 线程不直接执行阻塞 MLX 推理。
- 缓存 get/put/evict/unload 和全局 memory limit 修改均在 executor 内执行。
- 任务运行中不得由其他线程修改进程级 MLX memory limit。
- 在 step、token、tile、chunk 和 eval 边界检查取消；无法中断的长算子必须记录。
- panic、OOM、心跳超时和 worker 崩溃使当前任务失败，不触发跨引擎重试。

现状（2026-11-17 复核）：`danmo-mlx/src/executor.rs` 目前只是「专用线程 + channel + panic 捕获」，闭包要求 `Send + 'static`，没有模型/条件缓存的存放位置、取消 token、进度回调和心跳。P2 前必须先确定 executor 状态模型：executor 线程内类型化 `ModelCache`（revision key、字节预算、unload）、step 边界取消 token、进度/日志事件通道；禁止各 family 自建 `thread_local` 缓存。

assistant 不与生成 DiT 共用同一串行 executor，否则聊天会排在分钟级视频任务后面。assistant 使用独立 worker（与 `main` 的 `backend_llm` sidecar 相同的隔离方式），显存预算由控制面统一分配并显式记录。

### 1.4 Rust server 核心先行

当前分支已有大量 server 资产，不采用“Python server 保持到最后”的临时路线。

执行顺序：

```text
P0 技术决策门
→ P1 Rust server 核心控制面 + Python 纯推理 worker
→ P2/P3 MLX runtime/common
→ P4–P8 家族逐条切入进程内 Rust engine
→ P9 外围、本地 assistant、Rust 训练
→ P10 Python 过渡栈退役
```

P1 是核心控制面先行，不要求所有外围占位路由一次完成后才能开始 P2。外围按 family 依赖和 P9 清单持续补齐。

**推理与控制面进程隔离（复核建议，待确认）**：Metal/CUDA 命令缓冲失败、驱动 abort 和部分 OOM 不会变成 C API 错误，而是直接终止进程；长时间不返回的 kernel 也只能靠杀进程可靠取消。若 MLX 推理与 SQLite 唯一写者、SSE、队列、鉴权同进程，一次推理崩溃会拖垮整个控制面。建议 Rust 推理同样以独立 worker 子进程运行，复用 §5.2 的协议（版本化请求、心跳、文件 manifest、不写正式 DB）；控制面只负责调度与持久化。这样 §5 的双引擎桥成为长期架构而非过渡代码，并同时覆盖强制取消、Metal/CUDA 构建并存和训练/推理隔离。采纳前，§0.2 中「进程内 Rust engine」的表述保持不变但视为待决策。

### 1.5 权重格式不变

- 保留 MLX safetensors、fp16/bf16/fp32、int8/int4、mlx-q4/mlx-q8 及 scales/biases 布局。
- 保留 main 的 `remap_*`、sanitize、LoRA 和 bundle 语义，在 Rust 权重层实现。
- 不引入 GGUF、repack 或 ggml 权重布局。
- ggml 分支只借鉴架构、共享组件、tokenizer、测试语料和治理，不继承其计算后端和权重格式。

### 1.5.1 稠密权重保持原生 dtype —— 纠正 main 的 F32 放大（错误设计）

**结论：`main` 把本为 BF16/FP16 的稠密权重加载成 F32 是错误设计，不是要迁移的行为。** Rust 生产路径不得照搬；但 `Native` 不等于强制 BF16，官方 checkpoint 为 F32 的 tensor 必须继续保持 F32。

来源：`main` 共享加载器 `backend/engine/common/model/base.py::_assign_param_tensor` 最终执行 `param[:] = tensor`。`nn.Linear` 等参数初始化为 F32，原地切片赋值只拷贝数值、不改变目标 dtype，于是发布的 BF16 权重被**静默放大为 F32**。加载器、`ZImageConfig` 均无「保持 F32」的意图或开关；`main` 内部也不一致——MiniMax H3、LTX VAE 按参数自身 dtype 处理，只有走默认加载器的 family 被放大。Z-Image 激活本身是 BF16（`latent_noise_dtype="bfloat16"`），与 F32 权重相乘被类型提升为 F32 计算。

**2026-11-17 实测纠正**：本机按 registry `Tongyi-MAI/Z-Image-Turbo` 下载的官方 DiT 三个 shard 共 521 个 tensor，**全部为 F32**，总字节 `24,619,634,944`，与官方 index 的 `total_size` 完全一致；不是下载错误或 BF16 被转换。VAE 244 个 tensor 为 BF16（`167,639,366` 字节），Qwen3 文本编码器 398 个 tensor 为 BF16（`8,044,936,192` 字节）。因此 Z-Image 的 F32 DiT 不是 main 放大造成的，不能声称 Native DiT 会从约 24GB 降至约 12GB；registry 的 `fp16` 名称不能作为 tensor dtype 证据。

一般代价（仅针对 checkpoint 本为 BF16/FP16 的 tensor）：

- **显存翻倍**：低精度权重被放大后常驻字节翻倍；Z-Image Qwen3 文本编码器与 VAE 属于此类。DiT 是否节省空间必须以各官方 shard 实测 dtype 为准。
- **速度更慢**：被放大的权重增加内存带宽，且不能利用 BF16 matmul 路径（含 M5 NAX）。
- **精度无收益**：BF16 权重放大不会恢复精度；MLX BF16 matmul 内部已用 F32 累加。

决策：

1. 权重 dtype 是 `weights/mlx.rs` 共享加载层的显式策略（`WeightDtype::Native | F32`），**不是** family 分支。
2. `Native` 为生产默认：逐 tensor 按 checkpoint 原样保留 BF16/FP16/F32；norm、RoPE、timestep 等精度敏感小张量在算子内部按需转 F32。
3. `F32` **只用于** parity 测试复现 `main` 现状数值。共享策略已在 `9859706` 落地：`weights/mlx.rs::WeightDtype::{Native,F32}` 统一供 Qwen3、VAE 和 Z-Image DiT 使用，`F32` 仅在 `cfg(test)` 可用，生产加载只能选择 `Native`。
4. parity 分两层：机制层 F32 模式对 `main` 逐 ULP；生产层 `Native` 按 checkpoint 实际 dtype 单独设门，对照显式恢复原生 dtype 的参考输出，并加 `make bench-eval` PickScore 质量门。
5. 性能门与显存门**以 checkpoint 原生 dtype 路径为准**；F32 对 F32 的热性能对比不构成生产性能结论。
6. `main` 同步修复（P0H）：参数按 checkpoint dtype 赋值（或整体替换数组而非切片拷贝），只恢复原本低精度 tensor 的显存收益，不把官方 F32 tensor 强制降级。
7. 这是对 `main` 行为基线的有意偏差（§0.1「main 不是不可质疑的数学真相」），矩阵中每个条目必须记录峰值显存。

### 1.6 训练最终迁移 Rust

`mlx-c` 的 `value_and_grad`、VJP/JVP、checkpoint、custom VJP 等能力作为训练基座。

Python trainer 仅是过渡行为基线。最终需要实现：

- trainable/frozen 参数树和 LoRA 注册。
- loss、梯度、gradient accumulation/clipping。
- optimizer、LR schedule 和状态更新。
- gradient checkpointing/custom VJP。
- 显式 RNG 状态。
- safetensors + optimizer/RNG/metadata checkpoint。
- 数据集、裁剪、增强和 latent/text cache。
- 现有 checkpoint 的完全恢复或明确区分的权重恢复。
- candidate evaluation 直接复用 Rust inference。
- Metal/CUDA 训练按后端分别验收；不得因推理支持 CUDA 自动宣称训练支持。

复核建议（待确认）：Rust 训练需自建 optimizer、checkpoint、数据增强、梯度检查点，并按两后端分别验收，投入大、收益不确定。推理全部迁移后，Python trainer 可作为不写正式 DB、只产出 LoRA 文件的独立 worker 长期存在。建议将「退役 Python 训练」从 P10 必选降为可选，推理稳定后按收益评估再定。

## 2. 现有资产的直接复用与前置优化

### 2.1 server 与控制面

直接复用：

- `danmo-server/src/app.rs`：generation submit、health、queue、SSE 和进程内 HTTP 测试模式。
- `product.rs`：tasks/models/assets/canvas/settings/download/loras/assistant 的广泛路由覆盖。
- `download.rs`：下载、安装、LoRA 搜索和进度。
- `engine_host.rs`：`ResolvePort`、`ArtifactSink`、`RunObserver` 适配。
- `danmo-store`：tasks/logs/assets/asset_groups/canvas。
- `danmo-contracts`、`danmo-catalog`、`danmo-schedule`。

在 P1 直接优化，不留到二次架构整理：

- 将 HTTP DTO、领域 contract、EngineRequest 的转换集中为机械 adapter，消除重复字段漂移。
- 统一 `ApiError`/EngineError 分类、locale、HTTP status 和日志上下文。
- 路由、registry projection、engine capability 使用同一份能力数据，避免三处手工同步。
- 真实 main v3 DB fixture 验证字段、时间、状态、priority、分页、索引和 lineage，而非只比较表名。
- scheduler 补 EWMA、ETA、queue band、孤儿恢复、preview、取消和优先级。
- SSE 改为实时事件流，保留事件名、顺序、重连和完成语义。
- 补 main 的 loopback/remote HTTP/MCP 分离鉴权；设置 DTO 不等于鉴权中间件已经完成。
- 大文件结果使用文件 manifest，不把视频/音频 base64 嵌入 JSON。
- 所有产物经 work dir 原子完成和路径边界检查，再由 ArtifactSink 正式持久化。

### 2.2 engine ports 和三形状 pipeline

直接复用并保留：

- `ResolvePort`、`ArtifactSink`、`RunObserver`。
- `ResolvedRun`、`ExecutionContext`、通用 `Artifact`、`EngineResult`。
- `DitPipeline`、`JobPipeline`、`GenerationFacade`。
- 单点 `driver(family, action)`。
- family-owned session 和标准 phase 顺序。

前置优化：

- `EngineRequest` 使用强类型公共字段和严格 family options；server 不知道 ACE/LTX 等家族输出类型。
- Artifact 同时支持进程内 bytes 与受控 file-backed payload，持久化语义保持一致。
- `FamilySpec` 不再用单个 family 级布尔值掩盖 model/version/action 差异；能力投影关联迁移矩阵。
- shared pipeline 只处理真正共同的 phase；不同 denoise、布局、guidance、readout 留在 family session。
- 不为了套形状引入大量 callback/mode switch。差异大时使用 Facade，而不是污染 DitPipeline。
- engine 对工作区、DB、SSE 无业务知识；文件访问限定于 ResolvePort 返回的 bundle、输入媒体和受控 work dir。

### 2.3 common、tokenizer 和测试资产

直接复用/转录：

- `common/qwen3`、`common/qwen25vl` 的参数化 trunk 思路。
- `common/vae` 的 AutoencoderKL + VaeSpec 组织。
- `common/ops` 中运行时无关的 scheduler、embedding、算法参考。
- umT5 Viterbi、Gemma BPE、通用 BPE。
- host-reference fixture、cache revision、conditioning 统计、capacity/probe 思路。
- 架构守卫和 registry/factory 同步测试。

迁移时先做算法归属分析：

- 两个以上 family 数学相同且只差配置，进入 common 参数化组件。
- 只有一个 family 使用或拓扑不同的实现留在 family，避免“为了共享而共享”。
- 权重文件名和 tensor layout 是文件格式，不等于 family 业务。
- scheduler、norm、attention、VAE、tokenizer 共享时通过配置、层索引、输出模式、VaeSpec 等表达差异。
- 每个共享算法必须带独立 host/Python fixture，不允许仅靠最终图片推断正确性。

### 2.4 不借鉴的 ggml 资产

- ggml graph/cgraph、GGUF parser、tensor folding、F16 workaround。
- 多后端 ggml auto 机制。
- ggml mmap cache 本体。
- SSE 轮询和过度简化的 scheduler。
- GGUF bundle/install/packaging 逻辑。
- 与 MLX 数学不一致的近似实现。

## 3. 架构与依赖守卫

P0/P1 建立并持续执行：

1. Cargo 依赖图守卫：只有 `danmo-mlx` 依赖 `danmo-mlx-sys`。
2. 源码守卫：families/common 无 `mlx_sys`、unsafe FFI 和原生 handle。
3. `danmo-mlx-native` 是唯一 C++ shim，且无 family 字符串。
4. `families/mod.rs` 是唯一 family/action 工厂分派。
5. family 不得 import 另一个 family。
6. common/runtime/pipeline/session 不含 family 业务实现类型或 family 分支。
7. server adapter 不含 family session/output 类型。
8. registry、family specs、driver、路由能力矩阵保持同步。
9. 不存在 Python 自动 fallback、设备自动 fallback 或占位成功结果。
9a. 生产加载路径不得把稠密权重放大到 F32（§1.5.1）；F32 仅允许出现在显式 parity 模式。
10. 新 family 的默认修改面仅为 registry/config、`families/<id>/`、工厂行和能力矩阵；公共组件复用不得要求修改其他 family。

优先使用 Cargo metadata 和 Rust AST/结构化检查；现有 substring 测试可保留为补充，但不作为唯一防线。

## 4. 能力矩阵与执行记录

建立并维护 `docs/rust_mlx_migration_matrix.json`，每个条目至少包含：

```text
model_id
version
family
media
action
architecture_variant
quant
adapters_options
python_baseline_status
rust_driver_status
metal_parity
cuda_parity
metal_performance
cuda_performance
metal_peak_memory   # 待补字段：原生 dtype 路径峰值显存及对照
cuda_peak_memory    # 待补字段
e2e
route
evidence
notes
```

状态只能使用矩阵 `status_values` 中的受控值：

```text
work:         unstarted / in_progress / blocked / parity_failed / ready / routed / unavailable
verification: pending / in_progress / passed / failed / blocked / not_applicable
route:        pending / python / rust / unavailable
```

本文 §11 阶段表使用独立的阶段状态：`pending / in_progress / done / existing_failure`，不与矩阵状态混用。

矩阵待修正项（2026-11-17 复核）：

- `python_baseline_status` 全部为 `registry_declared`，不在 `status_values` 中；需补入受控值或改为 verification 值。
- 227 条 `architecture_variant` 全部为 `unclassified`；Flux1 Fill/Redux/Depth/Kontext 等结构变体需按 model card 归类。
- 38 个无 action 的组件 card（LoRA/ControlNet/Redux/depth-pro/VEDA 等）未登记。它们是 structural guide、Z-Image Scheme 4（DistillPatch 自动附加）、H3 turbo、lightning 等生产行为的依赖，需以组件条目或 `adapters_options` 维度登记验收。
- `z-image-turbo/fp16/create` 的 `metal_parity=passed`：16×16、1024×1024、768×1280 的 F32 parity，16×16 和 768×1280 的 checkpoint-Native parity/峰值显存，以及默认 9 步 TeaCache 和强制 skip 的 gate/residual 行为均已通过。该状态仅覆盖 Metal 数值与显存；正式性能、生产路由/E2E、CUDA 和其他版本仍为独立待办。
- 证据中的 `out/parity-zimage-*` 位于被 `.gitignore`（`/out/`）忽略的目录，无法从仓库复现；关键 fixture 需把 SHA、shape、误差摘要写入矩阵，或存放到可提交位置。

执行规则：

- 每阶段开始将本文对应状态改为 `in_progress`。
- 完成时记录 commit、命令、平台、MLX/CUDA 版本、fixture 和结果路径。
- 失败或未验证不得标为 done。
- 本机无 CUDA 硬件时，CUDA runtime 项保持 pending/blocked，不能用 macOS 构建结果代替。
- 偏差必须记录原因、影响、用户决策和后续阶段；不得静默缩小交付。
- 每完成一个条目同步 registry projection、路由和前端能力。

## 5. 双引擎过渡与 IPC

### 5.1 路由

```text
model_id × version × action → rust | python | unavailable
```

- Rust 条目进程内执行。
- Python 条目进入常驻 Python 纯推理 worker。
- 入队前解析并固定引擎，写入任务 metadata。
- 失败不自动切换。
- 测试覆盖显式引擎选择和禁止 fallback。

### 5.2 Python 纯推理 worker

- versioned JSON 请求。
- 常驻进程，避免每任务重复加载 Python 环境。
- 日志/进度/心跳事件。
- cancel、timeout、崩溃和孤儿处理。
- work dir 文件产物和 result manifest。
- 不直接写正式 DB。
- 内部原 Python asset persistence 与正式资产库解耦。
- Rust ArtifactSink 检查产物后统一持久化。
- 路径逃逸、超大消息、非原子产物和协议版本不兼容均失败。

### 5.3 控制面与训练过渡

- `DANQING_SERVER=python|rust` 只用于显式控制面切换/回滚。
- 同一 DB 不得同时运行两个消费队列的控制面。
- Rust 活跃时 Rust 独占正式写入。
- 过渡 Python training worker 同样改为事件和 manifest，不直接写正式 DB。
- 训练 Rust 化后删除该过渡桥。

## 6. 权重与缓存设计

### 6.1 加载与 remap

- safetensors 单文件/分片。
- tensor 名、shape、dtype、重复和缺失严格校验。
- 稠密权重默认逐 tensor 保持 checkpoint 原生 dtype（BF16/FP16/F32）；`main` 的低精度权重 F32 放大是错误设计（§1.5.1），只在显式 parity 模式复现。
- 加载错误一律返回错误，不 panic（`weights/mlx.rs::load_directory` 的目录项 `unwrap()` 已在 `9859706` 修正）。
- affine int4/int8 与实际发布的量化模式。
- LoRA、多 LoRA、系数、量化和非量化绑定。
- 未完成安装的 bundle 不得进入缓存。

### 6.2 revision

缓存键包含：

- 所有 shard 的 canonical path、长度和 mtime nanos。
- manifest install generation 或内容摘要。
- tokenizer/config/preprocessor revision。
- backend/device/dtype/quant，以及权重 dtype 策略（Native / F32 parity）。
- LoRA 集合及各自 revision。

安装后的 bundle 不可变，最后发布 manifest。

### 6.3 缓存所有权

- MLX 权重、模型、compiled function、MLX conditioning 均归 executor。
- host bytes 缓存只有在测量证明收益时保留。
- conditioning key 包含 token IDs、negative、layers、width、thinking、mask/padding、guidance 和所有影响输出的配置。
- 除条目数外设置字节预算。
- health/capacity 暴露 entries/hits/misses/bytes/idle/expires。

## 7. 分阶段执行

### P0 — mlx-c spike 与决策门

状态：`in_progress`

当前执行记录：macOS 源码 Metal 构建最初因本机缺少 Apple Metal Toolchain 组件而失败，P0 runtime probe 曾使用 PyPI 官方 `mlx-metal 0.32.4` macOS 14 wheel（SHA-256 `8019b2db622599267d66cc168b2f0dbf5af929790a6c3b3913e00eabc42a7947`）配合源码 `mlx-c`。2026-11-17 经用户确认安装 `xcodebuild -downloadComponent MetalToolchain`（Build 27B47）后，**源码 Metal 构建门通过**：`cargo clean -p danmo-mlx-sys` 后从 pin 的 MLX/mlx-c/fmt/json 源码完整构建（本机 Apple M5 Max / macOS 27.2，deployment target 27.2，NAX kernels 已编入），`danmo-mlx --features metal` 16 单测 + 2 守卫 + 3 safetensors、engine `mlx::` 31 项（含全部真实 fixture 与端到端）全部通过，且不需要 `DANQING_MLX_PREBUILT`。源码构建热性能与 wheel 一致（compiled 8 步 0.640s）。干净打包验证仍未完成，保留在 S0/S4 门中。

**构建环境 gotcha（已踩坑并记录）**：在 prebuilt 与源码两种链接模式间切换后，若不重新链接测试二进制，`@rpath/libmlx.dylib` 可能解析到旧产物，出现「同一测试时过时不过」的假数值分叉（曾表现为 SDPA 8e-4、Qwen3 caption 误差 4 的假失败）。切换后必须 `cargo clean -p danmo-mlx-sys`（或触发重链接）再跑；确认失败前先做单测隔离 + A/B 两种链接各跑一遍。另外 MLX 0.32.4 默认 `MLX_ENABLE_TF32=1`，NAX 芯片（M5 系）上 F32 matmul/SDPA 可能走 TF32；host 参考微探针在 TF32 开启时可能超 F32 容差，可用 `MLX_ENABLE_TF32=0` 做精确对照（真实 BF16 模型路径不受影响，main 与 Rust 同为 MLX 默认策略）。

Z-Image 资产检查：最初本机安装只有 GGUF DiT；经用户明确授权，已按 main registry 的 `Tongyi-MAI/Z-Image-Turbo` 下载 3 个官方 safetensors DiT shard 到用户工作区 `models/Image/z-image-turbo-fp16/transformer`（`main` 的 `local_path` 布局）。Rust `SafeTensors` 已验证 521 个 tensors、关键 shape、跨 shard 唯一性和 remap 无冲突。

Z-Image Metal 固定注入噪声垂直切片已通过（提交于 `dbcfaff`）：`tests/parity/zimage_real_mlx.py` 从固定 main 基线生成 16×16、8 步真实权重的 caption/latent/RGB fixture，`real_zimage_create_matches_main_latents_and_pixels` 使用同一 token IDs、初始 latent 和 sigmas 串联 Rust Qwen3、完整 DiT、Euler 与 AutoencoderKL。2026-11-17 复跑结果为 caption 最大误差 0、最终 latent 最大误差 `0.015625`（逐元素四个 BF16 ULP 门）、VAE 输出最大误差 `0.0030458868`；768 个 8-bit RGB 通道中 34 个不同，最大通道差 1。使用 main 最终 latent 单独解码时 VAE 最大误差为 `0.0000015497208`，确认残差来自逐步 DiT 数值累积。产物为 `out/parity-zimage-rust.png`，测试位于 `native/crates/danmo-engine/src/families/z_image/mlx/model.rs`。

垂直切片过程中确认并修复了三处与 main 的加载/调度语义差异（不是放宽容差）：

1. **稠密权重 F32 加载（仅 parity 模式）**：main 的 MLX `Module` 加载会把发布的 BF16 VAE/DiT 参数写成 F32（`_param_map` 切片赋值 `param[:] = tensor` 的副作用）。**这是 main 的错误设计，显存翻倍且无精度收益，见 §1.5.1。** Rust 此前保留 BF16，导致真实 VAE 首个分叉出现在 `conv_in`（误差 0.0054 超门）。为对齐 main 数值，parity 测试显式使用共享 `WeightDtype::F32`；该策略已在 `9859706` 迁入 `weights/mlx.rs`，生产默认 `Native` 保持 BF16，且 `F32` 只在测试构建可用。独立真实 VAE 解码对 main 的最大误差降至 `0.0000014305115`，分阶段 trace（conv_in/mid/up blocks）最大 `0.00019836426`。
2. **真实 pad token**：DiT 图像/文本尾部 padding 必须读 checkpoint 的 `x_pad_token` / `cap_pad_token`，不能写零。修复后首轮 velocity 误差从 `1.496` 降到 `0.000932`。
3. **Euler BF16 语义**：main 每步把 velocity 与 `dt` 转回 latent 的 BF16 再更新；`families/z_image/mlx/denoise.rs::euler` 已对齐（输出与 delta 均 cast 到当前 latent dtype），CPU/Metal 单测覆盖 F32 与 BF16 两条路径。该实现目前是 **Z-Image 家族私有**（错误文案和测试名均为 Z-Image），出现第二个使用方时按共享资产清单 §7 模板迁入 common。

组合测试的 VAE 拓扑必须取真实配置（该 VAE `layers_per_block=2`，不是默认 3）；此前硬编码 3 造成的 `0.305` 输出差已排除，与跨组件 MLX 状态无关。

固定 seed 切片同样通过：`out/parity-zimage-seeded.json`（seed 42）由同一 main 基线脚本生成，Rust `RandomKey::new(42).normal` 产生的初始噪声与 main **逐位一致**（测试内 `assert_eq`）。修复下述 `x_embedder` dtype 后，2026-11-17 复跑固定 seed 与固定注入两条 16×16 路径均为最终 latent、768 个 8-bit RGB 通道**逐位一致**；VAE 用 main 最终 latent 单独解码的最大误差分别为 `0.0000010728836` / `0.0000015497208`。`danmo-mlx` 的 seeded normal/split 在 CPU/Metal 均有对 main 的固定值单测。

该结果只点亮 fp16 turbo create 在 16×16、8 步、F32 parity 模式下的 Metal 数值 parity（固定注入噪声与固定 seed 两类均通过），不代表 9 步默认 TeaCache、生产路由或完整 E2E 已开放。checkpoint 原生 dtype 验收见下节；至少一个已发布量化版本、连续任务/切换/取消/缓存、CUDA 和 encode/VAE 热性能门仍待完成。

共享权重与量化路径已在 `9859706` 落地：`WeightDtype::Native` 为生产默认，F32 widening 仅限测试；`LinearWeight` 对 dense/affine 提供统一 `apply`，affine uint32 codes + scales + biases 全程保持 packed 并通过 QMM 在 kernel 内解量化，不在加载层或 family 中还原稠密副本。加载器校验 shard `quantization_level`、registry bits/group size、重复/缺失 tensor 和歧义形状；CPU/Metal tiny DiT packed-vs-dequantized、compiled-vs-eager 以及 >32K QMM 均通过。架构守卫禁止绕过 `LinearWeight`。**这尚未替代已发布量化 checkpoint 的真实端到端 parity/显存验收；本机目前只有官方 fp16 Z-Image bundle。**

#### compile 实测与热性能观察（2026-11-17，16×16 / 8 步 / 官方 fp16 权重 / Metal）

- `danmo-mlx` 新增多输入 `Closure`（`Fn(&[Array], &Stream)`，与 `UnaryClosure` 同一 FFI/panic 边界），CPU/Metal 各有「captured 常量 + 多输入 + shapeless 重放 + 形状变化重编译」与「闭包 panic 变错误不崩进程」单测；`Stream` 实现 `Clone`（共享同一 MLX stream handle）。
- `families/z_image/mlx/model.rs` 拆出 `embed_timestep` + `forward_with_timestep`，新增 `CompiledForward`：weights/geometry/caption 捕获为常量，仅 latent 与 timestep embedding 是 traced 输入。**pinned MLX 0.32.4 上 `Split`（AdaLN 分段）与 `Slice`（RoPE 分量、FinalLayer 裁剪）都没有 `output_shapes`，shapeless compile 对完整 DiT 显式失败**；使用 shape 特化 compile（形状变化自动重编译），这是已验证路径而非静默降级。
- tiny DiT compiled vs eager：CPU/Metal 跨 3 个 sigma 均 < 1e-5。真实权重 `real_compiled_dit_matches_eager_and_reports_warm_timings`：**8 步 velocity compiled vs eager 逐位一致（max error 0）**，eager 对 main 的 parity 与既有门一致。
- main 热基线（`tests/parity/zimage_warm_mlx.py`，同进程 3 passes，权重常驻）：eager 热 denoise `0.671s/0.669s`，encode `0.022s/0.018s`，VAE `0.005s`。修正 `--compile` 让其真正经过 `before_denoise` 后，实际 compiled 热 denoise 为 `0.782s/0.783s`，并未加速；证据 `out/parity-zimage-warm-compile-real.json`（SHA `2c4538c0b3928050f453eaa6480d516f6a1dd8a498456420ed1fa336600ce7cb`）。此前未经过 `before_denoise` 的 compile 计时无效。
- **更正（2026-11-17 复核）**：此前写「main 生产策略是 8 步 turbo 不开 compile，fixture 的 eager 基线即生产路径」，不成立。`main` registry 中 `z-image-turbo` 默认 **9 步**、默认分辨率 **768×1280**、默认 scheduler `linear`；8 步只出现在加载 Base 训练 LoRA 时自动附加 DistillPatch（Scheme 4）的场景。`ZImageConfig.use_mlx_compile_auto=True`，但默认 `teacache_mode="auto"` 在 9 步会启用 step cache，继而关闭 compile；因此 **9 步默认优化路径是 TeaCache，不是单纯 eager，也不是 compile**。`teacache_mode="none"` 且显式 `use_mlx_compile=True` 时才走 compiled forward。8 步 eager、9 步无 TeaCache eager、9 步显式 compile、9 步默认 TeaCache 必须分别验收。
- Rust 热观察（warm-up pass 后）：eager 8 步 `0.706s`，compiled `0.662s`（eager 的 93.7%；对 main 8 步 eager 热基线约 **101%**）。证据：`out/parity-zimage-warm.json`、`out/parity-zimage-warm-compile.json`、`out/parity-zimage-rust-warm.json`（本地产物，未入库）。
- **边界**：以上是 16×16 小图单 run、双方均为 F32 稠密权重的观察；F32 对 F32 不构成性能结论（§1.5.1），encode/VAE 的正式热对比与 checkpoint-Native 真实分辨率性能门仍未完成；不得据此宣称 S4 全量热性能门通过。

#### 真实分辨率 1024×1024：完整递推 parity 已通过（2026-11-17）

`tests/parity/zimage_real_mlx.py` 已参数化 `--width/--height/--steps/--compile`，Rust 端到端测试从 payload 读分辨率。1024×1024 / 8 步 / seed 42（4096 image tokens + 32 cap tokens）：

- main 基线：encode 0.26s、denoise 35.3s、VAE 0.84s（冷）；main 自身 TF32 on/off 输出**逐位一致**。完整 fixture SHA-256：`out/parity-zimage-1024.json` = `ba22df8f6f68659ed6b36753d2c50d8d9672308746158eb190e501878c2b2931`。
- 新增 `tests/parity/zimage_stages_mlx.py --step` 与 Rust `real_dit_stage_fixtures_localize_production_resolution_divergence`，对 timestep、两层 noise/context refiner、30 个 unified block、final layer 和 velocity 逐点比较。step-0 fixture SHA：`expected.json` = `74d391aa7f1a86e47e7b2a931e3168b640afcf7eb78816db131f9ecf653d0883`、`stages.safetensors` = `5020d5c4bb0ea07aa0cfd4dcebb3ca06119173af217ec2c01546c1034b0d784b`；step-1 fixture SHA 分别为 `5f53dac9585615401dcf8a0142ceb82e1866d2f382092ea8c03c9fdf6a507744`、`fba8816d1f472c421cbd9e58fbd485fa10c7a6af69adb5db3ec3fab8b9ef286a`。
- **三处数值边界已修复**：`x_embedder` 输出转回 latent BF16；RoPE 的 `arange/power/cos/sin` 改为 MLX 算子而非 Rust 主机数学；timestep 的 `(1-sigma)*t_scale`、频率 `exp` 和 `cos/sin` 同样改为 MLX 算子。修复后 step 0、step 1 的所有 37 个阶段均逐位一致，完整 8 步每个 velocity、每步 latent 与最终 latent 也全部逐位一致。
- 最终 VAE 输出最大误差 `0.00002604723`，3,145,728 个 8-bit RGB 通道中 1,678 个不同，最大通道差 1，通过原有 `0.004` 输出门和 RGB±1 门；没有放宽任何阈值。产物 `out/parity-zimage-1024-rust.png`。
- 曾排除的其他因素：TF32、源码 vs wheel 构建、sigma 公式、position ids / patchify 顺序、pad token、modulation unpack、Euler BF16 cast。该通过仍是 F32 parity 权重下的机制层结果；checkpoint-Native parity/显存见下节，正式热性能门仍独立待验收。

#### 768×1280 / 9 步：无 TeaCache eager 与显式 compile parity 已通过

- `out/parity-zimage-default-768x1280.json`（SHA `e6ace0a35d162e28a32fc475fdc74cc2e19aa479a08dcefe5ba1e9667c1074a2`）覆盖无 TeaCache eager；9 个 velocity、全部 latent 与最终 latent 逐位一致，VAE 最大误差 `0.0000141859055`，1,109 / 2,949,120 个 RGB 通道不同且最大差 1。
- `out/parity-zimage-default-768x1280-compile.json`（SHA `7b856aa0450246d4b988528a93dce227f037103e7119d9de1b21861e6070b1e0`）显式 `use_mlx_compile=True`、`teacache_mode="none"`；main compiled 与 main eager 逐位一致，Rust 9 步 compiled 与 Rust eager/main 也全部逐位一致。
- compiled 数值修复点：RoPE 表必须在 compiled closure 外预计算并显式 `eval`，否则 lazy 表会被追踪并产生 `4.0e-5` 的 step-0 差，随后递推放大。表现在按 image/caption/unified 三种几何复用，未引入全局或 `thread_local` 缓存。
- 性能仅作观察：Rust warm eager 9 步 `43.650s`、compiled `47.367s`（compiled/eager 108.5%），证据 `out/parity-zimage-default-rust-warm.json`（SHA `fcc938dd05bf6b07ab1c3c7348f1733883c664f289d68bc162dec63d00485190`）；main fixture 冷 `dit_load_and_denoise` eager `36.484s`、compiled `38.609s`。缓存状态与计时边界不同，且双方均为 F32 parity 权重，**性能门保持 pending**。
- 默认 `teacache_mode="auto"` 在 9 步启用 step cache并关闭 compile；该默认 TeaCache 路径尚未在 Rust 实现/验收，不能因上述两条路径通过而宣称完整默认优化路径通过。

#### checkpoint 原生 dtype：16×16 与 768×1280 parity / 显存门已通过

- `tests/parity/zimage_real_mlx.py --native-dtype` 显式恢复每个 checkpoint tensor 的原 dtype，生成生产层参考。官方 DiT 为 F32，VAE/Qwen3 为 BF16；没有强制把 DiT 转成 BF16。16×16 fixture `out/parity-zimage-native-16.json` SHA `2147a6118ac72b82d26383f6dad37021373cb1d90cd186f5f383ddfabb4e6f01`，默认分辨率 fixture `out/parity-zimage-native-default-768x1280.json` SHA `3a720fe5ab42813951fa81dc84ed7c7a65c9a7d0488c75c99473dd1f012c0360`。
- 两条完整切片均通过：DiT velocity 和全部 recurrent latent 逐位一致，VAE 输出和全部 RGB 通道也逐位一致。默认分辨率为 9 步、2,949,120 个 RGB 通道零差异；16×16 为 8 步、768 个通道零差异。
- 原生 VAE 对齐了四个实现细节：GroupNorm 使用与 main 相同的 reshape + `mx.fast.layer_norm`；dense Linear 使用 `addmm`；1×1 shortcut 保持 `conv2d`；padding 按 `[O,I,kH,kW]` 的最后两维判断。小图所有 norm/conv/shortcut/upsample 阶段均逐位一致。VAE fixture `out/parity-vae-native.json` SHA `fc0a2c61d41f890279863bfe62923ad867f20ce25a6b72127d9c3a5fb4d06f6b`。
- `danmo-mlx` 新增 active/peak memory 安全读取。默认分辨率 Rust peak `26,151,442,176` bytes，主侧 peak `31,793,682,130` bytes，比例 **82.25%**，通过不超过主侧 105% 的门；DiT 权重常驻 `24,619,634,944` bytes。报告 `out/parity-zimage-native-default-rust-memory.json` SHA `9cc2dab06d2df789ea01fcde06a512dd298fbe6e1d0a292f4a83694e81525139`。
- 这些结果证明 checkpoint-Native 路径，不代表默认 TeaCache、已发布量化 checkpoint、正式热性能或生产路由已完成。

#### 默认 9 步 TeaCache 与 residual skip 已通过

- `families/z_image/mlx/step_cache.rs` 实现 Z-Image 私有 gate：首尾强制 compute、非有限输入 fail-open、rel-L1、五次多项式 predicted distance、累计距离、compute 后更新输入/residual，以及 skip 时复用 residual 且不覆盖上一输入。主 DiT forward 复用同一路径，没有第二套模型实现或全局缓存。
- 默认阈值 `0.11` 的 768×1280 / 9 步 fixture `out/parity-zimage-native-default-teacache-768x1280.json`（SHA `aa70a1f117483ff54f8104e27ae61184729dcd6d74d3144721d2fdf7e2732f82`）中，双方均为 9 次 compute、0 次 skip；所有 gate kind、rel-L1、predicted/accumulated distance、latent、VAE 和全部 RGB 通道一致。
- 显式阈值 `0.5` 的 skip fixture `out/parity-zimage-native-teacache-skip-768x1280.json`（SHA `1aabbbf2918a99ffad8fe2c4468ee3303a2f707c6c8a01edf3ca09c63552925a`）中，双方均为 3 次 compute、6 次 skip，完整覆盖连续 residual 复用；全部数值和 RGB 通道同样一致。该阈值只用于缓存行为验收，不改变默认 0.11 结论。
- 默认 TeaCache peak：Rust `26,270,390,016` bytes，主侧 `31,915,106,770` bytes（约 82.3%），报告 `out/parity-zimage-native-default-teacache-rust-memory.json`（SHA `07a78f175dff29604e39cce3a9ca1b3649fc9100742a0e6f656e916081387daa`）。
- 注意：主侧 `DiTRuntimeSession.from_plan` 不传递自定义 threshold，skip fixture 在 runtime 创建后显式覆盖阈值；这是测试专用行为，已保留记录。
- `z-image-turbo/fp16/create` 的 Metal parity 因此标为 `passed`；正式热性能、已发布量化 checkpoint、连续会话/路由 E2E 和 CUDA 仍独立待办。

#### 正式热性能门已通过（2026-11-18，Metal / release / checkpoint-Native / 默认 TeaCache）

- **生产优化路径落地**：`CaptionCache` 在每次生成开始时预计算 caption 分支（`cap_norm → cap_embedder → context refiner`，对齐 main `before_denoise` 的 `needs_precompute_cap` 边界）；`CompiledCompute` 用 `mx.compile` 编译 30 个 unified blocks + final layer，**TeaCache gate、residual 复用、caption 预计算保持 eager**（对齐 main「gate 在 compile 外」的边界）。skip 步只跑编译后的 final layer closure。`StepCache::reset` 支持常驻模型连续生成。
- **compiled TeaCache 数值验收**：`real_compiled_teacache_matches_eager_cache_semantics` 在真实权重 768×1280 下，默认阈值 0.11（9 compute / 0 skip）与强制阈值 0.5（3 compute / 6 skip）两条路径，compiled 与 eager cached 的全部 9 步 velocity **逐位一致**，gate kind/compute/skip 计数与 main fixture 一致。
- **正式热门（§8.2，p50 ≤ main 105%）**：`real_native_warm_passes_match_main_and_report_performance`，5 pass 权重常驻，每 pass 对 main warm 基线逐位校验（caption、全部 velocity、全部递推 latent、全部 RGB）：
  - encode `0.0735s` vs main `0.0759s`（**96.8%**）
  - denoise `18.299s` vs main `28.798s`（**63.5%**）
  - VAE `0.2707s` vs main `0.3184s`（**85.0%**）
  - total `18.641s` vs main `29.194s`（**63.9%**）
  - 证据：`out/parity-zimage-native-default-teacache-rust-warm.json`（SHA `69ba38dc6d5227216e87037768562c292cde5657bc06a799c2dfde6d016327fe`）对 `out/parity-zimage-native-default-teacache-main-warm.json`（SHA `db0a92bb07a368a11abe9eef3ad3ab06bf93f85192b376e7c05ebf9d509a143d`）。
- **稳态显存门**：双方 `clear_cache` 清空可复用 buffer 池后，Rust 常驻 `32,663,479,666` bytes vs main `32,907,992,549` bytes（**99.3%**，低于 105% 门）。多 pass peak（Rust 39.85GB vs main 39.71GB，100.35%）主要是双方 buffer 池随 pass 数累积，不是常驻权重差异；单 run peak 门（82.25%）仍有效。
- **测量方法学（重要 gotcha）**：M5 Max 持续负载有显著热节流——同一 main 基线在会话内 denoise p50 从 15.2s 漂移到 28.8s（近 2×），Rust 同样漂移。因此**性能门必须 main/Rust 相邻同热态窗口测量**（本次为 Rust 5 pass 后立即跑 main 5 pass），且 parity 读回（`to_vec_f32`/`to_raw_bytes`）全部放在计时段外。此前「Rust 111–124%」的失败读数即热窗错位 + 计时段内读回所致，均已修复而非放宽阈值。eager Native 路径（`DANQING_TEST_ZIMAGE_WARM_EAGER=1` / full-compile 探针）同窗口也达标（denoise 82.0%），但生产默认是 compiled TeaCache。
- **共享封装新增**：`danmo-mlx` `MultiClosure`（多输出 `mx.compile`，输出向量用 `mlx_vector_array_set` 整体写入——`set_value` 是整体替换不是追加）、`Array::eval_all`（`mx.eval(a,b,…)`）、`Array::to_raw_bytes`（BF16 逐字节精确比较）、`clear_cache`。
- **测试基建修复**：`weights/mlx.rs` 测试 `Scratch` 临时目录名加入进程内原子计数器——macOS 粗时钟让并行测试拿到相同 pid+nanos 目录名，造成跨测试 shard 串扰（`duplicate tensor`）与文件被删的假失败。
- **TF32 gotcha 复核**：源码 Metal 构建下，host 参考 F32 微探针（`danmo-mlx` SDPA、z_image attention/blocks）在 M5 NAX 上需 `MLX_ENABLE_TF32=0`；真实 BF16 模型路径不受影响。全套件 TF32-off 绿：`danmo-mlx` 16+2+3+3、engine `mlx::` 32、architecture 5。
- **fixture SHA 漂移说明**：重新生成的 TeaCache fixture 不再匹配 2026-11-17 记录的 SHA，因为 fixture 脚本新增了 timing/memory 字段；parity 断言均对同轮新生成的 main 输出执行（default SHA `3017444cb4a9f84a76de0bed98ed4ca41564315de2b13e1ca2ef804e172e98bc`，skip SHA `4fc6931b5f457b687b4359cfb2a669db836509fc5db364a5a15e654a8ff4d274`）。
- `z-image-turbo/fp16/create` 的 `metal_performance` 与 `metal_peak_memory` 标为 `passed`；已发布量化 checkpoint、连续会话/生产路由 E2E、CUDA 仍独立待办。

#### 会话生命周期验收通过 + padding 语义修复（2026-11-18）

- **真实 bug 修复（padding 语义）**：main 在 `x_embedder`/`cap_embedder` **之前**用 `pad_tail_with_last` 重复最后一行 patch/caption 到 32 的倍数，再用 `x_pad_token`/`cap_pad_token` 覆盖 embed 后的 pad 位置。Rust 此前只对 embed 后的真实 token 做线性、pad 行直接拼 pad_token——当 pad 行参与注意力（小图，如 16×16 的 4 image token + 28 pad）时产生 1 BF16 ULP 的 `x_embed` 差，经 skip 步 residual 放大为可见 velocity 差。修复后 `embed_image`/`embed_caption` 与 main 完全同序；16×16 9 步 TeaCache（5 compute / 4 skip）全部 velocity、VAE、768 个 RGB 通道**逐位一致**。无 pad 几何（768×1280、1024×1024）数值不变，既有编译/门结论不受影响。定位方法：stage fixture 增加 TeaCache replay（cache 状态移植 + 记录步输入 latent），子阶段 dump 对比锁定 `x_embed` 单点分叉。
- **生命周期门（`real_session_lifecycle_consecutive_switch_cancel_unload`，真实权重、release）**：
  1. 常驻权重连续两次 768×1280 默认 TeaCache 生成：逐步逐位一致；
  2. 切换 16×16 几何（compiled compute 重建）再切回：均逐位一致；
  3. 另一线程 2.5s 后取消：`cancelled at step 2` 显式失败，取消后的下一次完整生成仍逐位一致（cache/compiled 状态无污染）；
  4. unload：drop 权重与 compiled closure + `clear_cache` 后 active 从 24.82GB 回落到 119MB（≥90% DiT 常驻字节释放）。
- **取消原语**：`danmo-mlx::CancelToken`（跨线程只传普通数据，§1.3）；`z_image` denoise 增加 `euler_cancellable`，在 step 边界检查取消；in-flight forward 不中断（边界已记录）。CPU 单测覆盖「取消后不再启动新 forward」。
- **热性能测量补充**：M5 Max 持续负载后的热恢复需要数分钟（Rust denoise 从 43s 逐步回落到 15s 量级）；warm 脚本/测试新增 `--warmup` / `DANQING_TEST_ZIMAGE_WARM_WARMUP`，p50 窗口取热恢复后的稳态段。
- **padding 修复后性能门复验通过**（双方各 5 warmup + 5 计量 pass、相邻热窗）：warm p50 encode **99.7%**、denoise **92.9%**、VAE **102.2%**、total **93.0%**；每个计量 pass 逐位一致；稳态常驻显存 32.66GB vs 32.91GB（99.3%）。证据 SHA：Rust `1d17b0edcdb34bd3deed1949350007ce834d4b705a8f8f97a65e20b294e66526` / main `7c6d4fc6dc7a7facef5d93b7373f1b9c74dfc1e4aaa36ecafd3a5a7cdc49e5b0`，摘要 `tests/parity/evidence/zimage-native-default-teacache-warm-gate-2026-11-18.json`；生命周期证据 `tests/parity/evidence/zimage-session-lifecycle-2026-11-18.json`。

#### S0 原生构建与最小分发

- vendor/pin mlx-c + MLX。
- 新建 `danmo-mlx-sys`、最小 `danmo-mlx`。
- macOS Metal Release、CPU smoke。
- Linux CUDA Release、CPU smoke；实际 CUDA runtime 在有硬件环境执行。
- 最小 macOS app 分发检查 dylib、metallib、rpath、deployment target、codesign。
- Linux 检查 MLX/CUDA/cuDNN 动态库加载。
- health 输出实际版本、backend、device 和 probe。

#### S1 安全封装与算子探针

- RAII、array ownership、dtype/shape、stream/device guard、error、eval、closure、panic 边界、safetensors、RNG、memory。
- Metal/CUDA 分别验证 matmul、QMM、index/gather/scatter、norm、RoPE、SDPA/GQA/mask、conv2d/3d、conv transpose、FFT/STFT、compile/eager、custom kernel。
- 单独验证超过 32K packed rows 的 QMM。
- 建立长期 executor 和跨任务缓存原型。

#### S2 Z-Image turbo 垂直切片

- 9 步默认 create（768×1280、默认 scheduler、`main` 实际 compile 状态）与 8 步 DistillPatch 场景分别验收。
- tokenizer/text encoder、DiT、VAE。
- fp16 和至少一个已发布量化版本。
- fp16 以 checkpoint 原生 dtype 为生产路径（§1.5.1；本官方 DiT 实测为 F32），记录峰值显存；F32 parity 模式仅用于机制层对齐。
- 固定注入噪声与固定 seed 两类测试。
- Metal/CUDA parity。
- 连续任务、模型切换、取消、缓存命中和 unload。

fixture：

```text
token IDs → text embeddings → initial noise → sigmas/timesteps
→ 选定 denoise step → final latent → RGB → PNG
```

#### S3 代表性缺口

- ACE 5Hz LM decode。
- H3 大 QMM。
- SeedVR2/Wan/LTX conv3d/conv transpose。
- 非标准 head_dim、sliding/additive mask、BSA。
- 一个 Metal custom kernel 和对应 CUDA kernel。
- DiT/FFN/norm/elementwise compile 实测。
- 长序列峰值内存。
- 在真实 CUDA 机器上尽早用一个视频 family（SeedVR2 或 Wan）跑通 conv3d、QMM、attention 算子探针，再决定 Linux 是否承诺全量。

#### S4 决策门

- 两后端构建和实际运行通过。
- Z-Image 与关键微基准 parity 通过。
- 热运行性能达到 Python 约 95%。
- C API、compile、custom kernel 缺口均有已验证路径。
- executor 与缓存稳定。
- 最小分发通过。

只有 Metal 通过时可继续实验，但不得宣称 CUDA 门通过或全量方案已验证。

### P0H — main 收割

状态：`pending`

与 spike 独立：

- 扩展现有 governance，不重复造一套。
- ModelCache/TextEmbeddingCache revision。
- submit gate、structural guide、capacity。
- tokenizer 和前端小件。
- fixture 转录。
- **修复 `main` 稠密权重 F32 放大**（§1.5.1）：`_assign_param_tensor` 按 checkpoint dtype 赋值，修复后重新生成 BF16 parity fixture，作为 Rust 生产层 parity 基线。
- ACE 新编辑模式作为独立产品特性，不阻塞 spike。

### P1 — Rust server 核心控制面与双引擎桥

状态：`pending`

必须先完成：

- contracts/catalog/store/schedule 核心对齐。
- 真实 DB fixture。
- 实时 SSE、错误、本地化、鉴权。
- EngineRouter 与能力矩阵。
- Python 纯推理 worker 和文件 manifest。
- Rust server 唯一正式 DB 写者。
- 过渡训练 worker 去正式 DB。
- 核心路由进程内 E2E：registry/tasks/assets/canvas/settings/download/generation。
- Rust 进程内 engine 与 Python worker 都走同一 scheduler/ArtifactSink/SSE。

外围空实现按依赖继续补，不以“路由存在”作为 parity。

P1 出口后，家族迁移只改 engine/common/family/能力路由，不再改控制面架构。

### P2 — MLX runtime 完整化

状态：`pending`

- executor/compile/shapeless/stream/memory；先定 executor 状态模型（§1.3：类型化 ModelCache、取消 token、进度事件）。
- pipeline/session trait 去 ggml 类型，改为后端中立；MLX 路径接入 driver（目前 `families/z_image/mlx` 仅在测试中使用）。
- attention 统一请求和原生/通用/custom/unavailable 分派。
- 权重、remap、量化、LoRA、revision cache；共享 `WeightDtype` 策略（生产 Native，F32 仅 parity）。
- conditioning/capacity/probe。
- TeaCache/step-cache 原语。
- Metal/CUDA custom kernel 管理。
- panic/OOM/cancel/restart 回归。

### P3 — common 共享层

状态：`pending`

- ops/schedulers/CFG/norm/embedding。
- Qwen3/Qwen2.5-VL/Qwen3-VL trunk。
- umT5/Gemma/CLIP/T5/BERT/byT5/whisper 所需组件。
- AutoencoderKL/VaeSpec、2D/3D VAE、tiling、stream cache、TAESD。
- PNG/JPEG/WebP、WAV/FLAC/MP3/OPUS/AAC、ffmpeg 视频和音轨。
- DiffRhythm 中文 G2P 的 ONNX Runtime C API 路径，去除动态执行 bundle Python。
- 所有共享算法独立 fixture 和至少两个使用方适配检查；单使用方组件不过早抽象。

### P4 — 图像核心

状态：`pending`

- `z_image`：create/edit/LoRA/controlnet union/structural guide/latent refine/LEMICA/TeaCache。
- `flux2`：Klein 4B/9B/base、create/edit/LoRA。
- `qwen_image`：create、edit/edit-2511、lightning、多参考、LoRA。

核心 trainer spike 可在本阶段对应 family forward 稳定后启动，不等视频家族。

### P5 — 图像长尾

状态：`pending`

- `flux1` 所有已发布 model card 和结构变体。
- `fibo`、`ernie_image`、`cogview4`、`hidream_o1`。
- `step1x_edit`、`boogu_image`、`longcat`、`esrgan`。

LongCat BSA 必须分别证明 Metal/CUDA 正确性和性能。

### P6 — 音频

状态：`pending`

- ACE create/cover/cover-nofsq/repaint/lego/extract/complete。
- 5Hz LM、PMI、APG/Haar-DCW/DTW/FSQ、vocal、LoRA。
- 多格式、denoise、near-silent。
- DiffRhythm、mulan、BERT、中文 polyphone G2P。

### P7 — 超分

状态：`pending`

- SeedVR2 image/video。
- tiled causal 3D VAE、temporal window、尾块、per-chunk seed。
- wavelet + LAB、ffmpeg 保音轨。
- Metal/CUDA conv3d 性能和长任务内存。

### P8 — 视频

状态：`pending`

- Wan t2v/i2v/14B/MoE/distill/LoRA/bernini。
- LTX 两阶段 A/V、i2v、per-token timestep、mask、IC-LoRA、长视频、BigVGAN+BWE。
- LTX25。
- Hunyuan animate、byT5、1080p SR。
- MiniMax H3 Ref2VA、Qwen3-VL、turbo、大 QMM。
- LongCat Avatar whisper 和 A/V conditioning。
- 长任务心跳、取消、内存增长、ffmpeg 尾块回归。

### P9 — 外围、本地 assistant 和 Rust 训练

状态：`pending`

外围：

- download/install/components/manifest/assert bundle/batch/setup。
- workspace registry、user LoRA、Civitai、presets、gallery、metadata。
- observability、diagnose/diagnostic、MCP、CLI。

Assistant：

- qwen3/qwen3_vl、chat template、vision preprocess、sampler、KV cache、SSE。
- TTL/unload、Metal/CUDA、云代理和 503 语义。
- parity 后退役 `backend_llm`。

训练：

- T0：小模型 loss/grad/optimizer/resume spike，再做 Z-Image 短训练。
- T1：共享 training runtime、参数树、optimizer、schedule、checkpoint、RNG、memory。
- T2：数据集、crop/augmentation、latent/text cache、旧 checkpoint 导入和恢复语义。
- T3：z_image、flux1、qwen_image trainer。
- candidate evaluation 复用 Rust inference。
- 每个 trainer 做 loss、gradient、多步参数、resume、best checkpoint、输出 LoRA 推理和 Metal/CUDA 验收。
- 完成后删除过渡 Python trainer 和其 Python family 依赖。

### P10 — 切换、退役与发布

状态：`pending`

- 23 个 runtime family 和训练能力矩阵全部达到发布要求。
- 默认 Rust server/engine/assistant/training。
- 保留明确观察期和显式回滚，不做自动 fallback。
- 删除 Python serving/inference/training 和 `backend_llm`（Python training 是否删除见 §1.6 待确认建议）。
- 删除 ggml 计算栈：`danmo-ggml-sys`、`native/third_party/ggml`、各 family 的 ggml 实现、GGUF 解析/bundle/下载逻辑。规则：每个 family 的 MLX 版点亮时，在同一 commit 删除其 ggml 实现，P10 只做最终清点，不允许长期维护 Python/ggml/MLX 三套实现。
- 更新 AGENTS、engine architecture、Cursor rules、打包和 CI。
- 干净 macOS Metal 与 Linux CUDA 安装、生成、训练和升级测试。
- 全量回归通过后才结束目标。

## 8. 每个 model/version/action 的统一验收门

### 8.1 数值

- token IDs 精确一致。
- scheduler、conditioning、noise、关键 step、final latent、decode 分阶段比较。
- fp16/bf16/int4/int8/mlx-q4/mlx-q8 分别设置阈值。
- 固定注入噪声隔离模型误差，另测 RNG。
- Metal/CUDA 分别对 Python 同后端比较，不要求跨 GPU 后端逐 bit 一致。
- 音频检查 waveform、时长、非静音和必要频谱指标；视频检查帧序列、fps、时长和音轨。

### 8.2 性能与内存

- 同硬件、MLX core、权重和缓存状态。
- 热运行 p50 延迟不超过 Python 105%，或吞吐不低于 95%。
- p95 无未解释异常退化。
- 峰值内存原则上不超过 Python 105%，超限必须复核。对照对象是 **checkpoint 原生 dtype 路径**：`main` 未修复低精度权重 F32 放大前，其显存数字不能作为上限（否则 Rust 照搬放大也能「通过」）；以各 shard 实测 dtype 的权重占用加实测激活为基线，不得假设所有 DiT 都是 BF16。
- 冷启动、首次 compile、加载单独报告。
- 不得测试失败后事后放宽阈值。

### 8.3 E2E

- submit/progress/trace/cancel/success/expected failure。
- artifact、lineage、metadata、restart/orphan。
- 连续运行/cache hit/model switch/unload。
- Metal/CUDA 各自运行。
- LoRA/edit/controlnet/多参考/批量等条目相关选项。

### 8.4 训练

- loss 和每个可训练 tensor 的 gradient。
- frozen 参数无梯度。
- accumulation/clipping/optimizer/schedule。
- 多步训练曲线和参数。
- checkpoint round-trip、完全恢复与权重恢复的区别。
- validation/best/candidate。
- 输出 LoRA 被 Rust 推理加载。
- 峰值内存、连续训练、取消和异常恢复。

## 9. 家族扩展 checklist

新增或迁移 family 时必须回答：

1. registry 中有哪些 model/version/action，不只是 family 名称？
2. 与现有 family 共享的是数学、权重格式、tokenizer 还是仅名字相似？
3. DiT/Job/Facade 哪个形状最自然，是否会迫使 shared pipeline 增加家族分支？
4. text encoder、VAE、scheduler、attention 是否可参数化复用？
5. 共享后是否有独立 fixture，且不需要知道 family id？
6. 权重 remap、quant、LoRA 和 bundle 规则是否严格？
7. Metal/CUDA 各自需要哪些 probe/custom kernel？
8. 取消、缓存、内存和长任务语义是否完整？
9. 能力矩阵、driver、registry projection 和路由是否同步？
10. 新 family 是否只增加自己的目录、配置、工厂行和必要的中立 common，而未修改其他 family？

## 10. 风险与控制

| 风险 | 控制 |
|---|---|
| 自有 mlx-c 封装维护 | 单点 sys/safe 层、RAII、fixture、精确 pin |
| C API 缺口 | 唯一 native shim，不扩散到 family |
| compile 性能 | 每闭包实测，通用算子/custom kernel/shim |
| Array 跨线程 | `!Send` + 长期 executor |
| Metal kernel 不可复用 | 通用实现或独立 CUDA kernel |
| CUDA 算子/性能差异 | 独立 probe/parity/performance/unavailable |
| server 范围膨胀 | 核心控制面先行，外围按依赖并行，不重复迁移 |
| 双栈 DB 竞争 | 活跃控制面单写者，worker 无正式 DB |
| 大媒体 IPC | 文件 manifest，不传 base64 |
| family 平均通过掩盖变体 | model/version/action 矩阵 |
| 训练长期双栈 | 核心推理稳定后启动训练波次，最终删除 Python trainer |
| 分发失败 | P0 最小包装，P10 全量干净环境验证 |
| 执行遗漏 | 每阶段更新本文、矩阵、证据和架构文档 |
| 照搬 main 的 F32 权重放大 | 共享 `WeightDtype`，生产 Native；F32 仅 parity；显存门以原生 dtype 为准（§1.5.1） |
| 推理崩溃拖垮控制面 | 推理 worker 与控制面进程隔离（§1.4，待确认） |
| Python/ggml/MLX 三套实现并存 | family 点亮即删 ggml 实现；P10 删除 ggml 栈 |
| 分支 registry 与矩阵基线漂移 | P1 前恢复 MLX 版本键并做同源校验（§0.1） |

## 11. 阶段执行日志

| 阶段 | 状态 | 证据/备注 |
|---|---|---|
| 分支与基线 | done | `feat/rust-mlx-c-migration`；保留已有未提交 ggml 工作 |
| 方案落盘 | done | `docs/rust_mlx_migration_plan.md` |
| 能力矩阵 | in_progress | 从固定 main registry 生成：92 个 card 中 54 个有 action 的 model、23 family、227 个 model/version/action 条目：`docs/rust_mlx_migration_matrix.json`。38 个组件 card 未登记、`architecture_variant` 未分类、峰值显存字段未建、Z-Image 条目状态需拆分，见 §4 |
| 文档复核 | done | 2026-11-17：纠正 Euler 归属、Z-Image 默认 9 步与 compile 策略、`dbcfaff` 提交边界、分支 registry 偏差；确立稠密权重原生 dtype 决策（§1.5.1，main F32 放大为错误设计）；增补 executor 状态模型、assistant 独立 worker、ggml 退役；推理进程隔离与 Python 训练去留标记为待确认 |
| 共享资产盘点 | done | 复用/重写/算法归属与 family 扩展边界：`docs/rust_mlx_shared_asset_inventory.md` |
| 全量 workspace 回归 | existing_failure | 373 项 engine 测试：370 通过、2 忽略、1 个既有 ggml Hunyuan smoke（随 `dbcfaff` 一并提交） 因缺少 `txt_in.individual_token_refiner.blocks.0.self_attn.qkv.weight` 失败；新增 MLX crate 的 CPU/Metal 测试均单独通过，不在 P0 修改该 ggml family |
| P0 | in_progress | **源码 Metal 构建通过**；runtime、safetensors、>32K QMM、compile/eager、executor、seeded RNG 探针通过；官方 DiT/text/VAE/tokenizer 已下载；真实 Qwen3 与 VAE fixture 通过。`9859706` 已落地共享 `WeightDtype` 与全程 packed affine QMM。F32 parity 已通过 16×16、1024×1024、768×1280 9 步 eager/compile。checkpoint-Native 16×16 与 768×1280 完整切片、默认 9 步 TeaCache 及阈值 0.5 的 6 次 residual skip 均通过，latent/VAE/RGB 一致；官方 DiT 实测为 F32，VAE/Qwen3 为 BF16。Native Metal peak 26.15GB vs main 31.79GB（82.25%）。**2026-11-18 正式热性能门通过**：compiled TeaCache 生产路径（CaptionCache + CompiledCompute，gate eager）稳态复验 warm p50 denoise 92.9% / encode 99.7% / VAE 102.2% / total 93.0%，每 pass 逐位一致；稳态显存 99.3%；`metal_performance`/`metal_peak_memory` 标为 passed。**会话生命周期门通过**（连续生成/几何切换/跨线程取消/unload，真实权重 release）；修复 padding 语义（embed 前重复末行，小图 pad 参与注意力时 1 ULP 分叉），16×16 TeaCache 5c/4s 逐位一致。剩余：已发布量化 checkpoint 端到端、custom kernel、干净打包、CUDA runtime、executor 级 ModelCache（P2）|
| P0H | pending | main 收割独立进行 |
| P1 | pending | Rust server 核心先行 |
| P2 | pending | MLX runtime |
| P3 | pending | common 共享层 |
| P4 | pending | 图像核心 |
| P5 | pending | 图像长尾 |
| P6 | pending | 音频 |
| P7 | pending | 超分 |
| P8 | pending | 视频 |
| P9 | pending | 外围、assistant、训练 |
| P10 | pending | 切换与退役 |
