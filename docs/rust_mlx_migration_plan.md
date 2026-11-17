# Rust + MLX-C 全量迁移执行方案

> 状态：执行中  
> 执行分支：`feat/rust-mlx-c-migration`  
> 本文是唯一阶段计划与验收索引。执行过程中同步更新状态、证据、偏差和能力矩阵，不凭记忆缩减范围。

## 0. 基线与目标

### 0.1 固定基线

- ggml Rust 基线 commit：`67495d3353d64f1ffd605e3586d0eea74837aebe`。
- `main` 行为基线 commit：`ab4456cb1a712b9e8a56a0629862170c967bef52`。
- 创建迁移分支时工作树已有未提交的 ggml Rust 修改，全部保留；不得把已有修改误记为本迁移新增成果。
- `main` registry 基线：92 个 model card，23 个 runtime family，其中 21 个生成 family，另有 `qwen3`、`qwen3_vl` 两个 assistant family。
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

### 1.5 权重格式不变

- 保留 MLX safetensors、fp16/bf16/fp32、int8/int4、mlx-q4/mlx-q8 及 scales/biases 布局。
- 保留 main 的 `remap_*`、sanitize、LoRA 和 bundle 语义，在 Rust 权重层实现。
- 不引入 GGUF、repack 或 ggml 权重布局。
- ggml 分支只借鉴架构、共享组件、tokenizer、测试语料和治理，不继承其计算后端和权重格式。

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
adapters/options
python_baseline_status
rust_driver_status
metal_parity
cuda_parity
metal_performance
cuda_performance
e2e
route
evidence
notes
```

状态只能使用受控值，例如：

```text
unstarted / in_progress / blocked / parity_failed / ready / routed / unavailable
```

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
- BF16/FP16/F32 widen 按 main 家族约定。
- affine int4/int8 与实际发布的量化模式。
- LoRA、多 LoRA、系数、量化和非量化绑定。
- 未完成安装的 bundle 不得进入缓存。

### 6.2 revision

缓存键包含：

- 所有 shard 的 canonical path、长度和 mtime nanos。
- manifest install generation 或内容摘要。
- tokenizer/config/preprocessor revision。
- backend/device/dtype/quant。
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

当前执行记录：macOS 源码 Metal 构建最初因本机缺少 Apple Metal Toolchain 组件而失败，P0 runtime probe 曾使用 PyPI 官方 `mlx-metal 0.32.4` macOS 14 wheel（SHA-256 `8019b2db622599267d66cc168b2f0dbf5af929790a6c3b3913e00eabc42a7947`）配合源码 `mlx-c`。2026-11-17 经用户确认安装 `xcodebuild -downloadComponent MetalToolchain`（Build 27B47）后，**源码 Metal 构建门通过**：`cargo clean -p danmo-mlx-sys` 后从 pin 的 MLX/mlx-c/fmt/json 源码完整构建（本机 Apple M5 Max / macOS 27.2，deployment target 27.2，NAX kernels 已编入），`danmo-mlx --features metal` 16 单测 + 2 守卫 + 2 safetensors、engine `mlx::` 31 项（含全部真实 fixture 与端到端）全部通过，且不需要 `DANQING_MLX_PREBUILT`。源码构建热性能与 wheel 一致（compiled 8 步 0.640s）。干净打包验证仍未完成，保留在 S0/S4 门中。

**构建环境 gotcha（已踩坑并记录）**：在 prebuilt 与源码两种链接模式间切换后，若不重新链接测试二进制，`@rpath/libmlx.dylib` 可能解析到旧产物，出现「同一测试时过时不过」的假数值分叉（曾表现为 SDPA 8e-4、Qwen3 caption 误差 4 的假失败）。切换后必须 `cargo clean -p danmo-mlx-sys`（或触发重链接）再跑；确认失败前先做单测隔离 + A/B 两种链接各跑一遍。另外 MLX 0.32.4 默认 `MLX_ENABLE_TF32=1`，NAX 芯片（M5 系）上 F32 matmul/SDPA 可能走 TF32；host 参考微探针在 TF32 开启时可能超 F32 容差，可用 `MLX_ENABLE_TF32=0` 做精确对照（真实 BF16 模型路径不受影响，main 与 Rust 同为 MLX 默认策略）。

Z-Image 资产检查：最初本机安装只有 GGUF DiT；经用户明确授权，已按 main registry 的 `Tongyi-MAI/Z-Image-Turbo` 下载 3 个官方 safetensors DiT shard 到 `/Users/nil.luo/Workspace/studio-workspace/models/Image/z-image-turbo-fp16/transformer`。Rust `SafeTensors` 已验证 521 个 tensors、关键 shape、跨 shard 唯一性和 remap 无冲突。

Z-Image Metal 固定注入噪声垂直切片已在当前未提交工作树通过：`tests/parity/zimage_real_mlx.py` 从固定 main 基线生成 16×16、8 步真实权重的 caption/latent/RGB fixture，`real_zimage_create_matches_main_latents_and_pixels` 使用同一 token IDs、初始 latent 和 sigmas 串联 Rust Qwen3、完整 DiT、Euler 与 AutoencoderKL。2026-11-17 复跑结果为 caption 最大误差 0、最终 latent 最大误差 `0.015625`（逐元素四个 BF16 ULP 门）、VAE 输出最大误差 `0.0030458868`；768 个 8-bit RGB 通道中 34 个不同，最大通道差 1。使用 main 最终 latent 单独解码时 VAE 最大误差为 `0.0000015497208`，确认残差来自逐步 DiT 数值累积。产物为 `out/parity-zimage-rust.png`，测试位于 `native/crates/danmo-engine/src/families/z_image/mlx/model.rs`。

垂直切片过程中确认并修复了三处与 main 的加载/调度语义差异（不是放宽容差）：

1. **稠密权重 F32 加载**：main 的 MLX `Module` 加载会把发布的 BF16 VAE/DiT 参数写成 F32（自定义 `_param_map` 切片赋值）。Rust 此前保留 BF16，导致真实 VAE 首个分叉出现在 `conv_in`（误差 0.0054 超门）。现 `common::vae::mlx::load_dense` 与 `families/z_image/mlx` DiT 加载统一 `astype(F32)`。独立真实 VAE 解码对 main 的最大误差降至 `0.0000014305115`，分阶段 trace（conv_in/mid/up blocks）最大 `0.00019836426`。
2. **真实 pad token**：DiT 图像/文本尾部 padding 必须读 checkpoint 的 `x_pad_token` / `cap_pad_token`，不能写零。修复后首轮 velocity 误差从 `1.496` 降到 `0.000932`。
3. **Euler BF16 语义**：main 每步把 velocity 与 `dt` 转回 latent 的 BF16 再更新；共享 `denoise::euler` 已对齐（输出与 delta 均 cast 到当前 latent dtype），CPU/Metal 单测覆盖 F32 与 BF16 两条路径。

组合测试的 VAE 拓扑必须取真实配置（该 VAE `layers_per_block=2`，不是默认 3）；此前硬编码 3 造成的 `0.305` 输出差已排除，与跨组件 MLX 状态无关。

固定 seed 切片同样通过：`out/parity-zimage-seeded.json`（seed 42）由同一 main 基线脚本生成，Rust `RandomKey::new(42).normal` 产生的初始噪声与 main **逐位一致**（测试内 `assert_eq`），2026-11-17 运行最终 latent 最大误差 `0.0078125`（四个 BF16 ULP 门内）、VAE 输出最大误差 `0.0027973056`、26/768 RGB 通道差且最大通道差 1，产物 `out/parity-zimage-seeded-rust.png`。`danmo-mlx` 的 seeded normal/split 在 CPU/Metal 均有对 main 的固定值单测。

该结果只点亮 fp16 turbo create 的 Metal 数值 parity（固定注入噪声与固定 seed 两类均通过），不代表生产路由或完整 E2E 已开放。至少一个已发布量化版本、连续任务/切换/取消/缓存、CUDA 和 encode/VAE 热性能门仍待完成。

#### compile 实测与热性能观察（2026-11-17，16×16 / 8 步 / 官方 fp16 权重 / Metal）

- `danmo-mlx` 新增多输入 `Closure`（`Fn(&[Array], &Stream)`，与 `UnaryClosure` 同一 FFI/panic 边界），CPU/Metal 各有「captured 常量 + 多输入 + shapeless 重放 + 形状变化重编译」与「闭包 panic 变错误不崩进程」单测；`Stream` 实现 `Clone`（共享同一 MLX stream handle）。
- `families/z_image/mlx/model.rs` 拆出 `embed_timestep` + `forward_with_timestep`，新增 `CompiledForward`：weights/geometry/caption 捕获为常量，仅 latent 与 timestep embedding 是 traced 输入。**pinned MLX 0.32.4 上 `Split`（AdaLN 分段）与 `Slice`（RoPE 分量、FinalLayer 裁剪）都没有 `output_shapes`，shapeless compile 对完整 DiT 显式失败**；使用 shape 特化 compile（形状变化自动重编译），这是已验证路径而非静默降级。
- tiny DiT compiled vs eager：CPU/Metal 跨 3 个 sigma 均 < 1e-5。真实权重 `real_compiled_dit_matches_eager_and_reports_warm_timings`：**8 步 velocity compiled vs eager 逐位一致（max error 0）**，eager 对 main 的 parity 与既有门一致。
- main 热基线（`tests/parity/zimage_warm_mlx.py`，同进程 3 passes，权重常驻）：eager 热 denoise `0.671s/0.669s`，encode `0.022s/0.018s`，VAE `0.005s`；`--compile`（`use_mlx_compile=True`）热 denoise 同为 `0.671s/0.673s`——main 的 compile 在该小图热态无额外收益。main 生产策略本来就是 8 步 turbo 不开 compile（`compile_policy.resolve_use_mlx_compile`，无 step_distill 时 `num_steps<=8` 返回 False），fixture 的 eager 基线即生产路径。
- Rust 热观察（warm-up pass 后）：eager 8 步 `0.706s`，compiled `0.662s`（eager 的 93.7%；对 main 热基线约 **101%**）。证据：`out/parity-zimage-warm.json`、`out/parity-zimage-warm-compile.json`、`out/parity-zimage-rust-warm.json`。
- **边界**：以上是 16×16 小图单 run 观察，encode/VAE 的正式热对比与真实分辨率（如 1024×1024）性能门未测；不得据此宣称 S4 全量热性能门通过。

#### 真实分辨率 1024×1024：当前未解决分叉（2026-11-17，parity_failed 切片）

`tests/parity/zimage_real_mlx.py` 已参数化 `--width/--height/--steps`，Rust 端到端测试改为从 payload 读分辨率。1024×1024 / 8 步 / seed 42（4096 image tokens + 32 cap tokens）：

- main 基线：encode 0.26s、denoise 35.3s、VAE 0.84s（冷）；main 自身 TF32 on/off 输出**逐位一致**（排除 TF32 因素）。
- Rust：encode parity 通过（caption 误差 ≤1e-6）；**step 0 velocity max error 0.024**（16×16 时为 0.00093），逐步放大至 step 7 的 1.08；最终 latent max error 0.0117，在小值元素上超 4 BF16 ULP + 4e-3 门（allowed 0.00595）。**未通过，不得放宽容差宣称完成**。
- 已排除：TF32（main on/off 逐位一致）；源码 vs wheel 构建（两种链接同误差）；RoPE 迭代乘法漂移（已改为 main 的 `pos * theta^(-2k/d)` 公式，误差不变，该修正保留）；sigma 公式（host 复算与 fixture 最大差 6e-8，仅 1 ULP，不足以解释 velocity 0.024）。
- 已核对无差异：position ids / patchify 顺序、pad token 应用、modulation unpack、Euler BF16 cast、timestep features 公式。
- 下一步：main 侧分阶段 fixture（x_embedder 后、noise_refiner 每层后、context_refiner 后、unified 每层后、final_layer 后），同一 1024 输入逐段对比定位首个分叉；重点怀疑长序列 attention kernel 分派（4128 tokens 时 main 与 Rust 的 mask/布局是否触发不同 steel/NAX 路径）与 F32 权重下大 K matmul 的 kernel 选择。16×16 全部门保持通过。

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

- 8 步 create。
- tokenizer/text encoder、DiT、VAE。
- fp16 和至少一个已发布量化版本。
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

- executor/compile/shapeless/stream/memory。
- attention 统一请求和原生/通用/custom/unavailable 分派。
- 权重、remap、量化、LoRA、revision cache。
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
- 删除 Python serving/inference/training 和 `backend_llm`。
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
- 峰值内存原则上不超过 Python 105%，超限必须复核。
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

## 11. 阶段执行日志

| 阶段 | 状态 | 证据/备注 |
|---|---|---|
| 分支与基线 | done | `feat/rust-mlx-c-migration`；保留已有未提交 ggml 工作 |
| 方案落盘 | done | `docs/rust_mlx_migration_plan.md` |
| 能力矩阵 | done | 从固定 main registry 生成 92 model、23 family、227 个 model/version/action 条目：`docs/rust_mlx_migration_matrix.json` |
| 共享资产盘点 | done | 复用/重写/算法归属与 family 扩展边界：`docs/rust_mlx_shared_asset_inventory.md` |
| 全量 workspace 回归 | existing_failure | 373 项 engine 测试：370 通过、2 忽略、1 个既有未提交 Hunyuan smoke 因缺少 `txt_in.individual_token_refiner.blocks.0.self_attn.qkv.weight` 失败；新增 MLX crate 的 CPU/Metal 测试均单独通过，不在 P0 修改该 ggml family |
| P0 | in_progress | **源码 Metal 构建通过**（MetalToolchain 27B47 已装，M5 Max/NAX，无需 prebuilt）；runtime、safetensors、>32K QMM、compile/eager、executor、seeded RNG 探针通过；官方 DiT/text/VAE/tokenizer 已下载；tiny/full DiT、真实 8 步 denoise 与 main tiny DiT BF16 parity 通过；真实 Qwen3 在 layer 1/2/10/20/30/35 与 main 输出均零误差；真实 VAE decode 最大误差 1.4e-6（F32 加载对齐）；16×16 完整图片端到端（固定注入 + 固定 seed 42）latent/RGB parity 通过，最大 8-bit 通道差 1；真实 DiT shape 特化 compile 8 步逐位一致，源码构建热 8 步 0.640s ≈ main 热基线 0.67s（shapeless 在 0.32.4 因 Split/Slice 缺 output_shapes 显式失败，已记录）。**1024×1024 端到端当前 parity_failed**（step0 velocity 0.024 → step7 1.08，latent 超 ULP 门；TF32/构建差异已排除，待分阶段定位）。剩余（按用户指示 fp16 验证优先，量化押后）：1024 分叉定位与修复、正式热性能门、连续任务/切换/取消/缓存、已发布量化版本（共享加载层自动识别，非家族分支）、custom kernel、干净打包和 CUDA runtime |
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
