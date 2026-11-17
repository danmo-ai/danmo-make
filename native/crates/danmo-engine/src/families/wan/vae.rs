//! Wan 2.2 causal 3D VAE decoder over the official `Wan2.2_VAE.safetensors`
//! key layout (PyTorch `[OC, IC, KD, KH, KW]` convs; the flat bytes double as
//! the ggml `[KW, KH, KD, IC * OC]` kernel).
//!
//! Semantics are a port of main's `wan_vae_mlx.py` (layout "wan22"):
//! - decode un-scales the latent per channel (`z * std + mean` with the
//!   hardcoded 48-dim Wan2.2 statistics), then `conv2` (1x1x1), then the
//!   decoder tower, then `unpatchify(2)`: `12` channels -> `3` at 2x.
//! - One latent frame is decoded per graph replay. Every CausalConv3d keeps a
//!   two-frame rolling cache of its pre-conv input; chunk `i` sees
//!   `[cache0, cache1, x_i..]` with a zero front pad on chunk 0. The cache
//!   slots ride in one flat graph input and the graph appends the next cache
//!   state to its output, so a replay is one compute plus one readback.
//! - `Resample` upsample3d (blocks 0 and 1): frame 0 bypasses `time_conv`;
//!   later frames go through it and split into two output frames (channel
//!   halves), so `T` latent frames give `1 + 2(T - 1)` after one block and
//!   `1 + 4(T - 1)` after both. The spatial half is nearest x2 then a 3x3
//!   conv2d over the frame batch.
//! - `DupUp3D` shortcut: blocks 0/1 (`in == out`, `ft=2`) are a pure nearest
//!   `(2, 2, 2)` duplicate with the first duplicated frame dropped on chunk 0;
//!   block 2 (`in == 2*out`, `ft=1`) folds channels:
//!   `out[2w+dw, 2h+dh, t, c] = x[w, h, t, 2c + dh]`.
//! - `WanVAERMSNorm` normalizes over channels with `x / sqrt(sum(x^2) +
//!   1e-12) * sqrt(C) * gamma`, which is ggml `rms_norm` with
//!   `eps = 1e-12 / C` plus the gamma multiply.
//! - The mid attention block is single-head over the `W * H` tokens of one
//!   frame, scale `C^-0.5`, fused `to_qkv` 1x1 conv.
//!
//! The first chunk and steady chunks differ in frame counts (time_conv bypass
//! and the DupUp frame drop), so decode keeps two reserved graphs. Activations
//! are `[W, H, frames, C]` F32 throughout (ne3 is channels). There is no
//! spatial tiling yet: a 480p decode holds the full 8x-resolution tower.

use crate::runtime::{shape, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const GRAPH_MEM: usize = 64 << 20;

/// Official Wan2.2 VAE latent statistics (main `_WAN22_VAE_MEAN`).
pub const WAN22_VAE_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];

/// Official Wan2.2 VAE latent statistics (main `_WAN22_VAE_STD`).
pub const WAN22_VAE_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

#[derive(Clone)]
pub struct WanVaeSpec {
    pub z_dim: i64,
    /// Decoder channel widths: `dims[0]` after `conv1`, `dims[i + 1]` the
    /// output of up block `i`. Four up blocks.
    pub dims: [i64; 5],
    /// Which of the first three up blocks upsample in time (upsample3d).
    pub temporal_up: [bool; 3],
    /// Head output channels before unpatchify (`3 * patch^2`).
    pub pixel_out: i64,
    pub patch: i64,
    /// Residual blocks per up block (`num_res_blocks + 1`).
    pub up_res_blocks: usize,
}

impl WanVaeSpec {
    /// The published `Wan2.2_VAE.safetensors` topology (TI2V-5B).
    pub fn wan22() -> Self {
        Self {
            z_dim: 48,
            dims: [1024, 1024, 1024, 512, 256],
            temporal_up: [true, true, false],
            pixel_out: 12,
            patch: 2,
            up_res_blocks: 3,
        }
    }

    /// Same topology at 1/128 the width for the host-reference test.
    pub fn smoke() -> Self {
        Self {
            z_dim: 4,
            dims: [8, 8, 8, 4, 2],
            temporal_up: [true, true, false],
            pixel_out: 12,
            patch: 2,
            up_res_blocks: 3,
        }
    }

    /// Infer the topology from the folded ggml shapes. The resample index is
    /// the resblock count, and the wan22 decoder always carries 3 residual
    /// blocks per up block, so any other count fails loud. Temporal upsampling
    /// is the presence of `time_conv` on the resample.
    pub fn infer(weights: &GgufWeights) -> Result<Self, WeightError> {
        let conv2 = weights.shape("conv2.weight")?;
        let z = (conv2[3] as f64).sqrt() as i64;
        if z < 1 || z * z != conv2[3] {
            return Err(WeightError::NotBuilt(format!(
                "wan vae conv2.weight is {conv2:?}, not [1, 1, 1, z*z]"
            )));
        }
        let conv1 = weights.shape("decoder.conv1.weight")?;
        if conv1[3] % z != 0 {
            return Err(WeightError::NotBuilt(format!(
                "wan vae decoder.conv1.weight is {conv1:?}, z_dim {z}"
            )));
        }
        let mut dims = [0i64; 5];
        dims[0] = conv1[3] / z;
        for i in 1..5 {
            let block = i - 1;
            let name = format!("decoder.upsamples.{block}.upsamples.0.residual.2.weight");
            let shape = weights.shape(&name)?;
            let ic = dims[i - 1];
            if ic < 1 || shape[3] % ic != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "wan vae {name} is {shape:?}, in channels {ic}"
                )));
            }
            dims[i] = shape[3] / ic;
        }
        for j in 0..3 {
            let gamma = format!("decoder.upsamples.0.upsamples.{j}.residual.0.gamma");
            if !weights.contains(&gamma) {
                return Err(WeightError::NotBuilt(format!(
                    "wan vae is missing {gamma}; the decoder carries 3 residual blocks per up block"
                )));
            }
        }
        if weights.contains("decoder.upsamples.0.upsamples.3.residual.0.gamma") {
            return Err(WeightError::NotBuilt(
                "wan vae has a fourth residual block; the wan22 topology carries 3".into(),
            ));
        }
        let mut temporal_up = [false; 3];
        for (i, slot) in temporal_up.iter_mut().enumerate() {
            *slot = weights.contains(&format!(
                "decoder.upsamples.{i}.upsamples.3.time_conv.weight"
            ));
        }
        let head = weights.shape("decoder.head.2.weight")?;
        if dims[4] < 1 || head[3] % dims[4] != 0 {
            return Err(WeightError::NotBuilt(format!(
                "wan vae decoder.head.2.weight is {head:?}, dim {}",
                dims[4]
            )));
        }
        let pixel_out = head[3] / dims[4];
        let patch = ((pixel_out / 3) as f64).sqrt() as i64;
        if patch < 1 || patch * patch * 3 != pixel_out {
            return Err(WeightError::NotBuilt(format!(
                "wan vae pixel_out {pixel_out} is not 3 * patch^2"
            )));
        }
        Ok(Self {
            z_dim: z,
            dims,
            temporal_up,
            pixel_out,
            patch,
            up_res_blocks: 3,
        })
    }

    /// Output frames for `t` latent frames: each temporal upsample maps
    /// `t -> 2t - 1`, so the wan22 pattern gives `1 + 4 * (t - 1)`.
    pub fn decode_frames(&self, t: i64) -> i64 {
        let mut frames = t.max(1);
        for &up in &self.temporal_up {
            if up {
                frames = 2 * frames - 1;
            }
        }
        frames
    }
}

enum Kind {
    Vector(i64),
    Conv2d {
        k: i64,
        ic: i64,
        oc: i64,
    },
    Conv3d {
        kw: i64,
        kh: i64,
        kd: i64,
        ic: i64,
        oc: i64,
    },
}

struct Param {
    name: String,
    kind: Kind,
}

impl Param {
    /// The ggml shape the weight file folds to (`[KW, KH, KD, IC * OC]`).
    fn ggml_ne(&self) -> [i64; 4] {
        match self.kind {
            Kind::Vector(c) => [c, 1, 1, 1],
            Kind::Conv2d { k, ic, oc } => [k, k, ic, oc],
            Kind::Conv3d { kw, kh, kd, ic, oc } => [kw, kh, kd, ic * oc],
        }
    }
}

fn push_conv3d(out: &mut Vec<Param>, prefix: &str, kw: i64, kh: i64, kd: i64, ic: i64, oc: i64) {
    out.push(Param {
        name: format!("{prefix}.weight"),
        kind: Kind::Conv3d { kw, kh, kd, ic, oc },
    });
    out.push(Param {
        name: format!("{prefix}.bias"),
        kind: Kind::Vector(oc),
    });
}

fn push_conv2d(out: &mut Vec<Param>, prefix: &str, k: i64, ic: i64, oc: i64) {
    out.push(Param {
        name: format!("{prefix}.weight"),
        kind: Kind::Conv2d { k, ic, oc },
    });
    out.push(Param {
        name: format!("{prefix}.bias"),
        kind: Kind::Vector(oc),
    });
}

fn push_resblock(out: &mut Vec<Param>, prefix: &str, ic: i64, oc: i64) {
    out.push(Param {
        name: format!("{prefix}.residual.0.gamma"),
        kind: Kind::Vector(ic),
    });
    push_conv3d(out, &format!("{prefix}.residual.2"), 3, 3, 3, ic, oc);
    out.push(Param {
        name: format!("{prefix}.residual.3.gamma"),
        kind: Kind::Vector(oc),
    });
    push_conv3d(out, &format!("{prefix}.residual.6"), 3, 3, 3, oc, oc);
    if ic != oc {
        push_conv3d(out, &format!("{prefix}.shortcut"), 1, 1, 1, ic, oc);
    }
}

/// The decoder parameter list in the official key layout (`conv2`,
/// `decoder.conv1`, `decoder.middle.{0,1,2}`, `decoder.upsamples.{i}.
/// upsamples.{j}`, `decoder.head.{0,2}`).
pub fn decoder_params(spec: &WanVaeSpec) -> Vec<Param> {
    let mut out = Vec::new();
    let d = spec.dims;
    push_conv3d(&mut out, "conv2", 1, 1, 1, spec.z_dim, spec.z_dim);
    push_conv3d(&mut out, "decoder.conv1", 3, 3, 3, spec.z_dim, d[0]);
    push_resblock(&mut out, "decoder.middle.0", d[0], d[0]);
    out.push(Param {
        name: "decoder.middle.1.norm.gamma".into(),
        kind: Kind::Vector(d[0]),
    });
    push_conv2d(&mut out, "decoder.middle.1.to_qkv", 1, d[0], 3 * d[0]);
    push_conv2d(&mut out, "decoder.middle.1.proj", 1, d[0], d[0]);
    push_resblock(&mut out, "decoder.middle.2", d[0], d[0]);
    for i in 0..4 {
        for j in 0..spec.up_res_blocks {
            let ic = if j == 0 { d[i] } else { d[i + 1] };
            push_resblock(
                &mut out,
                &format!("decoder.upsamples.{i}.upsamples.{j}"),
                ic,
                d[i + 1],
            );
        }
        if i < 3 {
            push_conv2d(
                &mut out,
                &format!("decoder.upsamples.{i}.upsamples.3.resample.1"),
                3,
                d[i + 1],
                d[i + 1],
            );
            if spec.temporal_up[i] {
                push_conv3d(
                    &mut out,
                    &format!("decoder.upsamples.{i}.upsamples.3.time_conv"),
                    1,
                    1,
                    3,
                    d[i + 1],
                    2 * d[i + 1],
                );
            }
        }
    }
    out.push(Param {
        name: "decoder.head.0.gamma".into(),
        kind: Kind::Vector(d[4]),
    });
    push_conv3d(&mut out, "decoder.head.2", 3, 3, 3, d[4], spec.pixel_out);
    out
}

/// Every declared tensor must exist at its folded ggml shape; the fourth up
/// block must not carry a resample. Anything else fails loud.
pub fn validate(weights: &GgufWeights, spec: &WanVaeSpec) -> Result<(), WeightError> {
    for param in decoder_params(spec) {
        let got = weights.shape(&param.name)?;
        let want = param.ggml_ne();
        if got != want {
            return Err(WeightError::Shape {
                name: param.name,
                expected: want.to_vec(),
                got: got.to_vec(),
            });
        }
    }
    if weights.contains("decoder.upsamples.3.upsamples.3.resample.1.weight") {
        return Err(WeightError::NotBuilt(
            "wan vae has a resample in the last up block; the wan22 topology does not".into(),
        ));
    }
    Ok(())
}

/// Load every decoder tensor widened to F32.
pub fn load(
    weights: &GgufWeights,
    spec: &WanVaeSpec,
) -> Result<Vec<(String, Vec<f32>)>, WeightError> {
    let mut out = Vec::new();
    for param in decoder_params(spec) {
        let data = weights.f32_widened(&param.name, &param.ggml_ne())?;
        out.push((param.name, data));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// cache slots
// ---------------------------------------------------------------------------

/// One cached CausalConv3d: two frames of its pre-conv input at `[w, h, c]`.
#[derive(Clone, Copy)]
struct Slot {
    w: i64,
    h: i64,
    c: i64,
}

impl Slot {
    fn len(&self) -> i64 {
        2 * self.w * self.h * self.c
    }
}

/// Cache slots in graph traversal order (the order the flat cache input is
/// laid out in and the order the graph appends the next cache state).
fn cache_slots(spec: &WanVaeSpec, w: i64, h: i64) -> Vec<Slot> {
    let mut out = Vec::new();
    let d = spec.dims;
    out.push(Slot {
        w,
        h,
        c: spec.z_dim,
    }); // decoder.conv1
        // middle.0 conv1 + conv2, middle.2 conv1 + conv2
    for _ in 0..4 {
        out.push(Slot { w, h, c: d[0] });
    }
    let (mut cw, mut ch) = (w, h);
    for i in 0..4 {
        for j in 0..spec.up_res_blocks {
            let ic = if j == 0 { d[i] } else { d[i + 1] };
            out.push(Slot {
                w: cw,
                h: ch,
                c: ic,
            });
            out.push(Slot {
                w: cw,
                h: ch,
                c: d[i + 1],
            });
        }
        if i < 3 {
            if spec.temporal_up[i] {
                out.push(Slot {
                    w: cw,
                    h: ch,
                    c: d[i + 1],
                });
            }
            cw *= 2;
            ch *= 2;
        }
    }
    out.push(Slot {
        w: cw,
        h: ch,
        c: d[4],
    }); // decoder.head.2
    out
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

struct Built {
    graph: ReusedGraph,
    frames: i64,
    pixels: i64,
    cache: i64,
}

fn nn(t: Tensor) -> Result<Tensor, RuntimeError> {
    if t.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(t)
    }
}

struct Decode<'g> {
    b: &'g GraphBuilder,
    w: &'g HashMap<String, Tensor>,
    cache: Option<Tensor>,
    offsets: Vec<usize>,
    slots: Vec<Slot>,
    slot: usize,
    next: Vec<Tensor>,
    first: bool,
    dup_rows: Tensor,
}

impl Decode<'_> {
    fn wt(&self, name: &str) -> Result<Tensor, RuntimeError> {
        self.w
            .get(name)
            .copied()
            .ok_or_else(|| RuntimeError::Message(format!("wan vae graph is missing {name}")))
    }

    fn shaped(&self, tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
        let got = shape(tensor);
        if got != ne {
            return Err(RuntimeError::Message(format!(
                "{what} is {got:?}, the wan vae expects {ne:?}"
            )));
        }
        Ok(tensor)
    }

    /// The three-frame-plus window for a cached CausalConv3d and this chunk's
    /// next cache state. `zero_next` is for the first-chunk time_conv slot,
    /// whose "Rep" state carries no real frames.
    fn take_window(
        &mut self,
        x: Tensor,
        w: i64,
        h: i64,
        f: i64,
        c: i64,
        zero_next: bool,
    ) -> Result<Tensor, RuntimeError> {
        let slot = *self
            .slots
            .get(self.slot)
            .ok_or_else(|| RuntimeError::Message("wan vae ran out of cache slots".into()))?;
        if slot.w != w || slot.h != h || slot.c != c {
            return Err(RuntimeError::Message(format!(
                "wan vae cache slot {} is [{}, {}, {}] but the conv input is [{}, {}, {}]",
                self.slot, slot.w, slot.h, slot.c, w, h, c
            )));
        }
        let offset = self.offsets[self.slot];
        self.slot += 1;
        if self.first {
            if f != 1 {
                return Err(RuntimeError::Message(
                    "wan vae first chunk must stay single-frame".into(),
                ));
            }
            let z = nn(self.b.scale(x, 0.0))?;
            let win = nn(self.b.concat(nn(self.b.concat(z, z, 2))?, x, 2))?;
            let next = if zero_next {
                nn(self.b.concat(z, z, 2))?
            } else {
                nn(self.b.concat(z, x, 2))?
            };
            self.next.push(nn(self.b.cont(next))?);
            return Ok(win);
        }
        let cache = self.cache.ok_or_else(|| {
            RuntimeError::Message("wan vae steady graph has no cache input".into())
        })?;
        let slot_t = nn(self.b.view_1d(cache, slot.len(), offset * 4))?;
        let slot_t = nn(self.b.reshape_4d(slot_t, w, h, 2, c))?;
        let win = nn(self.b.concat(slot_t, x, 2))?;
        let next = if f == 1 {
            let last = nn(self.b.cont(nn(self.b.view_4d(
                slot_t,
                w,
                h,
                1,
                c,
                (w * 4) as usize,
                (w * h * 4) as usize,
                (w * h * 2 * 4) as usize,
                (w * h * 4) as usize,
            ))?))?;
            nn(self.b.concat(last, x, 2))?
        } else {
            nn(self.b.cont(nn(self.b.view_4d(
                x,
                w,
                h,
                2,
                c,
                (w * 4) as usize,
                (w * h * 4) as usize,
                (w * h * f * 4) as usize,
                ((f - 2) * w * h * 4) as usize,
            ))?))?
        };
        self.next.push(nn(self.b.cont(next))?);
        Ok(win)
    }

    fn add_bias(&self, y: Tensor, name: &str, oc: i64) -> Result<Tensor, RuntimeError> {
        let bias = nn(self
            .b
            .cont(nn(self.b.reshape_4d(self.wt(name)?, 1, 1, 1, oc))?))?;
        let bias = nn(self.b.repeat(bias, y))?;
        nn(self.b.add(y, bias))
    }

    /// Cached 3x3x3 causal conv: windowed input, symmetric spatial pad 1.
    #[allow(clippy::too_many_arguments)]
    fn cc(
        &mut self,
        x: Tensor,
        prefix: &str,
        w: i64,
        h: i64,
        f: i64,
        ic: i64,
        oc: i64,
    ) -> Result<Tensor, RuntimeError> {
        let win = self.take_window(x, w, h, f, ic, false)?;
        let k = self.wt(&format!("{prefix}.weight"))?;
        let y = nn(self
            .b
            .conv3d_direct(k, win, 1, 1, 1, 1, 1, 0, 1, 1, 1, ic as i32, 1, oc as i32))?;
        let y = nn(self.b.cont(y))?;
        let y = self.shaped(y, [w, h, f, oc], &format!("{prefix}.weight"))?;
        self.add_bias(y, &format!("{prefix}.bias"), oc)
    }

    /// 1x1x1 conv without cache (`conv2`, residual shortcuts).
    #[allow(clippy::too_many_arguments)]
    fn point(
        &self,
        x: Tensor,
        prefix: &str,
        w: i64,
        h: i64,
        f: i64,
        ic: i64,
        oc: i64,
    ) -> Result<Tensor, RuntimeError> {
        let k = self.wt(&format!("{prefix}.weight"))?;
        let y = nn(self
            .b
            .conv3d_direct(k, x, 1, 1, 1, 0, 0, 0, 1, 1, 1, ic as i32, 1, oc as i32))?;
        let y = nn(self.b.cont(y))?;
        let y = self.shaped(y, [w, h, f, oc], &format!("{prefix}.weight"))?;
        self.add_bias(y, &format!("{prefix}.bias"), oc)
    }

    /// `WanVAERMSNorm` on channels (ne3): `rms_norm(eps = 1e-12 / C) * gamma`.
    fn rms(&self, x: Tensor, name: &str, c: i64) -> Result<Tensor, RuntimeError> {
        let gamma = self.wt(name)?;
        let y = nn(self.b.cont(nn(self.b.permute(x, 1, 2, 3, 0))?))?;
        let y = nn(self.b.rms_norm(y, 1e-12 / c as f32))?;
        let gamma = nn(self.b.cont(nn(self.b.reshape_4d(gamma, c, 1, 1, 1))?))?;
        let y = nn(self.b.mul(y, nn(self.b.repeat(gamma, y))?))?;
        nn(self.b.cont(nn(self.b.permute(y, 3, 0, 1, 2))?))
    }

    /// Single-head attention over the `w * h` tokens of one frame.
    fn attention(
        &self,
        x: Tensor,
        prefix: &str,
        c: i64,
        w: i64,
        h: i64,
    ) -> Result<Tensor, RuntimeError> {
        let n = self.rms(x, &format!("{prefix}.norm.gamma"), c)?;
        let t = nn(self.b.cont(nn(self.b.permute(n, 1, 2, 3, 0))?))?; // [c, w, h, 1]
        let t = nn(self.b.reshape_2d(t, c, w * h))?;
        let qkv_w = nn(self
            .b
            .reshape_2d(self.wt(&format!("{prefix}.to_qkv.weight"))?, c, 3 * c))?;
        let qkv = nn(self.b.mul_mat(qkv_w, t))?;
        let bias = nn(self.b.cont(nn(self.b.reshape_2d(
            self.wt(&format!("{prefix}.to_qkv.bias"))?,
            3 * c,
            1,
        ))?))?;
        let qkv = nn(self.b.add(qkv, nn(self.b.repeat(bias, qkv))?))?;
        let qkv = self.shaped(qkv, [3 * c, w * h, 1, 1], "wan vae qkv")?;
        let row = (3 * c * 4) as usize;
        let q = nn(self.b.cont(nn(self.b.view_2d(qkv, c, w * h, row, 0))?))?;
        let k = nn(self
            .b
            .cont(nn(self.b.view_2d(qkv, c, w * h, row, (c * 4) as usize))?))?;
        let v = nn(self.b.cont(nn(self.b.view_2d(
            qkv,
            c,
            w * h,
            row,
            (2 * c * 4) as usize,
        ))?))?;
        let scores = nn(self.b.mul_mat(k, q))?;
        let scores = nn(self.b.scale(scores, (c as f32).sqrt().recip()))?;
        let probs = nn(self.b.soft_max(scores))?;
        let v_t = nn(self.b.cont(nn(self.b.permute(v, 1, 0, 2, 3))?))?; // [wh, c]
        let mixed = nn(self.b.mul_mat(v_t, probs))?; // [c, wh]
        let proj_w = nn(self
            .b
            .reshape_2d(self.wt(&format!("{prefix}.proj.weight"))?, c, c))?;
        let out = nn(self.b.mul_mat(proj_w, mixed))?;
        let bias = nn(self.b.cont(nn(self.b.reshape_2d(
            self.wt(&format!("{prefix}.proj.bias"))?,
            c,
            1,
        ))?))?;
        let out = nn(self.b.add(out, nn(self.b.repeat(bias, out))?))?;
        let out = nn(self.b.reshape_4d(out, c, w, h, 1))?;
        let out = nn(self.b.cont(nn(self.b.permute(out, 3, 0, 1, 2))?))?;
        let out = self.shaped(out, [w, h, 1, c], "wan vae attention out")?;
        nn(self.b.add(x, out))
    }

    #[allow(clippy::too_many_arguments)]
    fn resblock(
        &mut self,
        x: Tensor,
        prefix: &str,
        ic: i64,
        oc: i64,
        w: i64,
        h: i64,
        f: i64,
    ) -> Result<Tensor, RuntimeError> {
        let mut y = self.rms(x, &format!("{prefix}.residual.0.gamma"), ic)?;
        y = nn(self.b.silu(y))?;
        y = self.cc(y, &format!("{prefix}.residual.2"), w, h, f, ic, oc)?;
        y = self.rms(y, &format!("{prefix}.residual.3.gamma"), oc)?;
        y = nn(self.b.silu(y))?;
        y = self.cc(y, &format!("{prefix}.residual.6"), w, h, f, oc, oc)?;
        let skip = if ic != oc {
            self.point(x, &format!("{prefix}.shortcut"), w, h, f, ic, oc)?
        } else {
            x
        };
        nn(self.b.add(skip, y))
    }

    /// time_conv output `[w, h, f, 2c]` -> interleaved frames `[w, h, 2f, c]`;
    /// output frame `2i + s` is channels `[s*c, (s+1)*c)` of input frame `i`.
    fn time_interleave(
        &self,
        tc: Tensor,
        w: i64,
        h: i64,
        f: i64,
        c: i64,
    ) -> Result<Tensor, RuntimeError> {
        let mut frames = Vec::with_capacity((2 * f) as usize);
        for i in 0..f {
            for s in 0..2i64 {
                let offset = ((i + f * c * s) * w * h * 4) as usize;
                let slice = nn(self.b.cont(nn(self.b.view_4d(
                    tc,
                    w,
                    h,
                    1,
                    c,
                    (w * 4) as usize,
                    (w * h * 4) as usize,
                    (w * h * f * 4) as usize,
                    offset,
                ))?))?;
                frames.push(slice);
            }
        }
        let mut it = frames.into_iter();
        let mut out = it.next().expect("at least one frame");
        for frame in it {
            out = nn(self.b.concat(out, frame, 2))?;
        }
        self.shaped(out, [w, h, 2 * f, c], "wan vae time interleave")
    }

    /// `Resample` upsample2d/upsample3d. The temporal half runs first; chunk 0
    /// bypasses time_conv entirely (the "Rep" cache state). The spatial half
    /// is nearest x2 and a 3x3 conv2d over the frame batch.
    #[allow(clippy::too_many_arguments)]
    fn resample(
        &mut self,
        x: Tensor,
        prefix: &str,
        c: i64,
        w: i64,
        h: i64,
        f: i64,
        temporal: bool,
    ) -> Result<(Tensor, i64), RuntimeError> {
        let mut y = x;
        let mut f = f;
        if temporal {
            if self.first {
                // "Rep": no time_conv, and the slot carries no real frames.
                let z = nn(self.b.scale(y, 0.0))?;
                let next = nn(self.b.cont(nn(self.b.concat(z, z, 2))?))?;
                self.slot += 1;
                self.next.push(next);
            } else {
                let win = self.take_window(y, w, h, f, c, true)?;
                let k = self.wt(&format!("{prefix}.time_conv.weight"))?;
                let tc = nn(self.b.conv3d_direct(
                    k,
                    win,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    c as i32,
                    1,
                    (2 * c) as i32,
                ))?;
                let tc = nn(self.b.cont(tc))?;
                let tc =
                    self.shaped(tc, [w, h, f, 2 * c], &format!("{prefix}.time_conv.weight"))?;
                let tc = self.add_bias(tc, &format!("{prefix}.time_conv.bias"), 2 * c)?;
                y = self.time_interleave(tc, w, h, f, c)?;
                f *= 2;
            }
        }
        let y2 = nn(self.b.cont(nn(self.b.permute(y, 0, 1, 3, 2))?))?; // [w, h, c, f]
        let y2 = nn(self.b.upscale_nearest(y2))?;
        let k = self.wt(&format!("{prefix}.resample.1.weight"))?;
        let y2 = nn(self.b.conv2d_direct(k, y2, 1, 1, 1, 1, 1, 1))?;
        let y2 = nn(self.b.cont(y2))?;
        let y2 = self.shaped(
            y2,
            [2 * w, 2 * h, c, f],
            &format!("{prefix}.resample.1.weight"),
        )?;
        let bias = nn(self.b.cont(nn(self.b.reshape_4d(
            self.wt(&format!("{prefix}.resample.1.bias"))?,
            1,
            1,
            c,
            1,
        ))?))?;
        let y2 = nn(self.b.add(y2, nn(self.b.repeat(bias, y2))?))?;
        let y2 = nn(self.b.cont(nn(self.b.permute(y2, 0, 1, 3, 2))?))?; // [2w, 2h, f, c]
        Ok((y2, f))
    }

    /// `DupUp3D` average shortcut. Blocks 0/1 (`ic == oc`, `ft == 2`) are a
    /// nearest (2, 2, 2) duplicate; the first chunk keeps one frame. Block 2
    /// (`ic == 2 * oc`, `ft == 1`) folds channels:
    /// `out[2w+dw, 2h+dh, t, c] = x[w, h, t, 2c + dh]`.
    #[allow(clippy::too_many_arguments)]
    fn dupup(
        &mut self,
        x: Tensor,
        ic: i64,
        oc: i64,
        ft: i64,
        w: i64,
        h: i64,
        f: i64,
    ) -> Result<(Tensor, i64), RuntimeError> {
        if ft == 2 && ic == oc {
            let y = nn(self.b.cont(nn(self.b.permute(x, 0, 1, 3, 2))?))?; // [w, h, c, f]
            let y = nn(self.b.upscale_nearest(y))?;
            let y = nn(self.b.cont(nn(self.b.permute(y, 0, 1, 3, 2))?))?; // [2w, 2h, f, c]
            if self.first {
                return Ok((y, f));
            }
            let mut frames = Vec::with_capacity((2 * f) as usize);
            for i in 0..f {
                for _ in 0..2 {
                    let frame = nn(self.b.cont(nn(self.b.view_4d(
                        y,
                        2 * w,
                        2 * h,
                        1,
                        oc,
                        (2 * w * 4) as usize,
                        (2 * w * 2 * h * 4) as usize,
                        (2 * w * 2 * h * f * 4) as usize,
                        (i * 2 * w * 2 * h * 4) as usize,
                    ))?))?;
                    frames.push(frame);
                }
            }
            let mut it = frames.into_iter();
            let mut out = it.next().expect("at least one frame");
            for frame in it {
                out = nn(self.b.concat(out, frame, 2))?;
            }
            let out = self.shaped(out, [2 * w, 2 * h, 2 * f, oc], "wan vae dupup")?;
            return Ok((out, 2 * f));
        }
        if ft != 1 || ic != 2 * oc {
            return Err(RuntimeError::Message(format!(
                "wan vae dupup factor_t {ft} with channels {ic} -> {oc} is not the wan22 shortcut"
            )));
        }
        // Channel fold. Split channels into (dh, c) with dh fastest, merge dh
        // into the height axis, then duplicate the width with a row gather.
        let y = nn(self.b.reshape_3d(x, w, h * f, ic))?;
        let y = nn(self.b.reshape_4d(y, w, h * f, 2, oc))?;
        let y = nn(self.b.cont(nn(self.b.permute(y, 0, 2, 1, 3))?))?; // [w, dh, hf, c]
        let y = nn(self.b.reshape_3d(y, w, 2 * h * f, oc))?;
        let y = nn(self.b.reshape_4d(y, w, 2 * h, f, oc))?;
        let m = 2 * h * f * oc;
        let y = nn(self.b.reshape_2d(y, w, m))?;
        let y = nn(self.b.cont(nn(self.b.permute(y, 1, 0, 2, 3))?))?; // [m, w]
        let y = nn(self.b.get_rows(y, self.dup_rows))?; // [m, 2w]
        let y = nn(self.b.cont(nn(self.b.permute(y, 1, 0, 2, 3))?))?; // [2w, m]
        let y = nn(self.b.reshape_4d(y, 2 * w, 2 * h, f, oc))?;
        let y = self.shaped(y, [2 * w, 2 * h, f, oc], "wan vae dupup fold")?;
        Ok((y, f))
    }

    /// `out[2w+dw, 2h+dh, t, c0] = x[w, h, t, 4*c0 + 2*dw + dh]`.
    fn unpatchify(
        &self,
        x: Tensor,
        w: i64,
        h: i64,
        f: i64,
        patch: i64,
        pixel_out: i64,
    ) -> Result<Tensor, RuntimeError> {
        if patch != 2 || pixel_out != 12 {
            return Err(RuntimeError::Message(
                "wan vae unpatchify is written for patch 2 and 12 head channels".into(),
            ));
        }
        let y = nn(self.b.reshape_3d(x, w, h * f, 12))?;
        let y = nn(self.b.reshape_4d(y, w, h * f, 2, 6))?; // (dh, dw + 2c0)
        let y = nn(self.b.cont(nn(self.b.permute(y, 0, 2, 1, 3))?))?; // [w, dh, hf, 6]
        let y = nn(self.b.reshape_3d(y, w, 2 * h * f, 6))?;
        let y = nn(self.b.reshape_4d(y, w, 2 * h * f, 2, 3))?; // (dw, c0)
        let y = nn(self.b.cont(nn(self.b.permute(y, 2, 0, 1, 3))?))?; // [hf2, dw, w, c0]
        let y = nn(self.b.reshape_3d(y, 2 * h * f, 2 * w, 3))?;
        let y = nn(self.b.cont(nn(self.b.permute(y, 1, 0, 2, 3))?))?; // [2w, hf2, 3]
        let y = nn(self.b.reshape_4d(y, 2 * w, 2 * h, f, 3))?;
        self.shaped(y, [2 * w, 2 * h, f, 3], "wan vae unpatchify")
    }
}

fn build(
    ctx: &GgmlContext,
    spec: &WanVaeSpec,
    w: i64,
    h: i64,
    first: bool,
) -> Result<Built, WeightError> {
    let mut b = ctx.builder(GRAPH_MEM)?;
    let mut wmap = HashMap::new();
    for param in decoder_params(spec) {
        let tensor = match param.kind {
            Kind::Vector(c) => b.f32_1d(&param.name, c, true),
            Kind::Conv2d { k, ic, oc } => b.f32_4d(&param.name, k, k, ic, oc, true),
            Kind::Conv3d { kw, kh, kd, ic, oc } => b.f32_4d(&param.name, kw, kh, kd, ic * oc, true),
        };
        wmap.insert(param.name.clone(), tensor);
    }
    let frame = b.f32_4d("wan_vae_frame", w, h, 1, spec.z_dim, true);
    let slots = cache_slots(spec, w, h);
    let mut offsets = Vec::with_capacity(slots.len());
    let mut cache_total = 0i64;
    for slot in &slots {
        offsets.push(cache_total as usize);
        cache_total += slot.len();
    }
    let cache = if first {
        None
    } else {
        Some(b.f32_1d("wan_vae_cache", cache_total, true))
    };
    // Up block 2 folds channels at width 4w; the row gather duplicates it to 8w.
    let dup_rows = b.i32_1d("wan_vae_dup_rows", 8 * w, true);
    let mut g = Decode {
        b: &b,
        w: &wmap,
        cache,
        offsets,
        slots,
        slot: 0,
        next: Vec::new(),
        first,
        dup_rows,
    };
    let d = spec.dims;
    let x = g.point(frame, "conv2", w, h, 1, spec.z_dim, spec.z_dim)?;
    let mut x = g.cc(x, "decoder.conv1", w, h, 1, spec.z_dim, d[0])?;
    let (mut cw, mut ch, mut f) = (w, h, 1i64);
    x = g.resblock(x, "decoder.middle.0", d[0], d[0], cw, ch, f)?;
    x = g.attention(x, "decoder.middle.1", d[0], cw, ch)?;
    x = g.resblock(x, "decoder.middle.2", d[0], d[0], cw, ch, f)?;
    for i in 0..4 {
        let x_in = x;
        let mut y = x_in;
        for j in 0..spec.up_res_blocks {
            let ic = if j == 0 { d[i] } else { d[i + 1] };
            y = g.resblock(
                y,
                &format!("decoder.upsamples.{i}.upsamples.{j}"),
                ic,
                d[i + 1],
                cw,
                ch,
                f,
            )?;
        }
        if i < 3 {
            let (yy, ff) = g.resample(
                y,
                &format!("decoder.upsamples.{i}.upsamples.3"),
                d[i + 1],
                cw,
                ch,
                f,
                spec.temporal_up[i],
            )?;
            y = yy;
            let ft = if spec.temporal_up[i] { 2 } else { 1 };
            let (sc, sc_f) = g.dupup(x_in, d[i], d[i + 1], ft, cw, ch, f)?;
            if sc_f != ff {
                return Err(RuntimeError::Message(format!(
                    "wan vae up block {i} main path has {ff} frames but the shortcut has {sc_f}"
                ))
                .into());
            }
            x = nn(b.add(y, sc))?;
            cw *= 2;
            ch *= 2;
            f = ff;
        } else {
            x = y;
        }
    }
    x = g.rms(x, "decoder.head.0.gamma", d[4])?;
    x = nn(b.silu(x))?;
    x = g.cc(x, "decoder.head.2", cw, ch, f, d[4], spec.pixel_out)?;
    let pix = g.unpatchify(x, cw, ch, f, spec.patch, spec.pixel_out)?;
    let pixels = 2 * cw * 2 * ch * f * 3;
    let out = nn(b.cont(pix))?;
    let mut out = nn(b.reshape_2d(out, pixels, 1))?;
    if g.slot != g.slots.len() {
        return Err(RuntimeError::Message(format!(
            "wan vae consumed {} cache slots of {}",
            g.slot,
            g.slots.len()
        ))
        .into());
    }
    for next in &g.next {
        let flat = nn(b.cont(*next))?;
        let len = shape(flat).iter().product::<i64>();
        let flat = nn(b.reshape_2d(flat, len, 1))?;
        out = nn(b.concat(out, flat, 0))?;
    }
    let graph = b.finish(out)?;
    Ok(Built {
        graph,
        frames: f,
        pixels,
        cache: cache_total,
    })
}

fn set_weights(
    graph: &ReusedGraph,
    params: &[(String, Vec<f32>)],
    w: i64,
) -> Result<(), WeightError> {
    for (name, data) in params {
        graph.set_f32(name, data)?;
    }
    let rows: Vec<i32> = (0..8 * w).map(|i| (i / 2) as i32).collect();
    graph.set_i32("wan_vae_dup_rows", &rows)?;
    Ok(())
}

/// Decode `latents` (`[w, h, t, z]`, w fastest) to pixels
/// `[16w, 16h, 1 + 4(t - 1), 3]` in `[-1, 1]`. `mean`/`std` are per channel.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &GgmlContext,
    params: &[(String, Vec<f32>)],
    spec: &WanVaeSpec,
    latents: &[f32],
    w: i64,
    h: i64,
    t: i64,
    mean: &[f32],
    std: &[f32],
) -> Result<Vec<f32>, WeightError> {
    let z = spec.z_dim;
    if t < 1 || w < 1 || h < 1 {
        return Err(WeightError::Shape {
            name: "wan vae decode geometry".into(),
            expected: vec![1],
            got: vec![w, h, t],
        });
    }
    if mean.len() != z as usize || std.len() != z as usize {
        return Err(WeightError::Shape {
            name: "wan vae scale".into(),
            expected: vec![z],
            got: vec![mean.len() as i64, std.len() as i64],
        });
    }
    if latents.len() != (w * h * t * z) as usize {
        return Err(WeightError::Shape {
            name: "wan vae latent".into(),
            expected: vec![w, h, t, z],
            got: vec![latents.len() as i64],
        });
    }
    let mut first_g = build(ctx, spec, w, h, true)?;
    set_weights(&first_g.graph, params, w)?;
    let mut steady_g = if t > 1 {
        let g = build(ctx, spec, w, h, false)?;
        set_weights(&g.graph, params, w)?;
        Some(g)
    } else {
        None
    };
    let mut cache = vec![0f32; first_g.cache as usize];
    let wh = (w * h) as usize;
    let frames_want = spec.decode_frames(t);
    let plane = (16 * w * 16 * h) as usize;
    let mut frames = vec![0f32; plane * frames_want as usize * 3];
    let mut frame_base = 0i64;
    for i in 0..t {
        let mut buf = vec![0f32; wh * z as usize];
        for c in 0..z as usize {
            for px in 0..wh {
                buf[px + wh * c] =
                    latents[px + wh * (i as usize + t as usize * c)] * std[c] + mean[c];
            }
        }
        let built = if i == 0 {
            &mut first_g
        } else {
            steady_g.as_mut().expect("steady graph exists for t > 1")
        };
        built.graph.set_f32("wan_vae_frame", &buf)?;
        if i > 0 {
            built.graph.set_f32("wan_vae_cache", &cache)?;
        }
        let out = built.graph.compute()?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(WeightError::NotBuilt(
                "wan vae decode produced a non-finite sample".into(),
            ));
        }
        let pixels = built.pixels as usize;
        // Scatter the chunk's [16w, 16h, f_i, 3] block into the assembled
        // [16w, 16h, F, 3] volume: channels are outermost.
        for c in 0..3usize {
            for f in 0..built.frames {
                let src = plane * (f as usize + built.frames as usize * c);
                let dst = plane * ((frame_base + f) as usize + frames_want as usize * c);
                frames[dst..dst + plane].copy_from_slice(&out[src..src + plane]);
            }
        }
        frame_base += built.frames;
        cache.copy_from_slice(&out[pixels..]);
    }
    if frame_base != frames_want {
        return Err(WeightError::NotBuilt(format!(
            "wan vae decoded {frame_base} frames, expected {frames_want}"
        )));
    }
    for v in frames.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    Ok(frames)
}

/// The Wan 2.2 VAE decoder bound to a weight file.
pub struct WanVaeDecoder {
    spec: WanVaeSpec,
    params: Vec<(String, Vec<f32>)>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl WanVaeDecoder {
    /// Open the published `Wan2.2_VAE.safetensors` (or a GGUF with the same
    /// tensor names). The topology is inferred from the folded shapes; the
    /// published 48-channel normalization constants are the only ones known,
    /// so any other `z_dim` fails loud.
    pub fn open(path: &Path) -> Result<Self, WeightError> {
        let weights = GgufWeights::open(path)?;
        let spec = WanVaeSpec::infer(&weights)?;
        if spec.z_dim as usize != WAN22_VAE_MEAN.len() {
            return Err(WeightError::NotBuilt(format!(
                "wan vae z_dim {} has no published normalization constants (only {} are known)",
                spec.z_dim,
                WAN22_VAE_MEAN.len()
            )));
        }
        validate(&weights, &spec)?;
        let params = load(&weights, &spec)?;
        Ok(Self {
            spec,
            params,
            mean: WAN22_VAE_MEAN.to_vec(),
            std: WAN22_VAE_STD.to_vec(),
        })
    }

    fn open_with(
        path: &Path,
        spec: WanVaeSpec,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Self, WeightError> {
        let weights = GgufWeights::open(path)?;
        validate(&weights, &spec)?;
        let params = load(&weights, &spec)?;
        Ok(Self {
            spec,
            params,
            mean,
            std,
        })
    }

    pub fn spec(&self) -> &WanVaeSpec {
        &self.spec
    }

    /// Decode `latents` (`[w, h, t, 48]`) to `[16w, 16h, 1 + 4(t-1), 3]`.
    pub fn decode(
        &self,
        ctx: &GgmlContext,
        latents: &[f32],
        w: i64,
        h: i64,
        t: i64,
    ) -> Result<Vec<f32>, WeightError> {
        decode(
            ctx,
            &self.params,
            &self.spec,
            latents,
            w,
            h,
            t,
            &self.mean,
            &self.std,
        )
    }
}

// ---------------------------------------------------------------------------
// smoke writer (session smoke bundle)
// ---------------------------------------------------------------------------

/// Write a deterministic smoke decoder safetensors: the wan22 topology at
/// `z_dim = 48` with 1/128-wide channels, so `WanVaeDecoder::open` accepts it
/// and the published normalization constants apply.
pub fn write_smoke(dir: &Path) -> Result<PathBuf, WeightError> {
    use crate::weights::{write_safetensors, SafeTensor};
    let spec = WanVaeSpec {
        z_dim: 48,
        ..WanVaeSpec::smoke()
    };
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let mut owned: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut seed = 1.0f32;
    for param in decoder_params(&spec) {
        let shape: Vec<u64> = match param.kind {
            Kind::Vector(c) => vec![c as u64],
            Kind::Conv2d { k, ic, oc } => vec![oc as u64, ic as u64, k as u64, k as u64],
            Kind::Conv3d { kw, kh, kd, ic, oc } => {
                vec![oc as u64, ic as u64, kd as u64, kh as u64, kw as u64]
            }
        };
        let n: usize = shape.iter().product::<u64>() as usize;
        let data: Vec<u8> = (0..n)
            .flat_map(|i| ((((i as f32 + seed) * 0.37).sin()) * 0.05).to_le_bytes())
            .collect();
        owned.push((param.name.clone(), shape, data));
        seed += 1.0;
    }
    let tensors: Vec<SafeTensor> = owned
        .iter()
        .map(|(name, shape, data)| SafeTensor {
            name: name.as_str(),
            dtype: "F32",
            shape: shape.as_slice(),
            data: data.as_slice(),
        })
        .collect();
    let path = dir.join("Wan2.2_VAE.safetensors");
    write_safetensors(&path, &tensors)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;
    use crate::weights::{write_safetensors, SafeTensor};
    use std::path::PathBuf;

    fn torch_shape(param: &Param) -> Vec<u64> {
        match param.kind {
            Kind::Vector(c) => vec![c as u64],
            Kind::Conv2d { k, ic, oc } => vec![oc as u64, ic as u64, k as u64, k as u64],
            Kind::Conv3d { kw, kh, kd, ic, oc } => {
                vec![oc as u64, ic as u64, kd as u64, kh as u64, kw as u64]
            }
        }
    }

    /// Deterministic smoke safetensors in the official key layout. The flat
    /// bytes are identical in the PyTorch and ggml layouts.
    fn write_smoke(dir: &PathBuf, spec: &WanVaeSpec) -> PathBuf {
        let mut tensors: Vec<SafeTensor<'static>> = Vec::new();
        let mut seed = 1.0f32;
        for param in decoder_params(spec) {
            let shape = torch_shape(&param);
            let n: usize = shape.iter().product::<u64>() as usize;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i as f32 + seed) * 0.37).sin()) * 0.05)
                .collect();
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let name: &'static str = Box::leak(param.name.clone().into_boxed_str());
            let shape: &'static [u64] = Box::leak(shape.into_boxed_slice());
            let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            tensors.push(SafeTensor {
                name,
                dtype: "F32",
                shape,
                data: bytes,
            });
            seed += 1.0;
        }
        let path = dir.join("Wan2.2_VAE.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        path
    }

    // ----- host reference (full sequence, no cache) -----
    // Layout: [C, T, H, W] planar, channel outermost.

    #[derive(Clone)]
    struct Vol {
        d: Vec<f32>,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
    }

    impl Vol {
        fn new(c: usize, t: usize, h: usize, w: usize) -> Self {
            Self {
                d: vec![0.0; c * t * h * w],
                c,
                t,
                h,
                w,
            }
        }
        fn at(&self, c: usize, t: usize, y: usize, x: usize) -> f32 {
            self.d[((c * self.t + t) * self.h + y) * self.w + x]
        }
        fn set(&mut self, c: usize, t: usize, y: usize, x: usize, v: f32) {
            self.d[((c * self.t + t) * self.h + y) * self.w + x] = v;
        }
    }

    fn from_ggml(flat: &[f32], w: usize, h: usize, t: usize, c: usize) -> Vol {
        let mut v = Vol::new(c, t, h, w);
        for cc in 0..c {
            for tt in 0..t {
                for y in 0..h {
                    for x in 0..w {
                        let val = flat[x + w * (y + h * (tt + t * cc))];
                        v.set(cc, tt, y, x, val);
                    }
                }
            }
        }
        v
    }

    fn to_ggml(v: &Vol) -> Vec<f32> {
        let mut out = vec![0f32; v.c * v.t * v.h * v.w];
        for cc in 0..v.c {
            for tt in 0..v.t {
                for y in 0..v.h {
                    for x in 0..v.w {
                        out[x + v.w * (y + v.h * (tt + v.t * cc))] = v.at(cc, tt, y, x);
                    }
                }
            }
        }
        out
    }

    fn h_conv3d(
        x: &Vol,
        wt: &[f32],
        bias: &[f32],
        kt: usize,
        kh: usize,
        kw: usize,
        oc: usize,
    ) -> Vol {
        let (pt, ph, pw) = (kt - 1, kh / 2, kw / 2);
        let mut out = Vol::new(oc, x.t, x.h, x.w);
        for o in 0..oc {
            for f in 0..x.t {
                for yy in 0..x.h {
                    for xx in 0..x.w {
                        let mut acc = bias[o];
                        for i in 0..x.c {
                            for zd in 0..kt {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let sf = f as i64 + zd as i64 - pt as i64;
                                        let sy = yy as i64 + ky as i64 - ph as i64;
                                        let sx = xx as i64 + kx as i64 - pw as i64;
                                        if sf < 0
                                            || sy < 0
                                            || sx < 0
                                            || sy >= x.h as i64
                                            || sx >= x.w as i64
                                        {
                                            continue;
                                        }
                                        let xv = x.at(i, sf as usize, sy as usize, sx as usize);
                                        let wv =
                                            wt[(((o * x.c + i) * kt + zd) * kh + ky) * kw + kx];
                                        acc += xv * wv;
                                    }
                                }
                            }
                        }
                        out.set(o, f, yy, xx, acc);
                    }
                }
            }
        }
        out
    }

    fn h_conv2d(x: &Vol, wt: &[f32], bias: &[f32], k: usize, oc: usize) -> Vol {
        let pad = k / 2;
        let mut out = Vol::new(oc, x.t, x.h, x.w);
        for o in 0..oc {
            for f in 0..x.t {
                for yy in 0..x.h {
                    for xx in 0..x.w {
                        let mut acc = bias[o];
                        for i in 0..x.c {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let sy = yy as i64 + ky as i64 - pad as i64;
                                    let sx = xx as i64 + kx as i64 - pad as i64;
                                    if sy < 0 || sx < 0 || sy >= x.h as i64 || sx >= x.w as i64 {
                                        continue;
                                    }
                                    let xv = x.at(i, f, sy as usize, sx as usize);
                                    let wv = wt[((o * x.c + i) * k + ky) * k + kx];
                                    acc += xv * wv;
                                }
                            }
                        }
                        out.set(o, f, yy, xx, acc);
                    }
                }
            }
        }
        out
    }

    fn h_rms(x: &mut Vol, gamma: &[f32]) {
        let scale = (x.c as f32).sqrt();
        for f in 0..x.t {
            for y in 0..x.h {
                for xx in 0..x.w {
                    let mut sum = 0f32;
                    for c in 0..x.c {
                        let v = x.at(c, f, y, xx);
                        sum += v * v;
                    }
                    let inv = 1.0 / (sum + 1e-12).sqrt();
                    for c in 0..x.c {
                        let v = x.at(c, f, y, xx);
                        x.set(c, f, y, xx, v * inv * scale * gamma[c]);
                    }
                }
            }
        }
    }

    fn h_silu(x: &mut Vol) {
        for v in x.d.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    fn at<'a>(map: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a Vec<f32> {
        map.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    fn h_resblock(
        x: &Vol,
        prefix: &str,
        ic: usize,
        oc: usize,
        map: &HashMap<String, Vec<f32>>,
    ) -> Vol {
        let mut y = x.clone();
        h_rms(&mut y, at(map, &format!("{prefix}.residual.0.gamma")));
        h_silu(&mut y);
        let mut y = h_conv3d(
            &y,
            at(map, &format!("{prefix}.residual.2.weight")),
            at(map, &format!("{prefix}.residual.2.bias")),
            3,
            3,
            3,
            oc,
        );
        h_rms(&mut y, at(map, &format!("{prefix}.residual.3.gamma")));
        h_silu(&mut y);
        let y = h_conv3d(
            &y,
            at(map, &format!("{prefix}.residual.6.weight")),
            at(map, &format!("{prefix}.residual.6.bias")),
            3,
            3,
            3,
            oc,
        );
        let skip = if ic != oc {
            h_conv3d(
                x,
                at(map, &format!("{prefix}.shortcut.weight")),
                at(map, &format!("{prefix}.shortcut.bias")),
                1,
                1,
                1,
                oc,
            )
        } else {
            x.clone()
        };
        let mut out = y;
        for i in 0..out.d.len() {
            out.d[i] += skip.d[i];
        }
        out
    }

    fn h_attention(x: &Vol, prefix: &str, map: &HashMap<String, Vec<f32>>) -> Vol {
        let c = x.c;
        let tokens = x.h * x.w;
        let gamma = at(map, &format!("{prefix}.norm.gamma"));
        let qkv_w = at(map, &format!("{prefix}.to_qkv.weight"));
        let qkv_b = at(map, &format!("{prefix}.to_qkv.bias"));
        let proj_w = at(map, &format!("{prefix}.proj.weight"));
        let proj_b = at(map, &format!("{prefix}.proj.bias"));
        let mut out = x.clone();
        for f in 0..x.t {
            // normed tokens [c][px]
            let mut normed = vec![0f32; c * tokens];
            for y in 0..x.h {
                for xx in 0..x.w {
                    let px = y * x.w + xx;
                    let mut sum = 0f32;
                    for cc in 0..c {
                        let v = x.at(cc, f, y, xx);
                        sum += v * v;
                    }
                    let inv = (x.c as f32).sqrt() / (sum + 1e-12).sqrt();
                    for cc in 0..c {
                        normed[cc * tokens + px] = x.at(cc, f, y, xx) * inv * gamma[cc];
                    }
                }
            }
            // qkv [3c][px]
            let mut qkv = vec![0f32; 3 * c * tokens];
            for o in 0..3 * c {
                for px in 0..tokens {
                    let mut acc = qkv_b[o];
                    for i in 0..c {
                        acc += qkv_w[o * c + i] * normed[i * tokens + px];
                    }
                    qkv[o * tokens + px] = acc;
                }
            }
            let q = &qkv[..c * tokens];
            let k = &qkv[c * tokens..2 * c * tokens];
            let v = &qkv[2 * c * tokens..];
            let scale = (c as f32).sqrt().recip();
            let mut mixed = vec![0f32; c * tokens];
            for tq in 0..tokens {
                let mut scores = vec![0f32; tokens];
                let mut max = f32::NEG_INFINITY;
                for tk in 0..tokens {
                    let mut s = 0f32;
                    for cc in 0..c {
                        s += q[cc * tokens + tq] * k[cc * tokens + tk];
                    }
                    scores[tk] = s * scale;
                    max = max.max(scores[tk]);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for cc in 0..c {
                    let mut acc = 0f32;
                    for tk in 0..tokens {
                        acc += v[cc * tokens + tk] * scores[tk] / sum;
                    }
                    mixed[cc * tokens + tq] = acc;
                }
            }
            for o in 0..c {
                for px in 0..tokens {
                    let mut acc = proj_b[o];
                    for i in 0..c {
                        acc += proj_w[o * c + i] * mixed[i * tokens + px];
                    }
                    let y = px / x.w;
                    let xx = px % x.w;
                    out.set(o, f, y, xx, x.at(o, f, y, xx) + acc);
                }
            }
        }
        out
    }

    fn h_resample(x: &Vol, prefix: &str, temporal: bool, map: &HashMap<String, Vec<f32>>) -> Vol {
        let c = x.c;
        let y = if temporal {
            // frame 0 bypasses time_conv; frames 1.. split into two each
            let rest = Vol {
                d: {
                    let mut d = Vec::with_capacity(c * (x.t - 1) * x.h * x.w);
                    for cc in 0..c {
                        for f in 1..x.t {
                            for p in 0..x.h * x.w {
                                d.push(x.at(cc, f, p / x.w, p % x.w));
                            }
                        }
                    }
                    d
                },
                c,
                t: x.t - 1,
                h: x.h,
                w: x.w,
            };
            let tc = h_conv3d(
                &rest,
                at(map, &format!("{prefix}.time_conv.weight")),
                at(map, &format!("{prefix}.time_conv.bias")),
                3,
                1,
                1,
                2 * c,
            );
            let mut out = Vol::new(c, 1 + 2 * (x.t - 1), x.h, x.w);
            for cc in 0..c {
                for p in 0..x.h * x.w {
                    out.set(cc, 0, p / x.w, p % x.w, x.at(cc, 0, p / x.w, p % x.w));
                }
            }
            for i in 0..x.t - 1 {
                for s in 0..2 {
                    for cc in 0..c {
                        for p in 0..x.h * x.w {
                            out.set(
                                cc,
                                1 + 2 * i + s,
                                p / x.w,
                                p % x.w,
                                tc.at(s * c + cc, i, p / x.w, p % x.w),
                            );
                        }
                    }
                }
            }
            out
        } else {
            x.clone()
        };
        // nearest x2 then 3x3 conv2d
        let mut up = Vol::new(c, y.t, y.h * 2, y.w * 2);
        for cc in 0..c {
            for f in 0..y.t {
                for yy in 0..up.h {
                    for xx in 0..up.w {
                        up.set(cc, f, yy, xx, y.at(cc, f, yy / 2, xx / 2));
                    }
                }
            }
        }
        h_conv2d(
            &up,
            at(map, &format!("{prefix}.resample.1.weight")),
            at(map, &format!("{prefix}.resample.1.bias")),
            3,
            c,
        )
    }

    fn h_dupup(x: &Vol, ft: usize, fs: usize, ic: usize, oc: usize) -> Vol {
        let repeats = oc * ft * fs * fs / ic;
        let mut out = Vol::new(oc, x.t * ft, x.h * fs, x.w * fs);
        for c_old in 0..ic {
            for r in 0..repeats {
                let c_rep = c_old * repeats + r;
                let c_new = c_rep / (ft * fs * fs);
                let inner = c_rep % (ft * fs * fs);
                let ft_i = inner / (fs * fs);
                let fs_a = (inner / fs) % fs;
                let fs_b = inner % fs;
                for f in 0..x.t {
                    for y in 0..x.h {
                        for xx in 0..x.w {
                            out.set(
                                c_new,
                                f * ft + ft_i,
                                y * fs + fs_a,
                                xx * fs + fs_b,
                                x.at(c_old, f, y, xx),
                            );
                        }
                    }
                }
            }
        }
        // first chunk drops the first ft - 1 duplicated frames
        if ft > 1 {
            let keep = out.t - (ft - 1);
            let mut trimmed = Vol::new(oc, keep, out.h, out.w);
            for cc in 0..oc {
                for f in 0..keep {
                    for p in 0..out.h * out.w {
                        trimmed.set(
                            cc,
                            f,
                            p / out.w,
                            p % out.w,
                            out.at(cc, f + ft - 1, p / out.w, p % out.w),
                        );
                    }
                }
            }
            return trimmed;
        }
        out
    }

    fn h_unpatchify(x: &Vol, patch: usize) -> Vol {
        let q = patch;
        let c0 = x.c / (q * q);
        let mut out = Vol::new(c0, x.t, x.h * q, x.w * q);
        for cc in 0..c0 {
            for f in 0..x.t {
                for y in 0..x.h {
                    for xx in 0..x.w {
                        for dh in 0..q {
                            for dw in 0..q {
                                // c12 = c0 * q*q + dw * q + dh (diffusers 5D layout)
                                let c12 = cc * q * q + dw * q + dh;
                                out.set(cc, f, y * q + dh, xx * q + dw, x.at(c12, f, y, xx));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn h_decode(
        latent: &Vol,
        spec: &WanVaeSpec,
        mean: &[f32],
        std: &[f32],
        map: &HashMap<String, Vec<f32>>,
    ) -> Vol {
        let d: Vec<usize> = spec.dims.iter().map(|v| *v as usize).collect();
        let mut z = latent.clone();
        for c in 0..z.c {
            for i in 0..z.t * z.h * z.w {
                z.d[i + z.t * z.h * z.w * c] = z.d[i + z.t * z.h * z.w * c] * std[c] + mean[c];
            }
        }
        let mut x = h_conv3d(
            &z,
            at(map, "conv2.weight"),
            at(map, "conv2.bias"),
            1,
            1,
            1,
            z.c,
        );
        x = h_conv3d(
            &x,
            at(map, "decoder.conv1.weight"),
            at(map, "decoder.conv1.bias"),
            3,
            3,
            3,
            d[0],
        );
        x = h_resblock(&x, "decoder.middle.0", d[0], d[0], map);
        x = h_attention(&x, "decoder.middle.1", map);
        x = h_resblock(&x, "decoder.middle.2", d[0], d[0], map);
        for i in 0..4 {
            let x_in = x.clone();
            let mut y = x_in.clone();
            for j in 0..spec.up_res_blocks {
                let ic = if j == 0 { d[i] } else { d[i + 1] };
                y = h_resblock(
                    &y,
                    &format!("decoder.upsamples.{i}.upsamples.{j}"),
                    ic,
                    d[i + 1],
                    map,
                );
            }
            if i < 3 {
                y = h_resample(
                    &y,
                    &format!("decoder.upsamples.{i}.upsamples.3"),
                    spec.temporal_up[i],
                    map,
                );
                let ft = if spec.temporal_up[i] { 2 } else { 1 };
                let sc = h_dupup(&x_in, ft, 2, d[i], d[i + 1]);
                assert_eq!(y.t, sc.t, "up block {i} frame counts");
                for k in 0..y.d.len() {
                    y.d[k] += sc.d[k];
                }
            }
            x = y;
        }
        h_rms(&mut x, at(map, "decoder.head.0.gamma"));
        h_silu(&mut x);
        let x = h_conv3d(
            &x,
            at(map, "decoder.head.2.weight"),
            at(map, "decoder.head.2.bias"),
            3,
            3,
            3,
            spec.pixel_out as usize,
        );
        let mut out = h_unpatchify(&x, spec.patch as usize);
        for v in out.d.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what} length");
        let mut worst = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((x - y).abs());
        }
        assert!(worst < tol, "{what} max abs diff {worst}");
    }

    fn smoke_decoder(
        dir: &PathBuf,
    ) -> (WanVaeDecoder, HashMap<String, Vec<f32>>, Vec<f32>, Vec<f32>) {
        let spec = WanVaeSpec::smoke();
        let path = write_smoke(dir, &spec);
        let mean: Vec<f32> = (0..spec.z_dim).map(|c| 0.05 * (c as f32) - 0.1).collect();
        let std: Vec<f32> = (0..spec.z_dim).map(|c| 0.9 + 0.07 * (c as f32)).collect();
        let decoder =
            WanVaeDecoder::open_with(&path, spec.clone(), mean.clone(), std.clone()).unwrap();
        let map: HashMap<String, Vec<f32>> = decoder.params.iter().cloned().collect();
        (decoder, map, mean, std)
    }

    #[test]
    fn smoke_decode_matches_the_full_sequence_host_reference() {
        let dir = std::env::temp_dir().join(format!("danmo-wan-vae-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (decoder, map, mean, std) = smoke_decoder(&dir);
        let spec = WanVaeSpec::smoke();
        // T=3 exercises the steady chunk on a cache of real frames (chunk 2
        // sees [x1, x2], not the zero-padded [0, x1] chunk 1 gets).
        let (w, h, t) = (3usize, 3usize, 3usize);
        let latents: Vec<f32> = (0..w * h * t * spec.z_dim as usize)
            .map(|i| (i as f32 * 0.23).cos() * 0.8)
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = decoder
            .decode(&ctx, &latents, w as i64, h as i64, t as i64)
            .unwrap();
        let frames = spec.decode_frames(t as i64) as usize;
        assert_eq!(frames, 9);
        assert_eq!(got.len(), 16 * w * 16 * h * frames * 3);

        let latent = from_ggml(&latents, w, h, t, spec.z_dim as usize);
        let want = h_decode(&latent, &spec, &mean, &std, &map);
        assert_eq!((want.t, want.h, want.w), (frames, 16 * h, 16 * w));
        let want_flat = to_ggml(&want);
        close(&got, &want_flat, 1e-4, "wan vae decode");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_latent_frame_decodes_to_one_frame() {
        let dir = std::env::temp_dir().join(format!("danmo-wan-vae-one-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (decoder, map, mean, std) = smoke_decoder(&dir);
        let spec = WanVaeSpec::smoke();
        let (w, h, t) = (3usize, 3usize, 1usize);
        let latents: Vec<f32> = (0..w * h * t * spec.z_dim as usize)
            .map(|i| (i as f32 * 0.31).sin())
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = decoder
            .decode(&ctx, &latents, w as i64, h as i64, t as i64)
            .unwrap();
        assert_eq!(got.len(), 16 * w * 16 * h * 3);
        let latent = from_ggml(&latents, w, h, t, spec.z_dim as usize);
        let want = h_decode(&latent, &spec, &mean, &std, &map);
        assert_eq!(want.t, 1);
        close(&got, &to_ggml(&want), 1e-4, "wan vae single frame");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_is_causal_along_the_frame_axis() {
        let dir = std::env::temp_dir().join(format!("danmo-wan-vae-causal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (decoder, _map, _mean, _std) = smoke_decoder(&dir);
        let spec = WanVaeSpec::smoke();
        let (w, h, t) = (3usize, 3usize, 3usize);
        let latents: Vec<f32> = (0..w * h * t * spec.z_dim as usize)
            .map(|i| (i as f32 * 0.19).sin() * 0.7)
            .collect();
        // Perturb the whole last latent frame; output frames 0..=4 must not move.
        let mut perturbed = latents.clone();
        let wh = w * h;
        let tt = t;
        for c in 0..spec.z_dim as usize {
            for px in 0..wh {
                perturbed[px + wh * (2 + tt * c)] += 0.5;
            }
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let base = decoder
            .decode(&ctx, &latents, w as i64, h as i64, t as i64)
            .unwrap();
        let changed = decoder
            .decode(&ctx, &perturbed, w as i64, h as i64, t as i64)
            .unwrap();
        let frames = spec.decode_frames(t as i64) as usize;
        assert_eq!(frames, 9);
        let plane = 16 * w * 16 * h;
        let frame_diff = |vals_a: &[f32], vals_b: &[f32], f: usize| -> f32 {
            let mut worst = 0f32;
            for c in 0..3 {
                let base = plane * (f + frames * c);
                for p in 0..plane {
                    worst = worst.max((vals_a[base + p] - vals_b[base + p]).abs());
                }
            }
            worst
        };
        for f in 0..5 {
            assert!(
                frame_diff(&base, &changed, f) < 1e-5,
                "output frame {f} must not depend on latent frame 2 (diff {})",
                frame_diff(&base, &changed, f)
            );
        }
        // The tail frames come from latent frame 2; clipping can saturate one
        // frame, so any of them moving proves causality holds directionally.
        let tail = (5..9)
            .map(|f| frame_diff(&base, &changed, f))
            .fold(0f32, f32::max);
        assert!(
            tail > 1e-4,
            "output frames 5..9 must see latent frame 2 (max diff {tail})"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_fails_loud_on_a_missing_tensor() {
        let dir = std::env::temp_dir().join(format!("danmo-wan-vae-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spec = WanVaeSpec::smoke();
        let path = write_smoke(&dir, &spec);
        // Truncate: write a file with only conv2.
        let tensors = [SafeTensor {
            name: "conv2.weight",
            dtype: "F32",
            shape: &[4, 4, 1, 1, 1],
            data: &[0u8; 4 * 4 * 4],
        }];
        let bad = dir.join("bad.safetensors");
        write_safetensors(&bad, &tensors).unwrap();
        let err = WanVaeDecoder::open_with(&bad, spec, vec![0.0; 4], vec![1.0; 4])
            .err()
            .expect("open must fail on a missing tensor");
        let msg = err.to_string();
        assert!(msg.contains("conv2.bias"), "{msg}");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = path;
    }
}
