use super::edit::VisionGrid;
use crate::runtime::{flash_attend, rope_nd, GraphBuilder, RuntimeError, Tensor, TYPE_F32};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;

const GRAPH_MEM: usize = 64 * 1024 * 1024;
const FULL_ATTENTION: [i64; 4] = [7, 15, 23, 31];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionConfig {
    pub embed: i64,
    pub depth: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub mlp_hidden: i64,
    pub hidden: i64,
}

impl VisionConfig {
    pub fn published() -> Self {
        Self {
            embed: 1280,
            depth: 32,
            heads: 16,
            head_dim: 80,
            mlp_hidden: 3420,
            hidden: 3584,
        }
    }

    pub fn smoke() -> Self {
        Self {
            embed: 4,
            depth: 1,
            heads: 1,
            head_dim: 4,
            mlp_hidden: 10,
            hidden: 8,
        }
    }
}

pub struct VisionForward {
    graph: crate::runtime::ReusedGraph,
    config: VisionConfig,
    seq: i64,
    groups: i64,
    window: Vec<i64>,
    reverse: Vec<i32>,
    cu_windows: Vec<i64>,
}

impl VisionForward {
    pub fn build(
        ctx: &crate::runtime::GgmlContext,
        config: VisionConfig,
        grid: VisionGrid,
    ) -> Result<Self, RuntimeError> {
        if config.embed < 1
            || config.heads < 1
            || config.head_dim < 2
            || config.head_dim % 4 != 0
            || config.heads * config.head_dim != config.embed
            || grid.t != 1
            || grid.h % 2 != 0
            || grid.w % 2 != 0
        {
            return Err(RuntimeError::Message(format!(
                "qwen vision config or grid is invalid: {config:?} {grid:?}"
            )));
        }
        let seq = (grid.t * grid.h * grid.w) as i64;
        let groups = seq / 4;
        let (window, reverse, cu_windows) = window_plan(grid);
        let mut builder = ctx.builder(GRAPH_MEM)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in weight_layout(config) {
            let tensor = builder.tensor_2d(&name, TYPE_F32, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        let pixel = nn(builder.f32_2d("pixel", 3 * 2 * 14 * 14, seq, true))?;
        let rope_h = nn(builder.i32_1d("rope_h", seq, true))?;
        let rope_w = nn(builder.i32_1d("rope_w", seq, true))?;
        let mask = nn(builder.f32_2d("attn_mask", seq, seq, true))?;
        let reverse_tensor = nn(builder.i32_1d("reverse", groups, true))?;
        let output = vision_graph(
            &builder,
            &weights,
            config,
            pixel,
            rope_h,
            rope_w,
            mask,
            reverse_tensor,
            seq,
            groups,
        )?;
        let graph = builder.finish(output)?;
        Ok(Self {
            graph,
            config,
            seq,
            groups,
            window,
            reverse,
            cu_windows,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    pub fn compute(
        &mut self,
        values: &[f32],
        positions_h: &[i32],
        positions_w: &[i32],
    ) -> Result<Vec<f32>, RuntimeError> {
        let patch_dim = 3 * 2 * 14 * 14;
        if values.len() != self.seq as usize * patch_dim
            || positions_h.len() != self.seq as usize
            || positions_w.len() != self.seq as usize
        {
            return Err(RuntimeError::Message(format!(
                "qwen vision input lengths pixel {} h {} w {}, expected seq {} by {patch_dim}",
                values.len(),
                positions_h.len(),
                positions_w.len(),
                self.seq
            )));
        }
        let window = expand_window(&self.window);
        let values = reorder(values, patch_dim, &window);
        let positions_h = reorder_indices(positions_h, &window);
        let positions_w = reorder_indices(positions_w, &window);
        self.graph.set_f32("pixel", &values)?;
        self.graph.set_i32("rope_h", &positions_h)?;
        self.graph.set_i32("rope_w", &positions_w)?;
        self.graph.set_i32("reverse", &self.reverse)?;
        for layer in 0..self.config.depth {
            let mask = if FULL_ATTENTION.contains(&layer) {
                vec![0.0; (self.seq * self.seq) as usize]
            } else {
                window_mask(self.seq, &self.cu_windows)
            };
            self.graph.set_f32("attn_mask", &mask)?;
            self.graph.compute_device()?;
        }
        self.graph.read_output()
    }
}

pub fn position_ids(grid: VisionGrid) -> (Vec<i32>, Vec<i32>) {
    let mut h = Vec::with_capacity(grid.t * grid.h * grid.w);
    let mut w = Vec::with_capacity(h.capacity());
    for _ in 0..grid.t {
        for gh in 0..grid.h / 2 {
            for gw in 0..grid.w / 2 {
                for mh in 0..2 {
                    for mw in 0..2 {
                        h.push((gh * 2 + mh) as i32);
                        w.push((gw * 2 + mw) as i32);
                    }
                }
            }
        }
    }
    (h, w)
}

pub fn encode(
    ctx: &crate::runtime::GgmlContext,
    weights: &GgufWeights,
    config: VisionConfig,
    input: &super::edit::VisionInput,
) -> Result<Vec<f32>, WeightError> {
    let mut forward = VisionForward::build(ctx, config, input.grid)?;
    for (name, ne0, ne1) in weight_layout(config) {
        let data = weights.f32_widened(&name, &[ne0, ne1])?;
        forward.load_f32(&name, &data)?;
    }
    let (h, w) = position_ids(input.grid);
    Ok(forward.compute(&input.values, &h, &w)?)
}

/// True when every vision tensor is present with the expected shape.
pub fn layout_matches(weights: &GgufWeights, config: VisionConfig) -> bool {
    for (name, ne0, ne1) in weight_layout(config) {
        let Ok(got) = weights.shape(&name) else {
            return false;
        };
        if got[0] != ne0 || got[1] != ne1 {
            return false;
        }
    }
    true
}

pub fn config_from_weights(weights: &GgufWeights) -> Result<VisionConfig, WeightError> {
    for candidate in [VisionConfig::published(), VisionConfig::smoke()] {
        if layout_matches(weights, candidate) {
            return Ok(candidate);
        }
    }
    Err(WeightError::NotBuilt(
        "Qwen2.5-VL vision tower layout is not the published or smoke tower".into(),
    ))
}

fn vision_graph(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: VisionConfig,
    pixel: Tensor,
    rope_h: Tensor,
    rope_w: Tensor,
    mask: Tensor,
    reverse: Tensor,
    seq: i64,
    groups: i64,
) -> Result<Tensor, RuntimeError> {
    let mut hidden = nn(b.project(wt(weights, "visual.patch_embedding.proj.weight")?, pixel))?;
    for layer in 0..config.depth {
        hidden = block(b, weights, config, layer, hidden, rope_h, rope_w, mask, seq)?;
    }
    hidden = rms(
        b,
        hidden,
        wt(weights, "visual.merger.ln_q.weight")?,
        config.embed,
    )?;
    hidden = nn(b.reshape_2d(hidden, config.embed * 4, groups))?;
    hidden = linear(
        b,
        hidden,
        wt(weights, "visual.merger.mlp.0.weight")?,
        wt(weights, "visual.merger.mlp.0.bias")?,
    )?;
    hidden = nn(b.gelu(hidden))?;
    hidden = linear(
        b,
        hidden,
        wt(weights, "visual.merger.mlp.1.weight")?,
        wt(weights, "visual.merger.mlp.1.bias")?,
    )?;
    nn(b.get_rows(hidden, reverse))
}

fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: VisionConfig,
    layer: i64,
    hidden: Tensor,
    rope_h: Tensor,
    rope_w: Tensor,
    mask: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let prefix = format!("visual.blocks.{layer}");
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{prefix}.norm1.weight"))?,
        config.embed,
    )?;
    let qkv = linear(
        b,
        normed,
        wt(weights, &format!("{prefix}.attn.qkv.weight"))?,
        wt(weights, &format!("{prefix}.attn.qkv.bias"))?,
    )?;
    let elem = std::mem::size_of::<f32>();
    let stride = (config.embed * 3) as usize * elem;
    let q = nn(b.cont(b.view_2d(qkv, config.embed, seq, stride, 0)))?;
    let k = nn(b.cont(b.view_2d(qkv, config.embed, seq, stride, config.embed as usize * elem)))?;
    let v = nn(b.cont(b.view_2d(
        qkv,
        config.embed,
        seq,
        stride,
        (config.embed * 2) as usize * elem,
    )))?;
    let q = nn(b.reshape_3d(q, config.head_dim, config.heads, seq))?;
    let k = nn(b.reshape_3d(k, config.head_dim, config.heads, seq))?;
    let v = nn(b.reshape_3d(v, config.head_dim, config.heads, seq))?;
    let dims = [config.head_dim / 2, config.head_dim / 2];
    let q = rope_nd(b, q, &[rope_h, rope_w], &dims, 10_000.0)?;
    let k = rope_nd(b, k, &[rope_h, rope_w], &dims, 10_000.0)?;
    let attended = flash_attend(b, q, k, v, mask)?;
    let attended = nn(b.reshape_2d(attended, config.embed, seq))?;
    let attended = linear(
        b,
        attended,
        wt(weights, &format!("{prefix}.attn.proj.weight"))?,
        wt(weights, &format!("{prefix}.attn.proj.bias"))?,
    )?;
    let hidden = nn(b.add(hidden, attended))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{prefix}.norm2.weight"))?,
        config.embed,
    )?;
    let gate = linear(
        b,
        normed,
        wt(weights, &format!("{prefix}.mlp.gate_proj.weight"))?,
        wt(weights, &format!("{prefix}.mlp.gate_proj.bias"))?,
    )?;
    let up = linear(
        b,
        normed,
        wt(weights, &format!("{prefix}.mlp.up_proj.weight"))?,
        wt(weights, &format!("{prefix}.mlp.up_proj.bias"))?,
    )?;
    let mlp = linear(
        b,
        nn(b.mul(nn(b.silu(gate))?, up))?,
        wt(weights, &format!("{prefix}.mlp.down_proj.weight"))?,
        wt(weights, &format!("{prefix}.mlp.down_proj.bias"))?,
    )?;
    nn(b.add(hidden, mlp))
}

fn linear(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Result<Tensor, RuntimeError> {
    let projected = nn(b.project(weight, x))?;
    nn(b.add(projected, nn(b.repeat(bias, projected))?))
}

fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor, channels: i64) -> Result<Tensor, RuntimeError> {
    let normed = nn(b.rms_norm(x, 1e-6))?;
    let weight = nn(b.reshape_2d(weight, channels, 1))?;
    nn(b.mul(normed, nn(b.repeat(weight, normed))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

pub(crate) fn weight_layout(config: VisionConfig) -> Vec<(String, i64, i64)> {
    let mut out = vec![(
        "visual.patch_embedding.proj.weight".into(),
        3 * 2 * 14 * 14,
        config.embed,
    )];
    for layer in 0..config.depth {
        let prefix = format!("visual.blocks.{layer}");
        for (name, ne0, ne1) in [
            ("norm1.weight", config.embed, 1),
            ("norm2.weight", config.embed, 1),
            ("attn.qkv.weight", config.embed, config.embed * 3),
            ("attn.qkv.bias", config.embed * 3, 1),
            ("attn.proj.weight", config.embed, config.embed),
            ("attn.proj.bias", config.embed, 1),
            ("mlp.gate_proj.weight", config.embed, config.mlp_hidden),
            ("mlp.gate_proj.bias", config.mlp_hidden, 1),
            ("mlp.up_proj.weight", config.embed, config.mlp_hidden),
            ("mlp.up_proj.bias", config.mlp_hidden, 1),
            ("mlp.down_proj.weight", config.mlp_hidden, config.embed),
            ("mlp.down_proj.bias", config.embed, 1),
        ] {
            out.push((format!("{prefix}.{name}"), ne0, ne1));
        }
    }
    out.extend([
        ("visual.merger.ln_q.weight".into(), config.embed, 1),
        (
            "visual.merger.mlp.0.weight".into(),
            config.embed * 4,
            config.embed * 4,
        ),
        ("visual.merger.mlp.0.bias".into(), config.embed * 4, 1),
        (
            "visual.merger.mlp.1.weight".into(),
            config.embed * 4,
            config.hidden,
        ),
        ("visual.merger.mlp.1.bias".into(), config.hidden, 1),
    ]);
    out
}

fn window_plan(grid: VisionGrid) -> (Vec<i64>, Vec<i32>, Vec<i64>) {
    let llm_h = grid.h / 2;
    let llm_w = grid.w / 2;
    let window_size = 4usize;
    let mut window = Vec::with_capacity(grid.t * grid.h * grid.w);
    let mut cu = vec![0i64];
    for _ in 0..grid.t {
        for wy in 0..llm_h.div_ceil(window_size) {
            for wx in 0..llm_w.div_ceil(window_size) {
                let mut count = 0i64;
                for y in wy * window_size..(wy * window_size + window_size).min(llm_h) {
                    for x in wx * window_size..(wx * window_size + window_size).min(llm_w) {
                        window.push((y * llm_w + x) as i64);
                        count += 1;
                    }
                }
                let next = cu.last().copied().unwrap_or(0) + count * 4;
                cu.push(next);
            }
        }
    }
    let mut reverse: Vec<i32> = (0..window.len() as i32).collect();
    reverse.sort_by_key(|index| window[*index as usize]);
    (window, reverse, cu)
}

fn window_mask(seq: i64, cu: &[i64]) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; (seq * seq) as usize];
    for start in 0..cu.len() - 1 {
        let begin = cu[start];
        let end = cu[start + 1];
        for query in begin..end {
            for key in begin..end {
                mask[(key + query * seq) as usize] = 0.0;
            }
        }
    }
    mask
}

fn expand_window(indices: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(indices.len() * 4);
    for index in indices {
        for offset in 0..4 {
            out.push(index * 4 + offset);
        }
    }
    out
}

fn reorder(values: &[f32], dim: usize, indices: &[i64]) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for (target, source) in indices.iter().enumerate() {
        let source = *source as usize;
        out[target * dim..(target + 1) * dim]
            .copy_from_slice(&values[source * dim..(source + 1) * dim]);
    }
    out
}

fn reorder_indices(values: &[i32], indices: &[i64]) -> Vec<i32> {
    indices
        .iter()
        .map(|index| values[*index as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{BackendKind, GgmlContext};

    #[test]
    fn published_layout_has_all_vision_blocks() {
        let layout = weight_layout(VisionConfig::published());
        assert_eq!(layout.len(), 1 + 32 * 12 + 5);
        assert!(layout
            .iter()
            .any(|(name, _, _)| name == "visual.blocks.31.attn.qkv.weight"));
        assert!(layout
            .iter()
            .any(|(name, _, _)| name == "visual.merger.mlp.1.bias"));
    }

    #[test]
    fn smoke_vision_runs_patch_attention_and_merger() {
        let config = VisionConfig::smoke();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let grid = VisionGrid { t: 1, h: 2, w: 2 };
        let mut forward = VisionForward::build(&ctx, config, grid).unwrap();
        for (name, ne0, ne1) in weight_layout(config) {
            let data = if name.ends_with("norm1.weight")
                || name.ends_with("norm2.weight")
                || name == "visual.merger.ln_q.weight"
            {
                vec![1.0; (ne0 * ne1) as usize]
            } else {
                vec![0.0; (ne0 * ne1) as usize]
            };
            forward.load_f32(&name, &data).unwrap();
        }
        let (h, w) = position_ids(grid);
        let values = vec![0.25; 4 * 3 * 2 * 14 * 14];
        let output = forward.compute(&values, &h, &w).unwrap();
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn positions_follow_spatial_merge_order() {
        let (h, w) = position_ids(VisionGrid { t: 1, h: 4, w: 4 });
        assert_eq!(h, [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]);
        assert_eq!(w, [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3]);
    }
}
