use super::{BackendKind, GgmlContext, ReusedGraph, RuntimeError};
use std::time::Instant;

const HEAD_DIM: i64 = 64;
const HEADS: i64 = 2;
const SEQ: i64 = 32;
const RUNS: usize = 5;

#[derive(Debug, Clone)]
pub struct AttentionProbeReport {
    pub backend: BackendKind,
    pub max_abs_error: f32,
    pub reference_ms: f64,
    pub flash_ms: f64,
    pub validated: bool,
}

impl AttentionProbeReport {
    pub fn summary(&self) -> String {
        format!(
            "attention probe backend={} max_abs_error={:.6} host_reference={:.3}ms flash={:.3}ms validated={}",
            self.backend.name(),
            self.max_abs_error,
            self.reference_ms,
            self.flash_ms,
            self.validated
        )
    }
}

pub fn probe_attention(ctx: &GgmlContext) -> Result<AttentionProbeReport, RuntimeError> {
    probe_attention_shape(ctx, HEAD_DIM, HEADS, SEQ)
}

fn probe_attention_shape(
    ctx: &GgmlContext,
    head_dim: i64,
    heads: i64,
    seq: i64,
) -> Result<AttentionProbeReport, RuntimeError> {
    let (q, k, v) = inputs(head_dim, heads, seq);
    let reference_started = Instant::now();
    let expected = host_attention(&q, &k, &v, head_dim, heads, seq);
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1000.0;
    let mut flash = build(ctx, head_dim, heads, seq)?;
    set_inputs(&mut flash, &q, &k, &v)?;
    let actual = flash.compute()?;
    if expected.len() != actual.len() {
        return Err(RuntimeError::Message(format!(
            "attention probe output lengths differ: {} vs {}",
            expected.len(),
            actual.len()
        )));
    }
    let max_abs_error = expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if !max_abs_error.is_finite() || max_abs_error > 0.02 {
        return Err(RuntimeError::Message(format!(
            "flash attention probe mismatch: max abs error {max_abs_error}"
        )));
    }
    let flash_ms = measure(&mut flash)?;
    Ok(AttentionProbeReport {
        backend: ctx.kind(),
        max_abs_error,
        reference_ms,
        flash_ms,
        validated: true,
    })
}

fn build(
    ctx: &GgmlContext,
    head_dim: i64,
    heads: i64,
    seq: i64,
) -> Result<ReusedGraph, RuntimeError> {
    let mut builder = ctx.builder(32 * 1024 * 1024)?;
    let q = builder.f32_3d("q", head_dim, heads, seq, true);
    let k = builder.f32_3d("k", head_dim, heads, seq, true);
    let v = builder.f32_3d("v", head_dim, heads, seq, true);
    let output = super::flash_attend(&builder, q, k, v, std::ptr::null_mut())?;
    builder.finish(output)
}

fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: i64,
    heads: i64,
    seq: i64,
) -> Vec<f32> {
    let d = head_dim as usize;
    let heads = heads as usize;
    let seq = seq as usize;
    let mut output = vec![0.0f32; d * heads * seq];
    let scale = 1.0 / (d as f32).sqrt();
    for head in 0..heads {
        for query in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for key in 0..seq {
                let mut score = 0.0f32;
                for dim in 0..d {
                    score +=
                        q[dim + d * (head + heads * query)] * k[dim + d * (head + heads * key)];
                }
                scores[key] = score * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                total += *score;
            }
            for dim in 0..d {
                let mut value = 0.0f32;
                for key in 0..seq {
                    value += scores[key] * v[dim + d * (head + heads * key)];
                }
                output[dim + d * (head + heads * query)] = value / total;
            }
        }
    }
    output
}

fn inputs(head_dim: i64, heads: i64, seq: i64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = (head_dim * heads * seq) as usize;
    let mut q = Vec::with_capacity(len);
    let mut k = Vec::with_capacity(len);
    let mut v = Vec::with_capacity(len);
    for index in 0..len {
        let x = index as f32;
        q.push((x * 0.013).sin());
        k.push((x * 0.017).cos());
        v.push((x * 0.019).sin() * 0.5);
    }
    (q, k, v)
}

fn set_inputs(
    graph: &mut ReusedGraph,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<(), RuntimeError> {
    graph.set_f32("q", q)?;
    graph.set_f32("k", k)?;
    graph.set_f32("v", v)
}

fn measure(graph: &mut ReusedGraph) -> Result<f64, RuntimeError> {
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        graph.compute_device()?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[RUNS / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_probe_validates_flash_against_host_reference() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let report = probe_attention(&ctx).unwrap();
        assert!(report.validated);
        assert!(report.max_abs_error <= 0.02);
        assert!(report.reference_ms >= 0.0);
        assert!(report.flash_ms >= 0.0);
        assert!(!report.summary().is_empty());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_probe_validates_flash_against_host_reference() {
        let ctx = GgmlContext::new(BackendKind::Metal).unwrap();
        let report = probe_attention(&ctx).unwrap();
        assert!(report.validated);
        assert!(report.max_abs_error <= 0.02);
        assert!(report.flash_ms >= 0.0);
        println!("{}", report.summary());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_small_head_fallback_matches_host_reference() {
        let ctx = GgmlContext::new(BackendKind::Metal).unwrap();
        let report = probe_attention_shape(&ctx, 4, 1, 17).unwrap();
        assert!(report.validated);
        assert!(report.max_abs_error <= 0.02);
    }
}
