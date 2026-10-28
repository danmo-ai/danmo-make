//! DiT cross-attention → DTW lyric timestamps (acestep.cpp `lrc-alignment.h`,
//! main `vocals/lyrics_alignment.py`).

use super::lyrics::{LyricsAlignment, LyricsSegment, DIT_ALIGNMENT_MODE};
use crate::weights::WeightError;

/// Bidirectional consensus denoise of `[heads, tokens, frames]` → `[tokens, frames]`.
pub fn consensus_alignment_matrix(
    weights: &[f32],
    n_heads: usize,
    n_tokens: usize,
    n_frames: usize,
    violence_level: f32,
) -> Result<Vec<f32>, WeightError> {
    let expected = n_heads * n_tokens * n_frames;
    if weights.len() != expected {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step CA weights length {} != heads*tokens*frames {expected}",
            weights.len()
        )));
    }
    if n_heads == 0 || n_tokens == 0 || n_frames == 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step CA consensus needs positive heads/tokens/frames".into(),
        ));
    }
    let txf = n_tokens * n_frames;
    let mut processed = vec![0f32; expected];
    let mut row_prob = vec![0f32; txf];
    let mut col_prob = vec![0f32; txf];

    for h in 0..n_heads {
        let src = &weights[h * txf..(h + 1) * txf];
        let dst = &mut processed[h * txf..(h + 1) * txf];
        softmax_rows(src, &mut row_prob, n_tokens, n_frames);
        softmax_cols(src, &mut col_prob, n_tokens, n_frames);
        for i in 0..txf {
            dst[i] = row_prob[i] * col_prob[i];
        }
        let mut r_med = vec![0f32; n_tokens];
        row_medians(dst, &mut r_med, n_tokens, n_frames);
        for t in 0..n_tokens {
            let threshold = violence_level * r_med[t];
            for f in 0..n_frames {
                let v = dst[t * n_frames + f] - threshold;
                dst[t * n_frames + f] = if v > 0.0 { v } else { 0.0 };
            }
        }
        let mut c_med = vec![0f32; n_frames];
        col_medians(dst, &mut c_med, n_tokens, n_frames);
        for t in 0..n_tokens {
            for f in 0..n_frames {
                let v = dst[t * n_frames + f] - violence_level * c_med[f];
                dst[t * n_frames + f] = if v > 0.0 { v } else { 0.0 };
            }
        }
        for v in dst.iter_mut() {
            *v *= *v;
        }
    }

    let n = expected as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &v in &processed {
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
    }
    let mean = (sum / n) as f32;
    let var = (sum_sq / n - (mean as f64) * (mean as f64)) as f32;
    let std = if var > 0.0 { var.sqrt() } else { 0.0 };
    let inv = 1.0 / (std + 1e-9);
    for v in processed.iter_mut() {
        *v = (*v - mean) * inv;
    }

    let mut calc = vec![0f32; txf];
    let inv_h = 1.0 / n_heads as f32;
    for h in 0..n_heads {
        let src = &processed[h * txf..(h + 1) * txf];
        for i in 0..txf {
            calc[i] += src[i] * inv_h;
        }
    }
    Ok(calc)
}

/// Monotonic DTW path on cost `[N, M]` (acestep.cpp `dtw_cpu` tie-break).
pub fn dtw_path(cost: &[f64], n: usize, m: usize) -> Result<(Vec<i64>, Vec<i64>), WeightError> {
    if n == 0 || m == 0 || cost.len() != n * m {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step DTW cost shape {n}x{m} length {}",
            cost.len()
        )));
    }
    let cols = m + 1;
    let mut acc = vec![f32::INFINITY; (n + 1) * cols];
    let mut trace = vec![-1i8; (n + 1) * cols];
    acc[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let c0 = acc[(i - 1) * cols + (j - 1)];
            let c1 = acc[(i - 1) * cols + j];
            let c2 = acc[i * cols + (j - 1)];
            let (best, t) = if c0 < c1 && c0 < c2 {
                (c0, 0i8)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            acc[i * cols + j] = cost[(i - 1) * m + (j - 1)] as f32 + best;
            trace[i * cols + j] = t;
        }
    }
    for j in 0..=m {
        trace[j] = 2;
    }
    for i in 0..=n {
        trace[i * cols] = 1;
    }

    let mut text = Vec::new();
    let mut time = Vec::new();
    let mut i = n as i64;
    let mut j = m as i64;
    while i > 0 || j > 0 {
        text.push(i - 1);
        time.push(j - 1);
        let t = trace[(i as usize) * cols + (j as usize)];
        match t {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            2 => j -= 1,
            _ => break,
        }
    }
    text.reverse();
    time.reverse();
    Ok((text, time))
}

/// `attention` is `[heads, lyric_tokens, frames]` row-major.
pub fn align_lyrics_from_attention(
    attention: &[f32],
    n_heads: usize,
    lyric_token_ids: &[i32],
    n_frames: usize,
    total_duration_sec: f32,
    decode: &dyn Fn(&[i32]) -> String,
    violence_level: f32,
) -> Result<LyricsAlignment, WeightError> {
    let n_tokens = lyric_token_ids.len();
    if n_tokens == 0 {
        return Ok(LyricsAlignment {
            mode: "empty".into(),
            segments: Vec::new(),
        });
    }
    if attention.len() != n_heads * n_tokens * n_frames {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step attention length {} does not match heads={} tokens={} frames={}",
            attention.len(),
            n_heads,
            n_tokens,
            n_frames
        )));
    }
    if !(total_duration_sec > 0.0) || !total_duration_sec.is_finite() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step DTW duration {total_duration_sec} must be positive finite"
        )));
    }

    let calc = consensus_alignment_matrix(
        attention,
        n_heads,
        n_tokens,
        n_frames,
        violence_level,
    )?;
    let mut neg = vec![0f64; calc.len()];
    for (i, &v) in calc.iter().enumerate() {
        neg[i] = -(v as f64);
    }
    let (text_idx, time_idx) = dtw_path(&neg, n_tokens, n_frames)?;
    let sec_per_frame = total_duration_sec / n_frames as f32;

    let mut starts = vec![0f32; n_tokens];
    let mut ends = vec![0f32; n_tokens];
    for k in 0..n_tokens {
        let mut first = None;
        let mut last = 0f32;
        for p in 0..text_idx.len() {
            if text_idx[p] == k as i64 {
                let t = time_idx[p] as f32 * sec_per_frame;
                if first.is_none() {
                    first = Some(t);
                }
                last = t;
            }
        }
        if let Some(s) = first {
            starts[k] = s;
            ends[k] = last.max(s);
        } else if k > 0 {
            starts[k] = ends[k - 1];
            ends[k] = ends[k - 1];
        }
    }

    let pieces: Vec<String> = (0..n_tokens)
        .map(|i| {
            let cur = decode(&lyric_token_ids[..=i]);
            let prev = if i == 0 {
                String::new()
            } else {
                decode(&lyric_token_ids[..i])
            };
            incremental_piece(&prev, &cur)
        })
        .collect();

    let mut segments = Vec::new();
    let mut label = "body".to_string();
    let mut group: Vec<usize> = Vec::new();
    let mut first_last = (0usize, 0usize);

    let flush = |group: &mut Vec<usize>,
                 label: &mut String,
                 first_last: (usize, usize),
                 segments: &mut Vec<LyricsSegment>| {
        if group.is_empty() {
            return;
        }
        let ids: Vec<i32> = group.iter().map(|&k| lyric_token_ids[k]).collect();
        group.clear();
        let text = decode(&ids).trim().to_string();
        if text.is_empty() {
            return;
        }
        if let Some(sec) = section_header(&text) {
            *label = if sec.is_empty() {
                "section".into()
            } else {
                sec
            };
            return;
        }
        let (k0, k1) = first_last;
        segments.push(LyricsSegment {
            label: label.clone(),
            start_sec: (starts[k0] * 1000.0).round() / 1000.0,
            end_sec: (ends[k1].max(starts[k0]) * 1000.0).round() / 1000.0,
            text,
        });
    };

    for (k, piece) in pieces.iter().enumerate() {
        if group.is_empty() {
            first_last.0 = k;
        }
        group.push(k);
        first_last.1 = k;
        if piece.contains('\n') {
            flush(&mut group, &mut label, first_last, &mut segments);
        }
    }
    flush(&mut group, &mut label, first_last, &mut segments);

    Ok(LyricsAlignment {
        mode: DIT_ALIGNMENT_MODE.into(),
        segments,
    })
}

fn incremental_piece(prev: &str, cur: &str) -> String {
    let pb = prev.as_bytes();
    let cb = cur.as_bytes();
    if cb.len() >= pb.len() && cb.starts_with(pb) {
        String::from_utf8_lossy(&cb[pb.len()..]).into_owned()
    } else {
        String::new()
    }
}

fn section_header(text: &str) -> Option<String> {
    let t = text.trim();
    if t.starts_with('[') && t.ends_with(']') && t.len() >= 2 {
        Some(t[1..t.len() - 1].trim().to_string())
    } else {
        None
    }
}

fn softmax_rows(x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &x[i * cols..(i + 1) * cols];
        let dst = &mut out[i * cols..(i + 1) * cols];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..cols {
            dst[j] = (row[j] - mx).exp();
            sum += dst[j];
        }
        let inv = 1.0 / (sum + 1e-9);
        for v in dst.iter_mut() {
            *v *= inv;
        }
    }
}

fn softmax_cols(x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        let mut mx = x[j];
        for i in 1..rows {
            mx = mx.max(x[i * cols + j]);
        }
        let mut sum = 0.0f32;
        for i in 0..rows {
            let v = (x[i * cols + j] - mx).exp();
            out[i * cols + j] = v;
            sum += v;
        }
        let inv = 1.0 / (sum + 1e-9);
        for i in 0..rows {
            out[i * cols + j] *= inv;
        }
    }
}

fn median_small(buf: &mut [f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let mid = buf.len() / 2;
    buf.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    buf[mid]
}

fn row_medians(x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    let mut buf = vec![0f32; cols];
    for i in 0..rows {
        buf.copy_from_slice(&x[i * cols..(i + 1) * cols]);
        out[i] = median_small(&mut buf);
    }
}

fn col_medians(x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    let mut buf = vec![0f32; rows];
    for j in 0..cols {
        for i in 0..rows {
            buf[i] = x[i * cols + j];
        }
        out[j] = median_small(&mut buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_diagonal_on_neg_identity() {
        // Strong diagonal preference → identity path.
        let n = 3usize;
        let m = 3usize;
        let mut cost = vec![1.0f64; n * m];
        for i in 0..n {
            cost[i * m + i] = -10.0;
        }
        let (ti, tj) = dtw_path(&cost, n, m).unwrap();
        assert!(!ti.is_empty());
        assert_eq!(ti.len(), tj.len());
        assert_eq!(*ti.first().unwrap(), 0);
        assert_eq!(*ti.last().unwrap(), 2);
    }

    #[test]
    fn align_from_diagonal_attention_groups_lines() {
        let heads = 1usize;
        let tokens = 5usize;
        let frames = 10usize;
        // Tokens: "a\n" "b\n" roughly — use decode that emits newlines on ids 1 and 3.
        let ids = [10, 11, 12, 13, 14];
        let mut attn = vec![0.1f32; heads * tokens * frames];
        for t in 0..tokens {
            let f = (t * frames) / tokens;
            attn[t * frames + f] = 5.0;
        }
        let decode = |ids: &[i32]| -> String {
            let mut s = String::new();
            for &id in ids {
                match id {
                    10 => s.push('a'),
                    11 => {
                        s.push('1');
                        s.push('\n');
                    }
                    12 => s.push('b'),
                    13 => {
                        s.push('2');
                        s.push('\n');
                    }
                    14 => s.push('c'),
                    _ => {}
                }
            }
            s
        };
        let align = align_lyrics_from_attention(
            &attn, heads, &ids, frames, 20.0, &decode, 2.0,
        )
        .unwrap();
        assert_eq!(align.mode, DIT_ALIGNMENT_MODE);
        assert!(align.segments.len() >= 2, "{:?}", align.segments);
        assert!(align.segments[0].text.contains('a'));
    }
}
