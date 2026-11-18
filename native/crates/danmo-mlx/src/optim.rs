use crate::{Array, Result, Stream};

/// AdamW hyperparameters (MLX / PyTorch-compatible defaults).
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Per-parameter first/second moment state for AdamW.
pub struct AdamWState {
    pub m: Array,
    pub v: Array,
    pub step: u64,
}

impl AdamWState {
    pub fn zeros_like(param: &Array, stream: &Stream) -> Result<Self> {
        Ok(Self {
            m: param.zeros_like(stream)?,
            v: param.zeros_like(stream)?,
            step: 0,
        })
    }
}

/// Apply one AdamW update in-place on `param`, returning the updated array.
///
/// Decoupled weight decay: `param *= (1 - lr * wd)` then subtract the
/// bias-corrected adaptive step.
pub fn adamw_update(
    param: &Array,
    grad: &Array,
    state: &mut AdamWState,
    config: &AdamWConfig,
    stream: &Stream,
) -> Result<Array> {
    state.step = state.step.saturating_add(1);
    let step = state.step as f32;
    let (beta1, beta2) = config.betas;

    let one_m_b1 = Array::scalar_f32(1.0 - beta1)?;
    let one_m_b2 = Array::scalar_f32(1.0 - beta2)?;
    let beta1_t = Array::scalar_f32(beta1)?;
    let beta2_t = Array::scalar_f32(beta2)?;

    let m_scaled = Array::multiply(&state.m, &beta1_t, stream)?;
    let g_scaled = Array::multiply(grad, &one_m_b1, stream)?;
    state.m = Array::add(&m_scaled, &g_scaled, stream)?;

    let v_scaled = Array::multiply(&state.v, &beta2_t, stream)?;
    let g2 = grad.square(stream)?;
    let g2_scaled = Array::multiply(&g2, &one_m_b2, stream)?;
    state.v = Array::add(&v_scaled, &g2_scaled, stream)?;

    let bias1 = 1.0 - beta1.powf(step);
    let bias2 = 1.0 - beta2.powf(step);
    let m_hat = Array::divide(&state.m, &Array::scalar_f32(bias1)?, stream)?;
    let v_hat = Array::divide(&state.v, &Array::scalar_f32(bias2)?, stream)?;

    let denom = Array::add(&v_hat.sqrt(stream)?, &Array::scalar_f32(config.eps)?, stream)?;
    let step_size = Array::divide(&m_hat, &denom, stream)?;
    let lr_step = Array::multiply(&step_size, &Array::scalar_f32(config.learning_rate)?, stream)?;

    let decayed = if config.weight_decay != 0.0 {
        let scale = Array::scalar_f32(1.0 - config.learning_rate * config.weight_decay)?;
        Array::multiply(param, &scale, stream)?
    } else {
        param.clone()
    };
    Array::subtract(&decayed, &lr_step, stream)
}

/// Update a list of parameters with matching grads and states.
pub fn adamw_update_all(
    params: &mut [Array],
    grads: &[Array],
    states: &mut [AdamWState],
    config: &AdamWConfig,
    stream: &Stream,
) -> Result<()> {
    if params.len() != grads.len() || params.len() != states.len() {
        return Err(crate::Error::new(
            "adamw_update_all",
            format!(
                "params/grads/states length mismatch: {}/{}/{}",
                params.len(),
                grads.len(),
                states.len()
            ),
        ));
    }
    for ((param, grad), state) in params.iter_mut().zip(grads).zip(states.iter_mut()) {
        *param = adamw_update(param, grad, state, config, stream)?;
    }
    let refs: Vec<&Array> = params.iter().collect();
    Array::eval_all(&refs)?;
    for state in states.iter() {
        Array::eval_all(&[&state.m, &state.v])?;
    }
    Ok(())
}
