pub fn rms_norm(input: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), rows * dim);
    assert_eq!(weight.len(), dim);
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let mean = (0..dim)
            .map(|column| input[row * dim + column].powi(2))
            .sum::<f32>()
            / dim as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for column in 0..dim {
            output[row * dim + column] = input[row * dim + column] * scale * weight[column];
        }
    }
    output
}

pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * input_dim);
    assert_eq!(weight.len(), output_dim * input_dim);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), output_dim);
    }
    let mut output = vec![0.0; rows * output_dim];
    for row in 0..rows {
        for out in 0..output_dim {
            let mut value = bias.map_or(0.0, |bias| bias[out]);
            for inner in 0..input_dim {
                value += input[row * input_dim + inner] * weight[out * input_dim + inner];
            }
            output[row * output_dim + out] = value;
        }
    }
    output
}
