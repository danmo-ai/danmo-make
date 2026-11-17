mod array;
mod cancel;
mod compile;
mod device;
mod error;
mod executor;
mod io;
mod linear;
mod memory;
mod random;
mod stream;

use std::ffi::CStr;

pub use array::{Array, Dtype, QuantizedArray};
pub use cancel::CancelToken;
pub use compile::{Closure, MultiClosure, UnaryClosure};
pub use device::{Device, DeviceType};
pub use error::{Error, Result};
pub use executor::Executor;
pub use io::{save_safetensors, SafeTensors};
pub use linear::{AffineQuantization, LinearWeight};
pub use memory::{active_memory, clear_cache, peak_memory, reset_peak_memory};
pub use random::RandomKey;
pub use stream::Stream;

pub fn version() -> Result<String> {
    error::initialize();
    error::clear_error();
    let mut raw = unsafe { danmo_mlx_sys::mlx_string_new() };
    error::check(
        unsafe { danmo_mlx_sys::mlx_version(&mut raw) },
        "mlx_version",
    )?;
    let value = unsafe {
        let data = danmo_mlx_sys::mlx_string_data(raw);
        if data.is_null() {
            return Err(Error::new("mlx_version", "returned a null string"));
        }
        CStr::from_ptr(data).to_string_lossy().into_owned()
    };
    unsafe {
        let _ = danmo_mlx_sys::mlx_string_free(raw);
    }
    error::clear_error();
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }

    fn matmul_probe(stream: &Stream) -> Result<Vec<f32>> {
        let lhs = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let rhs = Array::from_f32_slice(
            &[
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            &[3, 4],
        )?;
        let output = Array::matmul(&lhs, &rhs, stream)?;
        assert_eq!(output.shape(), vec![2, 4]);
        output.to_vec_f32(stream)
    }

    fn conv3d_probe(stream: &Stream) -> Result<(Vec<f32>, Vec<f32>)> {
        let input =
            Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2, 1])?;
        let kernel = Array::from_f32_slice(&[1.0; 8], &[1, 2, 2, 2, 1])?;
        let convolution =
            Array::conv3d(&input, &kernel, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, stream)?;
        assert_eq!(convolution.shape(), vec![1, 1, 1, 1, 1]);

        let point = Array::from_f32_slice(&[2.0], &[1, 1, 1, 1, 1])?;
        let transposed =
            Array::conv_transpose3d(&point, &kernel, [1, 1, 1], [0, 0, 0], [0, 0, 0], stream)?;
        assert_eq!(transposed.shape(), vec![1, 2, 2, 2, 1]);
        Ok((
            convolution.to_vec_f32(stream)?,
            transposed.to_vec_f32(stream)?,
        ))
    }

    fn quantized_matmul_probe(stream: &Stream) -> Result<Vec<f32>> {
        let input_values: Vec<f32> = (0..32).map(|index| index as f32 - 16.0).collect();
        let input = Array::from_f32_slice(&input_values, &[1, 32])?;
        let mut weight_values = vec![1.0; 32];
        weight_values.extend(vec![-1.0; 32]);
        let dense = Array::from_f32_slice(&weight_values, &[2, 32])?;
        let quantized = dense.quantize_affine(32, 4, stream)?;
        let output = Array::quantized_matmul(&input, &quantized, true, 32, 4, stream)?;
        assert_eq!(output.shape(), vec![1, 2]);
        output.to_vec_f32(stream)
    }

    fn large_quantized_matmul_probe(stream: &Stream) -> Result<()> {
        const ROWS: i32 = 32_769;
        let input = Array::from_f32_slice(&vec![1.0; ROWS as usize * 32], &[ROWS, 32])?;
        let mut weight_values = vec![1.0; 32];
        weight_values.extend(vec![-1.0; 32]);
        let dense = Array::from_f32_slice(&weight_values, &[2, 32])?;
        let quantized = dense.quantize_affine(32, 4, stream)?;
        let output = Array::quantized_matmul(&input, &quantized, true, 32, 4, stream)?;
        assert_eq!(output.shape(), vec![ROWS, 2]);
        let values = output.to_vec_f32(stream)?;
        assert_eq!(values.len(), ROWS as usize * 2);
        for pair in values.chunks_exact(2) {
            assert_close(pair, &[32.0, -32.0], 2e-3);
        }
        Ok(())
    }

    fn compile_probe(device_type: DeviceType) -> Result<()> {
        let device = Device::new(device_type, 0)?;
        let eager =
            UnaryClosure::new(Stream::new(&device)?, |input, stream| input.sigmoid(stream))?;
        let compiled =
            UnaryClosure::new(Stream::new(&device)?, |input, stream| input.sigmoid(stream))?
                .compile(true)?;
        let readback = Stream::new(&device)?;
        for values in [
            vec![-2.0, -0.5, 0.0, 0.5, 2.0, 3.0],
            vec![3.0, 2.0, 0.5, 0.0, -0.5, -2.0],
        ] {
            let input = Array::from_f32_slice(&values, &[2, 3])?;
            let expected: Vec<f32> = values
                .iter()
                .map(|value| 1.0 / (1.0 + (-value).exp()))
                .collect();
            let eager_values = eager.apply(&input)?.to_vec_f32(&readback)?;
            let compiled_values = compiled.apply(&input)?.to_vec_f32(&readback)?;
            assert_close(&eager_values, &expected, 1e-6);
            assert_close(&compiled_values, &expected, 1e-6);
            assert_close(&compiled_values, &eager_values, 1e-6);
        }
        Ok(())
    }

    #[test]
    fn conv1x1_without_padding_preserves_spatial_shape() {
        let stream = Stream::cpu().unwrap();
        let input = Array::from_f32_slice(&vec![0.0; 8 * 8 * 4], &[1, 8, 8, 4]).unwrap();
        let weight = Array::from_f32_slice(&vec![0.0; 2 * 4], &[2, 1, 1, 4]).unwrap();
        let output = Array::conv2d(&input, &weight, [1, 1], [0, 0], [1, 1], 1, &stream).unwrap();
        assert_eq!(output.shape(), vec![1, 8, 8, 2]);
    }

    #[test]
    fn cpu_matmul_matches_host_reference() {
        let stream = Stream::cpu().unwrap();
        let output = matmul_probe(&stream).unwrap();
        assert_close(
            &output,
            &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
            1e-6,
        );
    }

    #[test]
    fn cpu_conv3d_transpose_and_quantized_matmul_match_host_reference() {
        let stream = Stream::cpu().unwrap();
        let (convolution, transposed) = conv3d_probe(&stream).unwrap();
        assert_close(&convolution, &[36.0], 1e-6);
        assert_close(&transposed, &[2.0; 8], 1e-6);
        let sum: f32 = (0..32).map(|index| index as f32 - 16.0).sum();
        assert_close(
            &quantized_matmul_probe(&stream).unwrap(),
            &[sum, -sum],
            2e-3,
        );
    }

    #[test]
    fn cpu_quantized_matmul_supports_more_than_32k_rows() {
        let stream = Stream::cpu().unwrap();
        large_quantized_matmul_probe(&stream).unwrap();
    }

    #[test]
    fn cpu_shapeless_compile_matches_eager_across_replays() {
        compile_probe(DeviceType::Cpu).unwrap();
    }

    fn variadic_compile_probe(device_type: DeviceType) -> Result<()> {
        let device = Device::new(device_type, 0)?;
        let stream = Stream::new(&device)?;
        // Captured constants stay outside the traced inputs, like model weights.
        let weight = Array::from_f32_slice(&[0.5, -0.25, 0.75, 1.25, -0.5, 0.125], &[3, 2])?;
        let bias = Array::from_f32_slice(&[0.1, -0.2, 0.3], &[3])?;
        let function = {
            let weight = weight.clone();
            let bias = bias.clone();
            move |inputs: &[Array], stream: &Stream| -> Result<Array> {
                assert_eq!(inputs.len(), 2);
                let projected = Array::linear(&inputs[0], &weight, Some(&bias), stream)?;
                let gated = Array::multiply(&projected, &inputs[1].silu(stream)?, stream)?;
                gated.rms_norm(None, 1e-5, stream)
            }
        };
        let eager = Closure::new(Stream::new(&device)?, function.clone())?;
        let compiled = Closure::new(Stream::new(&device)?, function)?.compile(true)?;
        let readback = Stream::new(&device)?;
        for step in 0..3 {
            let scale = 1.0 + step as f32 * 0.5;
            let x = Array::from_f32_slice(
                &[0.3 * scale, -0.7 * scale, 1.1 * scale, 0.2 * scale],
                &[2, 2],
            )?;
            let gate = Array::from_f32_slice(&[scale, -scale, 0.5 * scale], &[1, 3])?;
            let inputs = [x, gate];
            let eager_values = eager.apply(&inputs)?.to_vec_f32(&readback)?;
            let compiled_values = compiled.apply(&inputs)?.to_vec_f32(&readback)?;
            assert_close(&compiled_values, &eager_values, 1e-5);
        }
        // A different input shape must recompile cleanly under shapeless mode.
        let x = Array::from_f32_slice(&[0.4, 0.9, -0.2, 0.6, 1.3, -0.8], &[3, 2])?;
        let gate = Array::from_f32_slice(&[0.7, 0.2, -0.4], &[1, 3])?;
        let inputs = [x, gate];
        let eager_values = eager.apply(&inputs)?.to_vec_f32(&readback)?;
        let compiled_values = compiled.apply(&inputs)?.to_vec_f32(&readback)?;
        assert_close(&compiled_values, &eager_values, 1e-5);
        drop(eager);
        drop(compiled);
        drop(weight);
        drop(bias);
        drop(stream);
        Ok(())
    }

    #[test]
    fn cpu_variadic_compile_matches_eager_across_replays() {
        variadic_compile_probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_variadic_compile_matches_eager_across_replays() {
        variadic_compile_probe(DeviceType::Gpu).unwrap();
    }

    #[test]
    fn variadic_closure_panic_is_an_error_not_a_crash() {
        let stream = Stream::cpu().unwrap();
        let closure = Closure::new(Stream::cpu().unwrap(), |_inputs, _stream| {
            panic!("closure body panicked");
        })
        .unwrap();
        let input = Array::from_f32_slice(&[1.0], &[1]).unwrap();
        let error = match closure.apply(&[input]) {
            Err(error) => error,
            Ok(_) => panic!("panicking closure unexpectedly succeeded"),
        };
        // mlx-c reports the non-zero closure return; the panic is caught at the
        // FFI boundary so the process survives and the next call stays clean.
        assert!(error.to_string().contains("mlx_closure_apply"), "{error}");
        // The thread-local error state must not poison later calls.
        let stream2 = Stream::cpu().unwrap();
        let value = Array::from_f32_slice(&[2.0], &[1])
            .unwrap()
            .sigmoid(&stream2)
            .unwrap()
            .to_vec_f32(&stream2)
            .unwrap();
        assert!((value[0] - 0.8807971).abs() < 1e-6);
        let _ = stream;
    }

    #[test]
    fn failed_mlx_call_does_not_poison_the_next_call() {
        let stream = Stream::cpu().unwrap();
        let lhs = Array::from_f32_slice(&[1.0; 6], &[2, 3]).unwrap();
        let rhs = Array::from_f32_slice(&[2.0; 6], &[2, 3]).unwrap();
        let error = match Array::matmul(&lhs, &rhs, &stream) {
            Err(error) => error,
            Ok(_) => panic!("incompatible matmul shapes unexpectedly succeeded"),
        };
        assert!(error.to_string().contains("mlx_matmul"));
        assert_close(
            &matmul_probe(&stream).unwrap(),
            &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
            1e-6,
        );
    }

    #[test]
    fn version_is_reported() {
        assert!(!version().unwrap().is_empty());
    }

    #[test]
    fn executor_reuses_one_thread_and_recovers_from_job_panics() {
        let executor = Executor::spawn(DeviceType::Cpu).unwrap();
        let first = executor
            .execute("executor thread", |_, _| Ok(std::thread::current().id()))
            .unwrap();
        let second = executor
            .execute("executor thread", |_, _| Ok(std::thread::current().id()))
            .unwrap();
        assert_eq!(first, second);
        assert_ne!(first, std::thread::current().id());

        let error = executor
            .execute("executor panic", |_, _| -> Result<()> {
                panic!("probe panic")
            })
            .unwrap_err();
        assert!(error.to_string().contains("job panicked"));
        for _ in 0..2 {
            let output = executor
                .execute("executor matmul", |_, stream| matmul_probe(stream))
                .unwrap();
            assert_close(
                &output,
                &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
                1e-6,
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matmul_and_sdpa_match_host_reference() {
        let device = Device::gpu().unwrap();
        assert!(device.is_available().unwrap(), "Metal GPU is unavailable");
        let stream = Stream::new(&device).unwrap();
        assert_close(
            &matmul_probe(&stream).unwrap(),
            &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
            1e-5,
        );

        let queries = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let keys = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let values = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let output = Array::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            std::f32::consts::FRAC_1_SQRT_2,
            "",
            None,
            &stream,
        )
        .unwrap()
        .to_vec_f32(&stream)
        .unwrap();
        assert_close(&output, &[1.6604775, 2.6604775, 2.3395226, 3.3395226], 2e-4);

        let (convolution, transposed) = conv3d_probe(&stream).unwrap();
        assert_close(&convolution, &[36.0], 1e-5);
        assert_close(&transposed, &[2.0; 8], 1e-5);
        let sum: f32 = (0..32).map(|index| index as f32 - 16.0).sum();
        assert_close(
            &quantized_matmul_probe(&stream).unwrap(),
            &[sum, -sum],
            2e-3,
        );

        let executor = Executor::spawn(DeviceType::Gpu).unwrap();
        let output = executor
            .execute("Metal executor matmul", |_, stream| matmul_probe(stream))
            .unwrap();
        assert_close(
            &output,
            &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
            1e-5,
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_quantized_matmul_supports_more_than_32k_rows() {
        let stream = Stream::gpu().unwrap();
        large_quantized_matmul_probe(&stream).unwrap();
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_shapeless_compile_matches_eager_across_replays() {
        compile_probe(DeviceType::Gpu).unwrap();
    }
}
