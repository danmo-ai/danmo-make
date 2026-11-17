use std::path::{Path, PathBuf};

use danmo_mlx::{SafeTensors, Stream};

fn scratch_file() -> PathBuf {
    std::env::temp_dir().join(format!(
        "danmo-mlx-safetensors-{}-{}.safetensors",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn write_f32_fixture(path: &Path, values: &[f32]) {
    let mut header = format!(
        "{{\"weight\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,{}]}},\"__metadata__\":{{\"format\":\"pt\"}}}}",
        values.len() * 4
    );
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn safetensors_loads_names_shapes_and_values() {
    let path = scratch_file();
    let values = [1.0, 2.0, 3.0, 4.0];
    write_f32_fixture(&path, &values);
    let stream = Stream::cpu().unwrap();
    let tensors = SafeTensors::load(&path, &stream).unwrap();
    assert_eq!(tensors.names().unwrap(), vec!["weight"]);
    assert!(tensors.get("missing").unwrap().is_none());
    let weight = tensors.get("weight").unwrap().unwrap();
    assert_eq!(weight.shape(), vec![2, 2]);
    assert_eq!(weight.to_vec_f32(&stream).unwrap(), values);
    drop(tensors);
    let _ = std::fs::remove_file(path);
}

#[test]
fn safetensors_round_trip_keeps_packed_dtypes_and_metadata() {
    let path = scratch_file();
    let stream = Stream::cpu().unwrap();
    let values: Vec<f32> = (0..2 * 64).map(|index| index as f32 * 0.01 - 0.5).collect();
    let dense = danmo_mlx::Array::from_f32_slice(&values, &[2, 64])
        .unwrap()
        .astype(danmo_mlx::Dtype::BFloat16, &stream)
        .unwrap();
    let packed = dense.quantize_affine(64, 4, &stream).unwrap();
    danmo_mlx::save_safetensors(
        &path,
        &[
            ("layer.weight", &packed.weight),
            ("layer.scales", &packed.scales),
            ("layer.biases", &packed.biases),
        ],
        &[("quantization_level", "4")],
    )
    .unwrap();
    let tensors = SafeTensors::load(&path, &stream).unwrap();
    assert_eq!(
        tensors.metadata("quantization_level").unwrap().as_deref(),
        Some("4")
    );
    assert!(tensors.metadata("missing").unwrap().is_none());
    let weight = tensors.get("layer.weight").unwrap().unwrap();
    assert_eq!(weight.dtype(), danmo_mlx::Dtype::UInt32);
    assert_eq!(weight.shape(), vec![2, 8]);
    let scales = tensors.get("layer.scales").unwrap().unwrap();
    assert_eq!(scales.dtype(), danmo_mlx::Dtype::BFloat16);
    assert_eq!(scales.shape(), vec![2, 1]);
    drop(tensors);
    let _ = std::fs::remove_file(path);
}

#[test]
fn missing_safetensors_fails_loud() {
    let stream = Stream::cpu().unwrap();
    let path = scratch_file();
    let error = match SafeTensors::load(&path, &stream) {
        Err(error) => error,
        Ok(_) => panic!("missing safetensors unexpectedly loaded"),
    };
    assert!(error.to_string().contains("mlx_load_safetensors"));
}

#[test]
#[ignore]
fn real_qwen_safetensors_shard_probe() {
    let root = std::env::var("DANQING_TEST_QWEN_SAFETENSORS").unwrap();
    let stream = Stream::cpu().unwrap();
    let tensors = SafeTensors::load(Path::new(&root), &stream).unwrap();
    assert!(!tensors.names().unwrap().is_empty());
    let embedding = tensors
        .get("model.embed_tokens.weight")
        .unwrap()
        .expect("Qwen shard is missing model.embed_tokens.weight");
    assert_eq!(embedding.shape().len(), 2);
    assert!(embedding.shape()[0] > 0 && embedding.shape()[1] > 0);
}
