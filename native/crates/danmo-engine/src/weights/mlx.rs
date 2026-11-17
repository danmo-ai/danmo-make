use std::collections::BTreeMap;
use std::path::Path;

use danmo_mlx::{Array, Error, Result, SafeTensors, Stream};

pub fn load_directory(directory: &Path) -> Result<BTreeMap<String, Array>> {
    let mut paths = std::fs::read_dir(directory)
        .map_err(|error| Error::new("MLX safetensors directory", error.to_string()))?
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("safetensors"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(Error::new(
            "MLX safetensors directory",
            format!("no safetensors shards in {}", directory.display()),
        ));
    }
    let io = Stream::cpu()?;
    let mut output = BTreeMap::new();
    for path in paths {
        let tensors = SafeTensors::load(&path, &io)?;
        for name in tensors.names()? {
            let tensor = tensors
                .get(&name)?
                .ok_or_else(|| Error::new("MLX safetensors", format!("missing {name}")))?;
            if output.insert(name.clone(), tensor).is_some() {
                return Err(Error::new(
                    "MLX safetensors directory",
                    format!("duplicate tensor {name}"),
                ));
            }
        }
    }
    Ok(output)
}
