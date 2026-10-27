//! Read and write GGUF. Keys are stored and looked up exactly as written.

use super::WeightError;
use danmo_ggml_sys as g;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::path::Path;

pub struct GgufWeights {
    parts: Vec<Part>,
}

struct Part {
    origin: String,
    map: MappedFile,
    tensors: BTreeMap<String, TensorLoc>,
}

struct TensorLoc {
    ggml_type: u32,
    ne: [i64; 4],
    offset: usize,
    nbytes: usize,
}

impl GgufWeights {
    pub fn open(path: &Path) -> Result<Self, WeightError> {
        Ok(Self {
            parts: vec![Part::open(path)?],
        })
    }

    pub fn open_parts(paths: &[std::path::PathBuf]) -> Result<Self, WeightError> {
        let mut parts = Vec::with_capacity(paths.len());
        let mut seen = BTreeMap::<String, String>::new();
        for path in paths {
            let part = Part::open(path)?;
            for name in part.tensors.keys() {
                if let Some(prev) = seen.get(name) {
                    return Err(WeightError::Duplicate {
                        name: name.clone(),
                        path: format!("{prev} and {}", part.origin),
                    });
                }
                seen.insert(name.clone(), part.origin.clone());
            }
            parts.push(part);
        }
        Ok(Self { parts })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.parts
            .iter()
            .flat_map(|part| part.tensors.keys().map(|name| name.as_str()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.parts
            .iter()
            .any(|part| part.tensors.contains_key(name))
    }

    /// Copy an F32 tensor whose stored name and shape match the graph input.
    pub fn f32_exact(&self, name: &str, ne: &[i64]) -> Result<Vec<f32>, WeightError> {
        let (part, loc) = self.locate(name, ne)?;
        if loc.ggml_type != g::ggml_type_GGML_TYPE_F32 {
            return Err(WeightError::Type {
                name: name.to_string(),
                ggml_type: loc.ggml_type,
            });
        }
        let raw = part.raw(name, loc)?;
        decode_f32(name, &part.origin, raw)
    }

    /// F32, F16, or BF16 widened to F32. Other stored types fail.
    /// Safetensors BF16 and F16 stay packed on disk until this read.
    pub fn f32_widened(&self, name: &str, ne: &[i64]) -> Result<Vec<f32>, WeightError> {
        let (part, loc) = self.locate(name, ne)?;
        let raw = part.raw(name, loc)?;
        if loc.ggml_type == g::ggml_type_GGML_TYPE_F32 {
            return decode_f32(name, &part.origin, raw);
        }
        if loc.ggml_type == g::ggml_type_GGML_TYPE_F16 {
            return decode_pairs(name, &part.origin, raw, f16_to_f32);
        }
        if loc.ggml_type == g::ggml_type_GGML_TYPE_BF16 {
            return decode_pairs(name, &part.origin, raw, bf16_to_f32);
        }
        Err(WeightError::NotBuilt(format!(
            "{name} is stored as ggml type {}. This read accepts F32, F16, and BF16",
            loc.ggml_type
        )))
    }

    pub fn shape(&self, name: &str) -> Result<[i64; 4], WeightError> {
        Ok(self.lookup(name)?.1.ne)
    }

    pub fn ggml_type_of(&self, name: &str) -> Result<u32, WeightError> {
        Ok(self.lookup(name)?.1.ggml_type)
    }

    /// Bytes as stored. The ggml type is not converted.
    pub fn raw_exact(&self, name: &str, ne: &[i64]) -> Result<Vec<u8>, WeightError> {
        let (part, loc) = self.lookup(name)?;
        if !shape_matches(&loc.ne, ne) {
            return Err(WeightError::Shape {
                name: name.to_string(),
                expected: ne.to_vec(),
                got: loc.ne.to_vec(),
            });
        }
        let raw = part
            .map
            .bytes(loc.offset, loc.nbytes)
            .map_err(|detail| WeightError::Open {
                path: part.origin.clone(),
                detail,
            })?;
        Ok(raw.to_vec())
    }

    fn locate(&self, name: &str, ne: &[i64]) -> Result<(&Part, &TensorLoc), WeightError> {
        let (part, loc) = self.lookup(name)?;
        if !shape_matches(&loc.ne, ne) {
            return Err(WeightError::Shape {
                name: name.to_string(),
                expected: ne.to_vec(),
                got: loc.ne.to_vec(),
            });
        }
        Ok((part, loc))
    }

    fn lookup(&self, name: &str) -> Result<(&Part, &TensorLoc), WeightError> {
        for part in &self.parts {
            if let Some(loc) = part.tensors.get(name) {
                return Ok((part, loc));
            }
        }
        let path = self
            .parts
            .first()
            .map(|part| part.origin.clone())
            .unwrap_or_default();
        Err(WeightError::MissingTensor {
            path,
            name: name.to_string(),
        })
    }
}

impl Part {
    fn open(path: &Path) -> Result<Self, WeightError> {
        let origin = path.display().to_string();
        let ext = path.extension().and_then(|ext| ext.to_str());
        if ext != Some("gguf") && ext != Some("safetensors") {
            return Err(WeightError::NotGguf { path: origin });
        }
        let map = MappedFile::open(path).map_err(|detail| WeightError::Open {
            path: origin.clone(),
            detail,
        })?;
        let tensors = if ext == Some("gguf") {
            let c_path = CString::new(origin.as_str()).map_err(|_| WeightError::Open {
                path: origin.clone(),
                detail: "path is not a C string".into(),
            })?;
            unsafe { read_index(c_path.as_ptr(), &origin, map.len)? }
        } else {
            read_safetensors(&map, &origin)?
        };
        Ok(Self {
            origin,
            map,
            tensors,
        })
    }

    fn raw<'a>(&'a self, name: &str, loc: &TensorLoc) -> Result<&'a [u8], WeightError> {
        self.map
            .bytes(loc.offset, loc.nbytes)
            .map_err(|detail| WeightError::Open {
                path: self.origin.clone(),
                detail: format!("{name}: {detail}"),
            })
    }
}

/// I64 safetensors scalars such as `bn.num_batches_tracked` are indexed and never widened.
const SAFETENSORS_I64: u32 = 0x5349_3634;

fn read_safetensors(
    map: &MappedFile,
    origin: &str,
) -> Result<BTreeMap<String, TensorLoc>, WeightError> {
    if map.len < 8 {
        return Err(WeightError::Open {
            path: origin.to_string(),
            detail: "safetensors header is shorter than 8 bytes".into(),
        });
    }
    let len_bytes = map.bytes(0, 8).map_err(|detail| WeightError::Open {
        path: origin.to_string(),
        detail,
    })?;
    let len_bytes: [u8; 8] = len_bytes.try_into().map_err(|_| WeightError::Open {
        path: origin.to_string(),
        detail: "safetensors header is shorter than 8 bytes".into(),
    })?;
    let n = u64::from_le_bytes(len_bytes) as usize;
    let header_end = 8usize.checked_add(n).ok_or_else(|| WeightError::Open {
        path: origin.to_string(),
        detail: "safetensors header length overflows".into(),
    })?;
    if header_end > map.len {
        return Err(WeightError::Open {
            path: origin.to_string(),
            detail: "safetensors header extends past the end of the file".into(),
        });
    }
    let header = map.bytes(8, n).map_err(|detail| WeightError::Open {
        path: origin.to_string(),
        detail,
    })?;
    let header = std::str::from_utf8(header).map_err(|_| WeightError::Open {
        path: origin.to_string(),
        detail: "safetensors header is not utf-8".into(),
    })?;
    let value: serde_json::Value =
        serde_json::from_str(header.trim_end()).map_err(|err| WeightError::Open {
            path: origin.to_string(),
            detail: format!("safetensors header is not json: {err}"),
        })?;
    let obj = value.as_object().ok_or_else(|| WeightError::Open {
        path: origin.to_string(),
        detail: "safetensors header is not an object".into(),
    })?;
    let mut tensors = BTreeMap::new();
    for (name, meta) in obj {
        if name.starts_with("__") {
            continue;
        }
        let dtype = meta
            .get("dtype")
            .and_then(|item| item.as_str())
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} has no dtype"),
            })?;
        let shape_v = meta
            .get("shape")
            .and_then(|item| item.as_array())
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} has no shape"),
            })?;
        let offsets = meta
            .get("data_offsets")
            .and_then(|item| item.as_array())
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} has no data_offsets"),
            })?;
        if offsets.len() != 2 {
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} data_offsets needs two numbers"),
            });
        }
        let start = json_usize(&offsets[0]).ok_or_else(|| WeightError::Open {
            path: origin.to_string(),
            detail: format!("{name} data_offsets start is not a byte offset"),
        })?;
        let end = json_usize(&offsets[1]).ok_or_else(|| WeightError::Open {
            path: origin.to_string(),
            detail: format!("{name} data_offsets end is not a byte offset"),
        })?;
        if end < start {
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} data_offsets end is before the start"),
            });
        }
        let nbytes = end - start;
        let offset = header_end
            .checked_add(start)
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} offset overflows"),
            })?;
        let file_end = offset
            .checked_add(nbytes)
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} extent overflows"),
            })?;
        if file_end > map.len {
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} extends past the end of the file"),
            });
        }
        if shape_v.len() > 4 {
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} has more than 4 dimensions"),
            });
        }
        let mut dims = Vec::with_capacity(shape_v.len());
        for dim in shape_v {
            let dim = dim.as_u64().ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} shape is not a list of lengths"),
            })?;
            if dim == 0 || dim > i64::MAX as u64 {
                return Err(WeightError::Open {
                    path: origin.to_string(),
                    detail: format!("{name} dimension {dim} is not usable"),
                });
            }
            dims.push(dim as i64);
        }
        let ggml_type = match dtype {
            "F32" => g::ggml_type_GGML_TYPE_F32,
            "F16" => g::ggml_type_GGML_TYPE_F16,
            "BF16" => g::ggml_type_GGML_TYPE_BF16,
            "I64" => SAFETENSORS_I64,
            other => {
                return Err(WeightError::Open {
                    path: origin.to_string(),
                    detail: format!("{name} dtype {other} is not loaded"),
                })
            }
        };
        let elem = match dtype {
            "F32" => 4usize,
            "F16" | "BF16" => 2,
            "I64" => 8,
            _ => unreachable!(),
        };
        let elems = dims
            .iter()
            .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} shape overflows"),
            })?;
        let expect = (elems as usize)
            .checked_mul(elem)
            .ok_or_else(|| WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} byte length overflows"),
            })?;
        if nbytes != expect {
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!(
                    "{name} has {nbytes} bytes, {dtype} {:?} needs {expect}",
                    dims
                ),
            });
        }
        let mut ne = [1i64; 4];
        for (index, dim) in dims.iter().rev().enumerate() {
            ne[index] = *dim;
        }
        if tensors
            .insert(
                name.clone(),
                TensorLoc {
                    ggml_type,
                    ne,
                    offset,
                    nbytes,
                },
            )
            .is_some()
        {
            return Err(WeightError::Duplicate {
                name: name.clone(),
                path: origin.to_string(),
            });
        }
    }
    Ok(tensors)
}

fn json_usize(value: &serde_json::Value) -> Option<usize> {
    value.as_u64().and_then(|item| usize::try_from(item).ok())
}

fn decode_f32(name: &str, origin: &str, raw: &[u8]) -> Result<Vec<f32>, WeightError> {
    if raw.len() % 4 != 0 {
        return Err(WeightError::Open {
            path: origin.to_string(),
            detail: format!("{name} byte length {} is not a multiple of 4", raw.len()),
        });
    }
    let mut values = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn decode_pairs(
    name: &str,
    origin: &str,
    raw: &[u8],
    convert: fn(u16) -> f32,
) -> Result<Vec<f32>, WeightError> {
    if raw.len() % 2 != 0 {
        return Err(WeightError::Open {
            path: origin.to_string(),
            detail: format!("{name} byte length {} is not a multiple of 2", raw.len()),
        });
    }
    let mut values = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        values.push(convert(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(values)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            sign | (e << 23) | (f << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

unsafe fn read_index(
    path: *const i8,
    origin: &str,
    file_len: usize,
) -> Result<BTreeMap<String, TensorLoc>, WeightError> {
    let params = g::gguf_init_params {
        no_alloc: true,
        ctx: std::ptr::null_mut(),
    };
    let gctx = g::gguf_init_from_file(path, params);
    if gctx.is_null() {
        return Err(WeightError::Open {
            path: origin.to_string(),
            detail: "gguf_init_from_file failed".into(),
        });
    }
    let n = g::gguf_get_n_tensors(gctx);
    let data_off = g::gguf_get_data_offset(gctx) as usize;
    let mut tensors = BTreeMap::new();
    for i in 0..n {
        let name_ptr = g::gguf_get_tensor_name(gctx, i);
        if name_ptr.is_null() {
            g::gguf_free(gctx);
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!("tensor {i} has no name"),
            });
        }
        let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
        let ne_ptr = g::gguf_get_tensor_ne(gctx, i);
        let mut ne = [1i64; 4];
        if !ne_ptr.is_null() {
            for dim in 0..4 {
                ne[dim] = *ne_ptr.add(dim);
            }
        }
        let ggml_type = g::gguf_get_tensor_type(gctx, i);
        let nbytes = g::gguf_get_tensor_size(gctx, i);
        let tensor_off = g::gguf_get_tensor_offset(gctx, i) as usize;
        let offset = data_off.saturating_add(tensor_off);
        let end = offset.saturating_add(nbytes);
        if end > file_len {
            g::gguf_free(gctx);
            return Err(WeightError::Open {
                path: origin.to_string(),
                detail: format!("{name} extends past the end of the file"),
            });
        }
        if tensors
            .insert(
                name.clone(),
                TensorLoc {
                    ggml_type,
                    ne,
                    offset,
                    nbytes,
                },
            )
            .is_some()
        {
            g::gguf_free(gctx);
            return Err(WeightError::Duplicate {
                name,
                path: origin.to_string(),
            });
        }
    }
    g::gguf_free(gctx);
    Ok(tensors)
}

fn shape_matches(got: &[i64; 4], expected: &[i64]) -> bool {
    if expected.len() > 4 {
        return false;
    }
    for dim in 0..4 {
        let want = expected.get(dim).copied().unwrap_or(1);
        if got[dim] != want {
            return false;
        }
    }
    true
}

pub struct F32Tensor<'a> {
    pub name: &'a str,
    pub ne0: i64,
    pub ne1: i64,
    pub data: &'a [f32],
}

pub fn write_f32_gguf(
    path: &Path,
    name: &str,
    ne0: i64,
    ne1: i64,
    data: &[f32],
) -> Result<(), WeightError> {
    write_f32_tensors(
        path,
        &[F32Tensor {
            name,
            ne0,
            ne1,
            data,
        }],
    )
}

pub struct TypedTensor<'a> {
    pub name: &'a str,
    pub ne0: i64,
    pub ne1: i64,
    pub ggml_type: u32,
    pub data: &'a [u8],
}

pub fn matmul_type(ggml_type: u32) -> bool {
    // Q4_K_M stores some rows as Q5_K. Both are ggml matmul types.
    matches!(
        ggml_type,
        g::ggml_type_GGML_TYPE_F32
            | g::ggml_type_GGML_TYPE_F16
            | g::ggml_type_GGML_TYPE_BF16
            | g::ggml_type_GGML_TYPE_Q8_0
            | g::ggml_type_GGML_TYPE_Q4_K
            | g::ggml_type_GGML_TYPE_Q5_K
    )
}

pub fn type_block(ggml_type: u32) -> i64 {
    unsafe { g::ggml_blck_size(ggml_type) as i64 }
}

/// Quantize F32 rows to Q8_0. `ne0` is the contiguous axis and must be a multiple of 32.
pub fn quantize_q8_0(ne0: i64, ne1: i64, values: &[f32]) -> Result<Vec<u8>, WeightError> {
    let expect = (ne0 as usize).saturating_mul(ne1 as usize);
    if values.len() != expect || ne0 <= 0 || ne1 <= 0 || ne0 % 32 != 0 {
        return Err(WeightError::Shape {
            name: "q8_0".into(),
            expected: vec![ne0, ne1],
            got: vec![values.len() as i64],
        });
    }
    let row = unsafe { g::ggml_row_size(g::ggml_type_GGML_TYPE_Q8_0, ne0) };
    let mut dst = vec![0u8; row.saturating_mul(ne1 as usize)];
    let wrote = unsafe {
        g::ggml_quantize_chunk(
            g::ggml_type_GGML_TYPE_Q8_0,
            values.as_ptr(),
            dst.as_mut_ptr().cast(),
            0,
            ne1,
            ne0,
            std::ptr::null(),
        )
    };
    if wrote == 0 {
        return Err(WeightError::Open {
            path: "q8_0".into(),
            detail: "ggml_quantize_chunk wrote nothing".into(),
        });
    }
    Ok(dst)
}

pub fn write_typed_tensors(path: &Path, tensors: &[TypedTensor<'_>]) -> Result<(), WeightError> {
    const MAX_NAME: usize = 256;
    if tensors.is_empty() {
        return Err(WeightError::Open {
            path: path.display().to_string(),
            detail: "no tensors to write".into(),
        });
    }
    let mut bytes = 0usize;
    for tensor in tensors {
        if tensor.name.len() >= MAX_NAME {
            return Err(WeightError::NameTooLong {
                name: tensor.name.to_string(),
                max: MAX_NAME - 1,
            });
        }
        if tensor.ne0 <= 0 || tensor.ne1 <= 0 {
            return Err(WeightError::Shape {
                name: tensor.name.to_string(),
                expected: vec![tensor.ne0, tensor.ne1],
                got: vec![tensor.data.len() as i64],
            });
        }
        let row = unsafe { g::ggml_row_size(tensor.ggml_type, tensor.ne0) };
        let expect = row.saturating_mul(tensor.ne1 as usize);
        if tensor.data.len() != expect {
            return Err(WeightError::Shape {
                name: tensor.name.to_string(),
                expected: vec![tensor.ne0, tensor.ne1, expect as i64],
                got: vec![tensor.data.len() as i64],
            });
        }
        bytes = bytes.saturating_add(expect);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| WeightError::Open {
            path: path.display().to_string(),
            detail: err.to_string(),
        })?;
    }
    let path_string = path.display().to_string();
    let c_path = CString::new(path_string.as_str()).map_err(|_| WeightError::Open {
        path: path_string,
        detail: "path is not a C string".into(),
    })?;
    unsafe {
        let params = g::ggml_init_params {
            mem_size: bytes.saturating_add(16 * 1024 * 1024),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        };
        let ctx = g::ggml_init(params);
        if ctx.is_null() {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "ggml_init failed".into(),
            });
        }
        let gctx = g::gguf_init_empty();
        if gctx.is_null() {
            g::ggml_free(ctx);
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "gguf_init_empty failed".into(),
            });
        }
        for tensor in tensors {
            let c_name = match CString::new(tensor.name) {
                Ok(name) => name,
                Err(_) => {
                    g::gguf_free(gctx);
                    g::ggml_free(ctx);
                    return Err(WeightError::Open {
                        path: path.display().to_string(),
                        detail: "tensor name is not a C string".into(),
                    });
                }
            };
            let created = g::ggml_new_tensor_2d(ctx, tensor.ggml_type, tensor.ne0, tensor.ne1);
            if created.is_null() {
                g::gguf_free(gctx);
                g::ggml_free(ctx);
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("ggml_new_tensor_2d failed for {}", tensor.name),
                });
            }
            g::ggml_set_name(created, c_name.as_ptr());
            let dst = g::ggml_get_data(created);
            if dst.is_null() {
                g::gguf_free(gctx);
                g::ggml_free(ctx);
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("{} has no data", tensor.name),
                });
            }
            std::ptr::copy_nonoverlapping(tensor.data.as_ptr(), dst.cast(), tensor.data.len());
            g::gguf_add_tensor(gctx, created);
        }
        let ok = g::gguf_write_to_file(gctx, c_path.as_ptr(), false);
        g::gguf_free(gctx);
        g::ggml_free(ctx);
        if !ok {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "gguf_write_to_file failed".into(),
            });
        }
    }
    Ok(())
}

pub fn write_f32_tensors(path: &Path, tensors: &[F32Tensor<'_>]) -> Result<(), WeightError> {
    // Kept in sync with -DGGML_MAX_NAME=256 in danmo-ggml-sys/build.rs.
    const MAX_NAME: usize = 256;
    if tensors.is_empty() {
        return Err(WeightError::Open {
            path: path.display().to_string(),
            detail: "no tensors to write".into(),
        });
    }
    let mut bytes = 0usize;
    for tensor in tensors {
        if tensor.name.len() >= MAX_NAME {
            return Err(WeightError::NameTooLong {
                name: tensor.name.to_string(),
                max: MAX_NAME - 1,
            });
        }
        let expect = (tensor.ne0 as usize).saturating_mul(tensor.ne1 as usize);
        if tensor.data.len() != expect || tensor.ne0 <= 0 || tensor.ne1 <= 0 {
            return Err(WeightError::Shape {
                name: tensor.name.to_string(),
                expected: vec![tensor.ne0, tensor.ne1],
                got: vec![tensor.data.len() as i64],
            });
        }
        bytes = bytes.saturating_add(tensor.data.len() * 4);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| WeightError::Open {
            path: path.display().to_string(),
            detail: err.to_string(),
        })?;
    }
    let path_string = path.display().to_string();
    let c_path = CString::new(path_string.as_str()).map_err(|_| WeightError::Open {
        path: path_string,
        detail: "path is not a C string".into(),
    })?;
    unsafe {
        let params = g::ggml_init_params {
            mem_size: bytes.saturating_add(16 * 1024 * 1024),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        };
        let ctx = g::ggml_init(params);
        if ctx.is_null() {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "ggml_init failed".into(),
            });
        }
        let gctx = g::gguf_init_empty();
        if gctx.is_null() {
            g::ggml_free(ctx);
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "gguf_init_empty failed".into(),
            });
        }
        for tensor in tensors {
            let c_name = match CString::new(tensor.name) {
                Ok(name) => name,
                Err(_) => {
                    g::gguf_free(gctx);
                    g::ggml_free(ctx);
                    return Err(WeightError::Open {
                        path: path.display().to_string(),
                        detail: "tensor name is not a C string".into(),
                    });
                }
            };
            let created =
                g::ggml_new_tensor_2d(ctx, g::ggml_type_GGML_TYPE_F32, tensor.ne0, tensor.ne1);
            if created.is_null() {
                g::gguf_free(gctx);
                g::ggml_free(ctx);
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("ggml_new_tensor_2d failed for {}", tensor.name),
                });
            }
            g::ggml_set_name(created, c_name.as_ptr());
            let dst = g::ggml_get_data_f32(created);
            if dst.is_null() {
                g::gguf_free(gctx);
                g::ggml_free(ctx);
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("{} has no data", tensor.name),
                });
            }
            std::ptr::copy_nonoverlapping(tensor.data.as_ptr(), dst, tensor.data.len());
            g::gguf_add_tensor(gctx, created);
        }
        let ok = g::gguf_write_to_file(gctx, c_path.as_ptr(), false);
        g::gguf_free(gctx);
        g::ggml_free(ctx);
        if !ok {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "gguf_write_to_file failed".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
pub struct F32Nd<'a> {
    pub name: &'a str,
    pub ne: [i64; 4],
    pub data: &'a [f32],
}

/// Write F32 tensors with an explicit ggml shape. `ne[0]` is the fastest dimension.
#[cfg(test)]
pub fn write_f32_nd(path: &Path, tensors: &[F32Nd<'_>]) -> Result<(), WeightError> {
    const MAX_NAME: usize = 256;
    if tensors.is_empty() {
        return Err(WeightError::Open {
            path: path.display().to_string(),
            detail: "no tensors to write".into(),
        });
    }
    let mut bytes = 0usize;
    for tensor in tensors {
        if tensor.name.len() >= MAX_NAME {
            return Err(WeightError::NameTooLong {
                name: tensor.name.to_string(),
                max: MAX_NAME - 1,
            });
        }
        if tensor.ne.iter().any(|dim| *dim <= 0) {
            return Err(WeightError::Shape {
                name: tensor.name.to_string(),
                expected: tensor.ne.to_vec(),
                got: vec![tensor.data.len() as i64],
            });
        }
        let expect = (tensor.ne[0] as usize)
            .checked_mul(tensor.ne[1] as usize)
            .and_then(|n| n.checked_mul(tensor.ne[2] as usize))
            .and_then(|n| n.checked_mul(tensor.ne[3] as usize));
        if expect != Some(tensor.data.len()) {
            return Err(WeightError::Shape {
                name: tensor.name.to_string(),
                expected: tensor.ne.to_vec(),
                got: vec![tensor.data.len() as i64],
            });
        }
        bytes = bytes.saturating_add(tensor.data.len() * 4);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| WeightError::Open {
            path: path.display().to_string(),
            detail: err.to_string(),
        })?;
    }
    let path_string = path.display().to_string();
    let c_path = CString::new(path_string.as_str()).map_err(|_| WeightError::Open {
        path: path_string,
        detail: "path is not a C string".into(),
    })?;
    unsafe {
        let params = g::ggml_init_params {
            mem_size: bytes.saturating_add(16 * 1024 * 1024),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        };
        let ctx = g::ggml_init(params);
        if ctx.is_null() {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "ggml_init failed".into(),
            });
        }
        let gctx = g::gguf_init_empty();
        if gctx.is_null() {
            g::ggml_free(ctx);
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "gguf_init_empty failed".into(),
            });
        }
        for tensor in tensors {
            let c_name = match CString::new(tensor.name) {
                Ok(name) => name,
                Err(_) => {
                    g::gguf_free(gctx);
                    g::ggml_free(ctx);
                    return Err(WeightError::Open {
                        path: path.display().to_string(),
                        detail: "tensor name is not a C string".into(),
                    });
                }
            };
            let created = g::ggml_new_tensor_4d(
                ctx,
                g::ggml_type_GGML_TYPE_F32,
                tensor.ne[0],
                tensor.ne[1],
                tensor.ne[2],
                tensor.ne[3],
            );
            if created.is_null() {
                g::gguf_free(gctx);
                g::ggml_free(ctx);
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("ggml_new_tensor_4d failed for {}", tensor.name),
                });
            }
            g::ggml_set_name(created, c_name.as_ptr());
            let dst = g::ggml_get_data_f32(created);
            if dst.is_null() {
                g::gguf_free(gctx);
                g::ggml_free(ctx);
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("{} has no data", tensor.name),
                });
            }
            std::ptr::copy_nonoverlapping(tensor.data.as_ptr(), dst, tensor.data.len());
            g::gguf_add_tensor(gctx, created);
        }
        let ok = g::gguf_write_to_file(gctx, c_path.as_ptr(), false);
        g::gguf_free(gctx);
        g::ggml_free(ctx);
        if !ok {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "gguf_write_to_file failed".into(),
            });
        }
    }
    Ok(())
}

struct MappedFile {
    ptr: *mut u8,
    len: usize,
    #[cfg(windows)]
    mapping: isize,
}

unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|err| err.to_string())?;
        let len = file.metadata().map_err(|err| err.to_string())?.len() as usize;
        if len == 0 {
            return Err("file is empty".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            const PROT_READ: i32 = 1;
            const MAP_PRIVATE: i32 = 2;
            extern "C" {
                fn mmap(
                    addr: *mut u8,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: i64,
                ) -> *mut u8;
            }
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr.is_null() || ptr == usize::MAX as *mut u8 {
                return Err("mmap failed".into());
            }
            Ok(Self { ptr, len })
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            const PAGE_READONLY: u32 = 0x02;
            const FILE_MAP_READ: u32 = 0x0004;
            extern "system" {
                fn CreateFileMappingW(
                    hFile: isize,
                    lpAttr: *mut u8,
                    flProtect: u32,
                    dwMaxHigh: u32,
                    dwMaxLow: u32,
                    name: *const u16,
                ) -> isize;
                fn MapViewOfFile(
                    hMap: isize,
                    access: u32,
                    offHigh: u32,
                    offLow: u32,
                    bytes: usize,
                ) -> *mut u8;
                fn CloseHandle(h: isize) -> i32;
            }
            let handle = file.as_raw_handle() as isize;
            let mapping = unsafe {
                CreateFileMappingW(
                    handle,
                    std::ptr::null_mut(),
                    PAGE_READONLY,
                    0,
                    0,
                    std::ptr::null(),
                )
            };
            if mapping == 0 {
                return Err("CreateFileMappingW failed".into());
            }
            let ptr = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
            if ptr.is_null() {
                unsafe { CloseHandle(mapping) };
                return Err("MapViewOfFile failed".into());
            }
            Ok(Self { ptr, len, mapping })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = file;
            Err("mmap is implemented for unix and windows".into())
        }
    }

    fn bytes(&self, offset: usize, len: usize) -> Result<&[u8], String> {
        let end = offset.checked_add(len).ok_or("offset overflow")?;
        if end > self.len {
            return Err(format!("read {offset}+{len} past {}", self.len));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) })
    }
}

#[cfg(test)]
pub struct SafeTensor<'a> {
    pub name: &'a str,
    pub dtype: &'a str,
    pub shape: &'a [u64],
    pub data: &'a [u8],
}

/// Write a safetensors file. `shape` is PyTorch order. `data` is that order's contiguous bytes.
#[cfg(test)]
pub fn write_safetensors(path: &Path, tensors: &[SafeTensor<'_>]) -> Result<(), WeightError> {
    if tensors.is_empty() {
        return Err(WeightError::Open {
            path: path.display().to_string(),
            detail: "no tensors to write".into(),
        });
    }
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let elem = match tensor.dtype {
            "F32" => 4usize,
            "F16" | "BF16" => 2,
            "I64" => 8,
            other => {
                return Err(WeightError::Open {
                    path: path.display().to_string(),
                    detail: format!("{} dtype {other} is not written", tensor.name),
                })
            }
        };
        let elems = tensor
            .shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim as usize))
            .ok_or_else(|| WeightError::Open {
                path: path.display().to_string(),
                detail: format!("{} shape overflows", tensor.name),
            })?;
        let expect = elems.saturating_mul(elem);
        if tensor.shape.is_empty() {
            if tensor.data.len() != elem {
                return Err(WeightError::Shape {
                    name: tensor.name.to_string(),
                    expected: vec![elem as i64],
                    got: vec![tensor.data.len() as i64],
                });
            }
        } else if tensor.data.len() != expect {
            return Err(WeightError::Shape {
                name: tensor.name.to_string(),
                expected: tensor.shape.iter().map(|dim| *dim as i64).collect(),
                got: vec![tensor.data.len() as i64],
            });
        }
        let start = align8(data.len());
        data.resize(start, 0);
        let end = start + tensor.data.len();
        data.extend_from_slice(tensor.data);
        header.insert(
            tensor.name.to_string(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let mut json = serde_json::to_vec(&header).map_err(|err| WeightError::Open {
        path: path.display().to_string(),
        detail: err.to_string(),
    })?;
    while (8 + json.len()) % 8 != 0 {
        json.push(b' ');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| WeightError::Open {
            path: path.display().to_string(),
            detail: err.to_string(),
        })?;
    }
    let mut file = Vec::with_capacity(8 + json.len() + data.len());
    file.extend_from_slice(&(json.len() as u64).to_le_bytes());
    file.extend_from_slice(&json);
    file.extend_from_slice(&data);
    std::fs::write(path, file).map_err(|err| WeightError::Open {
        path: path.display().to_string(),
        detail: err.to_string(),
    })
}

#[cfg(test)]
fn align8(n: usize) -> usize {
    n.saturating_add(7) & !7
}

#[cfg(test)]
mod tests {
    use super::{bf16_to_f32, write_safetensors, GgufWeights, SafeTensor};
    use crate::weights::WeightError;

    #[test]
    fn safetensors_reverses_pytorch_dims_and_widens_bf16() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-st-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let f32_values = [1.5f32, -2.0, 0.5, 3.0];
        let mut f32_bytes = Vec::new();
        for value in f32_values {
            f32_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let bf16_values = [1.0f32, -2.0, 0.5, 4.0];
        let mut bf16_bytes = Vec::new();
        for value in bf16_values {
            let bits = (value.to_bits() >> 16) as u16;
            assert_eq!(bf16_to_f32(bits), value);
            bf16_bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let i64_bytes = 7i64.to_le_bytes();
        write_safetensors(
            &path,
            &[
                SafeTensor {
                    name: "model.embed_tokens.weight",
                    dtype: "F32",
                    shape: &[4, 1],
                    data: &f32_bytes,
                },
                SafeTensor {
                    name: "decoder.conv_in.weight",
                    dtype: "BF16",
                    shape: &[2, 2, 1, 1],
                    data: &bf16_bytes,
                },
                SafeTensor {
                    name: "bn.num_batches_tracked",
                    dtype: "I64",
                    shape: &[],
                    data: &i64_bytes,
                },
            ],
        )
        .unwrap();
        let weights = GgufWeights::open(&path).unwrap();
        assert!(weights.contains("model.embed_tokens.weight"));
        let embed = weights
            .f32_exact("model.embed_tokens.weight", &[1, 4])
            .unwrap();
        assert_eq!(embed, f32_values);
        let err = weights
            .f32_exact("decoder.conv_in.weight", &[1, 1, 2, 2])
            .unwrap_err();
        assert!(err.to_string().contains("ggml type"), "{err}");
        let widened = weights
            .f32_widened("decoder.conv_in.weight", &[1, 1, 2, 2])
            .unwrap();
        assert_eq!(widened, bf16_values);
        let err = weights
            .f32_widened("bn.num_batches_tracked", &[1])
            .unwrap_err();
        assert!(err.to_string().contains("F32, F16, and BF16"), "{err}");
        let garbage = dir.join("w.safetensors");
        std::fs::write(&garbage, b"not-gguf").unwrap();
        let err = match GgufWeights::open(&garbage) {
            Err(WeightError::Open { detail, .. }) => detail,
            Err(err) => panic!("garbage safetensors returned {err}"),
            Ok(_) => panic!("garbage safetensors was accepted"),
        };
        assert!(err.contains("safetensors"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.len == 0 {
            return;
        }
        #[cfg(unix)]
        unsafe {
            extern "C" {
                fn munmap(addr: *mut u8, len: usize) -> i32;
            }
            munmap(self.ptr, self.len);
        }
        #[cfg(windows)]
        unsafe {
            extern "system" {
                fn UnmapViewOfFile(addr: *const u8) -> i32;
                fn CloseHandle(h: isize) -> i32;
            }
            UnmapViewOfFile(self.ptr);
            CloseHandle(self.mapping);
        }
    }
}
