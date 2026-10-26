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
        let (part, loc) = self.lookup(name)?;
        if !shape_matches(&loc.ne, ne) {
            return Err(WeightError::Shape {
                name: name.to_string(),
                expected: ne.to_vec(),
                got: loc.ne.to_vec(),
            });
        }
        if loc.ggml_type != g::ggml_type_GGML_TYPE_F32 {
            return Err(WeightError::Type {
                name: name.to_string(),
                ggml_type: loc.ggml_type,
            });
        }
        let raw = part
            .map
            .bytes(loc.offset, loc.nbytes)
            .map_err(|detail| WeightError::Open {
                path: part.origin.clone(),
                detail,
            })?;
        if raw.len() % 4 != 0 {
            return Err(WeightError::Open {
                path: part.origin.clone(),
                detail: format!("{name} byte length {} is not a multiple of 4", raw.len()),
            });
        }
        let mut values = Vec::with_capacity(raw.len() / 4);
        for chunk in raw.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }

    pub fn shape(&self, name: &str) -> Result<[i64; 4], WeightError> {
        Ok(self.lookup(name)?.1.ne)
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
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            return Err(WeightError::NotGguf { path: origin });
        }
        let map = MappedFile::open(path).map_err(|detail| WeightError::Open {
            path: origin.clone(),
            detail,
        })?;
        let c_path = CString::new(origin.as_str()).map_err(|_| WeightError::Open {
            path: origin.clone(),
            detail: "path is not a C string".into(),
        })?;
        let tensors = unsafe { read_index(c_path.as_ptr(), &origin, map.len)? };
        Ok(Self {
            origin,
            map,
            tensors,
        })
    }
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
