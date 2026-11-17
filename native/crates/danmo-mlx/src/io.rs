use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;

use danmo_mlx_sys as sys;

use crate::error::{check, check_infallible, clear_error, initialize};
use crate::{Array, Error, Result, Stream};

pub struct SafeTensors {
    arrays: sys::mlx_map_string_to_array,
    metadata: sys::mlx_map_string_to_string,
    _thread_affine: PhantomData<Rc<()>>,
}

impl SafeTensors {
    pub fn load(path: &Path, stream: &Stream) -> Result<Self> {
        initialize();
        let path_text = path
            .to_str()
            .ok_or_else(|| Error::new("mlx_load_safetensors", "path is not valid UTF-8"))?;
        let path_c = CString::new(path_text)
            .map_err(|_| Error::new("mlx_load_safetensors", "path contains a null byte"))?;
        clear_error();
        let arrays = unsafe { sys::mlx_map_string_to_array_new() };
        let arrays = check_infallible("mlx_map_string_to_array_new", arrays)?;
        clear_error();
        let metadata = unsafe { sys::mlx_map_string_to_string_new() };
        let metadata = check_infallible("mlx_map_string_to_string_new", metadata)?;
        let mut output = Self {
            arrays,
            metadata,
            _thread_affine: PhantomData,
        };
        clear_error();
        check(
            unsafe {
                sys::mlx_load_safetensors(
                    &mut output.arrays,
                    &mut output.metadata,
                    path_c.as_ptr(),
                    stream.raw,
                )
            },
            "mlx_load_safetensors",
        )?;
        Ok(output)
    }

    pub fn get(&self, name: &str) -> Result<Option<Array>> {
        let name = CString::new(name)
            .map_err(|_| Error::new("mlx_map_string_to_array_get", "key contains a null byte"))?;
        let mut output = Array::empty()?;
        clear_error();
        let status = unsafe {
            sys::mlx_map_string_to_array_get(&mut output.raw, self.arrays, name.as_ptr())
        };
        match status {
            0 => Ok(Some(output)),
            2 => Ok(None),
            _ => {
                check(status, "mlx_map_string_to_array_get")?;
                unreachable!()
            }
        }
    }

    pub fn names(&self) -> Result<Vec<String>> {
        clear_error();
        let iterator = unsafe { sys::mlx_map_string_to_array_iterator_new(self.arrays) };
        let iterator = check_infallible("mlx_map_string_to_array_iterator_new", iterator)?;
        let mut names = Vec::new();
        loop {
            let mut key = std::ptr::null();
            let mut value = Array::empty()?;
            clear_error();
            let status = unsafe {
                sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value.raw, iterator)
            };
            match status {
                0 => {
                    if key.is_null() {
                        unsafe {
                            let _ = sys::mlx_map_string_to_array_iterator_free(iterator);
                        }
                        return Err(Error::new(
                            "mlx_map_string_to_array_iterator_next",
                            "returned a null tensor name",
                        ));
                    }
                    names.push(
                        unsafe { CStr::from_ptr(key) }
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
                2 => break,
                _ => {
                    unsafe {
                        let _ = sys::mlx_map_string_to_array_iterator_free(iterator);
                    }
                    check(status, "mlx_map_string_to_array_iterator_next")?;
                }
            }
        }
        clear_error();
        unsafe {
            let _ = sys::mlx_map_string_to_array_iterator_free(iterator);
        }
        clear_error();
        names.sort();
        Ok(names)
    }
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        clear_error();
        unsafe {
            let _ = sys::mlx_map_string_to_array_free(self.arrays);
            let _ = sys::mlx_map_string_to_string_free(self.metadata);
        }
        clear_error();
    }
}
