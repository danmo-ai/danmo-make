use danmo_mlx_sys as sys;

use crate::array::{Array, Dtype};
use crate::error::{check, clear_error, Result};
use crate::stream::Stream;

pub struct RandomKey(Array);

impl RandomKey {
    pub fn new(seed: u64) -> Result<Self> {
        let mut output = Array::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_random_key(&mut output.raw, seed) },
            "mlx_random_key",
        )?;
        Ok(Self(output))
    }

    pub fn split(&self, stream: &Stream) -> Result<(Self, Self)> {
        let mut left = Array::empty()?;
        let mut right = Array::empty()?;
        clear_error();
        check(
            unsafe { sys::mlx_random_split(&mut left.raw, &mut right.raw, self.0.raw, stream.raw) },
            "mlx_random_split",
        )?;
        Ok((Self(left), Self(right)))
    }

    pub fn normal(&self, shape: &[i32], dtype: Dtype, stream: &Stream) -> Result<Array> {
        if dtype == Dtype::Int32 {
            return Err(crate::Error::new(
                "mlx_random_normal",
                "normal random arrays require Float32 or BFloat16",
            ));
        }
        let mut output = Array::empty()?;
        clear_error();
        check(
            unsafe {
                sys::mlx_random_normal(
                    &mut output.raw,
                    shape.as_ptr(),
                    shape.len(),
                    dtype.raw(),
                    0.0,
                    1.0,
                    self.0.raw,
                    stream.raw,
                )
            },
            "mlx_random_normal",
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(stream: &Stream) {
        let key = RandomKey::new(42).unwrap();
        let expected = [
            0.612265408039093,
            1.1225886344909668,
            1.1373323202133179,
            -0.8127323985099792,
            -0.8904047012329102,
            0.12623175978660583,
        ];
        let first = key
            .normal(&[2, 3], Dtype::Float32, stream)
            .unwrap()
            .to_vec_f32(stream)
            .unwrap();
        let second = key
            .normal(&[2, 3], Dtype::Float32, stream)
            .unwrap()
            .to_vec_f32(stream)
            .unwrap();
        assert_eq!(first, expected);
        assert_eq!(second, expected);

        let bf16 = key
            .normal(&[2, 3], Dtype::BFloat16, stream)
            .unwrap()
            .to_vec_f32(stream)
            .unwrap();
        assert_eq!(
            bf16,
            [
                0.61328125,
                1.125,
                1.140625,
                -0.8125,
                -0.890625,
                0.1259765625,
            ]
        );

        let (left, right) = key.split(stream).unwrap();
        assert_eq!(
            left.normal(&[2], Dtype::Float32, stream)
                .unwrap()
                .to_vec_f32(stream)
                .unwrap(),
            [0.6491709351539612, -0.6725232601165771]
        );
        assert_eq!(
            right
                .normal(&[2], Dtype::Float32, stream)
                .unwrap()
                .to_vec_f32(stream)
                .unwrap(),
            [1.7058420181274414, -0.2448853701353073]
        );
        assert!(key.normal(&[1], Dtype::Int32, stream).is_err());
    }

    #[test]
    fn cpu_seeded_normal_and_split_match_main() {
        probe(&Stream::cpu().unwrap());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_seeded_normal_and_split_match_main() {
        probe(&Stream::gpu().unwrap());
    }
}
