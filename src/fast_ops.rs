//! Fast contiguous f32 ops bypassing candle's generic dispatch.
//!
//! Provides `fast_add` for residual connections — skips type checking and
//! layout validation overhead on the hot path.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};

struct FastAddOp;

impl CustomOp2 for FastAddOp {
    fn name(&self) -> &'static str {
        "fast-add"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("fast_add only supports f32")
        }
        let a = s1.as_slice::<f32>()?;
        let b = s2.as_slice::<f32>()?;
        let n = l1.shape().elem_count();
        let a = &a[l1.start_offset()..l1.start_offset() + n];
        let b = &b[l2.start_offset()..l2.start_offset() + n];
        let mut dst = vec![0f32; n];
        for i in 0..n {
            dst[i] = a[i] + b[i];
        }
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }
}

/// Element-wise add bypassing candle's generic binary op dispatch.
/// Falls back to standard add on Metal/GPU where custom ops have overhead.
pub fn fast_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !matches!(a.device(), Device::Cpu) {
        a + b
    } else {
        a.apply_op2_no_bwd(b, &FastAddOp)
    }
}
