pub mod parakeet;
pub mod moonshine;
pub mod kokoro;
pub mod silero;
pub mod streaming;

#[cfg(feature = "fast-cpu")]
pub mod fast_matmul;
#[cfg(feature = "fast-cpu")]
pub mod fast_ops;
#[cfg(feature = "triton-metal")]
pub mod triton_kernels;
#[cfg(feature = "triton-d3d12")]
pub mod triton_d3d12_kernels;
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
pub mod kernel_archive;

mod napi;
