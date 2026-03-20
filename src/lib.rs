pub mod parakeet;
pub mod moonshine;
pub mod silero;
pub mod streaming;

#[cfg(feature = "fast-cpu")]
pub mod fast_matmul;
#[cfg(feature = "fast-cpu")]
pub mod fast_ops;
#[cfg(feature = "triton-metal")]
pub mod triton_kernels;

mod napi;
