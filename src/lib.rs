pub mod parakeet;
pub mod moonshine;
pub mod silero;

#[cfg(feature = "fast-cpu")]
pub mod fast_matmul;
#[cfg(feature = "fast-cpu")]
pub mod fast_ops;

mod napi;
