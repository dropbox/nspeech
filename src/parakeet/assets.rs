//! assets.rs — generic Zstd asset loader with optional compile-time embedding.

use once_cell::sync::OnceCell;
use std::path::{Path, PathBuf};

#[cfg(not(feature = "embed-assets"))]
use std::fs;

#[cfg(not(feature = "embed-assets"))]
use memmap2::Mmap;

/// A Zstandard-compressed asset that is decompressed at most once and
/// cached for the program's lifetime.
pub struct Asset {
    #[cfg(feature = "embed-assets")]
    compressed: &'static [u8],

    #[cfg(not(feature = "embed-assets"))]
    path: &'static str,

    decompressed: OnceCell<Result<Vec<u8>, ()>>,
}

impl Asset {
    #[cfg(feature = "embed-assets")]
    pub const fn new_embedded(compressed: &'static [u8]) -> Self {
        Self {
            compressed,
            decompressed: OnceCell::new(),
        }
    }

    #[cfg(not(feature = "embed-assets"))]
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            decompressed: OnceCell::new(),
        }
    }

    /// Return the decompressed bytes; performs work only on first call.
    /// Uses streaming decompression to reduce peak memory usage.
    pub fn bytes(&'static self, _assets: &PathBuf) -> Result<&'static [u8], ()> {
        self.decompressed
            .get_or_init(|| {
                // Decompress in streaming fashion to reduce memory pressure
                #[cfg(feature = "embed-assets")]
                {
                    // Decompress directly from embedded static slice (no copy)
                    match decompress_streaming(self.compressed) {
                        Ok(d) => Ok(d),
                        Err(e) => {
                            log::warn!("failed to decompress embedded asset: {}", e);
                            Err(())
                        }
                    }
                }

                #[cfg(not(feature = "embed-assets"))]
                {
                    // Read and decompress from file in streaming fashion
                    match fs::read(Path::new(&_assets.join(self.path))) {
                        Ok(compressed) => match decompress_streaming(&compressed) {
                            Ok(d) => Ok(d),
                            Err(e) => {
                                log::warn!("failed to decompress asset {}: {}", self.path, e);
                                Err(())
                            }
                        },
                        Err(e) => {
                            log::warn!("failed to read speech asset {}: {}", self.path, e);
                            Err(())
                        }
                    }
                }
            })
            .as_deref()
            .map_err(|_| ())
    }
}

/// Streaming decompression that reduces peak memory usage.
///
/// This function:
/// 1. Reads compressed data directly from slice (no intermediate copy)
/// 2. Decompresses in chunks to avoid large allocations
/// 3. Grows output buffer incrementally
///
/// Benefits over decode_all():
/// - No intermediate Vec copy of compressed data
/// - Processes in 64KB chunks instead of allocating full output upfront
/// - Lower peak memory usage for large models
fn decompress_streaming(compressed: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use std::io::Read;

    // Create streaming decoder directly from slice
    let mut decoder = zstd::stream::read::Decoder::new(compressed)?;

    // Start with reasonable initial capacity (will grow as needed)
    // 1MB initial buffer - grows exponentially if needed
    let mut decompressed = Vec::with_capacity(1024 * 1024);

    // Decompress in 64KB chunks to reduce memory pressure
    const CHUNK_SIZE: usize = 65536; // 64KB chunks
    let mut buffer = vec![0u8; CHUNK_SIZE];

    loop {
        match decoder.read(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(n) => {
                decompressed.extend_from_slice(&buffer[..n]);
            }
            Err(e) => return Err(e),
        }
    }

    // Shrink to fit to release excess capacity
    decompressed.shrink_to_fit();

    Ok(decompressed)
}

#[macro_export]
macro_rules! embed_zst_asset {
    ($vis:vis $name:ident, $path:literal) => {
        #[cfg(feature = "embed-assets")]
        $vis static $name: $crate::parakeet::assets::Asset =
            $crate::parakeet::assets::Asset::new_embedded(include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", $path)
            ));

        #[cfg(not(feature = "embed-assets"))]
        $vis static $name: $crate::parakeet::assets::Asset =
            $crate::parakeet::assets::Asset::new_file($path);
    };
}

/// An uncompressed asset that can be memory-mapped for efficient access.
///
/// For embedded builds, returns the embedded static slice directly.
/// For file-based builds, memory-maps the file for zero-copy access.
///
/// This is ideal for large GGUF model files where we want efficient access
/// without loading the entire file into memory.
pub struct UncompressedAsset {
    #[cfg(feature = "embed-assets")]
    data: &'static [u8],

    #[cfg(not(feature = "embed-assets"))]
    path: &'static str,

    #[cfg(not(feature = "embed-assets"))]
    mmap: OnceCell<Result<Mmap, ()>>,
}

impl UncompressedAsset {
    #[cfg(feature = "embed-assets")]
    pub const fn new_embedded(data: &'static [u8]) -> Self {
        Self { data }
    }

    #[cfg(not(feature = "embed-assets"))]
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            mmap: OnceCell::new(),
        }
    }

    /// Return the uncompressed bytes.
    ///
    /// For embedded builds: returns the static slice directly.
    /// For file-based builds: memory-maps the file on first access.
    pub fn bytes(&'static self, _assets: &PathBuf) -> Result<&'static [u8], ()> {
        #[cfg(feature = "embed-assets")]
        {
            Ok(self.data)
        }

        #[cfg(not(feature = "embed-assets"))]
        {
            self.mmap
                .get_or_init(|| {
                    let full_path = _assets.join(self.path);
                    match fs::File::open(&full_path) {
                        Ok(file) => {
                            // Memory-map the file for efficient access
                            match unsafe { Mmap::map(&file) } {
                                Ok(mmap) => Ok(mmap),
                                Err(e) => {
                                    log::warn!("failed to mmap asset {}: {}", self.path, e);
                                    Err(())
                                }
                            }
                        }
                        Err(e) => {
                            log::warn!("failed to open asset {}: {}", self.path, e);
                            Err(())
                        }
                    }
                })
                .as_ref()
                .map(|mmap| &mmap[..])
                .map_err(|_| ())
        }
    }
}

#[macro_export]
macro_rules! embed_asset {
    ($vis:vis $name:ident, $path:literal) => {
        #[cfg(feature = "embed-assets")]
        $vis static $name: $crate::parakeet::assets::UncompressedAsset =
            $crate::parakeet::assets::UncompressedAsset::new_embedded(include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", $path)
            ));

        #[cfg(not(feature = "embed-assets"))]
        $vis static $name: $crate::parakeet::assets::UncompressedAsset =
            $crate::parakeet::assets::UncompressedAsset::new_file($path);
    };
}
