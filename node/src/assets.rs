/// Asset loading for Parakeet models (similar to rust-warp pattern)

use once_cell::sync::OnceCell;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
};
use std::fs;

/// A Zstandard-compressed asset that is decompressed at most once and
/// cached for the program's lifetime.
pub struct Asset {
    path: &'static str,
    decompressed: OnceCell<Result<Vec<u8>, ()>>,
}

impl Asset {
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            decompressed: OnceCell::new(),
        }
    }

    /// Return the decompressed bytes; performs work only on first call.
    pub fn bytes(&'static self, assets: &PathBuf) -> Result<&'static [u8], ()> {
        self.decompressed
            .get_or_init(|| {
                let compressed: Vec<u8> = match fs::read(Path::new(&assets.join(self.path))) {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("failed to read asset {}: {}", self.path, e);
                        return Err(());
                    }
                };

                let mut cursor = Cursor::new(compressed);
                match zstd::stream::decode_all(&mut cursor) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        log::warn!("failed to decompress asset: {}", e);
                        Err(())
                    }
                }
            })
            .as_deref()
            .map_err(|_| ())
    }
}

/// Load a raw (non-compressed) asset file
pub fn load_raw_asset(assets: &PathBuf, filename: &str) -> Result<Vec<u8>, String> {
    fs::read(assets.join(filename))
        .map_err(|e| format!("Failed to read {}: {}", filename, e))
}
