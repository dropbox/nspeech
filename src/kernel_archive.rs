//! Kernel archive loader: looks up kernels by name from a zstd-compressed tar.

use std::borrow::Cow;
use std::sync::OnceLock;

/// Load a kernel by name from a zstd-compressed tar archive.
///
/// The archive is decompressed once on first access and cached for the process lifetime.
/// `name` is the kernel stem without extension (e.g. "matmul_fp16_64x64x32").
/// `ext` is "metallib" or "dxil".
/// `compressed` is the embedded zstd-compressed tar bytes.
pub fn load_kernel(name: &str, ext: &str, compressed: &'static [u8], cache: &OnceLock<Vec<u8>>) -> Option<Cow<'static, [u8]>> {
    let tar = cache.get_or_init(|| {
        zstd::bulk::decompress(compressed, 8 * 1024 * 1024)
            .expect("failed to decompress kernel archive")
    });
    let filename = format!("{name}.{ext}");
    tar_lookup(tar, &filename).map(|slice| {
        // Safety: the OnceLock keeps the Vec alive for 'static, so we can
        // extend the slice lifetime. The Vec is never mutated after init.
        let static_slice: &'static [u8] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr(), slice.len())
        };
        Cow::Borrowed(static_slice)
    })
}

fn tar_lookup<'a>(archive: &'a [u8], filename: &str) -> Option<&'a [u8]> {
    let mut offset = 0;
    while offset + 512 <= archive.len() {
        let header = &archive[offset..offset + 512];

        if header.iter().all(|&b| b == 0) {
            return None;
        }

        let name_end = header[..100].iter().position(|&b| b == 0).unwrap_or(100);
        let entry_name = std::str::from_utf8(&header[..name_end]).ok()?;

        let size_str = std::str::from_utf8(&header[124..136]).ok()?;
        let size =
            usize::from_str_radix(size_str.trim_matches(|c: char| c == '\0' || c == ' '), 8)
                .ok()?;

        let data_offset = offset + 512;
        let padded_size = (size + 511) & !511;

        let basename = entry_name.rsplit('/').next().unwrap_or(entry_name);
        if basename == filename {
            if data_offset + size <= archive.len() {
                return Some(&archive[data_offset..data_offset + size]);
            }
        }

        offset = data_offset + padded_size;
    }
    None
}
