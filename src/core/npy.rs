//! Minimal `.npy` (NumPy array) header parsing for memory-mapped tensors.
//!
//! Grok-1 JAX checkpoints are often distributed as per-tensor `.npy` files.
//! This module parses the header to obtain `dtype`, `shape`, and the byte
//! offset where raw array data begins, so large tensors can be accessed via
//! `mmap` without loading the full array into RAM.

use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::error::{GrokOzempicError, Result};

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Logical dtype of the `.npy` payload (only types used by the quantizer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpyDtype {
    F32,
    F16,
    BF16,
    Other,
}

/// Memory-mapped `.npy` with parsed header; raw values are in [`MmapNpy::data`].
pub struct MmapNpy {
    mmap: Mmap,
    dtype: NpyDtype,
    shape: Vec<usize>,
    data_offset: usize,
}

impl MmapNpy {
    pub fn map_path(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(GrokOzempicError::Io)?;
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(GrokOzempicError::Io)? };
        let (dtype, shape, data_offset) = parse_npy_header(&mmap)?;
        Ok(Self {
            mmap,
            dtype,
            shape,
            data_offset,
        })
    }

    pub fn dtype(&self) -> NpyDtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[u8] {
        &self.mmap[self.data_offset..]
    }
}

/// Parse NumPy `.npy` header prefix; returns `(dtype, shape, data_byte_offset)`.
pub fn parse_npy_header(buf: &[u8]) -> Result<(NpyDtype, Vec<usize>, usize)> {
    if buf.len() < 10 {
        return Err(GrokOzempicError::InvalidConfig(
            "npy: file too small for header".into(),
        ));
    }
    if &buf[0..6] != NPY_MAGIC {
        return Err(GrokOzempicError::InvalidConfig(
            "npy: missing \\x93NUMPY magic".into(),
        ));
    }
    let major = buf[6];
    let (header_len, header_start) = match major {
        1 => {
            let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
            (hlen, 10)
        }
        2 => {
            if buf.len() < 12 {
                return Err(GrokOzempicError::InvalidConfig("npy: truncated v2".into()));
            }
            let hlen = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
            (hlen, 12)
        }
        _ => {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "npy: unsupported version {major}"
            )));
        }
    };
    let header_end = header_start + header_len;
    if buf.len() < header_end {
        return Err(GrokOzempicError::InvalidConfig(
            "npy: header length exceeds file".into(),
        ));
    }
    let header_str = std::str::from_utf8(&buf[header_start..header_end]).map_err(|_| {
        GrokOzempicError::InvalidConfig("npy: header is not valid UTF-8".into())
    })?;
    let descr = parse_descr(header_str).ok_or_else(|| {
        GrokOzempicError::InvalidConfig("npy: could not parse descr".into())
    })?;
    let shape = parse_shape(header_str).ok_or_else(|| {
        GrokOzempicError::InvalidConfig("npy: could not parse shape".into())
    })?;
    let dtype = npy_descr_to_dtype(descr);
    let preamble = header_start;
    let total = preamble + header_len;
    let pad_to_64 = (64 - (total % 64)) % 64;
    let data_offset = total + pad_to_64;
    if buf.len() < data_offset {
        return Err(GrokOzempicError::InvalidConfig(
            "npy: file ends before data section".into(),
        ));
    }
    Ok((dtype, shape, data_offset))
}

fn parse_descr(header: &str) -> Option<&str> {
    let key = "'descr'";
    let i = header.find(key)?;
    let rest = &header[i + key.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &rest[quote.len_utf8()..];
    let end = rest.find(quote)?;
    Some(&rest[..end])
}

fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let key = "'shape'";
    let i = header.find(key)?;
    let rest = &header[i + key.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let rest = rest.strip_prefix('(')?;
    let end_paren = rest.find(')')?;
    let inner = &rest[..end_paren];
    if inner.is_empty() {
        return Some(vec![]);
    }
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let n: usize = p.parse().ok()?;
        dims.push(n);
    }
    Some(dims)
}

fn npy_descr_to_dtype(descr: &str) -> NpyDtype {
    let d = descr.trim();
    let body = d
        .strip_prefix('<')
        .or_else(|| d.strip_prefix('>'))
        .or_else(|| d.strip_prefix('|'))
        .or_else(|| d.strip_prefix('='))
        .unwrap_or(d);
    match body {
        "f4" => NpyDtype::F32,
        "f2" => NpyDtype::F16,
        "u2" | "bfloat16" | "bf16" => NpyDtype::BF16,
        _ => NpyDtype::Other,
    }
}

/// Map a filename stem like `blk__0__weight` back to a logical tensor name `blk.0.weight`.
pub fn npy_stem_to_tensor_name(stem: &str) -> String {
    stem.replace("__", ".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_v1_header() {
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }";
        let mut header = Vec::new();
        header.extend_from_slice(NPY_MAGIC);
        header.push(1);
        header.push(0);
        let hlen = dict.len() as u16;
        header.extend_from_slice(&hlen.to_le_bytes());
        header.extend_from_slice(dict.as_bytes());
        let preamble = header.len();
        let pad = (64 - (preamble % 64)) % 64;
        header.extend(std::iter::repeat(b' ').take(pad));
        let data_offset = header.len();
        header.extend_from_slice(&[0u8; 48]);
        let (dtype, shape, off) = parse_npy_header(&header).unwrap();
        assert_eq!(dtype, NpyDtype::F32);
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(off, data_offset);
    }
}
