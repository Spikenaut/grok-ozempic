//! Minimal GGUF v3 writer for grok-ozempic quantized tensors.
//!
//! Implements just enough of the GGUF binary format to store:
//! - Key-value metadata (strings and u32 scalars).
//! - Ternary-quantized (2-bit packed) tensors.
//! - FP16 pass-through tensors for MoE routing gates.
//!
//! # GGUF format overview (v3)
//! ```text
//! magic            u32  = 0x46554747 ("GGUF")
//! version          u32  = 3
//! tensor_count     u64
//! metadata_kv_count u64
//! [metadata KV entries …]
//! [tensor info entries …]  ← names, shapes, types, byte offsets
//! <alignment padding to DATA_ALIGNMENT>
//! [tensor data …]
//! ```

use std::{
    collections::BTreeMap,
    io::{self, Seek, SeekFrom, Write},
};

use crate::error::{GrokOzempicError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic number: the bytes 'G','G','U','F' (0x47,0x47,0x55,0x46) read as
/// a little-endian u32 give 0x4655_4747.
const GGUF_MAGIC: u32 = 0x4655_4747;
/// Writer targets GGUF version 3.
const GGUF_VERSION: u32 = 3;
/// Tensor data is aligned to this boundary in bytes.
const DATA_ALIGNMENT: u64 = 32;

// ---------------------------------------------------------------------------
// GGUF value type tags (GGUFMetadataValueType)
// ---------------------------------------------------------------------------
const GGUF_TYPE_UINT32: u32 = 5;
const GGUF_TYPE_STRING: u32 = 8;

// ---------------------------------------------------------------------------
// GGUF tensor type tags (GGUFTensorType)
// ---------------------------------------------------------------------------

/// Our custom 2-bit ternary type.  We assign it the value 30 which is beyond
/// the standard llama.cpp types (0-28 as of spec v3) so downstream tools will
/// know they need custom handling.
pub const GGUF_TENSOR_TYPE_TERNARY: u32 = 30;
/// IEEE 754 half-precision, matching llama.cpp's `GGML_TYPE_F16 = 1`.
pub const GGUF_TENSOR_TYPE_F16: u32 = 1;

// ---------------------------------------------------------------------------
// Metadata KV entry
// ---------------------------------------------------------------------------

/// A single key-value pair written into the GGUF metadata section.
pub enum GgufMetaValue {
    U32(u32),
    Str(String),
}

// ---------------------------------------------------------------------------
// Tensor descriptor queued for writing
// ---------------------------------------------------------------------------

/// Records everything needed to write one tensor's info header and data blob.
pub struct TensorEntry {
    pub name: String,
    /// Shape as a list of dimensions (slowest-varying last, per GGUF spec).
    pub shape: Vec<u64>,
    pub tensor_type: u32,
    /// Raw bytes to be written in the data section.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// GgufWriter
// ---------------------------------------------------------------------------

/// Streaming GGUF writer.
///
/// Tensors are added one at a time via [`GgufWriter::add_tensor`]; the file is
/// finalised by calling [`GgufWriter::finish`] which flushes the header (with
/// correct byte offsets) followed by all tensor data blobs.
pub struct GgufWriter {
    metadata: BTreeMap<String, GgufMetaValue>,
    tensors: Vec<TensorEntry>,
}

impl GgufWriter {
    /// Create a new, empty writer.
    pub fn new() -> Self {
        Self { metadata: BTreeMap::new(), tensors: Vec::new() }
    }

    /// Insert a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: GgufMetaValue) {
        self.metadata.insert(key.into(), value);
    }

    /// Queue a tensor for inclusion in the GGUF file.
    pub fn add_tensor(&mut self, entry: TensorEntry) {
        self.tensors.push(entry);
    }

    /// Write the complete GGUF file to `writer`.
    ///
    /// This performs two passes over the header area:
    /// 1. Write a placeholder header with dummy offsets.
    /// 2. Compute real data offsets from the actual header size.
    /// 3. Seek back and rewrite tensor info with correct offsets.
    /// 4. Pad to alignment, then stream tensor data blobs.
    pub fn finish<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        // --- Pass 1: write header with placeholder offsets ---
        write_u32(writer, GGUF_MAGIC)?;
        write_u32(writer, GGUF_VERSION)?;
        write_u64(writer, self.tensors.len() as u64)?;
        write_u64(writer, self.metadata.len() as u64)?;

        // Metadata KV entries.
        for (key, value) in &self.metadata {
            write_gguf_string(writer, key)?;
            match value {
                GgufMetaValue::U32(v) => {
                    write_u32(writer, GGUF_TYPE_UINT32)?;
                    write_u32(writer, *v)?;
                }
                GgufMetaValue::Str(s) => {
                    write_u32(writer, GGUF_TYPE_STRING)?;
                    write_gguf_string(writer, s)?;
                }
            }
        }

        // Tensor info entries — record the position of each offset field so we
        // can seek back and fix it up in pass 2.
        let mut offset_positions: Vec<u64> = Vec::with_capacity(self.tensors.len());
        for entry in &self.tensors {
            write_gguf_string(writer, &entry.name)?;
            write_u32(writer, entry.shape.len() as u32)?;
            for &dim in &entry.shape {
                write_u64(writer, dim)?;
            }
            write_u32(writer, entry.tensor_type)?;
            // Placeholder offset — we'll overwrite this in pass 2.
            offset_positions.push(writer.stream_position().map_err(GrokOzempicError::Io)?);
            write_u64(writer, 0u64)?;
        }

        // --- Alignment padding before data section ---
        let header_end = writer.stream_position().map_err(GrokOzempicError::Io)?;
        let padding_needed = (DATA_ALIGNMENT - (header_end % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        let zeroes = vec![0u8; padding_needed as usize];
        writer.write_all(&zeroes).map_err(GrokOzempicError::Io)?;

        let data_section_start = writer.stream_position().map_err(GrokOzempicError::Io)?;

        // --- Pass 2: write tensor data and record real offsets ---
        let mut real_offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        for entry in &self.tensors {
            let pos = writer.stream_position().map_err(GrokOzempicError::Io)?;
            real_offsets.push(pos - data_section_start);
            writer.write_all(&entry.data).map_err(GrokOzempicError::Io)?;
            // Align after each tensor blob.
            let cur = writer.stream_position().map_err(GrokOzempicError::Io)?;
            let pad = (DATA_ALIGNMENT - (cur % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
            writer.write_all(&vec![0u8; pad as usize]).map_err(GrokOzempicError::Io)?;
        }

        // --- Seek back and fix up offsets in tensor info headers ---
        for (offset_pos, real_offset) in offset_positions.iter().zip(real_offsets.iter()) {
            writer
                .seek(SeekFrom::Start(*offset_pos))
                .map_err(GrokOzempicError::Io)?;
            write_u64(writer, *real_offset)?;
        }

        Ok(())
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Low-level write helpers
// ---------------------------------------------------------------------------

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Write a GGUF string: u64 length followed by UTF-8 bytes (no null terminator).
fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_writer_with_one_tensor() -> (GgufWriter, TensorEntry) {
        let mut gw = GgufWriter::new();
        gw.set_metadata("general.name", GgufMetaValue::Str("grok-ozempic-test".into()));
        gw.set_metadata("general.quantization_version", GgufMetaValue::U32(1));

        let entry = TensorEntry {
            name: "blk.0.ffn_gate.weight".to_string(),
            shape: vec![64, 32],
            tensor_type: GGUF_TENSOR_TYPE_TERNARY,
            data: vec![0xAB; 64], // dummy packed data
        };
        (gw, entry)
    }

    #[test]
    fn write_and_check_magic() {
        let (mut gw, entry) = make_writer_with_one_tensor();
        gw.add_tensor(entry);
        let mut buf = Cursor::new(Vec::<u8>::new());
        gw.finish(&mut buf).unwrap();

        let bytes = buf.into_inner();
        assert!(bytes.len() > 16, "output is too short");
        // First 4 bytes must be GGUF magic in little-endian.
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, GGUF_MAGIC);
        // Next 4 bytes: version = 3.
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, 3);
    }

    #[test]
    fn tensor_count_is_correct() {
        let (mut gw, entry) = make_writer_with_one_tensor();
        gw.add_tensor(entry);
        let mut buf = Cursor::new(Vec::<u8>::new());
        gw.finish(&mut buf).unwrap();
        let bytes = buf.into_inner();
        // Bytes 8-15: tensor_count u64.
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 1);
    }

    #[test]
    fn data_section_is_aligned() {
        let (mut gw, entry) = make_writer_with_one_tensor();
        gw.add_tensor(entry);
        let mut buf = Cursor::new(Vec::<u8>::new());
        gw.finish(&mut buf).unwrap();
        // The file must be non-empty and the total length a multiple of DATA_ALIGNMENT
        // (because we pad after each tensor blob).
        let len = buf.into_inner().len() as u64;
        assert_eq!(len % DATA_ALIGNMENT, 0, "file length not aligned to {DATA_ALIGNMENT}");
    }

    #[test]
    fn empty_writer_produces_valid_header() {
        let gw = GgufWriter::new();
        let mut buf = Cursor::new(Vec::<u8>::new());
        gw.finish(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, GGUF_MAGIC);
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 0);
    }
}
