//! Minimal GGUF v3 writer for grok-ozempic quantized tensors.
//!
//! Implements just enough of the GGUF binary format to store:
//! - Key-value metadata (strings and u32 scalars).
//! - Ternary-quantized (2-bit packed) tensors.
//! - FP16 pass-through tensors for MoE routing gates.
//!
//! # Streaming
//! [`GgufStreamWriter`] writes tensor **info** with placeholder offsets, then
//! accepts each tensor's payload **once** via [`GgufStreamWriter::write_tensor_data`]
//! (no accumulation of all tensors in RAM). Offsets are patched in a final
//! [`GgufStreamWriter::finalize`] pass.
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
pub const DATA_ALIGNMENT: u64 = 32;

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
// Tensor header (no payload — data is streamed separately)
// ---------------------------------------------------------------------------

/// Static tensor metadata written into the GGUF info section (payload follows
/// in the data section via [`GgufStreamWriter::write_tensor_data`]).
#[derive(Clone, Debug)]
pub struct TensorHeader {
    pub name: String,
    /// Shape as a list of dimensions (slowest-varying last, per GGUF spec).
    pub shape: Vec<u64>,
    pub tensor_type: u32,
}

// ---------------------------------------------------------------------------
// GgufStreamWriter
// ---------------------------------------------------------------------------

/// Single-pass GGUF file writer: header with placeholder offsets, stream each
/// tensor blob, then seek-back offset fix-up. Holds **no** tensor payloads.
pub struct GgufStreamWriter<'a, W: Write + Seek> {
    writer: &'a mut W,
    tensor_count: usize,
    tensors_written: usize,
    offset_field_positions: Vec<u64>,
    real_offsets: Vec<u64>,
    data_section_start: u64,
}

impl<'a, W: Write + Seek> GgufStreamWriter<'a, W> {
    /// Write the metadata and tensor info headers (placeholder offsets), pad
    /// to [`DATA_ALIGNMENT`], and leave the writer positioned at the start of
    /// the tensor data section.
    pub fn begin(
        writer: &'a mut W,
        metadata: &BTreeMap<String, GgufMetaValue>,
        tensor_headers: &[TensorHeader],
    ) -> Result<Self> {
        write_u32(writer, GGUF_MAGIC)?;
        write_u32(writer, GGUF_VERSION)?;
        write_u64(writer, tensor_headers.len() as u64)?;
        write_u64(writer, metadata.len() as u64)?;

        for (key, value) in metadata {
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

        let mut offset_field_positions: Vec<u64> = Vec::with_capacity(tensor_headers.len());
        for entry in tensor_headers {
            write_gguf_string(writer, &entry.name)?;
            write_u32(writer, entry.shape.len() as u32)?;
            for &dim in &entry.shape {
                write_u64(writer, dim)?;
            }
            write_u32(writer, entry.tensor_type)?;
            offset_field_positions.push(writer.stream_position().map_err(GrokOzempicError::Io)?);
            write_u64(writer, 0u64)?;
        }

        let header_end = writer.stream_position().map_err(GrokOzempicError::Io)?;
        let padding_needed = (DATA_ALIGNMENT - (header_end % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        writer
            .write_all(&vec![0u8; padding_needed as usize])
            .map_err(GrokOzempicError::Io)?;

        let data_section_start = writer.stream_position().map_err(GrokOzempicError::Io)?;

        Ok(Self {
            writer,
            tensor_count: tensor_headers.len(),
            tensors_written: 0,
            offset_field_positions,
            real_offsets: Vec::with_capacity(tensor_headers.len()),
            data_section_start,
        })
    }

    /// Append one tensor's raw bytes (quantized payload) in tensor-header order.
    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        if self.tensors_written >= self.tensor_count {
            return Err(GrokOzempicError::GgufWrite(
                "write_tensor_data: more blobs than tensor headers".into(),
            ));
        }
        let pos = self.writer.stream_position().map_err(GrokOzempicError::Io)?;
        self.real_offsets.push(pos - self.data_section_start);
        self.writer.write_all(data).map_err(GrokOzempicError::Io)?;
        let cur = self.writer.stream_position().map_err(GrokOzempicError::Io)?;
        let pad = (DATA_ALIGNMENT - (cur % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        self.writer
            .write_all(&vec![0u8; pad as usize])
            .map_err(GrokOzempicError::Io)?;
        self.tensors_written += 1;
        Ok(())
    }

    /// Patch tensor offset fields and ensure blob count matches headers.
    pub fn finalize(self) -> Result<()> {
        if self.tensors_written != self.tensor_count {
            return Err(GrokOzempicError::GgufWrite(format!(
                "finalize: expected {} tensor blobs, got {}",
                self.tensor_count, self.tensors_written
            )));
        }
        if self.real_offsets.len() != self.offset_field_positions.len() {
            return Err(GrokOzempicError::GgufWrite(
                "internal: offset bookkeeping mismatch".into(),
            ));
        }
        for (offset_pos, real_offset) in self
            .offset_field_positions
            .iter()
            .zip(self.real_offsets.iter())
        {
            self.writer
                .seek(SeekFrom::Start(*offset_pos))
                .map_err(GrokOzempicError::Io)?;
            write_u64(self.writer, *real_offset)?;
        }
        self.writer
            .seek(SeekFrom::End(0))
            .map_err(GrokOzempicError::Io)?;
        Ok(())
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

    fn sample_metadata() -> BTreeMap<String, GgufMetaValue> {
        let mut m = BTreeMap::new();
        m.insert("general.name".into(), GgufMetaValue::Str("grok-ozempic-test".into()));
        m.insert("general.quantization_version".into(), GgufMetaValue::U32(1));
        m
    }

    #[test]
    fn stream_writer_magic_and_version() {
        let headers = vec![TensorHeader {
            name: "blk.0.ffn_gate.weight".into(),
            shape: vec![64, 32],
            tensor_type: GGUF_TENSOR_TYPE_TERNARY,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = GgufStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&vec![0xAB; 64]).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, GGUF_MAGIC);
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 3);
    }

    #[test]
    fn stream_writer_tensor_count() {
        let headers = vec![TensorHeader {
            name: "t".into(),
            shape: vec![1],
            tensor_type: GGUF_TENSOR_TYPE_F16,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = GgufStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[0u8; 2]).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 1);
    }

    #[test]
    fn stream_writer_two_tensors_no_payload_buffering() {
        let headers = vec![
            TensorHeader {
                name: "a".into(),
                shape: vec![2],
                tensor_type: GGUF_TENSOR_TYPE_TERNARY,
            },
            TensorHeader {
                name: "b".into(),
                shape: vec![4],
                tensor_type: GGUF_TENSOR_TYPE_F16,
            },
        ];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = GgufStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[1, 2]).unwrap();
            w.write_tensor_data(&[0u8; 8]).unwrap();
            w.finalize().unwrap();
        }
        let len = buf.into_inner().len() as u64;
        assert_eq!(len % DATA_ALIGNMENT, 0);
    }

    #[test]
    fn stream_writer_empty_tensors() {
        let headers: Vec<TensorHeader> = vec![];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let w = GgufStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, GGUF_MAGIC);
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 0);
    }
}
