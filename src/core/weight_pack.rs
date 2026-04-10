//! **GOZ1** — grok-ozempic packed checkpoint format (not GGUF).
//!
//! Little-endian container: metadata key/value pairs, tensor table with
//! placeholder offsets, alignment padding, then streamed tensor blobs.
//!
//! # Streaming
//! [`PackStreamWriter`] writes tensor info with placeholder offsets, then each
//! payload once via [`PackStreamWriter::write_tensor_data`], and patches offsets
//! in [`PackStreamWriter::finalize`].

use std::{
    collections::BTreeMap,
    io::{self, Seek, SeekFrom, Write},
};

use crate::error::{GrokOzempicError, Result};

/// File magic: ASCII `GOZ1` (grok-ozempic container version 1), little-endian.
const OZ1_MAGIC: u32 = u32::from_le_bytes([b'G', b'O', b'Z', b'1']);
const OZ1_VERSION: u32 = 1;

/// Tensor blob alignment in bytes.
pub const DATA_ALIGNMENT: u64 = 32;

const META_U32: u32 = 0;
const META_STR: u32 = 1;

/// Packed tensor payload kinds (GOZ1 tensor table).
pub const TENSOR_F16: u32 = 0;
/// 2-bit ternary {-1,0,+1}, four values per byte (same packing as quantizer).
pub const TENSOR_TERNARY: u32 = 1;

/// Metadata value in the GOZ1 header.
#[derive(Debug, Clone)]
pub enum PackMetaValue {
    U32(u32),
    Str(String),
}

/// One row in the tensor table (payload written separately).
#[derive(Clone, Debug)]
pub struct PackTensorHeader {
    pub name: String,
    /// Row-major shape, slowest index first.
    pub shape: Vec<u64>,
    pub tensor_type: u32,
}

/// Streams a GOZ1 file without buffering all tensor payloads in RAM.
pub struct PackStreamWriter<'a, W: Write + Seek> {
    writer: &'a mut W,
    tensor_count: usize,
    tensors_written: usize,
    offset_field_positions: Vec<u64>,
    real_offsets: Vec<u64>,
    data_section_start: u64,
}

impl<'a, W: Write + Seek> PackStreamWriter<'a, W> {
    pub fn begin(
        writer: &'a mut W,
        metadata: &BTreeMap<String, PackMetaValue>,
        tensor_headers: &[PackTensorHeader],
    ) -> Result<Self> {
        write_u32(writer, OZ1_MAGIC)?;
        write_u32(writer, OZ1_VERSION)?;
        write_u64(writer, tensor_headers.len() as u64)?;
        write_u64(writer, metadata.len() as u64)?;

        for (key, value) in metadata {
            write_str(writer, key)?;
            match value {
                PackMetaValue::U32(v) => {
                    write_u32(writer, META_U32)?;
                    write_u32(writer, *v)?;
                }
                PackMetaValue::Str(s) => {
                    write_u32(writer, META_STR)?;
                    write_str(writer, s)?;
                }
            }
        }

        let mut offset_field_positions: Vec<u64> = Vec::with_capacity(tensor_headers.len());
        for entry in tensor_headers {
            write_str(writer, &entry.name)?;
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

    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        if self.tensors_written >= self.tensor_count {
            return Err(GrokOzempicError::PackWrite(
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

    pub fn finalize(self) -> Result<()> {
        if self.tensors_written != self.tensor_count {
            return Err(GrokOzempicError::PackWrite(format!(
                "finalize: expected {} tensor blobs, got {}",
                self.tensor_count, self.tensors_written
            )));
        }
        if self.real_offsets.len() != self.offset_field_positions.len() {
            return Err(GrokOzempicError::PackWrite(
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

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_metadata() -> BTreeMap<String, PackMetaValue> {
        let mut m = BTreeMap::new();
        m.insert(
            "oz.name".into(),
            PackMetaValue::Str("grok-ozempic-test".into()),
        );
        m.insert("oz.quantization_version".into(), PackMetaValue::U32(1));
        m
    }

    #[test]
    fn stream_writer_magic_and_version() {
        let headers = vec![PackTensorHeader {
            name: "blk.0.ffn_gate.weight".into(),
            shape: vec![64, 32],
            tensor_type: TENSOR_TERNARY,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&vec![0xAB; 64]).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        assert_eq!(&bytes[0..4], b"GOZ1");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 1);
    }

    #[test]
    fn stream_writer_tensor_count() {
        let headers = vec![PackTensorHeader {
            name: "t".into(),
            shape: vec![1],
            tensor_type: TENSOR_F16,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[0u8; 2]).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 1);
    }

    #[test]
    fn stream_writer_two_tensors() {
        let headers = vec![
            PackTensorHeader {
                name: "a".into(),
                shape: vec![2],
                tensor_type: TENSOR_TERNARY,
            },
            PackTensorHeader {
                name: "b".into(),
                shape: vec![4],
                tensor_type: TENSOR_F16,
            },
        ];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[1, 2]).unwrap();
            w.write_tensor_data(&[0u8; 8]).unwrap();
            w.finalize().unwrap();
        }
        let len = buf.into_inner().len() as u64;
        assert_eq!(len % DATA_ALIGNMENT, 0);
    }

    #[test]
    fn stream_writer_empty_tensors() {
        let headers: Vec<PackTensorHeader> = vec![];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        assert_eq!(&bytes[0..4], b"GOZ1");
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 0);
    }
}
