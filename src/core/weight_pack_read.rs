//! Verifier for **GOZ1** packed checkpoints (see [`crate::core::weight_pack`]).
//!
//! Parses header + tensor table and checks blob layout against file size.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{
    core::weight_pack::{DATA_ALIGNMENT, TENSOR_F16, TENSOR_TERNARY},
    error::{GrokOzempicError, Result},
};

const OZ1_MAGIC: u32 = u32::from_le_bytes([b'G', b'O', b'Z', b'1']);

const META_U32: u32 = 0;
const META_STR: u32 = 1;

#[derive(Debug, Clone)]
pub struct PackVerifyReport {
    pub version: u32,
    pub tensor_count: usize,
    pub metadata_keys: Vec<String>,
    pub tensor_names: Vec<String>,
    pub file_size: u64,
}

pub fn verify_pack_file(path: &Path) -> Result<PackVerifyReport> {
    let mut f = File::open(path).map_err(GrokOzempicError::Io)?;
    let file_size = f.metadata().map_err(GrokOzempicError::Io)?.len();

    let magic = read_u32(&mut f)?;
    if magic != OZ1_MAGIC {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: bad magic (expected GOZ1), got {magic:#x}"
        )));
    }
    let version = read_u32(&mut f)?;
    let tensor_count = read_u64(&mut f)? as usize;
    let meta_count = read_u64(&mut f)? as usize;

    let mut metadata_keys = Vec::with_capacity(meta_count);
    for _ in 0..meta_count {
        let key = read_str(&mut f)?;
        metadata_keys.push(key.clone());
        let vtype = read_u32(&mut f)?;
        skip_meta_value(&mut f, vtype)?;
    }

    let mut tensor_names = Vec::with_capacity(tensor_count);
    let mut infos: Vec<TensorInfoParsed> = Vec::with_capacity(tensor_count);

    for _ in 0..tensor_count {
        let name = read_str(&mut f)?;
        tensor_names.push(name.clone());
        let ndim = read_u32(&mut f)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(&mut f)?);
        }
        let tensor_type = read_u32(&mut f)?;
        let offset = read_u64(&mut f)?;
        infos.push(TensorInfoParsed {
            name,
            shape,
            tensor_type,
            data_offset: offset,
        });
    }

    let header_end = f.stream_position().map_err(GrokOzempicError::Io)?;
    let padding_needed = (DATA_ALIGNMENT - (header_end % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
    f.seek(SeekFrom::Current(padding_needed as i64))
        .map_err(GrokOzempicError::Io)?;
    let data_section_start = f.stream_position().map_err(GrokOzempicError::Io)?;

    let mut expected_rel: u64 = 0;
    for (i, info) in infos.iter().enumerate() {
        if info.data_offset != expected_rel {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: tensor {} ({}) data_offset {} != expected cumulative {}",
                i, info.name, info.data_offset, expected_rel
            )));
        }
        let nbytes = tensor_nbytes(info)?;
        let abs = data_section_start + info.data_offset;
        let end = abs.saturating_add(nbytes);
        if end > file_size {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: tensor {} ({}) blob end {} exceeds file size {}",
                i, info.name, end, file_size
            )));
        }
        expected_rel = expected_rel.saturating_add(align_up(nbytes, DATA_ALIGNMENT));
    }

    if data_section_start + expected_rel > file_size {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: declared tensor payloads need {} bytes past data section start, file has {}",
            expected_rel,
            file_size.saturating_sub(data_section_start)
        )));
    }

    Ok(PackVerifyReport {
        version,
        tensor_count,
        metadata_keys,
        tensor_names,
        file_size,
    })
}

struct TensorInfoParsed {
    name: String,
    shape: Vec<u64>,
    tensor_type: u32,
    data_offset: u64,
}

fn tensor_nbytes(info: &TensorInfoParsed) -> Result<u64> {
    let n: u64 = info.shape.iter().product();
    match info.tensor_type {
        TENSOR_F16 => n.checked_mul(2).ok_or_else(|| {
            GrokOzempicError::InvalidConfig("GOZ1 verify: shape overflow (f16)".into())
        }),
        TENSOR_TERNARY => Ok(n.div_ceil(4)),
        other => Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: unknown tensor type {other} (cannot compute size)"
        ))),
    }
}

fn align_up(n: u64, align: u64) -> u64 {
    (n + align - 1) / align * align
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
    Ok(u64::from_le_bytes(b))
}

fn read_str<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(GrokOzempicError::Io)?;
    String::from_utf8(buf)
        .map_err(|e| GrokOzempicError::InvalidConfig(format!("GOZ1: invalid utf8: {e}")))
}

fn skip_meta_value<R: Read>(r: &mut R, vtype: u32) -> Result<()> {
    match vtype {
        META_U32 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
        }
        META_STR => {
            let _s = read_str(r)?;
        }
        _ => {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: unsupported metadata value type {vtype}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{BufWriter, Write};

    use crate::core::weight_pack::{PackMetaValue, PackStreamWriter, PackTensorHeader, TENSOR_TERNARY};

    #[test]
    fn verify_round_trip_stream_writer() {
        let dir = std::env::temp_dir();
        let path = dir.join("grok_ozempic_verify_test.goz1");
        let _ = std::fs::remove_file(&path);

        let mut meta = BTreeMap::new();
        meta.insert("oz.name".into(), PackMetaValue::Str("t".into()));
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![8],
            tensor_type: TENSOR_TERNARY,
        }];
        let f = File::create(&path).unwrap();
        let mut bw = BufWriter::new(f);
        {
            let mut w = PackStreamWriter::begin(&mut bw, &meta, &headers).unwrap();
            w.write_tensor_data(&[0xFF; 2]).unwrap();
            w.finalize().unwrap();
        }
        bw.flush().unwrap();
        drop(bw);

        let report = verify_pack_file(&path).unwrap();
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.version, 1);
        assert!(report.metadata_keys.contains(&"oz.name".to_string()));
        let _ = std::fs::remove_file(&path);
    }
}
