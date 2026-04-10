//! Saliency-Aware Ternary Quantizer
//!
//! Converts FP32 / FP16 weight tensors into a 2-bit ternary representation
//! {-1, 0, +1} using a Generalised Integrate-and-Fire (GIF) threshold.
//!
//! # Algorithm
//! 1. Compute the **saliency** of a layer as the root-mean-square of its
//!    absolute weight values:  `rms = sqrt( mean(w²) )`
//! 2. Derive the **GIF threshold**:  `τ = gif_threshold × rms`
//! 3. For each weight `w`:
//!    - `|w| < τ`  →  0.0  (silenced — "dead neuron")
//!    - `w  ≥  τ`  →  +1.0 (positive spike)
//!    - `w  ≤ -τ`  →  -1.0 (negative spike)
//! 4. Pack four ternary values into each byte using 2 bits per value:
//!    `0b00 = 0`, `0b01 = +1`, `0b10 = -1`.

use half::f16;

/// Outcome of quantizing a single tensor.
pub struct QuantizedTensor {
    /// Packed 2-bit ternary data (4 values per byte, LSB-first).
    pub packed: Vec<u8>,
    /// Number of individual weight values represented.
    pub num_elements: usize,
    /// The RMS saliency value used to derive the threshold.
    pub rms: f32,
    /// The actual firing threshold τ = gif_threshold × rms.
    pub threshold: f32,
    /// Fraction of weights silenced to zero.
    pub sparsity: f32,
}

/// Encode a single ternary value {-1, 0, +1} as 2 bits.
///
/// Encoding: 0 → 0b00, +1 → 0b01, -1 → 0b10.
#[inline]
fn encode_trit(v: f32) -> u8 {
    if v > 0.0 {
        0b01
    } else if v < 0.0 {
        0b10
    } else {
        0b00
    }
}

/// Decode a 2-bit trit back to f32.
#[allow(dead_code)]
#[inline]
pub fn decode_trit(bits: u8) -> f32 {
    match bits & 0b11 {
        0b01 => 1.0,
        0b10 => -1.0,
        _ => 0.0,
    }
}

/// Pack a slice of ternary floats into a byte vector (4 values per byte).
///
/// The slice length does not need to be a multiple of 4; the final byte is
/// zero-padded.
fn pack_trits(ternary: &[f32]) -> Vec<u8> {
    let num_bytes = ternary.len().div_ceil(4);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in ternary.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_shift = (i % 4) * 2;
        packed[byte_idx] |= encode_trit(v) << bit_shift;
    }
    packed
}

/// Apply saliency-aware GIF ternary quantization to FP32 weights.
///
/// # Arguments
/// * `weights`       - Raw FP32 weight slice.
/// * `gif_threshold` - Threshold multiplier applied to the layer RMS.
///
/// # Returns
/// A [`QuantizedTensor`] with packed ternary data and diagnostic statistics.
pub fn quantize_f32(weights: &[f32], gif_threshold: f32) -> QuantizedTensor {
    let n = weights.len();
    if n == 0 {
        return QuantizedTensor {
            packed: vec![],
            num_elements: 0,
            rms: 0.0,
            threshold: 0.0,
            sparsity: 1.0,
        };
    }

    // Step 1: compute RMS saliency.
    let sum_sq: f64 = weights.iter().map(|&w| (w as f64) * (w as f64)).sum();
    let rms = ((sum_sq / n as f64).sqrt()) as f32;

    // Step 2: derive firing threshold.
    let threshold = gif_threshold * rms;

    // Step 3: apply GIF ternary gate.
    let mut ternary = Vec::with_capacity(n);
    let mut zeros: usize = 0;
    for &w in weights {
        let t = if w >= threshold {
            1.0f32
        } else if w <= -threshold {
            -1.0f32
        } else {
            zeros += 1;
            0.0f32
        };
        ternary.push(t);
    }
    let sparsity = zeros as f32 / n as f32;

    // Step 4: pack into 2-bit representation.
    let packed = pack_trits(&ternary);

    QuantizedTensor { packed, num_elements: n, rms, threshold, sparsity }
}

/// Apply saliency-aware GIF ternary quantization to FP16 weights.
///
/// Converts to FP32 internally; the computational overhead is negligible
/// compared with I/O for expert-sized tensors.
pub fn quantize_f16(weights: &[f16], gif_threshold: f32) -> QuantizedTensor {
    let f32_weights: Vec<f32> = weights.iter().map(|&h| h.to_f32()).collect();
    quantize_f32(&f32_weights, gif_threshold)
}

/// Re-encode FP16 weights as raw bytes for mixed-precision (FP16-pass-through)
/// tensors such as MoE routing gates.
///
/// The bytes are in little-endian IEEE 754 half-precision order (same as GOZ1 `TENSOR_F16`).
pub fn passthrough_f16(weights: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(weights.len() * 2);
    for &h in weights {
        let bits = h.to_bits();
        out.push(bits as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

/// Re-encode FP32 weights as FP16 bytes for mixed-precision pass-through.
pub fn convert_f32_to_f16_bytes(weights: &[f32]) -> Vec<u8> {
    let halves: Vec<f16> = weights.iter().map(|&v| f16::from_f32(v)).collect();
    passthrough_f16(&halves)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trit_encoding_round_trip() {
        assert_eq!(decode_trit(encode_trit(1.0)), 1.0);
        assert_eq!(decode_trit(encode_trit(-1.0)), -1.0);
        assert_eq!(decode_trit(encode_trit(0.0)), 0.0);
    }

    #[test]
    fn pack_trits_basic() {
        // Four values fit exactly in one byte.
        let vals = [1.0f32, -1.0, 0.0, 1.0];
        let packed = pack_trits(&vals);
        assert_eq!(packed.len(), 1);
        assert_eq!(decode_trit((packed[0]) & 0b11), 1.0);
        assert_eq!(decode_trit((packed[0] >> 2) & 0b11), -1.0);
        assert_eq!(decode_trit((packed[0] >> 4) & 0b11), 0.0);
        assert_eq!(decode_trit((packed[0] >> 6) & 0b11), 1.0);
    }

    #[test]
    fn quantize_uniform_above_threshold() {
        // All weights are ±2.0; rms = 2.0.
        // With gif_threshold = 0.5: threshold = 0.5 × 2.0 = 1.0 < 2.0 → no weights zeroed.
        // With gif_threshold = 1.0: threshold = 1.0 × 2.0 = 2.0 = |w| → boundary, treated as spike.
        let weights = vec![2.0f32; 8];
        let qt = quantize_f32(&weights, 0.5); // threshold = 1.0, all |w|=2.0 exceed it
        assert_eq!(qt.sparsity, 0.0);
        assert_eq!(qt.num_elements, 8);
    }

    #[test]
    fn quantize_all_silenced() {
        // Very high gif_threshold silences everything.
        let weights: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let qt = quantize_f32(&weights, 1000.0);
        assert_eq!(qt.sparsity, 1.0);
    }

    #[test]
    fn sparsity_mixed() {
        // weights = [small, small, large, large]; threshold cuts the small ones.
        let weights = vec![0.01f32, -0.01, 2.0, -2.0];
        let qt = quantize_f32(&weights, 1.0);
        // rms ≈ sqrt((0.0001+0.0001+4+4)/4) ≈ sqrt(2.00005) ≈ 1.414
        // threshold ≈ 1.414  =>  0.01 < threshold, 2.0 > threshold
        assert_eq!(qt.sparsity, 0.5);
    }

    #[test]
    fn passthrough_f16_roundtrip() {
        let originals = vec![f16::from_f32(1.5), f16::from_f32(-3.14)];
        let bytes = passthrough_f16(&originals);
        assert_eq!(bytes.len(), 4);
        // Reconstruct.
        let reconstructed: Vec<f16> = bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(reconstructed[0], originals[0]);
        assert_eq!(reconstructed[1], originals[1]);
    }
}
