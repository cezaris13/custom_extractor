//! CAVLC residual decoding, sync-only (thesis Chapter 4 §4.4.1).
//!
//! Motion-vector extraction does not need residual coefficient *values* — it
//! only needs to (a) consume exactly the residual bits so the bitstream stays
//! in sync, and (b) recover `TotalCoeff` per block, which feeds the `nC`
//! neighbour predictor for the next block's `coeff_token`. So this decodes the
//! `coeff_token` / level / `total_zeros` / `run_before` syntax purely to advance
//! the reader and returns `TotalCoeff`; it never reconstructs the block.
//!
//! The VLC tables are ported verbatim from FFmpeg's `libavcodec/h264_cavlc.c`
//! (the authoritative, well-tested source already vendored in this repo) rather
//! than re-typed from the standard — that is the one place a transcription typo
//! would silently desync the whole slice. The level/zeros/run *algorithm*
//! follows ITU-T H.264 §9.2.2 directly (FFmpeg's version is the same logic with
//! a precomputed level table we don't need here).
use super::BitReader;
use std::sync::OnceLock;

/// O(1) VLC decoder: a flat table indexed by the next `max_len` peeked bits,
/// each slot packing `(symbol << 5) | code_len` (0 = no entry). Built once from
/// the `(len, bits)` tables — replaces the per-symbol O(max_len × entries) scan
/// that dominated CAVLC residual decoding (~57% of its time).
struct Vlc {
    lut: Box<[u16]>,
    max_len: u8,
}

impl Vlc {
    fn build(len: &[u8], bits: &[u8]) -> Vlc {
        let max_len = *len.iter().max().unwrap_or(&0);
        let mut lut = vec![0u16; 1usize << max_len];
        for k in 0..len.len() {
            let l = len[k];
            if l == 0 {
                continue;
            }
            let shift = max_len - l;
            let base = (bits[k] as usize) << shift;
            let packed = ((k as u16) << 5) | l as u16; // code_len fits in 5 bits (≤16)
            for slot in &mut lut[base..base + (1usize << shift)] {
                *slot = packed;
            }
        }
        Vlc { lut: lut.into_boxed_slice(), max_len }
    }

    #[inline]
    fn decode(&self, r: &mut BitReader) -> Option<usize> {
        let packed = self.lut[r.peek_bits(self.max_len as u32) as usize];
        let l = (packed & 0x1f) as usize;
        if l == 0 {
            return None;
        }
        r.skip_bits(l);
        Some((packed >> 5) as usize)
    }
}

/// All CAVLC residual VLCs, built once from the const tables on first use.
struct Vlcs {
    coeff_token: [Vlc; 4],
    chroma_dc_coeff_token: Vlc,
    chroma422_dc_coeff_token: Vlc,
    total_zeros: [Vlc; 15],
    chroma_dc_total_zeros: [Vlc; 3],
    chroma422_dc_total_zeros: [Vlc; 7],
    run: [Vlc; 7],
}

fn vlcs() -> &'static Vlcs {
    static V: OnceLock<Vlcs> = OnceLock::new();
    V.get_or_init(|| Vlcs {
        coeff_token: std::array::from_fn(|i| Vlc::build(&COEFF_TOKEN_LEN[i], &COEFF_TOKEN_BITS[i])),
        chroma_dc_coeff_token: Vlc::build(&CHROMA_DC_COEFF_TOKEN_LEN, &CHROMA_DC_COEFF_TOKEN_BITS),
        chroma422_dc_coeff_token: Vlc::build(
            &CHROMA422_DC_COEFF_TOKEN_LEN,
            &CHROMA422_DC_COEFF_TOKEN_BITS,
        ),
        total_zeros: std::array::from_fn(|i| Vlc::build(&TOTAL_ZEROS_LEN[i], &TOTAL_ZEROS_BITS[i])),
        chroma_dc_total_zeros: std::array::from_fn(|i| {
            Vlc::build(&CHROMA_DC_TOTAL_ZEROS_LEN[i], &CHROMA_DC_TOTAL_ZEROS_BITS[i])
        }),
        chroma422_dc_total_zeros: std::array::from_fn(|i| {
            Vlc::build(&CHROMA422_DC_TOTAL_ZEROS_LEN[i], &CHROMA422_DC_TOTAL_ZEROS_BITS[i])
        }),
        run: std::array::from_fn(|i| Vlc::build(&RUN_LEN[i], &RUN_BITS[i])),
    })
}

/// `coeff_token` (ITU-T Table 9-5) -> (total_coeff, trailing_ones). `n_c` is the
/// neighbour-derived predictor: >=0 selects the luma bucket, -1 = chroma DC
/// 4:2:0 (2x2), -2 = chroma DC 4:2:2 (2x4).
fn coeff_token(r: &mut BitReader, n_c: i32) -> Option<(i32, i32)> {
    let idx = if n_c == -1 {
        return chroma_dc_coeff_token(r);
    } else if n_c == -2 {
        return chroma422_dc_coeff_token(r);
    } else if n_c < 2 {
        0
    } else if n_c < 4 {
        1
    } else if n_c < 8 {
        2
    } else {
        3
    };
    let k = vlcs().coeff_token[idx].decode(r)?;
    // FFmpeg layout: index = total_coeff*4 + trailing_ones.
    Some(((k / 4) as i32, (k % 4) as i32))
}

fn chroma_dc_coeff_token(r: &mut BitReader) -> Option<(i32, i32)> {
    let k = vlcs().chroma_dc_coeff_token.decode(r)?;
    Some(((k / 4) as i32, (k % 4) as i32))
}

fn chroma422_dc_coeff_token(r: &mut BitReader) -> Option<(i32, i32)> {
    let k = vlcs().chroma422_dc_coeff_token.decode(r)?;
    Some(((k / 4) as i32, (k % 4) as i32))
}

/// Count leading zero bits up to the terminating 1 (`level_prefix`, §9.2.2.1).
fn level_prefix(r: &mut BitReader) -> u32 {
    let mut n = 0u32;
    while r.read_bit() == 0 {
        n += 1;
        if n > 63 {
            break;
        }
    }
    n
}

/// Decode one residual block for synchronisation only and return `TotalCoeff`.
///
/// `n_c` is the neighbour predictor, `max_coeff` the maximum coefficient count
/// for the block type (16 luma, 15 luma-AC, 4 chroma-DC 4:2:0, 8 chroma-DC
/// 4:2:2). On desync returns `None`.
pub fn residual_block(r: &mut BitReader, n_c: i32, max_coeff: usize) -> Option<i32> {
    let (total_coeff, trailing_ones) = coeff_token(r, n_c)?;
    if total_coeff == 0 {
        return Some(0);
    }

    // trailing_ones sign bits (1 each); we only need to skip them.
    r.skip_bits(trailing_ones as usize);

    // Levels (§9.2.2.1). We compute |level| only to drive suffix_length.
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };
    for i in trailing_ones..total_coeff {
        let prefix = level_prefix(r);
        let mut level_suffix_size = suffix_length;
        if prefix == 14 && suffix_length == 0 {
            level_suffix_size = 4;
        } else if prefix >= 15 {
            level_suffix_size = prefix - 3;
        }
        let level_suffix = if level_suffix_size > 0 {
            r.read_bits(level_suffix_size)
        } else {
            0
        };
        let mut level_code = (prefix.min(15) << suffix_length) + level_suffix;
        if prefix >= 15 && suffix_length == 0 {
            level_code += 15;
        }
        if prefix >= 16 {
            level_code += (1u32 << (prefix - 3)) - 4096;
        }
        // First non-trailing-one coefficient bumps the code when T1 < 3.
        if i == trailing_ones && trailing_ones < 3 {
            level_code += 2;
        }
        let abs_level = (level_code / 2) + 1; // |level| from level_code
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if abs_level > (3u32 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    // total_zeros (§9.2.3).
    let mut zeros_left: usize = if total_coeff as usize == max_coeff {
        0
    } else {
        let tc = total_coeff as usize;
        if max_coeff == 4 {
            vlcs().chroma_dc_total_zeros[tc - 1].decode(r)?
        } else if max_coeff <= 8 {
            vlcs().chroma422_dc_total_zeros[tc - 1].decode(r)?
        } else {
            vlcs().total_zeros[tc - 1].decode(r)?
        }
    };

    // run_before per coefficient (§9.2.4).
    let mut i = 1;
    while i < total_coeff && zeros_left > 0 {
        let row = if zeros_left < 7 { zeros_left - 1 } else { 6 };
        let run_before = vlcs().run[row].decode(r)?;
        zeros_left = zeros_left.saturating_sub(run_before);
        i += 1;
    }

    Some(total_coeff)
}

// ── VLC tables, ported from libavcodec/h264_cavlc.c ──────────────────────────
// Rows padded to fixed width with zeros (len==0 ⇒ no entry), matching the C
// `[N][M]` zero-initialised layout.

#[rustfmt::skip]
const CHROMA_DC_COEFF_TOKEN_LEN: [u8; 4 * 5] = [
    2,0,0,0, 6,1,0,0, 6,6,3,0, 6,7,7,6, 6,8,8,7,
];
#[rustfmt::skip]
const CHROMA_DC_COEFF_TOKEN_BITS: [u8; 4 * 5] = [
    1,0,0,0, 7,1,0,0, 4,6,1,0, 3,3,2,5, 2,3,2,0,
];

#[rustfmt::skip]
const CHROMA422_DC_COEFF_TOKEN_LEN: [u8; 4 * 9] = [
    1,0,0,0, 7,2,0,0, 7,7,3,0, 9,7,7,5, 9,9,7,6,
    10,10,9,7, 11,11,10,7, 12,12,11,10, 13,12,12,11,
];
#[rustfmt::skip]
const CHROMA422_DC_COEFF_TOKEN_BITS: [u8; 4 * 9] = [
    1,0,0,0, 15,1,0,0, 14,13,1,0, 7,12,11,1, 6,5,10,1,
    7,6,4,9, 7,6,5,8, 7,6,5,4, 7,5,4,4,
];

#[rustfmt::skip]
const COEFF_TOKEN_LEN: [[u8; 4 * 17]; 4] = [
    [
        1,0,0,0,
        6,2,0,0,  8,6,3,0,  9,8,7,5,  10,9,8,6,
        11,10,9,7, 13,11,10,8, 13,13,11,9, 13,13,13,10,
        14,14,13,11, 14,14,14,13, 15,15,14,14, 15,15,15,14,
        16,15,15,15, 16,16,16,15, 16,16,16,16, 16,16,16,16,
    ],
    [
        2,0,0,0,
        6,2,0,0,  6,5,3,0,  7,6,6,4,  8,6,6,4,
        8,7,7,5,  9,8,8,6,  11,9,9,6, 11,11,11,7,
        12,11,11,9, 12,12,12,11, 12,12,12,11, 13,13,13,12,
        13,13,13,13, 13,14,13,13, 14,14,14,13, 14,14,14,14,
    ],
    [
        4,0,0,0,
        6,4,0,0,  6,5,4,0,  6,5,5,4,  7,5,5,4,
        7,5,5,4,  7,6,6,4,  7,6,6,4,  8,7,7,5,
        8,8,7,6,  9,8,8,7,  9,9,8,8,  9,9,9,8,
        10,9,9,9, 10,10,10,10, 10,10,10,10, 10,10,10,10,
    ],
    [
        6,0,0,0,
        6,6,0,0,  6,6,6,0,  6,6,6,6,  6,6,6,6,
        6,6,6,6,  6,6,6,6,  6,6,6,6,  6,6,6,6,
        6,6,6,6,  6,6,6,6,  6,6,6,6,  6,6,6,6,
        6,6,6,6,  6,6,6,6,  6,6,6,6,  6,6,6,6,
    ],
];
#[rustfmt::skip]
const COEFF_TOKEN_BITS: [[u8; 4 * 17]; 4] = [
    [
        1,0,0,0,
        5,1,0,0,  7,4,1,0,  7,6,5,3,  7,6,5,3,
        7,6,5,4,  15,6,5,4, 11,14,5,4, 8,10,13,4,
        15,14,9,4, 11,10,13,12, 15,14,9,12, 11,10,13,8,
        15,1,9,12, 11,14,13,8, 7,10,9,12, 4,6,5,8,
    ],
    [
        3,0,0,0,
        11,2,0,0, 7,7,3,0,  7,10,9,5, 7,6,5,4,
        4,6,5,6,  7,6,5,8,  15,6,5,4, 11,14,13,4,
        15,10,9,4, 11,14,13,12, 8,10,9,8, 15,14,13,12,
        11,10,9,12, 7,11,6,8, 9,8,10,1, 7,6,5,4,
    ],
    [
        15,0,0,0,
        15,14,0,0, 11,15,13,0, 8,12,14,12, 15,10,11,11,
        11,8,9,10, 9,14,13,9, 8,10,9,8, 15,14,13,13,
        11,14,10,12, 15,10,13,12, 11,14,9,12, 8,10,13,8,
        13,7,9,12, 9,12,11,10, 5,8,7,6, 1,4,3,2,
    ],
    [
        3,0,0,0,
        0,1,0,0,  4,5,6,0,  8,9,10,11, 12,13,14,15,
        16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31,
        32,33,34,35, 36,37,38,39, 40,41,42,43, 44,45,46,47,
        48,49,50,51, 52,53,54,55, 56,57,58,59, 60,61,62,63,
    ],
];

#[rustfmt::skip]
const TOTAL_ZEROS_LEN: [[u8; 16]; 16] = [
    [1,3,3,4,4,5,5,6,6,7,7,8,8,9,9,9],
    [3,3,3,3,3,4,4,4,4,5,5,6,6,6,6,0],
    [4,3,3,3,4,4,3,3,4,5,5,6,5,6,0,0],
    [5,3,4,4,3,3,3,4,3,4,5,5,5,0,0,0],
    [4,4,4,3,3,3,3,3,4,5,4,5,0,0,0,0],
    [6,5,3,3,3,3,3,3,4,3,6,0,0,0,0,0],
    [6,5,3,3,3,2,3,4,3,6,0,0,0,0,0,0],
    [6,4,5,3,2,2,3,3,6,0,0,0,0,0,0,0],
    [6,6,4,2,2,3,2,5,0,0,0,0,0,0,0,0],
    [5,5,3,2,2,2,4,0,0,0,0,0,0,0,0,0],
    [4,4,3,3,1,3,0,0,0,0,0,0,0,0,0,0],
    [4,4,2,1,3,0,0,0,0,0,0,0,0,0,0,0],
    [3,3,1,2,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];
#[rustfmt::skip]
const TOTAL_ZEROS_BITS: [[u8; 16]; 16] = [
    [1,3,2,3,2,3,2,3,2,3,2,3,2,3,2,1],
    [7,6,5,4,3,5,4,3,2,3,2,3,2,1,0,0],
    [5,7,6,5,4,3,4,3,2,3,2,1,1,0,0,0],
    [3,7,5,4,6,5,4,3,3,2,2,1,0,0,0,0],
    [5,4,3,7,6,5,4,3,2,1,1,0,0,0,0,0],
    [1,1,7,6,5,4,3,2,1,1,0,0,0,0,0,0],
    [1,1,5,4,3,3,2,1,1,0,0,0,0,0,0,0],
    [1,1,1,3,3,2,2,1,0,0,0,0,0,0,0,0],
    [1,0,1,3,2,1,1,1,0,0,0,0,0,0,0,0],
    [1,0,1,3,2,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,2,1,3,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

#[rustfmt::skip]
const CHROMA_DC_TOTAL_ZEROS_LEN: [[u8; 4]; 3] = [
    [1,2,3,3],
    [1,2,2,0],
    [1,1,0,0],
];
#[rustfmt::skip]
const CHROMA_DC_TOTAL_ZEROS_BITS: [[u8; 4]; 3] = [
    [1,1,1,0],
    [1,1,0,0],
    [1,0,0,0],
];

#[rustfmt::skip]
const CHROMA422_DC_TOTAL_ZEROS_LEN: [[u8; 8]; 7] = [
    [1,3,3,4,4,4,5,5],
    [3,2,3,3,3,3,3,0],
    [3,3,2,2,3,3,0,0],
    [3,2,2,2,3,0,0,0],
    [2,2,2,2,0,0,0,0],
    [2,2,1,0,0,0,0,0],
    [1,1,0,0,0,0,0,0],
];
#[rustfmt::skip]
const CHROMA422_DC_TOTAL_ZEROS_BITS: [[u8; 8]; 7] = [
    [1,2,3,2,3,1,1,0],
    [0,1,1,4,5,6,7,0],
    [0,1,1,2,6,7,0,0],
    [6,0,1,2,7,0,0,0],
    [0,1,2,3,0,0,0,0],
    [0,1,1,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
];

#[rustfmt::skip]
const RUN_LEN: [[u8; 16]; 7] = [
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,3,3,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,3,3,3,3,0,0,0,0,0,0,0,0,0,0],
    [2,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0],
    [3,3,3,3,3,3,3,4,5,6,7,8,9,10,11,0],
];
#[rustfmt::skip]
const RUN_BITS: [[u8; 16]; 7] = [
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,3,2,1,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,1,3,2,5,4,0,0,0,0,0,0,0,0,0],
    [7,6,5,4,3,2,1,1,1,1,1,1,1,1,1,0],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(bits: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in bits.chars().filter(|c| *c == '0' || *c == '1') {
            cur = (cur << 1) | (c == '1') as u8;
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            out.push(cur << (8 - n));
        }
        out
    }

    #[test]
    fn coeff_token_trivial_entries() {
        // bucket 0 (nC<2): "1" -> (0,0); "01" -> (1,1); "001" -> (2,2)
        let d = reader("1");
        assert_eq!(coeff_token(&mut BitReader::new(&d), 0), Some((0, 0)));
        let d = reader("01");
        assert_eq!(coeff_token(&mut BitReader::new(&d), 0), Some((1, 1)));
        let d = reader("001");
        assert_eq!(coeff_token(&mut BitReader::new(&d), 0), Some((2, 2)));
    }

    #[test]
    fn empty_block_consumes_one_bit() {
        // coeff_token "1" (nC<2) => TotalCoeff 0, exactly one bit consumed.
        let d = reader("1100000000");
        let mut r = BitReader::new(&d);
        assert_eq!(residual_block(&mut r, 0, 16), Some(0));
        // next bit still readable as the following '1'
        assert_eq!(r.read_bit(), 1);
    }

    #[test]
    fn single_trailing_one_block() {
        // TotalCoeff=1, T1=1: coeff_token "01", sign bit "0" (=+1),
        // total_zeros (tc=1) "1" (=0 zeros). Block has one coefficient.
        // bits: 01 0 1
        let d = reader("01011111");
        let mut r = BitReader::new(&d);
        assert_eq!(residual_block(&mut r, 0, 16), Some(1));
        // 4 bits consumed; the rest ('1111') remain.
        assert_eq!(r.read_bits(4), 0b1111);
    }

    #[test]
    fn thesis_worked_example() {
        // ITU/thesis §4.4.1.3 worked example, nC=1, 4x4 luma block.
        // Coded bitstream (24 bits) decodes to TotalCoeff=5 and consumes all 24.
        let d = reader("000010001110010111101101");
        let mut r = BitReader::new(&d);
        let tc = residual_block(&mut r, 1, 16).expect("should decode");
        assert_eq!(tc, 5, "TotalCoeff");
        assert_eq!(r.bits_left(), 0, "must consume exactly 24 bits");
    }

    #[test]
    fn nc8_fixed_length_empty() {
        // bucket 3 (nC>=8): "000011" -> (0,0)
        let d = reader("000011");
        assert_eq!(coeff_token(&mut BitReader::new(&d), 8), Some((0, 0)));
    }
}
