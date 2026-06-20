//! HEVC CABAC context initialisation (thesis-style HEVC extension, rung 2).
//!
//! HEVC's arithmetic decoding engine is identical to H.264's (FFmpeg's HEVC
//! decoder uses the same `libavcodec/cabac.h` core), so the engine itself is
//! reused verbatim from the H.264 path — see `crate::thesis::cabac::Cabac`.
//! Only context *initialisation* differs: 179 contexts, three init types, and
//! `m`/`n` derived from a single init byte (ITU-T H.265 §9.3.2.2) rather than
//! looked up as a pair. Init values come from `hevc_cabac_tables.rs` (generated
//! from FFmpeg by scripts/gen_hevc_cabac_tables.py).

pub use crate::thesis::cabac::Cabac;
use crate::hevc::SliceType;
use crate::hevc_cabac_tables::{HEVC_CONTEXTS, INIT_VALUES};

/// Initialise the HEVC CABAC context states (ITU-T H.265 §9.3.2.2).
///
/// `init_type` selection: `2 - slice_type` (I→0, P→1, B→2), swapped between P/B
/// when `cabac_init_flag` is set on a non-I slice. The per-context state is the
/// FFmpeg combined byte (`pStateIdx<<1 | valMPS`), exactly as the H.264 engine
/// expects, so the reused `Cabac::get` consumes it directly.
pub fn init_states(states: &mut [u8; HEVC_CONTEXTS], slice_type: SliceType, cabac_init_flag: bool, slice_qp: i32) {
    let st_num = match slice_type {
        SliceType::B => 0,
        SliceType::P => 1,
        SliceType::I => 2,
    };
    let mut init_type = 2 - st_num;
    if cabac_init_flag && slice_type != SliceType::I {
        init_type ^= 3; // swaps the P and B init tables (both in {1,2})
    }
    let qp = slice_qp.clamp(0, 51);
    for i in 0..HEVC_CONTEXTS {
        let iv = INIT_VALUES[init_type as usize][i] as i32;
        let m = (iv >> 4) * 5 - 45;
        let n = ((iv & 15) << 3) - 16;
        let mut pre = 2 * (((m * qp) >> 4) + n) - 127;
        pre ^= pre >> 31; // abs via sign-smear (matches FFmpeg)
        if pre > 124 {
            pre = 124 + (pre & 1);
        }
        states[i] = pre as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_formula_known_value() {
        // I-slice (init_type 0), ctx 0 = sao_merge_flag, init_value 153, qp 26.
        // m=(153>>4)*5-45=0, n=((153&15)<<3)-16=56, pre=2*(0+56)-127=-15,
        // abs-smear -> 14 (<=124). Hand-derived from §9.3.2.2.
        let mut st = [0u8; HEVC_CONTEXTS];
        init_states(&mut st, SliceType::I, false, 26);
        assert_eq!(st[0], 14);
        assert!(st.iter().all(|&s| s <= 125));
    }

    #[test]
    fn cabac_init_flag_swaps_p_b() {
        // With cabac_init_flag, a P slice must use the B init table and vice
        // versa, so the two results cross over.
        let mut p_plain = [0u8; HEVC_CONTEXTS];
        let mut p_swapped = [0u8; HEVC_CONTEXTS];
        let mut b_plain = [0u8; HEVC_CONTEXTS];
        init_states(&mut p_plain, SliceType::P, false, 32);
        init_states(&mut p_swapped, SliceType::P, true, 32);
        init_states(&mut b_plain, SliceType::B, false, 32);
        assert_eq!(p_swapped, b_plain);
        assert_ne!(p_plain, b_plain);
    }
}
