//! HEVC CTU slice decode (rung 3) — coding_quadtree → CU → PU → transform_tree
//! → residual, ported from FFmpeg libavcodec/hevc/{hevcdec.c,cabac.c} for the
//! 4:2:0 Main-profile, single-tile, non-WPP common case (the dashcam/stickman
//! clips). Targets CABAC *sync* first: every bin is parsed exactly as FFmpeg's
//! `motion_vectors_only` path does, so the arithmetic decoder lands precisely at
//! `end_of_slice_flag` after the last CTU. MV *values* (merge/AMVP derivation)
//! are layered on top in rung 4 — they don't affect bit positions, so sync is
//! validated independently here.

use crate::hevc::{Pps, SliceHeader, SliceType, Sps};
use crate::hevc_cabac::{init_states, Cabac};
use crate::hevc_cabac_tables::*;
#[allow(unused_imports)]
use crate::ffmpeg_common::MvFilter;
use mv_types::motion_vector::MvCompact;

const MODE_INTER: u8 = 0;
const MODE_INTRA: u8 = 1;
const MODE_SKIP: u8 = 2;
const PART_2NX2N: u8 = 0;
const PART_2NXN: u8 = 1;
const PART_NX2N: u8 = 2;
const PART_NXN: u8 = 3;
const PART_2NXNU: u8 = 4;
const PART_2NXND: u8 = 5;
const PART_NLX2N: u8 = 6;
const PART_NRX2N: u8 = 7;

const PF_L0: u8 = 1;
const PF_L1: u8 = 2;
const PF_BI: u8 = 3;
const PF_INTRA: u8 = 4;

/// Per-min-PU motion field entry (HEVC `MvField`).
#[derive(Clone, Copy, Default)]
pub struct MvField {
    pub mv: [[i16; 2]; 2], // [list][x,y], quarter-pel (HEVC MVs fit in i16)
    pub ref_idx: [i8; 2],
    pub pred_flag: u8,
}

/// Reference POC lists for the current slice + the collocated picture's motion
/// field, supplied by the frame manager (DPB) in `extractor9`.
pub struct RefInfo<'a> {
    pub poc: i32,
    pub ref_poc: [Vec<i32>; 2],
    pub ref_long: [Vec<bool>; 2],
    pub col: Option<&'a FrameMv>,
}

/// A decoded picture's motion field, retained in the DPB for temporal MVP.
pub struct FrameMv {
    pub poc: i32,
    pub tab_mvf: Vec<MvField>,
    pub min_pu_width: usize,
    pub min_pu_height: usize,
    pub ref_poc: [Vec<i32>; 2],
    pub ref_long: [Vec<bool>; 2],
    pub collocated_list: usize,
}

// scan tables (FFmpeg libavcodec/hevc/{data.c,cabac.c})
#[rustfmt::skip]
const DIAG4_X: [u8;16] = [0,0,1,0,1,2,0,1,2,3,1,2,3,2,3,3];
#[rustfmt::skip]
const DIAG4_Y: [u8;16] = [0,1,0,2,1,0,3,2,1,0,3,2,1,3,2,3];
#[rustfmt::skip]
const HORIZ4_X: [u8;16] = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
#[rustfmt::skip]
const HORIZ4_Y: [u8;16] = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3];
const HORIZ2_X: [u8;4] = [0,1,0,1];
const HORIZ2_Y: [u8;4] = [0,0,1,1];
const DIAG2_X: [u8;4] = [0,0,1,1];
const DIAG2_Y: [u8;4] = [0,1,0,1];
const DIAG2_INV: [[u8;2];2] = [[0,2],[1,3]];
const DIAG4_INV: [[u8;4];4] = [[0,2,5,9],[1,4,8,12],[3,7,11,14],[6,10,13,15]];
#[rustfmt::skip]
const DIAG8_INV: [[u8;8];8] = [
    [0,2,5,9,14,20,27,35],[1,4,8,13,19,26,34,42],[3,7,12,18,25,33,41,48],
    [6,11,17,24,32,40,47,53],[10,16,23,31,39,46,52,57],[15,22,30,38,45,51,56,60],
    [21,29,37,44,50,55,59,62],[28,36,43,49,54,58,61,63]];
#[rustfmt::skip]
const HORIZ8_INV: [[u8;8];8] = [
    [0,1,2,3,16,17,18,19],[4,5,6,7,20,21,22,23],[8,9,10,11,24,25,26,27],
    [12,13,14,15,28,29,30,31],[32,33,34,35,48,49,50,51],[36,37,38,39,52,53,54,55],
    [40,41,42,43,56,57,58,59],[44,45,46,47,60,61,62,63]];
const DIAG8_X: [u8;64] = [
    0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,
    1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7];
const DIAG8_Y: [u8;64] = [
    0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,5,4,3,2,1,0,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0,
    7,6,5,4,3,2,1,7,6,5,4,3,2,7,6,5,4,3,7,6,5,4,7,6,5,7,6,7];
const SCAN_1X1: [u8;1] = [0];

#[derive(Clone, Copy, PartialEq)]
enum Scan {
    Diag,
    Horiz,
    Vert,
}

// significant_coeff_flag ctx_idx_map (cabac.c), 5 rows of 16.
#[rustfmt::skip]
const CTX_IDX_MAP: [[u8;16];5] = [
    [0,1,4,5,2,3,4,5,6,6,8,8,7,7,8,8], // log2_trafo_size == 2
    [1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0], // prev_sig == 0
    [2,2,2,2,1,1,1,1,0,0,0,0,0,0,0,0], // prev_sig == 1
    [2,1,0,0,2,1,0,0,2,1,0,0,2,1,0,0], // prev_sig == 2
    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], // default
];

pub struct SyncResult {
    pub ctbs: u32,
    pub ok: bool,
}

/// Cached `E10_DBG` flag — avoids an env lookup + String alloc every frame.
fn dbg_enabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("E10_DBG").is_ok())
}

struct Ctx<'a> {
    c: Cabac<'a>,
    st: Box<[u8; HEVC_CONTEXTS]>,
    sps: &'a Sps,
    pps: &'a Pps,
    sh: &'a SliceHeader,
    log2_ctb: u32,
    log2_min_cb: u32,
    log2_min_tb: u32,
    log2_max_tb: u32,
    min_cb_width: usize,
    min_pu_width: usize,
    log2_min_pu: u32,
    min_pu_height: usize,
    skip_flag: Vec<u8>,
    ct_depth: Vec<u8>,
    ipm: Vec<u8>, // intra pred modes per min-PU (INTRA_DC=1 default)
    tab_mvf: Vec<MvField>,
    // reference info + collocated motion field (for MV derivation)
    ri: &'a RefInfo<'a>,
    // neighbour availability for the current PU (set_neighbour_available)
    na_up: bool,
    na_left: bool,
    na_up_left: bool,
    na_up_right: bool,
    na_up_right_sap: bool,
    na_bottom_left: bool,
    cu_x: i32,
    cu_y: i32,
    // within-CTB z-scan (Morton) lookup, precomputed once per slice
    zs_s: u32,
    zs_lut: Vec<i64>,
    ctb_w_i64: i64,
    // emitted motion vectors
    out: Vec<MvCompact>,
    frame_index: i32,
    // per-CTB neighbour availability
    ctb_left: bool,
    ctb_up: bool,
    ctb_up_right: bool,
    ctb_up_left: bool,
    // current CU/PU/TU scalar state
    ct_cur_depth: i32,
    cu_pred_mode: u8,
    cu_part_mode: u8,
    cu_intra_split: bool,
    cu_transquant_bypass: bool,
    cu_max_trafo_depth: i32,
    pu_merge_flag: bool,
    pu_intra_pred_mode: [u8; 4],
    pu_intra_pred_mode_c: u8,
    tu_intra_pred_mode: u8,
    tu_intra_pred_mode_c: u8,
    tu_is_cu_qp_delta_coded: bool,
    greater1_ctx: i32,
}

impl<'a> Ctx<'a> {
    #[inline]
    fn get(&mut self, ctx: usize) -> u32 {
        self.c.get(&mut self.st[ctx])
    }
    #[inline]
    fn bypass(&mut self) -> u32 {
        self.c.bypass()
    }
    #[inline]
    fn terminate(&mut self) -> bool {
        self.c.terminate()
    }
}

/// Reusable per-decode scratch buffers (avoids reallocating the large grids
/// every frame). `mvf_pool` recycles `tab_mvf` buffers evicted from the DPB.
#[derive(Default)]
pub struct HevcScratch {
    pub skip_flag: Vec<u8>,
    pub ct_depth: Vec<u8>,
    pub ipm: Vec<u8>,
    pub mvf_pool: Vec<Vec<MvField>>,
    zs_s: u32,
    zs_lut: Vec<i64>,
}

/// Decode one HEVC slice: parse the CTU tree (CABAC sync) AND reconstruct motion
/// vectors (merge/AMVP/temporal). Appends per-PU MVs to `out` and returns the
/// frame's motion field for the DPB. `ri` carries POC + reference lists;
/// `scratch` supplies reusable grid buffers.
pub fn decode_slice(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    ri: &RefInfo,
    frame_index: i32,
    out: Vec<MvCompact>,
    scratch: &mut HevcScratch,
) -> (SyncResult, FrameMv, Vec<MvCompact>) {
    let slice_type = sh.slice_type.unwrap_or(SliceType::I);
    let ctb_w = sps.pic_width_in_ctbs();
    let ctb_h = sps.pic_height_in_ctbs();
    let min_cb_width = (sps.pic_width_in_luma_samples >> sps.min_cb_log2_size_y) as usize;
    let min_cb_height = (sps.pic_height_in_luma_samples >> sps.min_cb_log2_size_y) as usize;
    let log2_min_pu = sps.min_cb_log2_size_y - 1;
    let min_pu_width = (sps.pic_width_in_luma_samples >> log2_min_pu) as usize;
    let min_pu_height = (sps.pic_height_in_luma_samples >> log2_min_pu) as usize;

    // Reuse scratch grids (cleared+resized) instead of fresh allocations.
    let mut skip_flag = std::mem::take(&mut scratch.skip_flag);
    skip_flag.clear();
    skip_flag.resize(min_cb_width * min_cb_height, 0);
    let mut ct_depth = std::mem::take(&mut scratch.ct_depth);
    ct_depth.clear();
    ct_depth.resize(min_cb_width * min_cb_height, 0);
    let mut ipm = std::mem::take(&mut scratch.ipm);
    ipm.clear();
    ipm.resize(min_pu_width * min_pu_height, 1);
    let mut tab_mvf = scratch.mvf_pool.pop().unwrap_or_default();
    tab_mvf.clear();
    tab_mvf.resize(min_pu_width * min_pu_height, MvField::default());

    // Within-CTB z-scan (Morton) lookup — depends only on the (constant)
    // CTB/min-TB ratio, so build it once and keep it in the scratch.
    let zs_s = sps.ctb_log2_size_y - sps.min_tb_log2_size_y;
    if scratch.zs_s != zs_s || scratch.zs_lut.is_empty() {
        let ctb_tb = 1usize << zs_s;
        let mut lut = vec![0i64; ctb_tb * ctb_tb];
        for yin in 0..ctb_tb {
            for xin in 0..ctb_tb {
                let mut m = 0i64;
                for i in 0..zs_s {
                    m |= (((xin >> i) & 1) as i64) << (2 * i) | (((yin >> i) & 1) as i64) << (2 * i + 1);
                }
                lut[yin * ctb_tb + xin] = m;
            }
        }
        scratch.zs_lut = lut;
        scratch.zs_s = zs_s;
    }
    let zs_lut = std::mem::take(&mut scratch.zs_lut);

    let mut cx = Ctx {
        c: Cabac::new(rbsp, sh.data_offset),
        st: Box::new([0u8; HEVC_CONTEXTS]),
        sps,
        pps,
        sh,
        log2_ctb: sps.ctb_log2_size_y,
        log2_min_cb: sps.min_cb_log2_size_y,
        log2_min_tb: sps.min_tb_log2_size_y,
        log2_max_tb: sps.max_tb_log2_size_y,
        log2_min_pu,
        min_pu_height,
        min_cb_width,
        min_pu_width,
        skip_flag,
        ct_depth,
        ipm,
        tab_mvf,
        ri,
        zs_s,
        zs_lut,
        ctb_w_i64: ctb_w as i64,
        na_up: false,
        na_left: false,
        na_up_left: false,
        na_up_right: false,
        na_up_right_sap: false,
        na_bottom_left: false,
        cu_x: 0,
        cu_y: 0,
        out,
        frame_index,
        ctb_left: false,
        ctb_up: false,
        ctb_up_right: false,
        ctb_up_left: false,
        ct_cur_depth: 0,
        cu_pred_mode: MODE_INTRA,
        cu_part_mode: PART_2NX2N,
        cu_intra_split: false,
        cu_transquant_bypass: false,
        cu_max_trafo_depth: 0,
        pu_merge_flag: false,
        pu_intra_pred_mode: [1; 4],
        pu_intra_pred_mode_c: 1,
        tu_intra_pred_mode: 1,
        tu_intra_pred_mode_c: 1,
        tu_is_cu_qp_delta_coded: false,
        greater1_ctx: 1,
    };
    init_states(&mut cx.st, slice_type, sh.cabac_init_flag, sh.slice_qp);

    let pic_ctbs = ctb_w * ctb_h;
    let ctb_size = 1u32 << cx.log2_ctb;
    let mut ctb_addr = sh.slice_segment_address;
    let mut more_data = true;
    let mut decoded = 0u32;

    while more_data && ctb_addr < pic_ctbs {
        let rx = ctb_addr % ctb_w;
        let ry = ctb_addr / ctb_w;
        let x_ctb = rx * ctb_size;
        let y_ctb = ry * ctb_size;
        cx.ctb_left = rx > 0 && ctb_addr > 0;
        cx.ctb_up = ry > 0 && ctb_addr >= ctb_w;
        cx.ctb_up_right = ry > 0 && (ctb_addr + 1) >= ctb_w && rx + 1 < ctb_w;
        cx.ctb_up_left = rx > 0 && ry > 0 && ctb_addr - 1 >= ctb_w;

        sao_param(&mut cx, rx, ry);
        let log2_ctb = cx.log2_ctb;
        more_data = coding_quadtree(&mut cx, x_ctb as i32, y_ctb as i32, log2_ctb, 0);
        ctb_addr += 1;
        decoded += 1;
    }

    let ok = ctb_addr == pic_ctbs;
    export_frame_mvs(&mut cx);
    if dbg_enabled() {
        let nz = cx.out.iter().filter(|m| m.src_x != m.dst_x || m.src_y != m.dst_y).count();
        eprintln!(
            "  [sync] type={:?} ctbs={}/{} ok={} poc={} tmvp={} col={} nb_refs={:?} mvs={} nz={}",
            slice_type, decoded, pic_ctbs, ok, ri.poc, sh.slice_temporal_mvp_enabled,
            ri.col.is_some(), sh.nb_refs, cx.out.len(), nz
        );
    }
    // Return the reusable grids to the scratch pool (tab_mvf is retained in the
    // returned FrameMv for temporal MVP).
    scratch.skip_flag = std::mem::take(&mut cx.skip_flag);
    scratch.ct_depth = std::mem::take(&mut cx.ct_depth);
    scratch.ipm = std::mem::take(&mut cx.ipm);
    scratch.zs_lut = std::mem::take(&mut cx.zs_lut);
    let frame = FrameMv {
        poc: ri.poc,
        tab_mvf: std::mem::take(&mut cx.tab_mvf),
        min_pu_width,
        min_pu_height,
        ref_poc: [ri.ref_poc[0].clone(), ri.ref_poc[1].clone()],
        ref_long: [ri.ref_long[0].clone(), ri.ref_long[1].clone()],
        collocated_list: sh.collocated_list,
    };
    (SyncResult { ctbs: decoded, ok }, frame, cx.out)
}

fn sao_param(cx: &mut Ctx, rx: u32, ry: u32) {
    let mut merge_left = false;
    let mut merge_up = false;
    if cx.sh.sao_luma || cx.sh.sao_chroma {
        if rx > 0 && cx.ctb_left {
            merge_left = cx.get(SAO_MERGE_FLAG_OFFSET) == 1;
        }
        if ry > 0 && !merge_left && cx.ctb_up {
            merge_up = cx.get(SAO_MERGE_FLAG_OFFSET) == 1;
        }
    }
    let nc = if cx.sps.chroma_format_idc != 0 { 3 } else { 1 };
    let mut sao_type_chroma = 0i32; // SAO_NOT_APPLIED
    for c_idx in 0..nc {
        let sao_on = match c_idx {
            0 => cx.sh.sao_luma,
            _ => cx.sh.sao_chroma,
        };
        if !sao_on {
            continue;
        }
        if merge_left || merge_up {
            continue;
        }
        let type_idx = if c_idx == 2 {
            sao_type_chroma
        } else {
            let t = sao_type_idx(cx);
            if c_idx == 1 {
                sao_type_chroma = t;
            }
            t
        };
        if type_idx == 0 {
            continue; // SAO_NOT_APPLIED
        }
        let mut offset_abs = [0i32; 4];
        for v in offset_abs.iter_mut() {
            *v = sao_offset_abs(cx);
        }
        if type_idx == 1 {
            // SAO_BAND
            for &a in &offset_abs {
                if a != 0 {
                    cx.bypass(); // offset_sign
                }
            }
            // band_position: 5 bypass bits
            for _ in 0..5 {
                cx.bypass();
            }
        } else if c_idx != 2 {
            // eo_class: 2 bypass bits
            cx.bypass();
            cx.bypass();
        }
    }
}

fn sao_type_idx(cx: &mut Ctx) -> i32 {
    if cx.get(SAO_TYPE_IDX_OFFSET) == 0 {
        return 0; // NOT_APPLIED
    }
    if cx.bypass() == 0 {
        1 // BAND
    } else {
        2 // EDGE
    }
}

fn sao_offset_abs(cx: &mut Ctx) -> i32 {
    // bit_depth assumed 8: length = (1<<(min(8,10)-5))-1 = 7
    let length = (1 << (cx.sps_bit_depth().min(10) - 5)) - 1;
    let mut i = 0;
    while i < length && cx.bypass() == 1 {
        i += 1;
    }
    i
}

impl<'a> Ctx<'a> {
    fn sps_bit_depth(&self) -> i32 {
        8 // Main/Main-still-picture; Main10 would be 10 (TODO if needed)
    }
}

/// coding_quadtree; returns more_data.
fn coding_quadtree(cx: &mut Ctx, x0: i32, y0: i32, log2_cb_size: u32, cb_depth: i32) -> bool {
    cx.ct_cur_depth = cb_depth;
    let cb_size = 1i32 << log2_cb_size;
    let w = cx.sps.pic_width_in_luma_samples as i32;
    let h = cx.sps.pic_height_in_luma_samples as i32;

    let split_cu = if x0 + cb_size <= w && y0 + cb_size <= h && log2_cb_size > cx.log2_min_cb {
        split_cu_flag(cx, cb_depth, x0, y0)
    } else {
        log2_cb_size > cx.log2_min_cb
    };

    if cx.pps.cu_qp_delta_enabled_flag
        && log2_cb_size >= cx.log2_ctb - cx.pps.diff_cu_qp_delta_depth
    {
        cx.tu_is_cu_qp_delta_coded = false;
    }

    if split_cu {
        let half = cb_size >> 1;
        let (x1, y1) = (x0 + half, y0 + half);
        let mut more = coding_quadtree(cx, x0, y0, log2_cb_size - 1, cb_depth + 1);
        if more && x1 < w {
            more = coding_quadtree(cx, x1, y0, log2_cb_size - 1, cb_depth + 1);
        }
        if more && y1 < h {
            more = coding_quadtree(cx, x0, y1, log2_cb_size - 1, cb_depth + 1);
        }
        if more && x1 < w && y1 < h {
            more = coding_quadtree(cx, x1, y1, log2_cb_size - 1, cb_depth + 1);
        }
        if more {
            (x1 + half) < w || (y1 + half) < h
        } else {
            false
        }
    } else {
        coding_unit(cx, x0, y0, log2_cb_size);
        let ctb_size = 1i32 << cx.log2_ctb;
        let at_ctb_edge_x = (x0 + cb_size) % ctb_size == 0 || (x0 + cb_size) >= w;
        let at_ctb_edge_y = (y0 + cb_size) % ctb_size == 0 || (y0 + cb_size) >= h;
        if at_ctb_edge_x && at_ctb_edge_y {
            !cx.terminate() // end_of_slice_flag
        } else {
            true
        }
    }
}

fn split_cu_flag(cx: &mut Ctx, ct_depth: i32, x0: i32, y0: i32) -> bool {
    let x0b = (x0 as u32) & ((1 << cx.log2_ctb) - 1);
    let y0b = (y0 as u32) & ((1 << cx.log2_ctb) - 1);
    let x_cb = (x0 >> cx.log2_min_cb) as usize;
    let y_cb = (y0 >> cx.log2_min_cb) as usize;
    let mut inc = 0;
    if cx.ctb_left || x0b != 0 {
        if cx.ct_depth[y_cb * cx.min_cb_width + x_cb - 1] as i32 > ct_depth {
            inc += 1;
        }
    }
    if cx.ctb_up || y0b != 0 {
        if cx.ct_depth[(y_cb - 1) * cx.min_cb_width + x_cb] as i32 > ct_depth {
            inc += 1;
        }
    }
    cx.get(SPLIT_CODING_UNIT_FLAG_OFFSET + inc) == 1
}

fn coding_unit(cx: &mut Ctx, x0: i32, y0: i32, log2_cb_size: u32) {
    let cb_size = 1i32 << log2_cb_size;
    let length = (cb_size >> cx.log2_min_cb) as usize;
    let x_cb = (x0 >> cx.log2_min_cb) as usize;
    let y_cb = (y0 >> cx.log2_min_cb) as usize;
    let idx = log2_cb_size - 2;

    cx.cu_pred_mode = MODE_INTRA;
    cx.cu_part_mode = PART_2NX2N;
    cx.cu_intra_split = false;
    cx.cu_x = x0;
    cx.cu_y = y0;

    cx.cu_transquant_bypass = if cx.pps.transquant_bypass_enabled_flag {
        cx.get(CU_TRANSQUANT_BYPASS_FLAG_OFFSET) == 1
    } else {
        false
    };

    let is_i = matches!(cx.sh.slice_type, Some(SliceType::I));
    let mut skip = false;
    if !is_i {
        let x0b = (x0 as u32) & ((1 << cx.log2_ctb) - 1);
        let y0b = (y0 as u32) & ((1 << cx.log2_ctb) - 1);
        skip = skip_flag_decode(cx, x0b != 0, y0b != 0, x_cb, y_cb) == 1;
        let sv = if skip { 1u8 } else { 0 };
        for y in 0..length {
            for x in 0..length {
                cx.skip_flag[(y_cb + y) * cx.min_cb_width + x_cb + x] = sv;
            }
        }
        cx.cu_pred_mode = if skip { MODE_SKIP } else { MODE_INTER };
    } else {
        for y in 0..length {
            for x in 0..length {
                cx.skip_flag[(y_cb + y) * cx.min_cb_width + x_cb + x] = 0;
            }
        }
    }

    let _ = idx;
    if skip {
        prediction_unit(cx, x0, y0, cb_size, cb_size, 0, log2_cb_size);
        set_ipm_default(cx, x0, y0, log2_cb_size);
    } else {
        let mut pcm_flag = false;
        if !is_i {
            cx.cu_pred_mode = if cx.get(PRED_MODE_FLAG_OFFSET) == 1 { MODE_INTRA } else { MODE_INTER };
        }
        if cx.cu_pred_mode != MODE_INTRA || log2_cb_size == cx.log2_min_cb {
            cx.cu_part_mode = part_mode_decode(cx, log2_cb_size);
            cx.cu_intra_split = cx.cu_part_mode == PART_NXN && cx.cu_pred_mode == MODE_INTRA;
        }

        if cx.cu_pred_mode == MODE_INTRA {
            if cx.cu_part_mode == PART_2NX2N
                && cx.sps.pcm_enabled
                && log2_cb_size >= cx.sps.pcm_log2_min_cb
                && log2_cb_size <= cx.sps.pcm_log2_max_cb
            {
                pcm_flag = cx.terminate(); // pcm_flag (terminate)
            }
            if pcm_flag {
                // I_PCM: byte-align + raw samples, then re-init engine. Rare in
                // these clips; mark and bail by leaving more_data handling to the
                // caller. (TODO: full PCM realign.)
                set_ipm_default(cx, x0, y0, log2_cb_size);
            } else {
                intra_prediction_unit(cx, x0, y0, log2_cb_size);
            }
        } else {
            set_ipm_default(cx, x0, y0, log2_cb_size);
            let cs = cb_size;
            let l = log2_cb_size;
            match cx.cu_part_mode {
                0 => prediction_unit(cx, x0, y0, cs, cs, 0, l), // 2Nx2N
                1 => {
                    prediction_unit(cx, x0, y0, cs, cs / 2, 0, l);
                    prediction_unit(cx, x0, y0 + cs / 2, cs, cs / 2, 1, l);
                }
                2 => {
                    prediction_unit(cx, x0, y0, cs / 2, cs, 0, l);
                    prediction_unit(cx, x0 + cs / 2, y0, cs / 2, cs, 1, l);
                }
                4 => {
                    prediction_unit(cx, x0, y0, cs, cs / 4, 0, l);
                    prediction_unit(cx, x0, y0 + cs / 4, cs, cs * 3 / 4, 1, l);
                }
                5 => {
                    prediction_unit(cx, x0, y0, cs, cs * 3 / 4, 0, l);
                    prediction_unit(cx, x0, y0 + cs * 3 / 4, cs, cs / 4, 1, l);
                }
                6 => {
                    prediction_unit(cx, x0, y0, cs / 4, cs, 0, l);
                    prediction_unit(cx, x0 + cs / 4, y0, cs * 3 / 4, cs, 1, l);
                }
                7 => {
                    prediction_unit(cx, x0, y0, cs * 3 / 4, cs, 0, l);
                    prediction_unit(cx, x0 + cs * 3 / 4, y0, cs / 4, cs, 1, l);
                }
                _ => {
                    // NxN
                    prediction_unit(cx, x0, y0, cs / 2, cs / 2, 0, l);
                    prediction_unit(cx, x0 + cs / 2, y0, cs / 2, cs / 2, 1, l);
                    prediction_unit(cx, x0, y0 + cs / 2, cs / 2, cs / 2, 2, l);
                    prediction_unit(cx, x0 + cs / 2, y0 + cs / 2, cs / 2, cs / 2, 3, l);
                }
            }
        }

        if !pcm_flag {
            let mut rqt_root_cbf = true;
            if cx.cu_pred_mode != MODE_INTRA
                && !(cx.cu_part_mode == PART_2NX2N && cx.pu_merge_flag)
            {
                rqt_root_cbf = cx.get(NO_RESIDUAL_DATA_FLAG_OFFSET) == 1;
            }
            if rqt_root_cbf {
                cx.cu_max_trafo_depth = if cx.cu_pred_mode == MODE_INTRA {
                    cx.sps.max_transform_hierarchy_depth_intra as i32 + cx.cu_intra_split as i32
                } else {
                    cx.sps.max_transform_hierarchy_depth_inter as i32
                };
                transform_tree(cx, x0, y0, x0, y0, log2_cb_size, log2_cb_size, 0, 0, [0, 0], [0, 0]);
            }
        }
    }

    // Mark intra CUs in the motion field so neighbouring inter PUs reject them
    // as merge/AMVP candidates (FFmpeg sets PF_INTRA via luma_intra_pred_mode).
    if cx.cu_pred_mode == MODE_INTRA {
        let w = (cb_size >> cx.log2_min_pu) as usize;
        let xp = (x0 >> cx.log2_min_pu) as usize;
        let yp = (y0 >> cx.log2_min_pu) as usize;
        for j in 0..w {
            for i in 0..w {
                cx.tab_mvf[(yp + j) * cx.min_pu_width + xp + i].pred_flag = PF_INTRA;
            }
        }
    }
    set_ct_depth(cx, x0, y0, log2_cb_size);
}

fn skip_flag_decode(cx: &mut Ctx, x0_nz: bool, y0_nz: bool, x_cb: usize, y_cb: usize) -> u32 {
    let mut inc = 0;
    if cx.ctb_left || x0_nz {
        if cx.skip_flag[y_cb * cx.min_cb_width + x_cb - 1] != 0 {
            inc += 1;
        }
    }
    if cx.ctb_up || y0_nz {
        if cx.skip_flag[(y_cb - 1) * cx.min_cb_width + x_cb] != 0 {
            inc += 1;
        }
    }
    cx.get(SKIP_FLAG_OFFSET + inc)
}

fn part_mode_decode(cx: &mut Ctx, log2_cb_size: u32) -> u8 {
    if cx.get(PART_MODE_OFFSET) == 1 {
        return 0; // 2Nx2N
    }
    if log2_cb_size == cx.log2_min_cb {
        if cx.cu_pred_mode == MODE_INTRA {
            return PART_NXN;
        }
        if cx.get(PART_MODE_OFFSET + 1) == 1 {
            return 1; // 2NxN
        }
        if log2_cb_size == 3 {
            return 2; // Nx2N
        }
        if cx.get(PART_MODE_OFFSET + 2) == 1 {
            return 2;
        }
        return PART_NXN;
    }
    if !cx.sps.amp_enabled {
        if cx.get(PART_MODE_OFFSET + 1) == 1 {
            return 1;
        }
        return 2;
    }
    if cx.get(PART_MODE_OFFSET + 1) == 1 {
        if cx.get(PART_MODE_OFFSET + 3) == 1 {
            return 1; // 2NxN
        }
        if cx.bypass() == 1 {
            return 5; // 2NxnD
        }
        return 4; // 2NxnU
    }
    if cx.get(PART_MODE_OFFSET + 3) == 1 {
        return 2; // Nx2N
    }
    if cx.bypass() == 1 {
        return 7; // nRx2N
    }
    6 // nLx2N
}

fn prediction_unit(cx: &mut Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, part_idx: u32, log2_cb_size: u32) {
    let x_cb = (x0 >> cx.log2_min_cb) as usize;
    let y_cb = (y0 >> cx.log2_min_cb) as usize;
    let skip = cx.skip_flag[y_cb * cx.min_cb_width + x_cb] != 0;
    if !skip {
        cx.pu_merge_flag = cx.get(MERGE_FLAG_OFFSET) == 1;
    } else {
        cx.pu_merge_flag = true;
    }
    let mut cur = MvField::default();
    if skip || cx.pu_merge_flag {
        let merge_idx = if cx.sh.max_num_merge_cand > 1 { merge_idx_decode(cx) } else { 0 };
        luma_mv_merge_mode(cx, x0, y0, npbw, npbh, log2_cb_size, part_idx as i32, merge_idx, &mut cur);
    } else {
        mvp_mode(cx, x0, y0, npbw, npbh, log2_cb_size, part_idx as i32, &mut cur);
    }
    store_mvf(cx, x0, y0, npbw, npbh, cur);
}

fn store_mvf(cx: &mut Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, mv: MvField) {
    let xp = (x0 >> cx.log2_min_pu) as usize;
    let yp = (y0 >> cx.log2_min_pu) as usize;
    let w = (npbw >> cx.log2_min_pu) as usize;
    let h = (npbh >> cx.log2_min_pu) as usize;
    for j in 0..h {
        for i in 0..w {
            cx.tab_mvf[(yp + j) * cx.min_pu_width + xp + i] = mv;
        }
    }
}

/// Export the frame's motion vectors, replicating FFmpeg
/// `ff_hevc_export_motion_vectors` exactly: greedily merge equal-MV min-PU
/// (4x4) blocks into rectangles (expand right, then down), then for each used
/// reference direction emit src = block centre, dst = src + (mv >> 2),
/// source = ±(ref_idx + 1).
fn export_frame_mvs(cx: &mut Ctx) {
    let mpw = cx.min_pu_width;
    let mph = cx.min_pu_height;
    let min_pu = 1i32 << cx.log2_min_pu;
    let max_pu = 255 / min_pu;
    let frame = cx.frame_index;
    let tab = std::mem::take(&mut cx.tab_mvf);
    let mut flt = MvFilter::new();
    let mut visited = vec![false; mpw * mph];
    for y in 0..mph {
        for x in 0..mpw {
            if visited[y * mpw + x] {
                continue;
            }
            let m = tab[y * mpw + x];
            if m.pred_flag & (PF_L0 | PF_L1) == 0 {
                visited[y * mpw + x] = true;
                continue;
            }
            let matches = |o: &MvField| o.mv == m.mv && o.ref_idx == m.ref_idx && o.pred_flag == m.pred_flag;
            let mut w = 1;
            while w < max_pu as usize && x + w < mpw && !visited[y * mpw + x + w] && matches(&tab[y * mpw + x + w]) {
                w += 1;
            }
            let mut h = 1;
            'down: while h < max_pu as usize && y + h < mph {
                for xx in x..x + w {
                    if visited[(y + h) * mpw + xx] || !matches(&tab[(y + h) * mpw + xx]) {
                        break 'down;
                    }
                }
                h += 1;
            }
            for yy in y..y + h {
                for xx in x..x + w {
                    visited[yy * mpw + xx] = true;
                }
            }
            let pixel_w = w as i32 * min_pu;
            let pixel_h = h as i32 * min_pu;
            if pixel_w > 255 || pixel_h > 255 {
                continue;
            }
            let src_x = x as i32 * min_pu + pixel_w / 2;
            let src_y = y as i32 * min_pu + pixel_h / 2;
            for dir in 0..2 {
                if m.pred_flag & (1 << dir) == 0 {
                    continue;
                }
                let source = if dir == 0 {
                    -(m.ref_idx[0] as i32 + 1)
                } else {
                    m.ref_idx[1] as i32 + 1
                };
                let dst_x = src_x + (m.mv[dir][0] as i32 >> 2);
                let dst_y = src_y + (m.mv[dir][1] as i32 >> 2);
                if dst_x == src_x && dst_y == src_y {
                    continue; // zero-size vector: no displacement, skip
                }
                if !flt.keep(src_x, src_y, dst_x, dst_y, src_x, src_y) {
                    continue; // MV_MIN_SIZE / MV_EVERY_NTH
                }
                cx.out.push(MvCompact {
                    frame,
                    source,
                    src_x: src_x as i16,
                    src_y: src_y as i16,
                    dst_x: dst_x as i16,
                    dst_y: dst_y as i16,
                });
            }
        }
    }
    cx.tab_mvf = tab;
}

fn merge_idx_decode(cx: &mut Ctx) -> u32 {
    let mut i = cx.get(MERGE_IDX_OFFSET);
    if i != 0 {
        while i < cx.sh.max_num_merge_cand - 1 && cx.bypass() == 1 {
            i += 1;
        }
    }
    i
}

/// AMVP: decode inter_pred_idc/ref_idx/mvd/mvp_flag, then mv = predictor + mvd.
fn mvp_mode(cx: &mut Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, log2_cb_size: u32, part_idx: i32, mv: &mut MvField) {
    let is_b = matches!(cx.sh.slice_type, Some(SliceType::B));
    let mut inter_pred_idc = 0u32; // PRED_L0
    if is_b {
        inter_pred_idc = inter_pred_idc_decode(cx, npbw, npbh);
    }
    mv.pred_flag = 0;
    if inter_pred_idc != 1 {
        // != PRED_L1 -> has L0
        if cx.sh.nb_refs[0] > 0 {
            mv.ref_idx[0] = ref_idx_lx_decode(cx, cx.sh.nb_refs[0]) as i8;
        }
        mv.pred_flag = PF_L0;
        let mvd = mvd_coding(cx);
        let mvp_flag = cx.get(MVP_LX_FLAG_OFFSET) as i32;
        let pred = amvp_predictor(cx, x0, y0, npbw, npbh, log2_cb_size, part_idx, 0, mv.ref_idx[0] as i32, mvp_flag);
        mv.mv[0] = [(pred[0] + mvd[0]) as i16, (pred[1] + mvd[1]) as i16];
    }
    if inter_pred_idc != 0 {
        // != PRED_L0 -> has L1
        if cx.sh.nb_refs[1] > 0 {
            mv.ref_idx[1] = ref_idx_lx_decode(cx, cx.sh.nb_refs[1]) as i8;
        }
        let mvd = if cx.sh.mvd_l1_zero_flag && inter_pred_idc == 2 {
            [0, 0]
        } else {
            mvd_coding(cx)
        };
        mv.pred_flag += PF_L1;
        let mvp_flag = cx.get(MVP_LX_FLAG_OFFSET) as i32;
        let pred = amvp_predictor(cx, x0, y0, npbw, npbh, log2_cb_size, part_idx, 1, mv.ref_idx[1] as i32, mvp_flag);
        mv.mv[1] = [(pred[0] + mvd[0]) as i16, (pred[1] + mvd[1]) as i16];
    }
}

fn inter_pred_idc_decode(cx: &mut Ctx, npbw: i32, npbh: i32) -> u32 {
    if npbw + npbh == 12 {
        return cx.get(INTER_PRED_IDC_OFFSET + 4);
    }
    let d = cx.ct_cur_depth as usize;
    if cx.get(INTER_PRED_IDC_OFFSET + d) == 1 {
        return 2; // PRED_BI
    }
    cx.get(INTER_PRED_IDC_OFFSET + 4)
}

fn ref_idx_lx_decode(cx: &mut Ctx, num_ref: u32) -> u32 {
    let max = num_ref as i32 - 1;
    let max_ctx = max.min(2);
    let mut i = 0i32;
    while i < max_ctx && cx.get(REF_IDX_L0_OFFSET + i as usize) == 1 {
        i += 1;
    }
    if i == 2 {
        while i < max && cx.bypass() == 1 {
            i += 1;
        }
    }
    i as u32
}

fn mvd_coding(cx: &mut Ctx) -> [i32; 2] {
    let gx = cx.get(ABS_MVD_GREATER0_FLAG_OFFSET);
    let gy = cx.get(ABS_MVD_GREATER0_FLAG_OFFSET);
    let mut x = gx;
    let mut y = gy;
    if gx != 0 {
        x += cx.get(ABS_MVD_GREATER1_FLAG_OFFSET + 1);
    }
    if gy != 0 {
        y += cx.get(ABS_MVD_GREATER1_FLAG_OFFSET + 1);
    }
    let vx = match x {
        2 => mvd_decode(cx),
        1 => cx.c.bypass_sign(-1),
        _ => 0,
    };
    let vy = match y {
        2 => mvd_decode(cx),
        1 => cx.c.bypass_sign(-1),
        _ => 0,
    };
    [vx, vy]
}

fn mvd_decode(cx: &mut Ctx) -> i32 {
    let mut ret = 2i32;
    let mut k = 1;
    while k < 32 && cx.bypass() == 1 {
        ret += 1 << k;
        k += 1;
    }
    while k > 0 {
        k -= 1;
        ret += (cx.bypass() as i32) << k;
    }
    cx.c.bypass_sign(-ret)
}

// ── MV derivation (ported from FFmpeg libavcodec/hevc/mvs.c) ─────────────────

impl<'a> Ctx<'a> {
    #[inline]
    fn mvf_at(&self, x: i32, y: i32) -> MvField {
        let xp = (x >> self.log2_min_pu) as usize;
        let yp = (y >> self.log2_min_pu) as usize;
        self.tab_mvf[yp * self.min_pu_width + xp]
    }
    #[inline]
    fn ref_poc(&self, list: usize, idx: i8) -> i32 {
        *self.ri.ref_poc[list].get(idx.max(0) as usize).unwrap_or(&0)
    }
    #[inline]
    fn ref_long(&self, list: usize, idx: i8) -> bool {
        *self.ri.ref_long[list].get(idx.max(0) as usize).unwrap_or(&false)
    }
}

fn is_diff_mer(cx: &Ctx, xn: i32, yn: i32, xp: i32, yp: i32) -> bool {
    let p = cx.pps.log2_parallel_merge_level;
    (xn >> p) == (xp >> p) && (yn >> p) == (yp >> p)
}

fn avail_pu(cx: &Ctx, cand: bool, x: i32, y: i32) -> bool {
    cand && cx.mvf_at(x, y).pred_flag != PF_INTRA
}

fn compare_mv_ref_idx(a: &MvField, b: &MvField) -> bool {
    if a.pred_flag != b.pred_flag {
        return false;
    }
    match a.pred_flag {
        PF_BI => a.ref_idx == b.ref_idx && a.mv == b.mv,
        PF_L0 => a.ref_idx[0] == b.ref_idx[0] && a.mv[0] == b.mv[0],
        PF_L1 => a.ref_idx[1] == b.ref_idx[1] && a.mv[1] == b.mv[1],
        _ => false,
    }
}

fn set_neighbour_available(cx: &mut Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32) {
    let ctb = 1i32 << cx.log2_ctb;
    let mask = ctb - 1;
    let x0b = x0 & mask;
    let y0b = y0 & mask;
    cx.na_up = cx.ctb_up || y0b != 0;
    cx.na_left = cx.ctb_left || x0b != 0;
    cx.na_up_left = if x0b != 0 || y0b != 0 {
        cx.na_left && cx.na_up
    } else {
        cx.ctb_up_left
    };
    cx.na_up_right_sap = if x0b + npbw == ctb {
        cx.ctb_up_right && y0b == 0
    } else {
        cx.na_up
    };
    cx.na_up_right = cx.na_up_right_sap && (x0 + npbw) < cx.sps.pic_width_in_luma_samples as i32;
    let eot_y = ((y0 & !mask) + ctb).min(cx.sps.pic_height_in_luma_samples as i32);
    cx.na_bottom_left = if (y0 + npbh) >= eot_y { false } else { cx.na_left };
}

/// min_tb_addr_zs (single tile): CTB raster address × CTB area + within-CTB
/// Morton (z) order — the §6.5.1 z-scan address for non-tiled streams. Uses the
/// precomputed `zs_lut` for the within-CTB Morton code.
#[inline]
fn min_tb_addr_zs(cx: &Ctx, x_tb: i32, y_tb: i32) -> i64 {
    let s = cx.zs_s;
    let ctb_tb = 1i32 << s;
    let ctb_addr = (y_tb >> s) as i64 * cx.ctb_w_i64 + (x_tb >> s) as i64;
    let within = (y_tb & (ctb_tb - 1)) as usize * ctb_tb as usize + (x_tb & (ctb_tb - 1)) as usize;
    ctb_addr * (1i64 << (2 * s)) + cx.zs_lut[within]
}

fn z_scan_block_avail(cx: &Ctx, xc: i32, yc: i32, xn: i32, yn: i32) -> bool {
    let l = cx.log2_ctb;
    if (yn >> l) < (yc >> l) || (xn >> l) < (xc >> l) {
        return true;
    }
    let cur = min_tb_addr_zs(cx, xc >> cx.sps.min_tb_log2_size_y, yc >> cx.sps.min_tb_log2_size_y);
    let n = min_tb_addr_zs(cx, xn >> cx.sps.min_tb_log2_size_y, yn >> cx.sps.min_tb_log2_size_y);
    n <= cur
}

fn clip(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

fn mv_scale(src: [i32; 2], td: i32, tb: i32) -> [i32; 2] {
    let td = clip(td, -128, 127);
    let tb = clip(tb, -128, 127);
    let tx = (0x4000 + (td.abs() / 2)) / td;
    let scale = clip((tb * tx + 32) >> 6, -4096, 4095);
    let f = |s: i32| -> i32 {
        clip((scale * s + 127 + if scale * s < 0 { 1 } else { 0 }) >> 8, -32768, 32767)
    };
    [f(src[0]), f(src[1])]
}

const MRG_MAX: usize = 5;

fn luma_mv_merge_mode(cx: &mut Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, log2_cb_size: u32, part_idx: i32, merge_idx: u32, out: &mut MvField) {
    let (mut x0, mut y0, mut npbw, mut npbh, mut part_idx) = (x0, y0, npbw, npbh, part_idx);
    let ncs = 1i32 << log2_cb_size;
    let npbw2 = npbw;
    let npbh2 = npbh;
    let mut single = false;
    if cx.pps.log2_parallel_merge_level > 2 && ncs == 8 {
        single = true;
        x0 = cx.cu_x;
        y0 = cx.cu_y;
        npbw = ncs;
        npbh = ncs;
        part_idx = 0;
    }
    set_neighbour_available(cx, x0, y0, npbw, npbh);
    let list = derive_spatial_merge(cx, x0, y0, npbw, npbh, single, part_idx, merge_idx as i32);
    let mut cand = list[merge_idx as usize];
    if cand.pred_flag == PF_BI && (npbw2 + npbh2) == 12 {
        cand.pred_flag = PF_L0;
    }
    *out = cand;
}

#[allow(clippy::too_many_arguments)]
fn derive_spatial_merge(cx: &Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, single: bool, part_idx: i32, merge_idx: i32) -> [MvField; MRG_MAX] {
    let mut list = [MvField::default(); MRG_MAX];
    let is_b = matches!(cx.sh.slice_type, Some(SliceType::B));
    let part_mode = cx.cu_part_mode;
    let (x_a1, y_a1) = (x0 - 1, y0 + npbh - 1);
    let (x_b1, y_b1) = (x0 + npbw - 1, y0 - 1);
    let (x_b0, y_b0) = (x0 + npbw, y0 - 1);
    let (x_a0, y_a0) = (x0 - 1, y0 + npbh);
    let (x_b2, y_b2) = (x0 - 1, y0 - 1);
    let nb_refs = if matches!(cx.sh.slice_type, Some(SliceType::P)) {
        cx.sh.nb_refs[0]
    } else {
        cx.sh.nb_refs[0].min(cx.sh.nb_refs[1])
    };
    let mut nb = 0usize;

    // A1 (left)
    let a1_blocked = (!single && part_idx == 1 && matches!(part_mode, PART_NX2N | PART_NLX2N | PART_NRX2N))
        || is_diff_mer(cx, x_a1, y_a1, x0, y0);
    let avail_a1 = !a1_blocked && avail_pu(cx, cx.na_left, x_a1, y_a1);
    if avail_a1 {
        list[nb] = cx.mvf_at(x_a1, y_a1);
        if merge_idx == 0 {
            return list;
        }
        nb += 1;
    }

    // B1 (up)
    let b1_blocked = (!single && part_idx == 1 && matches!(part_mode, PART_2NXN | PART_2NXNU | PART_2NXND))
        || is_diff_mer(cx, x_b1, y_b1, x0, y0);
    let avail_b1 = !b1_blocked && avail_pu(cx, cx.na_up, x_b1, y_b1);
    if avail_b1 && !(avail_a1 && compare_mv_ref_idx(&cx.mvf_at(x_b1, y_b1), &cx.mvf_at(x_a1, y_a1))) {
        list[nb] = cx.mvf_at(x_b1, y_b1);
        if merge_idx == nb as i32 {
            return list;
        }
        nb += 1;
    }

    // B0 (up-right)
    let avail_b0 = avail_pu(cx, cx.na_up_right_sap, x_b0, y_b0)
        && x_b0 < cx.sps.pic_width_in_luma_samples as i32
        && z_scan_block_avail(cx, x0, y0, x_b0, y_b0)
        && !is_diff_mer(cx, x_b0, y_b0, x0, y0);
    if avail_b0 && !(avail_b1 && compare_mv_ref_idx(&cx.mvf_at(x_b0, y_b0), &cx.mvf_at(x_b1, y_b1))) {
        list[nb] = cx.mvf_at(x_b0, y_b0);
        if merge_idx == nb as i32 {
            return list;
        }
        nb += 1;
    }

    // A0 (bottom-left)
    let avail_a0 = avail_pu(cx, cx.na_bottom_left, x_a0, y_a0)
        && y_a0 < cx.sps.pic_height_in_luma_samples as i32
        && z_scan_block_avail(cx, x0, y0, x_a0, y_a0)
        && !is_diff_mer(cx, x_a0, y_a0, x0, y0);
    if avail_a0 && !(avail_a1 && compare_mv_ref_idx(&cx.mvf_at(x_a0, y_a0), &cx.mvf_at(x_a1, y_a1))) {
        list[nb] = cx.mvf_at(x_a0, y_a0);
        if merge_idx == nb as i32 {
            return list;
        }
        nb += 1;
    }

    // B2 (up-left)
    let avail_b2 = avail_pu(cx, cx.na_up_left, x_b2, y_b2) && !is_diff_mer(cx, x_b2, y_b2, x0, y0);
    if avail_b2
        && !(avail_a1 && compare_mv_ref_idx(&cx.mvf_at(x_b2, y_b2), &cx.mvf_at(x_a1, y_a1)))
        && !(avail_b1 && compare_mv_ref_idx(&cx.mvf_at(x_b2, y_b2), &cx.mvf_at(x_b1, y_b1)))
        && nb != 4
    {
        list[nb] = cx.mvf_at(x_b2, y_b2);
        if merge_idx == nb as i32 {
            return list;
        }
        nb += 1;
    }

    // temporal
    if cx.sh.slice_temporal_mvp_enabled && nb < cx.sh.max_num_merge_cand as usize {
        let mut mv0 = [0i32; 2];
        let mut mv1 = [0i32; 2];
        let a0 = temporal_luma_mv(cx, x0, y0, npbw, npbh, 0, 0, &mut mv0);
        let a1 = if is_b {
            temporal_luma_mv(cx, x0, y0, npbw, npbh, 0, 1, &mut mv1)
        } else {
            false
        };
        if a0 || a1 {
            let mut c = MvField::default();
            c.pred_flag = a0 as u8 + ((a1 as u8) << 1);
            c.mv[0] = [mv0[0] as i16, mv0[1] as i16];
            c.mv[1] = [mv1[0] as i16, mv1[1] as i16];
            list[nb] = c;
            if merge_idx == nb as i32 {
                return list;
            }
            nb += 1;
        }
    }

    let nb_orig = nb;
    // combined bi-predictive (B slices)
    if is_b && nb_orig > 1 && nb < cx.sh.max_num_merge_cand as usize {
        const L0L1: [[usize; 2]; 12] = [
            [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
            [0, 3], [3, 0], [1, 3], [3, 1], [2, 3], [3, 2],
        ];
        let mut comb = 0;
        while nb < cx.sh.max_num_merge_cand as usize && comb < nb_orig * (nb_orig - 1) {
            let l0c = list[L0L1[comb][0]];
            let l1c = list[L0L1[comb][1]];
            if (l0c.pred_flag & PF_L0) != 0
                && (l1c.pred_flag & PF_L1) != 0
                && (cx.ref_poc(0, l0c.ref_idx[0]) != cx.ref_poc(1, l1c.ref_idx[1]) || l0c.mv[0] != l1c.mv[1])
            {
                let mut c = MvField::default();
                c.ref_idx[0] = l0c.ref_idx[0];
                c.ref_idx[1] = l1c.ref_idx[1];
                c.pred_flag = PF_BI;
                c.mv[0] = l0c.mv[0];
                c.mv[1] = l1c.mv[1];
                list[nb] = c;
                if merge_idx == nb as i32 {
                    return list;
                }
                nb += 1;
            }
            comb += 1;
        }
    }

    // zero candidates
    let mut zero_idx = 0i8;
    while nb < cx.sh.max_num_merge_cand as usize {
        let mut c = MvField::default();
        c.pred_flag = PF_L0 + (((is_b) as u8) << 1);
        let ri = if (zero_idx as u32) < nb_refs { zero_idx } else { 0 };
        c.ref_idx = [ri, ri];
        list[nb] = c;
        if merge_idx == nb as i32 {
            return list;
        }
        nb += 1;
        zero_idx += 1;
    }
    list
}

/// Temporal collocated MV (TMVP). Returns availability; fills `mv_out`.
fn temporal_luma_mv(cx: &Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, ref_idx: i32, x_list: usize, mv_out: &mut [i32; 2]) -> bool {
    let col = match cx.ri.col {
        Some(c) => c,
        None => return false,
    };
    let col_poc = col.poc;
    // bottom-right collocated
    let mut x = x0 + npbw;
    let mut y = y0 + npbh;
    let try_pos = |x: i32, y: i32, out: &mut [i32; 2]| -> bool {
        let xp = (x >> cx.log2_min_pu) as usize;
        let yp = (y >> cx.log2_min_pu) as usize;
        if xp >= col.min_pu_width || yp >= col.min_pu_height {
            return false;
        }
        let temp = col.tab_mvf[yp * col.min_pu_width + xp];
        derive_temporal_colocated(cx, &temp, ref_idx, x_list, col_poc, col, out)
    };
    if (y0 >> cx.log2_ctb) == (y >> cx.log2_ctb)
        && y < cx.sps.pic_height_in_luma_samples as i32
        && x < cx.sps.pic_width_in_luma_samples as i32
    {
        x &= !15;
        y &= !15;
        if try_pos(x, y, mv_out) {
            return true;
        }
    }
    // center collocated
    x = (x0 + (npbw >> 1)) & !15;
    y = (y0 + (npbh >> 1)) & !15;
    try_pos(x, y, mv_out)
}

fn derive_temporal_colocated(cx: &Ctx, temp: &MvField, ref_idx: i32, x_list: usize, col_poc: i32, col: &FrameMv, out: &mut [i32; 2]) -> bool {
    if temp.pred_flag == PF_INTRA || temp.pred_flag == 0 {
        return false;
    }
    // pick the collocated list (8.5.3.2.9)
    let l = if temp.pred_flag & PF_L0 == 0 {
        1
    } else if temp.pred_flag == PF_L0 {
        0
    } else {
        // BI: depends on whether any cur ref has poc > cur
        let mut has_future = false;
        for j in 0..2 {
            for &p in &cx.ri.ref_poc[j] {
                if p > cx.ri.poc {
                    has_future = true;
                }
            }
        }
        if !has_future {
            x_list
        } else if cx.sh.collocated_list == 1 {
            0
        } else {
            1
        }
    };
    let mv_col = [temp.mv[l][0] as i32, temp.mv[l][1] as i32];
    let cur_lt = cx.ref_long(x_list, ref_idx as i8);
    let col_ref_idx = temp.ref_idx[l];
    let col_lt = *col.ref_long[l].get(col_ref_idx.max(0) as usize).unwrap_or(&false);
    if cur_lt != col_lt {
        *out = [0, 0];
        return false;
    }
    let col_ref_poc = *col.ref_poc[l].get(col_ref_idx.max(0) as usize).unwrap_or(&0);
    let col_diff = col_poc - col_ref_poc;
    let cur_diff = cx.ri.poc - cx.ref_poc(x_list, ref_idx as i8);
    if cur_lt || col_diff == cur_diff || col_diff == 0 {
        *out = mv_col;
    } else {
        *out = mv_scale(mv_col, col_diff, cur_diff);
    }
    true
}

/// AMVP predictor (ITU-T §8.5.3.2.6-8). Returns the chosen MVP.
#[allow(clippy::too_many_arguments)]
fn amvp_predictor(cx: &mut Ctx, x0: i32, y0: i32, npbw: i32, npbh: i32, _log2_cb_size: u32, _part_idx: i32, lx: usize, ref_idx: i32, mvp_flag: i32) -> [i32; 2] {
    set_neighbour_available(cx, x0, y0, npbw, npbh);
    let mut cand = [[0i32; 2]; 2];
    let mut num = 0usize;
    let cur_ref_poc = cx.ref_poc(lx, ref_idx as i8);

    // A candidates (A0 bottom-left, A1 left)
    let (x_a0, y_a0) = (x0 - 1, y0 + npbh);
    let (x_a1, y_a1) = (x0 - 1, y0 + npbh - 1);
    let avail_a0 = avail_pu(cx, cx.na_bottom_left, x_a0, y_a0)
        && y_a0 < cx.sps.pic_height_in_luma_samples as i32
        && z_scan_block_avail(cx, x0, y0, x_a0, y_a0);
    let avail_a1 = avail_pu(cx, cx.na_left, x_a1, y_a1);
    let is_scaled = avail_a0 || avail_a1;

    let mut mxa = [0i32; 2];
    let mut a_ok = false;
    // pass 1: same-poc (no scaling)
    for (av, (x, y)) in [(avail_a0, (x_a0, y_a0)), (avail_a1, (x_a1, y_a1))] {
        if !av {
            continue;
        }
        if mp_mx(cx, x, y, lx, lx, ref_idx, &mut mxa, false) || mp_mx(cx, x, y, 1 - lx, lx, ref_idx, &mut mxa, false) {
            a_ok = true;
            break;
        }
    }
    // pass 2: long-term aware with scaling
    if !a_ok {
        for (av, (x, y)) in [(avail_a0, (x_a0, y_a0)), (avail_a1, (x_a1, y_a1))] {
            if !av {
                continue;
            }
            if mp_mx(cx, x, y, lx, lx, ref_idx, &mut mxa, true) || mp_mx(cx, x, y, 1 - lx, lx, ref_idx, &mut mxa, true) {
                a_ok = true;
                break;
            }
        }
    }

    // B candidates (B0 up-right, B1 up, B2 up-left)
    let (x_b0, y_b0) = (x0 + npbw, y0 - 1);
    let (x_b1, y_b1) = (x0 + npbw - 1, y0 - 1);
    let (x_b2, y_b2) = (x0 - 1, y0 - 1);
    let avail_b0 = avail_pu(cx, cx.na_up_right_sap, x_b0, y_b0)
        && x_b0 < cx.sps.pic_width_in_luma_samples as i32
        && z_scan_block_avail(cx, x0, y0, x_b0, y_b0);
    let avail_b1 = avail_pu(cx, cx.na_up, x_b1, y_b1);
    let avail_b2 = avail_pu(cx, cx.na_up_left, x_b2, y_b2);
    let mut mxb = [0i32; 2];
    let mut b_ok = false;
    for (av, (x, y)) in [(avail_b0, (x_b0, y_b0)), (avail_b1, (x_b1, y_b1)), (avail_b2, (x_b2, y_b2))] {
        if !av {
            continue;
        }
        if mp_mx(cx, x, y, lx, lx, ref_idx, &mut mxb, false) || mp_mx(cx, x, y, 1 - lx, lx, ref_idx, &mut mxb, false) {
            b_ok = true;
            break;
        }
    }

    if !is_scaled {
        if b_ok {
            a_ok = true;
            mxa = mxb;
        }
        b_ok = false;
        for (av, (x, y)) in [(avail_b0, (x_b0, y_b0)), (avail_b1, (x_b1, y_b1)), (avail_b2, (x_b2, y_b2))] {
            if !av || b_ok {
                continue;
            }
            if mp_mx(cx, x, y, lx, lx, ref_idx, &mut mxb, true) || mp_mx(cx, x, y, 1 - lx, lx, ref_idx, &mut mxb, true) {
                b_ok = true;
            }
        }
    }

    if a_ok {
        cand[num] = mxa;
        num += 1;
    }
    if b_ok && (!a_ok || mxa != mxb) {
        cand[num] = mxb;
        num += 1;
    }

    // temporal
    if num < 2 && cx.sh.slice_temporal_mvp_enabled && mvp_flag == num as i32 {
        let mut mc = [0i32; 2];
        if temporal_luma_mv(cx, x0, y0, npbw, npbh, ref_idx, lx, &mut mc) {
            cand[num] = mc;
            num += 1;
        }
    }
    let _ = (cur_ref_poc, num);
    cand[mvp_flag as usize]
}

/// mv_mp_mode_mx / _lt: a neighbour PU's MV usable as an AMVP candidate.
fn mp_mx(cx: &Ctx, x: i32, y: i32, pred_list: usize, ref_idx_curr: usize, ref_idx: i32, out: &mut [i32; 2], long_term: bool) -> bool {
    let nb = cx.mvf_at(x, y);
    if nb.pred_flag & (1 << pred_list) == 0 {
        return false;
    }
    let nb_ref_poc = cx.ref_poc(pred_list, nb.ref_idx[pred_list]);
    let cur_ref_poc = cx.ref_poc(ref_idx_curr, ref_idx as i8);
    if !long_term {
        if nb_ref_poc == cur_ref_poc {
            *out = [nb.mv[pred_list][0] as i32, nb.mv[pred_list][1] as i32];
            return true;
        }
        false
    } else {
        let cur_lt = cx.ref_long(ref_idx_curr, ref_idx as i8);
        let col_lt = cx.ref_long(pred_list, nb.ref_idx[pred_list]);
        if cur_lt == col_lt {
            let mut mv = [nb.mv[pred_list][0] as i32, nb.mv[pred_list][1] as i32];
            if !cur_lt && nb_ref_poc != cur_ref_poc {
                let mut pd = cx.ri.poc - nb_ref_poc;
                if pd == 0 {
                    pd = 1;
                }
                mv = mv_scale(mv, pd, cx.ri.poc - cur_ref_poc);
            }
            *out = mv;
            return true;
        }
        false
    }
}

// ── intra ────────────────────────────────────────────────────────────────────

fn set_ipm_default(cx: &mut Ctx, x0: i32, y0: i32, log2_cb_size: u32) {
    let log2_min_pu = cx.log2_min_cb - 1;
    let size = (1i32 << log2_cb_size) >> log2_min_pu;
    let xp = (x0 >> log2_min_pu) as usize;
    let yp = (y0 >> log2_min_pu) as usize;
    for j in 0..size as usize {
        for i in 0..size as usize {
            cx.ipm[(yp + j) * cx.min_pu_width + xp + i] = 1; // INTRA_DC
        }
    }
}

fn intra_prediction_unit(cx: &mut Ctx, x0: i32, y0: i32, log2_cb_size: u32) {
    let split = cx.cu_part_mode == PART_NXN;
    let pb_size = (1i32 << log2_cb_size) >> split as i32;
    let side = split as usize + 1;
    let mut prev_flag = [false; 4];
    for i in 0..side {
        for j in 0..side {
            prev_flag[2 * i + j] = cx.get(PREV_INTRA_LUMA_PRED_FLAG_OFFSET) == 1;
        }
    }
    for i in 0..side {
        for j in 0..side {
            let mut mpm_idx = 0;
            let mut rem = 0;
            if prev_flag[2 * i + j] {
                // mpm_idx: TU bypass up to 2
                while mpm_idx < 2 && cx.bypass() == 1 {
                    mpm_idx += 1;
                }
            } else {
                rem = cx.bypass();
                for _ in 0..4 {
                    rem = (rem << 1) | cx.bypass();
                }
            }
            let mode = luma_intra_pred_mode(
                cx,
                x0 + pb_size * j as i32,
                y0 + pb_size * i as i32,
                pb_size,
                prev_flag[2 * i + j],
                mpm_idx,
                rem as i32,
            );
            cx.pu_intra_pred_mode[2 * i + j] = mode;
        }
    }
    // chroma intra mode (4:2:0/4:2:2: one intra_chroma_pred_mode per CU). The
    // derived mode_c drives scan_idx_c, so it must be exact.
    const INTRA_CHROMA_TABLE: [u8; 4] = [0, 26, 10, 1];
    if cx.sps.chroma_format_idc != 0 {
        let chroma_mode = intra_chroma_pred_mode_decode(cx);
        let luma0 = cx.pu_intra_pred_mode[0];
        cx.pu_intra_pred_mode_c = if chroma_mode != 4 {
            let t = INTRA_CHROMA_TABLE[chroma_mode as usize];
            if luma0 == t {
                34
            } else {
                t
            }
        } else {
            luma0
        };
    }
}

fn intra_chroma_pred_mode_decode(cx: &mut Ctx) -> u32 {
    if cx.get(INTRA_CHROMA_PRED_MODE_OFFSET) == 0 {
        return 4;
    }
    let mut ret = cx.bypass() << 1;
    ret |= cx.bypass();
    ret
}

fn luma_intra_pred_mode(cx: &mut Ctx, x0: i32, y0: i32, pu_size: i32, prev: bool, mpm_idx: i32, rem: i32) -> u8 {
    const INTRA_DC: i32 = 1;
    const INTRA_PLANAR: i32 = 0;
    let log2_min_pu = cx.log2_min_cb - 1;
    let xp = (x0 >> log2_min_pu) as usize;
    let yp = (y0 >> log2_min_pu) as usize;
    let x0b = (x0 as u32) & ((1 << cx.log2_ctb) - 1);
    let y0b = (y0 as u32) & ((1 << cx.log2_ctb) - 1);
    let y_ctb = (y0 >> cx.log2_ctb) << cx.log2_ctb;

    let mut cand_up = if cx.ctb_up || y0b != 0 {
        cx.ipm[(yp - 1) * cx.min_pu_width + xp] as i32
    } else {
        INTRA_DC
    };
    let cand_left = if cx.ctb_left || x0b != 0 {
        cx.ipm[yp * cx.min_pu_width + xp - 1] as i32
    } else {
        INTRA_DC
    };
    if (y0 - 1) < y_ctb {
        cand_up = INTRA_DC;
    }

    let mut cand = [0i32; 3];
    if cand_left == cand_up {
        if cand_left < 2 {
            cand = [INTRA_PLANAR, INTRA_DC, 26];
        } else {
            cand[0] = cand_left;
            cand[1] = 2 + ((cand_left - 2 - 1 + 32) & 31);
            cand[2] = 2 + ((cand_left - 2 + 1) & 31);
        }
    } else {
        cand[0] = cand_left;
        cand[1] = cand_up;
        if cand[0] != INTRA_PLANAR && cand[1] != INTRA_PLANAR {
            cand[2] = INTRA_PLANAR;
        } else if cand[0] != INTRA_DC && cand[1] != INTRA_DC {
            cand[2] = INTRA_DC;
        } else {
            cand[2] = 26;
        }
    }

    let mode = if prev {
        cand[mpm_idx as usize]
    } else {
        let mut c = cand;
        c.sort_unstable();
        let mut m = rem;
        for &cc in &c {
            if m >= cc {
                m += 1;
            }
        }
        m
    };

    let size = (pu_size >> log2_min_pu).max(1) as usize;
    for i in 0..size {
        for j in 0..size {
            cx.ipm[(yp + i) * cx.min_pu_width + xp + j] = mode as u8;
        }
    }
    mode as u8
}

fn set_ct_depth(cx: &mut Ctx, x0: i32, y0: i32, log2_cb_size: u32) {
    let length = ((1i32 << log2_cb_size) >> cx.log2_min_cb) as usize;
    let x_cb = (x0 >> cx.log2_min_cb) as usize;
    let y_cb = (y0 >> cx.log2_min_cb) as usize;
    let d = cx.ct_cur_depth as u8;
    for y in 0..length {
        for x in 0..length {
            cx.ct_depth[(y_cb + y) * cx.min_cb_width + x_cb + x] = d;
        }
    }
}

// ── transform tree / unit / residual ─────────────────────────────────────────

fn transform_tree(
    cx: &mut Ctx,
    x0: i32,
    y0: i32,
    x_base: i32,
    y_base: i32,
    log2_cb_size: u32,
    log2_trafo_size: u32,
    trafo_depth: i32,
    blk_idx: i32,
    base_cbf_cb: [i32; 2],
    base_cbf_cr: [i32; 2],
) {
    let mut cbf_cb = base_cbf_cb;
    let mut cbf_cr = base_cbf_cr;

    if cx.cu_intra_split && trafo_depth == 1 {
        cx.tu_intra_pred_mode = cx.pu_intra_pred_mode[blk_idx as usize];
    } else if !cx.cu_intra_split {
        cx.tu_intra_pred_mode = cx.pu_intra_pred_mode[0];
    }
    cx.tu_intra_pred_mode_c = cx.pu_intra_pred_mode_c; // 4:2:0/4:2:2: one per CU

    let split_transform_flag;
    if log2_trafo_size <= cx.log2_max_tb
        && log2_trafo_size > cx.log2_min_tb
        && trafo_depth < cx.cu_max_trafo_depth
        && !(cx.cu_intra_split && trafo_depth == 0)
    {
        split_transform_flag = cx.get(SPLIT_TRANSFORM_FLAG_OFFSET + (5 - log2_trafo_size) as usize) == 1;
    } else {
        let inter_split = cx.sps.max_transform_hierarchy_depth_inter == 0
            && cx.cu_pred_mode == MODE_INTER
            && cx.cu_part_mode != PART_2NX2N
            && trafo_depth == 0;
        split_transform_flag = log2_trafo_size > cx.log2_max_tb
            || (cx.cu_intra_split && trafo_depth == 0)
            || inter_split;
    }

    let chroma = cx.sps.chroma_format_idc != 0;
    if chroma && (log2_trafo_size > 2 || cx.sps.chroma_format_idc == 3) {
        if trafo_depth == 0 || cbf_cb[0] != 0 {
            cbf_cb[0] = cx.get(CBF_CB_CR_OFFSET + trafo_depth as usize) as i32;
        }
        if trafo_depth == 0 || cbf_cr[0] != 0 {
            cbf_cr[0] = cx.get(CBF_CB_CR_OFFSET + trafo_depth as usize) as i32;
        }
    }

    if split_transform_flag {
        let split = 1i32 << (log2_trafo_size - 1);
        let x1 = x0 + split;
        let y1 = y0 + split;
        transform_tree(cx, x0, y0, x0, y0, log2_cb_size, log2_trafo_size - 1, trafo_depth + 1, 0, cbf_cb, cbf_cr);
        transform_tree(cx, x1, y0, x0, y0, log2_cb_size, log2_trafo_size - 1, trafo_depth + 1, 1, cbf_cb, cbf_cr);
        transform_tree(cx, x0, y1, x0, y0, log2_cb_size, log2_trafo_size - 1, trafo_depth + 1, 2, cbf_cb, cbf_cr);
        transform_tree(cx, x1, y1, x0, y0, log2_cb_size, log2_trafo_size - 1, trafo_depth + 1, 3, cbf_cb, cbf_cr);
    } else {
        let mut cbf_luma = 1;
        if cx.cu_pred_mode == MODE_INTRA || trafo_depth != 0 || cbf_cb[0] != 0 || cbf_cr[0] != 0 {
            cbf_luma = cx.get(CBF_LUMA_OFFSET + (trafo_depth == 0) as usize) as i32;
        }
        transform_unit(cx, x0, y0, x_base, y_base, log2_trafo_size, blk_idx, cbf_luma, cbf_cb, cbf_cr);
    }
}

fn transform_unit(
    cx: &mut Ctx,
    x0: i32,
    y0: i32,
    x_base: i32,
    y_base: i32,
    log2_trafo_size: u32,
    blk_idx: i32,
    cbf_luma: i32,
    cbf_cb: [i32; 2],
    cbf_cr: [i32; 2],
) {
    let log2_tc = log2_trafo_size - 1; // 4:2:0 chroma
    let chroma = cx.sps.chroma_format_idc != 0;
    let any = cbf_luma != 0 || cbf_cb[0] != 0 || cbf_cr[0] != 0;
    if any {
        if cx.pps.cu_qp_delta_enabled_flag && !cx.tu_is_cu_qp_delta_coded {
            cu_qp_delta_abs(cx);
            cx.tu_is_cu_qp_delta_coded = true;
        }
        // cu_chroma_qp_offset (rext): sh.cu_chroma_qp_offset_enabled assumed off.

        let mut scan_idx = Scan::Diag;
        let mut scan_idx_c = Scan::Diag;
        if cx.cu_pred_mode == MODE_INTRA && log2_trafo_size < 4 {
            let m = cx.tu_intra_pred_mode as i32;
            if (6..=14).contains(&m) {
                scan_idx = Scan::Vert;
            } else if (22..=30).contains(&m) {
                scan_idx = Scan::Horiz;
            }
            let mc = cx.tu_intra_pred_mode_c as i32;
            if (6..=14).contains(&mc) {
                scan_idx_c = Scan::Vert;
            } else if (22..=30).contains(&mc) {
                scan_idx_c = Scan::Horiz;
            }
        }

        if cbf_luma != 0 {
            residual_coding(cx, log2_trafo_size, scan_idx, 0);
        }
        if chroma && (log2_trafo_size > 2 || cx.sps.chroma_format_idc == 3) {
            if cbf_cb[0] != 0 {
                residual_coding(cx, log2_tc, scan_idx_c, 1);
            }
            if cbf_cr[0] != 0 {
                residual_coding(cx, log2_tc, scan_idx_c, 2);
            }
        } else if chroma && blk_idx == 3 {
            let _ = (x_base, y_base);
            if cbf_cb[0] != 0 {
                residual_coding(cx, log2_trafo_size, scan_idx_c, 1);
            }
            if cbf_cr[0] != 0 {
                residual_coding(cx, log2_trafo_size, scan_idx_c, 2);
            }
        }
    }
    let _ = (x0, y0);
}

fn cu_qp_delta_abs(cx: &mut Ctx) {
    let mut prefix = 0;
    let mut inc = 0;
    while prefix < 5 && cx.get(CU_QP_DELTA_OFFSET + inc) == 1 {
        prefix += 1;
        inc = 1;
    }
    if prefix >= 5 {
        let mut k = 0;
        while k < 7 && cx.bypass() == 1 {
            k += 1;
        }
        while k > 0 {
            k -= 1;
            cx.bypass();
        }
    }
    if prefix >= 1 {
        // cu_qp_delta != 0 -> sign flag
        cx.bypass();
    }
}

fn residual_coding(cx: &mut Ctx, log2_trafo_size: u32, scan_idx: Scan, c_idx: usize) {
    let trafo_size = 1i32 << log2_trafo_size;
    let mut transform_skip_flag = false;
    if !cx.cu_transquant_bypass && cx.pps.transform_skip_enabled_flag && log2_trafo_size <= 2 {
        transform_skip_flag = cx.get(TRANSFORM_SKIP_FLAG_OFFSET + (c_idx != 0) as usize) == 1;
    }
    let _ = transform_skip_flag;

    // last significant coeff prefix
    let (mut last_x, mut last_y) = last_sig_coeff_prefix(cx, c_idx, log2_trafo_size);
    if last_x > 3 {
        let suffix = last_sig_coeff_suffix(cx, last_x);
        last_x = (1 << ((last_x >> 1) - 1)) * (2 + (last_x & 1)) + suffix;
    }
    if last_y > 3 {
        let suffix = last_sig_coeff_suffix(cx, last_y);
        last_y = (1 << ((last_y >> 1) - 1)) * (2 + (last_y & 1)) + suffix;
    }
    if scan_idx == Scan::Vert {
        std::mem::swap(&mut last_x, &mut last_y);
    }

    let x_cg_last = (last_x >> 2) as usize;
    let y_cg_last = (last_y >> 2) as usize;

    let (scan_x_cg, scan_y_cg, scan_x_off, scan_y_off): (&[u8], &[u8], &[u8], &[u8]);
    let mut num_coeff;
    match scan_idx {
        Scan::Diag => {
            let lxc = (last_x & 3) as usize;
            let lyc = (last_y & 3) as usize;
            scan_x_off = &DIAG4_X;
            scan_y_off = &DIAG4_Y;
            num_coeff = DIAG4_INV[lyc][lxc] as i32;
            match trafo_size {
                4 => {
                    scan_x_cg = &SCAN_1X1;
                    scan_y_cg = &SCAN_1X1;
                }
                8 => {
                    num_coeff += (DIAG2_INV[y_cg_last][x_cg_last] as i32) << 4;
                    scan_x_cg = &DIAG2_X;
                    scan_y_cg = &DIAG2_Y;
                }
                16 => {
                    num_coeff += (DIAG4_INV[y_cg_last][x_cg_last] as i32) << 4;
                    scan_x_cg = &DIAG4_X;
                    scan_y_cg = &DIAG4_Y;
                }
                _ => {
                    num_coeff += (DIAG8_INV[y_cg_last][x_cg_last] as i32) << 4;
                    scan_x_cg = &DIAG8_X;
                    scan_y_cg = &DIAG8_Y;
                }
            }
        }
        Scan::Horiz => {
            scan_x_cg = &HORIZ2_X;
            scan_y_cg = &HORIZ2_Y;
            scan_x_off = &HORIZ4_X;
            scan_y_off = &HORIZ4_Y;
            num_coeff = HORIZ8_INV[last_y as usize][last_x as usize] as i32;
        }
        Scan::Vert => {
            scan_x_cg = &HORIZ2_Y;
            scan_y_cg = &HORIZ2_X;
            scan_x_off = &HORIZ4_Y;
            scan_y_off = &HORIZ4_X;
            num_coeff = HORIZ8_INV[last_x as usize][last_y as usize] as i32;
        }
    }
    num_coeff += 1;
    let num_last_subset = (num_coeff - 1) >> 4;

    let mut sig_cg = [[0u8; 8]; 8];
    let cg_dim = (1i32 << (log2_trafo_size - 2)) - 1;

    cx.greater1_ctx = 1;
    for i in (0..=num_last_subset).rev() {
        let offset = i << 4;
        let x_cg = scan_x_cg[i as usize] as usize;
        let y_cg = scan_y_cg[i as usize] as usize;
        let mut implicit_non_zero = false;

        if i < num_last_subset && i > 0 {
            let mut ctx_cg = 0;
            if (x_cg as i32) < cg_dim {
                ctx_cg += sig_cg[x_cg + 1][y_cg];
            }
            if (y_cg as i32) < cg_dim {
                ctx_cg += sig_cg[x_cg][y_cg + 1];
            }
            let inc = (ctx_cg.min(1) as usize) + if c_idx > 0 { 2 } else { 0 };
            sig_cg[x_cg][y_cg] = cx.get(SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET + inc) as u8;
            implicit_non_zero = true;
        } else {
            sig_cg[x_cg][y_cg] =
                ((x_cg == x_cg_last && y_cg == y_cg_last) || (x_cg == 0 && y_cg == 0)) as u8;
        }

        let last_scan_pos = num_coeff - offset - 1;
        let mut n_end;
        let mut sig_idx = [0u8; 16];
        let mut nb_sig = 0usize;
        if i == num_last_subset {
            n_end = last_scan_pos - 1;
            sig_idx[0] = last_scan_pos as u8;
            nb_sig = 1;
        } else {
            n_end = 15;
        }

        let mut prev_sig = 0;
        let lim = (((1i32 << log2_trafo_size) - 1) >> 2) as usize;
        if x_cg < lim {
            prev_sig = (sig_cg[x_cg + 1][y_cg] != 0) as i32;
        }
        if y_cg < lim {
            prev_sig += ((sig_cg[x_cg][y_cg + 1] != 0) as i32) << 1;
        }

        if sig_cg[x_cg][y_cg] != 0 && n_end >= 0 {
            let mut scf_offset = 0usize;
            let ctx_map_row: usize;
            if c_idx != 0 {
                scf_offset = 27;
            }
            if log2_trafo_size == 2 {
                ctx_map_row = 0;
            } else {
                ctx_map_row = (prev_sig + 1) as usize;
                if c_idx == 0 {
                    if x_cg > 0 || y_cg > 0 {
                        scf_offset += 3;
                    }
                    if log2_trafo_size == 3 {
                        scf_offset += if scan_idx == Scan::Diag { 9 } else { 15 };
                    } else {
                        scf_offset += 21;
                    }
                } else if log2_trafo_size == 3 {
                    scf_offset += 9;
                } else {
                    scf_offset += 12;
                }
            }
            let mut n = n_end;
            while n > 0 {
                let xc = scan_x_off[n as usize] as usize;
                let yc = scan_y_off[n as usize] as usize;
                let inc = CTX_IDX_MAP[ctx_map_row][(yc << 2) + xc] as usize + scf_offset;
                if cx.get(SIGNIFICANT_COEFF_FLAG_OFFSET + inc) == 1 {
                    sig_idx[nb_sig] = n as u8;
                    nb_sig += 1;
                    implicit_non_zero = false;
                }
                n -= 1;
            }
            if !implicit_non_zero {
                let scf0 = if c_idx == 0 {
                    if i == 0 { 0 } else { 2 + scf_offset }
                } else if i == 0 {
                    27
                } else {
                    2 + scf_offset
                };
                if cx.get(SIGNIFICANT_COEFF_FLAG_OFFSET + scf0) == 1 {
                    sig_idx[nb_sig] = 0;
                    nb_sig += 1;
                }
            } else {
                sig_idx[nb_sig] = 0;
                nb_sig += 1;
            }
        }

        n_end = nb_sig as i32;
        if n_end > 0 {
            let mut ctx_set = if i > 0 && c_idx == 0 { 2 } else { 0 };
            if i != num_last_subset && cx.greater1_ctx == 0 {
                ctx_set += 1;
            }
            cx.greater1_ctx = 1;
            let last_nz = sig_idx[0] as i32;
            let mut g1 = [0i32; 8];
            let mut first_g1 = -1i32;
            let lim_m = if n_end > 8 { 8 } else { n_end };
            for m in 0..lim_m {
                let inc = ((ctx_set << 2) + cx.greater1_ctx) as usize;
                let cidx_inc = if c_idx > 0 { inc + 16 } else { inc };
                g1[m as usize] = cx.get(COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET + cidx_inc) as i32;
                if g1[m as usize] != 0 {
                    cx.greater1_ctx = 0;
                    if first_g1 == -1 {
                        first_g1 = m;
                    }
                } else if cx.greater1_ctx > 0 && cx.greater1_ctx < 3 {
                    cx.greater1_ctx += 1;
                }
            }
            let first_nz = sig_idx[(n_end - 1) as usize] as i32;
            let sign_hidden = if cx.cu_transquant_bypass {
                false
            } else {
                last_nz - first_nz >= 4
            };
            if first_g1 != -1 {
                let inc = ctx_set as usize + if c_idx > 0 { 4 } else { 0 };
                g1[first_g1 as usize] += cx.get(COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc) as i32;
            }
            let nb_signs = if !cx.pps.sign_data_hiding_enabled || !sign_hidden {
                nb_sig
            } else {
                nb_sig - 1
            };
            for _ in 0..nb_signs {
                cx.bypass(); // coeff_sign_flag
            }

            // level remaining (mv_only path: track rice param)
            let mut c_rice = 0u32;
            for m in 0..n_end {
                let tcl;
                if m < 8 {
                    tcl = 1 + g1[m as usize];
                    let thr = if m == first_g1 { 3 } else { 2 };
                    if tcl == thr {
                        let rem = coeff_abs_level_remaining(cx, c_rice);
                        let tcl2 = tcl + rem;
                        if tcl2 > (3 << c_rice) {
                            c_rice = (c_rice + 1).min(4);
                        }
                    }
                } else {
                    let rem = coeff_abs_level_remaining(cx, c_rice);
                    tcl = 1 + rem;
                    if tcl > (3 << c_rice) as i32 {
                        c_rice = (c_rice + 1).min(4);
                    }
                }
            }
        }
    }
}

fn last_sig_coeff_prefix(cx: &mut Ctx, c_idx: usize, log2_size: u32) -> (i32, i32) {
    let max = (log2_size << 1) as i32 - 1;
    let (ctx_offset, ctx_shift) = if c_idx == 0 {
        (3 * (log2_size as i32 - 2) + ((log2_size as i32 - 1) >> 2), (log2_size as i32 + 1) >> 2)
    } else {
        (15, log2_size as i32 - 2)
    };
    let mut x = 0i32;
    while x < max
        && cx.get(LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET + (x >> ctx_shift) as usize + ctx_offset as usize) == 1
    {
        x += 1;
    }
    let mut y = 0i32;
    while y < max
        && cx.get(LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET + (y >> ctx_shift) as usize + ctx_offset as usize) == 1
    {
        y += 1;
    }
    (x, y)
}

fn last_sig_coeff_suffix(cx: &mut Ctx, prefix: i32) -> i32 {
    let length = (prefix >> 1) - 1;
    let mut value = cx.bypass() as i32;
    for _ in 1..length {
        value = (value << 1) | cx.bypass() as i32;
    }
    value
}

fn coeff_abs_level_remaining(cx: &mut Ctx, rice: u32) -> i32 {
    let mut prefix = 0;
    while prefix < 32 && cx.bypass() == 1 {
        prefix += 1;
    }
    if prefix < 3 {
        let mut suffix = 0i32;
        for _ in 0..rice {
            suffix = (suffix << 1) | cx.bypass() as i32;
        }
        (prefix << rice) as i32 + suffix
    } else {
        let pm3 = prefix - 3;
        let mut suffix = 0i32;
        for _ in 0..(pm3 + rice) {
            suffix = (suffix << 1) | cx.bypass() as i32;
        }
        ((((1u32 << pm3) + 3 - 1) << rice) as i32) + suffix
    }
}
