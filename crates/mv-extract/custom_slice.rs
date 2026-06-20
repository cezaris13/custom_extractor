//! CAVLC slice_data: macroblock layer + MV prediction (thesis Ch4 §4.3/§4.4.1).
//!
//! Decodes P/I slices for 4:2:0, frame-only (non-MBAFF) streams — the thesis's
//! Baseline target. Builds per-frame 4x4-block grids of motion vectors, ref
//! indices and non-zero counts, then exports motion vectors in exactly FFmpeg's
//! `AV_FRAME_DATA_MOTION_VECTORS` layout (see libavcodec/mpegutils.c) so the
//! output is directly comparable to extractor1's. CABAC and B slices are out of
//! scope here (rungs 3/4); the caller skips them.
//!
//! Sync correctness is self-checked: after a single-slice picture, exactly
//! `mb_w*mb_h` macroblocks must have been consumed and the reader must sit on
//! the rbsp trailing bits. Everything (mb_type branching, residual `nC`,
//! prediction availability) is ported from the ITU-T H.264 spec and FFmpeg's
//! h264_cavlc.c / h264_mvpred.h.

use super::cavlc::residual_block;
use super::{BitReader, Pps, SliceHeader, SliceType, Sps};
use mv_types::motion_vector::{MotionVector, MvCompact};

const PART_NA: i32 = -2; // PART_NOT_AVAILABLE
const INTRA: i32 = -1; // LIST_NOT_USED (intra / not list-0)

#[derive(Clone, Copy, Debug, PartialEq)]
enum Seg {
    None,
    Intra,
    P16x16,
    P16x8,
    P8x16,
    P8x8,
    Skip,
}

/// Per-frame decode state. All grids are indexed in 4x4-block raster units.
pub struct FrameGrids {
    mb_w: usize,
    mb_h: usize,
    bw: usize, // luma blocks per row = mb_w*4
    bh: usize,
    cw: usize, // chroma blocks per row = mb_w*2
    mv: Vec<[i32; 2]>,    // list 0 motion, per luma 4x4 block
    refi: Vec<i32>,       // list 0 ref idx: >=0 inter, INTRA, or unset
    nnz_l: Vec<u8>,       // luma non-zero counts
    nnz_c: [Vec<u8>; 2],  // chroma (Cb, Cr) non-zero counts
    seg: Vec<Seg>,        // per-MB segmentation (for export)
    slice_id: Vec<i32>,   // per-MB slice number
    decoded: Vec<bool>,   // per-MB decoded flag
    // CABAC-only neighbour context (unused by CAVLC):
    mvd_l: Vec<[i32; 2]>, // |mvd| per luma 4x4 block, for the mvd context
    cbp_mb: Vec<u16>,     // per-MB coded_block_pattern incl. DC flags, for cbf ctx
    chroma_pred: Vec<u8>, // per-MB intra chroma pred mode, for its CABAC context
}

#[inline]
fn block_xy(idx: usize) -> (usize, usize) {
    let i8x8 = idx / 4;
    let i4x4 = idx % 4;
    ((i8x8 & 1) * 2 + (i4x4 & 1), (i8x8 >> 1) * 2 + (i4x4 >> 1))
}

#[inline]
fn xy_scan(bx: usize, by: usize) -> usize {
    let i8x8 = (bx / 2) + (by / 2) * 2;
    let i4x4 = (bx & 1) + (by & 1) * 2;
    i4x4 + 4 * i8x8
}

#[inline]
fn mid_pred(a: i32, b: i32, c: i32) -> i32 {
    a + b + c - a.min(b).min(c) - a.max(b).max(c)
}

impl FrameGrids {
    pub fn new(mb_w: usize, mb_h: usize) -> Self {
        let bw = mb_w * 4;
        let bh = mb_h * 4;
        let cw = mb_w * 2;
        let ch = mb_h * 2;
        FrameGrids {
            mb_w,
            mb_h,
            bw,
            bh,
            cw,
            mv: vec![[0, 0]; bw * bh],
            refi: vec![PART_NA; bw * bh],
            nnz_l: vec![0; bw * bh],
            nnz_c: [vec![0; cw * ch], vec![0; cw * ch]],
            seg: vec![Seg::None; mb_w * mb_h],
            slice_id: vec![-1; mb_w * mb_h],
            decoded: vec![false; mb_w * mb_h],
            mvd_l: vec![[0, 0]; bw * bh],
            cbp_mb: vec![0; mb_w * mb_h],
            chroma_pred: vec![0; mb_w * mb_h],
        }
    }

    pub fn mb_count(&self) -> usize {
        self.mb_w * self.mb_h
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.mb_w, self.mb_h)
    }

    /// Reuse this allocation for the next picture instead of reallocating five
    /// large Vecs every frame. `mv`/`refi` are read only through availability
    /// checks gated on `decoded[]`, so they need no reset. `nnz_*`/`mvd_l` are
    /// read by neighbour contexts of *decoded* MBs that don't themselves write
    /// them (skip/intra), so they must be zeroed — done here in bulk rather than
    /// per-MB, which is cheaper on skip-heavy frames.
    pub fn reset(&mut self, cabac: bool) {
        self.decoded.fill(false);
        self.slice_id.fill(-1);
        self.seg.fill(Seg::None);
        self.cbp_mb.fill(0);
        self.chroma_pred.fill(0);
        self.nnz_l.fill(0);
        self.nnz_c[0].fill(0);
        self.nnz_c[1].fill(0);
        // mvd_l is read only by the CABAC mvd context; skip the (large) clear
        // for CAVLC pictures, which never touch it.
        if cabac {
            self.mvd_l.fill([0, 0]);
        }
    }

    #[inline]
    fn lidx(&self, gx: usize, gy: usize) -> usize {
        gy * self.bw + gx
    }

    /// Is luma 4x4 block (gx,gy) available as a neighbour of the block at
    /// scan-order `cur_scan` in macroblock `cur_mb` (slice `sid`)? Implements
    /// raster + within-MB scan availability (covers the top-right diagonal).
    fn avail(&self, gx: i32, gy: i32, cur_mb: usize, cur_scan: usize, sid: i32) -> bool {
        if gx < 0 || gy < 0 || gx >= self.bw as i32 || gy >= self.bh as i32 {
            return false;
        }
        let (gx, gy) = (gx as usize, gy as usize);
        let m = (gx / 4) + (gy / 4) * self.mb_w;
        if m == cur_mb {
            xy_scan(gx % 4, gy % 4) < cur_scan
        } else {
            m < cur_mb && self.decoded[m] && self.slice_id[m] == sid
        }
    }

    /// ref idx of a neighbour block: >=0 inter, INTRA, or PART_NA if unavailable.
    fn nref(&self, gx: i32, gy: i32, cur_mb: usize, cur_scan: usize, sid: i32) -> i32 {
        if self.avail(gx, gy, cur_mb, cur_scan, sid) {
            self.refi[self.lidx(gx as usize, gy as usize)]
        } else {
            PART_NA
        }
    }

    fn nmv(&self, gx: i32, gy: i32, cur_mb: usize, cur_scan: usize, sid: i32) -> [i32; 2] {
        if self.avail(gx, gy, cur_mb, cur_scan, sid) {
            self.mv[self.lidx(gx as usize, gy as usize)]
        } else {
            [0, 0]
        }
    }

    /// pred_motion (ITU-T §8.4.1.3 / FFmpeg pred_motion) for a partition whose
    /// origin 4x4 block is (bx0,by0) inside MB (mbx,mby), part width `pw` (in
    /// 4x4 units), given partition ref index `refr`.
    fn pred_motion(
        &self,
        mbx: usize,
        mby: usize,
        bx0: usize,
        by0: usize,
        pw: i32,
        refr: i32,
        sid: i32,
    ) -> (i32, i32) {
        let cur_mb = mby * self.mb_w + mbx;
        let cur_scan = xy_scan(bx0, by0);
        let x0 = (mbx * 4 + bx0) as i32;
        let y0 = (mby * 4 + by0) as i32;

        let left_ref = self.nref(x0 - 1, y0, cur_mb, cur_scan, sid);
        let a = self.nmv(x0 - 1, y0, cur_mb, cur_scan, sid);
        let top_ref = self.nref(x0, y0 - 1, cur_mb, cur_scan, sid);
        let b = self.nmv(x0, y0 - 1, cur_mb, cur_scan, sid);

        // diagonal C: top-right, else top-left
        let tr_ref = self.nref(x0 + pw, y0 - 1, cur_mb, cur_scan, sid);
        let (diag_ref, c) = if tr_ref != PART_NA {
            (tr_ref, self.nmv(x0 + pw, y0 - 1, cur_mb, cur_scan, sid))
        } else {
            (
                self.nref(x0 - 1, y0 - 1, cur_mb, cur_scan, sid),
                self.nmv(x0 - 1, y0 - 1, cur_mb, cur_scan, sid),
            )
        };

        let match_count =
            (diag_ref == refr) as i32 + (top_ref == refr) as i32 + (left_ref == refr) as i32;
        if match_count > 1 {
            (mid_pred(a[0], b[0], c[0]), mid_pred(a[1], b[1], c[1]))
        } else if match_count == 1 {
            if left_ref == refr {
                (a[0], a[1])
            } else if top_ref == refr {
                (b[0], b[1])
            } else {
                (c[0], c[1])
            }
        } else if top_ref == PART_NA && diag_ref == PART_NA && left_ref != PART_NA {
            (a[0], a[1])
        } else {
            (mid_pred(a[0], b[0], c[0]), mid_pred(a[1], b[1], c[1]))
        }
    }

    /// pred_pskip_motion (ITU-T §8.4.1.1 / FFmpeg pred_pskip_motion).
    fn pred_pskip(&self, mbx: usize, mby: usize, sid: i32) -> (i32, i32) {
        let cur_mb = mby * self.mb_w + mbx;
        let scan0 = 0usize; // MB origin block
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4) as i32;

        // left
        let left_ref = self.nref(x0 - 1, y0, cur_mb, scan0, sid);
        let a = self.nmv(x0 - 1, y0, cur_mb, scan0, sid);
        if left_ref == PART_NA {
            return (0, 0); // left MB unavailable -> zero MV
        }
        if left_ref == 0 && a == [0, 0] {
            return (0, 0);
        }
        // top
        let top_ref = self.nref(x0, y0 - 1, cur_mb, scan0, sid);
        let b = self.nmv(x0, y0 - 1, cur_mb, scan0, sid);
        if top_ref == PART_NA {
            return (0, 0);
        }
        if top_ref == 0 && b == [0, 0] {
            return (0, 0);
        }
        // diagonal: top-right else top-left
        let tr_ref = self.nref(x0 + 4, y0 - 1, cur_mb, scan0, sid);
        let (diag_ref, c) = if tr_ref != PART_NA {
            (tr_ref, self.nmv(x0 + 4, y0 - 1, cur_mb, scan0, sid))
        } else {
            (
                self.nref(x0 - 1, y0 - 1, cur_mb, scan0, sid),
                self.nmv(x0 - 1, y0 - 1, cur_mb, scan0, sid),
            )
        };

        let match_count =
            (left_ref == 0) as i32 + (top_ref == 0) as i32 + (diag_ref == 0) as i32;
        if match_count > 1 {
            (mid_pred(a[0], b[0], c[0]), mid_pred(a[1], b[1], c[1]))
        } else if match_count == 1 {
            if left_ref == 0 {
                (a[0], a[1])
            } else if top_ref == 0 {
                (b[0], b[1])
            } else {
                (c[0], c[1])
            }
        } else {
            (mid_pred(a[0], b[0], c[0]), mid_pred(a[1], b[1], c[1]))
        }
    }

    /// nC for a luma block, FFmpeg pred_non_zero_count (sentinel 64 = N/A).
    fn nnz_pred_luma(&self, gx: usize, gy: usize, cur_mb: usize, sid: i32) -> i32 {
        let scan = xy_scan(gx % 4, gy % 4);
        let l = if self.avail(gx as i32 - 1, gy as i32, cur_mb, scan, sid) {
            self.nnz_l[self.lidx(gx - 1, gy)] as i32
        } else {
            64
        };
        let t = if self.avail(gx as i32, gy as i32 - 1, cur_mb, scan, sid) {
            self.nnz_l[self.lidx(gx, gy - 1)] as i32
        } else {
            64
        };
        let i = l + t;
        if i < 64 {
            (i + 1) >> 1
        } else {
            i & 31
        }
    }

    fn nnz_pred_chroma(&self, c: usize, cx: usize, cy: usize, cur_mb: usize, sid: i32) -> i32 {
        // chroma availability mirrors luma at MB granularity (2x2 blocks/MB)
        let ch = self.nnz_c[c].len() / self.cw;
        let avail = |x: i32, y: i32| -> Option<usize> {
            if x < 0 || y < 0 || x >= self.cw as i32 || y >= ch as i32 {
                return None;
            }
            let m = (x as usize / 2) + (y as usize / 2) * self.mb_w;
            if m == cur_mb {
                Some(y as usize * self.cw + x as usize)
            } else if m < cur_mb && self.decoded[m] && self.slice_id[m] == sid {
                Some(y as usize * self.cw + x as usize)
            } else {
                None
            }
        };
        let l = avail(cx as i32 - 1, cy as i32).map_or(64, |i| self.nnz_c[c][i] as i32);
        let t = avail(cx as i32, cy as i32 - 1).map_or(64, |i| self.nnz_c[c][i] as i32);
        let i = l + t;
        if i < 64 {
            (i + 1) >> 1
        } else {
            i & 31
        }
    }

    fn fill_block(&mut self, mbx: usize, mby: usize, bx: usize, by: usize, w: usize, h: usize, mv: [i32; 2], refr: i32) {
        for dy in 0..h {
            for dx in 0..w {
                let gx = mbx * 4 + bx + dx;
                let gy = mby * 4 + by + dy;
                let idx = gy * self.bw + gx;
                self.mv[idx] = mv;
                self.refi[idx] = refr;
            }
        }
    }

    fn set_mb_refi(&mut self, mbx: usize, mby: usize, refr: i32) {
        for by in 0..4 {
            for bx in 0..4 {
                let idx = (mby * 4 + by) * self.bw + (mbx * 4 + bx);
                self.refi[idx] = refr;
                if refr == INTRA {
                    self.mv[idx] = [0, 0];
                    // mvd_l stays 0 (zeroed once per frame in reset()).
                }
            }
        }
    }
}

/// Result of decoding one slice: how many MBs were consumed (for the sync
/// oracle) plus whether parsing stayed valid.
pub struct SliceResult {
    pub mbs_decoded: usize,
    pub ok: bool,
}

/// Decode a CAVLC P/I slice into `g`. `r` must sit at slice_data() (i.e. just
/// after the parsed slice header). `sid` is this slice's number.
pub fn decode_slice(
    g: &mut FrameGrids,
    r: &mut BitReader,
    sh: &SliceHeader,
    _sps: &Sps,
    pps: &Pps,
    sid: i32,
) -> SliceResult {
    let pmbs = g.mb_w * g.mb_h;
    let inter = matches!(sh.slice_type, SliceType::P | SliceType::Sp);
    let mut cur = sh.first_mb_in_slice as usize;
    let mut count = 0usize;
    let mut ok = true;

    'slice: loop {
        if cur >= pmbs {
            break;
        }
        if inter {
            let run = r.read_ue() as usize;
            for _ in 0..run {
                if cur >= pmbs {
                    break 'slice;
                }
                decode_p_skip(g, cur, sid);
                cur += 1;
                count += 1;
            }
            if run > 0 && !r.more_rbsp_data() {
                break;
            }
            if cur >= pmbs {
                break;
            }
        }
        if !decode_macroblock(g, r, cur, sid, sh, pps) {
                ok = false;
            break;
        }
        cur += 1;
        count += 1;
        if !r.more_rbsp_data() {
            break;
        }
    }

    SliceResult {
        mbs_decoded: count,
        ok,
    }
}

fn decode_p_skip(g: &mut FrameGrids, mb: usize, sid: i32) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    g.slice_id[mb] = sid;
    let (mx, my) = g.pred_pskip(mbx, mby, sid);
    g.fill_block(mbx, mby, 0, 0, 4, 4, [mx, my], 0);
    // A skip MB's nnz/mvd stay 0 (zeroed once per frame in reset()).
    g.seg[mb] = Seg::Skip;
    g.decoded[mb] = true;
}

fn read_ref(r: &mut BitReader, num_active: u32) -> i32 {
    if num_active <= 1 {
        0
    } else if num_active == 2 {
        (r.read_bit() ^ 1) as i32
    } else {
        r.read_ue() as i32
    }
}

fn decode_macroblock(
    g: &mut FrameGrids,
    r: &mut BitReader,
    mb: usize,
    sid: i32,
    sh: &SliceHeader,
    pps: &Pps,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    g.slice_id[mb] = sid;

    let mut mb_type = r.read_ue();
    let is_i_slice = matches!(sh.slice_type, SliceType::I | SliceType::Si);

    if !is_i_slice {
        // P slice
        if mb_type < 5 {
            return decode_p_inter(g, r, mb, sid, sh, pps, mb_type);
        }
        mb_type -= 5; // intra mb_type in a P slice
    }
    decode_intra(g, r, mb, sid, mb_type, pps)
        .map(|_| {
            g.set_mb_refi(mbx, mby, INTRA);
            g.seg[mb] = Seg::Intra;
            g.decoded[mb] = true;
        })
        .is_some()
}

fn decode_p_inter(
    g: &mut FrameGrids,
    r: &mut BitReader,
    mb: usize,
    sid: i32,
    sh: &SliceHeader,
    pps: &Pps,
    mb_type: u32,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let n0 = sh.num_ref_idx_l0_active;

    let seg = match mb_type {
        0 => {
            // P_L0_16x16
            let refr = read_ref(r, n0);
            g.fill_block(mbx, mby, 0, 0, 4, 4, [0, 0], refr); // ref before pred
            let (mut mx, mut my) = g.pred_motion(mbx, mby, 0, 0, 4, refr, sid);
            mx += r.read_se();
            my += r.read_se();
            g.fill_block(mbx, mby, 0, 0, 4, 4, [mx, my], refr);
            Seg::P16x16
        }
        1 => {
            // P_L0_L0_16x8 — two 16x8 parts (origin (0,0) and (0,2))
            let r0 = read_ref(r, n0);
            let r1 = read_ref(r, n0);
            g.fill_block(mbx, mby, 0, 0, 4, 2, [0, 0], r0);
            g.fill_block(mbx, mby, 0, 2, 4, 2, [0, 0], r1);
            let (mut mx, mut my) = pred_16x8(g, mbx, mby, 0, r0, sid);
            mx += r.read_se();
            my += r.read_se();
            g.fill_block(mbx, mby, 0, 0, 4, 2, [mx, my], r0);
            let (mut mx, mut my) = pred_16x8(g, mbx, mby, 1, r1, sid);
            mx += r.read_se();
            my += r.read_se();
            g.fill_block(mbx, mby, 0, 2, 4, 2, [mx, my], r1);
            Seg::P16x8
        }
        2 => {
            // P_L0_L0_8x16 — two 8x16 parts (origin (0,0) and (2,0))
            let r0 = read_ref(r, n0);
            let r1 = read_ref(r, n0);
            g.fill_block(mbx, mby, 0, 0, 2, 4, [0, 0], r0);
            g.fill_block(mbx, mby, 2, 0, 2, 4, [0, 0], r1);
            let (mut mx, mut my) = pred_8x16(g, mbx, mby, 0, r0, sid);
            mx += r.read_se();
            my += r.read_se();
            g.fill_block(mbx, mby, 0, 0, 2, 4, [mx, my], r0);
            let (mut mx, mut my) = pred_8x16(g, mbx, mby, 1, r1, sid);
            mx += r.read_se();
            my += r.read_se();
            g.fill_block(mbx, mby, 2, 0, 2, 4, [mx, my], r1);
            Seg::P8x16
        }
        3 | 4 => {
            // P_8x8 / P_8x8ref0
            let ref0 = mb_type == 4;
            if !decode_p_8x8(g, r, mb, sid, n0, ref0) {
                return false;
            }
            Seg::P8x8
        }
        _ => return false,
    };

    g.seg[mb] = seg;
    g.decoded[mb] = true;

    // coded_block_pattern (inter) then residual (sync).
    let cbp_code = r.read_ue();
    if cbp_code as usize >= GOLOMB_TO_INTER_CBP.len() {
        return false;
    }
    let cbp = GOLOMB_TO_INTER_CBP[cbp_code as usize] as u32;

    // transform_size_8x8_flag (only High profile; Baseline never sets it).
    if pps.transform_8x8_mode_flag && (cbp & 15) != 0 {
        let _t8 = r.read_bit();
    }

    if cbp != 0 {
        let _mb_qp_delta = r.read_se();
        if !residual_non_i16x16(g, r, mb, sid, cbp) {
            return false;
        }
    } else {
        zero_mb_nnz(g, mb);
    }
    true
}

fn decode_p_8x8(
    g: &mut FrameGrids,
    r: &mut BitReader,
    mb: usize,
    sid: i32,
    n0: u32,
    ref0: bool,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let mut sub = [0u32; 4];
    for s in sub.iter_mut() {
        *s = r.read_ue();
        if *s >= 4 {
            return false;
        }
    }
    // refs for all 4 quadrants first
    let mut refs = [0i32; 4];
    for i in 0..4 {
        refs[i] = if ref0 { 0 } else { read_ref(r, n0) };
        let (qx, qy) = ((i & 1) * 2, (i >> 1) * 2);
        g.fill_block(mbx, mby, qx, qy, 2, 2, [0, 0], refs[i]);
    }
    // mvds + prediction per sub-partition
    for i in 0..4 {
        let (qx, qy) = ((i & 1) * 2, (i >> 1) * 2);
        let refr = refs[i];
        // (pw,ph in 4x4 units, list of origins) per sub_mb_type
        let parts: &[(usize, usize, i32, usize)] = match sub[i] {
            0 => &[(0, 0, 2, 2)],                       // 8x8
            1 => &[(0, 0, 2, 1), (0, 1, 2, 1)],         // 8x4
            2 => &[(0, 0, 1, 2), (1, 0, 1, 2)],         // 4x8
            _ => &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)], // 4x4
        };
        for &(ox, oy, pw, ph) in parts {
            let bx = qx + ox;
            let by = qy + oy;
            let (mut mx, mut my) = g.pred_motion(mbx, mby, bx, by, pw, refr, sid);
            mx += r.read_se();
            my += r.read_se();
            g.fill_block(mbx, mby, bx, by, pw as usize, ph, [mx, my], refr);
        }
    }
    true
}

/// pred_16x8 (FFmpeg): part 0 favours top, part 1 favours left, else median.
fn pred_16x8(g: &FrameGrids, mbx: usize, mby: usize, part: usize, refr: i32, sid: i32) -> (i32, i32) {
    let cur_mb = mby * g.mb_w + mbx;
    if part == 0 {
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4) as i32;
        let top_ref = g.nref(x0, y0 - 1, cur_mb, xy_scan(0, 0), sid);
        if top_ref == refr {
            let b = g.nmv(x0, y0 - 1, cur_mb, xy_scan(0, 0), sid);
            return (b[0], b[1]);
        }
        g.pred_motion(mbx, mby, 0, 0, 4, refr, sid)
    } else {
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4 + 2) as i32;
        let left_ref = g.nref(x0 - 1, y0, cur_mb, xy_scan(0, 2), sid);
        if left_ref == refr {
            let a = g.nmv(x0 - 1, y0, cur_mb, xy_scan(0, 2), sid);
            return (a[0], a[1]);
        }
        g.pred_motion(mbx, mby, 0, 2, 4, refr, sid)
    }
}

/// pred_8x16 (FFmpeg): part 0 favours left, part 1 favours top-right, else median.
fn pred_8x16(g: &FrameGrids, mbx: usize, mby: usize, part: usize, refr: i32, sid: i32) -> (i32, i32) {
    let cur_mb = mby * g.mb_w + mbx;
    if part == 0 {
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4) as i32;
        let left_ref = g.nref(x0 - 1, y0, cur_mb, xy_scan(0, 0), sid);
        if left_ref == refr {
            let a = g.nmv(x0 - 1, y0, cur_mb, xy_scan(0, 0), sid);
            return (a[0], a[1]);
        }
        g.pred_motion(mbx, mby, 0, 0, 2, refr, sid)
    } else {
        // diagonal (top-right of the 8x16 right part, origin (2,0))
        let x0 = (mbx * 4 + 2) as i32;
        let y0 = (mby * 4) as i32;
        let tr_ref = g.nref(x0 + 2, y0 - 1, cur_mb, xy_scan(2, 0), sid);
        let (diag_ref, c) = if tr_ref != PART_NA {
            (tr_ref, g.nmv(x0 + 2, y0 - 1, cur_mb, xy_scan(2, 0), sid))
        } else {
            (
                g.nref(x0 - 1, y0 - 1, cur_mb, xy_scan(2, 0), sid),
                g.nmv(x0 - 1, y0 - 1, cur_mb, xy_scan(2, 0), sid),
            )
        };
        if diag_ref == refr {
            return (c[0], c[1]);
        }
        g.pred_motion(mbx, mby, 2, 0, 2, refr, sid)
    }
}

/// Intra macroblock — parsed for sync only (no MVs). Returns Some(()) on success.
fn decode_intra(g: &mut FrameGrids, r: &mut BitReader, mb: usize, sid: i32, i_mb_type: u32, pps: &Pps) -> Option<()> {
    if i_mb_type > 25 {
        return None;
    }
    if i_mb_type == 25 {
        // I_PCM: byte-align, then 384 bytes (4:2:0 8-bit) of samples.
        while !r.byte_aligned() {
            r.read_bit();
        }
        r.skip_bits(384 * 8);
        set_mb_nnz(g, mb, 16);
        return Some(());
    }
    if i_mb_type == 0 {
        // I_NxN: intra4x4 pred modes (di=1, or 4 with 8x8 transform), chroma
        // pred mode, then me(v) coded_block_pattern.
        let di = if pps.transform_8x8_mode_flag && r.read_bit() == 1 {
            4
        } else {
            1
        };
        let mut i = 0;
        while i < 16 {
            if r.read_bit() == 0 {
                let _rem = r.read_bits(3);
            }
            i += di;
        }
        let _chroma_pred = r.read_ue(); // intra_chroma_pred_mode (ue, <=3)
        let cbp_code = r.read_ue();
        if cbp_code as usize >= GOLOMB_TO_INTRA_CBP.len() {
            return None;
        }
        let cbp = GOLOMB_TO_INTRA_CBP[cbp_code as usize] as u32;
        if cbp != 0 {
            let _mb_qp_delta = r.read_se();
            if !residual_non_i16x16(g, r, mb, sid, cbp) {
                return None;
            }
        } else {
            zero_mb_nnz(g, mb);
        }
        Some(())
    } else {
        // I_16x16 (1..24): cbp encoded in mb_type.
        let i0 = i_mb_type - 1;
        let cbp_chroma = (i0 / 4) % 3;
        let cbp_luma = if i0 / 12 != 0 { 0x0f } else { 0 };
        let cbp = cbp_luma | (cbp_chroma << 4);
        let _chroma_pred = r.read_ue();
        let _mb_qp_delta = r.read_se(); // always present for I_16x16
        if !residual_i16x16(g, r, mb, sid, cbp) {
            return None;
        }
        Some(())
    }
}

// ── residual driving (sync only, updates nnz grids) ──────────────────────────

fn residual_i16x16(g: &mut FrameGrids, r: &mut BitReader, mb: usize, sid: i32, cbp: u32) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    // luma DC (max_coeff 16, nC of block 0); count not stored.
    let nc = g.nnz_pred_luma(mbx * 4, mby * 4, mb, sid);
    if residual_block(r, nc, 16).is_none() {
        return false;
    }
    if cbp & 15 != 0 {
        for idx in 0..16 {
            let (bx, by) = block_xy(idx);
            let gx = mbx * 4 + bx;
            let gy = mby * 4 + by;
            let nc = g.nnz_pred_luma(gx, gy, mb, sid);
            match residual_block(r, nc, 15) {
                Some(tc) => g.nnz_l[gy * g.bw + gx] = tc as u8,
                None => return false,
            }
        }
    } else {
        zero_mb_luma_nnz(g, mb);
    }
    residual_chroma(g, r, mb, sid, cbp)
}

fn residual_non_i16x16(g: &mut FrameGrids, r: &mut BitReader, mb: usize, sid: i32, cbp: u32) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    for i8x8 in 0..4 {
        if cbp & (1 << i8x8) != 0 {
            for i4x4 in 0..4 {
                let (bx, by) = block_xy(i4x4 + 4 * i8x8);
                let gx = mbx * 4 + bx;
                let gy = mby * 4 + by;
                let nc = g.nnz_pred_luma(gx, gy, mb, sid);
                match residual_block(r, nc, 16) {
                    Some(tc) => g.nnz_l[gy * g.bw + gx] = tc as u8,
                    None => return false,
                }
            }
        } else {
            for i4x4 in 0..4 {
                let (bx, by) = block_xy(i4x4 + 4 * i8x8);
                g.nnz_l[(mby * 4 + by) * g.bw + (mbx * 4 + bx)] = 0;
            }
        }
    }
    residual_chroma(g, r, mb, sid, cbp)
}

fn residual_chroma(g: &mut FrameGrids, r: &mut BitReader, mb: usize, sid: i32, cbp: u32) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    if cbp & 0x30 != 0 {
        // chroma DC (n_c = -1, max_coeff 4), count not stored
        for _c in 0..2 {
            if residual_block(r, -1, 4).is_none() {
                return false;
            }
        }
    }
    if cbp & 0x20 != 0 {
        for c in 0..2 {
            for i4x4 in 0..4 {
                let cx = i4x4 & 1;
                let cy = i4x4 >> 1;
                let gx = mbx * 2 + cx;
                let gy = mby * 2 + cy;
                let nc = g.nnz_pred_chroma(c, gx, gy, mb, sid);
                match residual_block(r, nc, 15) {
                    Some(tc) => g.nnz_c[c][gy * g.cw + gx] = tc as u8,
                    None => return false,
                }
            }
        }
    } else {
        for c in 0..2 {
            for cy in 0..2 {
                for cx in 0..2 {
                    g.nnz_c[c][(mby * 2 + cy) * g.cw + (mbx * 2 + cx)] = 0;
                }
            }
        }
    }
    true
}

fn zero_mb_luma_nnz(g: &mut FrameGrids, mb: usize) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    for by in 0..4 {
        for bx in 0..4 {
            g.nnz_l[(mby * 4 + by) * g.bw + (mbx * 4 + bx)] = 0;
        }
    }
}

fn zero_mb_nnz(g: &mut FrameGrids, mb: usize) {
    zero_mb_luma_nnz(g, mb);
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    for c in 0..2 {
        for cy in 0..2 {
            for cx in 0..2 {
                g.nnz_c[c][(mby * 2 + cy) * g.cw + (mbx * 2 + cx)] = 0;
            }
        }
    }
}

fn set_mb_nnz(g: &mut FrameGrids, mb: usize, v: u8) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    for by in 0..4 {
        for bx in 0..4 {
            g.nnz_l[(mby * 4 + by) * g.bw + (mbx * 4 + bx)] = v;
        }
    }
    for c in 0..2 {
        for cy in 0..2 {
            for cx in 0..2 {
                g.nnz_c[c][(mby * 2 + cy) * g.cw + (mbx * 2 + cx)] = v;
            }
        }
    }
}

// ── MV export, matching libavcodec/mpegutils.c add_mb() ───────────────────────

impl FrameGrids {
    /// Iterate every exported motion vector as `(sx, sy, w, h, mv)` — `(sx,sy)`
    /// is the partition centre (dst), `mv` the quarter-pel motion. Shared by the
    /// full and compact exporters so they can't diverge.
    fn for_each_mv(&self, mut f: impl FnMut(i32, i32, i32, i32, [i32; 2])) {
        for mby in 0..self.mb_h {
            for mbx in 0..self.mb_w {
                let mb = mby * self.mb_w + mbx;
                let mv_at = |bx: usize, by: usize| -> [i32; 2] {
                    self.mv[(mby * 4 + by) * self.bw + (mbx * 4 + bx)]
                };
                let x = (mbx * 16) as i32;
                let y = (mby * 16) as i32;
                match self.seg[mb] {
                    Seg::Skip | Seg::P16x16 => f(x + 8, y + 8, 16, 16, mv_at(0, 0)),
                    Seg::P16x8 => {
                        f(x + 8, y + 4, 16, 8, mv_at(0, 0));
                        f(x + 8, y + 12, 16, 8, mv_at(0, 2));
                    }
                    Seg::P8x16 => {
                        f(x + 4, y + 8, 8, 16, mv_at(0, 0));
                        f(x + 12, y + 8, 8, 16, mv_at(2, 0));
                    }
                    Seg::P8x8 => {
                        for i in 0..4 {
                            let sx = x + 4 + 8 * (i & 1) as i32;
                            let sy = y + 4 + 8 * (i >> 1) as i32;
                            f(sx, sy, 8, 8, mv_at((i & 1) * 2, (i >> 1) * 2));
                        }
                    }
                    Seg::Intra | Seg::None => {}
                }
            }
        }
    }

    /// Full `AVMotionVector`-layout export (12 columns).
    pub fn export_mvs(&self, frame: i32, out: &mut Vec<MotionVector>) {
        self.for_each_mv(|sx, sy, w, h, mv| {
            out.push(MotionVector {
                frame,
                source: -1,
                w,
                h,
                src_x: (sx + mv[0] / 4) as f64,
                src_y: (sy + mv[1] / 4) as f64,
                dst_x: sx as f64,
                dst_y: sy as f64,
                flags: 0,
                motion_x: mv[0] as f64,
                motion_y: mv[1] as f64,
                motion_scale: 4.0,
            });
        });
    }

    /// Compact export matching the custom FFmpeg `AVMotionVectorCompact`
    /// (libavcodec add_mb_compact): 6 integer columns, ~5x smaller and faster to
    /// write than the full format.
    pub fn export_mvs_compact(&self, frame: i32, out: &mut Vec<MvCompact>) {
        self.for_each_mv(|sx, sy, _w, _h, mv| {
            out.push(MvCompact {
                frame,
                source: -1,
                src_x: (sx + mv[0] / 4) as i16,
                src_y: (sy + mv[1] / 4) as i16,
                dst_x: sx as i16,
                dst_y: sy as i16,
            });
        });
    }
}

const GOLOMB_TO_INTER_CBP: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];
const GOLOMB_TO_INTRA_CBP: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

// ═════════════════════════════════════════════════════════════════════════════
// CABAC slice decoding (thesis Ch4 §4.4.2). Reuses the FrameGrids/prediction/
// export above; only the entropy layer differs. Ported from FFmpeg
// h264_cabac.c for 4:2:0, frame-only (non-MBAFF), P/I slices.
// ═════════════════════════════════════════════════════════════════════════════

use super::cabac::{init_states, Cabac};
use super::custom_cabac_tables::{LAST_COEFF_8X8, SIG_OFF_8X8};

// Residual context base offsets (non-MBAFF row), categories 0..4:
// 0=LumaDC16, 1=LumaAC16, 2=Luma4x4, 3=ChromaDC, 4=ChromaAC.
const SIG_OFF: [usize; 6] = [105, 120, 134, 149, 152, 402];
const LAST_OFF: [usize; 6] = [166, 181, 195, 210, 213, 417];
const ABS_OFF: [usize; 6] = [227, 237, 247, 257, 266, 426];
const CBF_BASE: [usize; 5] = [85, 89, 93, 97, 101];
const ABS_L1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
const ABS_GT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
const TRANS0: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
const TRANS1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];

/// CABAC engine + the 1024 context states for one slice.
struct Cabd<'a> {
    c: Cabac<'a>,
    st: Box<[u8; 1024]>,
    last_qscale_nonzero: bool,
}

impl<'a> Cabd<'a> {
    #[inline]
    fn get(&mut self, ctx: usize) -> u32 {
        self.c.get(&mut self.st[ctx])
    }
}

impl FrameGrids {
    /// (left_mb, top_mb) that are available for context (in-bounds, decoded,
    /// same slice). `None` = unavailable.
    fn ctx_neighbors(&self, mb: usize, sid: i32) -> (Option<usize>, Option<usize>) {
        let mbx = mb % self.mb_w;
        let mby = mb / self.mb_w;
        let ok = |m: usize| self.decoded[m] && self.slice_id[m] == sid;
        let left = if mbx > 0 && ok(mb - 1) { Some(mb - 1) } else { None };
        let top = if mby > 0 && ok(mb - self.mb_w) {
            Some(mb - self.mb_w)
        } else {
            None
        };
        (left, top)
    }

    /// Effective cbp of a neighbour MB for context, applying FFmpeg's
    /// unavailable defaults (intra → 0x7CF, inter → 0x00F).
    /// neighbour_transform_size ctx for transform_size_8x8_flag: count of
    /// available neighbours coded with the 8x8 transform (cbp_mb bit 0x1000).
    fn neighbor_transform_size(&self, mb: usize, sid: i32) -> usize {
        let (l, t) = self.ctx_neighbors(mb, sid);
        let bit = |n: Option<usize>| n.map_or(false, |m| self.cbp_mb[m] & 0x1000 != 0);
        bit(l) as usize + bit(t) as usize
    }

    fn ctx_cbp(&self, n: Option<usize>, cur_intra: bool) -> u32 {
        match n {
            Some(m) => self.cbp_mb[m] as u32,
            None => {
                if cur_intra {
                    0x7CF
                } else {
                    0x00F
                }
            }
        }
    }
}

/// Decode a CABAC P/I slice into `g`. `rbsp` is the slice NAL RBSP; `byte_start`
/// is the byte offset of slice_data() (after cabac_alignment_one_bit).
pub fn decode_slice_cabac(
    g: &mut FrameGrids,
    rbsp: &[u8],
    byte_start: usize,
    sh: &SliceHeader,
    _sps: &Sps,
    pps: &Pps,
    sid: i32,
) -> SliceResult {
    let mut cd = Cabd {
        c: Cabac::new(rbsp, byte_start),
        st: Box::new([0u8; 1024]),
        last_qscale_nonzero: false,
    };
    let is_i = matches!(sh.slice_type, SliceType::I | SliceType::Si);
    init_states(&mut cd.st, is_i, sh.cabac_init_idc as usize, sh.slice_qp);

    let pmbs = g.mb_w * g.mb_h;
    let mut cur = sh.first_mb_in_slice as usize;
    let mut count = 0usize;
    let mut ok = true;

    loop {
        if cur >= pmbs {
            break;
        }
        let mut skipped = false;
        if !is_i {
            // mb_skip_flag
            let (l, t) = g.ctx_neighbors(cur, sid);
            let mut ctx = 0;
            if let Some(m) = l {
                if g.seg[m] != Seg::Skip {
                    ctx += 1;
                }
            }
            if let Some(m) = t {
                if g.seg[m] != Seg::Skip {
                    ctx += 1;
                }
            }
            if cd.get(11 + ctx) == 1 {
                decode_p_skip(g, cur, sid);
                // A skip MB carries no mb_qp_delta; FFmpeg resets last_qscale_diff
                // to 0 so the next MB's dqp context (ctx 60 vs 61) is correct.
                cd.last_qscale_nonzero = false;
                skipped = true;
            }
        }
        if !skipped && !decode_mb_cabac(g, &mut cd, cur, sid, sh, pps) {
            ok = false;
            break;
        }
        cur += 1;
        count += 1;
        if cur >= pmbs {
            break;
        }
        // end_of_slice_flag
        if cd.c.terminate() {
            break;
        }
    }

    SliceResult {
        mbs_decoded: count,
        ok,
    }
}

fn decode_mb_cabac(
    g: &mut FrameGrids,
    cd: &mut Cabd,
    mb: usize,
    sid: i32,
    sh: &SliceHeader,
    pps: &Pps,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    g.slice_id[mb] = sid;
    let is_i = matches!(sh.slice_type, SliceType::I | SliceType::Si);

    // ── mb_type ──
    let mut intra_i_mb_type: Option<u32> = None;
    let mut p_inter: Option<u32> = None;
    if is_i {
        intra_i_mb_type = Some(decode_intra_mb_type_cabac(g, cd, mb, sid, 3, true));
    } else if cd.get(14) == 0 {
        // P inter
        let t = if cd.get(15) == 0 {
            3 * cd.get(16)
        } else {
            2 - cd.get(17)
        };
        p_inter = Some(t);
    } else {
        intra_i_mb_type = Some(decode_intra_mb_type_cabac(g, cd, mb, sid, 17, false));
    }

    if let Some(t) = p_inter {
        return decode_p_inter_cabac(g, cd, mb, sid, sh, pps, t);
    }

    // ── intra ──
    let i_mb_type = intra_i_mb_type.unwrap();
    if i_mb_type == 25 {
        // I_PCM: re-init the CABAC engine after the raw bytes (384 bytes 4:2:0).
        let p = cabac_pcm_byte_pos(cd);
        let new_start = p + 384;
        cd.c = Cabac::new(cd.c_bytes(), new_start);
        cd.last_qscale_nonzero = false; // I_PCM resets last_qscale_diff (FFmpeg)
        set_mb_nnz(g, mb, 16);
        g.cbp_mb[mb] = 0x7EF | 0x800; // all coded + I16x16/PCM marker
        g.set_mb_refi(mbx, mby, INTRA);
        g.seg[mb] = Seg::Intra;
        g.decoded[mb] = true;
        return true;
    }
    // seg must be Intra *before* residual decode: the CABAC coded_block_flag
    // context derives its unavailable-neighbour default from the current MB's
    // intra-ness (is_intra), which is read off g.seg[mb] inside the residual
    // helpers (FFmpeg condTermFlag=1 for unavailable neighbours of an intra MB).
    g.seg[mb] = Seg::Intra;
    decode_intra_residual_cabac(g, cd, mb, sid, i_mb_type, pps);
    g.set_mb_refi(mbx, mby, INTRA);
    g.decoded[mb] = true;
    true
}

impl<'a> Cabd<'a> {
    fn c_bytes(&self) -> &'a [u8] {
        self.c.bytes_ref()
    }
}

fn cabac_pcm_byte_pos(cd: &Cabd) -> usize {
    cd.c.pcm_byte_pos()
}

/// decode_cabac_intra_mb_type (FFmpeg). ctx_base 3 (I-slice) or 17 (P intra).
fn decode_intra_mb_type_cabac(
    g: &FrameGrids,
    cd: &mut Cabd,
    mb: usize,
    sid: i32,
    ctx_base: usize,
    intra_slice: bool,
) -> u32 {
    let mut base = ctx_base;
    if intra_slice {
        let (l, t) = g.ctx_neighbors(mb, sid);
        let mut ctx = 0;
        if let Some(m) = l {
            if g.seg[m] == Seg::Intra && g.cbp_mb[m] & 0x100 != 0 {
                // approximate I16x16/PCM via DC-coded flag; refined below
            }
            if is_intra16x16_or_pcm(g, m) {
                ctx += 1;
            }
        }
        if let Some(m) = t {
            if is_intra16x16_or_pcm(g, m) {
                ctx += 1;
            }
        }
        if cd.get(base + ctx) == 0 {
            return 0; // I_NxN
        }
        base += 2;
    } else if cd.get(base) == 0 {
        return 0; // I_NxN
    }
    if cd.c.terminate() {
        return 25; // I_PCM
    }
    let isl = intra_slice as usize;
    let mut mb_type = 1u32;
    mb_type += 12 * cd.get(base + 1);
    if cd.get(base + 2) == 1 {
        mb_type += 4 + 4 * cd.get(base + 2 + isl);
    }
    mb_type += 2 * cd.get(base + 3 + isl);
    mb_type += cd.get(base + 3 + 2 * isl);
    mb_type
}

fn is_intra16x16_or_pcm(g: &FrameGrids, m: usize) -> bool {
    // I16x16 and PCM MBs have luma-DC or all-coeff flags; we track this via a
    // per-MB marker in the high bits of cbp_mb (0x800 = I16x16/PCM).
    g.seg[m] == Seg::Intra && (g.cbp_mb[m] & 0x800 != 0)
}

impl FrameGrids {
    fn amvd(&self, mb: usize, sid: i32, bx: usize, by: usize) -> (i32, i32) {
        let scan = xy_scan(bx, by);
        let x0 = (mb % self.mb_w * 4 + bx) as i32;
        let y0 = (mb / self.mb_w * 4 + by) as i32;
        let l = if self.avail(x0 - 1, y0, mb, scan, sid) {
            self.mvd_l[self.lidx((x0 - 1) as usize, y0 as usize)]
        } else {
            [0, 0]
        };
        let t = if self.avail(x0, y0 - 1, mb, scan, sid) {
            self.mvd_l[self.lidx(x0 as usize, (y0 - 1) as usize)]
        } else {
            [0, 0]
        };
        (l[0] + t[0], l[1] + t[1])
    }

    fn fill_mvd(&mut self, mbx: usize, mby: usize, bx: usize, by: usize, w: usize, h: usize, v: [i32; 2]) {
        for dy in 0..h {
            for dx in 0..w {
                let idx = (mby * 4 + by + dy) * self.bw + (mbx * 4 + bx + dx);
                self.mvd_l[idx] = v;
            }
        }
    }

    fn ref_neighbor(&self, mb: usize, sid: i32, bx: usize, by: usize) -> usize {
        let scan = xy_scan(bx, by);
        let x0 = (mb % self.mb_w * 4 + bx) as i32;
        let y0 = (mb / self.mb_w * 4 + by) as i32;
        let refa = self.nref(x0 - 1, y0, mb, scan, sid);
        let refb = self.nref(x0, y0 - 1, mb, scan, sid);
        (refa > 0) as usize + 2 * (refb > 0) as usize
    }
}

fn decode_mvd_cabac(cd: &mut Cabd, ctxbase: usize, amvd: i32) -> (i32, i32) {
    let c0 = ctxbase + (amvd > 2) as usize + (amvd > 32) as usize;
    if cd.get(c0) == 0 {
        return (0, 0);
    }
    let mut mvd = 1i32;
    let mut ctx = ctxbase + 3;
    while mvd < 9 && cd.get(ctx) == 1 {
        if mvd < 4 {
            ctx += 1;
        }
        mvd += 1;
    }
    if mvd >= 9 {
        let mut k = 3;
        while cd.c.bypass() == 1 {
            mvd += 1 << k;
            k += 1;
            if k > 24 {
                break;
            }
        }
        while k > 0 {
            k -= 1;
            mvd += (cd.c.bypass() as i32) << k;
        }
    }
    let clamped = mvd.min(70);
    let signed = cd.c.bypass_sign(-mvd);
    (signed, clamped)
}

fn decode_ref_cabac(cd: &mut Cabd, mut ctx: usize) -> i32 {
    let mut r = 0;
    while cd.get(54 + ctx) == 1 {
        r += 1;
        ctx = (ctx >> 2) + 4;
        if r >= 32 {
            break;
        }
    }
    r
}

/// decode ref_idx (0 when only one reference is active).
fn read_ref_cabac(g: &FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, bx: usize, by: usize, n0: u32) -> i32 {
    if n0 <= 1 {
        0
    } else {
        decode_ref_cabac(cd, g.ref_neighbor(mb, sid, bx, by))
    }
}

fn decode_dqp(cd: &mut Cabd) {
    let ctx0 = 60 + cd.last_qscale_nonzero as usize;
    if cd.get(ctx0) == 0 {
        cd.last_qscale_nonzero = false;
        return;
    }
    let mut ctx = 2;
    let mut val = 1;
    while cd.get(60 + ctx) == 1 {
        ctx = 3;
        val += 1;
        if val > 2 * 51 + 10 {
            break;
        }
    }
    cd.last_qscale_nonzero = true;
}

fn decode_p_inter_cabac(
    g: &mut FrameGrids,
    cd: &mut Cabd,
    mb: usize,
    sid: i32,
    sh: &SliceHeader,
    pps: &Pps,
    mb_type: u32,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let n0 = sh.num_ref_idx_l0_active;

    let do_mvd = |g: &mut FrameGrids, cd: &mut Cabd, bx, by, pw, ph, refr, mxp: i32, myp: i32| {
        let (ax, ay) = g.amvd(mb, sid, bx, by);
        let (dx, cx) = decode_mvd_cabac(cd, 40, ax);
        let (dy, cy) = decode_mvd_cabac(cd, 47, ay);
        g.fill_block(mbx, mby, bx, by, pw, ph, [mxp + dx, myp + dy], refr);
        g.fill_mvd(mbx, mby, bx, by, pw, ph, [cx, cy]);
    };

    let mut all_8x8 = true;
    let seg = match mb_type {
        0 => {
            let r = read_ref_cabac(g, cd, mb, sid, 0, 0, n0);
            g.fill_block(mbx, mby, 0, 0, 4, 4, [0, 0], r);
            let (mx, my) = g.pred_motion(mbx, mby, 0, 0, 4, r, sid);
            do_mvd(g, cd, 0, 0, 4, 4, r, mx, my);
            Seg::P16x16
        }
        1 => {
            let r0 = read_ref_cabac(g, cd, mb, sid, 0, 0, n0);
            g.fill_block(mbx, mby, 0, 0, 4, 2, [0, 0], r0);
            let r1 = read_ref_cabac(g, cd, mb, sid, 0, 2, n0);
            g.fill_block(mbx, mby, 0, 2, 4, 2, [0, 0], r1);
            let (mx0, my0) = pred_16x8(g, mbx, mby, 0, r0, sid);
            do_mvd(g, cd, 0, 0, 4, 2, r0, mx0, my0);
            let (mx1, my1) = pred_16x8(g, mbx, mby, 1, r1, sid);
            do_mvd(g, cd, 0, 2, 4, 2, r1, mx1, my1);
            Seg::P16x8
        }
        2 => {
            let r0 = read_ref_cabac(g, cd, mb, sid, 0, 0, n0);
            g.fill_block(mbx, mby, 0, 0, 2, 4, [0, 0], r0);
            let r1 = read_ref_cabac(g, cd, mb, sid, 2, 0, n0);
            g.fill_block(mbx, mby, 2, 0, 2, 4, [0, 0], r1);
            let (mx0, my0) = pred_8x16(g, mbx, mby, 0, r0, sid);
            do_mvd(g, cd, 0, 0, 2, 4, r0, mx0, my0);
            let (mx1, my1) = pred_8x16(g, mbx, mby, 1, r1, sid);
            do_mvd(g, cd, 2, 0, 2, 4, r1, mx1, my1);
            Seg::P8x16
        }
        _ => {
            all_8x8 = decode_p_8x8_cabac(g, cd, mb, sid, n0);
            Seg::P8x8
        }
    };

    g.seg[mb] = seg;
    g.decoded[mb] = true;

    let cbp = decode_cbp_cabac(g, cd, mb, sid, false);
    // transform_size_8x8_flag (High profile): allowed for 16x16/16x8/8x16, or
    // P_8x8 only when every sub-partition is 8x8.
    let dct8_allowed = pps.transform_8x8_mode_flag && (mb_type < 3 || all_8x8);
    let dct8 = dct8_allowed && cbp & 15 != 0 && {
        let nts = g.neighbor_transform_size(mb, sid);
        cd.get(399 + nts) == 1
    };
    g.cbp_mb[mb] = (cbp & 0xFF) as u16 | if dct8 { 0x1000 } else { 0 };
    if cbp != 0 {
        decode_dqp(cd);
        residual_luma_cabac(g, cd, mb, sid, false, cbp);
        residual_chroma_cabac(g, cd, mb, sid, cbp);
    } else {
        cd.last_qscale_nonzero = false;
        zero_mb_nnz(g, mb);
    }
    true
}

/// Returns true if every sub-partition is 8x8 (relevant for dct8x8_allowed).
fn decode_p_8x8_cabac(g: &mut FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, n0: u32) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let mut sub = [0u32; 4];
    for s in sub.iter_mut() {
        // decode_cabac_p_mb_sub_type
        *s = if cd.get(21) == 1 {
            0
        } else if cd.get(22) == 0 {
            1
        } else if cd.get(23) == 1 {
            2
        } else {
            3
        };
    }
    let mut refs = [0i32; 4];
    for i in 0..4 {
        let (qx, qy) = ((i & 1) * 2, (i >> 1) * 2);
        refs[i] = read_ref_cabac(g, cd, mb, sid, qx, qy, n0);
        g.fill_block(mbx, mby, qx, qy, 2, 2, [0, 0], refs[i]);
    }
    for i in 0..4 {
        let (qx, qy) = ((i & 1) * 2, (i >> 1) * 2);
        let refr = refs[i];
        let parts: &[(usize, usize, i32, usize)] = match sub[i] {
            0 => &[(0, 0, 2, 2)],
            1 => &[(0, 0, 2, 1), (0, 1, 2, 1)],
            2 => &[(0, 0, 1, 2), (1, 0, 1, 2)],
            _ => &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
        };
        for &(ox, oy, pw, ph) in parts {
            let bx = qx + ox;
            let by = qy + oy;
            let (mx, my) = g.pred_motion(mbx, mby, bx, by, pw, refr, sid);
            let (ax, ay) = g.amvd(mb, sid, bx, by);
            let (dx, cx) = decode_mvd_cabac(cd, 40, ax);
            let (dy, cy) = decode_mvd_cabac(cd, 47, ay);
            g.fill_block(mbx, mby, bx, by, pw as usize, ph, [mx + dx, my + dy], refr);
            g.fill_mvd(mbx, mby, bx, by, pw as usize, ph, [cx, cy]);
        }
    }
    sub.iter().all(|&s| s == 0)
}

fn decode_intra_residual_cabac(
    g: &mut FrameGrids,
    cd: &mut Cabd,
    mb: usize,
    sid: i32,
    i_mb_type: u32,
    pps: &Pps,
) {
    if i_mb_type == 0 {
        // I_NxN: transform_size_8x8_flag (High profile), then intra4x4 pred modes
        // (di=4 when 8x8-transform), chroma pred, cbp.
        let nts = g.neighbor_transform_size(mb, sid);
        let dct8 = pps.transform_8x8_mode_flag && cd.get(399 + nts) == 1;
        let di = if dct8 { 4 } else { 1 };
        let mut i = 0;
        while i < 16 {
            if cd.get(68) == 0 {
                cd.get(69);
                cd.get(69);
                cd.get(69);
            }
            i += di;
        }
        decode_chroma_pred_cabac(g, cd, mb, sid);
        let cbp = decode_cbp_cabac(g, cd, mb, sid, true);
        g.cbp_mb[mb] = (cbp & 0xFF) as u16 | if dct8 { 0x1000 } else { 0 };
        if cbp != 0 {
            decode_dqp(cd);
            residual_luma_cabac(g, cd, mb, sid, false, cbp);
            residual_chroma_cabac(g, cd, mb, sid, cbp);
        } else {
            cd.last_qscale_nonzero = false;
            zero_mb_nnz(g, mb);
        }
    } else {
        // I_16x16
        let i0 = i_mb_type - 1;
        let cbp_chroma = (i0 / 4) % 3;
        let cbp_luma = if i0 / 12 != 0 { 0x0f } else { 0 };
        let cbp = cbp_luma | (cbp_chroma << 4);
        g.cbp_mb[mb] = (cbp as u16) | 0x800; // mark I16x16 for neighbour ctx
        decode_chroma_pred_cabac(g, cd, mb, sid);
        decode_dqp(cd);
        residual_luma_cabac(g, cd, mb, sid, true, cbp);
        residual_chroma_cabac(g, cd, mb, sid, cbp);
    }
}

fn decode_chroma_pred_cabac(g: &mut FrameGrids, cd: &mut Cabd, mb: usize, sid: i32) {
    let (l, t) = g.ctx_neighbors(mb, sid);
    let mut ctx = 0;
    if let Some(m) = l {
        if g.chroma_pred[m] != 0 {
            ctx += 1;
        }
    }
    if let Some(m) = t {
        if g.chroma_pred[m] != 0 {
            ctx += 1; // ctxIdxInc = condTermFlagA + condTermFlagB, range 0..2
        }
    }
    let v = if cd.get(64 + ctx) == 0 {
        0
    } else if cd.get(64 + 3) == 0 {
        1
    } else if cd.get(64 + 3) == 0 {
        2
    } else {
        3
    };
    g.chroma_pred[mb] = v;
}

fn decode_cbp_cabac(g: &FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, is_intra: bool) -> u32 {
    let (l, t) = g.ctx_neighbors(mb, sid);
    let cbp_a = g.ctx_cbp(l, is_intra);
    let cbp_b = g.ctx_cbp(t, is_intra);
    let mut cbp = 0u32;
    let mut ctx = (cbp_a & 0x02 == 0) as usize + 2 * (cbp_b & 0x04 == 0) as usize;
    cbp |= cd.get(73 + ctx);
    ctx = (cbp & 0x01 == 0) as usize + 2 * (cbp_b & 0x08 == 0) as usize;
    cbp |= cd.get(73 + ctx) << 1;
    ctx = (cbp_a & 0x08 == 0) as usize + 2 * (cbp & 0x01 == 0) as usize;
    cbp |= cd.get(73 + ctx) << 2;
    ctx = (cbp & 0x04 == 0) as usize + 2 * (cbp & 0x02 == 0) as usize;
    cbp |= cd.get(73 + ctx) << 3;
    // chroma
    let ca = (cbp_a >> 4) & 3;
    let cb = (cbp_b >> 4) & 3;
    let mut ctx = 0;
    if ca > 0 {
        ctx += 1;
    }
    if cb > 0 {
        ctx += 2;
    }
    if cd.get(77 + ctx) == 1 {
        let mut ctx = 4;
        if ca == 2 {
            ctx += 1;
        }
        if cb == 2 {
            ctx += 2;
        }
        cbp |= (1 + cd.get(77 + ctx)) << 4;
    }
    cbp
}

// ── CABAC residual ───────────────────────────────────────────────────────────

impl FrameGrids {
    fn cbf_ctx_luma(&self, mb: usize, sid: i32, gx: usize, gy: usize, cat: usize, is_intra: bool) -> usize {
        let scan = xy_scan(gx % 4, gy % 4);
        let nza = if self.avail(gx as i32 - 1, gy as i32, mb, scan, sid) {
            self.nnz_l[self.lidx(gx - 1, gy)] > 0
        } else {
            is_intra
        };
        let nzb = if self.avail(gx as i32, gy as i32 - 1, mb, scan, sid) {
            self.nnz_l[self.lidx(gx, gy - 1)] > 0
        } else {
            is_intra
        };
        CBF_BASE[cat] + nza as usize + 2 * nzb as usize
    }

    fn cbf_ctx_chroma(&self, mb: usize, sid: i32, c: usize, cx: usize, cy: usize, is_intra: bool) -> usize {
        let ch = self.nnz_c[c].len() / self.cw;
        let av = |x: i32, y: i32| -> Option<usize> {
            if x < 0 || y < 0 || x >= self.cw as i32 || y >= ch as i32 {
                return None;
            }
            let m = (x as usize / 2) + (y as usize / 2) * self.mb_w;
            if m == mb || (m < mb && self.decoded[m] && self.slice_id[m] == sid) {
                Some(y as usize * self.cw + x as usize)
            } else {
                None
            }
        };
        let nza = av(cx as i32 - 1, cy as i32).map_or(is_intra, |i| self.nnz_c[c][i] > 0);
        let nzb = av(cx as i32, cy as i32 - 1).map_or(is_intra, |i| self.nnz_c[c][i] > 0);
        CBF_BASE[4] + nza as usize + 2 * nzb as usize
    }

    fn cbf_ctx_dc(&self, mb: usize, sid: i32, cat: usize, comp: usize, is_intra: bool) -> usize {
        let (l, t) = self.ctx_neighbors(mb, sid);
        let mask = if cat == 0 { 0x100u16 } else { 0x40u16 << comp };
        let bit = |n: Option<usize>| -> bool {
            match n {
                Some(m) => self.cbp_mb[m] & mask != 0,
                None => is_intra,
            }
        };
        CBF_BASE[cat] + bit(l) as usize + 2 * bit(t) as usize
    }
}

/// Decode significance map + coeff_abs levels for sync; returns coeff_count.
/// `is_8x8` selects the position-dependent significance offsets (cat 5).
fn residual_bins(cd: &mut Cabd, cat: usize, max_coeff: usize, is_8x8: bool) -> u8 {
    let sig = SIG_OFF[cat];
    let last_b = LAST_OFF[cat];
    let abs_b = ABS_OFF[cat];
    let coefs = max_coeff - 1;
    let mut count = 0usize;
    let mut last = 0usize;
    loop {
        if last >= coefs {
            break;
        }
        let (so, lo) = if is_8x8 {
            (SIG_OFF_8X8[last] as usize, LAST_COEFF_8X8[last] as usize)
        } else {
            (last, last)
        };
        if cd.get(sig + so) == 1 {
            count += 1;
            if cd.get(last_b + lo) == 1 {
                break;
            }
        }
        last += 1;
    }
    if last == max_coeff - 1 {
        count += 1;
    }
    // coeff_abs levels (reverse order), bins consumed only.
    let mut node = 0usize;
    for _ in 0..count {
        if cd.get(abs_b + ABS_L1_CTX[node]) == 0 {
            node = TRANS0[node];
            cd.c.bypass_sign(-1);
        } else {
            let cg = abs_b + ABS_GT1_CTX[node];
            node = TRANS1[node];
            let mut coeff_abs = 2;
            while coeff_abs < 15 && cd.get(cg) == 1 {
                coeff_abs += 1;
            }
            if coeff_abs >= 15 {
                let mut j = 0;
                while cd.c.bypass() == 1 && j < 23 {
                    j += 1;
                }
                let mut ca = 1i32;
                while j > 0 {
                    j -= 1;
                    ca = ca + ca + cd.c.bypass() as i32;
                }
            }
            cd.c.bypass_sign(-1);
        }
    }
    count as u8
}

fn residual_luma_cabac(g: &mut FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, is_i16x16: bool, cbp: u32) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let is_intra = g.seg[mb] == Seg::Intra;
    if is_i16x16 {
        // luma DC (cat0)
        let cbf = g.cbf_ctx_dc(mb, sid, 0, 0, is_intra);
        if cd.get(cbf) == 1 {
            g.cbp_mb[mb] |= 0x100;
            residual_bins(cd, 0, 16, false);
        }
        if cbp & 15 != 0 {
            for idx in 0..16 {
                let (bx, by) = block_xy(idx);
                let gx = mbx * 4 + bx;
                let gy = mby * 4 + by;
                let cbf = g.cbf_ctx_luma(mb, sid, gx, gy, 1, is_intra);
                if cd.get(cbf) == 1 {
                    g.nnz_l[gy * g.bw + gx] = residual_bins(cd, 1, 15, false);
                } else {
                    g.nnz_l[gy * g.bw + gx] = 0;
                }
            }
        } else {
            zero_mb_luma_nnz(g, mb);
        }
    } else if g.cbp_mb[mb] & 0x1000 != 0 {
        // 8x8 transform: one cat-5 block per coded 8x8 (no cbf bin in non-444).
        for i8x8 in 0..4 {
            let bx = (i8x8 & 1) * 2;
            let by = (i8x8 >> 1) * 2;
            let count = if cbp & (1 << i8x8) != 0 {
                residual_bins(cd, 5, 64, true)
            } else {
                0
            };
            for dy in 0..2 {
                for dx in 0..2 {
                    g.nnz_l[(mby * 4 + by + dy) * g.bw + (mbx * 4 + bx + dx)] = count;
                }
            }
        }
    } else {
        for i8x8 in 0..4 {
            if cbp & (1 << i8x8) != 0 {
                for i4x4 in 0..4 {
                    let (bx, by) = block_xy(i4x4 + 4 * i8x8);
                    let gx = mbx * 4 + bx;
                    let gy = mby * 4 + by;
                    let cbf = g.cbf_ctx_luma(mb, sid, gx, gy, 2, is_intra);
                    if cd.get(cbf) == 1 {
                        g.nnz_l[gy * g.bw + gx] = residual_bins(cd, 2, 16, false);
                    } else {
                        g.nnz_l[gy * g.bw + gx] = 0;
                    }
                }
            } else {
                for i4x4 in 0..4 {
                    let (bx, by) = block_xy(i4x4 + 4 * i8x8);
                    g.nnz_l[(mby * 4 + by) * g.bw + (mbx * 4 + bx)] = 0;
                }
            }
        }
    }
}

fn residual_chroma_cabac(g: &mut FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, cbp: u32) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let is_intra = g.seg[mb] == Seg::Intra;
    if cbp & 0x30 != 0 {
        for c in 0..2 {
            let cbf = g.cbf_ctx_dc(mb, sid, 3, c, is_intra);
            if cd.get(cbf) == 1 {
                g.cbp_mb[mb] |= 0x40 << c;
                residual_bins(cd, 3, 4, false);
            }
        }
    }
    if cbp & 0x20 != 0 {
        for c in 0..2 {
            for i4x4 in 0..4 {
                let cx = mbx * 2 + (i4x4 & 1);
                let cy = mby * 2 + (i4x4 >> 1);
                let cbf = g.cbf_ctx_chroma(mb, sid, c, cx, cy, is_intra);
                if cd.get(cbf) == 1 {
                    g.nnz_c[c][cy * g.cw + cx] = residual_bins(cd, 4, 15, false);
                } else {
                    g.nnz_c[c][cy * g.cw + cx] = 0;
                }
            }
        }
    } else {
        for c in 0..2 {
            for cy in 0..2 {
                for cx in 0..2 {
                    g.nnz_c[c][(mby * 2 + cy) * g.cw + (mbx * 2 + cx)] = 0;
                }
            }
        }
    }
}
