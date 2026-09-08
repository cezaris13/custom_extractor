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
use crate::ffmpeg_common::MvFilter;
use mv_types::motion_vector::{MotionVector, MvCompact};

const PART_NA: i32 = -2; // PART_NOT_AVAILABLE
const INTRA: i32 = -1; // LIST_NOT_USED (intra / not list-0)

/// Snapshot of RefPicList1[0] (the collocated picture used by B-slice spatial
/// direct mode). `mv1`/`refi1` are `Some` only when the collocated picture is
/// itself a *reference* B picture (hierarchical/pyramid B, `nal_ref_idc != 0`)
/// — those need the ITU-T §8.4.1.2.2 list-1 colZeroFlag fallback for blocks
/// that didn't use list 0. `None` for I/P collocated pictures, which never
/// have a list 1.
pub struct ColPic<'a> {
    pub bw: usize,
    pub mv0: &'a [[i32; 2]],
    pub refi0: &'a [i32],
    pub mv1: Option<&'a [[i32; 2]]>,
    pub refi1: Option<&'a [i32]>,
    /// Per-macroblock collocated shape category (ITU-T §8.4.1.2.2 / FFmpeg
    /// h264_direct.c's `pred_spatial_direct_motion`, `single_col` label):
    /// `ColShape::Big16x16` (16x16-or-intra), `Wide16x8`, `Tall8x16`, or
    /// `Small` (8x8-or-finer). Indexed by macroblock number, same raster
    /// order as the current picture (frame-only decode: the collocated
    /// macroblock is always at the same `mb` index). Used only as a
    /// tiebreaker in `decode_b_direct_whole_mb` when the 4 independently
    /// computed quadrant values aren't already uniform (which is FFmpeg's
    /// primary promote-to-16x16 signal — see that function's doc comment).
    pub col_shape: &'a [ColShape],
}

/// See `ColPic::col_shape`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ColShape {
    #[default]
    Small,
    Big16x16,
    Wide16x8,
    Tall8x16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Seg {
    None,
    Intra,
    P16x16,
    P16x8,
    P8x16,
    P8x8,
    Skip,
    /// B partitions, exported at the same granularity P8x8 already uses for
    /// its sub-8x8 splits (one row per 8x8 quadrant, using that quadrant's
    /// top-left 4x4 as the representative) — `B8x8` covers explicit B_8x8
    /// *and* whole-MB direct/skip that didn't promote to `B16x16` (ITU-T
    /// §8.4.1.2.2's per-quadrant colZeroFlag can make quadrants differ).
    B16x16,
    B16x8,
    B8x16,
    B8x8,
}

/// Per-frame decode state. All grids are indexed in 4x4-block raster units.
pub struct FrameGrids {
    mb_w: usize,
    mb_h: usize,
    bw: usize, // luma blocks per row = mb_w*4
    bh: usize,
    cw: usize, // chroma blocks per row = mb_w*2
    cvb: usize, // chroma blocks per MB column = 2 (4:2:0) or 4 (4:2:2)
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
    // B-slice-only (list 1). Allocated regardless (cheap relative to the L0
    // grids above) but only ever written/read when decoding B slices.
    mv1: Vec<[i32; 2]>,    // list 1 motion, per luma 4x4 block
    refi1: Vec<i32>,       // list 1 ref idx: >=0 inter, INTRA, or unset
    mvd_l1: Vec<[i32; 2]>, // |mvd| per luma 4x4 block, list 1 mvd context
    direct_l: Vec<bool>,   // per luma 4x4 block: decoded via B direct/skip mode
    mb_skip: Vec<bool>,    // per-MB: B_Skip (mb_skip_flag ctx of the *next* MB)
    mb_bdirect: Vec<bool>, // per-MB: B_Direct_16x16, explicit or skip-inferred (mb_type ctx of the *next* MB)
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
    /// `chroma422` selects the 2x4 (8-coefficient DC / 8 AC-block) chroma
    /// layout instead of the default 4:2:0 2x2 layout. 4:4:4
    /// (chroma_format_idc == 3) is out of scope and falls back to 4:2:0 sizing
    /// (unchanged pre-existing behaviour, still unsupported).
    pub fn new(mb_w: usize, mb_h: usize, chroma422: bool) -> Self {
        let bw = mb_w * 4;
        let bh = mb_h * 4;
        let cw = mb_w * 2;
        let cvb = if chroma422 { 4 } else { 2 };
        let ch = mb_h * cvb;
        FrameGrids {
            mb_w,
            mb_h,
            bw,
            bh,
            cw,
            cvb,
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
            mv1: vec![[0, 0]; bw * bh],
            refi1: vec![PART_NA; bw * bh],
            mvd_l1: vec![[0, 0]; bw * bh],
            direct_l: vec![false; bw * bh],
            mb_skip: vec![false; mb_w * mb_h],
            mb_bdirect: vec![false; mb_w * mb_h],
        }
    }

    /// Read-only snapshot of this picture's list-0 motion/ref grids, retained
    /// in a small DPB by the caller for future B pictures' spatial direct
    /// mode (colZeroFlag, ITU-T §8.4.1.2.2). `bw` is the stride needed to
    /// index the flat `mv`/`refi` vectors.
    pub fn mv_refi_snapshot(&self) -> (usize, Vec<[i32; 2]>, Vec<i32>) {
        (self.bw, self.mv.clone(), self.refi.clone())
    }

    /// List-1 counterpart of `mv_refi_snapshot`, for reference B pictures
    /// (hierarchical/pyramid B) retained in the DPB — a later B picture's
    /// direct mode may need this collocated picture's list-1 grids too (see
    /// `spatial_direct_quadrants`'s colZeroFlag list-1 fallback).
    pub fn mv_refi1_snapshot(&self) -> (Vec<[i32; 2]>, Vec<i32>) {
        (self.mv1.clone(), self.refi1.clone())
    }

    /// Per-macroblock collocated shape category, for `ColPic::col_shape`
    /// (see its doc comment). Retained in the DPB alongside the mv/ref
    /// grids so a later B picture's direct mode can look up this picture's
    /// macroblock shapes.
    pub fn col_shape_snapshot(&self) -> Vec<ColShape> {
        self.seg
            .iter()
            .map(|s| match s {
                Seg::Intra | Seg::P16x16 | Seg::Skip | Seg::B16x16 => ColShape::Big16x16,
                Seg::P16x8 | Seg::B16x8 => ColShape::Wide16x8,
                Seg::P8x16 | Seg::B8x16 => ColShape::Tall8x16,
                Seg::None | Seg::P8x8 | Seg::B8x8 => ColShape::Small,
            })
            .collect()
    }

    pub fn mb_count(&self) -> usize {
        self.mb_w * self.mb_h
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.mb_w, self.mb_h)
    }

    pub fn is_chroma422(&self) -> bool {
        self.cvb == 4
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
        // for CAVLC pictures, which never touch it. Same reasoning for the
        // list-1 grids, B-slice-only: unlike list 0 (where every decoded MB
        // unconditionally writes refi for all 4 of its blocks, so avail()
        // gating alone is enough), an L0-only B partition never touches
        // refi1/direct_l, so a stale non-PART_NA value from an earlier
        // picture at that slot would otherwise leak into this picture's
        // neighbour context.
        if cabac {
            self.mvd_l.fill([0, 0]);
            self.mvd_l1.fill([0, 0]);
            self.refi1.fill(PART_NA);
            self.direct_l.fill(false);
            self.mb_skip.fill(false);
            self.mb_bdirect.fill(false);
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
        self.nref_l(0, gx, gy, cur_mb, cur_scan, sid)
    }

    fn nmv(&self, gx: i32, gy: i32, cur_mb: usize, cur_scan: usize, sid: i32) -> [i32; 2] {
        self.nmv_l(0, gx, gy, cur_mb, cur_scan, sid)
    }

    /// `list`-parametrized ref idx of a neighbour block (0 = list 0, matching
    /// `nref`; 1 = list 1, B-slice-only).
    fn nref_l(&self, list: usize, gx: i32, gy: i32, cur_mb: usize, cur_scan: usize, sid: i32) -> i32 {
        if self.avail(gx, gy, cur_mb, cur_scan, sid) {
            let idx = self.lidx(gx as usize, gy as usize);
            if list == 0 { self.refi[idx] } else { self.refi1[idx] }
        } else {
            PART_NA
        }
    }

    fn nmv_l(&self, list: usize, gx: i32, gy: i32, cur_mb: usize, cur_scan: usize, sid: i32) -> [i32; 2] {
        if self.avail(gx, gy, cur_mb, cur_scan, sid) {
            let idx = self.lidx(gx as usize, gy as usize);
            if list == 0 { self.mv[idx] } else { self.mv1[idx] }
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
        self.pred_motion_l(0, mbx, mby, bx0, by0, pw, refr, sid)
    }

    /// `list`-parametrized `pred_motion` (0 = list 0, matching `pred_motion`;
    /// 1 = list 1, B-slice-only — same predictor, just reading the list-1
    /// neighbour grids).
    fn pred_motion_l(
        &self,
        list: usize,
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

        let left_ref = self.nref_l(list, x0 - 1, y0, cur_mb, cur_scan, sid);
        let a = self.nmv_l(list, x0 - 1, y0, cur_mb, cur_scan, sid);
        let top_ref = self.nref_l(list, x0, y0 - 1, cur_mb, cur_scan, sid);
        let b = self.nmv_l(list, x0, y0 - 1, cur_mb, cur_scan, sid);

        // diagonal C: top-right, else top-left
        let tr_ref = self.nref_l(list, x0 + pw, y0 - 1, cur_mb, cur_scan, sid);
        let (diag_ref, c) = if tr_ref != PART_NA {
            (tr_ref, self.nmv_l(list, x0 + pw, y0 - 1, cur_mb, cur_scan, sid))
        } else {
            (
                self.nref_l(list, x0 - 1, y0 - 1, cur_mb, cur_scan, sid),
                self.nmv_l(list, x0 - 1, y0 - 1, cur_mb, cur_scan, sid),
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
        // chroma availability mirrors luma at MB granularity (2 x cvb blocks/MB)
        let ch = self.nnz_c[c].len() / self.cw;
        let avail = |x: i32, y: i32| -> Option<usize> {
            if x < 0 || y < 0 || x >= self.cw as i32 || y >= ch as i32 {
                return None;
            }
            let m = (x as usize / 2) + (y as usize / self.cvb) * self.mb_w;
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
        self.fill_block_l(0, mbx, mby, bx, by, w, h, mv, refr);
    }

    /// `list`-parametrized `fill_block` (0 = list 0, matching `fill_block`;
    /// 1 = list 1, B-slice-only).
    fn fill_block_l(&mut self, list: usize, mbx: usize, mby: usize, bx: usize, by: usize, w: usize, h: usize, mv: [i32; 2], refr: i32) {
        for dy in 0..h {
            for dx in 0..w {
                let gx = mbx * 4 + bx + dx;
                let gy = mby * 4 + by + dy;
                let idx = gy * self.bw + gx;
                if list == 0 {
                    self.mv[idx] = mv;
                    self.refi[idx] = refr;
                } else {
                    self.mv1[idx] = mv;
                    self.refi1[idx] = refr;
                }
            }
        }
    }

    /// Mark every 4x4 block of an intra MB as INTRA on both lists (B slices
    /// can contain intra MBs; both list-0 and list-1 neighbour contexts need
    /// to see them as unavailable-for-inter, not stale/leftover data).
    fn set_mb_intra_both(&mut self, mbx: usize, mby: usize) {
        self.set_mb_refi(mbx, mby, INTRA);
        for by in 0..4 {
            for bx in 0..4 {
                let idx = (mby * 4 + by) * self.bw + (mbx * 4 + bx);
                self.refi1[idx] = INTRA;
                self.mv1[idx] = [0, 0];
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
    let cvb = g.cvb;
    if cbp & 0x30 != 0 {
        // chroma DC: n_c = -1/max_coeff 4 for 4:2:0 (2x2), -2/8 for 4:2:2 (2x4).
        // count not stored either way.
        let (n_c, max_coeff) = if cvb == 4 { (-2, 8) } else { (-1, 4) };
        for _c in 0..2 {
            if residual_block(r, n_c, max_coeff).is_none() {
                return false;
            }
        }
    }
    if cbp & 0x20 != 0 {
        for c in 0..2 {
            for i4x4 in 0..(2 * cvb) {
                let cx = i4x4 & 1;
                let cy = i4x4 >> 1;
                let gx = mbx * 2 + cx;
                let gy = mby * cvb + cy;
                let nc = g.nnz_pred_chroma(c, gx, gy, mb, sid);
                match residual_block(r, nc, 15) {
                    Some(tc) => g.nnz_c[c][gy * g.cw + gx] = tc as u8,
                    None => return false,
                }
            }
        }
    } else {
        for c in 0..2 {
            for cy in 0..cvb {
                for cx in 0..2 {
                    g.nnz_c[c][(mby * cvb + cy) * g.cw + (mbx * 2 + cx)] = 0;
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
    let cvb = g.cvb;
    for c in 0..2 {
        for cy in 0..cvb {
            for cx in 0..2 {
                g.nnz_c[c][(mby * cvb + cy) * g.cw + (mbx * 2 + cx)] = 0;
            }
        }
    }
}

fn set_mb_nnz(g: &mut FrameGrids, mb: usize, v: u8) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let cvb = g.cvb;
    for by in 0..4 {
        for bx in 0..4 {
            g.nnz_l[(mby * 4 + by) * g.bw + (mbx * 4 + bx)] = v;
        }
    }
    for c in 0..2 {
        for cy in 0..cvb {
            for cx in 0..2 {
                g.nnz_c[c][(mby * cvb + cy) * g.cw + (mbx * 2 + cx)] = v;
            }
        }
    }
}

// ── MV export, matching libavcodec/mpegutils.c add_mb() ───────────────────────

impl FrameGrids {
    /// Iterate every exported motion vector as `(sx, sy, w, h, mv)` — `(sx,sy)`
    /// is the partition centre (dst), `mv` the quarter-pel motion. Shared by the
    /// full and compact exporters so they can't diverge.
    /// `source` is -1 for list 0, +1 for list 1 (matching FFmpeg's
    /// `AVMotionVector.source` convention for backward/forward references).
    fn for_each_mv(&self, mut f: impl FnMut(i32, i32, i32, i32, i32, [i32; 2])) {
        for mby in 0..self.mb_h {
            for mbx in 0..self.mb_w {
                let mb = mby * self.mb_w + mbx;
                let mv_at = |bx: usize, by: usize| -> [i32; 2] {
                    self.mv[(mby * 4 + by) * self.bw + (mbx * 4 + bx)]
                };
                let x = (mbx * 16) as i32;
                let y = (mby * 16) as i32;
                match self.seg[mb] {
                    Seg::Skip | Seg::P16x16 => f(x + 8, y + 8, 16, 16, -1, mv_at(0, 0)),
                    Seg::P16x8 => {
                        f(x + 8, y + 4, 16, 8, -1, mv_at(0, 0));
                        f(x + 8, y + 12, 16, 8, -1, mv_at(0, 2));
                    }
                    Seg::P8x16 => {
                        f(x + 4, y + 8, 8, 16, -1, mv_at(0, 0));
                        f(x + 12, y + 8, 8, 16, -1, mv_at(2, 0));
                    }
                    Seg::P8x8 => {
                        for i in 0..4 {
                            let sx = x + 4 + 8 * (i & 1) as i32;
                            let sy = y + 4 + 8 * (i >> 1) as i32;
                            f(sx, sy, 8, 8, -1, mv_at((i & 1) * 2, (i >> 1) * 2));
                        }
                    }
                    // B partitions: emit an L0 row and/or an L1 row at the same
                    // representative position(s) P uses for the equivalent
                    // shape (P8x8's granularity for B8x8, which also covers
                    // direct/skip — see `Seg::B8x8`'s doc comment).
                    //
                    // List inclusion is gated at the *whole macroblock*, not
                    // per sub-block: FFmpeg's mpegutils.c checks mb_type's
                    // L0/L1 bits once per direction, outside the
                    // shape-specific per-partition loop (HAS_MV_EXT), so if
                    // *any* sub-block in the macroblock uses a list, *every*
                    // sub-block exports that list's motion — zero for the
                    // ones that don't actually use it, which is what
                    // FFmpeg's own "not used" fill already zeroes to. Below,
                    // B16x16 has a single representative position anyway
                    // (uniform by construction — see
                    // `decode_b_direct_whole_mb`), so `emit_b`'s existing
                    // per-position check is already whole-macroblock-level
                    // for it; B16x8/B8x16/B8x8 need the explicit OR since
                    // their partitions/quadrants can differ.
                    Seg::B16x16 => self.emit_b(&mut f, x + 8, y + 8, 16, 16, mbx, mby, 0, 0),
                    Seg::B16x8 => {
                        let i0 = mby * 4 * self.bw + mbx * 4;
                        let i1 = (mby * 4 + 2) * self.bw + mbx * 4;
                        let l0 = self.refi[i0] >= 0 || self.refi[i1] >= 0;
                        let l1 = self.refi1[i0] >= 0 || self.refi1[i1] >= 0;
                        if l0 {
                            f(x + 8, y + 4, 16, 8, -1, self.mv[i0]);
                            f(x + 8, y + 12, 16, 8, -1, self.mv[i1]);
                        }
                        if l1 {
                            f(x + 8, y + 4, 16, 8, 1, self.mv1[i0]);
                            f(x + 8, y + 12, 16, 8, 1, self.mv1[i1]);
                        }
                    }
                    Seg::B8x16 => {
                        let i0 = mby * 4 * self.bw + mbx * 4;
                        let i1 = mby * 4 * self.bw + mbx * 4 + 2;
                        let l0 = self.refi[i0] >= 0 || self.refi[i1] >= 0;
                        let l1 = self.refi1[i0] >= 0 || self.refi1[i1] >= 0;
                        if l0 {
                            f(x + 4, y + 8, 8, 16, -1, self.mv[i0]);
                            f(x + 12, y + 8, 8, 16, -1, self.mv[i1]);
                        }
                        if l1 {
                            f(x + 4, y + 8, 8, 16, 1, self.mv1[i0]);
                            f(x + 12, y + 8, 8, 16, 1, self.mv1[i1]);
                        }
                    }
                    Seg::B8x8 => {
                        let idxs = [
                            mby * 4 * self.bw + mbx * 4,
                            mby * 4 * self.bw + mbx * 4 + 2,
                            (mby * 4 + 2) * self.bw + mbx * 4,
                            (mby * 4 + 2) * self.bw + mbx * 4 + 2,
                        ];
                        let l0 = idxs.iter().any(|&i| self.refi[i] >= 0);
                        let l1 = idxs.iter().any(|&i| self.refi1[i] >= 0);
                        for (i, &idx) in idxs.iter().enumerate() {
                            let sx = x + 4 + 8 * (i as i32 & 1);
                            let sy = y + 4 + 8 * (i as i32 >> 1);
                            if l0 {
                                f(sx, sy, 8, 8, -1, self.mv[idx]);
                            }
                            if l1 {
                                f(sx, sy, 8, 8, 1, self.mv1[idx]);
                            }
                        }
                    }
                    Seg::Intra | Seg::None => {}
                }
            }
        }
    }

    /// Emit the L0 row (if that list is active at this 4x4 block) and/or the
    /// L1 row, at the same `(sx,sy,w,h)` representative position.
    #[allow(clippy::too_many_arguments)]
    fn emit_b(
        &self,
        f: &mut dyn FnMut(i32, i32, i32, i32, i32, [i32; 2]),
        sx: i32,
        sy: i32,
        w: i32,
        h: i32,
        mbx: usize,
        mby: usize,
        bx: usize,
        by: usize,
    ) {
        let idx = (mby * 4 + by) * self.bw + (mbx * 4 + bx);
        if self.refi[idx] >= 0 {
            f(sx, sy, w, h, -1, self.mv[idx]);
        }
        if self.refi1[idx] >= 0 {
            f(sx, sy, w, h, 1, self.mv1[idx]);
        }
    }

    /// Full `AVMotionVector`-layout export (12 columns). `l0_only` drops
    /// list-1 (`source > 0`, forward-reference) rows, matching the custom
    /// FFmpeg fork's `mv_l0_only` AVOption (see `mpegutils.c`'s
    /// `ff_print_debug_info2_optimized`).
    pub fn export_mvs(&self, frame: i32, l0_only: bool, out: &mut Vec<MotionVector>) {
        let mut flt = MvFilter::new();
        self.for_each_mv(|sx, sy, w, h, source, mv| {
            if l0_only && source > 0 {
                return;
            }
            let src_x = sx + mv[0] / 4;
            let src_y = sy + mv[1] / 4;
            if src_x == sx && src_y == sy {
                return; // zero-size vector: no displacement, skip
            }
            if !flt.keep(src_x, src_y, sx, sy, sx, sy) {
                return; // MV_MIN_SIZE / MV_EVERY_NTH
            }
            out.push(MotionVector {
                frame,
                source,
                w,
                h,
                src_x: src_x as f64,
                src_y: src_y as f64,
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
    /// write than the full format. `l0_only`: see `export_mvs`.
    pub fn export_mvs_compact(&self, frame: i32, l0_only: bool, out: &mut Vec<MvCompact>) {
        let mut flt = MvFilter::new();
        self.for_each_mv(|sx, sy, _w, _h, source, mv| {
            if l0_only && source > 0 {
                return;
            }
            let src_x = sx + mv[0] / 4;
            let src_y = sy + mv[1] / 4;
            if src_x == sx && src_y == sy {
                return; // zero-size vector: no displacement, skip
            }
            if !flt.keep(src_x, src_y, sx, sy, sx, sy) {
                return; // MV_MIN_SIZE / MV_EVERY_NTH
            }
            out.push(MvCompact {
                frame,
                source,
                src_x: src_x as i16,
                src_y: src_y as i16,
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
// Chroma-DC 4:2:2 uses a narrower context range (ctx 8, not 9, at node 7) —
// ffmpeg h264_cabac.c `coeff_abs_levelgt1_ctx[1]`.
const ABS_GT1_CTX_DC422: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 8];
const TRANS0: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
const TRANS1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];
// ctxIdxInc for chroma-DC significant_coeff_flag/last_significant_coeff_flag
// when ChromaArrayType == 2 (4:2:2): Min(levelListIdx / NumC8x8, 2), NumC8x8=2
// (spec Table 9-43 / ffmpeg `sig_coeff_offset_dc`).
const SIG_COEFF_OFFSET_DC_422: [usize; 7] = [0, 0, 1, 1, 2, 2, 2];

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
        self.amvd_l(0, mb, sid, bx, by)
    }

    /// `list`-parametrized `amvd` (0 = list 0, matching `amvd`; 1 = list 1,
    /// B-slice-only).
    fn amvd_l(&self, list: usize, mb: usize, sid: i32, bx: usize, by: usize) -> (i32, i32) {
        let scan = xy_scan(bx, by);
        let x0 = (mb % self.mb_w * 4 + bx) as i32;
        let y0 = (mb / self.mb_w * 4 + by) as i32;
        let mvd = if list == 0 { &self.mvd_l } else { &self.mvd_l1 };
        let l = if self.avail(x0 - 1, y0, mb, scan, sid) {
            mvd[self.lidx((x0 - 1) as usize, y0 as usize)]
        } else {
            [0, 0]
        };
        let t = if self.avail(x0, y0 - 1, mb, scan, sid) {
            mvd[self.lidx(x0 as usize, (y0 - 1) as usize)]
        } else {
            [0, 0]
        };
        (l[0] + t[0], l[1] + t[1])
    }

    fn fill_mvd(&mut self, mbx: usize, mby: usize, bx: usize, by: usize, w: usize, h: usize, v: [i32; 2]) {
        self.fill_mvd_l(0, mbx, mby, bx, by, w, h, v);
    }

    /// `list`-parametrized `fill_mvd` (0 = list 0, matching `fill_mvd`; 1 =
    /// list 1, B-slice-only).
    fn fill_mvd_l(&mut self, list: usize, mbx: usize, mby: usize, bx: usize, by: usize, w: usize, h: usize, v: [i32; 2]) {
        for dy in 0..h {
            for dx in 0..w {
                let idx = (mby * 4 + by + dy) * self.bw + (mbx * 4 + bx + dx);
                if list == 0 { self.mvd_l[idx] = v } else { self.mvd_l1[idx] = v };
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

    /// B-slice ref_idx context (ITU-T Table 9-34 / FFmpeg `decode_cabac_mb_
    /// ref`): like `ref_neighbor`, but a neighbour coded via direct mode
    /// doesn't count even if its ref idx is positive.
    fn ref_neighbor_b(&self, list: usize, mb: usize, sid: i32, bx: usize, by: usize) -> usize {
        let scan = xy_scan(bx, by);
        let x0 = (mb % self.mb_w * 4 + bx) as i32;
        let y0 = (mb / self.mb_w * 4 + by) as i32;
        let refa = self.nref_l(list, x0 - 1, y0, mb, scan, sid);
        let refb = self.nref_l(list, x0, y0 - 1, mb, scan, sid);
        let dira = self.avail(x0 - 1, y0, mb, scan, sid) && self.direct_l[self.lidx((x0 - 1) as usize, y0 as usize)];
        let dirb = self.avail(x0, y0 - 1, mb, scan, sid) && self.direct_l[self.lidx(x0 as usize, (y0 - 1) as usize)];
        (refa > 0 && !dira) as usize + 2 * (refb > 0 && !dirb) as usize
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

/// `list`-parametrized `pred_16x8` (0 = list 0; 1 = list 1, B-slice-only).
fn pred_16x8_l(g: &FrameGrids, list: usize, mbx: usize, mby: usize, part: usize, refr: i32, sid: i32) -> (i32, i32) {
    let cur_mb = mby * g.mb_w + mbx;
    if part == 0 {
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4) as i32;
        let top_ref = g.nref_l(list, x0, y0 - 1, cur_mb, xy_scan(0, 0), sid);
        if top_ref == refr {
            let b = g.nmv_l(list, x0, y0 - 1, cur_mb, xy_scan(0, 0), sid);
            return (b[0], b[1]);
        }
        g.pred_motion_l(list, mbx, mby, 0, 0, 4, refr, sid)
    } else {
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4 + 2) as i32;
        let left_ref = g.nref_l(list, x0 - 1, y0, cur_mb, xy_scan(0, 2), sid);
        if left_ref == refr {
            let a = g.nmv_l(list, x0 - 1, y0, cur_mb, xy_scan(0, 2), sid);
            return (a[0], a[1]);
        }
        g.pred_motion_l(list, mbx, mby, 0, 2, 4, refr, sid)
    }
}

/// `list`-parametrized `pred_8x16` (0 = list 0; 1 = list 1, B-slice-only).
fn pred_8x16_l(g: &FrameGrids, list: usize, mbx: usize, mby: usize, part: usize, refr: i32, sid: i32) -> (i32, i32) {
    let cur_mb = mby * g.mb_w + mbx;
    if part == 0 {
        let x0 = (mbx * 4) as i32;
        let y0 = (mby * 4) as i32;
        let left_ref = g.nref_l(list, x0 - 1, y0, cur_mb, xy_scan(0, 0), sid);
        if left_ref == refr {
            let a = g.nmv_l(list, x0 - 1, y0, cur_mb, xy_scan(0, 0), sid);
            return (a[0], a[1]);
        }
        g.pred_motion_l(list, mbx, mby, 0, 0, 2, refr, sid)
    } else {
        let x0 = (mbx * 4 + 2) as i32;
        let y0 = (mby * 4) as i32;
        let tr_ref = g.nref_l(list, x0 + 2, y0 - 1, cur_mb, xy_scan(2, 0), sid);
        let (diag_ref, c) = if tr_ref != PART_NA {
            (tr_ref, g.nmv_l(list, x0 + 2, y0 - 1, cur_mb, xy_scan(2, 0), sid))
        } else {
            (
                g.nref_l(list, x0 - 1, y0 - 1, cur_mb, xy_scan(2, 0), sid),
                g.nmv_l(list, x0 - 1, y0 - 1, cur_mb, xy_scan(2, 0), sid),
            )
        };
        if diag_ref == refr {
            return (c[0], c[1]);
        }
        g.pred_motion_l(list, mbx, mby, 2, 0, 2, refr, sid)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// B-slice CABAC decoding (ITU-T §7.3.5.1 mb_pred()/sub_mb_pred() for B slices,
// §8.4.1.2.2 spatial direct mode). Ported from FFmpeg h264_cabac.c /
// h264_direct.c. Scope, enforced by the caller before calling in: CABAC only,
// spatial direct mode only (direct_spatial_mv_pred_flag == 1),
// direct_8x8_inference_flag == 1, frame-only (no MBAFF/fields), short-term
// references only (matching the rest of this decoder), and pic_order_cnt_type
// == 0 (needed to order the DPB for the collocated picture).
// ═════════════════════════════════════════════════════════════════════════════

/// ff_h264_b_mb_type_info (h264data.c): (shape, p0l0, p0l1, p1l0, p1l1).
/// shape: 0=16x16, 1=16x8, 2=8x16, 3=8x8 (partition_count 4, via sub_mb_type).
/// Index 0 (B_Direct_16x16) is handled before this table is consulted.
const B_MB_TYPE_INFO: [(u8, bool, bool, bool, bool); 23] = [
    (0, false, false, false, false), // 0: B_Direct_16x16 (unused directly)
    (0, true, false, false, false),  // 1: B_L0_16x16
    (0, false, true, false, false),  // 2: B_L1_16x16
    (0, true, true, false, false),   // 3: B_Bi_16x16
    (1, true, false, true, false),   // 4: B_L0_L0_16x8
    (2, true, false, true, false),   // 5: B_L0_L0_8x16
    (1, false, true, false, true),   // 6: B_L1_L1_16x8
    (2, false, true, false, true),   // 7: B_L1_L1_8x16
    (1, true, false, false, true),   // 8: B_L0_L1_16x8
    (2, true, false, false, true),   // 9: B_L0_L1_8x16
    (1, false, true, true, false),   // 10: B_L1_L0_16x8
    (2, false, true, true, false),   // 11: B_L1_L0_8x16
    (1, true, false, true, true),    // 12: B_L0_Bi_16x8
    (2, true, false, true, true),    // 13: B_L0_Bi_8x16
    (1, false, true, true, true),    // 14: B_L1_Bi_16x8
    (2, false, true, true, true),    // 15: B_L1_Bi_8x16
    (1, true, true, true, false),    // 16: B_Bi_L0_16x8
    (2, true, true, true, false),    // 17: B_Bi_L0_8x16
    (1, true, true, false, true),    // 18: B_Bi_L1_16x8
    (2, true, true, false, true),    // 19: B_Bi_L1_8x16
    (1, true, true, true, true),     // 20: B_Bi_Bi_16x8
    (2, true, true, true, true),     // 21: B_Bi_Bi_8x16
    (3, true, true, true, true),     // 22: B_8x8 (via sub_mb_type)
];

/// ff_h264_b_sub_mb_type_info (h264data.c): (shape, p0l0, p0l1, p1l0, p1l1).
/// shape (relative to this 8x8): 0=undivided, 1=two 8x4 (top/bottom),
/// 2=two 4x8 (left/right), 3=four 4x4. Index 0 (B_Direct_8x8) is handled
/// before this table is consulted.
const B_SUB_MB_TYPE_INFO: [(u8, bool, bool, bool, bool); 13] = [
    (0, false, false, false, false), // 0: B_Direct_8x8 (unused directly)
    (0, true, false, false, false),  // 1: B_L0_8x8
    (0, false, true, false, false),  // 2: B_L1_8x8
    (0, true, true, false, false),   // 3: B_Bi_8x8
    (1, true, false, true, false),   // 4: B_L0_L0_8x4
    (2, true, false, true, false),   // 5: B_L0_L0_4x8
    (1, false, true, false, true),   // 6: B_L1_L1_8x4
    (2, false, true, false, true),   // 7: B_L1_L1_4x8
    (1, true, true, true, true),     // 8: B_Bi_Bi_8x4
    (2, true, true, true, true),     // 9: B_Bi_Bi_4x8
    (3, true, false, true, false),   // 10: B_L0_L0_4x4
    (3, false, true, false, true),   // 11: B_L1_L1_4x4
    (3, true, true, true, true),     // 12: B_Bi_Bi_4x4
];

/// mb_type, B slices (ITU-T Table 9-37 binarization, FFmpeg's exact bin
/// layout). `None` => this is actually an intra mb_type (bits==13); the
/// caller falls back to `decode_intra_mb_type_cabac(..., 32, false)`.
/// Otherwise `Some(index into B_MB_TYPE_INFO)`.
///
/// `left_cond`/`top_cond` are the bin-0 ctxIdxInc condTermFlags: true only
/// when that neighbour is *available and not itself direct/skip*. An
/// unavailable neighbour contributes 0, same as an actual direct/skip one —
/// FFmpeg gets this via `IS_DIRECT(left_type - 1)`, where the missing-
/// neighbour sentinel `left_type == 0` makes `0 - 1 == -1`, and `-1 &
/// MB_TYPE_DIRECT2` is nonzero (all bits set), so `!IS_DIRECT(-1)` is false.
fn decode_b_mb_type_cabac(cd: &mut Cabd, left_cond: bool, top_cond: bool) -> Option<u32> {
    let ctx = left_cond as usize + top_cond as usize;
    if cd.get(27 + ctx) == 0 {
        return Some(0); // B_Direct_16x16
    }
    if cd.get(27 + 3) == 0 {
        return Some(1 + cd.get(27 + 5));
    }
    let mut bits = cd.get(27 + 4) << 3;
    bits += cd.get(27 + 5) << 2;
    bits += cd.get(27 + 5) << 1;
    bits += cd.get(27 + 5);
    if bits < 8 {
        Some(bits + 3)
    } else if bits == 13 {
        None
    } else if bits == 14 {
        Some(11)
    } else if bits == 15 {
        Some(22)
    } else {
        let bits2 = (bits << 1) + cd.get(27 + 5);
        Some(bits2 - 4)
    }
}

/// sub_mb_type, B slices (FFmpeg decode_cabac_b_mb_sub_type). Returns an
/// index into B_SUB_MB_TYPE_INFO.
fn decode_b_sub_mb_type_cabac(cd: &mut Cabd) -> u32 {
    if cd.get(36) == 0 {
        return 0; // B_Direct_8x8
    }
    if cd.get(37) == 0 {
        return 1 + cd.get(39); // B_L0_8x8 / B_L1_8x8
    }
    let mut t = 3;
    if cd.get(38) == 1 {
        if cd.get(39) == 1 {
            return 11 + cd.get(39); // B_L1_4x4 / B_Bi_4x4
        }
        t += 4;
    }
    t += 2 * cd.get(39);
    t += cd.get(39);
    t
}

/// decode ref_idx for a B partition (0 when only one reference is active).
fn read_ref_cabac_b(g: &FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, bx: usize, by: usize, list: usize, n: u32) -> i32 {
    if n <= 1 {
        0
    } else {
        decode_ref_cabac(cd, g.ref_neighbor_b(list, mb, sid, bx, by))
    }
}

/// Minimum of the non-negative candidates (ITU-T §8.4.1.2.2's ref idx
/// derivation, treating unavailable/intra neighbours as not a candidate), or
/// -1 if none qualify.
fn min_nonneg3(a: i32, b: i32, c: i32) -> i32 {
    [a, b, c].into_iter().filter(|&v| v >= 0).min().unwrap_or(-1)
}

/// Spatial direct predictor + per-quadrant colZeroFlag (ITU-T §8.4.1.2.2,
/// direct_8x8_inference_flag path — the 4x4 luma block sampled from the
/// collocated picture for each 8x8 quadrant is the one at that quadrant's
/// *outer* corner of the macroblock). `col` is the collocated picture's
/// motion/ref snapshot. Returns `(ref0, ref1, [quadrant0..3: (mv0, mv1,
/// colZeroFlag)])`.
///
/// The flag is exposed (not just baked into mv0/mv1) because FFmpeg's
/// whole-MB promotion criterion is nominally "did colZeroFlag fire the same
/// way in all 4 quadrants" (`n == 0 || n == 16` in `pred_spatial_direct_
/// motion`'s per-quadrant loop) rather than "do the 4 resulting values
/// happen to be numerically equal" — tried switching the check in
/// `decode_b_direct_whole_mb` to use this flag instead of value equality,
/// which regressed dramatically (bus 84 → 148,276; a 4:2:2 test clip 7,360
/// → 2,457,483 mismatched list-0 vectors), so value equality is what's
/// actually used below — this decoder's colZeroFlag computation evidently
/// doesn't track FFmpeg's own closely enough for the flag-based check to be
/// safe, and unlike the flag, the *value* a quadrant ends up with is
/// unaffected by that: `ref_lx[list] != 0` makes colZeroFlag's zeroing a
/// no-op regardless of whether it fired, so value equality is the more
/// robust signal even where the two aren't theoretically identical. The
/// flag stays part of the return type in case a correct per-quadrant
/// colZeroFlag computation is worth revisiting later.
fn spatial_direct_quadrants(
    g: &FrameGrids,
    mb: usize,
    sid: i32,
    col: &ColPic,
) -> (i32, i32, [([i32; 2], [i32; 2], bool); 4]) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let x0 = (mbx * 4) as i32;
    let y0 = (mby * 4) as i32;

    let mut ref_lx = [-1i32; 2];
    let mut mv_lx = [[0i32; 2]; 2];
    for list in 0..2 {
        let left_ref = g.nref_l(list, x0 - 1, y0, mb, 0, sid);
        let a = g.nmv_l(list, x0 - 1, y0, mb, 0, sid);
        let top_ref = g.nref_l(list, x0, y0 - 1, mb, 0, sid);
        let b = g.nmv_l(list, x0, y0 - 1, mb, 0, sid);
        let tr_ref = g.nref_l(list, x0 + 4, y0 - 1, mb, 0, sid);
        let (diag_ref, c) = if tr_ref != PART_NA {
            (tr_ref, g.nmv_l(list, x0 + 4, y0 - 1, mb, 0, sid))
        } else {
            (g.nref_l(list, x0 - 1, y0 - 1, mb, 0, sid), g.nmv_l(list, x0 - 1, y0 - 1, mb, 0, sid))
        };
        let r = min_nonneg3(left_ref, top_ref, diag_ref);
        if r >= 0 {
            let mc = (left_ref == r) as i32 + (top_ref == r) as i32 + (diag_ref == r) as i32;
            ref_lx[list] = r;
            mv_lx[list] = if mc > 1 {
                [mid_pred(a[0], b[0], c[0]), mid_pred(a[1], b[1], c[1])]
            } else if left_ref == r {
                a
            } else if top_ref == r {
                b
            } else {
                c
            };
        }
    }
    if ref_lx[0] < 0 && ref_lx[1] < 0 {
        ref_lx[0] = 0;
        ref_lx[1] = 0;
    }

    let mut quadrants = [([0i32; 2], [0i32; 2], false); 4];
    for (i8, q) in quadrants.iter_mut().enumerate() {
        let x8 = i8 & 1;
        let y8 = i8 >> 1;
        let ccx = mbx * 4 + x8 * 3;
        let ccy = mby * 4 + y8 * 3;
        let cidx = ccy * col.bw + ccx;
        let col_ref0 = col.refi0.get(cidx).copied().unwrap_or(PART_NA);
        let col_mv0 = col.mv0.get(cidx).copied().unwrap_or([0, 0]);
        // ITU-T §8.4.1.2.2 / FFmpeg h264_direct.c: colZeroFlag is list-0's
        // (ref==0, |mv|<=1) OR — only reachable when the collocated picture
        // has a list 1, i.e. it's itself a reference B picture — list-0
        // unused there but list-1's (ref==0, |mv|<=1).
        let col_zero = if col_ref0 == 0 {
            col_mv0[0].abs() <= 1 && col_mv0[1].abs() <= 1
        } else if col_ref0 < 0 {
            match (col.refi1, col.mv1) {
                (Some(refi1), Some(mv1)) => {
                    let col_ref1 = refi1.get(cidx).copied().unwrap_or(PART_NA);
                    let col_mv1v = mv1.get(cidx).copied().unwrap_or([0, 0]);
                    col_ref1 == 0 && col_mv1v[0].abs() <= 1 && col_mv1v[1].abs() <= 1
                }
                _ => false,
            }
        } else {
            false
        };
        let mut a = mv_lx[0];
        let mut b = mv_lx[1];
        if col_zero {
            if ref_lx[0] == 0 {
                a = [0, 0];
            }
            if ref_lx[1] == 0 {
                b = [0, 0];
            }
        }
        *q = (a, b, col_zero);
    }
    (ref_lx[0], ref_lx[1], quadrants)
}

/// Fill one 8x8 quadrant's list-0/list-1 motion+ref and mark it direct-coded
/// (for the next MB's ref_idx context).
fn fill_direct_quadrant(g: &mut FrameGrids, mb: usize, i8: usize, ref0: i32, ref1: i32, mv0: [i32; 2], mv1: [i32; 2]) {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let qx = (i8 & 1) * 2;
    let qy = (i8 >> 1) * 2;
    g.fill_block_l(0, mbx, mby, qx, qy, 2, 2, mv0, ref0);
    g.fill_block_l(1, mbx, mby, qx, qy, 2, 2, mv1, ref1);
    for dy in 0..2 {
        for dx in 0..2 {
            let idx = (mby * 4 + qy + dy) * g.bw + (mbx * 4 + qx + dx);
            g.direct_l[idx] = true;
        }
    }
}

/// Whole-MB direct mode (B_Direct_16x16 mb_type, or the B_Skip inference —
/// same underlying computation). Returns the export `Seg` (the compact CSV
/// has no width/height column, so the exported row *count and position*
/// must match FFmpeg's).
///
/// Two rules, matching FFmpeg h264_direct.c's `pred_spatial_direct_motion`
/// (`single_col` label) in its actual order:
///
/// 1. If all 4 independently-computed quadrant values agree, `B16x16` —
///    this is meant to mirror FFmpeg's `if (!is_b8x8 && !(n & 15)) *mb_type
///    = ... MB_TYPE_16x16` check at the end of its per-quadrant colZero
///    loop (`n` counts, in steps of 4, how many quadrants' colZeroFlag
///    fired; `n==0`/`n==16` mean "fired nowhere"/"fired everywhere"), which
///    isn't quite the same thing as value equality (colZeroFlag firing only
///    changes a quadrant's value when the pre-zeroing neighbour-predicted
///    mv isn't already `[0, 0]`, so a fired/not-fired mix can still produce
///    equal values). Tried switching to a direct colZeroFlag-uniformity
///    check instead — regressed dramatically (bus 84 → 148,276; a 4:2:2
///    test clip 7,360 → 2,457,483 mismatched list-0 vectors), so this
///    decoder's colZeroFlag computation evidently doesn't track FFmpeg's own
///    closely enough for the flag-based check to be safe (see
///    `spatial_direct_quadrants`'s doc comment). Value equality stays
///    because it's what's actually verified to work.
/// 2. Otherwise, FFmpeg *statically* sets the shape to the collocated MB's
///    own 16x8/8x16 partitioning if it has one (`mb_type_col[0] &
///    (MB_TYPE_16x8|MB_TYPE_8x16)`, assigned *before* the colZero loop even
///    runs, no numeric check) — so a non-uniform result doesn't always mean
///    `B8x8`; it's `B16x8`/`B8x16` when the collocated MB was that shape,
///    and `B8x8` only when the collocated MB was itself 8x8-or-finer.
fn decode_b_direct_whole_mb(g: &mut FrameGrids, mb: usize, sid: i32, col: &ColPic) -> Seg {
    let (ref0, ref1, quadrants) = spatial_direct_quadrants(g, mb, sid, col);
    for (i8, &(a, b, _)) in quadrants.iter().enumerate() {
        fill_direct_quadrant(g, mb, i8, ref0, ref1, a, b);
    }
    let vals: [([i32; 2], [i32; 2]); 4] = std::array::from_fn(|i| (quadrants[i].0, quadrants[i].1));
    if vals.iter().all(|q| *q == vals[0]) {
        Seg::B16x16
    } else {
        match col.col_shape.get(mb).copied().unwrap_or_default() {
            ColShape::Wide16x8 => Seg::B16x8,
            ColShape::Tall8x16 => Seg::B8x16,
            ColShape::Big16x16 | ColShape::Small => Seg::B8x8,
        }
    }
}

/// B_8x8: per-quadrant sub_mb_type, then ref_idx and mv/mvd — both passes
/// ordered list-major then quadrant-minor to match the CABAC bitstream's
/// exact bin order (FFmpeg decodes all of list 0's ref_idx/mv before any of
/// list 1's, not interleaved per quadrant). Returns whether every quadrant
/// (direct ones included — direct never subdivides under
/// direct_8x8_inference_flag) is 8x8-shaped, for the caller's dct8x8_allowed.
fn decode_b_8x8_cabac(
    g: &mut FrameGrids,
    cd: &mut Cabd,
    mb: usize,
    sid: i32,
    sh: &SliceHeader,
    col: &ColPic,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    let n0 = sh.num_ref_idx_l0_active;
    let n1 = sh.num_ref_idx_l1_active;

    let mut sub = [0u32; 4];
    for s in sub.iter_mut() {
        *s = decode_b_sub_mb_type_cabac(cd);
    }
    let info: [(u8, bool, bool, bool, bool); 4] = std::array::from_fn(|i| B_SUB_MB_TYPE_INFO[sub[i] as usize]);
    let is_direct = |i: usize| sub[i] == 0;

    let any_direct = (0..4).any(is_direct);
    let mut dquad = [([0i32; 2], [0i32; 2], false); 4];
    let (dref0, dref1) = if any_direct {
        let (r0, r1, q) = spatial_direct_quadrants(g, mb, sid, col);
        dquad = q;
        (r0, r1)
    } else {
        (0, 0)
    };
    for i8 in 0..4 {
        if is_direct(i8) {
            let (a, b, _) = dquad[i8];
            fill_direct_quadrant(g, mb, i8, dref0, dref1, a, b);
        }
    }

    // ref_idx: list-major, quadrant-minor (matches the CABAC bit order).
    let mut refs = [[-1i32; 2]; 4]; // refs[i8][list]
    for list in 0..2 {
        for i8 in 0..4 {
            if is_direct(i8) {
                continue;
            }
            let (_, p0l0, p0l1, _, _) = info[i8];
            let (qx, qy) = ((i8 & 1) * 2, (i8 >> 1) * 2);
            if !(if list == 0 { p0l0 } else { p0l1 }) {
                // FFmpeg's LIST_NOT_USED fill — see the 16x16 branch's comment.
                g.fill_block_l(list, mbx, mby, qx, qy, 2, 2, [0, 0], INTRA);
                continue;
            }
            let n = if list == 0 { n0 } else { n1 };
            let r = read_ref_cabac_b(g, cd, mb, sid, qx, qy, list, n);
            refs[i8][list] = r;
            g.fill_block_l(list, mbx, mby, qx, qy, 2, 2, [0, 0], r);
        }
    }

    // mv/mvd: list-major, quadrant-minor, sub-partition-minor-most.
    for list in 0..2 {
        for i8 in 0..4 {
            if is_direct(i8) {
                continue;
            }
            let (shape, p0l0, p0l1, _, _) = info[i8];
            if !(if list == 0 { p0l0 } else { p0l1 }) {
                continue;
            }
            let (qx, qy) = ((i8 & 1) * 2, (i8 >> 1) * 2);
            let refr = refs[i8][list];
            let parts: &[(usize, usize, usize, usize)] = match shape {
                0 => &[(0, 0, 2, 2)],
                1 => &[(0, 0, 2, 1), (0, 1, 2, 1)],
                2 => &[(0, 0, 1, 2), (1, 0, 1, 2)],
                _ => &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
            };
            for &(ox, oy, pw, ph) in parts {
                let bx = qx + ox;
                let by = qy + oy;
                let (mx, my) = g.pred_motion_l(list, mbx, mby, bx, by, pw as i32, refr, sid);
                let (ax, ay) = g.amvd_l(list, mb, sid, bx, by);
                let (dx, cx) = decode_mvd_cabac(cd, 40, ax);
                let (dy, cy) = decode_mvd_cabac(cd, 47, ay);
                g.fill_block_l(list, mbx, mby, bx, by, pw, ph, [mx + dx, my + dy], refr);
                g.fill_mvd_l(list, mbx, mby, bx, by, pw, ph, [cx, cy]);
            }
        }
    }

    (0..4).all(|i| is_direct(i) || B_SUB_MB_TYPE_INFO[sub[i] as usize].0 == 0)
}

/// CBP + transform_size_8x8_flag + residual — identical to the tail of
/// `decode_p_inter_cabac`, shared verbatim since residual coding doesn't
/// depend on list direction, only on cbp/intra-ness. `mb_type_lt_3` mirrors
/// P's "mb_type < 3" dct8x8_allowed condition (true for 16x16/16x8/8x16 and
/// for whole-MB direct, which is never subdivided finer than 8x8 under
/// direct_8x8_inference_flag).
fn finish_b_inter(g: &mut FrameGrids, cd: &mut Cabd, mb: usize, sid: i32, pps: &Pps, mb_type_lt_3_or_all_8x8: bool) -> bool {
    let cbp = decode_cbp_cabac(g, cd, mb, sid, false);
    let dct8_allowed = pps.transform_8x8_mode_flag && mb_type_lt_3_or_all_8x8;
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

fn decode_mb_cabac_b(
    g: &mut FrameGrids,
    cd: &mut Cabd,
    mb: usize,
    sid: i32,
    sh: &SliceHeader,
    pps: &Pps,
    col: &ColPic,
) -> bool {
    let mbx = mb % g.mb_w;
    let mby = mb / g.mb_w;
    g.slice_id[mb] = sid;

    let (l, t) = g.ctx_neighbors(mb, sid);
    let left_cond = l.is_some_and(|m| !g.mb_bdirect[m]);
    let top_cond = t.is_some_and(|m| !g.mb_bdirect[m]);

    let Some(mb_type) = decode_b_mb_type_cabac(cd, left_cond, top_cond) else {
        // intra escape (same base pattern as P's intra escape, ctx_base 32)
        let i_mb_type = decode_intra_mb_type_cabac(g, cd, mb, sid, 32, false);
        if i_mb_type == 25 {
            let p = cabac_pcm_byte_pos(cd);
            let new_start = p + 384;
            cd.c = Cabac::new(cd.c_bytes(), new_start);
            cd.last_qscale_nonzero = false;
            set_mb_nnz(g, mb, 16);
            g.cbp_mb[mb] = 0x7EF | 0x800;
            g.set_mb_intra_both(mbx, mby);
            g.seg[mb] = Seg::Intra;
            g.decoded[mb] = true;
            return true;
        }
        g.seg[mb] = Seg::Intra;
        decode_intra_residual_cabac(g, cd, mb, sid, i_mb_type, pps);
        g.set_mb_intra_both(mbx, mby);
        g.decoded[mb] = true;
        return true;
    };

    if mb_type == 0 {
        let seg = decode_b_direct_whole_mb(g, mb, sid, col);
        g.mb_bdirect[mb] = true;
        g.seg[mb] = seg;
        g.decoded[mb] = true;
        return finish_b_inter(g, cd, mb, sid, pps, true);
    }

    if mb_type == 22 {
        let all_8x8 = decode_b_8x8_cabac(g, cd, mb, sid, sh, col);
        g.seg[mb] = Seg::B8x8;
        g.decoded[mb] = true;
        return finish_b_inter(g, cd, mb, sid, pps, all_8x8);
    }

    let n0 = sh.num_ref_idx_l0_active;
    let n1 = sh.num_ref_idx_l1_active;
    let (shape, p0l0, p0l1, p1l0, p1l1) = B_MB_TYPE_INFO[mb_type as usize];

    let seg = if shape == 0 {
        let uses = [(p0l0, 0usize), (p0l1, 1usize)];
        let mut refs = [-1i32; 2];
        for &(u, list) in &uses {
            if u {
                let n = if list == 0 { n0 } else { n1 };
                refs[list] = read_ref_cabac_b(g, cd, mb, sid, 0, 0, list, n);
                g.fill_block_l(list, mbx, mby, 0, 0, 4, 4, [0, 0], refs[list]);
            } else {
                // Explicitly mark unused (FFmpeg's LIST_NOT_USED): a same-MB
                // neighbour lookup from another partition must see "not
                // used", not stale data left over from a previous picture at
                // this grid slot.
                g.fill_block_l(list, mbx, mby, 0, 0, 4, 4, [0, 0], INTRA);
            }
        }
        for &(u, list) in &uses {
            if u {
                let refr = refs[list];
                let (mx, my) = g.pred_motion_l(list, mbx, mby, 0, 0, 4, refr, sid);
                let (ax, ay) = g.amvd_l(list, mb, sid, 0, 0);
                let (dx, cx) = decode_mvd_cabac(cd, 40, ax);
                let (dy, cy) = decode_mvd_cabac(cd, 47, ay);
                g.fill_block_l(list, mbx, mby, 0, 0, 4, 4, [mx + dx, my + dy], refr);
                g.fill_mvd_l(list, mbx, mby, 0, 0, 4, 4, [cx, cy]);
            }
        }
        Seg::B16x16
    } else {
        let parts: [(usize, usize, usize, usize); 2] = if shape == 1 {
            [(0, 0, 4, 2), (0, 2, 4, 2)]
        } else {
            [(0, 0, 2, 4), (2, 0, 2, 4)]
        };
        let part_uses = [(p0l0, p0l1), (p1l0, p1l1)];
        let mut refs = [[-1i32; 2]; 2]; // refs[partition][list]
        for list in 0..2 {
            for (pi, &(bx, by, pw, ph)) in parts.iter().enumerate() {
                let uses = if list == 0 { part_uses[pi].0 } else { part_uses[pi].1 };
                if uses {
                    let n = if list == 0 { n0 } else { n1 };
                    let r = read_ref_cabac_b(g, cd, mb, sid, bx, by, list, n);
                    refs[pi][list] = r;
                    g.fill_block_l(list, mbx, mby, bx, by, pw, ph, [0, 0], r);
                } else {
                    // FFmpeg's LIST_NOT_USED fill — see the 16x16 branch above.
                    g.fill_block_l(list, mbx, mby, bx, by, pw, ph, [0, 0], INTRA);
                }
            }
        }
        for list in 0..2 {
            for (pi, &(bx, by, pw, ph)) in parts.iter().enumerate() {
                let uses = if list == 0 { part_uses[pi].0 } else { part_uses[pi].1 };
                if uses {
                    let refr = refs[pi][list];
                    let (mx, my) = if shape == 1 {
                        pred_16x8_l(g, list, mbx, mby, pi, refr, sid)
                    } else {
                        pred_8x16_l(g, list, mbx, mby, pi, refr, sid)
                    };
                    let (ax, ay) = g.amvd_l(list, mb, sid, bx, by);
                    let (dx, cx) = decode_mvd_cabac(cd, 40, ax);
                    let (dy, cy) = decode_mvd_cabac(cd, 47, ay);
                    g.fill_block_l(list, mbx, mby, bx, by, pw, ph, [mx + dx, my + dy], refr);
                    g.fill_mvd_l(list, mbx, mby, bx, by, pw, ph, [cx, cy]);
                }
            }
        }
        if shape == 1 { Seg::B16x8 } else { Seg::B8x16 }
    };

    g.seg[mb] = seg;
    g.decoded[mb] = true;
    // 16x16/16x8/8x16 (shape 0/1/2, i.e. not B_8x8) always allow the 8x8
    // transform, mirroring P's unconditional "mb_type < 3" — only explicit
    // B_8x8 gates it on `all_8x8` (handled at its own call site above).
    finish_b_inter(g, cd, mb, sid, pps, true)
}

/// Decode a CABAC B slice into `g`. Mirrors `decode_slice_cabac`'s structure
/// (mb_skip_flag loop + end_of_slice_flag), but with the B-specific skip
/// context (ctx 24-26) and B_Skip resolved via spatial direct mode instead of
/// `pred_pskip`. `col` is RefPicList1[0]'s (the collocated picture) motion/ref
/// snapshot, needed by direct mode.
pub fn decode_slice_cabac_b(
    g: &mut FrameGrids,
    rbsp: &[u8],
    byte_start: usize,
    sh: &SliceHeader,
    pps: &Pps,
    sid: i32,
    col: &ColPic,
) -> SliceResult {
    let mut cd = Cabd {
        c: Cabac::new(rbsp, byte_start),
        st: Box::new([0u8; 1024]),
        last_qscale_nonzero: false,
    };
    init_states(&mut cd.st, false, sh.cabac_init_idc as usize, sh.slice_qp);

    let pmbs = g.mb_w * g.mb_h;
    let mut cur = sh.first_mb_in_slice as usize;
    let mut count = 0usize;
    let mut ok = true;

    loop {
        if cur >= pmbs {
            break;
        }
        let mut skipped = false;
        {
            // mb_skip_flag: same base ctx (11) as P, +13 for B, neighbour
            // "not skip" count (ITU-T Table 9-34 / FFmpeg ctx+=13).
            let (l, t) = g.ctx_neighbors(cur, sid);
            let mut ctx = 13;
            if let Some(m) = l {
                if !g.mb_skip[m] {
                    ctx += 1;
                }
            }
            if let Some(m) = t {
                if !g.mb_skip[m] {
                    ctx += 1;
                }
            }
            if cd.get(11 + ctx) == 1 {
                let seg = decode_b_direct_whole_mb(g, cur, sid, col);
                g.mb_bdirect[cur] = true;
                g.mb_skip[cur] = true;
                g.seg[cur] = seg;
                g.slice_id[cur] = sid;
                g.decoded[cur] = true;
                cd.last_qscale_nonzero = false;
                zero_mb_nnz(g, cur); // B_Skip carries no residual
                skipped = true;
            }
        }
        if !skipped && !decode_mb_cabac_b(g, &mut cd, cur, sid, sh, pps, col) {
            ok = false;
            break;
        }
        cur += 1;
        count += 1;
        if cur >= pmbs {
            break;
        }
        if cd.c.terminate() {
            break;
        }
    }

    SliceResult { mbs_decoded: count, ok }
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
            let m = (x as usize / 2) + (y as usize / self.cvb) * self.mb_w;
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
/// `chroma422_dc` selects the ChromaArrayType==2 chroma-DC ctxIdxInc formula
/// and abs-level-gt1 context range (cat 3 only, 8 coefficients instead of 4).
fn residual_bins(cd: &mut Cabd, cat: usize, max_coeff: usize, is_8x8: bool, chroma422_dc: bool) -> u8 {
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
        } else if chroma422_dc {
            let o = SIG_COEFF_OFFSET_DC_422[last];
            (o, o)
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
            let gt1 = if chroma422_dc { ABS_GT1_CTX_DC422[node] } else { ABS_GT1_CTX[node] };
            let cg = abs_b + gt1;
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
            residual_bins(cd, 0, 16, false, false);
        }
        if cbp & 15 != 0 {
            for idx in 0..16 {
                let (bx, by) = block_xy(idx);
                let gx = mbx * 4 + bx;
                let gy = mby * 4 + by;
                let cbf = g.cbf_ctx_luma(mb, sid, gx, gy, 1, is_intra);
                if cd.get(cbf) == 1 {
                    g.nnz_l[gy * g.bw + gx] = residual_bins(cd, 1, 15, false, false);
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
                residual_bins(cd, 5, 64, true, false)
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
                        g.nnz_l[gy * g.bw + gx] = residual_bins(cd, 2, 16, false, false);
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
    let cvb = g.cvb;
    let chroma422 = cvb == 4;
    let dc_max = if chroma422 { 8 } else { 4 };
    if cbp & 0x30 != 0 {
        for c in 0..2 {
            let cbf = g.cbf_ctx_dc(mb, sid, 3, c, is_intra);
            if cd.get(cbf) == 1 {
                g.cbp_mb[mb] |= 0x40 << c;
                residual_bins(cd, 3, dc_max, false, chroma422);
            }
        }
    }
    if cbp & 0x20 != 0 {
        for c in 0..2 {
            for i4x4 in 0..(2 * cvb) {
                let cx = mbx * 2 + (i4x4 & 1);
                let cy = mby * cvb + (i4x4 >> 1);
                let cbf = g.cbf_ctx_chroma(mb, sid, c, cx, cy, is_intra);
                if cd.get(cbf) == 1 {
                    g.nnz_c[c][cy * g.cw + cx] = residual_bins(cd, 4, 15, false, false);
                } else {
                    g.nnz_c[c][cy * g.cw + cx] = 0;
                }
            }
        }
    } else {
        for c in 0..2 {
            for cy in 0..cvb {
                for cx in 0..2 {
                    g.nnz_c[c][(mby * cvb + cy) * g.cw + (mbx * 2 + cx)] = 0;
                }
            }
        }
    }
}
