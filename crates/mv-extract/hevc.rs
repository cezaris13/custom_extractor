//! From-scratch H.265/HEVC compressed-domain parser — the thesis approach
//! extended to HEVC (the original thesis is H.264-only; this is custom).
//!
//! HEVC is a different codec from H.264 at every layer, so almost none of the
//! H.264 decoder is reusable: a 2-byte NAL header, VPS/SPS/PPS parameter sets,
//! CTU quadtrees (coding units up to 64x64, recursively split) instead of fixed
//! 16x16 macroblocks, CABAC-only entropy with its own context tables, and motion
//! that uses merge / AMVP candidate lists rather than H.264's median predictor.
//!
//! Staged like the H.264 build (see `thesis.rs`):
//!   rung 1 (this file): bitstream foundation — 2-byte NAL split (Annex-B +
//!     hvcC length-prefix), profile_tier_level, SPS, PPS, and slice-segment
//!     header far enough to classify every slice (I/P/B) and find picture
//!     boundaries. UNIT TESTED + verified against real clips by `extractor10`.
//!   rung 2: HEVC-CABAC engine + context init.
//!   rung 3: CTU coding_quadtree -> coding_unit -> prediction_unit parse, with
//!     transform_tree/residual_coding consumed for CABAC sync.
//!   rung 4: MV reconstruction — spatial+temporal merge lists and AMVP.
//!
//! Correct MV output is impossible before rung 3 (CABAC must stay in sync), so
//! this rung reports structure only — exactly as H.264 rung 1 did.

use crate::thesis::{ebsp_to_rbsp, BitReader};

// NAL unit types we care about (ITU-T H.265 Table 7-1). VCL slices are 0..=31;
// the IRAP range 16..=23 carries IDR/BLA/CRA pictures.
pub const NAL_VPS: u8 = 32;
pub const NAL_SPS: u8 = 33;
pub const NAL_PPS: u8 = 34;

#[inline]
pub fn is_vcl(nal_type: u8) -> bool {
    nal_type <= 31
}
#[inline]
pub fn is_irap(nal_type: u8) -> bool {
    (16..=23).contains(&nal_type)
}

/// One HEVC NAL unit, 2-byte header parsed (§7.3.1.2), payload as RBSP.
#[derive(Debug, Clone)]
pub struct Nal {
    pub nal_unit_type: u8,
    pub nuh_layer_id: u8,
    pub nuh_temporal_id_plus1: u8,
    pub rbsp: Vec<u8>,
}

impl Nal {
    fn from_ebsp(bytes: &[u8]) -> Option<Nal> {
        if bytes.len() < 2 || bytes[0] & 0x80 != 0 {
            return None; // need 2-byte header; forbidden_zero_bit must be 0
        }
        let nal_unit_type = (bytes[0] >> 1) & 0x3f;
        let nuh_layer_id = ((bytes[0] & 1) << 5) | (bytes[1] >> 3);
        let nuh_temporal_id_plus1 = bytes[1] & 0x07;
        Some(Nal {
            nal_unit_type,
            nuh_layer_id,
            nuh_temporal_id_plus1,
            rbsp: ebsp_to_rbsp(&bytes[2..]),
        })
    }
}

/// Split an Annex-B byte stream (`00 00 01` / `00 00 00 01` start codes).
pub fn split_annexb(data: &[u8]) -> Vec<Nal> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (idx, &s) in starts.iter().enumerate() {
        let end = if idx + 1 < starts.len() {
            let mut e = starts[idx + 1] - 3;
            while e > s && data[e - 1] == 0 {
                e -= 1;
            }
            e
        } else {
            data.len()
        };
        if end > s {
            if let Some(n) = Nal::from_ebsp(&data[s..end]) {
                nals.push(n);
            }
        }
    }
    nals
}

/// Split a length-prefixed stream (mp4 `hvc1`/`hev1`), `nal_length_size` bytes.
pub fn split_length_prefixed(data: &[u8], nal_length_size: usize) -> Vec<Nal> {
    let mut nals = Vec::new();
    let mut i = 0usize;
    while i + nal_length_size <= data.len() {
        let mut len = 0usize;
        for _ in 0..nal_length_size {
            len = (len << 8) | data[i] as usize;
            i += 1;
        }
        if len == 0 || i + len > data.len() {
            break;
        }
        if let Some(n) = Nal::from_ebsp(&data[i..i + len]) {
            nals.push(n);
        }
        i += len;
    }
    nals
}

/// Parsed `hvcC` (HEVCDecoderConfigurationRecord) — mp4 extradata for HEVC.
pub struct Hvcc {
    pub nal_length_size: usize,
    pub param_sets: Vec<Nal>, // VPS/SPS/PPS NALs carried in the record
}

/// Parse `hvcC`. Returns `None` if the blob is not an hvcC record (Annex-B
/// streams carry raw or empty extradata).
pub fn parse_hvcc(extradata: &[u8]) -> Option<Hvcc> {
    // configurationVersion == 1, then a fixed 22-byte header before numOfArrays.
    if extradata.len() < 23 || extradata[0] != 1 {
        return None;
    }
    let nal_length_size = (extradata[21] & 0x03) as usize + 1;
    let num_arrays = extradata[22] as usize;
    let mut i = 23usize;
    let mut param_sets = Vec::new();
    for _ in 0..num_arrays {
        if i + 3 > extradata.len() {
            break;
        }
        // array_completeness(1)/reserved(1)/NAL_unit_type(6)
        i += 1;
        let num_nalus = ((extradata[i] as usize) << 8) | extradata[i + 1] as usize;
        i += 2;
        for _ in 0..num_nalus {
            if i + 2 > extradata.len() {
                return Some(Hvcc { nal_length_size, param_sets });
            }
            let len = ((extradata[i] as usize) << 8) | extradata[i + 1] as usize;
            i += 2;
            if i + len > extradata.len() {
                return Some(Hvcc { nal_length_size, param_sets });
            }
            if let Some(n) = Nal::from_ebsp(&extradata[i..i + len]) {
                param_sets.push(n);
            }
            i += len;
        }
    }
    Some(Hvcc { nal_length_size, param_sets })
}

// ── profile_tier_level (§7.3.3) ──────────────────────────────────────────────

/// Consume `profile_tier_level(1, max_sub_layers_minus1)`, returning
/// `general_profile_idc`. The general block is a fixed 96 bits; sub-layer blocks
/// are 88 bits (profile) + 8 bits (level) when their present-flags are set.
fn profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> u8 {
    let _space = r.read_bits(2);
    let _tier = r.read_bit();
    let profile_idc = r.read_bits(5) as u8;
    r.skip_bits(32); // general_profile_compatibility_flag[32]
    r.skip_bits(4); // progressive/interlaced/non_packed/frame_only
    r.skip_bits(44); // general constraint/reserved (43) + inbld/reserved (1)
    let _general_level_idc = r.read_bits(8);

    let n = max_sub_layers_minus1.min(8) as usize;
    let mut sub_profile = [false; 8];
    let mut sub_level = [false; 8];
    for i in 0..n {
        sub_profile[i] = r.read_bit() == 1;
        sub_level[i] = r.read_bit() == 1;
    }
    if n > 0 {
        for _ in n..8 {
            r.skip_bits(2); // reserved_zero_2bits
        }
    }
    for i in 0..n {
        if sub_profile[i] {
            r.skip_bits(88);
        }
        if sub_level[i] {
            r.skip_bits(8);
        }
    }
    profile_idc
}

// ── SPS (§7.3.2.2) ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Sps {
    pub sps_id: u32,
    pub profile_idc: u8,
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub log2_max_poc_lsb: u32,
    pub ctb_log2_size_y: u32,
    pub min_cb_log2_size_y: u32,
    pub min_tb_log2_size_y: u32,
    pub max_tb_log2_size_y: u32,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    pub scaling_list_enabled: bool,
    pub amp_enabled: bool,
    pub sao_enabled: bool,
    pub pcm_enabled: bool,
    pub pcm_log2_min_cb: u32,
    pub pcm_log2_max_cb: u32,
    pub num_long_term_ref_pics_sps: u32,
    pub long_term_ref_pics_present: bool,
    pub temporal_mvp_enabled: bool,
    /// Full short-term RPS per SPS index (needed to build reference POC lists).
    pub st_rps: Vec<ShortRps>,
}

/// A resolved short-term reference picture set (ITU-T H.265 §7.4.8): signed POC
/// deltas relative to the current picture, with per-entry used-by-curr flags.
/// Negatives come first (sorted), then positives.
#[derive(Debug, Clone, Default)]
pub struct ShortRps {
    pub delta_poc: Vec<i32>,
    pub used: Vec<bool>,
    pub num_negative: u32,
}

impl ShortRps {
    pub fn num_delta_pocs(&self) -> u32 {
        self.delta_poc.len() as u32
    }
    pub fn num_used(&self) -> u32 {
        self.used.iter().filter(|&&u| u).count() as u32
    }
}

impl Sps {
    pub fn ctb_size_y(&self) -> u32 {
        1 << self.ctb_log2_size_y
    }
    pub fn pic_width_in_ctbs(&self) -> u32 {
        self.pic_width_in_luma_samples.div_ceil(self.ctb_size_y())
    }
    pub fn pic_height_in_ctbs(&self) -> u32 {
        self.pic_height_in_luma_samples.div_ceil(self.ctb_size_y())
    }
    pub fn pic_size_in_ctbs(&self) -> u32 {
        self.pic_width_in_ctbs() * self.pic_height_in_ctbs()
    }
}

/// Parse `seq_parameter_set_rbsp` far enough for slice classification (through
/// the CTB-size fields). Later rungs will extend past here.
pub fn parse_sps(rbsp: &[u8]) -> Sps {
    let mut r = BitReader::new(rbsp);
    let mut s = Sps::default();
    let _vps_id = r.read_bits(4);
    let max_sub_layers_minus1 = r.read_bits(3);
    let _temporal_id_nesting = r.read_bit();
    s.profile_idc = profile_tier_level(&mut r, max_sub_layers_minus1);

    s.sps_id = r.read_ue();
    s.chroma_format_idc = r.read_ue();
    if s.chroma_format_idc == 3 {
        s.separate_colour_plane_flag = r.read_bit() == 1;
    }
    s.pic_width_in_luma_samples = r.read_ue();
    s.pic_height_in_luma_samples = r.read_ue();
    if r.read_bit() == 1 {
        // conformance_window: 4 ue offsets (unused for MV positions at CTB res)
        let _ = r.read_ue();
        let _ = r.read_ue();
        let _ = r.read_ue();
        let _ = r.read_ue();
    }
    let _bit_depth_luma_minus8 = r.read_ue();
    let _bit_depth_chroma_minus8 = r.read_ue();
    s.log2_max_poc_lsb = r.read_ue() + 4;

    let sub_layer_ordering_info_present = r.read_bit() == 1;
    let lo = if sub_layer_ordering_info_present { 0 } else { max_sub_layers_minus1 };
    for _ in lo..=max_sub_layers_minus1 {
        let _max_dec_pic_buffering = r.read_ue();
        let _max_num_reorder_pics = r.read_ue();
        let _max_latency_increase = r.read_ue();
    }

    s.min_cb_log2_size_y = r.read_ue() + 3; // log2_min_luma_coding_block_size_minus3 + 3
    let log2_diff_max_min = r.read_ue();
    s.ctb_log2_size_y = s.min_cb_log2_size_y + log2_diff_max_min;

    s.min_tb_log2_size_y = r.read_ue() + 2;
    s.max_tb_log2_size_y = s.min_tb_log2_size_y + r.read_ue();
    s.max_transform_hierarchy_depth_inter = r.read_ue();
    s.max_transform_hierarchy_depth_intra = r.read_ue();
    s.scaling_list_enabled = r.read_bit() == 1;
    if s.scaling_list_enabled && r.read_bit() == 1 {
        skip_scaling_list_data(&mut r);
    }
    s.amp_enabled = r.read_bit() == 1;
    s.sao_enabled = r.read_bit() == 1;
    s.pcm_enabled = r.read_bit() == 1;
    if s.pcm_enabled {
        let _pcm_bd_luma = r.read_bits(4);
        let _pcm_bd_chroma = r.read_bits(4);
        s.pcm_log2_min_cb = r.read_ue() + 3;
        s.pcm_log2_max_cb = s.pcm_log2_min_cb + r.read_ue();
        let _pcm_loop_filter_disabled = r.read_bit();
    }
    let num_st_rps = r.read_ue();
    let mut sets: Vec<ShortRps> = Vec::with_capacity(num_st_rps as usize);
    for i in 0..num_st_rps {
        let rps = parse_short_term_rps(&mut r, i as usize, &sets, false, num_st_rps);
        sets.push(rps);
    }
    s.st_rps = sets;
    s.long_term_ref_pics_present = r.read_bit() == 1;
    if s.long_term_ref_pics_present {
        s.num_long_term_ref_pics_sps = r.read_ue();
        for _ in 0..s.num_long_term_ref_pics_sps {
            let _lt_ref_poc_lsb = r.read_bits(s.log2_max_poc_lsb);
            let _used_by_curr = r.read_bit();
        }
    }
    s.temporal_mvp_enabled = r.read_bit() == 1;
    let _strong_intra_smoothing = r.read_bit();
    // VUI + SPS extensions follow but are unused for MV extraction.
    s
}

/// `scaling_list_data()` (ITU-T H.265 §7.3.4) — consumed only to advance.
fn skip_scaling_list_data(r: &mut BitReader) {
    for size_id in 0..4 {
        let step = if size_id == 3 { 3 } else { 1 };
        let mut matrix_id = 0;
        while matrix_id < 6 {
            if r.read_bit() == 0 {
                let _pred_matrix_id_delta = r.read_ue();
            } else {
                let coef_num = 64.min(1 << (4 + (size_id << 1)));
                if size_id > 1 {
                    let _dc_coef_minus8 = r.read_se();
                }
                for _ in 0..coef_num {
                    let _delta_coef = r.read_se();
                }
            }
            matrix_id += step;
        }
    }
}

/// `st_ref_pic_set()` (ITU-T H.265 §7.3.7), fully resolved into signed POC
/// deltas + used flags (negatives first). `sps_sets` holds already-parsed SPS
/// sets (for the inter-RPS-prediction path); `idx` is this set's index.
fn parse_short_term_rps(r: &mut BitReader, idx: usize, sps_sets: &[ShortRps], is_slice: bool, sps_nb_st_rps: u32) -> ShortRps {
    let mut rps_predict = false;
    if (is_slice || idx != 0) && sps_nb_st_rps > 0 {
        rps_predict = r.read_bit() == 1;
    }
    if rps_predict {
        let refr = if is_slice {
            let delta_idx = r.read_ue() + 1;
            &sps_sets[(sps_nb_st_rps - delta_idx) as usize]
        } else {
            &sps_sets[idx - 1]
        };
        let ref_ndp = refr.num_delta_pocs() as i32;
        let delta_rps_sign = r.read_bit() as i32;
        let abs_delta_rps = r.read_ue() as i32 + 1;
        let delta_rps = (1 - 2 * delta_rps_sign) * abs_delta_rps;
        // Per §7.4.8 inter-RPS prediction with the FFmpeg sort.
        let mut delta = Vec::new();
        let mut used = Vec::new();
        let mut num_neg = 0u32;
        for i in 0..=ref_ndp as usize {
            let u = r.read_bit() == 1;
            let use_delta = if !u { r.read_bit() == 1 } else { false };
            if u || use_delta {
                let dp = if (i as i32) < ref_ndp {
                    delta_rps + refr.delta_poc[i]
                } else {
                    delta_rps
                };
                delta.push(dp);
                used.push(u);
                if dp < 0 {
                    num_neg += 1;
                }
            }
        }
        // sort increasing, keeping used[] aligned (insertion sort like FFmpeg)
        let n = delta.len();
        for i in 1..n {
            let (dp, uu) = (delta[i], used[i]);
            let mut k = i as i32 - 1;
            while k >= 0 && dp < delta[k as usize] {
                delta[k as usize + 1] = delta[k as usize];
                used[k as usize + 1] = used[k as usize];
                k -= 1;
            }
            delta[(k + 1) as usize] = dp;
            used[(k + 1) as usize] = uu;
        }
        // flip negatives to most-negative-... FFmpeg flips to largest-first then
        // overall increasing leaves negatives ascending; StCurrBefore wants
        // closest first, handled at ref-list build time.
        ShortRps { delta_poc: delta, used, num_negative: num_neg }
    } else {
        let num_neg = r.read_ue();
        let num_pos = r.read_ue();
        let mut delta = Vec::with_capacity((num_neg + num_pos) as usize);
        let mut used = Vec::with_capacity((num_neg + num_pos) as usize);
        let mut prev = 0i32;
        for _ in 0..num_neg {
            prev -= r.read_ue() as i32 + 1;
            delta.push(prev);
            used.push(r.read_bit() == 1);
        }
        prev = 0;
        for _ in 0..num_pos {
            prev += r.read_ue() as i32 + 1;
            delta.push(prev);
            used.push(r.read_bit() == 1);
        }
        ShortRps { delta_poc: delta, used, num_negative: num_neg }
    }
}

// ── PPS (§7.3.2.3) ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active: u32,
    pub num_ref_idx_l1_default_active: u32,
    pub pic_init_qp: i32,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub pic_slice_chroma_qp_offsets_present: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub num_tile_columns: u32,
    pub num_tile_rows: u32,
    pub loop_filter_across_slices_enabled: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub deblocking_filter_disabled_flag: bool,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level: u32,
    pub slice_header_extension_present_flag: bool,
}

/// Parse `pic_parameter_set_rbsp` (ITU-T H.265 §7.3.2.3) — fields needed for the
/// slice header and CTU decode. Range-extension / SCC PPS fields default to off
/// (Main/Main10 streams don't carry them before this point).
pub fn parse_pps(rbsp: &[u8]) -> Pps {
    let mut r = BitReader::new(rbsp);
    let mut p = Pps::default();
    p.pps_id = r.read_ue();
    p.sps_id = r.read_ue();
    p.dependent_slice_segments_enabled_flag = r.read_bit() == 1;
    p.output_flag_present_flag = r.read_bit() == 1;
    p.num_extra_slice_header_bits = r.read_bits(3);
    p.sign_data_hiding_enabled = r.read_bit() == 1;
    p.cabac_init_present_flag = r.read_bit() == 1;
    p.num_ref_idx_l0_default_active = r.read_ue() + 1;
    p.num_ref_idx_l1_default_active = r.read_ue() + 1;
    p.pic_init_qp = 26 + r.read_se();
    let _constrained_intra_pred = r.read_bit();
    p.transform_skip_enabled_flag = r.read_bit() == 1;
    p.cu_qp_delta_enabled_flag = r.read_bit() == 1;
    if p.cu_qp_delta_enabled_flag {
        p.diff_cu_qp_delta_depth = r.read_ue();
    }
    p.cb_qp_offset = r.read_se();
    p.cr_qp_offset = r.read_se();
    p.pic_slice_chroma_qp_offsets_present = r.read_bit() == 1;
    p.weighted_pred_flag = r.read_bit() == 1;
    p.weighted_bipred_flag = r.read_bit() == 1;
    p.transquant_bypass_enabled_flag = r.read_bit() == 1;
    p.tiles_enabled_flag = r.read_bit() == 1;
    p.entropy_coding_sync_enabled_flag = r.read_bit() == 1;
    p.num_tile_columns = 1;
    p.num_tile_rows = 1;
    if p.tiles_enabled_flag {
        p.num_tile_columns = r.read_ue() + 1;
        p.num_tile_rows = r.read_ue() + 1;
        let uniform_spacing = r.read_bit() == 1;
        if !uniform_spacing {
            for _ in 0..p.num_tile_columns - 1 {
                let _column_width = r.read_ue();
            }
            for _ in 0..p.num_tile_rows - 1 {
                let _row_height = r.read_ue();
            }
        }
        let _loop_filter_across_tiles = r.read_bit();
    }
    p.loop_filter_across_slices_enabled = r.read_bit() == 1;
    p.deblocking_filter_control_present_flag = r.read_bit() == 1;
    if p.deblocking_filter_control_present_flag {
        p.deblocking_filter_override_enabled_flag = r.read_bit() == 1;
        p.deblocking_filter_disabled_flag = r.read_bit() == 1;
        if !p.deblocking_filter_disabled_flag {
            let _beta_offset = r.read_se();
            let _tc_offset = r.read_se();
        }
    }
    if r.read_bit() == 1 {
        // pps_scaling_list_data_present_flag
        skip_scaling_list_data(&mut r);
    }
    p.lists_modification_present_flag = r.read_bit() == 1;
    p.log2_parallel_merge_level = r.read_ue() + 2;
    p.slice_header_extension_present_flag = r.read_bit() == 1;
    // pps_extension flags follow but are unused.
    p
}

// ── Slice segment header (§7.3.6.1), classification subset ───────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    B,
    P,
    I,
}

impl SliceType {
    fn from_ue(v: u32) -> SliceType {
        match v {
            0 => SliceType::B,
            1 => SliceType::P,
            _ => SliceType::I,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SliceHeader {
    pub first_slice_segment_in_pic_flag: bool,
    pub dependent_slice_segment_flag: bool,
    pub pps_id: u32,
    pub slice_segment_address: u32,
    /// `None` for dependent slice segments (they inherit the type).
    pub slice_type: Option<SliceType>,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub nb_refs: [u32; 2],
    pub slice_temporal_mvp_enabled: bool,
    pub collocated_list: usize,
    pub collocated_ref_idx: u32,
    pub mvd_l1_zero_flag: bool,
    pub cabac_init_flag: bool,
    pub max_num_merge_cand: u32,
    pub slice_qp: i32,
    /// Byte offset of `slice_segment_data()` (after cabac byte alignment).
    pub data_offset: usize,
    /// True if the header used a feature this decoder can't yet keep in sync
    /// (tiles, WPP) — caller should not decode MVs.
    pub unsupported: bool,
    // POC / reference fields (for MV reconstruction).
    pub nal_unit_type: u8,
    pub is_idr: bool,
    pub poc_lsb: u32,
    /// The slice's resolved short-term RPS (used to build reference POC lists).
    pub short_rps: ShortRps,
}

/// Parse the full `slice_segment_header()` (ITU-T H.265 §7.3.6.1) through to the
/// byte-aligned start of slice data. Needs the referenced SPS/PPS.
pub fn parse_slice_header(
    nal: &Nal,
    sps_for: &impl Fn(u32) -> Option<Sps>,
    pps_for: &impl Fn(u32) -> Option<Pps>,
) -> Option<SliceHeader> {
    let mut r = BitReader::new(&nal.rbsp);
    let mut sh = SliceHeader::default();
    sh.first_slice_segment_in_pic_flag = r.read_bit() == 1;
    if is_irap(nal.nal_unit_type) {
        let _no_output_of_prior_pics_flag = r.read_bit();
    }
    sh.pps_id = r.read_ue();
    let pps = pps_for(sh.pps_id)?;
    let sps = sps_for(pps.sps_id)?;

    if !sh.first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            sh.dependent_slice_segment_flag = r.read_bit() == 1;
        }
        sh.slice_segment_address = r.read_bits(ceil_log2(sps.pic_size_in_ctbs()));
    }

    let is_idr = matches!(nal.nal_unit_type, 19 | 20);
    sh.is_idr = is_idr;
    sh.nal_unit_type = nal.nal_unit_type;
    sh.max_num_merge_cand = 5;
    sh.slice_qp = pps.pic_init_qp;
    let mut num_pic_total_curr: u32 = 0;

    if !sh.dependent_slice_segment_flag {
        for _ in 0..pps.num_extra_slice_header_bits {
            let _ = r.read_bit();
        }
        let st = SliceType::from_ue(r.read_ue());
        sh.slice_type = Some(st);
        if pps.output_flag_present_flag {
            let _pic_output_flag = r.read_bit();
        }
        if sps.separate_colour_plane_flag {
            let _colour_plane_id = r.read_bits(2);
        }
        if !is_idr {
            sh.poc_lsb = r.read_bits(sps.log2_max_poc_lsb);
            let short_term_sps_flag = r.read_bit() == 1;
            let nb = sps.st_rps.len() as u32;
            if !short_term_sps_flag {
                sh.short_rps = parse_short_term_rps(&mut r, nb as usize, &sps.st_rps, true, nb);
            } else {
                let numbits = ceil_log2(nb);
                let rps_idx = if numbits > 0 { r.read_bits(numbits) } else { 0 };
                sh.short_rps = sps.st_rps.get(rps_idx as usize).cloned().unwrap_or_default();
            }
            num_pic_total_curr += sh.short_rps.num_used();
            if sps.long_term_ref_pics_present {
                let mut num_lt_sps = 0;
                if sps.num_long_term_ref_pics_sps > 0 {
                    num_lt_sps = r.read_ue();
                }
                let num_lt_pics = r.read_ue();
                for i in 0..num_lt_sps + num_lt_pics {
                    if i < num_lt_sps {
                        if sps.num_long_term_ref_pics_sps > 1 {
                            let _lt_idx_sps = r.read_bits(ceil_log2(sps.num_long_term_ref_pics_sps));
                        }
                        // used_by_curr for SPS lt entries isn't re-signalled here;
                        // conservatively count it (common encoders set it).
                        num_pic_total_curr += 1;
                    } else {
                        let _poc_lsb_lt = r.read_bits(sps.log2_max_poc_lsb);
                        if r.read_bit() == 1 {
                            num_pic_total_curr += 1; // used_by_curr_pic_lt_flag
                        }
                    }
                    if r.read_bit() == 1 {
                        // delta_poc_msb_present_flag
                        let _delta_poc_msb_cycle_lt = r.read_ue();
                    }
                }
            }
            if sps.temporal_mvp_enabled {
                sh.slice_temporal_mvp_enabled = r.read_bit() == 1;
            }
        }

        if sps.sao_enabled {
            sh.sao_luma = r.read_bit() == 1;
            if sps.chroma_format_idc != 0 {
                sh.sao_chroma = r.read_bit() == 1;
            }
        }

        if matches!(st, SliceType::P | SliceType::B) {
            sh.nb_refs[0] = pps.num_ref_idx_l0_default_active;
            if st == SliceType::B {
                sh.nb_refs[1] = pps.num_ref_idx_l1_default_active;
            }
            if r.read_bit() == 1 {
                // num_ref_idx_active_override_flag
                sh.nb_refs[0] = r.read_ue() + 1;
                if st == SliceType::B {
                    sh.nb_refs[1] = r.read_ue() + 1;
                }
            }
            if pps.lists_modification_present_flag && num_pic_total_curr > 1 {
                // ref_pic_list_modification (§7.3.6.2)
                let nbits = ceil_log2(num_pic_total_curr);
                if r.read_bit() == 1 {
                    for _ in 0..sh.nb_refs[0] {
                        let _list_entry_l0 = r.read_bits(nbits);
                    }
                }
                if st == SliceType::B && r.read_bit() == 1 {
                    for _ in 0..sh.nb_refs[1] {
                        let _list_entry_l1 = r.read_bits(nbits);
                    }
                }
            }
            if st == SliceType::B {
                sh.mvd_l1_zero_flag = r.read_bit() == 1;
            }
            if pps.cabac_init_present_flag {
                sh.cabac_init_flag = r.read_bit() == 1;
            }
            if sh.slice_temporal_mvp_enabled {
                sh.collocated_list = 0;
                if st == SliceType::B && r.read_bit() == 0 {
                    sh.collocated_list = 1; // collocated_from_l0_flag == 0 -> L1
                }
                if sh.nb_refs[sh.collocated_list] > 1 {
                    sh.collocated_ref_idx = r.read_ue();
                }
            }
            if (pps.weighted_pred_flag && st == SliceType::P)
                || (pps.weighted_bipred_flag && st == SliceType::B)
            {
                parse_pred_weight_table(&mut r, sps.chroma_format_idc != 0, &sh.nb_refs, st == SliceType::B);
            }
            sh.max_num_merge_cand = 5 - r.read_ue();
        }

        sh.slice_qp = pps.pic_init_qp + r.read_se();
        if pps.pic_slice_chroma_qp_offsets_present {
            let _cb = r.read_se();
            let _cr = r.read_se();
        }
        let mut disable_dbf = pps.deblocking_filter_disabled_flag;
        if pps.deblocking_filter_control_present_flag {
            let mut override_flag = false;
            if pps.deblocking_filter_override_enabled_flag {
                override_flag = r.read_bit() == 1;
            }
            if override_flag {
                disable_dbf = r.read_bit() == 1;
                if !disable_dbf {
                    let _beta = r.read_se();
                    let _tc = r.read_se();
                }
            }
        }
        if pps.loop_filter_across_slices_enabled && (sh.sao_luma || sh.sao_chroma || !disable_dbf) {
            let _slice_loop_filter_across = r.read_bit();
        }
    }

    if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
        let num_entry = r.read_ue();
        if num_entry > 0 {
            let offset_len = r.read_ue() + 1;
            for _ in 0..num_entry {
                let _offset = r.read_bits(offset_len);
            }
        }
        // Multi-substream CABAC (tiles/WPP) needs per-entry-point reinit; not
        // handled by the single-stream CTU decoder yet.
        if pps.num_tile_columns > 1 || pps.num_tile_rows > 1 || pps.entropy_coding_sync_enabled_flag {
            sh.unsupported = true;
        }
    }
    if pps.slice_header_extension_present_flag {
        let len = r.read_ue();
        for _ in 0..len {
            let _b = r.read_bits(8);
        }
    }

    let _alignment_bit_one = r.read_bit();
    sh.data_offset = r.pos().div_ceil(8);
    Some(sh)
}

/// `pred_weight_table()` (ITU-T H.265 §7.3.6.3) — consumed only to advance.
fn parse_pred_weight_table(r: &mut BitReader, has_chroma: bool, nb_refs: &[u32; 2], is_b: bool) {
    let _luma_log2_weight_denom = r.read_ue();
    if has_chroma {
        let _delta_chroma_log2_weight_denom = r.read_se();
    }
    let lists = if is_b { 2 } else { 1 };
    for l in 0..lists {
        let n = nb_refs[l] as usize;
        let mut luma_flag = [false; 16];
        let mut chroma_flag = [false; 16];
        for i in 0..n {
            luma_flag[i] = r.read_bit() == 1;
        }
        if has_chroma {
            for i in 0..n {
                chroma_flag[i] = r.read_bit() == 1;
            }
        }
        for i in 0..n {
            if luma_flag[i] {
                let _delta_luma_weight = r.read_se();
                let _luma_offset = r.read_se();
            }
            if chroma_flag[i] {
                for _ in 0..2 {
                    let _delta_chroma_weight = r.read_se();
                    let _delta_chroma_offset = r.read_se();
                }
            }
        }
    }
}

/// `Ceil(Log2(n))` (ITU-T usage): smallest `k` with `2^k >= n`. 0 for n<=1.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_header_2_byte() {
        // nal_unit_type = 33 (SPS): byte0 = 0<<7 | 33<<1 | layer_msb=0 = 0x42.
        let n = Nal::from_ebsp(&[0x42, 0x01, 0xAA]).unwrap();
        assert_eq!(n.nal_unit_type, NAL_SPS);
        assert_eq!(n.nuh_temporal_id_plus1, 1);
        assert_eq!(n.rbsp, vec![0xAA]);
    }

    #[test]
    fn annexb_finds_hevc_nals() {
        // start(4) VPS(0x40,..) start(3) SPS(0x42,..)
        let s = [0, 0, 0, 1, 0x40, 0x01, 0x0c, 0, 0, 1, 0x42, 0x01, 0x01];
        let nals = split_annexb(&s);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].nal_unit_type, NAL_VPS);
        assert_eq!(nals[1].nal_unit_type, NAL_SPS);
    }

    #[test]
    fn ceil_log2_known() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(8160), 13); // 1920x1080 @ 16x16 CTB-ish
    }

    #[test]
    fn slice_type_mapping() {
        assert_eq!(SliceType::from_ue(0), SliceType::B);
        assert_eq!(SliceType::from_ue(1), SliceType::P);
        assert_eq!(SliceType::from_ue(2), SliceType::I);
    }
}
