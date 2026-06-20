//! From-scratch H.264 compressed-domain motion-vector extractor.
//!
//! This is the thesis approach (Angheluta, PoliTo 2020/21, "Efficient Extraction
//! of Motion Vectors from H264 Video Streams", Chapter 4): parse the H.264
//! bitstream directly — NAL splitting, parameter sets, slice headers, entropy
//! decoding — and recover motion vectors *without* invoking libavcodec's decode
//! path. Unlike the `motion_vectors_only` extractors (0/1/2/3/5), nothing here
//! depends on FFmpeg beyond the container demux done by the `extractor4` binary.
//!
//! Build status — this is staged work (see the thesis's own intermediate vs
//! final split):
//!   rung 1 (this file): bitstream reader, Exp-Golomb, NAL split (Annex-B +
//!                       AVCC), SPS/PPS/slice-header parsing. UNIT TESTED.
//!   rung 2: CAVLC residual parse-for-sync + P/I macroblock layer.
//!   rung 3: CABAC engine + residual parse-for-sync.
//!   rung 4: MV prediction (median / skip / direct), DPB/MMCO/RPLR.
//!
//! Correct MV output is impossible before rung 2: the bitstream only stays in
//! sync if every macroblock's residual bits are consumed, so there is no
//! "partial but correct" output. Until then `extractor4` parses structure and
//! reports it; it does not emit MVs.
// ponytail: container demux is delegated to libavformat (extractor4.rs). We only
// hand-roll the compressed-domain extraction that is the thesis's actual point;
// reimplementing an mp4/annexb demuxer would be reinventing the stdlib.

// CAVLC residual decoding lives in a sibling file to keep the big VLC tables
// out of this parser. Declared with #[path] so `thesis` stays a single-file
// module (lib.rs: `pub mod thesis;`).
#[path = "thesis_cavlc.rs"]
pub mod cavlc;
#[path = "thesis_slice.rs"]
pub mod slice;
#[path = "thesis_cabac_tables.rs"]
pub mod thesis_cabac_tables;
#[path = "thesis_cabac.rs"]
pub mod cabac;

/// MSB-first bit reader over an RBSP buffer, with the Exp-Golomb helpers the
/// H.264 syntax is written in (ITU-T H.264 §9.1).
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    bit_pos: usize,
    /// Cached `rbsp_stop_one_bit` position (usize::MAX = not yet computed).
    rbsp_stop: std::cell::Cell<usize>,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0, rbsp_stop: std::cell::Cell::new(usize::MAX) }
    }

    #[inline]
    pub fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.bit_pos)
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.bit_pos
    }

    #[inline]
    pub fn byte_aligned(&self) -> bool {
        self.bit_pos % 8 == 0
    }

    /// Read a single bit. Returns 0 past end-of-buffer (matches how H.264
    /// readers behave at the RBSP trailing boundary).
    #[inline]
    pub fn read_bit(&mut self) -> u32 {
        let byte = self.bit_pos / 8;
        if byte >= self.data.len() {
            self.bit_pos += 1;
            return 0;
        }
        let shift = 7 - (self.bit_pos % 8);
        let bit = (self.data[byte] >> shift) & 1;
        self.bit_pos += 1;
        bit as u32
    }

    /// Read `n` bits (0..=32) MSB-first as an unsigned value: `u(n)`.
    pub fn read_bits(&mut self, n: u32) -> u32 {
        let n = n.min(32); // guard: desync can request absurd widths
        let mut v: u32 = 0;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    pub fn skip_bits(&mut self, n: usize) {
        self.bit_pos += n;
    }

    /// Read `n` bits (0..=24) MSB-first WITHOUT advancing — for VLC table lookup.
    /// Bits past end-of-buffer read as 0.
    pub fn peek_bits(&self, n: u32) -> u32 {
        let mut v = 0u32;
        let mut pos = self.bit_pos;
        for _ in 0..n {
            let byte = pos / 8;
            let bit = if byte < self.data.len() {
                (self.data[byte] >> (7 - (pos % 8))) & 1
            } else {
                0
            };
            v = (v << 1) | bit as u32;
            pos += 1;
        }
        v
    }

    /// Unsigned Exp-Golomb `ue(v)` (ITU-T H.264 §9.1).
    pub fn read_ue(&mut self) -> u32 {
        let mut leading_zeros: u32 = 0;
        // Count leading zero bits until the first 1; bail out if we run off the
        // end to avoid an infinite loop on corrupt input.
        while self.read_bit() == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 || self.bit_pos > self.data.len() * 8 {
                return 0;
            }
        }
        if leading_zeros == 0 {
            return 0;
        }
        let suffix = self.read_bits(leading_zeros);
        (1u32 << leading_zeros) - 1 + suffix
    }

    /// `more_rbsp_data()` (ITU-T H.264 §7.2): true if there are payload bits
    /// before the `rbsp_stop_one_bit`.
    pub fn more_rbsp_data(&self) -> bool {
        self.bit_pos < self.rbsp_stop_bit()
    }

    /// Bit position of the `rbsp_stop_one_bit` (the last set bit in the buffer),
    /// or 0 if none. The result is fixed for a given buffer, so it is computed
    /// once and cached — the CAVLC slice loop calls `more_rbsp_data()` per
    /// macroblock, which would otherwise be an O(n) rescan each time (O(n²) per
    /// slice). `usize::MAX` in the cell means "not computed yet".
    pub fn rbsp_stop_bit(&self) -> usize {
        if self.rbsp_stop.get() == usize::MAX {
            let total = self.data.len() * 8;
            let mut stop = 0;
            for i in (0..total).rev() {
                if (self.data[i / 8] >> (7 - (i % 8))) & 1 == 1 {
                    stop = i;
                    break;
                }
            }
            self.rbsp_stop.set(stop);
        }
        self.rbsp_stop.get()
    }

    /// Signed Exp-Golomb `se(v)` (ITU-T H.264 §9.1.1).
    pub fn read_se(&mut self) -> i32 {
        let code = self.read_ue();
        let k = ((code + 1) >> 1) as i32;
        if code & 1 == 1 {
            k
        } else {
            -k
        }
    }
}

/// Strip emulation-prevention bytes: `00 00 03 xx` -> `00 00 xx`
/// (ITU-T H.264 §7.4.1, EBSP -> RBSP).
pub fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0usize;
    let mut i = 0usize;
    while i < ebsp.len() {
        let b = ebsp[i];
        if zeros >= 2 && b == 0x03 && i + 1 < ebsp.len() && ebsp[i + 1] <= 0x03 {
            // drop the 0x03, reset the run
            zeros = 0;
            i += 1;
            continue;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        i += 1;
    }
    out
}

/// One NAL unit, header parsed (ITU-T H.264 §7.3.1) and payload already
/// converted to RBSP.
#[derive(Debug, Clone)]
pub struct Nal {
    pub nal_ref_idc: u8,
    pub nal_unit_type: u8,
    pub rbsp: Vec<u8>,
}

impl Nal {
    /// Build from a raw NAL (header byte + EBSP payload).
    fn from_ebsp(bytes: &[u8]) -> Option<Nal> {
        if bytes.is_empty() {
            return None;
        }
        let hdr = bytes[0];
        // forbidden_zero_bit must be 0; tolerate but skip if set.
        if hdr & 0x80 != 0 {
            return None;
        }
        Some(Nal {
            nal_ref_idc: (hdr >> 5) & 0x03,
            nal_unit_type: hdr & 0x1f,
            rbsp: ebsp_to_rbsp(&bytes[1..]),
        })
    }
}

/// Split an Annex-B byte stream (NALs separated by `00 00 01` / `00 00 00 01`
/// start codes) into NAL units. (ITU-T H.264 Annex B.)
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
            // back up over the next start code (and its leading zero bytes)
            let mut e = starts[idx + 1] - 3;
            while e > s && data[e - 1] == 0 {
                e -= 1;
            }
            e
        } else {
            data.len()
        };
        if e_valid(s, end) {
            if let Some(n) = Nal::from_ebsp(&data[s..end]) {
                nals.push(n);
            }
        }
    }
    nals
}

#[inline]
fn e_valid(s: usize, end: usize) -> bool {
    end > s
}

/// Split a length-prefixed AVCC bitstream. `nal_length_size` is 1..=4 bytes,
/// taken from the avcC `lengthSizeMinusOne` field.
pub fn split_avcc(data: &[u8], nal_length_size: usize) -> Vec<Nal> {
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

/// Parsed avcC (`AVCDecoderConfigurationRecord`) — the AVCC extradata that mp4
/// stores instead of in-band parameter sets.
pub struct Avcc {
    pub nal_length_size: usize,
    pub sps: Vec<Nal>,
    pub pps: Vec<Nal>,
}

/// Parse avcC extradata. Returns `None` if the blob is not avcC (e.g. Annex-B
/// streams carry empty or raw extradata).
pub fn parse_avcc(extradata: &[u8]) -> Option<Avcc> {
    // avcC starts with configurationVersion == 1.
    if extradata.len() < 7 || extradata[0] != 1 {
        return None;
    }
    let nal_length_size = (extradata[4] & 0x03) as usize + 1;
    let mut i = 5usize;
    let mut sps = Vec::new();
    let mut pps = Vec::new();

    let num_sps = (extradata[i] & 0x1f) as usize;
    i += 1;
    for _ in 0..num_sps {
        if i + 2 > extradata.len() {
            return None;
        }
        let len = ((extradata[i] as usize) << 8) | extradata[i + 1] as usize;
        i += 2;
        if i + len > extradata.len() {
            return None;
        }
        if let Some(n) = Nal::from_ebsp(&extradata[i..i + len]) {
            sps.push(n);
        }
        i += len;
    }
    if i >= extradata.len() {
        return Some(Avcc { nal_length_size, sps, pps });
    }
    let num_pps = extradata[i] as usize;
    i += 1;
    for _ in 0..num_pps {
        if i + 2 > extradata.len() {
            break;
        }
        let len = ((extradata[i] as usize) << 8) | extradata[i + 1] as usize;
        i += 2;
        if i + len > extradata.len() {
            break;
        }
        if let Some(n) = Nal::from_ebsp(&extradata[i..i + len]) {
            pps.push(n);
        }
        i += len;
    }
    Some(Avcc { nal_length_size, sps, pps })
}

// ── Parameter sets ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Sps {
    pub sps_id: u32,
    pub profile_idc: u8,
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub log2_max_frame_num: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_pic_order_cnt_lsb: u32,
    pub delta_pic_order_always_zero_flag: bool,
    pub max_num_ref_frames: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
}

impl Sps {
    /// Frame height in macroblocks (accounts for field/frame coding).
    pub fn frame_height_in_mbs(&self) -> u32 {
        (2 - self.frame_mbs_only_flag as u32) * self.pic_height_in_map_units
    }
}

fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

/// Consume a scaling list (ITU-T H.264 §7.3.2.1.1.1) without storing it — we
/// only need to keep the bit position correct.
fn skip_scaling_list(r: &mut BitReader, size: usize) {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            let delta = r.read_se();
            next_scale = (last_scale + delta + 256) % 256;
        }
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
}

/// Parse `seq_parameter_set_rbsp` (ITU-T H.264 §7.3.2.1.1) — only the fields the
/// extractor needs.
pub fn parse_sps(rbsp: &[u8]) -> Sps {
    let mut r = BitReader::new(rbsp);
    let mut s = Sps::default();
    s.profile_idc = r.read_bits(8) as u8;
    let _constraint_flags = r.read_bits(8);
    let _level_idc = r.read_bits(8);
    s.sps_id = r.read_ue();
    s.chroma_format_idc = 1; // default 4:2:0 for baseline/main/extended

    if is_high_profile(s.profile_idc) {
        s.chroma_format_idc = r.read_ue();
        if s.chroma_format_idc == 3 {
            s.separate_colour_plane_flag = r.read_bit() == 1;
        }
        let _bit_depth_luma = r.read_ue();
        let _bit_depth_chroma = r.read_ue();
        let _qpprime_y_zero_transform_bypass = r.read_bit();
        let scaling_matrix_present = r.read_bit() == 1;
        if scaling_matrix_present {
            let count = if s.chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                let present = r.read_bit() == 1;
                if present {
                    if i < 6 {
                        skip_scaling_list(&mut r, 16);
                    } else {
                        skip_scaling_list(&mut r, 64);
                    }
                }
            }
        }
    }

    s.log2_max_frame_num = r.read_ue() + 4;
    s.pic_order_cnt_type = r.read_ue();
    if s.pic_order_cnt_type == 0 {
        s.log2_max_pic_order_cnt_lsb = r.read_ue() + 4;
    } else if s.pic_order_cnt_type == 1 {
        s.delta_pic_order_always_zero_flag = r.read_bit() == 1;
        let _offset_for_non_ref_pic = r.read_se();
        let _offset_for_top_to_bottom_field = r.read_se();
        let num_ref_frames_in_cycle = r.read_ue();
        for _ in 0..num_ref_frames_in_cycle {
            let _ = r.read_se();
        }
    }
    s.max_num_ref_frames = r.read_ue();
    let _gaps_in_frame_num_allowed = r.read_bit();
    s.pic_width_in_mbs = r.read_ue() + 1;
    s.pic_height_in_map_units = r.read_ue() + 1;
    s.frame_mbs_only_flag = r.read_bit() == 1;
    if !s.frame_mbs_only_flag {
        s.mb_adaptive_frame_field_flag = r.read_bit() == 1;
    }
    // Remaining fields (direct_8x8, cropping, VUI) are unused for MV extraction.
    s
}

#[derive(Debug, Clone, Default)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups: u32,
    pub num_ref_idx_l0_default_active: u32,
    pub num_ref_idx_l1_default_active: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
}

/// Parse `pic_parameter_set_rbsp` (ITU-T H.264 §7.3.2.2) — fields needed for
/// slice parsing and entropy mode selection.
pub fn parse_pps(rbsp: &[u8]) -> Pps {
    let mut r = BitReader::new(rbsp);
    let mut p = Pps::default();
    p.pps_id = r.read_ue();
    p.sps_id = r.read_ue();
    p.entropy_coding_mode_flag = r.read_bit() == 1;
    p.bottom_field_pic_order_in_frame_present_flag = r.read_bit() == 1;
    p.num_slice_groups = r.read_ue() + 1;
    // Slice-group map syntax (num_slice_groups > 1) is rare; not handled here.
    p.num_ref_idx_l0_default_active = r.read_ue() + 1;
    p.num_ref_idx_l1_default_active = r.read_ue() + 1;
    p.weighted_pred_flag = r.read_bit() == 1;
    p.weighted_bipred_idc = r.read_bits(2);
    p.pic_init_qp = r.read_se() + 26;
    let _pic_init_qs = r.read_se();
    let _chroma_qp_index_offset = r.read_se();
    p.deblocking_filter_control_present_flag = r.read_bit() == 1;
    let _constrained_intra_pred_flag = r.read_bit();
    p.redundant_pic_cnt_present_flag = r.read_bit() == 1;
    // PPS extension (only present if there is more RBSP data). We need
    // transform_8x8_mode_flag for the macroblock layer's dct8x8 decision.
    if r.more_rbsp_data() {
        p.transform_8x8_mode_flag = r.read_bit() == 1;
        // pic_scaling_matrix / second_chroma_qp_index_offset follow but are
        // unused for MV extraction, so we stop reading here.
    }
    p
}

// ── Slice header ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    fn from_ue(v: u32) -> SliceType {
        match v % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            _ => SliceType::Si,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub is_idr: bool,
    /// Active reference counts (after any override), L0/L1.
    pub num_ref_idx_l0_active: u32,
    pub num_ref_idx_l1_active: u32,
    pub direct_spatial_mv_pred_flag: bool,
    pub cabac_init_idc: u32,
    /// SliceQP_Y = pic_init_qp + slice_qp_delta (ITU-T eq. 7-30).
    pub slice_qp: i32,
    pub disable_deblocking_filter_idc: u32,
}

/// Parse the leading, fixed portion of `slice_header()` (ITU-T H.264 §7.3.3):
/// enough to identify the slice and reach the reference-list / dec-ref-pic
/// marking syntax that rung 2 will need. Returns `None` if the referenced PPS
/// or SPS is missing.
pub fn parse_slice_header(
    nal: &Nal,
    sps_for: &impl Fn(u32) -> Option<Sps>,
    pps_for: &impl Fn(u32) -> Option<Pps>,
) -> Option<SliceHeader> {
    // Peek pps_id to resolve the parameter sets, then parse fully.
    let mut peek = BitReader::new(&nal.rbsp);
    let _first = peek.read_ue();
    let _stype = peek.read_ue();
    let pps_id = peek.read_ue();
    let pps = pps_for(pps_id)?;
    let sps = sps_for(pps.sps_id)?;
    let mut r = BitReader::new(&nal.rbsp);
    Some(parse_slice_header_into(&mut r, nal, &sps, &pps))
}

/// Parse `slice_header()` from an explicit reader, leaving it positioned at
/// `slice_data()`. Used by the decoder, which already holds the SPS/PPS.
pub fn parse_slice_header_into(r: &mut BitReader, nal: &Nal, sps: &Sps, pps: &Pps) -> SliceHeader {
    let is_idr = nal.nal_unit_type == 5;
    let first_mb_in_slice = r.read_ue();
    let slice_type = SliceType::from_ue(r.read_ue());
    let _pps_id = r.read_ue();
    let pps_id = pps.pps_id;

    if sps.separate_colour_plane_flag {
        let _colour_plane_id = r.read_bits(2);
    }
    let frame_num = r.read_bits(sps.log2_max_frame_num);

    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = r.read_bit() == 1;
        if field_pic_flag {
            bottom_field_flag = r.read_bit() == 1;
        }
    }

    let mut idr_pic_id = 0;
    if is_idr {
        idr_pic_id = r.read_ue();
    }

    let mut pic_order_cnt_lsb = 0;
    if sps.pic_order_cnt_type == 0 {
        pic_order_cnt_lsb = r.read_bits(sps.log2_max_pic_order_cnt_lsb);
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            let _delta_pic_order_cnt_bottom = r.read_se();
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        let _delta0 = r.read_se();
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            let _delta1 = r.read_se();
        }
    }

    if pps.redundant_pic_cnt_present_flag {
        let _redundant_pic_cnt = r.read_ue();
    }

    let mut direct_spatial_mv_pred_flag = false;
    if slice_type == SliceType::B {
        direct_spatial_mv_pred_flag = r.read_bit() == 1;
    }

    let mut num_ref_idx_l0_active = pps.num_ref_idx_l0_default_active;
    let mut num_ref_idx_l1_active = pps.num_ref_idx_l1_default_active;
    if matches!(slice_type, SliceType::P | SliceType::Sp | SliceType::B) {
        let override_flag = r.read_bit() == 1;
        if override_flag {
            num_ref_idx_l0_active = r.read_ue() + 1;
            if slice_type == SliceType::B {
                num_ref_idx_l1_active = r.read_ue() + 1;
            }
        }
    }

    // ref_pic_list_modification (§7.3.3.1) — present for all but I/SI.
    if !matches!(slice_type, SliceType::I | SliceType::Si) {
        ref_pic_list_modification(r); // list 0
        if slice_type == SliceType::B {
            ref_pic_list_modification(r); // list 1
        }
    }

    // pred_weight_table (§7.3.3.2) — only consumed to advance the reader.
    let chroma_array_type = if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    };
    if (pps.weighted_pred_flag && matches!(slice_type, SliceType::P | SliceType::Sp))
        || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B)
    {
        pred_weight_table(
            r,
            chroma_array_type != 0,
            num_ref_idx_l0_active,
            num_ref_idx_l1_active,
            slice_type == SliceType::B,
        );
    }

    // dec_ref_pic_marking (§7.3.3.3).
    if nal.nal_ref_idc != 0 {
        if is_idr {
            let _no_output_of_prior_pics = r.read_bit();
            let _long_term_reference = r.read_bit();
        } else {
            let adaptive = r.read_bit() == 1;
            if adaptive {
                loop {
                    let mmco = r.read_ue();
                    if mmco == 0 {
                        break;
                    }
                    if mmco == 1 || mmco == 3 {
                        let _difference_of_pic_nums_minus1 = r.read_ue();
                    }
                    if mmco == 2 {
                        let _long_term_pic_num = r.read_ue();
                    }
                    if mmco == 3 || mmco == 6 {
                        let _long_term_frame_idx = r.read_ue();
                    }
                    if mmco == 4 {
                        let _max_long_term_frame_idx_plus1 = r.read_ue();
                    }
                }
            }
        }
    }

    let mut cabac_init_idc = 0;
    if pps.entropy_coding_mode_flag && !matches!(slice_type, SliceType::I | SliceType::Si) {
        cabac_init_idc = r.read_ue();
    }

    let slice_qp_delta = r.read_se();
    let slice_qp = pps.pic_init_qp + slice_qp_delta;

    if matches!(slice_type, SliceType::Sp | SliceType::Si) {
        if slice_type == SliceType::Sp {
            let _sp_for_switch_flag = r.read_bit();
        }
        let _slice_qs_delta = r.read_se();
    }

    let mut disable_deblocking_filter_idc = 0;
    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = r.read_ue();
        if disable_deblocking_filter_idc != 1 {
            let _slice_alpha_c0_offset_div2 = r.read_se();
            let _slice_beta_offset_div2 = r.read_se();
        }
    }
    // slice_group_change_cycle (num_slice_groups > 1) is not handled — those
    // streams are vanishingly rare and unsupported by the macroblock layer too.

    // The reader now sits at slice_data(): for CABAC, cabac_alignment_one_bit
    // alignment happens in the entropy stage (rung 3); for CAVLC it starts here.

    SliceHeader {
        first_mb_in_slice,
        slice_type,
        pps_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        pic_order_cnt_lsb,
        is_idr,
        num_ref_idx_l0_active,
        num_ref_idx_l1_active,
        direct_spatial_mv_pred_flag,
        cabac_init_idc,
        slice_qp,
        disable_deblocking_filter_idc,
    }
}

/// ref_pic_list_modification() (ITU-T H.264 §7.3.3.1) — consumed only to
/// advance the reader.
fn ref_pic_list_modification(r: &mut BitReader) {
    if r.read_bit() == 1 {
        loop {
            let idc = r.read_ue();
            if idc == 3 {
                break;
            }
            match idc {
                0 | 1 => {
                    let _abs_diff_pic_num_minus1 = r.read_ue();
                }
                2 => {
                    let _long_term_pic_num = r.read_ue();
                }
                _ => break,
            }
        }
    }
}

/// pred_weight_table() (ITU-T H.264 §7.3.3.2) — consumed only to advance the
/// reader; weights are unused for MV extraction.
fn pred_weight_table(r: &mut BitReader, has_chroma: bool, n0: u32, n1: u32, is_b: bool) {
    let _luma_log2_weight_denom = r.read_ue();
    if has_chroma {
        let _chroma_log2_weight_denom = r.read_ue();
    }
    let consume_list = |r: &mut BitReader, n: u32| {
        for _ in 0..n {
            if r.read_bit() == 1 {
                let _luma_weight = r.read_se();
                let _luma_offset = r.read_se();
            }
            if has_chroma && r.read_bit() == 1 {
                for _ in 0..2 {
                    let _chroma_weight = r.read_se();
                    let _chroma_offset = r.read_se();
                }
            }
        }
    };
    consume_list(r, n0);
    if is_b {
        consume_list(r, n1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a BitReader over an explicit bit string like "0100".
    fn bits(s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in s.chars().filter(|c| *c == '0' || *c == '1') {
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
    fn read_bits_msb_first() {
        let d = bits("10110010");
        let mut r = BitReader::new(&d);
        assert_eq!(r.read_bits(4), 0b1011);
        assert_eq!(r.read_bits(4), 0b0010);
    }

    #[test]
    fn ue_golomb_known_vectors() {
        // ITU-T H.264 Table: codeNum 0..=6
        let cases = [
            ("1", 0u32),
            ("010", 1),
            ("011", 2),
            ("00100", 3),
            ("00101", 4),
            ("00110", 5),
            ("00111", 6),
            ("0001000", 7),
        ];
        for (s, want) in cases {
            let d = bits(s);
            let mut r = BitReader::new(&d);
            assert_eq!(r.read_ue(), want, "ue({s})");
        }
    }

    #[test]
    fn se_golomb_known_vectors() {
        // se mapping: 0->0, 1->1, 2->-1, 3->2, 4->-2 ...
        let cases = [
            ("1", 0i32),
            ("010", 1),
            ("011", -1),
            ("00100", 2),
            ("00101", -2),
            ("00110", 3),
        ];
        for (s, want) in cases {
            let d = bits(s);
            let mut r = BitReader::new(&d);
            assert_eq!(r.read_se(), want, "se({s})");
        }
    }

    #[test]
    fn emulation_prevention_removed() {
        assert_eq!(ebsp_to_rbsp(&[0, 0, 3, 1]), vec![0, 0, 1]);
        assert_eq!(ebsp_to_rbsp(&[0, 0, 3, 0, 0, 3, 2]), vec![0, 0, 0, 0, 2]);
        // 0x03 not after two zeros is preserved
        assert_eq!(ebsp_to_rbsp(&[1, 3, 4]), vec![1, 3, 4]);
    }

    #[test]
    fn annexb_split_two_nals() {
        // start(4) + NAL[0x67,...] + start(3) + NAL[0x68,...]
        let stream = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, //
            0, 0, 1, 0x68, 0xCE, //
        ];
        let nals = split_annexb(&stream);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].nal_unit_type, 7); // SPS
        assert_eq!(nals[1].nal_unit_type, 8); // PPS
    }

    #[test]
    fn avcc_split_roundtrip() {
        // 4-byte length prefix, one NAL of type 1 (non-IDR slice)
        let data = [0, 0, 0, 3, 0x61, 0xAA, 0xBB];
        let nals = split_avcc(&data, 4);
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].nal_unit_type, 1);
        assert_eq!(nals[0].rbsp, vec![0xAA, 0xBB]);
    }
}
