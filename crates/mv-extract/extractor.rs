//! extractor9 — the thesis from-scratch compressed-domain MV extractor.
//!
//! libavformat is used ONLY to demux the container into H.264 packets +
//! extradata. Everything past that — NAL split, parameter sets, slice headers,
//! CAVLC entropy decode, macroblock layer and MV prediction — is hand-rolled in
//! `mv_extract::custom`, never touching libavcodec's decode path (thesis Ch4).
//!
//! Status: CAVLC and CABAC P/I slices produce motion vectors in FFmpeg's
//! AV_FRAME_DATA_MOTION_VECTORS layout, bit-identical to extractor1 (verified on
//! High-profile CABAC clips). B slices are skipped (rung 4) — they emit no MVs.
//!
//! Codec routing: H.264 streams take the full extraction path above. H.265/HEVC
//! streams are handled by `mv_extract::{hevc, hevc_slice}` — a from-scratch HEVC
//! decoder (NAL/VPS/SPS/PPS/slice-header, HEVC-CABAC, CTU quadtree, merge/AMVP +
//! temporal MV reconstruction). It reproduces 100% of FFmpeg's HEVC motion
//! vectors (verified on dashcam: every FFmpeg MV row matched, frame-label offset
//! aside — we tag by decode order, FFmpeg by display order). Tiles/WPP (e.g. the
//! Rext MCTTR clip) are not yet supported and are skipped.
//!
//! Threading: the demux/NAL-split/parameter-set pass is inherently serial, but a
//! picture's decode is self-contained (spatial-only MV prediction, no DPB / no
//! cross-frame reads), so whole frames decode in parallel. The serial path (one
//! reused `FrameGrids`, lowest memory) is the default; passing the 5th CLI arg
//! `is_single_threaded = 0` enables a producer → worker-pool → reordering-writer
//! pipeline. Output is byte-identical either way (writer reorders by frame
//! index). `E9_THREADS` overrides the worker count.
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::slice;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use ffmpeg_sys_next as ff;

use mv_extract::ffmpeg_common::{get_current_rss_kb, open_mv_compact_writer, ExtractorArgs, FileMvCompactWriter};
use mv_extract::hevc;
use mv_extract::custom::slice::{decode_slice, decode_slice_cabac, FrameGrids};
use mv_extract::custom::{
    parse_avcc, parse_pps, parse_slice_header, parse_slice_header_into, parse_sps, split_annexb,
    split_avcc, BitReader, Nal, Pps, SliceType, Sps,
};
use mv_types::motion_vector::MvCompact;

fn main() {
    let Some(args) = ExtractorArgs::from_env() else {
        std::process::exit(255);
    };

    let writer = if args.do_print {
        match open_mv_compact_writer(&args.output_file) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("Failed to open output file: {}", e);
                std::process::exit(255);
            }
        }
    } else {
        None
    };

    let (stats, writer) = unsafe { run(&args, writer) };

    let total_mvs = writer.as_ref().map(|w| w.total()).unwrap_or(0);
    if let Some(mut w) = writer {
        let _ = w.flush();
    }

    // Benchmark harness contract: one stdout line "<frames> <mvs> <rss_kb>".
    println!("{} {} {}", stats.frames, total_mvs, get_current_rss_kb());

    if args.is_verbose {
        if stats.hevc {
            eprintln!(
                "extractor9 (hevc rung1): {}x{} profile_idc={} ctb={} pictures={}",
                stats.width, stats.height, stats.profile_idc, stats.ctb, stats.frames
            );
            eprintln!(
                "  slices: I={} P={} B={} dependent={}  slice_qp=[{}..{}] bad_headers={} unsupported={}",
                stats.i_slices, stats.p_slices, stats.b_slices, stats.dependent,
                stats.qp_min, stats.qp_max, stats.bad_header, stats.unsupported
            );
            eprintln!(
                "  CABAC sync: {}/{} slices decoded CTU tree to clean end_of_slice (MV values = next)",
                stats.sync_ok, stats.sync_attempt
            );
        } else {
            eprintln!(
                "extractor9 (thesis): pictures={} cavlc={} cabac={} b_skipped={} threads={}",
                stats.frames, stats.cavlc_decoded, stats.cabac_decoded, stats.b_skipped,
                if args.is_single_threaded { 1 } else { worker_count() },
            );
            eprintln!(
                "  sync: slices_incomplete={} parse_errors={} (both should be 0 on clean CAVLC)",
                stats.incomplete, stats.parse_errors
            );
        }
    }
}

#[derive(Default, Clone)]
struct Stats {
    frames: u64, // pictures (H.264) / pictures (HEVC)
    cavlc_decoded: u64,
    cabac_decoded: u64,
    b_skipped: u64,
    incomplete: u64,
    parse_errors: u64,
    // HEVC rung-1 report (only populated for H.265 inputs).
    hevc: bool,
    i_slices: u64,
    p_slices: u64,
    b_slices: u64,
    dependent: u64,
    width: u32,
    height: u32,
    ctb: u32,
    profile_idc: u8,
    qp_min: i32,
    qp_max: i32,
    unsupported: u64,
    bad_header: u64,
    sync_attempt: u64,
    sync_ok: u64,
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.cavlc_decoded += o.cavlc_decoded;
        self.cabac_decoded += o.cabac_decoded;
        self.incomplete += o.incomplete;
        self.parse_errors += o.parse_errors;
    }
}

/// One coded slice: the owned RBSP (in `nal`) plus the resolved parameter sets.
struct SliceJob {
    nal: Nal,
    sps: Sps,
    pps: Pps,
}

/// All non-B slices of one coded picture, plus its output frame index and size.
struct Picture {
    frame_index: i32,
    mb_w: usize,
    mb_h: usize,
    jobs: Vec<SliceJob>,
}

fn worker_count() -> usize {
    if let Ok(n) = std::env::var("E9_THREADS") {
        if let Ok(n) = n.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

unsafe fn run(
    args: &ExtractorArgs,
    writer: Option<FileMvCompactWriter>,
) -> (Stats, Option<FileMvCompactWriter>) {
    // Route by codec: HEVC has its own (rung-1) parser; everything else takes
    // the H.264 extraction path.
    if detect_codec(args) == ff::AVCodecID::AV_CODEC_ID_HEVC {
        return run_hevc(args, writer);
    }
    if args.is_single_threaded {
        run_serial(args, writer)
    } else {
        run_threaded(args, writer)
    }
}

/// Probe the container for the video stream's codec id (cheap header-only open).
unsafe fn detect_codec(args: &ExtractorArgs) -> ff::AVCodecID {
    let mut fmt_ctx: *mut ff::AVFormatContext = ptr::null_mut();
    let c_video = CString::new(args.video_file.as_str()).unwrap();
    if ff::avformat_open_input(&mut fmt_ctx, c_video.as_ptr(), ptr::null_mut(), ptr::null_mut()) < 0
    {
        eprintln!("Could not open input file.");
        std::process::exit(255);
    }
    let _ = ff::avformat_find_stream_info(fmt_ctx, ptr::null_mut());
    let mut id = ff::AVCodecID::AV_CODEC_ID_NONE;
    for i in 0..(*fmt_ctx).nb_streams as usize {
        let s = *(*fmt_ctx).streams.add(i);
        if (*(*s).codecpar).codec_type == ff::AVMediaType::AVMEDIA_TYPE_VIDEO {
            id = (*(*s).codecpar).codec_id;
            break;
        }
    }
    ff::avformat_close_input(&mut fmt_ctx as *mut _);
    id
}

/// Compute the HEVC picture order count (ITU-T H.265 §8.3.1).
fn hevc_compute_poc(log2_max_poc_lsb: u32, poc_tid0: i32, poc_lsb: i32, nal_type: u8) -> i32 {
    let max = 1i32 << log2_max_poc_lsb;
    let prev_lsb = poc_tid0 % max;
    let prev_msb = poc_tid0 - prev_lsb;
    let msb = if poc_lsb < prev_lsb && prev_lsb - poc_lsb >= max / 2 {
        prev_msb + max
    } else if poc_lsb > prev_lsb && poc_lsb - prev_lsb > max / 2 {
        prev_msb - max
    } else {
        prev_msb
    };
    // BLA types reset MSB to 0 (16/17/18).
    if matches!(nal_type, 16 | 17 | 18) {
        poc_lsb
    } else {
        msb + poc_lsb
    }
}

/// Build a reference POC list by cycling the primary then secondary candidate
/// lists until `nb` entries are filled (ITU-T H.265 §8.3.4, short-term only).
fn build_ref_list(primary: &[i32], secondary: &[i32], nb: usize) -> Vec<i32> {
    let mut tmp = Vec::new();
    if primary.is_empty() && secondary.is_empty() {
        return tmp;
    }
    while tmp.len() < nb {
        for &p in primary {
            tmp.push(p);
        }
        for &p in secondary {
            tmp.push(p);
        }
    }
    tmp.truncate(nb);
    tmp
}

/// HEVC MV extraction (rungs 1-4): demux, parse params/slice headers, decode the
/// CTU tree, reconstruct motion vectors (merge/AMVP/temporal) and write them.
unsafe fn run_hevc(
    args: &ExtractorArgs,
    mut writer: Option<FileMvCompactWriter>,
) -> (Stats, Option<FileMvCompactWriter>) {
    let mut st = Stats { hevc: true, qp_min: 999, ..Default::default() };

    let mut fmt_ctx: *mut ff::AVFormatContext = ptr::null_mut();
    let c_video = CString::new(args.video_file.as_str()).unwrap();
    if ff::avformat_open_input(&mut fmt_ctx, c_video.as_ptr(), ptr::null_mut(), ptr::null_mut()) < 0
    {
        eprintln!("Could not open input file.");
        std::process::exit(255);
    }
    if ff::avformat_find_stream_info(fmt_ctx, ptr::null_mut()) < 0 {
        eprintln!("Could not find stream info.");
        std::process::exit(255);
    }

    let mut vsi: i32 = -1;
    for i in 0..(*fmt_ctx).nb_streams as usize {
        let s = *(*fmt_ctx).streams.add(i);
        if (*(*s).codecpar).codec_type == ff::AVMediaType::AVMEDIA_TYPE_VIDEO {
            vsi = i as i32;
            break;
        }
    }
    if vsi < 0 {
        eprintln!("Could not find video stream");
        std::process::exit(255);
    }
    let par = (*(*(*fmt_ctx).streams.add(vsi as usize))).codecpar;

    let mut sps_map: HashMap<u32, hevc::Sps> = HashMap::new();
    let mut pps_map: HashMap<u32, hevc::Pps> = HashMap::new();

    let note_sps = |st: &mut Stats, s: &hevc::Sps| {
        if st.width == 0 {
            st.width = s.pic_width_in_luma_samples;
            st.height = s.pic_height_in_luma_samples;
            st.ctb = s.ctb_size_y();
            st.profile_idc = s.profile_idc;
        }
    };

    let extradata = if !(*par).extradata.is_null() && (*par).extradata_size > 0 {
        slice::from_raw_parts((*par).extradata, (*par).extradata_size as usize).to_vec()
    } else {
        Vec::new()
    };
    let hvcc = hevc::parse_hvcc(&extradata);
    let nal_length_size = hvcc.as_ref().map(|h| h.nal_length_size);
    if let Some(h) = &hvcc {
        for n in &h.param_sets {
            match n.nal_unit_type {
                hevc::NAL_SPS => {
                    let s = hevc::parse_sps(&n.rbsp);
                    note_sps(&mut st, &s);
                    sps_map.insert(s.sps_id, s);
                }
                hevc::NAL_PPS => {
                    let p = hevc::parse_pps(&n.rbsp);
                    pps_map.insert(p.pps_id, p);
                }
                _ => {}
            }
        }
    }

    use mv_extract::hevc_slice::{decode_slice, FrameMv, HevcScratch, RefInfo};
    let mut poc_tid0 = 0i32;
    let mut dpb: Vec<FrameMv> = Vec::new();
    let mut frame_index = -1i32;
    let mut total_mvs = 0u64;
    let mut scratch = HevcScratch::default();

    let pkt = ff::av_packet_alloc();
    let mut pkt = pkt;
    while ff::av_read_frame(fmt_ctx, pkt) >= 0 {
        if (*pkt).stream_index == vsi {
            let data = slice::from_raw_parts((*pkt).data, (*pkt).size as usize);
            let nals = match nal_length_size {
                Some(sz) => hevc::split_length_prefixed(data, sz),
                None => hevc::split_annexb(data),
            };
            for n in &nals {
                match n.nal_unit_type {
                    hevc::NAL_SPS => {
                        let s = hevc::parse_sps(&n.rbsp);
                        note_sps(&mut st, &s);
                        sps_map.insert(s.sps_id, s);
                    }
                    hevc::NAL_PPS => {
                        let p = hevc::parse_pps(&n.rbsp);
                        pps_map.insert(p.pps_id, p);
                    }
                    t if hevc::is_vcl(t) => {
                        let sps_for = |id: u32| sps_map.get(&id).cloned();
                        let pps_for = |id: u32| pps_map.get(&id).cloned();
                        let Some(sh) = hevc::parse_slice_header(n, &sps_for, &pps_for) else {
                            continue;
                        };
                        if sh.first_slice_segment_in_pic_flag {
                            st.frames += 1;
                            frame_index += 1;
                        }
                        match sh.slice_type {
                            Some(hevc::SliceType::I) => st.i_slices += 1,
                            Some(hevc::SliceType::P) => st.p_slices += 1,
                            Some(hevc::SliceType::B) => st.b_slices += 1,
                            None => st.dependent += 1,
                        }
                        st.qp_min = st.qp_min.min(sh.slice_qp);
                        st.qp_max = st.qp_max.max(sh.slice_qp);
                        if sh.unsupported {
                            st.unsupported += 1;
                        }
                        if sh.data_offset > n.rbsp.len() || sh.slice_qp < 0 || sh.slice_qp > 51 {
                            st.bad_header += 1;
                        }
                        if sh.unsupported || sh.slice_type.is_none() {
                            continue;
                        }
                        let Some(p) = pps_map.get(&sh.pps_id).cloned() else { continue };
                        let Some(s) = sps_map.get(&p.sps_id).cloned() else { continue };

                        // POC + short-term reference POC lists (§8.3.x).
                        let poc = if sh.is_idr {
                            0
                        } else {
                            hevc_compute_poc(s.log2_max_poc_lsb, poc_tid0, sh.poc_lsb as i32, n.nal_unit_type)
                        };
                        let mut st_before = Vec::new();
                        let mut st_after = Vec::new();
                        let rps = &sh.short_rps;
                        for i in 0..rps.delta_poc.len() {
                            if rps.used[i] {
                                if (i as u32) < rps.num_negative {
                                    st_before.push(poc + rps.delta_poc[i]);
                                } else {
                                    st_after.push(poc + rps.delta_poc[i]);
                                }
                            }
                        }
                        let l0 = build_ref_list(&st_before, &st_after, sh.nb_refs[0] as usize);
                        let l1 = build_ref_list(&st_after, &st_before, sh.nb_refs[1] as usize);
                        let long0 = vec![false; l0.len()];
                        let long1 = vec![false; l1.len()];

                        // collocated reference frame for TMVP
                        let col_poc = {
                            let list = if sh.collocated_list == 0 { &l0 } else { &l1 };
                            list.get(sh.collocated_ref_idx as usize).copied()
                        };
                        let col = if sh.slice_temporal_mvp_enabled {
                            col_poc.and_then(|cp| dpb.iter().find(|f| f.poc == cp))
                        } else {
                            None
                        };

                        let ri = RefInfo {
                            poc,
                            ref_poc: [l0, l1],
                            ref_long: [long0, long1],
                            col,
                        };
                        st.sync_attempt += 1;
                        let (res, frame, mvs) =
                            decode_slice(&n.rbsp, &sh, &s, &p, &ri, frame_index, Vec::new(), &mut scratch);
                        if res.ok {
                            st.sync_ok += 1;
                        }
                        if let Some(w) = writer.as_mut() {
                            for mv in &mvs {
                                let _ = w.write(mv);
                            }
                        }
                        total_mvs += mvs.len() as u64;
                        // retain frame for TMVP (bounded DPB); recycle the
                        // evicted frame's MV buffer.
                        dpb.push(frame);
                        if dpb.len() > 16 {
                            let old = dpb.remove(0);
                            scratch.mvf_pool.push(old.tab_mvf);
                        }
                        if n.nuh_temporal_id_plus1 == 1 {
                            poc_tid0 = poc;
                        }
                        let _ = total_mvs;
                    }
                    _ => {}
                }
            }
        }
        ff::av_packet_unref(pkt);
    }

    ff::av_packet_free(&mut pkt as *mut _);
    ff::avformat_close_input(&mut fmt_ctx as *mut _);
    (st, writer)
}

/// Serial path: one reused `FrameGrids`, decode + write inline. Lowest memory.
unsafe fn run_serial(
    args: &ExtractorArgs,
    mut writer: Option<FileMvCompactWriter>,
) -> (Stats, Option<FileMvCompactWriter>) {
    let mut stats = Stats::default();
    let mut grid: Option<FrameGrids> = None;
    let mut out = Vec::new();

    let (frames, b_skipped) = demux_pictures(args, |p| {
        out.clear();
        decode_picture(&p, &mut grid, &mut out, &mut stats);
        if let Some(w) = writer.as_mut() {
            for mv in &out {
                let _ = w.write(mv);
            }
        }
    });

    stats.frames = frames;
    stats.b_skipped = b_skipped;
    (stats, writer)
}

/// Threaded path: serial producer (demux) → worker pool (one reused `FrameGrids`
/// per worker) → writer thread that reorders results by frame index. Output is
/// identical to the serial path. Memory scales with worker count.
unsafe fn run_threaded(
    args: &ExtractorArgs,
    writer: Option<FileMvCompactWriter>,
) -> (Stats, Option<FileMvCompactWriter>) {
    let nthreads = worker_count();
    // Bounded so the producer can't race ahead and buffer every frame's RBSP in
    // memory; backpressure caps in-flight pictures at ~2x the worker count.
    let (work_tx, work_rx) = mpsc::sync_channel::<Picture>(nthreads * 2);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (res_tx, res_rx) = mpsc::channel::<(i32, Vec<MvCompact>, Stats)>();

    let mut workers = Vec::with_capacity(nthreads);
    for _ in 0..nthreads {
        let rx = Arc::clone(&work_rx);
        let tx = res_tx.clone();
        workers.push(std::thread::spawn(move || {
            let mut grid: Option<FrameGrids> = None;
            loop {
                // Hold the lock only across the (cheap) handoff; decoding runs
                // unlocked so workers don't serialise.
                let job = rx.lock().unwrap().recv();
                let Ok(p) = job else { break };
                let mut out = Vec::new();
                let mut st = Stats::default();
                decode_picture(&p, &mut grid, &mut out, &mut st);
                let _ = tx.send((p.frame_index, out, st));
            }
        }));
    }
    drop(res_tx); // results channel closes once every worker exits

    // Writer thread: reorder by frame index and write in order, owning the file.
    let writer_thread = std::thread::spawn(move || {
        let mut writer = writer;
        let mut agg = Stats::default();
        let mut next: i32 = 0;
        let mut pending: HashMap<i32, Vec<MvCompact>> = HashMap::new();
        for (idx, out, st) in res_rx {
            agg.merge(&st);
            pending.insert(idx, out);
            while let Some(mvs) = pending.remove(&next) {
                if let Some(w) = writer.as_mut() {
                    for mv in &mvs {
                        let _ = w.write(mv);
                    }
                }
                next += 1;
            }
        }
        (agg, writer)
    });

    // Producer (this thread): demux and dispatch whole pictures.
    let (frames, b_skipped) = demux_pictures(args, |p| {
        let _ = work_tx.send(p);
    });
    drop(work_tx); // workers exit when the work channel drains

    for w in workers {
        let _ = w.join();
    }
    let (mut stats, writer) = writer_thread.join().unwrap();
    stats.frames = frames;
    stats.b_skipped = b_skipped;
    (stats, writer)
}

/// Decode every slice of `p` into a reused `FrameGrids` and append its motion
/// vectors to `out`. Shared by both the serial and threaded paths.
fn decode_picture(p: &Picture, grid: &mut Option<FrameGrids>, out: &mut Vec<MvCompact>, st: &mut Stats) {
    if p.jobs.is_empty() {
        return; // B-only / empty picture: still a frame, but emits no MVs
    }
    let cabac = p.jobs[0].pps.entropy_coding_mode_flag;
    match grid {
        Some(g) if g.dims() == (p.mb_w, p.mb_h) => g.reset(cabac),
        _ => *grid = Some(FrameGrids::new(p.mb_w, p.mb_h)),
    }
    let g = grid.as_mut().unwrap();

    // sid is per-picture (slice index): only its distinctness within the
    // picture matters for same-slice neighbour checks.
    for (sid, job) in p.jobs.iter().enumerate() {
        let mut r = BitReader::new(&job.nal.rbsp);
        let sh = parse_slice_header_into(&mut r, &job.nal, &job.sps, &job.pps);
        let res = if job.pps.entropy_coding_mode_flag {
            // CABAC: slice_data starts at the next byte boundary
            // (cabac_alignment_one_bit).
            let byte_start = r.pos().div_ceil(8);
            st.cabac_decoded += 1;
            decode_slice_cabac(g, &job.nal.rbsp, byte_start, &sh, &job.sps, &job.pps, sid as i32)
        } else {
            st.cavlc_decoded += 1;
            decode_slice(g, &mut r, &sh, &job.sps, &job.pps, sid as i32)
        };
        if !res.ok {
            st.parse_errors += 1;
        }
        if sh.first_mb_in_slice as usize + res.mbs_decoded != g.mb_count() {
            st.incomplete += 1;
        }
    }
    g.export_mvs_compact(p.frame_index, out);
}

/// Demux the container, split NALs, track SPS/PPS, and group coded slices into
/// `Picture`s (one per coded frame). Calls `on_picture` once per picture, in
/// decode order. Returns `(picture_count, b_slices_skipped)`.
unsafe fn demux_pictures(args: &ExtractorArgs, mut on_picture: impl FnMut(Picture)) -> (u64, u64) {
    let mut fmt_ctx: *mut ff::AVFormatContext = ptr::null_mut();
    let c_video = CString::new(args.video_file.as_str()).unwrap();
    if ff::avformat_open_input(&mut fmt_ctx, c_video.as_ptr(), ptr::null_mut(), ptr::null_mut()) < 0
    {
        eprintln!("Could not open input file.");
        std::process::exit(255);
    }
    if ff::avformat_find_stream_info(fmt_ctx, ptr::null_mut()) < 0 {
        eprintln!("Could not find stream info.");
        std::process::exit(255);
    }

    let mut vsi: i32 = -1;
    let nb = (*fmt_ctx).nb_streams as usize;
    for i in 0..nb {
        let s = *(*fmt_ctx).streams.add(i);
        if (*(*s).codecpar).codec_type == ff::AVMediaType::AVMEDIA_TYPE_VIDEO {
            vsi = i as i32;
            break;
        }
    }
    if vsi < 0 {
        eprintln!("Could not find video stream");
        std::process::exit(255);
    }
    let par = (*(*(*fmt_ctx).streams.add(vsi as usize))).codecpar;

    let mut sps_map: HashMap<u32, Sps> = HashMap::new();
    let mut pps_map: HashMap<u32, Pps> = HashMap::new();

    let extradata = if !(*par).extradata.is_null() && (*par).extradata_size > 0 {
        slice::from_raw_parts((*par).extradata, (*par).extradata_size as usize).to_vec()
    } else {
        Vec::new()
    };
    let avcc = parse_avcc(&extradata);
    let nal_length_size = avcc.as_ref().map(|a| a.nal_length_size);
    if let Some(a) = &avcc {
        for n in &a.sps {
            let s = parse_sps(&n.rbsp);
            sps_map.insert(s.sps_id, s);
        }
        for n in &a.pps {
            let p = parse_pps(&n.rbsp);
            pps_map.insert(p.pps_id, p);
        }
    }

    let mut cur: Option<Picture> = None;
    let mut next_index: i32 = 0;
    let mut b_skipped: u64 = 0;

    let pkt = ff::av_packet_alloc();
    let mut pkt = pkt;
    while ff::av_read_frame(fmt_ctx, pkt) >= 0 {
        if (*pkt).stream_index == vsi {
            let data = slice::from_raw_parts((*pkt).data, (*pkt).size as usize);
            let nals = match nal_length_size {
                Some(sz) => split_avcc(data, sz),
                None => split_annexb(data),
            };
            for n in nals {
                match n.nal_unit_type {
                    7 => {
                        let s = parse_sps(&n.rbsp);
                        sps_map.insert(s.sps_id, s);
                    }
                    8 => {
                        let p = parse_pps(&n.rbsp);
                        pps_map.insert(p.pps_id, p);
                    }
                    1 | 5 => {
                        let sps_for = |id: u32| sps_map.get(&id).cloned();
                        let pps_for = |id: u32| pps_map.get(&id).cloned();
                        let Some(sh) = parse_slice_header(&n, &sps_for, &pps_for) else {
                            continue;
                        };
                        let Some(pps) = pps_map.get(&sh.pps_id).cloned() else {
                            continue;
                        };
                        let Some(sps) = sps_map.get(&pps.sps_id).cloned() else {
                            continue;
                        };

                        // New picture boundary: emit the previous picture.
                        if sh.first_mb_in_slice == 0 {
                            if let Some(prev) = cur.take() {
                                on_picture(prev);
                            }
                            cur = Some(Picture {
                                frame_index: next_index,
                                mb_w: sps.pic_width_in_mbs as usize,
                                mb_h: sps.frame_height_in_mbs() as usize,
                                jobs: Vec::new(),
                            });
                            next_index += 1;
                        }

                        // B slices need bi/direct prediction (rung 4) — skip,
                        // but the picture itself still counts as a frame.
                        if matches!(sh.slice_type, SliceType::B) {
                            b_skipped += 1;
                            continue;
                        }
                        if let Some(p) = cur.as_mut() {
                            p.jobs.push(SliceJob { nal: n, sps, pps });
                        }
                    }
                    _ => {}
                }
            }
        }
        ff::av_packet_unref(pkt);
    }
    if let Some(prev) = cur.take() {
        on_picture(prev);
    }

    ff::av_packet_free(&mut pkt as *mut _);
    ff::avformat_close_input(&mut fmt_ctx as *mut _);
    (next_index as u64, b_skipped)
}
