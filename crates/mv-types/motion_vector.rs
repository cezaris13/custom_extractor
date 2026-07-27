use std::fs;
use std::io::Write;

#[derive(Debug, Clone, Default)]
pub struct MotionVector {
    pub frame: i32,
    pub source: i32, // remove this?
    pub w: i32, // remove this?
    pub h: i32, // remove this?
    pub src_x: f64,
    pub src_y: f64,
    pub dst_x: f64,
    pub dst_y: f64,
    // `u64` to match `AVMotionVector::flags`. Serialized in hex (`0x...`) to
    // stay compatible with the C++ writer.
    pub flags: u64, // remove this?
    pub motion_x: f64,
    pub motion_y: f64,
    pub motion_scale: f64, // remove this??
}

/// Compact motion vector mirroring `AVMotionVectorCompact` from the custom
/// FFmpeg patch. Stores only the two endpoints (src + dst) and direction,
/// dropping w/h/flags/motion_x/motion_y/motion_scale.
#[derive(Debug, Clone, Default)]
pub struct MvCompact {
    pub frame: i32,
    pub source: i32,
    pub src_x: i16,
    pub src_y: i16,
    pub dst_x: i16,
    pub dst_y: i16,
}

fn parse_f64(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn parse_flags(s: &str) -> Option<u64> {
    let t = s.trim();
    if let Some(stripped) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(stripped, 16).ok()
    } else {
        t.parse::<u64>().ok()
    }
}

/// Which `MotionVector` field a CSV column position feeds. Precomputed once
/// from the header row so each row can dispatch fields in a single pass with
/// an array index instead of `Vec<&str>` + name lookups per row.
#[derive(Clone, Copy)]
enum Col {
    Frame,
    Source,
    W,
    H,
    SrcX,
    SrcY,
    DstX,
    DstY,
    Flags,
    MotionX,
    MotionY,
    MotionScale,
}

pub fn load_motion_vectors(
    csv_file: &str,
) -> Result<Vec<MotionVector>, Box<dyn std::error::Error>> {
    // Read once instead of allocating a String per line via BufRead::lines();
    // str::lines() below then borrows &str slices out of `content` for free.
    let content = fs::read_to_string(csv_file)?;
    let mut lines = content.lines();

    let header_line = lines.next().ok_or("Empty CSV file")?;
    let headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();

    let find_col = |name: &str| -> Option<usize> { headers.iter().position(|&h| h == name) };

    if find_col("frame").is_none() {
        return Err("Missing 'frame' column".into());
    }
    if find_col("src_x").is_none() {
        return Err("Missing 'src_x' column".into());
    }
    if find_col("src_y").is_none() {
        return Err("Missing 'src_y' column".into());
    }

    let mut roles: Vec<Option<Col>> = vec![None; headers.len()];
    for (name, role) in [
        ("frame", Col::Frame),
        ("source", Col::Source),
        ("w", Col::W),
        ("h", Col::H),
        ("src_x", Col::SrcX),
        ("src_y", Col::SrcY),
        ("dst_x", Col::DstX),
        ("dst_y", Col::DstY),
        ("flags", Col::Flags),
        ("motion_x", Col::MotionX),
        ("motion_y", Col::MotionY),
        ("motion_scale", Col::MotionScale),
    ] {
        if let Some(idx) = find_col(name) {
            roles[idx] = Some(role);
        }
    }

    let mut vectors = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let mut frame_f: Option<f64> = None;
        let mut source_f: Option<f64> = None;
        let mut w_f: Option<f64> = None;
        let mut h_f: Option<f64> = None;
        let mut src_x_f: Option<f64> = None;
        let mut src_y_f: Option<f64> = None;
        let mut dst_x_f: Option<f64> = None;
        let mut dst_y_f: Option<f64> = None;
        let mut flags_s: Option<&str> = None;
        let mut motion_x_f: Option<f64> = None;
        let mut motion_y_f: Option<f64> = None;
        let mut motion_scale_f: Option<f64> = None;

        for (i, field) in line.split(',').enumerate() {
            let Some(Some(role)) = roles.get(i) else {
                continue;
            };
            match role {
                Col::Frame => frame_f = parse_f64(field),
                Col::Source => source_f = parse_f64(field),
                Col::W => w_f = parse_f64(field),
                Col::H => h_f = parse_f64(field),
                Col::SrcX => src_x_f = parse_f64(field),
                Col::SrcY => src_y_f = parse_f64(field),
                Col::DstX => dst_x_f = parse_f64(field),
                Col::DstY => dst_y_f = parse_f64(field),
                Col::Flags => flags_s = Some(field),
                Col::MotionX => motion_x_f = parse_f64(field),
                Col::MotionY => motion_y_f = parse_f64(field),
                Col::MotionScale => motion_scale_f = parse_f64(field),
            }
        }

        let Some(frame_val) = frame_f else { continue };
        let Some(src_x_val) = src_x_f else { continue };
        let Some(src_y_val) = src_y_f else { continue };

        let motion_scale_val = motion_scale_f.unwrap_or(0.0);
        let motion_x_val = motion_x_f.unwrap_or(0.0);
        let motion_y_val = motion_y_f.unwrap_or(0.0);

        let derive_dst = |src: f64, motion: f64| -> f64 {
            if motion_scale_val > 0.0 {
                src + motion / motion_scale_val
            } else {
                src
            }
        };
        let dst_x_val = dst_x_f.unwrap_or_else(|| derive_dst(src_x_val, motion_x_val));
        let dst_y_val = dst_y_f.unwrap_or_else(|| derive_dst(src_y_val, motion_y_val));

        let mv = MotionVector {
            frame: frame_val as i32,
            source: source_f.unwrap_or(0.0) as i32,
            w: w_f.unwrap_or(0.0) as i32,
            h: h_f.unwrap_or(0.0) as i32,
            src_x: src_x_val,
            src_y: src_y_val,
            dst_x: dst_x_val,
            dst_y: dst_y_val,
            flags: flags_s.and_then(parse_flags).unwrap_or(0),
            motion_x: motion_x_val,
            motion_y: motion_y_val,
            motion_scale: motion_scale_val,
        };

        vectors.push(mv);
    }

    Ok(vectors)
}

/// Streaming CSV writer — writes the header on construction and lets callers
/// append rows incrementally. Used by the extractor binaries, which emit one
/// batch of motion vectors per decoded frame and do not want to buffer the
/// entire video in memory. Keeps a running `total` so the binary can print the
/// same `<frames> <mvs>` summary as the C++ version.
///
/// Rows are written directly to the underlying writer via `itoa`/`ryu` byte
/// formatters — no intermediate `String` or `Vec<String>` allocation per row.
/// Profiling showed the old `.to_string()`/`Vec::from_iter` path dominated
/// runtime on MV-heavy videos (≈30% of total), almost entirely from `malloc`
/// churn on the tiny per-column `String`s.
pub struct MotionVectorCsvWriter<W: Write> {
    inner: W,
    total: i64,
}

const MV_HEADER: &[u8] =
    b"frame,source,w,h,src_x,src_y,dst_x,dst_y,flags,motion_x,motion_y,motion_scale\n";

impl<W: Write> MotionVectorCsvWriter<W> {
    pub fn new(mut inner: W) -> std::io::Result<Self> {
        inner.write_all(MV_HEADER)?;
        Ok(Self { inner, total: 0 })
    }

    pub fn write(&mut self, v: &MotionVector) -> std::io::Result<()> {
        let mut ibuf = itoa::Buffer::new();
        let mut fbuf = ryu::Buffer::new();
        let w = &mut self.inner;
        w.write_all(ibuf.format(v.frame).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(ibuf.format(v.source).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(ibuf.format(v.w).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(ibuf.format(v.h).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.src_x).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.src_y).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.dst_x).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.dst_y).as_bytes())?;
        w.write_all(b",")?;
        // flags: hex, preserving "0x..." format for compatibility with the C++ writer.
        write!(w, "0x{:x}", v.flags)?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.motion_x).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.motion_y).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(fbuf.format(v.motion_scale).as_bytes())?;
        w.write_all(b"\n")?;
        self.total += 1;
        Ok(())
    }

    pub fn total(&self) -> i64 {
        self.total
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

pub struct MvCompactCsvWriter<W: Write> {
    inner: W,
    total: i64,
}

const MV_COMPACT_HEADER: &[u8] = b"frame,source,src_x,src_y,dst_x,dst_y\n";

impl<W: Write> MvCompactCsvWriter<W> {
    pub fn new(mut inner: W) -> std::io::Result<Self> {
        inner.write_all(MV_COMPACT_HEADER)?;
        Ok(Self { inner, total: 0 })
    }

    pub fn write(&mut self, v: &MvCompact) -> std::io::Result<()> {
        let mut buf = itoa::Buffer::new();
        let w = &mut self.inner;
        w.write_all(buf.format(v.frame).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(buf.format(v.source).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(buf.format(v.src_x).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(buf.format(v.src_y).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(buf.format(v.dst_x).as_bytes())?;
        w.write_all(b",")?;
        w.write_all(buf.format(v.dst_y).as_bytes())?;
        w.write_all(b"\n")?;
        self.total += 1;
        Ok(())
    }

    pub fn total(&self) -> i64 {
        self.total
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

pub fn get_frame_vectors(all_vectors: &[MotionVector], frame_number: i32) -> Vec<MotionVector> {
    all_vectors
        .iter()
        .filter(|v| v.frame == frame_number)
        .cloned()
        .collect()
}

pub fn get_max_frame(vectors: &[MotionVector]) -> i32 {
    vectors.iter().map(|v| v.frame).max().unwrap_or(0)
}

pub fn reduce_motion_vectors(
    frame_data: &[MotionVector],
    max_vectors: usize,
) -> Vec<MotionVector> {
    let mut mag_indices: Vec<(f64, usize)> = frame_data
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let dx = v.dst_x - v.src_x;
            let dy = v.dst_y - v.src_y;
            let mag = (dx * dx + dy * dy).sqrt();
            if mag > 2.0 {
                Some((mag, i))
            } else {
                None
            }
        })
        .collect();

    if mag_indices.len() > max_vectors {
        mag_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        mag_indices.truncate(max_vectors);
    }

    mag_indices
        .iter()
        .map(|(_, idx)| frame_data[*idx].clone())
        .collect()
}
