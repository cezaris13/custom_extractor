use crate::motion_vector::{MotionVector, load_motion_vectors};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

// Ord/PartialOrd are only needed for the final `diffs.sort_by` — the
// first_map/second_map lookups below use Hash+Eq (HashMap), not ordering.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub frame: i32,
    pub src_x: i64,
    pub src_y: i64,
    pub dst_x: i64,
    pub dst_y: i64,
    pub source: i32,
}

impl Key {
    pub fn from_mv(mv: &MotionVector) -> Self {
        Key {
            frame: mv.frame,
            src_x: mv.src_x as i64,
            src_y: mv.src_y as i64,
            dst_x: mv.dst_x as i64,
            dst_y: mv.dst_y as i64,
            source: mv.source,
        }
    }

    pub fn display(&self) -> String {
        format!(
            "Frame {} src=({},{}) dst=({},{}) source={}",
            self.frame, self.src_x, self.src_y, self.dst_x, self.dst_y, self.source
        )
    }
}

/// Zero-size vector: source and destination points coincide (no displacement).
/// `compare_frames` skips these on both sides; callers that report row counts
/// alongside its diff count must apply the same filter or the two disagree.
pub fn is_zero_size(mv: &MotionVector) -> bool {
    mv.src_x == mv.dst_x && mv.src_y == mv.dst_y
}

/// Compare the rows sharing one `Key` between the two files, field by field.
/// Takes slices directly (no intermediate per-row `Vec<(&str, f64)>`
/// allocation) since there are only ever four fixed float columns to check.
fn compare_rows(key: &Key, f_rows: &[&MotionVector], s_rows: &[&MotionVector], diffs: &mut Vec<(Key, String)>) {
    let count = f_rows.len().max(s_rows.len());
    for i in 0..count {
        match (f_rows.get(i), s_rows.get(i)) {
            (None, Some(_)) => diffs.push((
                key.clone(),
                format!("{}: extra row in second file (index {})", key.display(), i),
            )),
            (Some(_), None) => diffs.push((
                key.clone(),
                format!("{}: extra row in first file (index {})", key.display(), i),
            )),
            (Some(f), Some(s)) => {
                let mut cmp = |name: &str, v1: f64, v2: f64| {
                    if v1 != v2 {
                        diffs.push((
                            key.clone(),
                            format!("{}: '{}' differs (first={}, second={})", key.display(), name, v1, v2),
                        ));
                    }
                };
                cmp("src_x", f.src_x, s.src_x);
                cmp("src_y", f.src_y, s.src_y);
                cmp("dst_x", f.dst_x, s.dst_x);
                cmp("dst_y", f.dst_y, s.dst_y);
            }
            (None, None) => unreachable!(),
        }
    }
}

pub fn compare_frames(first: &[MotionVector], second: &[MotionVector]) -> Vec<(Key, String)> {
    // HashMap, not BTreeMap: nothing here depends on sorted iteration order —
    // `diffs` gets sorted explicitly at the end — so O(1) average lookup beats
    // the O(log n) tree traversal for large vector counts.
    let mut first_map: HashMap<Key, Vec<&MotionVector>> = HashMap::new();
    for mv in first.iter().filter(|mv| !is_zero_size(mv)) {
        first_map.entry(Key::from_mv(mv)).or_default().push(mv);
    }

    let mut second_map: HashMap<Key, Vec<&MotionVector>> = HashMap::new();
    for mv in second.iter().filter(|mv| !is_zero_size(mv)) {
        second_map.entry(Key::from_mv(mv)).or_default().push(mv);
    }

    let mut diffs: Vec<(Key, String)> = Vec::new();

    // Two passes over the two maps directly, instead of first collecting a
    // third union-of-keys map and then re-looking-up both maps per key.
    for (key, f_rows) in &first_map {
        match second_map.get(key) {
            None => diffs.push((key.clone(), format!("{}: missing in second file", key.display()))),
            Some(s_rows) => compare_rows(key, f_rows, s_rows, &mut diffs),
        }
    }
    for (key, _) in &second_map {
        if !first_map.contains_key(key) {
            diffs.push((key.clone(), format!("{}: missing in first file", key.display())));
        }
    }

    diffs.sort_by(|a, b| a.0.cmp(&b.0));
    diffs
}

pub fn write_results(differences: &[(Key, String)], output_path: &Path) -> Result<(), String> {
    let mut content = String::new();
    if differences.is_empty() {
        writeln!(content, "No differences found in frames in all frames.").unwrap();
    } else {
        for (_, msg) in differences {
            writeln!(content, "{}", msg).unwrap();
        }
    }
    fs::write(output_path, content).map_err(|e| format!("Failed to write output: {}", e))
}

pub fn compare(
    first_csv: &str,
    second_csv: &str,
    output_file: &str,
) -> Result<(), String> {
    let first = load_motion_vectors(first_csv)
        .map_err(|e| format!("Error loading first CSV: {}", e))?;
    let second = load_motion_vectors(second_csv)
        .map_err(|e| format!("Error loading second CSV: {}", e))?;

    let differences = compare_frames(&first, &second);
    let output_path = Path::new(output_file);

    write_results(&differences, output_path)?;

    println!(
        "Comparison complete. Results written to {}",
        output_path.display()
    );
    Ok(())
}
