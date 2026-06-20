use crate::motion_vector::{MotionVector, load_motion_vectors};
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
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

struct ValueColumns {
    src_x: f64,
    src_y: f64,
    dst_x: f64,
    dst_y: f64,
}

impl ValueColumns {
    fn from_mv(mv: &MotionVector) -> Self {
        ValueColumns {
            src_x: mv.src_x,
            src_y: mv.src_y,
            dst_x: mv.dst_x,
            dst_y: mv.dst_y,
        }
    }

    fn fields(&self) -> Vec<(&str, f64)> {
        vec![
            ("src_x", self.src_x),
            ("src_y", self.src_y),
            ("dst_x", self.dst_x),
            ("dst_y", self.dst_y),
        ]
    }
}

pub fn compare_frames(first: &[MotionVector], second: &[MotionVector]) -> Vec<(Key, String)> {
    let mut first_map: BTreeMap<Key, Vec<&MotionVector>> = BTreeMap::new();
    for mv in first {
        first_map.entry(Key::from_mv(mv)).or_default().push(mv);
    }

    let mut second_map: BTreeMap<Key, Vec<&MotionVector>> = BTreeMap::new();
    for mv in second {
        second_map.entry(Key::from_mv(mv)).or_default().push(mv);
    }

    let mut diffs: Vec<(Key, String)> = Vec::new();

    let all_keys: BTreeMap<&Key, ()> = first_map
        .keys()
        .chain(second_map.keys())
        .map(|k| (k, ()))
        .collect();

    for key in all_keys.keys() {
        let first_rows = first_map.get(key);
        let second_rows = second_map.get(key);

        match (first_rows, second_rows) {
            (Some(_), None) => {
                diffs.push((
                    (*key).clone(),
                    format!("{}: missing in second file", key.display()),
                ));
            }
            (None, Some(_)) => {
                diffs.push((
                    (*key).clone(),
                    format!("{}: missing in first file", key.display()),
                ));
            }
            (Some(f_rows), Some(s_rows)) => {
                let count = f_rows.len().max(s_rows.len());
                for i in 0..count {
                    if i >= f_rows.len() {
                        diffs.push((
                            (*key).clone(),
                            format!("{}: extra row in second file (index {})", key.display(), i),
                        ));
                        continue;
                    }
                    if i >= s_rows.len() {
                        diffs.push((
                            (*key).clone(),
                            format!("{}: extra row in first file (index {})", key.display(), i),
                        ));
                        continue;
                    }
                    let f_vals = ValueColumns::from_mv(f_rows[i]);
                    let s_vals = ValueColumns::from_mv(s_rows[i]);
                    for ((col, v1), (_, v2)) in f_vals.fields().iter().zip(s_vals.fields().iter()) {
                        if v1 != v2 {
                            diffs.push((
                                (*key).clone(),
                                format!(
                                    "{}: '{}' differs (first={}, second={})",
                                    key.display(),
                                    col,
                                    v1,
                                    v2
                                ),
                            ));
                        }
                    }
                }
            }
            (None, None) => unreachable!(),
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
