#!/usr/bin/env python3
"""Generate HEVC CABAC context-init tables from vendored FFmpeg.

Parses libavcodec/hevc/cabac.c — the `init_values[3][199]` table and the
`CABAC_ELEMS` element list (name + bin count) — and emits Rust:
  - INIT_VALUES: [[u8; 199]; 3]  (indexed by init_type)
  - per-element context-base offset constants (cumulative bin counts)

Run from the repo root:
  python3 scripts/gen_hevc_cabac_tables.py > crates/mv-extract/hevc_cabac_tables.rs
"""
import re
import sys

SRC = "ffmpeg/FFmpeg-8.0-custom/FFmpeg/libavcodec/hevc/cabac.c"
CNU = 154


def strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def main() -> None:
    text = open(SRC).read()

    # ── element offsets from CABAC_ELEMS(ELEM); their bin-count sum == #contexts
    #    (HEVC_CONTEXTS=199 in FFmpeg is an over-allocation; only the sum is real)
    em = re.search(r"#define CABAC_ELEMS\(ELEM\)(.*?)\n\n", text, re.S)
    if not em:
        sys.exit("CABAC_ELEMS macro not found")
    elems = re.findall(r"ELEM\((\w+),\s*(\d+)\)", em.group(1))
    offsets = []
    off = 0
    for name, nbins in elems:
        offsets.append((name, off))
        off += int(nbins)
    n_ctx = off

    # ── init_values[3][n_ctx] ──
    m = re.search(r"init_values\[3\]\[HEVC_CONTEXTS\]\s*=\s*\{(.*?)\n\};", text, re.S)
    if not m:
        sys.exit("init_values table not found")
    body = strip_comments(m.group(1))
    groups = re.findall(r"\{([^{}]*)\}", body)
    if len(groups) != 3:
        sys.exit(f"expected 3 init_type groups, got {len(groups)}")
    tables = []
    for gi, g in enumerate(groups):
        vals = [CNU if t == "CNU" else int(t) for t in g.replace(",", " ").split()]
        if len(vals) != n_ctx:
            sys.exit(f"init_type {gi}: {len(vals)} values != {n_ctx} contexts")
        tables.append(vals)
    HEVC_CONTEXTS = n_ctx

    # ── emit ──
    out = []
    out.append("//! HEVC CABAC context-init values + element offsets, generated from")
    out.append("//! FFmpeg libavcodec/hevc/cabac.c. DO NOT EDIT BY HAND.")
    out.append("//! Regenerate with scripts/gen_hevc_cabac_tables.py.")
    out.append("")
    out.append(f"pub const HEVC_CONTEXTS: usize = {HEVC_CONTEXTS};")
    out.append("")
    out.append("/// init_values[init_type][ctxIdx]; init_type = I:0 P:1 B:2 (pre cabac_init swap).")
    out.append(f"pub static INIT_VALUES: [[u8; {HEVC_CONTEXTS}]; 3] = [")
    for vals in tables:
        out.append("    [")
        for i in range(0, len(vals), 16):
            out.append("        " + ",".join(str(v) for v in vals[i : i + 16]) + ",")
        out.append("    ],")
    out.append("];")
    out.append("")
    out.append("// Context-base offset (ctxIdx 0) for each syntax element.")
    for name, o in offsets:
        out.append(f"pub const {name}_OFFSET: usize = {o};")
    print("\n".join(out))


if __name__ == "__main__":
    main()
