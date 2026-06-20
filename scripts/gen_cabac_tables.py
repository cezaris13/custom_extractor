#!/usr/bin/env python3
"""Generate crates/mv-extract/thesis_cabac_tables.rs from FFmpeg source.

Extracts the CABAC engine tables (ff_h264_cabac_tables) and the context-init
tables (cabac_context_init_I / _PB) verbatim, so they can't drift from a
hand-transcription. Run from the repo root after changing the custom FFmpeg.
"""
import os
import re

LAC = "ffmpeg/FFmpeg-8.0-custom/FFmpeg/libavcodec"
OUT = "crates/mv-extract/thesis_cabac_tables.rs"


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def extract_array(path, name):
    src = open(path).read()
    b = src.index("{", src.index(name))
    depth, j = 0, b
    while True:
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return [int(x) for x in re.findall(r"-?\d+", strip_comments(src[b:j + 1]))]


def main():
    cab = extract_array(f"{LAC}/cabac.c", "ff_h264_cabac_tables")
    ci = extract_array(f"{LAC}/h264_cabac.c", "cabac_context_init_I")
    pb = extract_array(f"{LAC}/h264_cabac.c", "cabac_context_init_PB")
    assert len(cab) == 1343, len(cab)
    assert len(ci) == 2048, len(ci)
    assert len(pb) == 6144, len(pb)

    out = [
        "//! CABAC engine + context-init tables, generated from FFmpeg's",
        "//! libavcodec/cabac.c and h264_cabac.c. DO NOT EDIT BY HAND.",
        "//! Regenerate with scripts/gen_cabac_tables.py if FFmpeg changes.",
    ]

    def emit(name, vals, ty="u8"):
        out.append(f"pub static {name}: [{ty}; {len(vals)}] = [")
        for k in range(0, len(vals), 16):
            out.append("    " + ",".join(str(v) for v in vals[k:k + 16]) + ",")
        out.append("];")

    # ff_h264_cabac_tables is uint8_t in C; some mlps_state entries are written
    # as negative literals (e.g. -128) that convert to uint8_t. Mask to match.
    u8 = lambda v: [x & 0xFF for x in v]
    emit("NORM_SHIFT", u8(cab[0:512]))
    emit("LPS_RANGE", u8(cab[512:1024]))
    emit("MLPS_STATE", u8(cab[1024:1280]))
    emit("CTX_INIT_I", ci, "i8")
    emit("CTX_INIT_PB", pb, "i8")
    # 8x8-transform significance offsets (High profile). last-coeff-flag offsets
    # live in ff_h264_cabac_tables[1280..1343]; the per-position significance
    # offsets are a separate frame-mode table in h264_cabac.c.
    emit("LAST_COEFF_8X8", u8(cab[1280:1343]))
    sig8 = extract_array(f"{LAC}/h264_cabac.c", "significant_coeff_flag_offset_8x8")
    emit("SIG_OFF_8X8", u8(sig8[0:63]))  # row 0 = frame (non-MBAFF)
    open(OUT, "w").write("\n".join(out) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
