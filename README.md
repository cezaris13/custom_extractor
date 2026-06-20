# custom_extractor

From-scratch compressed-domain motion-vector extractor (`extractor9`).

Parses the H.264 / HEVC bitstream directly — NAL splitting, parameter sets,
slice headers, CAVLC/CABAC entropy decoding, MV prediction — and recovers
motion vectors **without** invoking libavcodec's decode path. This is the
thesis approach (Angheluta, PoliTo 2020/21, "Efficient Extraction of Motion
Vectors from H264 Video Streams", Ch. 4), extended with a from-scratch HEVC
decoder that reuses the H.264 bit reader and CABAC engine.

libavformat (via `ffmpeg-sys-next`) is used **only** for container demux.

## Layout

    crates/mv-types     shared MV types (vendored from the parent repo)
    crates/mv-extract   the decoder library + `extractor9` binary
      thesis*.rs        H.264 path: bitreader, CAVLC, CABAC, slice/MV decode
      hevc*.rs          HEVC path
      ffmpeg_common.rs  container demux + MV output writer
    scripts/            CABAC init-table generators

## Build

    cargo build --release        # binary at target/release/extractor9

`ffmpeg-sys-next` links against whatever FFmpeg `pkg-config` finds; point
`PKG_CONFIG_PATH` at the desired install if it isn't the system one.
