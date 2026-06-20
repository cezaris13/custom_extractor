# Standalone build for the custom from-scratch MV extractor.
#
# ffmpeg-sys-next links against whatever FFmpeg pkg-config finds. Override
# FFMPEG_PREFIX to point at a specific install (its lib/pkgconfig is prepended
# to PKG_CONFIG_PATH and baked as an rpath); leave it empty to use the system
# FFmpeg already on pkg-config's path.
#   make                       # release build against system FFmpeg
#   make FFMPEG_PREFIX=/opt/ff # release build against a specific FFmpeg
#   make run ARGS="in.mp4 out.csv"
FFMPEG_PREFIX ?=

ifneq ($(FFMPEG_PREFIX),)
export PKG_CONFIG_PATH := $(FFMPEG_PREFIX)/lib/pkgconfig:$(PKG_CONFIG_PATH)
export RUSTFLAGS := -C link-arg=-Wl,-rpath,$(FFMPEG_PREFIX)/lib -C link-arg=-Wl,--disable-new-dtags
endif

.PHONY: build run clean
build:
	cargo build --release --bin extractor

run: build
	cargo run --release --bin extractor -- $(ARGS)

clean:
	cargo clean
