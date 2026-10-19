#!/usr/bin/env bash
set -euo pipefail

PARALLEL_ARGS=(--parallel)
if [[ -n "${NIMBUS_BUILD_JOBS+x}${CMAKE_BUILD_PARALLEL_LEVEL+x}" ]]; then
    BUILD_JOBS="${NIMBUS_BUILD_JOBS-${CMAKE_BUILD_PARALLEL_LEVEL-}}"
    if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Configured build worker count must be a positive integer." >&2
        exit 1
    fi
    PARALLEL_ARGS+=("$BUILD_JOBS")
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PREFIX="${NIMBUS_WINDOWS_PREFIX:-$ROOT_DIR/build/windows-prefix}"
WORK_DIR="$ROOT_DIR/build/windows-deps/zfp-1.0.1"
ARCHIVE="$ROOT_DIR/build/windows-deps/zfp-1.0.1.tar.gz"
URL="https://github.com/LLNL/zfp/releases/download/1.0.1/zfp-1.0.1.tar.gz"
SHA256="ca0f7b4ae88044ffdda12faead30723fe83dd8f5bb0db74125df84589e60e52b"

if [[ -f "$PREFIX/include/zfp.h" && -f "$PREFIX/lib/libzfp.a" ]]; then
    printf 'ZFP already installed at %s\n' "$PREFIX"
    exit 0
fi

mkdir -p "$(dirname "$ARCHIVE")" "$PREFIX"
if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 3 --output "$ARCHIVE" "$URL"
fi
printf '%s  %s\n' "$SHA256" "$ARCHIVE" | sha256sum --check --status || {
    printf 'ZFP archive digest mismatch: %s\n' "$ARCHIVE" >&2
    exit 1
}

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$WORK_DIR"
cmake -S "$WORK_DIR" -B "$WORK_DIR/build-nimbus" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DZFP_WITH_OPENMP=ON
cmake --build "$WORK_DIR/build-nimbus" "${PARALLEL_ARGS[@]}"
cmake --install "$WORK_DIR/build-nimbus"
