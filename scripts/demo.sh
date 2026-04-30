#!/usr/bin/env bash
# demo.sh -- One-command supercell simulation + live 3D visualization
#
# Usage:
#   ./scripts/demo.sh                          # default: supercell_30min, cinematic B&W
#   ./scripts/demo.sh --config configs/student/student_cartesian.yaml
#   ./scripts/demo.sh --style default           # color instead of B&W
#   ./scripts/demo.sh --record                  # record video (requires ffmpeg)
#   ./scripts/demo.sh --preset quick            # 5-min sim, fast preview
#   ./scripts/demo.sh --preset beast            # overnight production run (headless)
#   ./scripts/demo.sh --headless-only           # sim only, no viewer
#
# The script starts the simulation in the background, waits for shared
# memory to initialize, then launches the volumetric viewer. When the
# viewer window is closed, the simulation is automatically stopped.
# In --headless-only mode, the simulation runs in the foreground until
# completion or Ctrl+C.
#
# Video recording uses ffmpeg's avfoundation capture. Install with:
#   brew install ffmpeg

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Defaults ─────────────────────────────────────────────────────────
CONFIG="configs/teaching/supercell_30min.yaml"
STYLE="cinematic-bw"
VOLUME_MODE="supercell"
CAMERA_MODE="orbit"
RAY_STEPS=256
SHM_FIELDS="qc,qr"
RECORD=false
HEADLESS_ONLY=false
RECORD_OUTPUT="demo_$(date +%Y%m%d_%H%M%S).mp4"
DURATION=""
EXTRA_SIM_ARGS=""
EXTRA_VIEWER_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --style)        STYLE="$2"; shift 2 ;;
        --volume-mode)  VOLUME_MODE="$2"; shift 2 ;;
        --camera-mode)  CAMERA_MODE="$2"; shift 2 ;;
        --ray-steps)    RAY_STEPS="$2"; shift 2 ;;
        --fields)       SHM_FIELDS="$2"; shift 2 ;;
        --record)       RECORD=true; shift ;;
        --record-output) RECORD_OUTPUT="$2"; shift 2 ;;
        --duration)     DURATION="$2"; shift 2 ;;
        --headless-only) HEADLESS_ONLY=true; shift ;;
        --preset)
            case "$2" in
                quick)
                    CONFIG="configs/student/student_cartesian.yaml"
                    DURATION="300"
                    RAY_STEPS=128
                    SHM_FIELDS="qc,qr"
                    ;;
                supercell)
                    CONFIG="configs/teaching/supercell_30min.yaml"
                    ;;
                production)
                    CONFIG="configs/simulation/production.yaml"
                    RAY_STEPS=384
                    ;;
                beast)
                    CONFIG="configs/simulation/production.yaml"
                    ;;
                tornado)
                    CONFIG="configs/teaching/tornado_genesis.yaml"
                    ;;
                *)
                    echo "Unknown preset: $2 (available: quick, supercell, production, beast, tornado)"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --help|-h)
            head -16 "$0" | tail -14
            echo ""
            echo "Presets: quick, supercell (default), production, beast, tornado"
            echo "Styles: cinematic-bw (default), default (color)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1
            ;;
    esac
done

# ── Build ────────────────────────────────────────────────────────────
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "Building simulation + GPU shaders..."
make -j"$NPROC" 2>&1 | tail -1
if [[ ! -f "bin/tornado_sim" ]]; then
    echo "Error: bin/tornado_sim failed to build."
    exit 1
fi

if ! $HEADLESS_ONLY; then
    echo "Building viewer..."
    make -C vulkan -j"$NPROC" 2>&1 | tail -1
    if [[ ! -f "bin/vulkan_viewer" ]]; then
        echo "Error: bin/vulkan_viewer failed to build."
        exit 1
    fi
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found: $CONFIG"
    exit 1
fi
if $RECORD && ! command -v ffmpeg &>/dev/null; then
    echo "Error: --record requires ffmpeg. Install with: brew install ffmpeg"
    exit 1
fi

# ── Cleanup handler ──────────────────────────────────────────────────
SIM_PID=""
FFMPEG_PID=""
cleanup() {
    if [[ -n "$FFMPEG_PID" ]]; then
        kill "$FFMPEG_PID" 2>/dev/null || true
        wait "$FFMPEG_PID" 2>/dev/null || true
        echo "Recording saved: $RECORD_OUTPUT"
    fi
    if [[ -n "$SIM_PID" ]]; then
        kill "$SIM_PID" 2>/dev/null || true
        wait "$SIM_PID" 2>/dev/null || true
    fi
    echo "Demo stopped."
}
trap cleanup EXIT

# ── Build simulation command ─────────────────────────────────────────
SIM_CMD=(
    ./bin/tornado_sim --headless
    "--config=$CONFIG"
)
if ! $HEADLESS_ONLY; then
    SIM_CMD+=(--live-shm "--live-shm-fields=$SHM_FIELDS")
fi
if [[ -n "$DURATION" ]]; then
    SIM_CMD+=("--duration=$DURATION")
fi

# ── Build viewer command ─────────────────────────────────────────────
VIEWER_CMD=(
    ./bin/vulkan_viewer
    --window-test
    --window-width 1920
    --window-height 1080
    --render-backend volume
    --volume-mode "$VOLUME_MODE"
    --style "$STYLE"
    --camera-mode "$CAMERA_MODE"
    --camera-distance 1.8
    --camera-height 0.4
    --ray-steps "$RAY_STEPS"
    --ray-threshold 0.01
    --ray-opacity 1.2
    --ray-brightness 1.8
    --ray-ambient 1.0
    --ray-anisotropy 0.45
    --fields "$SHM_FIELDS"
)

# ── Launch ───────────────────────────────────────────────────────────
CASE_NAME=$(grep 'case_name:' "$CONFIG" 2>/dev/null | awk '{print $2}' || basename "$CONFIG" .yaml)
echo "============================================"
echo "  Nimbus Demo: $CASE_NAME"
echo "============================================"
echo "Config:  $CONFIG"
echo "Mode:    $($HEADLESS_ONLY && echo "headless-only" || echo "sim + viewer")"
echo "Style:   $STYLE"
echo "Fields:  $SHM_FIELDS"
echo "Record:  $RECORD"
echo ""

SIM_LOG="data/demo_sim_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data

if $HEADLESS_ONLY; then
    # Headless-only: run simulation in foreground until completion or Ctrl+C
    echo "Starting simulation (headless-only)..."
    echo "Output: data/"
    echo "Press Ctrl+C to stop."
    echo ""
    "${SIM_CMD[@]}" 2>&1 | tee "$SIM_LOG"
    echo ""
    echo "Simulation complete."
else
    echo "Starting simulation (log: $SIM_LOG)..."
    "${SIM_CMD[@]}" > "$SIM_LOG" 2>&1 &
    SIM_PID=$!

    # Wait for simulation to finish initialization and open SHM
    echo "Waiting for simulation to initialize..."
    for i in $(seq 1 60); do
        if ! kill -0 "$SIM_PID" 2>/dev/null; then
            echo "Error: simulation exited during initialization. Check $SIM_LOG"
            SIM_PID=""
            exit 1
        fi
        if grep -q "ADVECTION" "$SIM_LOG" 2>/dev/null; then
            break
        fi
        sleep 0.25
    done
    echo "Simulation running (PID $SIM_PID)"

    # Start video recording if requested
    if $RECORD; then
        echo "Recording to: $RECORD_OUTPUT"
        if [[ "$(uname -s)" == "Darwin" ]]; then
            # Auto-detect screen capture device (avoid grabbing the webcam)
            SCREEN_IDX=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 \
                         | grep -i "capture screen" | head -1 \
                         | sed 's/.*\[\([0-9]*\)\].*/\1/')
            if [[ -z "$SCREEN_IDX" ]]; then
                echo "Warning: could not detect screen capture device, skipping recording."
                RECORD=false
            else
                echo "  Screen capture device: [$SCREEN_IDX]"
                ffmpeg -f avfoundation -framerate 30 -capture_cursor 0 \
                       -i "${SCREEN_IDX}:none" -vcodec libx264 -preset fast -crf 18 \
                       -pix_fmt yuv420p "$RECORD_OUTPUT" </dev/null &>/dev/null &
                FFMPEG_PID=$!
            fi
        else
            ffmpeg -f x11grab -framerate 30 -video_size 1920x1080 \
                   -i "${DISPLAY:-:0}" -vcodec libx264 -preset fast -crf 18 \
                   -pix_fmt yuv420p "$RECORD_OUTPUT" </dev/null &>/dev/null &
            FFMPEG_PID=$!
        fi
    fi

    # Launch viewer in foreground (blocks until window closed)
    echo "Launching viewer... (close window or Ctrl+C to stop)"
    echo ""
    "${VIEWER_CMD[@]}" || true

    echo ""
    echo "Viewer closed."
fi
