#!/usr/bin/env bash
# demo.sh -- One-command supercell simulation + live 3D visualization
#
# Usage:
#   ./scripts/demo.sh                          # default: supercell_30min, cinematic B&W
#   ./scripts/demo.sh --config configs/student_cartesian.yaml
#   ./scripts/demo.sh --style default           # color instead of B&W
#   ./scripts/demo.sh --record                  # record video (requires ffmpeg)
#   ./scripts/demo.sh --preset quick            # 5-min sim, fast preview
#
# The script starts the simulation in the background, waits for shared
# memory to initialize, then launches the volumetric viewer. When the
# viewer window is closed, the simulation is automatically stopped.
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
SHM_FIELDS="w,theta,qc,qr,vorticity_z"
RECORD=false
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
        --preset)
            case "$2" in
                quick)
                    CONFIG="configs/student_cartesian.yaml"
                    DURATION="300"
                    RAY_STEPS=128
                    ;;
                supercell)
                    CONFIG="configs/teaching/supercell_30min.yaml"
                    ;;
                production)
                    CONFIG="configs/physical_supercell.yaml"
                    RAY_STEPS=384
                    ;;
                tornado)
                    CONFIG="configs/teaching/tornado_genesis.yaml"
                    ;;
                *)
                    echo "Unknown preset: $2 (available: quick, supercell, production, tornado)"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --help|-h)
            head -14 "$0" | tail -12
            echo ""
            echo "Presets: quick, supercell (default), production, tornado"
            echo "Styles: cinematic-bw (default), default (color)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1
            ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────
if [[ ! -f "bin/tornado_sim" ]]; then
    echo "Error: bin/tornado_sim not found. Run 'make' first."
    exit 1
fi
if [[ ! -f "bin/vulkan_viewer" ]]; then
    echo "Error: bin/vulkan_viewer not found. Run 'make vulkan' first."
    exit 1
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
    --live-shm
    "--live-shm-fields=$SHM_FIELDS"
)
if [[ -n "$DURATION" ]]; then
    SIM_CMD+=("--duration=$DURATION")
fi

# ── Build viewer command ─────────────────────────────────────────────
VIEWER_CMD=(
    ./bin/vulkan_viewer
    --render-backend volume
    "--volume-mode=$VOLUME_MODE"
    "--style=$STYLE"
    "--camera-mode=$CAMERA_MODE"
    "--ray-steps=$RAY_STEPS"
    "--fields=$SHM_FIELDS"
)

# ── Launch ───────────────────────────────────────────────────────────
CASE_NAME=$(grep 'case_name:' "$CONFIG" 2>/dev/null | awk '{print $2}' || basename "$CONFIG" .yaml)
echo "============================================"
echo "  Nimbus Demo: $CASE_NAME"
echo "============================================"
echo "Config:  $CONFIG"
echo "Style:   $STYLE"
echo "Fields:  $SHM_FIELDS"
echo "Record:  $RECORD"
echo ""

SIM_LOG="data/demo_sim_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data

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
    if grep -q "step=" "$SIM_LOG" 2>/dev/null; then
        break
    fi
    sleep 0.25
done
echo "Simulation running (PID $SIM_PID)"

# Start video recording if requested
if $RECORD; then
    echo "Recording to: $RECORD_OUTPUT"
    # Capture screen 0 (main display) at 30 fps. The viewer window will be
    # the primary content. Crop/trim in post if needed.
    ffmpeg -f avfoundation -framerate 30 -capture_cursor 0 \
           -i "1:none" -vcodec libx264 -preset fast -crf 18 \
           -pix_fmt yuv420p "$RECORD_OUTPUT" </dev/null &>/dev/null &
    FFMPEG_PID=$!
fi

# Launch viewer in foreground (blocks until window closed)
echo "Launching viewer... (close window or Ctrl+C to stop)"
echo ""
"${VIEWER_CMD[@]}" || true

echo ""
echo "Viewer closed."
