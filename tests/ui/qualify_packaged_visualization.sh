#!/usr/bin/env bash
set -euo pipefail

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../ui" && pwd)"
PROFILE="smoke"
MODE="debug"
OUTPUT="$UI_DIR/build/qualification/packaged-visualization.json"
PROGRESS=""
SKIP_BUILD=0
APP_BUNDLE_OVERRIDE=""
PYTHON_CHECK=0
CONFIG_PREFLIGHT_INPUT=""
TIMEOUT_SECONDS=180

while (($#)); do
  case "$1" in
    --profile) PROFILE="${2:?missing profile}"; shift 2 ;;
    --mode) MODE="${2:?missing mode}"; shift 2 ;;
    --output) OUTPUT="${2:?missing output}"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="${2:?missing timeout}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --app-bundle) APP_BUNDLE_OVERRIDE="${2:?missing app bundle}"; SKIP_BUILD=1; shift 2 ;;
    --python) PYTHON_CHECK=1; shift ;;
    --config-preflight) CONFIG_PREFLIGHT_INPUT="${2:?missing configuration batch}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$APP_BUNDLE_OVERRIDE" && "$(uname -s)" != "Darwin" ]]; then
  echo "--app-bundle requires a macOS application bundle." >&2
  exit 2
fi

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "production" && "$PROFILE" != "startup" ]]; then
  echo "Profile must be smoke, production or startup." >&2
  exit 2
fi
if [[ "$PROFILE" == "startup" && ( "$PYTHON_CHECK" != 0 || -n "$CONFIG_PREFLIGHT_INPUT" ) ]]; then
  echo "Startup checks opening the app; use smoke or production for Python/configuration checks." >&2
  exit 2
fi
if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
  echo "Mode must be debug or release." >&2
  exit 2
fi
if ! [[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Timeout must be a positive integer." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUTPUT")"
if [[ -n "$CONFIG_PREFLIGHT_INPUT" ]]; then
  CONFIG_PREFLIGHT_INPUT="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$CONFIG_PREFLIGHT_INPUT")"
fi
PROGRESS="${OUTPUT}.progress.ndjson"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing evidence: $OUTPUT" >&2
  exit 2
fi
if [[ -e "$PROGRESS" ]]; then
  echo "Refusing to reuse existing progress evidence: $PROGRESS" >&2
  exit 2
fi

cd "$UI_DIR"
if [[ "$SKIP_BUILD" -eq 0 ]]; then
  BUILD_ARGS=(build --ci)
  if [[ "$MODE" == "debug" ]]; then
    BUILD_ARGS+=(--debug)
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    BUILD_ARGS+=(--bundles app)
  else
    BUILD_ARGS+=(--no-bundle)
  fi
  npx tauri "${BUILD_ARGS[@]}"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  APP_BUNDLE="${APP_BUNDLE_OVERRIDE:-$UI_DIR/src-tauri/target/$MODE/bundle/macos/Nimbus Studio.app}"
  if [[ -z "$APP_BUNDLE_OVERRIDE" && "$SKIP_BUILD" -eq 1 && ! -x "$APP_BUNDLE/Contents/MacOS/nimbus-studio" \
      && -x "/Applications/Nimbus Studio.app/Contents/MacOS/nimbus-studio" ]]; then
    APP_BUNDLE="/Applications/Nimbus Studio.app"
  fi
  BINARY="$APP_BUNDLE/Contents/MacOS/nimbus-studio"
else
  APP_BUNDLE=""
  BINARY="$UI_DIR/src-tauri/target/$MODE/nimbus-studio"
fi
if [[ ! -x "$BINARY" ]]; then
  echo "Packaged qualification binary is missing: $BINARY" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  open -n -W \
    --env "NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT=$OUTPUT" \
    --env "NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE=$PROFILE" \
    --env "NIMBUS_STUDIO_PYTHON_CHECK=$PYTHON_CHECK" \
    --env "NIMBUS_STUDIO_CONFIG_QUALIFICATION_INPUT=$CONFIG_PREFLIGHT_INPUT" \
    "$APP_BUNDLE" &
  # `open -W` is the owned launcher process. The native PID is published in
  # the progress stream and termination resolves through `terminate_packaged_app`.
  for _attempt in $(seq 1 50); do
    if kill -0 "$!" 2>/dev/null; then
      osascript -e 'tell application id "dev.nimbus.studio" to activate' >/dev/null 2>&1 || true
      break
    fi
    sleep 0.1
  done
else
  NIMBUS_STUDIO_PYTHON_CHECK="$PYTHON_CHECK" \
  NIMBUS_STUDIO_CONFIG_QUALIFICATION_INPUT="$CONFIG_PREFLIGHT_INPUT" \
  NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT="$OUTPUT" \
  NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE="$PROFILE" \
  NIMBUS_STUDIO_WHOLE_APP_SESSION=1 \
    setsid "$BINARY" &
fi
APP_PID=$!
ACTIVATOR_PID=""
NATIVE_PID=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  (
    while kill -0 "$APP_PID" 2>/dev/null; do
      osascript -e 'tell application id "dev.nimbus.studio" to activate' >/dev/null 2>&1 || true
      sleep 0.2
    done
  ) &
  ACTIVATOR_PID=$!
fi

stop_activator() {
  if [[ -n "$ACTIVATOR_PID" ]]; then
    kill "$ACTIVATOR_PID" 2>/dev/null || true
    wait "$ACTIVATOR_PID" 2>/dev/null || true
    ACTIVATOR_PID=""
  fi
}

terminate_native_from_progress() {
  if [[ -f "$PROGRESS" ]]; then
    NATIVE_PID="$(tail -1 "$PROGRESS" | sed -E 's/.*"pid":([0-9]+).*/\1/')"
    if [[ "$NATIVE_PID" =~ ^[1-9][0-9]*$ ]]; then
      kill "$NATIVE_PID" 2>/dev/null || true
      for _attempt in $(seq 1 20); do
        if ! kill -0 "$NATIVE_PID" 2>/dev/null; then
          return
        fi
        sleep 0.1
      done
      kill -KILL "$NATIVE_PID" 2>/dev/null || true
    fi
  fi
}

terminate_packaged_app() {
  terminate_native_from_progress
  kill "$APP_PID" 2>/dev/null || true
  stop_activator
}

deadline=$((SECONDS + TIMEOUT_SECONDS))
while kill -0 "$APP_PID" 2>/dev/null; do
  if ((SECONDS >= deadline)); then
    terminate_packaged_app
    wait "$APP_PID" 2>/dev/null || true
    echo "Packaged visualization qualification timed out after ${TIMEOUT_SECONDS}s." >&2
    if [[ -f "$PROGRESS" ]]; then
      echo "Last packaged visualization progress events:" >&2
      tail -20 "$PROGRESS" >&2
    fi
    exit 1
  fi
  sleep 1
done

set +e
wait "$APP_PID"
APP_STATUS=$?
set -e
stop_activator
if [[ "$(uname -s)" == "Darwin" ]]; then
  terminate_native_from_progress
fi
if [[ ! -f "$OUTPUT" ]]; then
  echo "Packaged application exited without writing qualification evidence (status $APP_STATUS)." >&2
  if [[ -f "$PROGRESS" ]]; then
    echo "Last packaged visualization progress events:" >&2
    tail -20 "$PROGRESS" >&2
  fi
  exit 1
fi

case "$(uname -s)" in
  Darwin) TARGET_OS="macos" ;;
  Linux) TARGET_OS="linux" ;;
  *) echo "Unsupported packaged qualification OS: $(uname -s)" >&2; exit 2 ;;
esac
case "$(uname -m)" in
  arm64|aarch64) TARGET_ARCH="aarch64" ;;
  x86_64|amd64) TARGET_ARCH="x86_64" ;;
  *) echo "Unsupported packaged qualification architecture: $(uname -m)" >&2; exit 2 ;;
esac
if [[ "$PROFILE" == "startup" ]]; then
  node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
    ../tests/ui/release/validate_packaged_startup_evidence.ts "$OUTPUT" "$MODE" "$TARGET_OS" "$TARGET_ARCH"
else
  node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
    ../tests/ui/release/validate_packaged_visualization_evidence.ts \
    "$OUTPUT" --profile "$PROFILE" --mode "$MODE" \
    --target-os "$TARGET_OS" --target-arch "$TARGET_ARCH" --require-python "$PYTHON_CHECK" \
    --config-preflight "$CONFIG_PREFLIGHT_INPUT"
fi

if [[ "$APP_STATUS" -ne 0 ]]; then
  echo "Packaged application reported failure status $APP_STATUS." >&2
  exit "$APP_STATUS"
fi
rm -f "$PROGRESS"
