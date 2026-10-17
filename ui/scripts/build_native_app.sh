#!/usr/bin/env bash
# Build Nimbus Studio for the current supported desktop platform. On macOS the
# verified bundle atomically replaces the one canonical /Applications copy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
UI_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
ROOT_DIR="$(cd "$UI_DIR/.." && pwd -P)"
cd "$UI_DIR"

BUNDLE_ONLY=false
BUNDLE_DIR=""
PRINT_PLAN=false

usage()
{
    cat <<'USAGE'
Usage: bash scripts/build_native_app.sh [options]

  --bundle-only  Build the native bundle without installing it. This is the
                 default behavior on Linux and useful for macOS release jobs.
  --bundle-dir DIR  Copy and sign the macOS bundle in this directory without
                 installing it; useful outside Desktop file-provider folders.
  --print-plan   Print the detected bundle/install route without building.
  -h, --help     Show this message.

On macOS, this builds pinned runtime libraries and a portable simulator for
macOS 11 before bundling. NIMBUS_MACOS_RUNTIME_WORK selects the dependency cache;
NIMBUS_BUILD_JOBS sets its build worker count (default 8).
On other platforms the simulator must already exist at ../bin/nimbus_sim.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle-only) BUNDLE_ONLY=true; shift ;;
        --bundle-dir) BUNDLE_DIR="${2:?missing bundle directory}"; BUNDLE_ONLY=true; shift 2 ;;
        --print-plan) PRINT_PLAN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

if [[ -n "${NIMBUS_BUILD_JOBS+x}" ]] && [[ ! "$NIMBUS_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NIMBUS_BUILD_JOBS must be a positive integer." >&2
    exit 1
fi

HOST_UNAME="$(uname -s)"
case "$HOST_UNAME" in
    Darwin)
        PLATFORM="macos"
        BUNDLES="${NIMBUS_STUDIO_BUNDLES:-app}"
        ;;
    Linux)
        PLATFORM="linux"
        # AppImage is the one portable desktop artifact shared by Debian,
        # Fedora, Arch and WSLg. Distro packages remain release-channel work.
        BUNDLES="${NIMBUS_STUDIO_BUNDLES:-appimage}"
        BUNDLE_ONLY=true
        ;;
    MINGW*|MSYS*|CYGWIN*)
        [[ -z "$BUNDLE_DIR" ]] || { echo "--bundle-dir requires macOS." >&2; exit 1; }
        args=()
        $BUNDLE_ONLY && args+=("-BundleOnly")
        $PRINT_PLAN && args+=("-PrintPlan")
        exec powershell.exe -NoProfile -ExecutionPolicy Bypass \
            -File "$UI_DIR/scripts/build_native_app.ps1" "${args[@]}"
        ;;
    *)
        echo "Unsupported desktop build host: $HOST_UNAME." >&2
        echo "Nimbus supports macOS, Linux, and native Windows." >&2
        exit 1
        ;;
esac

if [[ -n "$BUNDLE_DIR" && "$PLATFORM" != "macos" ]]; then
    echo "--bundle-dir requires a macOS application bundle." >&2
    exit 1
fi

SIMULATOR="$ROOT_DIR/bin/nimbus_sim"
if [[ "$PLATFORM" != "macos" ]] && { [[ ! -f "$SIMULATOR" ]] || [[ ! -x "$SIMULATOR" ]]; }; then
    echo "Nimbus simulator is not built or executable: $SIMULATOR" >&2
    echo "Run 'make -j' from the repository root before building Studio." >&2
    exit 1
fi

if $PRINT_PLAN; then
    printf 'platform=%s\nbundles=%s\ninstall=%s\nsimulator=%s\n' \
        "$PLATFORM" "$BUNDLES" "$([[ "$PLATFORM" == "macos" ]] && ! $BUNDLE_ONLY && echo yes || echo no)" "$SIMULATOR"
    [[ -z "$BUNDLE_DIR" ]] || printf 'bundle_directory=%s\n' "$BUNDLE_DIR"
    exit 0
fi

if [[ "$PLATFORM" == "macos" ]]; then
    [[ "$BUNDLES" == "app" ]] || {
        echo "The verified macOS runtime must be assembled as an app before creating other release containers." >&2
        exit 1
    }
    export MACOSX_DEPLOYMENT_TARGET=11.0
    RUNTIME_WORK="${NIMBUS_MACOS_RUNTIME_WORK:-$ROOT_DIR/build/macos-runtime}"
    BUILD_JOBS="${NIMBUS_BUILD_JOBS:-8}"
    python3 "$ROOT_DIR/scripts/dependencies/build_macos_runtime.py" \
        --work "$RUNTIME_WORK" --jobs "$BUILD_JOBS"
    RUNTIME_WORK="$(cd "$RUNTIME_WORK" && pwd -P)"
    RUNTIME_PREFIX="$RUNTIME_WORK/prefix"
    MOLTENVK_PREFIX="${NIMBUS_MOLTENVK_PREFIX:-$(brew --prefix molten-vk)}"
    [[ -f "$MOLTENVK_PREFIX/lib/libMoltenVK.dylib" && -f "$MOLTENVK_PREFIX/LICENSE" ]] || {
        echo "MoltenVK runtime and distribution license are required: $MOLTENVK_PREFIX" >&2
        exit 1
    }
    make -C "$ROOT_DIR" -j"$BUILD_JOBS" PORTABLE=1 DISTRIBUTION=1 REQUIRE_FULL_STACK=1 \
        MACOSX_DEPLOYMENT_TARGET=11.0 BUILDDIR=build/macos-release BIN=bin/nimbus_sim_macos \
        LIBOMP_PATH="$RUNTIME_PREFIX" ZFP_PREFIX="$RUNTIME_PREFIX" \
        NETCDF_CONFIG="$RUNTIME_PREFIX/bin/nc-config"
    cp "$ROOT_DIR/bin/nimbus_sim_macos" "$SIMULATOR"
fi

SIMULATOR_CAPABILITIES="$($SIMULATOR --version)"
for required in \
    "OpenMP runtime: available" \
    "ZFP compression: available" \
    "netCDF4 history output: available" \
    "Vulkan compute: available" \
    "live shared memory: available"
do
    if ! grep -Fq "$required" <<<"$SIMULATOR_CAPABILITIES"; then
        echo "Refusing to package a reduced simulator: missing '$required'." >&2
        echo "Build through scripts/setup.sh so the complete product contract is enforced." >&2
        exit 1
    fi
done

npx tauri build --bundles "$BUNDLES"

if [[ "$PLATFORM" == "linux" ]]; then
    artifact="$(find "$UI_DIR/src-tauri/target/release/bundle" -type f -name '*.AppImage' -print | head -1)"
    if [[ -z "$artifact" ]] || [[ ! -f "$artifact" ]]; then
        echo "Tauri completed without producing the expected AppImage." >&2
        exit 1
    fi
    chmod +x "$artifact"
    echo "Nimbus Studio AppImage: $artifact"
    exit 0
fi

SOURCE_APP="$UI_DIR/src-tauri/target/release/bundle/macos/Nimbus Studio.app"
SOURCE_ICON="$UI_DIR/src-tauri/icons/icon.icns"
INSTALL_PARENT="/Applications"
INSTALL_APP="$INSTALL_PARENT/Nimbus Studio.app"
BUNDLE_ID="dev.nimbus.studio"

bundle_id()
{
    /usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$1/Contents/Info.plist" 2>/dev/null || true
}

validate_bundle()
{
    local app="$1" expected_parent="$2" resolved_parent
    [[ -n "$app" ]] || { echo "Refusing an empty app path." >&2; return 1; }
    [[ "$app" == "$expected_parent/Nimbus Studio.app" ]] || {
        echo "Refusing unexpected app path: $app" >&2
        return 1
    }
    [[ -d "$app" ]] || { echo "App bundle is missing: $app" >&2; return 1; }
    [[ ! -L "$app" ]] || { echo "Refusing symlinked app bundle: $app" >&2; return 1; }
    resolved_parent="$(cd "$app/.." && pwd -P)"
    [[ "$resolved_parent" == "$expected_parent" ]] || {
        echo "App bundle escaped its expected parent: $app" >&2
        return 1
    }
    [[ "$(bundle_id "$app")" == "$BUNDLE_ID" ]] || {
        echo "Refusing bundle with an unexpected identifier: $app" >&2
        return 1
    }
}

validate_embedded_payload()
{
    local app="$1" bundled_icon bundled_simulator bundled_configs
    bundled_icon="$app/Contents/Resources/icon.icns"
    bundled_simulator="$app/Contents/Resources/bin/nimbus_sim"
    bundled_configs="$app/Contents/Resources/configs"
    if [[ ! -f "$bundled_icon" ]] || ! cmp -s "$SOURCE_ICON" "$bundled_icon"; then
        echo "Bundled icon does not match the Nimbus source artwork: $bundled_icon" >&2
        return 1
    fi
    if [[ ! -f "$bundled_simulator" ]] || [[ ! -x "$bundled_simulator" ]]; then
        echo "Bundled simulator is missing or not executable: $bundled_simulator" >&2
        return 1
    fi
    if [[ ! -f "$bundled_configs/500meterResolution/kessler/baseline.yaml" ]]; then
        echo "Bundled configuration catalog is missing or empty: $bundled_configs" >&2
        return 1
    fi
    python3 "$SCRIPT_DIR/macos_runtime.py" verify --app "$app" \
        --simulator "$SIMULATOR" --minimum-macos 11.0
}

strip_unsealed_bundle_metadata()
{
    local app="$1"
    command -v xattr >/dev/null 2>&1 || return 0
    # Desktop/iCloud workspaces can immediately attach these unsealed root
    # attributes. They are not application data and make strict verification
    # reject an otherwise valid bundle.
    xattr -d 'com.apple.fileprovider.fpfs#P' "$app" 2>/dev/null || true
    xattr -d com.apple.FinderInfo "$app" 2>/dev/null || true
    xattr -d com.apple.ResourceFork "$app" 2>/dev/null || true
}

validate_bundle "$SOURCE_APP" "$UI_DIR/src-tauri/target/release/bundle/macos"
python3 "$SCRIPT_DIR/macos_runtime.py" prepare --app "$SOURCE_APP" \
    --simulator "$SIMULATOR" --runtime-prefix "$RUNTIME_PREFIX" \
    --extra-library "$MOLTENVK_PREFIX/lib/libMoltenVK.dylib" --minimum-macos 11.0
ditto "$RUNTIME_PREFIX/share/nimbus-runtime-licenses" "$SOURCE_APP/Contents/Resources/runtime-licenses"
mkdir -p "$SOURCE_APP/Contents/Resources/runtime-licenses/MoltenVK"
cp "$MOLTENVK_PREFIX/LICENSE" "$SOURCE_APP/Contents/Resources/runtime-licenses/MoltenVK/LICENSE"
cp "$RUNTIME_PREFIX/nimbus-runtime-build.json" "$SOURCE_APP/Contents/Resources/macos-runtime-build.json"
validate_embedded_payload "$SOURCE_APP"

if $BUNDLE_ONLY; then
    if [[ -n "$BUNDLE_DIR" ]]; then
        mkdir -p "$BUNDLE_DIR"
        bundle_parent="$(cd "$BUNDLE_DIR" && pwd -P)"
        bundle_destination="$bundle_parent/Nimbus Studio.app"
        [[ ! -e "$bundle_destination" && ! -L "$bundle_destination" ]] || {
            echo "Refusing to overwrite an existing bundle: $bundle_destination" >&2
            exit 1
        }
        ditto --norsrc "$SOURCE_APP" "$bundle_destination"
        SOURCE_APP="$bundle_destination"
        validate_bundle "$SOURCE_APP" "$bundle_parent"
        validate_embedded_payload "$SOURCE_APP"
    fi
    # A locally linked Mach-O carries only a linker signature, which does not
    # seal Info.plist or bundled resources. Preserve a valid release identity
    # when Tauri supplied one; otherwise give CI/local artifacts a complete
    # ad-hoc bundle signature so the uploaded .app is internally verifiable.
    strip_unsealed_bundle_metadata "$SOURCE_APP"
    if ! codesign --verify --deep --strict "$SOURCE_APP" >/dev/null 2>&1; then
        codesign --force --deep --sign "${APPLE_SIGNING_IDENTITY:--}" "$SOURCE_APP"
    fi
    strip_unsealed_bundle_metadata "$SOURCE_APP"
    codesign --verify --deep --strict "$SOURCE_APP"
    echo "Nimbus Studio bundle: $SOURCE_APP"
    exit 0
fi

[[ -d "$INSTALL_PARENT" ]] || { echo "Missing install directory: $INSTALL_PARENT" >&2; exit 1; }
[[ ! -L "$INSTALL_PARENT" ]] || { echo "Refusing symlinked install directory: $INSTALL_PARENT" >&2; exit 1; }

if [[ -e "$INSTALL_APP" ]]; then
    validate_bundle "$INSTALL_APP" "$INSTALL_PARENT"
fi

INSTALL_TMP="$(mktemp -d "$INSTALL_PARENT/.nimbus-studio-install.XXXXXX")"
NEW_APP="$INSTALL_TMP/Nimbus Studio.app"
BACKUP_DIR="$INSTALL_TMP/previous"
OLD_APP="$BACKUP_DIR/Nimbus Studio.app"

cleanup_install_tmp()
{
    if [[ -d "$INSTALL_TMP" ]] && [[ ! -L "$INSTALL_TMP" ]] &&
       [[ "$(cd "$INSTALL_TMP/.." && pwd -P)" == "$INSTALL_PARENT" ]] &&
       [[ "$(basename "$INSTALL_TMP")" == .nimbus-studio-install.* ]]; then
        rm -rf "$INSTALL_TMP"
    fi
}
trap cleanup_install_tmp EXIT

# --norsrc deliberately leaves FinderInfo/resource-fork metadata behind, so
# codesign receives clean files without recursively deleting unrelated bundle
# attributes. A unique build number plus a fresh icon timestamp gives
# IconServices an identity that actually changes between local rebuilds.
ditto --norsrc "$SOURCE_APP" "$NEW_APP"
validate_bundle "$NEW_APP" "$INSTALL_TMP"
validate_embedded_payload "$NEW_APP"
BUILD_NUMBER="${NIMBUS_STUDIO_BUILD_NUMBER:-$(date -u +%s)}"
case "$BUILD_NUMBER" in
    ""|*[!0-9.]*) echo "Invalid NIMBUS_STUDIO_BUILD_NUMBER: $BUILD_NUMBER" >&2; exit 1 ;;
esac
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $BUILD_NUMBER" "$NEW_APP/Contents/Info.plist"
touch "$NEW_APP/Contents/Resources/icon.icns" "$NEW_APP"
codesign --force --deep --sign "${APPLE_SIGNING_IDENTITY:--}" "$NEW_APP"
codesign --verify --deep --strict "$NEW_APP"

# Ask the running copy to close cleanly only after its verified replacement is
# ready. If it remains open, stop instead of forcing termination and risking
# user work.
if pgrep -f '/Nimbus Studio\.app/Contents/MacOS/nimbus-studio$' >/dev/null 2>&1; then
    osascript -e 'tell application id "dev.nimbus.studio" to quit' >/dev/null 2>&1 || true
    attempts=0
    while pgrep -f '/Nimbus Studio\.app/Contents/MacOS/nimbus-studio$' >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ "$attempts" -ge 50 ]]; then
            echo "Nimbus Studio is still running. Close it normally, then rebuild again." >&2
            exit 1
        fi
        sleep 0.1
    done
fi

if [[ -d "$INSTALL_APP" ]]; then
    mkdir "$BACKUP_DIR"
    mv "$INSTALL_APP" "$OLD_APP"
fi

if ! mv "$NEW_APP" "$INSTALL_APP"; then
    [[ -d "$OLD_APP" ]] && mv "$OLD_APP" "$INSTALL_APP"
    echo "Could not install the rebuilt Nimbus Studio bundle." >&2
    exit 1
fi

if ! codesign --verify --deep --strict "$INSTALL_APP" || ! validate_embedded_payload "$INSTALL_APP"; then
    rm -rf "$INSTALL_APP"
    [[ -d "$OLD_APP" ]] && mv "$OLD_APP" "$INSTALL_APP"
    echo "Installed bundle failed verification; the previous copy was restored." >&2
    exit 1
fi

if [[ -d "$OLD_APP" ]]; then
    validate_bundle "$OLD_APP" "$BACKUP_DIR"
    rm -rf "$OLD_APP"
fi

# Register the new build identity instead of restarting Finder or Dock and
# disrupting the desktop session. The bundle version and icon mtime above make
# this registration invalidate stale IconServices entries.
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
if [[ -x "$LSREGISTER" ]]; then
    "$LSREGISTER" -f "$INSTALL_APP" >/dev/null 2>&1 || true
fi
touch "$INSTALL_APP"

# The Tauri bundle is only a staging artifact. Keeping it is what made macOS
# index two different Nimbus Studio versions, so remove it after installation.
validate_bundle "$SOURCE_APP" "$UI_DIR/src-tauri/target/release/bundle/macos"
rm -rf "$SOURCE_APP"

echo "Installed one canonical Nimbus Studio app: $INSTALL_APP"
