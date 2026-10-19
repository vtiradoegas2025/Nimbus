#!/usr/bin/env bash
# setup.sh -- Check and install Nimbus dependencies for the current platform.
#
# Nimbus Studio is the default product: setup installs the complete dependency
# stack, builds the simulator, then builds the native desktop app around that
# exact binary. --headless-only is the explicit secondary terminal workflow.
# The default flow is interactive before it changes the machine; --yes is the
# explicit non-interactive automation path.
#
# Supported platforms: macOS (Homebrew), Ubuntu/Debian/Pop!_OS (apt),
#                      Fedora/RHEL (dnf), Arch Linux (pacman).
#
# Written for bash 3.2 so it runs on a stock macOS /bin/bash.

set -euo pipefail

cd "$(dirname "$0")/.."

# The next-steps output quotes repo-relative build commands, so refuse to run
# from a copy of this script that has been lifted out of the tree.
if [[ ! -f Makefile ]] || [[ ! -f mk/options.mk ]] || \
   [[ ! -f mk/dependencies.mk ]] || [[ ! -d include/core ]]; then
    echo "setup.sh: not inside a Nimbus checkout (looked in $PWD)" >&2
    exit 1
fi

# ── Options ──────────────────────────────────────────────────────────
CHECK_ONLY=false
ASSUME_YES=false
HEADLESS_ONLY=false
DEPENDENCIES_ONLY=false
SHOW_ART=auto
USE_COLOR=auto

usage()
{
    cat <<'USAGE'
Usage: ./scripts/setup.sh [options]

  --check       Report dependency status; install nothing. Exits non-zero if
                any dependency in the complete Nimbus stack is missing. This
                mode is non-interactive.
  -y, --yes     Approve package-manager installation without prompting. Manual
                prerequisites are still reported and must be completed. This
                also accepts the detected platform route.
  --headless-only
                Build the simulator without Nimbus Studio or the legacy
                windowed Vulkan viewer. This is the secondary terminal route.
  --dependencies-only
                Install and verify dependencies but do not build products.
  --no-color    Disable ANSI colour.
  --no-art      Disable the banner artwork.
  --art         Force the banner even when stdout is not a terminal.
  -h, --help    Show this message.

By default the script builds the simulator and Nimbus Studio desktop app after
dependencies are satisfied. It explains the detected platform route, asks
whether to continue, and separately confirms any package-manager changes.
Set NIMBUS_BUILD_JOBS to a positive integer to limit C++ build parallelism.
See docs/BUILDING.md.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)     CHECK_ONLY=true; shift ;;
        --yes|-y)    ASSUME_YES=true; shift ;;
        --headless-only) HEADLESS_ONLY=true; shift ;;
        --dependencies-only) DEPENDENCIES_ONLY=true; shift ;;
        --no-color)  USE_COLOR=false; shift ;;
        --no-art)    SHOW_ART=false; shift ;;
        --art)       SHOW_ART=true; shift ;;
        --help|-h)   usage; exit 0 ;;
        *)
            echo "Unknown option: $1 (try --help)" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${NIMBUS_BUILD_JOBS+x}" ]] && [[ ! "$NIMBUS_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NIMBUS_BUILD_JOBS must be a positive integer." >&2
    exit 1
fi

# ── Presentation ─────────────────────────────────────────────────────
# Colour is on for an interactive terminal unless NO_COLOR is set
# (https://no-color.org). The 256-colour palette degrades to the basic 8.
if [[ "$USE_COLOR" == "auto" ]]; then
    if [[ -n "${NO_COLOR:-}" ]] || [[ ! -t 1 ]]; then
        USE_COLOR=false
    else
        USE_COLOR=true
    fi
fi

if [[ "$SHOW_ART" == "auto" ]]; then
    if [[ -t 1 ]]; then SHOW_ART=true; else SHOW_ART=false; fi
fi

TERM_COLORS=0
if $USE_COLOR; then
    TERM_COLORS="$(tput colors 2>/dev/null || echo 8)"
fi

# Palette. The storm is monochrome -- a greyscale ramp from the bright ice
# canopy aloft down to the shadowed low-level core -- so the one saturated
# colour in the banner, the lightning, is the only thing that draws the eye.
if [[ "$TERM_COLORS" -ge 256 ]]; then
    ANVIL=$'\033[38;5;253m';   CLOUD=$'\033[38;5;249m'
    STORM=$'\033[38;5;246m';   STORM_MID=$'\033[38;5;243m'
    STORM_DARK=$'\033[38;5;240m'
    BOLT=$'\033[38;5;227m';    FUNNEL=$'\033[38;5;244m'
    GROUND=$'\033[38;5;238m'
    MARK=$'\033[38;5;250m';    MARK2=$'\033[38;5;244m'
    RED=$'\033[38;5;203m'; GREEN=$'\033[38;5;77m'; YELLOW=$'\033[38;5;221m'
    DIM=$'\033[38;5;244m'
elif [[ "$TERM_COLORS" -ge 8 ]]; then
    ANVIL=$'\033[1;37m';   CLOUD=$'\033[1;37m'
    STORM=$'\033[0;37m';   STORM_MID=$'\033[0;37m'
    STORM_DARK=$'\033[1;30m'
    BOLT=$'\033[1;33m';    FUNNEL=$'\033[0;37m'
    GROUND=$'\033[1;30m'
    MARK=$'\033[0;37m';    MARK2=$'\033[1;30m'
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
    DIM=$'\033[0;37m'
else
    ANVIL=''; CLOUD=''; STORM=''; STORM_MID=''; STORM_DARK=''
    BOLT=''; FUNNEL=''; GROUND=''; MARK=''; MARK2=''
    RED=''; GREEN=''; YELLOW=''; DIM=''
fi
BOLD=''; RESET=''
if [[ "$TERM_COLORS" -ge 8 ]]; then BOLD=$'\033[1m'; RESET=$'\033[0m'; fi

# Every message goes through printf '%s\n' so that colour codes and payload
# text are never interpreted as a format string.
say()  { printf '%s\n' "$1"; }
ok()   { say "  ${GREEN}ok${RESET}     $1"; }
warn() { say "  ${YELLOW}miss${RESET}   $1"; }
fail() { say "  ${RED}FAIL${RESET}   $1"; }
info() { say "${BOLD}$1${RESET}"; }
note() { say "         ${DIM}$1${RESET}"; }

# The banner uses U+2591..U+2588 block shading, so it only renders where the
# locale is UTF-8. Anywhere else falls back to a pure-ASCII wordmark.
is_utf8()
{
    local enc
    enc="$(locale charmap 2>/dev/null || echo "")"
    [[ "$enc" == "UTF-8" ]] && return 0
    [[ "${LC_ALL:-}${LC_CTYPE:-}${LANG:-}" == *[Uu][Tt][Ff]* ]]
}

# Widest line in each art variant, plus the margin that keeps a full-width
# banner off the right edge. Below ART_MIN_COLS even the compact storm would
# wrap, and wrapped block art is unreadable noise, so it is dropped entirely.
ART_FULL_COLS=56
ART_MIN_COLS=38

# Terminal width at the moment of printing. There is no live redraw: the banner
# is written once and scrolls away, so a later window resize cannot reflow it.
# Choosing the variant here is the only sizing control available.
term_cols()
{
    local cols
    cols="$(tput cols 2>/dev/null || true)"
    if [[ "$cols" =~ ^[0-9]+$ ]] && [[ "$cols" -gt 0 ]]; then
        echo "$cols"
    else
        echo 80
    fi
}

# Supercell cross-section: overshooting top, downshear-spreading anvil, tapering
# updraft tower, wall cloud, and a tornado touching down at the same column the
# ground line darkens. 53 columns wide.
art_storm_full()
{
    say "                         ${ANVIL}▄▄▄▄${RESET}"
    say "                    ${ANVIL}▄▄▄████████▄▄▄${RESET}"
    say "              ${ANVIL}▄▄▄███████████████████▄▄▄▄▄▄▄▄▄${RESET}"
    say "       ${CLOUD}▄▄▄███████████████████████████████████████▄▄${RESET}"
    say "     ${CLOUD}▄██████████████████████████████████████████████▄${RESET}"
    say "    ${STORM}████████████████████████████████${CLOUD}▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀${RESET}"
    say "    ${STORM}███████████████████████████${STORM_MID}▀▀${RESET}      ${BOLT}▟▛${RESET}"
    say "     ${STORM_MID}██████████████████████${STORM_DARK}▀▀${RESET}         ${BOLT}▟▛${RESET}"
    say "      ${STORM_MID}█████████████████${STORM_DARK}▀${RESET}            ${BOLT}▀▜▛▀${RESET}"
    say "       ${STORM_DARK}▀███████████▀${RESET}               ${BOLT}▟▛${RESET}"
    say "         ${STORM_DARK}▀███████▀${RESET}                ${BOLT}▟▘${RESET}"
    say "          ${STORM_DARK}▐█████▌${RESET}"
    say "           ${FUNNEL}▝███▘${RESET}"
    say "            ${FUNNEL}▝█▘${RESET}"
    say "    ${GROUND}▁▂▃▄▅▆▇█${FUNNEL}▒▓▒${GROUND}█▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁${RESET}"
}

# The same storm redrawn for narrow terminals. 32 columns wide.
art_storm_compact()
{
    say "             ${ANVIL}▄▄▄▄${RESET}"
    say "         ${ANVIL}▄▄▄████████▄▄▄${RESET}"
    say "    ${CLOUD}▄▄███████████████▄▄▄▄▄▄▄${RESET}"
    say "  ${CLOUD}▄███████████████████████████▄${RESET}"
    say "  ${STORM}██████████████████${CLOUD}▀▀▀▀▀▀▀▀▀▀${RESET}"
    say "  ${STORM}███████████████${STORM_MID}▀▀${RESET}  ${BOLT}▟▛${RESET}"
    say "   ${STORM_MID}███████████${STORM_DARK}▀▀${RESET}    ${BOLT}▀▜▛${RESET}"
    say "    ${STORM_DARK}▀███████▀${RESET}        ${BOLT}▟▘${RESET}"
    say "      ${STORM_DARK}▀███▀${RESET}"
    say "      ${STORM_DARK}▐███▌${RESET}"
    say "       ${FUNNEL}▝█▘${RESET}"
    say "  ${GROUND}▁▂▃▄▅${FUNNEL}▒▓▒${GROUND}▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁${RESET}"
}

# ANSI-shadow wordmark, 53 columns wide. Needs UTF-8 box-drawing glyphs.
art_wordmark_block()
{
    say "   ${MARK} ███╗   ██╗██╗███╗   ███╗██████╗ ██╗   ██╗███████╗${RESET}"
    say "   ${MARK} ████╗  ██║██║████╗ ████║██╔══██╗██║   ██║██╔════╝${RESET}"
    say "   ${MARK} ██╔██╗ ██║██║██╔████╔██║██████╔╝██║   ██║███████╗${RESET}"
    say "   ${MARK2} ██║╚██╗██║██║██║╚██╔╝██║██╔══██╗██║   ██║╚════██║${RESET}"
    say "   ${MARK2} ██║ ╚████║██║██║ ╚═╝ ██║██████╔╝╚██████╔╝███████║${RESET}"
    say "   ${MARK2} ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚══════╝${RESET}"
}

# Pure-ASCII wordmark, 35 columns wide. Used for narrow terminals and for any
# locale that is not UTF-8.
art_wordmark_ascii()
{
    say "   ${MARK}  _  _ ___ __  __ ___ _   _ ___${RESET}"
    say "   ${MARK} | \\| |_ _|  \\/  | _ ) | | / __|${RESET}"
    say "   ${MARK2} | .\` || || |\\/| | _ \\ |_| \\__ \\${RESET}"
    say "   ${MARK2} |_|\\_|___|_|  |_|___/\\___/|___/${RESET}"
}

banner()
{
    $SHOW_ART || return 0

    local cols
    cols="$(term_cols)"
    [[ "$cols" -lt "$ART_MIN_COLS" ]] && return 0

    say ""
    if is_utf8 && [[ "$cols" -ge "$ART_FULL_COLS" ]]; then
        art_storm_full
        say ""
        art_wordmark_block
        say ""
        say "   ${DIM}A modern approach to supercell modeling.${RESET}"
        say "   ${DIM}Convective-scale atmospheric simulation in C++17.${RESET}"
    else
        if is_utf8; then
            art_storm_compact
            say ""
        fi
        art_wordmark_ascii
        say ""
        say "   ${DIM}Supercell modeling in C++17.${RESET}"
    fi
    say ""
}

# ── Dependency bookkeeping ───────────────────────────────────────────
# Space-separated token lists rather than arrays: bash 3.2 treats an empty
# "${arr[@]}" as an unbound variable under `set -u`.
MISSING_REQUIRED=""
REQUIRED_IMPACTS=""

require_missing() { MISSING_REQUIRED="$MISSING_REQUIRED $1"; }
required_impact() { REQUIRED_IMPACTS="$REQUIRED_IMPACTS|$1"; }

# Word-splitting on the token list is the point here, so the argument is
# intentionally unquoted.
# shellcheck disable=SC2086
count_tokens() { set -- $1; echo $#; }

# plural 1 "dependency" -> "1 dependency"; plural 3 "dependency" -> "3 dependencies"
plural()
{
    local n="$1" noun="$2"
    if [[ "$n" -eq 1 ]]; then
        echo "$n $noun"
    elif [[ "$noun" == *y ]]; then
        echo "$n ${noun%y}ies"
    else
        echo "$n ${noun}s"
    fi
}

check_cmd() { command -v "$1" >/dev/null 2>&1; }

# ── Platform detection ───────────────────────────────────────────────
detect_platform()
{
    local uname_s kernel_release
    uname_s="$(uname -s)"

    case "$uname_s" in
        Darwin)
            PLATFORM="macos"
            PLATFORM_LABEL="macOS ($(uname -m))"
            ;;
        Linux)
            kernel_release="$(uname -r)"
            if check_cmd apt-get; then
                PLATFORM="debian"
                case "$kernel_release" in
                    *[Mm]icrosoft*) PLATFORM_LABEL="WSL 2 / Debian/Ubuntu (apt)" ;;
                    *) PLATFORM_LABEL="Debian/Ubuntu (apt)" ;;
                esac
            elif check_cmd dnf; then
                PLATFORM="fedora"; PLATFORM_LABEL="Fedora/RHEL (dnf)"
            elif check_cmd pacman; then
                PLATFORM="arch"; PLATFORM_LABEL="Arch Linux (pacman)"
            elif $CHECK_ONLY; then
                PLATFORM="linux"; PLATFORM_LABEL="Linux (unrecognized package manager)"
            else
                fail "Unsupported Linux distribution (need apt, dnf, or pacman)"
                echo "     Install the dependencies listed in docs/BUILDING.md by hand,"
                echo "     or re-run with --check to see the dependency report anyway."
                exit 1
            fi
            ;;
        *)
            fail "Unsupported OS: $uname_s"
            echo "     On Windows use WSL 2 with Ubuntu. See docs/BUILDING.md."
            exit 1
            ;;
    esac
}

print_platform_route()
{
    echo ""
    info "Guided setup route"
    case "$PLATFORM" in
        macos)
            say "  Package path: Homebrew, with Xcode Command Line Tools when needed"
            say "  GPU path:     native Metal plus Vulkan through MoltenVK"
            if ! $HEADLESS_ONLY; then
                say "  Studio path:  Node/npm + Rust; macOS supplies the system WebKit view"
            fi
            ;;
        debian)
            say "  Package path: apt packages; Rust remains an explicit rustup action"
            say "  GPU path:     Vulkan headers/loader and the installed hardware driver"
            if ! $HEADLESS_ONLY; then
                say "  Studio path:  Node/npm + Rust + webkit2gtk-4.1 development packages"
            fi
            case "$PLATFORM_LABEL" in
                WSL*) say "  WSL note:     windowed Vulkan requires WSLg (Windows 11 baseline)" ;;
            esac
            ;;
        fedora)
            say "  Package path: dnf packages; Rust remains an explicit rustup action"
            say "  GPU path:     Vulkan headers/loader and the installed hardware driver"
            if ! $HEADLESS_ONLY; then
                say "  Studio path:  Node/npm + Rust + webkit2gtk4.1 development packages"
            fi
            ;;
        arch)
            say "  Package path: pacman; ZFP requires an AUR or local installation"
            say "  GPU path:     Vulkan headers/loader and the installed hardware driver"
            if ! $HEADLESS_ONLY; then
                say "  Studio path:  Node/npm + Rust + webkit2gtk-4.1 development packages"
            fi
            ;;
        linux)
            say "  Package path: manual; no supported package manager was detected"
            say "  Follow:       docs/BUILDING.md, then rerun this script with --check"
            ;;
    esac
    if $HEADLESS_ONLY; then
        say "  Product path: simulator binary only (--headless-only)"
        say "  Launch path:  explicit terminal command with --headless"
    else
        say "  Product path: simulator engine bundled inside Nimbus Studio"
        case "$PLATFORM" in
            macos) say "  Launch path:  /Applications/Nimbus Studio.app" ;;
            *) say "  Launch path:  portable Nimbus Studio AppImage" ;;
        esac
    fi
    if $DEPENDENCIES_ONLY; then
        say "  Build path:   skipped by --dependencies-only"
    elif $HEADLESS_ONLY; then
        say "  Build path:   GNU Make simulator build"
    else
        say "  Build path:   GNU Make engine, then locked npm/Tauri desktop build"
    fi
}

confirm_platform_route()
{
    $CHECK_ONLY && return 0
    if $ASSUME_YES; then
        note "Platform route accepted by --yes."
        return 0
    fi
    if [[ ! -t 0 ]]; then
        fail "Guided setup needs an interactive terminal."
        echo "     Re-run in a terminal, use --check for a read-only report," >&2
        echo "     or use --yes to accept the detected route for automation." >&2
        return 1
    fi

    local reply
    while true; do
        printf '  Continue with this guided setup route? [Y/n] '
        IFS= read -r reply || reply="n"
        case "$reply" in
            ""|y|Y|yes|YES|Yes) return 0 ;;
            n|N|no|NO|No)
                warn "Setup cancelled before checking or changing the machine."
                return 1
                ;;
            *) say "  Please answer yes or no." ;;
        esac
    done
}

# ── Core checks ──────────────────────────────────────────────────────
# Scratch directory for the compile probes below. A directory (rather than a
# bare mktemp file) is what lets the probes carry a .cpp suffix, which every
# compiler needs to pick the C++ frontend.
PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nimbus_setup.XXXXXX")"
cleanup_probe_dir()
{
    [[ -n "$PROBE_DIR" ]] || return 0
    [[ -d "$PROBE_DIR" ]] || return 0
    [[ ! -L "$PROBE_DIR" ]] || return 0
    case "$PROBE_DIR" in
        "${TMPDIR:-/tmp}"/nimbus_setup.*) rm -rf "$PROBE_DIR" ;;
        *) echo "setup.sh: refusing unexpected probe cleanup path: $PROBE_DIR" >&2 ;;
    esac
}
trap cleanup_probe_dir EXIT

host_cxx()
{
    command -v "${CXX:-}" 2>/dev/null \
        || command -v g++ 2>/dev/null \
        || command -v clang++ 2>/dev/null \
        || command -v c++ 2>/dev/null \
        || echo ""
}

# Compile a real C++17 translation unit instead of parsing version strings:
# Apple Clang, LLVM Clang, and GCC all number themselves differently, and what
# the build actually needs is the language level.
check_compiler()
{
    local cxx src
    cxx="$(host_cxx)"
    if [[ -z "$cxx" ]]; then
        warn "C++ compiler (g++ / clang++) not found"
        require_missing compiler
        return
    fi

    src="$PROBE_DIR/cxx17_probe.cpp"
    cat > "$src" <<'CPP'
#include <tuple>
int main() {
    auto [a, b] = std::pair<int, int>{1, 2};
    if constexpr (sizeof(int) > 0) { return a * 0 + b * 0; }
    return 0;
}
CPP
    if "$cxx" -std=c++17 "$src" -o /dev/null 2>/dev/null; then
        ok "C++17 compiler: $("$cxx" --version 2>&1 | head -1)"
    else
        warn "C++ compiler found but rejects -std=c++17: $cxx"
        note "Nimbus needs g++ >= 7.0 or clang++ >= 9.0; override with CXX=..."
        require_missing compiler
    fi
}

check_make()
{
    if check_cmd make; then
        ok "GNU Make: $(make --version 2>&1 | head -1)"
    else
        warn "GNU Make not found"
        require_missing make
    fi
}

# Compiler caching is deliberately optional: it changes iteration speed, not
# the produced artifact. Report the exact wrapper selected by mk/options.mk
# without making a complete-stack setup fail when neither is installed.
check_compiler_cache()
{
    if check_cmd sccache; then
        ok "Compiler cache: $(sccache --version 2>&1 | head -1)"
    elif check_cmd ccache; then
        ok "Compiler cache: $(ccache --version 2>&1 | head -1)"
    else
        note "Compiler cache: not installed (optional; install sccache or ccache)"
    fi
}

# python3 is required, not optional: the default `make` target runs
# scripts/dev/shaders/verify_spirv.py and scripts/dev/shaders/verify_kernel_manifest.py whenever a
# compute shader is recompiled, and the qualification targets are Python.
# Both use the standard library only -- no pip packages needed.
check_python()
{
    if check_cmd python3; then
        if [[ "$PLATFORM" == "macos" ]] && ! $HEADLESS_ONLY &&
           ! python3 -c 'import tarfile; assert hasattr(tarfile, "data_filter")' >/dev/null 2>&1; then
            warn "Python needs safe tar extraction for the macOS runtime build; update Homebrew python3"
            require_missing python3
        else
            ok "Python 3: $(python3 --version 2>&1)"
        fi
    else
        warn "python3 not found (shader manifest verification + qualification tools)"
        require_missing python3
    fi
}

check_macos_cmake()
{
    [[ "$PLATFORM" == "macos" ]] && ! $HEADLESS_ONLY || return 0
    local version=""
    if check_cmd cmake; then version="$(cmake --version | head -1)"; fi
    if [[ "$version" =~ ([0-9]+)\.([0-9]+) ]] &&
       (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 26) )); then
        ok "CMake: $version"
    else
        warn "CMake >= 3.26 is required to build the portable macOS runtime libraries"
        require_missing cmake
    fi
}

check_openmp()
{
    local cxx src compiled=false
    cxx="$(host_cxx)"
    if [[ -z "$cxx" ]]; then
        warn "OpenMP: cannot test (no compiler found)"
        require_missing openmp
        required_impact "OpenMP is required for the qualified multicore simulation path"
        return
    fi

    src="$PROBE_DIR/openmp_probe.cpp"
    cat > "$src" <<'CPP'
#include <omp.h>
int main() { return omp_get_max_threads() > 0 ? 0 : 1; }
CPP

    # GCC and Linux clang accept -fopenmp directly; macOS clang needs Homebrew libomp.
    if "$cxx" -fopenmp "$src" -o /dev/null 2>/dev/null; then
        compiled=true
    elif [[ "$PLATFORM" == "macos" ]]; then
        local libomp
        libomp="$(brew --prefix libomp 2>/dev/null || echo "")"
        if [[ -n "$libomp" ]] && [[ -f "$libomp/include/omp.h" ]]; then
            if "$cxx" -Xpreprocessor -fopenmp -I"$libomp/include" \
                      -L"$libomp/lib" -lomp "$src" -o /dev/null 2>/dev/null; then
                compiled=true
            fi
        fi
    fi

    if $compiled; then
        ok "OpenMP: available"
    else
        warn "OpenMP: not found"
        require_missing openmp
        required_impact "OpenMP is required for the qualified multicore simulation path"
    fi
}

# ── GPU / viewer checks ──────────────────────────────────────────────
check_pkg_config()
{
    if check_cmd pkg-config; then
        ok "pkg-config: $(pkg-config --version)"
    elif [[ "$PLATFORM" == "macos" ]]; then
        ok "pkg-config: not required (Homebrew prefixes are queried directly)"
    else
        warn "pkg-config: not found (used to locate Vulkan/GLFW on Linux)"
        require_missing pkgconfig
        required_impact "pkg-config is required to resolve the Linux GPU and Studio toolchains"
    fi
}

check_vulkan()
{
    local found=false

    if [[ -n "${VULKAN_SDK:-}" ]] && [[ -f "$VULKAN_SDK/include/vulkan/vulkan.h" ]]; then
        found=true
    elif [[ "$PLATFORM" == "macos" ]]; then
        local prefix
        prefix="$(brew --prefix vulkan-headers 2>/dev/null || echo "")"
        [[ -n "$prefix" ]] && [[ -f "$prefix/include/vulkan/vulkan.h" ]] && found=true
    else
        if [[ -f /usr/include/vulkan/vulkan.h ]] || [[ -f /usr/local/include/vulkan/vulkan.h ]]; then
            found=true
        elif check_cmd pkg-config && pkg-config --exists vulkan 2>/dev/null; then
            found=true
        fi
    fi

    if $found; then
        ok "Vulkan headers: found"
    else
        warn "Vulkan headers: not found"
        require_missing vulkan
        required_impact "Vulkan is required for GPU compute and the native 3D viewer"
    fi
}

check_glslang()
{
    if check_cmd glslangValidator; then
        ok "glslangValidator: $(glslangValidator --version 2>&1 | head -1)"
    else
        warn "glslangValidator: not found"
        note "Pre-compiled SPIR-V ships in the repo, so this is only needed to"
        note "rebuild vulkan/shaders/**/*.comp or to run \`make verify-shaders\`."
        require_missing glslang
        required_impact "glslangValidator is required to rebuild and verify the shipped compute shaders"
    fi
}

check_glfw()
{
    local found=false

    if [[ "$PLATFORM" == "macos" ]]; then
        local prefix
        prefix="$(brew --prefix glfw 2>/dev/null || echo "")"
        [[ -n "$prefix" ]] && [[ -f "$prefix/include/GLFW/glfw3.h" ]] && found=true
    else
        if check_cmd pkg-config && pkg-config --exists glfw3 2>/dev/null; then
            found=true
        elif [[ -f /usr/include/GLFW/glfw3.h ]] || [[ -f /usr/local/include/GLFW/glfw3.h ]]; then
            found=true
        fi
    fi

    if $found; then
        ok "GLFW: found"
    else
        warn "GLFW: not found"
        require_missing glfw
        required_impact "GLFW is required for the windowed Vulkan viewer"
    fi
}

# ── Output-format checks ─────────────────────────────────────────────
# The Makefile auto-detects ZFP from brew, ~/.local, /usr/local, or ZFP_PREFIX,
# and validates each candidate by looking for <prefix>/include/zfp.h.
check_zfp()
{
    local prefix
    for prefix in "${ZFP_PREFIX:-}" "$(brew --prefix zfp 2>/dev/null || echo "")" \
                  "$HOME/.local" /usr/local /usr; do
        if [[ -n "$prefix" ]] && [[ -f "$prefix/include/zfp.h" ]]; then
            ok "ZFP: found in $prefix"
            return
        fi
    done

    warn "ZFP: not found"
    require_missing zfp
    required_impact "ZFP is required for compressed high-fidelity history output"
}

# netCDF detection is compile-time in the Makefile via nc-config, so the binary
# reports its true output capability. ncdump backs `make qualify-netcdf-history`.
check_netcdf()
{
    if check_cmd nc-config; then
        ok "netCDF: $(nc-config --version 2>&1 | head -1)"
        if check_cmd ncdump; then
            ok "ncdump: present (netCDF history qualification)"
        else
            warn "ncdump: not found"
            require_missing netcdf
            required_impact "ncdump is required to qualify netCDF history output"
        fi
    else
        warn "netCDF: nc-config not found"
        require_missing netcdf
        required_impact "netCDF is required for the supported self-describing history workflow"
    fi
}

# ── Nimbus Studio checks ─────────────────────────────────────────────
# Studio is the Tauri 2 desktop app in ui/: Rust shell + React/TypeScript
# frontend. See ui/README.md.
NODE_22_MIN=22.13.0
RUST_MIN=1.77.2

# Numeric dotted-version comparison: true when $1 >= $2.
version_ge()
{
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)" == "$2" ]]
}

check_node()
{
    if ! check_cmd node; then
        warn "Node.js: not found (Nimbus Studio frontend)"
        require_missing node
        required_impact "Node.js is required to build and run Nimbus Studio"
        return
    fi

    local version major
    version="$(node --version 2>/dev/null | sed 's/^v//')"
    major="${version%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]] && {
        [[ "$major" -ge 24 ]] || { [[ "$major" -eq 22 ]] && version_ge "$version" "$NODE_22_MIN"; }
    }; then
        ok "Node.js: v$version"
    else
        warn "Node.js: v$version is unsupported; use v${NODE_22_MIN}+ within v22, or v24+"
        note "Distro packages often lag; nvm or NodeSource gives a current runtime."
        require_missing node
        required_impact "Nimbus Studio requires Node.js v${NODE_22_MIN}+ within v22, or v24+"
    fi

    if check_cmd npm; then
        ok "npm: $(npm --version 2>/dev/null)"
    else
        warn "npm: not found"
        require_missing npm
        required_impact "npm is required to install and build Nimbus Studio"
    fi
}

# Rust is deliberately not auto-installed: rustup is the toolchain manager Tauri
# expects, distro `cargo` packages lag behind the required version, and piping a
# remote installer into a shell is not something this script should do silently.
check_rust()
{
    if ! check_cmd cargo; then
        warn "Rust/Cargo: not found (Nimbus Studio shell)"
        require_missing rust
        required_impact "Rust/Cargo is required to build the Nimbus Studio native shell"
        return
    fi

    local version
    version="$(cargo --version 2>/dev/null | awk '{print $2}')"
    if version_ge "$version" "$RUST_MIN"; then
        ok "Rust/Cargo: $version"
    else
        warn "Rust/Cargo: $version is older than the required $RUST_MIN"
        require_missing rust
        required_impact "Rust >= $RUST_MIN is required by the Nimbus Studio native shell"
    fi
}

check_ffmpeg()
{
    if check_cmd ffmpeg; then
        # "ffmpeg version 8.1 Copyright (c) ..." -- keep the version, drop the notice.
        ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
    else
        warn "ffmpeg: not found"
        require_missing ffmpeg
        required_impact "ffmpeg is required for Nimbus Studio movie export"
    fi
}

# Tauri needs a WebKit web view plus a handful of build tools on Linux. macOS
# gets the web view from the system and the toolchain from the Xcode CLT.
check_tauri_system_deps()
{
    [[ "$PLATFORM" == "macos" ]] && return 0

    local found=false
    if check_cmd pkg-config; then
        if pkg-config --exists webkit2gtk-4.1 2>/dev/null; then
            found=true
        fi
    fi

    if $found; then
        ok "webkit2gtk-4.1: found (Tauri web view)"
    else
        warn "webkit2gtk-4.1: not found"
        require_missing tauri-sys
        required_impact "webkit2gtk-4.1 is required by the Nimbus Studio desktop shell on Linux"
    fi
}

# ── Package mapping ──────────────────────────────────────────────────
# Echoes the package names that provide $1 on $PLATFORM, or nothing when the
# platform has no package for it (the caller then prints a manual hint).
packages_for()
{
    case "$PLATFORM:$1" in
        macos:compiler|macos:make)  echo "" ;;          # Xcode Command Line Tools
        macos:python3)  echo "python3" ;;
        macos:cmake)    echo "cmake" ;;
        macos:openmp)   echo "libomp" ;;
        macos:vulkan)   echo "vulkan-headers vulkan-loader molten-vk" ;;
        macos:glslang)  echo "glslang" ;;
        macos:glfw)     echo "glfw" ;;
        macos:pkgconfig) echo "pkg-config" ;;
        macos:zfp)      echo "zfp" ;;
        macos:netcdf)   echo "netcdf" ;;
        macos:node|macos:npm) echo "node" ;;
        macos:ffmpeg)   echo "ffmpeg" ;;
        macos:rust|macos:tauri-sys) echo "" ;;

        debian:compiler) echo "g++ build-essential" ;;
        debian:make)     echo "make" ;;
        debian:python3)  echo "python3" ;;
        debian:openmp)   echo "libomp-dev" ;;
        debian:vulkan)   echo "libvulkan-dev vulkan-tools" ;;
        debian:glslang)  echo "glslang-tools" ;;
        debian:glfw)     echo "libglfw3-dev" ;;
        debian:pkgconfig) echo "pkg-config" ;;
        debian:zfp)      echo "libzfp-dev" ;;
        debian:netcdf)   echo "libnetcdf-dev netcdf-bin" ;;
        debian:node|debian:npm) echo "nodejs npm" ;;
        debian:ffmpeg)   echo "ffmpeg" ;;
        debian:tauri-sys) echo "libwebkit2gtk-4.1-dev librsvg2-dev libssl-dev libayatana-appindicator3-dev patchelf file" ;;
        debian:rust)     echo "" ;;

        fedora:compiler) echo "gcc-c++" ;;
        fedora:make)     echo "make" ;;
        fedora:python3)  echo "python3" ;;
        fedora:openmp)   echo "libomp-devel" ;;
        fedora:vulkan)   echo "vulkan-headers vulkan-loader-devel" ;;
        fedora:glslang)  echo "glslang" ;;
        fedora:glfw)     echo "glfw-devel" ;;
        fedora:pkgconfig) echo "pkgconf-pkg-config" ;;
        fedora:zfp)      echo "zfp-devel" ;;
        fedora:netcdf)   echo "netcdf-devel" ;;
        fedora:node|fedora:npm) echo "nodejs npm" ;;
        fedora:ffmpeg)   echo "ffmpeg-free" ;;
        fedora:tauri-sys) echo "webkit2gtk4.1-devel librsvg2-devel openssl-devel libappindicator-gtk3-devel patchelf file" ;;
        fedora:rust)     echo "" ;;

        arch:compiler)   echo "gcc" ;;
        arch:make)       echo "make" ;;
        arch:python3)    echo "python" ;;
        arch:openmp)     echo "openmp" ;;
        arch:vulkan)     echo "vulkan-headers vulkan-icd-loader" ;;
        arch:glslang)    echo "glslang" ;;
        arch:glfw)       echo "glfw" ;;
        arch:pkgconfig)  echo "pkgconf" ;;
        arch:netcdf)     echo "netcdf" ;;
        arch:node|arch:npm) echo "nodejs npm" ;;
        arch:ffmpeg)     echo "ffmpeg" ;;
        arch:tauri-sys)  echo "webkit2gtk-4.1 librsvg openssl libappindicator-gtk3 patchelf file" ;;
        arch:zfp|arch:rust) echo "" ;;          # zfp lives in the AUR

        *) echo "" ;;
    esac
}

# Instruction for anything the package manager cannot supply.
manual_hint_for()
{
    case "$PLATFORM:$1" in
        macos:compiler|macos:make)
            echo "run 'xcode-select --install', finish Apple's installer, then re-run setup" ;;
        *:rust)
            echo "install rustup: https://rustup.rs" ;;
        arch:zfp)
            echo "ZFP is in the AUR: 'yay -S zfp', or build https://github.com/LLNL/zfp into ~/.local" ;;
        *)
            echo "" ;;
    esac
}

# ── Installation ─────────────────────────────────────────────────────
# Collects packages for every token in $1, prints manual hints for the rest,
# and echoes the package list on stdout.
resolve_packages()
{
    local dep dep_pkgs pkg pkgs=""
    # shellcheck disable=SC2086  # token list, splitting is intended
    for dep in $1; do
        dep_pkgs="$(packages_for "$dep")"
        if [[ -z "$dep_pkgs" ]]; then
            continue
        fi
        # Several deps share a package (node/npm, compiler/make); keep one copy.
        # shellcheck disable=SC2086
        for pkg in $dep_pkgs; do
            case " $pkgs " in
                *" $pkg "*) ;;
                *) pkgs="$pkgs $pkg" ;;
            esac
        done
    done
    echo "${pkgs# }"
}

# The default installation path must be a conversation, not an unattended
# mutation. CI and provisioning can opt in explicitly with --yes.
confirm_package_install()
{
    local pkgs="$1" reply
    [[ -z "${pkgs// /}" ]] && return 0
    $ASSUME_YES && return 0

    if [[ ! -t 0 ]]; then
        fail "Installation needs confirmation, but stdin is not interactive."
        echo "     Re-run in a terminal, use --check for a report, or use --yes for automation."
        return 1
    fi

    echo ""
    info "Action proposed"
    say "  The platform package manager will install:"
    say "  ${BOLD}$pkgs${RESET}"
    printf '  Continue with package installation? [y/N] '
    IFS= read -r reply || reply=""
    case "$reply" in
        y|Y|yes|YES|Yes) return 0 ;;
        *)
            warn "Installation cancelled; no packages were changed."
            return 1
            ;;
    esac
}

print_manual_actions()
{
    local dep dep_pkgs hint printed=""
    # shellcheck disable=SC2086  # token list, splitting is intended
    for dep in $1; do
        dep_pkgs="$(packages_for "$dep")"
        [[ -n "$dep_pkgs" ]] && continue
        hint="$(manual_hint_for "$dep")"
        [[ -z "$hint" ]] && continue
        case " $printed " in
            *" $dep "*) continue ;;
        esac
        say "  ${YELLOW}action${RESET}  $dep: $hint"
        printed="$printed $dep"
    done
}

ensure_package_manager()
{
    local pkgs="$1"
    [[ -z "${pkgs// /}" ]] && return 0
    case "$PLATFORM" in
        macos)
            if ! check_cmd brew; then
                fail "Homebrew is required to install the missing packages."
                echo "     Install it from https://brew.sh, then re-run this script."
                return 1
            fi
            ;;
    esac
}

install_packages()
{
    local pkgs="$1"
    [[ -z "${pkgs// /}" ]] && return 0

    case "$PLATFORM" in
        macos)
            info "Installing via Homebrew: $pkgs"
            # shellcheck disable=SC2086  # deliberate word splitting into argv
            brew install $pkgs
            ;;
        debian)
            info "Installing via apt: $pkgs"
            sudo apt-get update -qq
            # shellcheck disable=SC2086
            sudo apt-get install -y $pkgs
            ;;
        fedora)
            info "Installing via dnf: $pkgs"
            # shellcheck disable=SC2086
            sudo dnf install -y $pkgs
            ;;
        arch)
            info "Installing via pacman: $pkgs"
            # shellcheck disable=SC2086
            sudo pacman -S --needed --noconfirm $pkgs
            ;;
    esac
}

# ── Check driver ─────────────────────────────────────────────────────
run_all_checks()
{
    MISSING_REQUIRED=""
    REQUIRED_IMPACTS=""

    echo ""
    info "Core (required to build and test the simulator)"
    check_compiler
    check_make
    check_python
    check_macos_cmake

    echo ""
    info "Build acceleration (optional)"
    check_compiler_cache

    echo ""
    info "Parallelism (required for the qualified runtime)"
    check_openmp

    echo ""
    if $HEADLESS_ONLY; then
        info "GPU compute (required for the default simulator build)"
    else
        info "GPU compute and native viewer toolchain (required)"
    fi
    check_pkg_config
    check_vulkan
    check_glslang
    if ! $HEADLESS_ONLY; then
        check_glfw
    fi

    echo ""
    info "Scientific output formats (required)"
    check_zfp
    check_netcdf

    if ! $HEADLESS_ONLY; then
        echo ""
        info "Nimbus Studio desktop app, ui/ (default product)"
        check_node
        check_rust
        check_tauri_system_deps
        check_ffmpeg
    fi
}

print_required_impacts()
{
    [[ -z "$REQUIRED_IMPACTS" ]] && return 0
    echo ""
    info "Why action is required"
    local IFS='|' item
    # shellcheck disable=SC2086  # split on the '|' set above
    for item in $REQUIRED_IMPACTS; do
        [[ -n "$item" ]] && say "  ${DIM}-${RESET} $item"
    done
}

processor_count()
{
    local count=""
    if check_cmd nproc; then count="$(nproc 2>/dev/null || true)"; fi
    if [[ -z "$count" ]] && check_cmd getconf; then
        count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi
    if [[ -z "$count" ]] && check_cmd sysctl; then
        count="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]] || count=1
    echo "$count"
}

build_products()
{
    $CHECK_ONLY && return 0
    if $DEPENDENCIES_ONLY; then
        note "Product build skipped by --dependencies-only."
        return 0
    fi

    local jobs
    jobs="${NIMBUS_BUILD_JOBS:-$(processor_count)}"
    echo ""
    info "Building Nimbus simulation engine ($jobs parallel jobs)"
    make -j"$jobs" REQUIRE_FULL_STACK=1

    if $HEADLESS_ONLY; then
        echo ""
        info "Headless Nimbus build complete: $PWD/bin/nimbus_sim"
        return 0
    fi

    echo ""
    info "Installing locked Nimbus Studio dependencies"
    (cd ui && npm ci)

    echo ""
    info "Building Nimbus Studio with the simulator bundled inside"
    (cd ui && npm run tauri:build)
}

print_next_steps()
{
    echo ""
    if $CHECK_ONLY; then
        info "Dependency check complete"
        if $HEADLESS_ONLY; then
            say "  Build the secondary terminal product: ./scripts/setup.sh --headless-only"
        else
            say "  Build the default desktop product: ./scripts/setup.sh"
        fi
    elif $DEPENDENCIES_ONLY; then
        info "Dependencies are ready"
        if $HEADLESS_ONLY; then
            say "  Build the secondary terminal product: ./scripts/setup.sh --headless-only"
        else
            say "  Build the default desktop product: ./scripts/setup.sh"
        fi
    elif $HEADLESS_ONLY; then
        info "Run Nimbus from the terminal (secondary workflow)"
        say ""
        say "  ./bin/nimbus_sim --headless --config=configs/1000meterResolution/kessler/baseline.yaml"
    else
        info "Launch Nimbus Studio (default workflow)"
        case "$PLATFORM" in
            macos)
                say ""
                say "  open '/Applications/Nimbus Studio.app'"
                ;;
            *)
                local artifact
                artifact="$(find ui/src-tauri/target/release/bundle -type f -name '*.AppImage' -print 2>/dev/null | head -1)"
                say ""
                if [[ -n "$artifact" ]]; then
                    say "  '$PWD/$artifact'"
                else
                    say "  Run the AppImage reported by the Studio build above."
                fi
                ;;
        esac
        say ""
        say "  ${DIM}Terminal-only alternative: ./scripts/setup.sh --headless-only${RESET}"
    fi

    cat <<'STEPS'

  make test                                          # run the test suite
  make build-fingerprint                             # capture qualification build identity
STEPS
    say ""
    say "  ${DIM}Full build documentation: docs/BUILDING.md${RESET}"
    say ""
}

# ── Main ─────────────────────────────────────────────────────────────
main()
{
    banner
    detect_platform
    info "Platform: $PLATFORM_LABEL"
    print_platform_route
    confirm_platform_route || return 1

    run_all_checks

    local n_required
    n_required="$(count_tokens "$MISSING_REQUIRED")"

    print_required_impacts

    if [[ "$n_required" -eq 0 ]]; then
        echo ""
        if $HEADLESS_ONLY; then
            info "Nimbus headless dependency stack satisfied."
        else
            info "Complete Nimbus Studio dependency stack satisfied."
        fi
        build_products
        print_next_steps
        return 0
    fi

    if $CHECK_ONLY; then
        echo ""
        fail "Missing $(plural "$n_required" "required dependency")."
        print_manual_actions "$MISSING_REQUIRED"
        echo "     Re-run without --check to install them."
        return 1
    fi

    # ── Install ──
    echo ""
    info "Found $(plural "$n_required" "missing required dependency")"
    local pkgs
    pkgs="$(resolve_packages "$MISSING_REQUIRED")"
    print_manual_actions "$MISSING_REQUIRED"
    ensure_package_manager "$pkgs" || return 1
    confirm_package_install "$pkgs" || return 1
    install_packages "$pkgs"

    # ── Verify ──
    echo ""
    info "Re-checking after installation"
    run_all_checks
    n_required="$(count_tokens "$MISSING_REQUIRED")"
    print_required_impacts

    echo ""
    if [[ "$n_required" -gt 0 ]]; then
        fail "$(plural "$n_required" "required dependency") still missing -- see the messages above."
        print_manual_actions "$MISSING_REQUIRED"
        echo "     Manual installation steps: docs/BUILDING.md"
        return 1
    fi

    if $HEADLESS_ONLY; then
        info "Setup complete. The Nimbus headless dependency stack is available."
    else
        info "Setup complete. The full Nimbus Studio dependency stack is available."
    fi
    build_products
    print_next_steps
    return 0
}

main
