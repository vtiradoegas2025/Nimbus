#!/usr/bin/env bash
# setup.sh -- Install Nimbus dependencies for the current platform
#
# Usage:
#   ./scripts/setup.sh              # install core + viewer dependencies
#   ./scripts/setup.sh --headless   # skip Vulkan/viewer dependencies
#   ./scripts/setup.sh --check      # check what is installed without changing anything
#
# Supported platforms:
#   macOS        (Homebrew)
#   Ubuntu/Debian/Pop!_OS  (apt)
#   Fedora/RHEL  (dnf)
#   Arch Linux   (pacman)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
HEADLESS=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)   HEADLESS=true; shift ;;
        --check)      CHECK_ONLY=true; shift ;;
        --help|-h)
            head -12 "$0" | tail -10
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1
            ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

ok()   { printf "${GREEN}[OK]${RESET}    %s\n" "$1"; }
warn() { printf "${YELLOW}[MISS]${RESET}  %s\n" "$1"; }
fail() { printf "${RED}[FAIL]${RESET}  %s\n" "$1"; }
info() { printf "${BOLD}==> %s${RESET}\n" "$1"; }

MISSING_CORE=()
MISSING_VIEWER=()

check_cmd() 
{
    command -v "$1" &>/dev/null
}

# ── Detect platform ──────────────────────────────────────────────────
detect_platform() 
{
    local uname_s
    uname_s="$(uname -s)"

    case "$uname_s" in
        Darwin)
            PLATFORM="macos"
            if ! check_cmd brew; then
                fail "Homebrew not found. Install from https://brew.sh"
                exit 1
            fi
            ;;
        Linux)
            if check_cmd apt-get; then
                PLATFORM="debian"
            elif check_cmd dnf; then
                PLATFORM="fedora"
            elif check_cmd pacman; then
                PLATFORM="arch"
            else
                fail "Unsupported Linux distribution (need apt, dnf, or pacman)"
                exit 1
            fi
            ;;
        *)
            fail "Unsupported OS: $uname_s"
            echo "On Windows, use WSL 2 with Ubuntu. See BUILDING.md for details."
            exit 1
            ;;
    esac

    info "Detected platform: $PLATFORM"
}

# ── Check functions ──────────────────────────────────────────────────
check_compiler() 
{
    if check_cmd g++ || check_cmd clang++; then
        local cxx
        cxx="$(command -v c++ 2>/dev/null || command -v g++ 2>/dev/null || command -v clang++ 2>/dev/null)"
        local version
        version="$($cxx --version 2>&1 | head -1)"
        ok "C++ compiler: $version"
    else
        warn "C++ compiler (g++ or clang++) not found"
        MISSING_CORE+=("compiler")
    fi
}

check_make() 
{
    if check_cmd make; then
        ok "GNU Make: $(make --version 2>&1 | head -1)"
    else
        warn "GNU Make not found"
        MISSING_CORE+=("make")
    fi
}

check_openmp() 
{
    local cxx
    cxx="$(command -v c++ 2>/dev/null || command -v g++ 2>/dev/null || command -v clang++ 2>/dev/null || echo "")"
    if [[ -z "$cxx" ]]; then
        warn "OpenMP: cannot test (no compiler found)"
        MISSING_CORE+=("openmp")
        return
    fi

    local test_src
    test_src=$(mktemp /tmp/omp_test.XXXXXX.cpp)
    cat > "$test_src" <<'CPP'
#include <omp.h>
int main() { return omp_get_max_threads(); }
CPP

    local compiled=false
    # Try standard -fopenmp first (GCC, Linux clang)
    if $cxx -fopenmp "$test_src" -o /dev/null 2>/dev/null; then
        compiled=true
    fi
    # Try macOS clang with Homebrew libomp
    if ! $compiled && [[ "$PLATFORM" == "macos" ]]; then
        local libomp_prefix
        libomp_prefix="$(brew --prefix libomp 2>/dev/null || echo "")"
        if [[ -n "$libomp_prefix" ]]; then
            if $cxx -Xpreprocessor -fopenmp -I"$libomp_prefix/include" -L"$libomp_prefix/lib" -lomp "$test_src" -o /dev/null 2>/dev/null; then
                compiled=true
            fi
        fi
    fi
    rm -f "$test_src"

    if $compiled; then
        ok "OpenMP: available"
    else
        warn "OpenMP: not available (simulation will run single-threaded)"
        MISSING_CORE+=("openmp")
    fi
}

check_zfp() 
{
    local found=false

    if [[ -f "$HOME/.local/include/zfp.h" ]]; then
        found=true
        ok "ZFP: found in ~/.local"
    elif [[ -f "/usr/local/include/zfp.h" ]]; then
        found=true
        ok "ZFP: found in /usr/local"
    elif [[ -f "/usr/include/zfp.h" ]]; then
        found=true
        ok "ZFP: found in /usr"
    fi

    if ! $found; then
        warn "ZFP: not found (required for compressed output)"
        MISSING_CORE+=("zfp")
    fi
}

check_pkg_config() 
{
    if check_cmd pkg-config; then
        ok "pkg-config: $(pkg-config --version)"
    elif [[ "$PLATFORM" != "macos" ]]; then
        # pkg-config is only important on Linux; macOS uses Homebrew prefix detection
        warn "pkg-config not found (needed for Vulkan/GLFW detection on Linux)"
        MISSING_VIEWER+=("pkg-config")
    else
        ok "pkg-config: not needed (Homebrew handles paths)"
    fi
}

check_vulkan() 
{

    local found=false

    # Check for vulkan.h header
    if [[ "$PLATFORM" == "macos" ]]; then
        local vk_prefix
        vk_prefix="$(brew --prefix vulkan-headers 2>/dev/null || echo "")"
        if [[ -n "$vk_prefix" ]] && [[ -f "$vk_prefix/include/vulkan/vulkan.h" ]]; then
            found=true
        fi
    else
        if [[ -f /usr/include/vulkan/vulkan.h ]] || [[ -f /usr/local/include/vulkan/vulkan.h ]]; then
            found=true
        fi
        # Also check via pkg-config
        if ! $found && check_cmd pkg-config && pkg-config --exists vulkan 2>/dev/null; then
            found=true
        fi
    fi

    if $found; then
        ok "Vulkan SDK: headers found"
    else
        warn "Vulkan SDK: headers not found"
        MISSING_VIEWER+=("vulkan")
    fi
}

check_glslang() 
{
    if check_cmd glslangValidator; then
        ok "glslangValidator: $(glslangValidator --version 2>&1 | head -1)"
    else
        warn "glslangValidator not found (needed to compile GPU shaders)"
        MISSING_VIEWER+=("glslang")
    fi
}

check_glfw() 
{
    local found=false

    if [[ "$PLATFORM" == "macos" ]]; then
        local glfw_prefix
        glfw_prefix="$(brew --prefix glfw 2>/dev/null || echo "")"
        if [[ -n "$glfw_prefix" ]] && [[ -f "$glfw_prefix/include/GLFW/glfw3.h" ]]; then
            found=true
        fi
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
        warn "GLFW: not found (windowed viewer requires GLFW)"
        MISSING_VIEWER+=("glfw")
    fi
}

# ── Install functions ────────────────────────────────────────────────
install_macos() 
{
    local pkgs=()

    for dep in "${MISSING_CORE[@]}"; do
        case "$dep" in
            compiler) ;; # Xcode CLT handled separately
            make)     ;; # Comes with Xcode CLT
            openmp)   pkgs+=(libomp) ;;
        esac
    done

    if ! $HEADLESS; then
        for dep in "${MISSING_VIEWER[@]}"; do
            case "$dep" in
                vulkan)     pkgs+=(vulkan-headers vulkan-loader molten-vk) ;;
                glslang)    pkgs+=(glslang) ;;
                glfw)       pkgs+=(glfw) ;;
                pkg-config) pkgs+=(pkg-config) ;;
            esac
        done
    fi

    # Xcode CLT for compiler + make
    for dep in "${MISSING_CORE[@]}"; do
        if [[ "$dep" == "compiler" ]] || [[ "$dep" == "make" ]]; then
            info "Installing Xcode Command Line Tools..."
            xcode-select --install 2>/dev/null || true
            echo "    If a dialog appeared, complete the installation and re-run this script."
            break
        fi
    done

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing via Homebrew: ${pkgs[*]}"
        brew install "${pkgs[@]}"
    fi
}

install_debian() 
{
    local pkgs=()

    for dep in "${MISSING_CORE[@]}"; do
        case "$dep" in
            compiler) pkgs+=(g++) ;;
            make)     pkgs+=(make) ;;
            openmp)   pkgs+=(libomp-dev) ;;
        esac
    done

    if ! $HEADLESS; then
        for dep in "${MISSING_VIEWER[@]}"; do
            case "$dep" in
                vulkan)     pkgs+=(libvulkan-dev vulkan-tools) ;;
                glslang)    pkgs+=(glslang-tools) ;;
                glfw)       pkgs+=(libglfw3-dev) ;;
                pkg-config) pkgs+=(pkg-config) ;;
            esac
        done
    fi

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing via apt: ${pkgs[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y "${pkgs[@]}"
    fi
}

install_fedora() 
{
    local pkgs=()

    for dep in "${MISSING_CORE[@]}"; do
        case "$dep" in
            compiler) pkgs+=(gcc-c++) ;;
            make)     pkgs+=(make) ;;
            openmp)   pkgs+=(libomp-devel) ;;
        esac
    done

    if ! $HEADLESS; then
        for dep in "${MISSING_VIEWER[@]}"; do
            case "$dep" in
                vulkan)     pkgs+=(vulkan-headers vulkan-loader-devel) ;;
                glslang)    pkgs+=(glslang) ;;
                glfw)       pkgs+=(glfw-devel) ;;
                pkg-config) pkgs+=(pkgconf-pkg-config) ;;
            esac
        done
    fi

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing via dnf: ${pkgs[*]}"
        sudo dnf install -y "${pkgs[@]}"
    fi
}

install_arch() 
{
    local pkgs=()

    for dep in "${MISSING_CORE[@]}"; do
        case "$dep" in
            compiler) pkgs+=(gcc) ;;
            make)     pkgs+=(make) ;;
            openmp)   pkgs+=(openmp) ;;
        esac
    done

    if ! $HEADLESS; then
        for dep in "${MISSING_VIEWER[@]}"; do
            case "$dep" in
                vulkan)     pkgs+=(vulkan-headers vulkan-icd-loader) ;;
                glslang)    pkgs+=(glslang) ;;
                glfw)       pkgs+=(glfw-x11) ;;
                pkg-config) pkgs+=(pkgconf) ;;
            esac
        done
    fi

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing via pacman: ${pkgs[*]}"
        sudo pacman -S --noconfirm "${pkgs[@]}"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────
main() 
{
    echo ""
    info "Nimbus dependency setup"
    echo ""

    detect_platform

    # ── Check core dependencies ──
    echo ""
    info "Core dependencies (simulation engine)"
    check_compiler
    check_make
    check_openmp
    check_zfp

    # ── Check viewer dependencies ──
    if ! $HEADLESS; then
        echo ""
        info "Viewer dependencies (Vulkan renderer)"
        check_pkg_config
        check_vulkan
        check_glslang
        check_glfw
    fi

    # ── Summary ──
    echo ""
    local total_missing=$(( ${#MISSING_CORE[@]} + ${#MISSING_VIEWER[@]} ))

    if [[ $total_missing -eq 0 ]]; then
        info "All dependencies satisfied. Ready to build:"
        echo ""
        echo "    make -j\$(nproc 2>/dev/null || sysctl -n hw.ncpu)    # build simulation"
        if ! $HEADLESS; then
            echo "    make vulkan                                          # build viewer"
        fi
        echo "    make test                                              # run tests"
        echo ""
        return 0
    fi

    if $CHECK_ONLY; then
        echo ""
        warn "Missing ${total_missing} dependency group(s). Run without --check to install."
        return 1
    fi

    # ── Install ──
    echo ""
    info "Installing ${total_missing} missing dependency group(s)..."

    case "$PLATFORM" in
        macos)  install_macos  ;;
        debian) install_debian ;;
        fedora) install_fedora ;;
        arch)   install_arch   ;;
    esac

    # ── Verify ──
    echo ""
    info "Verifying installation..."
    MISSING_CORE=()
    MISSING_VIEWER=()

    check_compiler
    check_make
    check_openmp
    check_zfp
    if ! $HEADLESS; then
        check_pkg_config
        check_vulkan
        check_glslang
        check_glfw
    fi

    local still_missing=$(( ${#MISSING_CORE[@]} + ${#MISSING_VIEWER[@]} ))
    if [[ $still_missing -gt 0 ]]; then
        echo ""
        fail "Some dependencies could not be installed. See messages above."
        echo "    Consult BUILDING.md for manual installation steps."
        return 1
    fi

    echo ""
    info "Setup complete. Build with:"
    echo ""
    echo "    make -j\$(nproc 2>/dev/null || sysctl -n hw.ncpu)    # build simulation"
    if ! $HEADLESS; then
        echo "    make vulkan                                          # build viewer"
    fi
    echo "    make test                                              # run tests"
    echo ""
}

main
