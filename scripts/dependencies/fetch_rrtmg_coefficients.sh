#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DESTINATION_DIR="${1:-${REPOSITORY_ROOT}/data/rrtmg}"

LONGWAVE_COMMIT="958edb1ad1618d76d0fcc5a9dc2e3c2f791a54dd"
SHORTWAVE_COMMIT="286e84ed14f61e2279ba819f4ce512a30b48f0b3"
LONGWAVE_SHA256="5ea5b5d4129d31952404985dc52059573f5d674cd8ee20bf5038f5e929f514a1"
SHORTWAVE_SHA256="6bbdcaf3339d98b6da64980677b8f72332d128cc9576fae301cca852c97de717"

require_command()
{
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command '$1' was not found" >&2
        exit 1
    fi
}

sha256_file()
{
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo "error: shasum or sha256sum is required" >&2
        exit 1
    fi
}

fetch_verified()
{
    local url="$1"
    local expected_sha256="$2"
    local destination="$3"
    local temporary
    temporary="$(mktemp "${destination}.tmp.XXXXXX")"

    if ! curl --fail --location --silent --show-error \
        --output "${temporary}" "${url}"; then
        rm -f -- "${temporary}"
        return 1
    fi

    local actual_sha256
    actual_sha256="$(sha256_file "${temporary}")"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "error: checksum mismatch for ${destination}" >&2
        echo "expected: ${expected_sha256}" >&2
        echo "actual:   ${actual_sha256}" >&2
        rm -f -- "${temporary}"
        return 1
    fi
    mv -f -- "${temporary}" "${destination}"
}

require_command curl
require_command mktemp
require_command awk
mkdir -p -- "${DESTINATION_DIR}"

fetch_verified \
    "https://raw.githubusercontent.com/AER-RC/RRTMG_LW/${LONGWAVE_COMMIT}/data/rrtmg_lw.nc" \
    "${LONGWAVE_SHA256}" \
    "${DESTINATION_DIR}/rrtmg_lw.nc"
fetch_verified \
    "https://raw.githubusercontent.com/AER-RC/RRTMG_SW/${SHORTWAVE_COMMIT}/data/rrtmg_sw.nc" \
    "${SHORTWAVE_SHA256}" \
    "${DESTINATION_DIR}/rrtmg_sw.nc"

echo "RRTMG coefficients installed in ${DESTINATION_DIR}"
