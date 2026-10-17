#!/usr/bin/env bash
# Supply Homebrew's MoltenVK loader paths on Apple Silicon, then run a command.
# Linux/native-Vulkan environments and explicit caller overrides are unchanged.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

if [ "$(uname -s)" = "Darwin" ]; then
    DEFAULT_ICD=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
    if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -s "$DEFAULT_ICD" ]; then
        export VK_ICD_FILENAMES="$DEFAULT_ICD"
    fi

    VULKAN_DYLD=/opt/homebrew/opt/vulkan-loader/lib:/opt/homebrew/opt/molten-vk/lib
    export DYLD_LIBRARY_PATH="$VULKAN_DYLD${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

exec "$@"
