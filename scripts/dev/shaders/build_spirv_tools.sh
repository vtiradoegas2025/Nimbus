#!/usr/bin/env bash
set -euo pipefail

shader_root="${1:?Usage: build_spirv_tools.sh BUILD_DIRECTORY}"
mkdir -p "$shader_root"
shader_root="$(cd "$shader_root" && pwd -P)"
tools_revision=fbe4f3ad913c44fe8700545f8ffe35d1382b7093
headers_revision=04f10f650d514df88b76d25e83db360142c7b174

checkout() {
    local repository="$1" revision="$2" destination="$3"
    if [[ -d "$destination/.git" ]]; then
        [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
            echo "Existing shader tool checkout has a different revision; use a new build directory." >&2
            exit 1
        }
    else
        git init -q "$destination"
        git -C "$destination" remote add origin "$repository"
        git -C "$destination" fetch -q --depth 1 origin "$revision"
        git -C "$destination" checkout -q --detach FETCH_HEAD
    fi
}

checkout https://github.com/KhronosGroup/SPIRV-Tools.git "$tools_revision" "$shader_root/source"
checkout https://github.com/KhronosGroup/SPIRV-Headers.git "$headers_revision" "$shader_root/source/external/spirv-headers"
cmake -S "$shader_root/source" -B "$shader_root/build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=OFF
cmake --build "$shader_root/build" --parallel "${NIMBUS_SHADER_BUILD_JOBS:-2}" \
    --target spirv-as spirv-dis spirv-val spirv-opt spirv-link
"$shader_root/build/tools/spirv-val" --version
echo "Shader tool executables: $shader_root/build/tools"
