#!/usr/bin/env bash
set -euo pipefail

shader_root="${1:?Usage: build_spirv_tools.sh BUILD_DIRECTORY}"
mkdir -p "$shader_root"
shader_root="$(cd "$shader_root" && pwd -P)"
tools_revision=fbe4f3ad913c44fe8700545f8ffe35d1382b7093
headers_revision=04f10f650d514df88b76d25e83db360142c7b174
glslang_revision=f0bd0257c308b9a26562c1a30c4748a0219cc951

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
checkout https://github.com/KhronosGroup/glslang.git "$glslang_revision" "$shader_root/glslang"
mkdir -p "$shader_root/toolchain" "$shader_root/bin"
# glslang -Os embeds the optimizer; putting newer CLI tools on PATH is insufficient.
cat > "$shader_root/toolchain/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.22.1)
project(NimbusShaderToolchain LANGUAGES C CXX)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../source" spirv-tools)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../glslang" glslang)
CMAKE
cmake -S "$shader_root/toolchain" -B "$shader_root/toolchain-build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=OFF \
    -DGLSLANG_TESTS=OFF -DGLSLANG_ENABLE_INSTALL=OFF -DBUILD_EXTERNAL=OFF -DENABLE_OPT=ON
cmake --build "$shader_root/toolchain-build" --parallel "${NIMBUS_SHADER_BUILD_JOBS:-2}" \
    --target spirv-as spirv-dis spirv-val spirv-opt spirv-link glslang-standalone
for executable in spirv-as spirv-dis spirv-val spirv-opt spirv-link; do
    ln -sf "$shader_root/toolchain-build/spirv-tools/tools/$executable" "$shader_root/bin/$executable"
done
ln -sf "$shader_root/toolchain-build/glslang/StandAlone/glslang" "$shader_root/bin/glslangValidator"
"$shader_root/bin/spirv-val" --version
"$shader_root/bin/glslangValidator" --version
echo "Shader tool executables: $shader_root/bin"
