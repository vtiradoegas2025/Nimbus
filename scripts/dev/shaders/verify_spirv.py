#!/usr/bin/env python3
"""Rebuild, validate, reflect, and provenance-check every Nimbus shader.

The checked SPIR-V remains the runtime artifact. Verification compiles each
GLSL source into a temporary directory with the repository's declared flags,
validates both copies, canonicalizes away debug records/ID numbering, ignores
only the generator word in the SPIR-V header, and requires byte equality.

`--refresh` replaces only canonically drifted artifacts, then rewrites the
manifest. Artifacts that already represent the same module are left untouched,
so a harmless compiler-generator version change does not churn 31 binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

import division_workaround


ROOT = pathlib.Path(__file__).resolve().parents[3]
SHADER_ROOT = ROOT / "vulkan" / "shaders"
MANIFEST = SHADER_ROOT / "manifest.json"
SOURCE_SUFFIXES = {".comp": "comp", ".vert": "vert", ".frag": "frag"}
COMPILE_FLAGS = ["-V", "-Os"]


def compile_flags(source: pathlib.Path) -> list[str]:
    # Full inlining of the compensated cold process family exhausts glslang's optimizer.
    if source.name in ("milbrandt_cold.comp", "milbrandt_sedimentation.comp"):
        return ["-V"]
    return COMPILE_FLAGS


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required shader tool not found: {name}")
    return path


def run(args: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout if capture else ""


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64(path: pathlib.Path) -> str:
    value = 0xCBF29CE484222325
    for byte in path.read_bytes():
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def sources() -> list[pathlib.Path]:
    return sorted(
        path for path in SHADER_ROOT.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def compile_shader(source: pathlib.Path, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = SOURCE_SUFFIXES[source.suffix]
    run([
        tool("glslangValidator"),
        *compile_flags(source),
        "-S", stage,
        str(source),
        "-o", str(output),
    ])
    if source.name == "acoustic_retained_flux_fp64.comp":
        # spirv_instruction does not propagate GLSL precise to extension ops.
        assembly = run([tool("spirv-dis"), str(output)], capture=True)
        decorations = "".join(
            f"OpDecorate {result} NoContraction\n"
            for result in re.findall(r"(%\S+) = OpFmaKHR", assembly)
            if f"OpDecorate {result} NoContraction" not in assembly)
        at = re.search(r"^\s*%\S+ = OpType", assembly, flags=re.M).start()
        assembly = assembly[:at] + decorations + assembly[at:]
        with tempfile.TemporaryDirectory(prefix="nimbus-fp64-modes-") as temporary:
            path = pathlib.Path(temporary) / "shader.spvasm"
            path.write_text(assembly)
            run([tool("spirv-as"), "--target-env", "spv1.0", str(path), "-o", str(output)])


def validate(path: pathlib.Path) -> None:
    run([tool("spirv-val"), str(path)])


def canonical_bytes(path: pathlib.Path, temp_dir: pathlib.Path) -> bytes:
    output = temp_dir / f"{path.name}.{hashlib.sha1(str(path).encode()).hexdigest()}.canon"
    run([
        tool("spirv-opt"),
        "--strip-debug",
        "--compact-ids",
        str(path),
        "-o", str(output),
    ])
    data = bytearray(output.read_bytes())
    if len(data) < 20:
        raise RuntimeError(f"SPIR-V header truncated: {path}")
    # Word 2 is the generator magic/version. Module semantics and every other
    # header word remain compared after spirv-opt canonicalization.
    data[8:12] = b"\0\0\0\0"
    return bytes(data)


def generator_word(path: pathlib.Path) -> str:
    data = path.read_bytes()[:12]
    if len(data) < 12:
        raise RuntimeError(f"SPIR-V header truncated: {path}")
    return f"0x{struct.unpack('<I', data[8:12])[0]:08x}"


def reflection(path: pathlib.Path) -> dict[str, Any]:
    text = run([tool("spirv-dis"), str(path), "-o", "-"], capture=True)
    descriptor_sets: dict[str, int] = {}
    bindings: dict[str, int] = {}
    member_offsets: dict[tuple[str, int], int] = {}
    array_strides: dict[str, int] = {}
    definitions: dict[str, list[str]] = {}
    constants: dict[str, int] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        tokens = line.split()
        if " = " in line:
            lhs, rhs = line.split(" = ", 1)
            definitions[lhs.strip()] = rhs.split()
            rhs_tokens = rhs.split()
            if rhs_tokens and rhs_tokens[0] == "OpConstant" and len(rhs_tokens) >= 3:
                try:
                    constants[lhs.strip()] = int(rhs_tokens[2], 0)
                except ValueError:
                    pass
            continue
        if tokens[0] == "OpDecorate" and len(tokens) >= 4:
            target, kind = tokens[1], tokens[2]
            if kind == "DescriptorSet":
                descriptor_sets[target] = int(tokens[3])
            elif kind == "Binding":
                bindings[target] = int(tokens[3])
            elif kind == "ArrayStride":
                array_strides[target] = int(tokens[3])
        elif tokens[0] == "OpMemberDecorate" and len(tokens) >= 5:
            if tokens[3] == "Offset":
                member_offsets[(tokens[1], int(tokens[2]))] = int(tokens[4])

    def type_size(type_id: str, active: set[str] | None = None) -> int:
        active = set() if active is None else set(active)
        if type_id in active:
            raise RuntimeError(f"recursive SPIR-V type in {path}: {type_id}")
        active.add(type_id)
        op = definitions.get(type_id)
        if not op:
            raise RuntimeError(f"missing SPIR-V type {type_id} in {path}")
        kind = op[0]
        if kind in {"OpTypeInt", "OpTypeFloat"}:
            return int(op[1]) // 8
        if kind == "OpTypeBool":
            return 4
        if kind == "OpTypeVector":
            return type_size(op[1], active) * int(op[2])
        if kind == "OpTypeMatrix":
            return type_size(op[1], active) * int(op[2])
        if kind == "OpTypeArray":
            stride = array_strides.get(type_id, type_size(op[1], active))
            return stride * constants[op[2]]
        if kind == "OpTypeStruct":
            occupied = 0
            for index, member_type in enumerate(op[1:]):
                offset = member_offsets.get((type_id, index), occupied)
                occupied = max(occupied, offset + type_size(member_type, active))
            return occupied
        if kind == "OpTypePointer":
            return type_size(op[2], active)
        raise RuntimeError(f"unsupported SPIR-V type {kind} in {path}")

    push_sizes: list[int] = []
    for definition in definitions.values():
        if len(definition) >= 3 and definition[0] == "OpVariable" and definition[2] == "PushConstant":
            pointer = definitions.get(definition[1], [])
            if len(pointer) >= 3 and pointer[0] == "OpTypePointer":
                push_sizes.append(type_size(pointer[2]))

    descriptor_bindings = sorted(
        {
            (descriptor_sets[target], binding)
            for target, binding in bindings.items()
            if target in descriptor_sets
        }
    )
    return {
        "descriptor_bindings": [
            {"set": set_index, "binding": binding}
            for set_index, binding in descriptor_bindings
        ],
        "push_constant_bytes": max(push_sizes, default=0),
    }


def entry(source: pathlib.Path, artifact: pathlib.Path) -> dict[str, Any]:
    reflected = reflection(artifact)
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "artifact": artifact.relative_to(ROOT).as_posix(),
        "stage": SOURCE_SUFFIXES[source.suffix],
        "entry_point": "main",
        "flags": compile_flags(source),
        "source_sha256": sha256(source),
        "spirv_sha256": sha256(artifact),
        "spirv_fnv1a64": fnv1a64(artifact),
        "generator_word": generator_word(artifact),
        **reflected,
    }


def tool_version(name: str) -> str:
    return run([tool(name), "--version"], capture=True).splitlines()[0]


def manifest_payload(shader_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "verification_toolchain": {
            "glslangValidator": tool_version("glslangValidator"),
            "spirv-val": tool_version("spirv-val"),
            "spirv-opt": tool_version("spirv-opt"),
        },
        "canonicalization": ["spirv-opt --strip-debug --compact-ids", "ignore generator word"],
        "include_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted(SHADER_ROOT.rglob('*.glsl'))
        },
        "shaders": shader_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="replace canonically drifted SPIR-V and rewrite the manifest",
    )
    args = parser.parse_args()

    shader_sources = sources()
    if not shader_sources:
        raise RuntimeError("no shader sources found")
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="nimbus-spirv-") as raw_temp:
        temp = pathlib.Path(raw_temp)
        for source in shader_sources:
            artifact = source.with_name(source.name + ".spv")
            if not artifact.is_file():
                failures.append(f"missing artifact: {artifact.relative_to(ROOT)}")
                continue
            generated = temp / artifact.relative_to(SHADER_ROOT)
            compile_shader(source, generated)
            validate(generated)
            validate(artifact)
            if canonical_bytes(generated, temp) != canonical_bytes(artifact, temp):
                if args.refresh:
                    shutil.copyfile(generated, artifact)
                    validate(artifact)
                    print(f"refreshed {artifact.relative_to(ROOT)}")
                else:
                    failures.append(f"artifact drift: {artifact.relative_to(ROOT)}")
                    continue
            if reflection(generated) != reflection(artifact):
                failures.append(f"reflection drift: {artifact.relative_to(ROOT)}")

        helper = SHADER_ROOT / "support/exact_division.glsl"
        library = division_workaround.build_library(helper, temp)
        variants = []
        for source in shader_sources:
            original = source.with_name(source.name + ".spv")
            generated = temp / "division-variants" / original.relative_to(SHADER_ROOT)
            count = division_workaround.lower(original, library, generated, temp)
            if not count:
                continue
            relative = pathlib.Path("variants") / division_workaround.STRATEGY / original.relative_to(SHADER_ROOT / "compute")
            artifact = SHADER_ROOT / "compute" / relative
            if not artifact.is_file() or canonical_bytes(generated, temp) != canonical_bytes(artifact, temp):
                if args.refresh:
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(generated, artifact)
                else:
                    failures.append(f"division variant drift: {artifact.relative_to(ROOT)}")
                    continue
            validate(artifact)
            if reflection(original) != reflection(artifact):
                failures.append(f"division variant ABI drift: {artifact.relative_to(ROOT)}")
            variants.append({
                "source_artifact": original.relative_to(SHADER_ROOT / "compute").as_posix(),
                "source_hash": fnv1a64(original),
                "artifact": relative.as_posix(),
                "artifact_hash": fnv1a64(artifact),
                "helper_sha256": sha256(helper),
                "arithmetic_rewrites": count,
                "requires_fma": "OpFmaKHR" in division_workaround.run("spirv-dis", str(artifact)),
            })
        header = ROOT / "vulkan/include/compute/generated_division_variants.hpp"
        header_text = division_workaround.render_header(variants)
        if args.refresh:
            header.write_text(header_text)
        elif not header.is_file() or header.read_text() != header_text:
            failures.append("division variant header drift")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print("Run scripts/dev/shaders/verify_spirv.py --refresh after reviewing shader changes.", file=sys.stderr)
        return 1

    entries = [entry(source, source.with_name(source.name + ".spv")) for source in shader_sources]
    payload = manifest_payload(entries)
    payload["division_variants"] = variants
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.refresh:
        MANIFEST.write_text(rendered, encoding="utf-8")
    elif not MANIFEST.is_file():
        print(f"ERROR: missing manifest: {MANIFEST.relative_to(ROOT)}", file=sys.stderr)
        return 1
    else:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        # Tool versions are provenance for the last refresh, not a portability
        # restriction. Canonical module equality above is the cross-version gate.
        expected["verification_toolchain"] = payload["verification_toolchain"]
        if expected != payload:
            print("ERROR: shader manifest drift; run scripts/dev/shaders/verify_spirv.py --refresh", file=sys.stderr)
            return 1

    print(
        f"SPIR-V verification passed: {len(entries)} sources/artifacts, "
        "all validated and canonically reproducible."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
