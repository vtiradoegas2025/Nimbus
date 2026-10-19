#!/usr/bin/env python3
"""Fail when the compiled backend-neutral kernel contract drifts from GLSL/SPIR-V."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CPP = ROOT / "src/compute/kernel_manifest.cpp"
SPIRV_MANIFEST = ROOT / "vulkan/shaders/manifest.json"


def bit_range(first: int, last: int) -> int:
    return sum(1 << binding for binding in range(first, last + 1))


def parse_masks(expression: str) -> tuple[int, int, int]:
    expression = re.sub(
        r"bits\((\d+),\s*(\d+)\)",
        lambda match: str(bit_range(int(match.group(1)), int(match.group(2)))),
        expression,
    )
    parts = [part.strip() for part in expression.split(",")]
    if len(parts) != 3:
        raise ValueError(f"cannot parse resources({expression})")
    return tuple(int(eval(part, {"__builtins__": {}}, {})) for part in parts)  # noqa: S307


def glsl_resources(path: Path) -> tuple[int, int, int]:
    source = path.read_text(encoding="utf-8")
    bindings = re.findall(
        r"layout\s*\([^)]*binding\s*=\s*(\d+)[^)]*\)\s*"
        r"(readonly|writeonly)?\s*buffer",
        source,
        re.DOTALL,
    )
    if not bindings:
        raise ValueError(f"no storage-buffer bindings found in {path}")
    read = 0
    write = 0
    for binding_text, qualifier in bindings:
        bit = 1 << int(binding_text)
        if qualifier != "writeonly":
            read |= bit
        if qualifier != "readonly":
            write |= bit
    return max(int(binding) for binding, _ in bindings) + 1, read, write


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="synchronize compiled C++ SPIR-V hashes from the artifact manifest",
    )
    args = parser.parse_args()

    text = CPP.read_text(encoding="utf-8")
    starts = list(re.finditer(r"\{KernelId::[A-Za-z0-9_]+,", text))
    records: dict[str, tuple[str, str, int, int, int, int]] = {}
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else text.index("}};", start.start())
        block = text[start.start():end]
        strings = re.findall(r'"([^"]*)"', block)
        source_index = next((i for i, value in enumerate(strings) if value.endswith(".comp")), None)
        if source_index is None or source_index + 2 >= len(strings):
            raise ValueError(f"kernel block lacks source/artifact/hash: {block[:80]}")
        source, artifact, hash_value = strings[source_index:source_index + 3]
        resources_match = re.search(r"resources\((.*?)\),\s*S::", block, re.DOTALL)
        constants_match = re.search(
            r"S::[A-Za-z]+,\s*\{(\d+)(?:,\s*\d+(?:,\s*\d+)?)?\}", block)
        if resources_match is None or constants_match is None:
            raise ValueError(f"kernel block lacks resources/constants: {source}")
        span, read, write = parse_masks(resources_match.group(1))
        records[source] = (artifact, hash_value, int(constants_match.group(1)), span, read, write)

    artifact_manifest = json.loads(SPIRV_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        item["source"].split("vulkan/shaders/compute/", 1)[1]: item
        for item in artifact_manifest["shaders"]
        if "vulkan/shaders/compute/" in item["source"]
    }
    failures: list[str] = []
    refreshed = 0
    if set(records) != set(expected):
        failures.append(
            f"kernel set differs: missing={sorted(set(expected) - set(records))}, "
            f"extra={sorted(set(records) - set(expected))}"
        )

    for source in sorted(set(records) & set(expected)):
        artifact, hash_value, constants, span, read, write = records[source]
        item = expected[source]
        expected_artifact = item["artifact"].split("vulkan/shaders/compute/", 1)[1]
        if artifact != expected_artifact:
            failures.append(f"{source}: artifact {artifact} != {expected_artifact}")
        if hash_value != item["spirv_fnv1a64"]:
            if args.refresh:
                # Artifact and hash literals are allowed to wrap independently
                # under clang-format. Replace the parsed hash token itself,
                # after proving it identifies exactly one manifest record.
                old = f'"{hash_value}"'
                new = f'"{item["spirv_fnv1a64"]}"'
                if text.count(old) != 1:
                    failures.append(
                        f"{source}: cannot locate compiled hash {hash_value} for refresh"
                    )
                else:
                    text = text.replace(old, new, 1)
                    refreshed += 1
            else:
                failures.append(f"{source}: hash {hash_value} != {item['spirv_fnv1a64']}")
        if constants != item["push_constant_bytes"]:
            failures.append(f"{source}: constants {constants} != {item['push_constant_bytes']}")
        glsl_span, glsl_read, glsl_write = glsl_resources(ROOT / item["source"])
        if (span, read, write) != (glsl_span, glsl_read, glsl_write):
            failures.append(
                f"{source}: resources span/read/write "
                f"{(span, read, write)} != GLSL {(glsl_span, glsl_read, glsl_write)}"
            )

    if failures:
        print("Kernel manifest verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if refreshed:
        CPP.write_text(text, encoding="utf-8")
        print(f"Refreshed {refreshed} compiled kernel hash(es).")
    print(f"Kernel manifest verification passed: {len(records)} compute operations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
