#!/usr/bin/env python3
"""Generate compile_commands.json from the live Makefile build flags.

modularity cleanup finding 8 (Phase 0): the checked-in compile DB went stale
(130 entries naming files that no longer exist, dated months back) because it
was produced once by an external tool and never refreshed, so clangd /
clang-tidy / IDE tooling silently drifted. This regenerates it from the exact
flags the Makefile computes for the current platform, and is wired to
`make compile-commands` so it never goes stale again.

It is intentionally self-maintaining: it globs the source trees rather than
reading a hand-kept list, so a newly added .cpp is covered automatically.

Usage:
    gen_compile_commands.py <compiler> <cxxflags> <test_cxxflags> [root]

  <cxxflags>      full CXXFLAGS string (already platform-resolved: -std, the
                  -I include dirs, Vulkan/OpenMP/ZFP flags, defines, ...).
  <test_cxxflags> CXXFLAGS plus the test recipe's extras (-O0 -I tests); applied
                  to translation units under tests/ so clangd resolves
                  <catch2/catch.hpp> and the test-only include path.
  [root]          repository root (default: current directory).

Emits a JSON array on stdout (clangd's compilation database format).
"""
import json
import os
import shlex
import sys

# (glob roots relative to repo root, flags string) — first match wins per file.
SOURCE_GROUPS = [
    (("tests",), "test"),
    (("src", "vulkan/src"), "main"),
]
SOURCE_EXTENSIONS = (".cpp", ".cc", ".cxx")


def main() -> int:
    if len(sys.argv) < 4:
        sys.stderr.write(
            "usage: gen_compile_commands.py <compiler> <cxxflags> "
            "<test_cxxflags> [root]\n"
        )
        return 2

    compiler = sys.argv[1]
    flags = {
        "main": shlex.split(sys.argv[2]),
        "test": shlex.split(sys.argv[3]),
    }
    root = os.path.abspath(sys.argv[4] if len(sys.argv) > 4 else ".")

    entries = []
    seen = set()
    for roots, kind in SOURCE_GROUPS:
        for rel_root in roots:
            abs_root = os.path.join(root, rel_root)
            if not os.path.isdir(abs_root):
                continue
            for dirpath, _dirs, files in os.walk(abs_root):
                for name in sorted(files):
                    if not name.endswith(SOURCE_EXTENSIONS):
                        continue
                    path = os.path.join(dirpath, name)
                    if path in seen:
                        continue
                    seen.add(path)
                    entries.append(
                        {
                            "directory": root,
                            "file": path,
                            "arguments": [compiler, *flags[kind], "-c", path],
                        }
                    )

    entries.sort(key=lambda e: e["file"])
    json.dump(entries, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
