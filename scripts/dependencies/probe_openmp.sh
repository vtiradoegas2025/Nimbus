#!/usr/bin/env bash
set -euo pipefail

probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-openmp.XXXXXX")"
trap 'rm -rf -- "$probe_dir"' EXIT
cat > "$probe_dir/probe.cpp" <<'CPP'
#include <omp.h>
int main() { return omp_get_max_threads() > 0 ? 0 : 1; }
CPP
"$@" -x c++ -fopenmp "$probe_dir/probe.cpp" -o "$probe_dir/probe.exe" >/dev/null 2>&1
