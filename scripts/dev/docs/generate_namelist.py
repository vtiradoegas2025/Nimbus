#!/usr/bin/env python3
"""Render the reviewed configuration inventory and check its source/default identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).with_name('namelist.json')
DESTINATION = ROOT / 'docs/NAMELIST.md'

INTRODUCTION = '''# Nimbus namelist: complete v1 configuration reference

This is the parameter reference for a Nimbus `schema: nimbus.v1` input file.
It includes **every strict-v1 key**, including options that are guarded,
unfinished, conditional or retained only to reject an obsolete request.
Nimbus reads nested YAML mappings; it does not read a CM1 Fortran `&param`
namelist. Similar names do not promise the same equations, defaults, units or
numeric option codes as CM1. The [mathematics guide](MATHEMATICS.pdf) explains
those differences and the [architecture guide](ARCHITECTURE.pdf) explains how
the selected components connect.

The inventory contains **302 finite keys**: 288 fixed keys and 14 permitted
chaos-amplitude map entries. It also documents the open
`output.zfp_field_tolerance.<field>` mapping. The command-line and legacy
compatibility sections distinguish other input routes from this v1 namespace.
Studio's form is a view of this configuration, not its complete definition;
some settings require editing YAML.

## How to read this reference

**When omitted** reports the C++ configuration default or the explicit
conditional rule, not a Studio draft or a resolution preset. A required value
has no usable omission default. A dormant parameter can have a typed default
even when writing it explicitly for the wrong scheme is rejected. Consequently,
this whole reference must not be pasted into a single configuration: mutually
exclusive and rejected keys are intentionally listed together.

The type column describes the scalar/list parser. `Real > 0` and `real ≥ 0`
also require finite values. Section and field notes add coupled restrictions;
scientific admission can be narrower still. Enum aliases are listed explicitly
when the parser retains them. Unknown v1 keys are rejected. Parameter names
are case-sensitive: `Cs`, `Pr_t`, `Sc_t`, `K_h`, `K_v`, `H` and `h0` retain
their spelling.

Use two-space-indented mappings, plain scalar values and inline lists such as
`[w, reflectivity_dbz]`. The native reader is a restricted indentation/flattening
reader, not a general YAML implementation: do not use anchors, multiline
scalars or block sequences. It treats the first `#` as a comment delimiter
even inside quotes. Avoid duplicate keys; duplicate flattened names currently
overwrite earlier values. Avoid `#` inside paths or other string values.

For example, this shows the syntax of the required fields; it is not a
recommended physical experiment or a qualified configuration:

```yaml
schema: nimbus.v1
grid:
  nx: 64
  ny: 64
  nz: 64
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 60
  dt_large: 1
  output_interval: 10
```

Start an actual experiment from a [maintained configuration](../configs/README.md).
For the [build-and-open check](BUILDING.md#build-and-open-only), no input file,
simulation or external coefficient tables are necessary.

## Availability and scientific status

A recognized key, a parsed configuration, an admitted profile and a Fully
Qualified capability are different things. Use the actual executable's
`--scheme-catalog=json` and ordinary validation path for the complete combination.
`--validate-only` prepares a one-dimensional initial-condition context but runs
no timesteps; it is separate from merely opening Studio.

The runtime value `extensions.execution.mode: qualified` names a strict execution
policy. It is **not** a claim of Fully Qualified science. That status additionally
requires all applicable engineering, bounded tests, long campaigns, checked
evidence and independent scientific review. The required reporting states are
Engineering incomplete, Ready, Campaign validated, Externally reviewed and
Fully qualified; see [AGENTS.md](../AGENTS.md).

Tracked broader gaps include remaining native physics combinations, moist
two-way nesting, the complete bounded-history ZFP scope, the retained vertical-only
SGS gate, Milbrandt triple moments and exact catalog/CLI/Studio parity. Some
narrow profiles within these families can be admitted without completing the
whole tracked item. The [readiness registry](../src/core/runtime/readiness.cpp)
and runtime catalog retain that distinction. Acknowledging a limitation does
not authorize a rejected option.

Two particularly easy mistakes are documented in the tables: the loader rejects
an explicit `microphysics.milbrandt.triple_moment` key even when false, and the
vortex seed's `center_z_m` currently does not control its vertical placement.

## Parameter index
'''


def cell(value: str) -> str:
    return value.replace('|', '\\|').replace('\n', ' ')


def kind(row: dict) -> str:
    if row.get('choices'):
        return ', '.join(f'`{value}`' for value in row['choices'])
    return {
        'enum': 'Selector', 'coordinate_system': 'Selector', 'stagger_type': 'Selector',
        'int': 'Integer', 'positive_int': 'Integer > 0', 'non_negative_int': 'Integer ≥ 0',
        'double': 'Finite real', 'positive_double': 'Real > 0',
        'non_negative_double': 'Real ≥ 0', 'bool': 'Boolean',
        'string': 'String', 'runtime_seconds': 'Whole seconds', 'list': 'Inline list',
    }[row['kind']]


def legacy_inventory(data: dict) -> str:
    legacy = data['legacy']
    chunks = [f'''The compatibility inventory below lists all {legacy['count']} literal keys read by
the legacy runtime/output owners, including the schema branch selector. It is
an index for understanding old files, not a second recommended input schema.
Legacy omission defaults and accepted values can differ even when a spelling
also exists in v1. Unknown legacy keys lack v1's consumed-key rejection pass;
silently tolerating a key does not mean an operator uses it.

Only directly traced migration renames/conversions are labeled as such. Other
entries require inspecting migration diagnostics; a missing mapping in this
table does not establish that migration is impossible. Legacy `nested.enabled:
true` is rejected by the nested-grid owner, separately from modern
`domain_motion.nesting`. Legacy Milbrandt `triple_moment: false` can be consumed,
while true is rejected; strict v1 rejects the explicit key regardless of value.

<details>
<summary>Complete legacy reader inventory ({legacy['count']} keys)</summary>

| Legacy key | Relationship to v1 |
| --- | --- |
''']
    for row in legacy['keys']:
        target = f"../{row['source']}#L{row['line']}"
        relation = 'Same spelling exists; compare semantics and omission defaults.' if row['same_v1_spelling'] else 'Legacy-only spelling; inspect migration diagnostics.'
        if row['migrations']:
            relation = '; '.join(
                f"[{'Unit conversion to' if item['operation'] == 'warn_unit' else 'Migrates to'} `{item['destination']}`](../{item['source']}#L{item['line']})"
                for item in row['migrations']) + '.'
        chunks.append(f"| [`{row['path']}`]({target}) | {relation} |\n")
    chunks.append('\n</details>\n\nLegacy dynamic families are also read:\n\n| Pattern | Meaning |\n| --- | --- |\n')
    for row in legacy['dynamic_families']:
        source, line = row['source'].rsplit(':', 1)
        chunks.append(f"| [`{row['pattern']}`](../{source}#L{line}) | {cell(row['description'])} |\n")
    return ''.join(chunks) + '\n'


def render(data: dict) -> str:
    chunks = [INTRODUCTION]
    for section in data['sections']:
        chunks.append(f"- [{section['title']}](#{section['id'].replace('_', '-')})\n")
    chunks.append('- [Per-field ZFP map](#per-field-zfp-map)\n')
    chunks.append('- [Command-line companion](#command-line-companion)\n')
    chunks.append('- [Legacy compatibility](#legacy-compatibility)\n')
    for section in data['sections']:
        prefix = section['id'] + '.'
        rows = [row for row in data['fields'] if row['path'] == section['id'] or row['path'].startswith(prefix)]
        location = ('Top-level scalar; do not nest it under another `schema` key.\n\n'
                    if section['id'] == 'schema' else
                    f"Prefix: `{section['id']}`. Each name below is relative to that prefix.\n\n")
        chunks.extend([f"\n## {section['title']}\n\n", section['description'] + '\n\n', location,
                       '| Key | Type / choices | When omitted | Meaning and restrictions |\n',
                       '| --- | --- | --- | --- |\n'])
        for row in rows:
            name = row['path'].removeprefix(prefix)
            link = f"[`{name}`](../src/init/config/v1_loader.cpp#L{row['loader_line']})"
            description = row['description']
            if row.get('aliases'):
                description += ' Aliases: ' + '; '.join(f"`{alias}` → `{canonical}`" for alias, canonical in row['aliases'].items()) + '.'
            chunks.append('| ' + ' | '.join(map(cell, [link, kind(row), row['default_text'], description])) + ' |\n')
    chunks.append('''
## Per-field ZFP map

`output.zfp_field_tolerance.<field>` supplies a **strictly positive finite**
absolute tolerance for a named field, in that field's native units. The map is
empty by default. A nonempty field name passes the map-key parser; the output
policy must still accept that field and its relationship to the selected
preset, format, tolerance mode and qualified scope. This is an open mapping,
not a fixed list of eight Studio controls.

For example, the syntax of one entry is:

```yaml
output:
  zfp_field_tolerance:
    theta: 0.001
```

This fragment is not a complete experiment and does not select a ZFP output
format by itself. A tolerance for potential temperature in K is not
interchangeable with a mixing-ratio tolerance in kg kg⁻¹ dry air.

## Command-line companion

Command-line overrides are separate from the YAML keys. In particular,
`--threads` configures OpenMP workers, output/checkpoint paths select filesystem
destinations, and several diagnostics/live-stream controls exist only at this
boundary. Inspect `--help` and `--version` on the executable actually being used.
The current maintained help is reproduced below. Internal qualification flags
require their special test builds; they are not ordinary configuration escape
hatches or instructions to bypass a readiness guard.

```text
''')
    source = (ROOT / 'src/core/runtime/nimbus_sim.cpp').read_text()
    help_text = re.search(r'kRuntimeUsage = R"\((.*?)\)";', source, re.DOTALL)
    if help_text is None:
        raise ValueError('cannot locate maintained runtime help')
    chunks.append(help_text.group(1).rstrip() + '\n```\n')
    chunks.append('''
Build-worker limits (`NIMBUS_BUILD_JOBS`, `CARGO_BUILD_JOBS` and CMake's
`CMAKE_BUILD_PARALLEL_LEVEL`) belong to compilation, not the namelist or
simulation thread count. Runtime environment also supplies library/driver
discovery, OpenMP policy and optional coefficient locations such as
`NIMBUS_RRTMG_LW_COEFFICIENTS` and `NIMBUS_RRTMG_SW_COEFFICIENTS`.
Use [BUILDING.md](BUILDING.md) for provisioning. These operating-environment
settings are not additional strict-v1 keys.

## Legacy compatibility

Unversioned legacy configuration loading and v0 migration still exist. Their
spellings are not additional `nimbus.v1` keys: adding an old flat key to a v1
document causes an unknown-key error. Migrate a copy with `--migrate-config`,
inspect its warnings and resulting YAML, then validate the new document.
Migration can translate, resolve, drop or reject legacy settings; it does not
certify that the old experiment had the same physical interpretation.

The maintained compatibility owners are
[the legacy runtime loader](../src/core/runtime/runtime_config.cpp) and
[the migration implementation](../src/init/config/v0_to_v1.cpp), including its
[physics](../src/init/config/v0_to_v1_physics.cpp),
[initialization](../src/init/config/v0_to_v1_initialization.cpp) and
[runtime/output](../src/init/config/v0_to_v1_runtime.cpp) translators.

## Keeping this list current

The [strict loader](../src/init/config/v1_loader.cpp),
[typed configuration](../include/init/config/config_v1.hpp),
[microphysics defaults](../include/microphysics/config.hpp), enum parsers and
ordinary runtime admission are authoritative. Linked key names lead to their
loader consumers. The reviewed inventory and document generator live under
`scripts/dev/docs/`; `make check-docs` checks their source identities and the
rendered reference. A source change requires reviewing affected entries rather
than silently carrying old descriptions forward.

The optional native-default check constructs `ConfigV1` and reads its fields;
it runs no atmosphere and loads no scientific data:

```sh
python3 scripts/dev/docs/generate_namelist.py --check --verify-defaults
```

Constructor-default verification establishes the omission-value inventory,
not correctness of the physical assumptions or qualification of a combination.
''')
    return ''.join(chunks).replace('## Keeping this list current', legacy_inventory(data) + '\n## Keeping this list current')


def verify_defaults(data: dict, compiler: str) -> None:
    header = r'''#include "init/config/config_v1.hpp"
#include <iostream>
#include <iomanip>
#include <type_traits>
using namespace tmv::init::config;
template<class T> void value(const T& v) { if constexpr(std::is_enum_v<T>) std::cout << static_cast<int>(v); else if constexpr(std::is_same_v<T,std::string>) std::cout<<std::quoted(v); else std::cout<<v; }
template<class T> void value(const std::optional<T>& v) { if(v) value(*v); else std::cout<<"null"; }
template<class T> void value(const std::vector<T>& v) { std::cout<<"["; bool first=true; for(const auto& x:v) { if(!first) std::cout<<","; value(x); first=false; } std::cout<<"]"; }
std::optional<double> entry(const std::unordered_map<std::string,double>& map,const std::string& key) { auto it=map.find(key); return it==map.end()?std::nullopt:std::optional<double>(it->second); }
int main(){ ConfigV1 c; std::cout<<std::setprecision(17)<<std::boolalpha<<"{\n";
'''
    for index, row in enumerate(data['fields']):
        header += 'std::cout<<' + json.dumps((',' if index else '') + json.dumps(row['path']) + ':')
        header += '; value(' + row['cpp_expression'] + '); std::cout<<"\\n";\n'
    header += 'std::cout<<"}\\n";}\n'
    with tempfile.TemporaryDirectory(prefix='nimbus-namelist-') as temp:
        source = Path(temp) / 'defaults.cpp'
        binary = Path(temp) / ('defaults.exe' if os.name == 'nt' else 'defaults')
        source.write_text(header)
        subprocess.run([*shlex.split(compiler), '-std=c++17', '-I' + str(ROOT / 'include'), str(source), '-o', str(binary)], check=True)
        actual = json.loads(subprocess.check_output([str(binary)], text=True))
    expected = {row['path']: row['constructor_default'] for row in data['fields']}
    changed = [key for key in expected if expected[key] != actual.get(key)]
    if changed or actual.keys() != expected.keys():
        raise ValueError('native defaults differ: ' + ', '.join(changed))
    print(f'Native constructor defaults verified: {len(expected)} keys.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--verify-defaults', action='store_true')
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    data = json.loads(DATA.read_text())
    stale = [name for name, digest in data['sources'].items()
             if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if stale:
        parser.error('configuration source changed; review namelist.json before recording new identities: ' + ', '.join(stale))
    paths = [row['path'] for row in data['fields']]
    if len(paths) != len(set(paths)) or len(paths) != data['finite_key_count'] or any(not row['description'] for row in data['fields']):
        parser.error('incomplete or duplicate namelist entries')
    produced = render(data)
    if args.write:
        DESTINATION.write_text(produced)
    elif not DESTINATION.exists() or DESTINATION.read_text() != produced:
        parser.error('NAMELIST.md differs from reviewed inventory; run --write after reviewing inputs')
    if args.verify_defaults:
        verify_defaults(data, args.cxx)
    print(f'Namelist verified: {len(paths)} finite v1 keys and the open ZFP field map.')


if __name__ == '__main__':
    main()
