"""Reproducible strict FP32 variants for NVIDIA division and compensated FMA."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess

STRATEGY = "nvidia-rte-precise-v1"


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def build_library(helper: pathlib.Path, temp: pathlib.Path) -> pathlib.Path:
    source = temp / "division-library.comp"
    source.write_text(
        '#version 450\n#extension GL_ARB_gpu_shader_int64 : require\n'
        'layout(local_size_x=1) in;\n'
        'layout(set=0,binding=0) buffer B { float values[]; };\n'
        + helper.read_text()
        + '\nvoid main() { values[0]=exact_divide(values[1],values[2]); }\n'
    )
    artifact = temp / "division-library.spv"
    run("glslangValidator", "-V", str(source), "-o", str(artifact))
    asm = run("spirv-dis", str(artifact))
    asm = re.sub(r'^.*Op(?:EntryPoint|ExecutionMode|Name|MemberName|Source|SourceExtension|String).*\n', '', asm, flags=re.M)
    asm = re.sub(r'%main = OpFunction.*?OpFunctionEnd', '', asm, flags=re.S)
    asm = re.sub(r'^\s*OpDecorate %\S+ BuiltIn WorkgroupSize\s*\n', '', asm, flags=re.M)
    asm = asm.replace('OpCapability Shader', 'OpCapability Shader\nOpCapability Linkage', 1)
    asm = asm.replace('OpTypeFunction %float %_ptr_Function_float %_ptr_Function_float', 'OpTypeFunction %float %float %float')
    asm = asm.replace('OpFunctionParameter %_ptr_Function_float', 'OpFunctionParameter %float')
    asm = re.sub(r'(%\S+) = OpLoad %float (%a|%b)\s*\n', r'\1 = OpCopyObject %float \2\n', asm)
    asm = asm.replace('OpDecorate', 'OpDecorate %exact_divide_f1_f1_ LinkageAttributes "nimbus_divide_rte32" Export\nOpDecorate', 1)
    assembly = temp / "division-library.spvasm"
    assembly.write_text(asm)
    run("spirv-as", "--target-env", "spv1.0", str(assembly), "-o", str(artifact))
    run("spirv-opt", "--eliminate-dead-variables", "--eliminate-dead-functions", "--eliminate-dead-const", "--ssa-rewrite", str(artifact), "-o", str(artifact))
    run("spirv-val", "--target-env", "spv1.0", str(artifact))
    return artifact


def lower(source: pathlib.Path, library: pathlib.Path, output: pathlib.Path, temp: pathlib.Path) -> int:
    asm = run("spirv-dis", "--raw-id", str(source))
    if not re.search(r'OpExecutionMode %\S+ RoundingModeRTE 32', asm):
        return 0
    divisions = re.findall(r'(%\S+) = OpFDiv (%\S+) (%\S+) (%\S+)', asm)
    fmas = re.findall(r'(%\S+) = OpExtInst (%\S+) %\S+ Fma (%\S+) (%\S+) (%\S+)', asm)
    if not divisions and not fmas:
        return 0
    float_type = re.search(r'(%\S+) = OpTypeFloat 32\b', asm)
    if not float_type or any(t != float_type[1] for _, t, _, _ in divisions):
        raise RuntimeError(f"unsupported division type in {source}")
    float_id = float_type[1]
    asm = asm.replace('OpCapability Shader', 'OpCapability Shader\nOpCapability Linkage', 1)
    first_type = re.search(r'^\s*%\S+ = OpType', asm, flags=re.M).start()
    asm = asm[:first_type] + '\nOpDecorate %nimbus_divide LinkageAttributes "nimbus_divide_rte32" Import\n' + asm[first_type:]
    first_fn = re.search(r'^\s*%\S+ = OpFunction ', asm, flags=re.M).start()
    declaration = (
        f'\n%nimbus_divide_type = OpTypeFunction {float_id} {float_id} {float_id}\n'
        f'%nimbus_divide = OpFunction {float_id} None %nimbus_divide_type\n'
        f'%nimbus_a = OpFunctionParameter {float_id}\n%nimbus_b = OpFunctionParameter {float_id}\nOpFunctionEnd\n'
    )
    asm = asm[:first_fn] + declaration + asm[first_fn:]
    # NoContraction on the original FDiv cannot decorate its replacement function call.
    division_ids = {result for result, _, _, _ in divisions}
    asm = re.sub(r'^\s*OpDecorate (%\S+) NoContraction\s*\n',
                 lambda match: '' if match[1] in division_ids else match[0], asm, flags=re.M)
    asm = re.sub(r'OpFDiv (%\S+) (%\S+) (%\S+)', r'OpFunctionCall \1 %nimbus_divide \2 \3', asm)
    if fmas:
        asm = asm.replace('OpCapability Shader', 'OpCapability Shader\nOpCapability FMAKHR', 1)
        asm = asm.replace('OpExtension ', 'OpExtension "SPV_KHR_fma"\nOpExtension ', 1)
        asm = re.sub(r'OpExtInst (%\S+) %\S+ Fma (%\S+) (%\S+) (%\S+)', r'OpFmaKHR \1 \2 \3 \4', asm)
        first_type = re.search(r'^\s*%\S+ = OpType', asm, flags=re.M).start()
        decorations = ''.join(f'OpDecorate {result} NoContraction\n' for result, *_ in fmas)
        asm = asm[:first_type] + '\n' + decorations + asm[first_type:]
    assembly = temp / "division-input.spvasm"
    imported = temp / "division-input.spv"
    assembly.write_text(asm)
    run("spirv-as", "--target-env", "spv1.0", str(assembly), "-o", str(imported))
    output.parent.mkdir(parents=True, exist_ok=True)
    run("spirv-link", str(imported), str(library), "-o", str(output))
    run("spirv-opt", "--compact-ids", str(output), "-o", str(output))
    run("spirv-val", "--target-env", "vulkan1.1", str(output))
    after = run("spirv-dis", str(output))
    if 'OpFDiv' in after:
        raise RuntimeError(f"unlowered division in {output}")
    before_modes = sorted(re.findall(r'OpExecutionMode %\S+ (\S+)([^\n]*)', asm))
    after_modes = sorted(re.findall(r'OpExecutionMode %\S+ (\S+)([^\n]*)', after))
    if before_modes != after_modes:
        raise RuntimeError(f"execution modes changed in {output}")
    return len(divisions) + len(fmas)


def render_header(entries: list[dict]) -> str:
    signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]
    rows = '\n'.join(
        '    {' + ', '.join(json.dumps(e[k]) for k in ('source_artifact', 'source_hash', 'artifact', 'artifact_hash')) + ', ' + str(e.get('requires_fma', False)).lower() + '},'
        for e in entries
    )
    return (
        '// Generated by scripts/dev/shaders/verify_spirv.py; regenerate with --refresh.\n'
        '#pragma once\n#include <array>\n#include <string_view>\n\nnamespace tmv_vulkan {\n'
        'struct DivisionVariant { std::string_view source_artifact, source_hash, artifact, artifact_hash; bool requires_fma; };\n'
        f'inline constexpr std::string_view division_variant_signature = "{STRATEGY}-{signature}";\n'
        f'inline constexpr std::array<DivisionVariant, {len(entries)}> division_variants = {{{{\n{rows}\n}}}};\n'
        '}\n'
    )
