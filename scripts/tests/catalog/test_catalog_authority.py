#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
REFERENCE = ROOT / "tests/fixtures/reference/catalog/authority_v1.json"


def canonical_surface(catalog: dict[str, object]) -> dict[str, object]:
    capabilities = catalog.get("runtimeCapabilities")
    if not isinstance(capabilities, dict):
        raise RuntimeError("catalog runtimeCapabilities is malformed")
    output_formats = capabilities.get("outputFormats")
    if not isinstance(output_formats, list):
        raise RuntimeError("catalog outputFormats is malformed")
    format_ids = []
    for value in output_formats:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise RuntimeError("catalog output format is malformed")
        format_ids.append(value["id"])
    return {
        "schema": catalog.get("schema"),
        "catalogVersion": catalog.get("catalogVersion"),
        "runtimeStateFields": catalog.get("runtimeStateFields"),
        "modules": catalog.get("modules"),
        "interactionPolicy": capabilities.get("interactionPolicy"),
        "outputFormatIds": sorted(format_ids),
    }


def fingerprint(value: dict[str, object]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized(value: str) -> bool:
    return bool(value) and value == value.strip().lower()


def validate_catalog(catalog: dict[str, object]) -> tuple[int, int, list[dict[str, object]]]:
    modules = catalog.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("catalog has no modules")
    module_ids: set[str] = set()
    selection_paths: set[str] = set()
    snapshot: list[dict[str, object]] = []
    scheme_count = 0
    for module in modules:
        if not isinstance(module, dict):
            raise RuntimeError("catalog module is malformed")
        module_id = module.get("id")
        selection_path = module.get("selectionPath")
        studio = module.get("studioSelection")
        schemes = module.get("schemes")
        if (
            not isinstance(module_id, str)
            or not normalized(module_id)
            or module_id in module_ids
            or not isinstance(selection_path, str)
            or not selection_path
            or selection_path in selection_paths
            or not isinstance(studio, dict)
            or not isinstance(schemes, list)
            or not schemes
        ):
            raise RuntimeError(f"catalog module is inconsistent: {module_id!r}")
        module_ids.add(module_id)
        selection_paths.add(selection_path)
        mode = studio.get("mode")
        fixed = studio.get("fixedSchemeId")
        reason = studio.get("reason")
        if mode not in ("form", "product_set_v1", "fixed_v1"):
            raise RuntimeError(f"module {module_id} has non-runnable Studio mode {mode!r}")
        if not isinstance(fixed, str) or not isinstance(reason, str):
            raise RuntimeError(f"module {module_id} has malformed Studio policy")

        identities: set[str] = set()
        canonical_ids: list[str] = []
        for scheme in schemes:
            if not isinstance(scheme, dict):
                raise RuntimeError(f"module {module_id} has malformed scheme")
            scheme_id = scheme.get("id")
            aliases = scheme.get("aliases")
            if not isinstance(scheme_id, str) or not normalized(scheme_id):
                raise RuntimeError(f"module {module_id} has invalid scheme id")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) and normalized(alias) for alias in aliases
            ):
                raise RuntimeError(f"scheme {module_id}/{scheme_id} has invalid aliases")
            for identity in (scheme_id, *aliases):
                if identity in identities:
                    raise RuntimeError(
                        f"module {module_id} repeats scheme identity {identity}"
                    )
                identities.add(identity)
            if (
                scheme.get("maturity") != "qualified"
                or scheme.get("constructible") is not True
                or scheme.get("runtimeSelectable") is not True
                or scheme.get("readinessState") != "ready"
            ):
                raise RuntimeError(f"scheme {module_id}/{scheme_id} is not Ready")
            canonical_ids.append(scheme_id)
            scheme_count += 1

        if mode == "form" and (fixed or reason):
            raise RuntimeError(f"form module {module_id} has fixed-policy residue")
        if mode == "fixed_v1" and (fixed not in canonical_ids or not reason.strip()):
            raise RuntimeError(f"fixed module {module_id} has invalid fixed selection")
        if mode == "product_set_v1" and (fixed or not reason.strip()):
            raise RuntimeError(f"product-set module {module_id} has invalid policy")
        snapshot.append(
            {
                "id": module_id,
                "selectionPath": selection_path,
                "studioMode": mode,
                "schemes": canonical_ids,
            }
        )
    return len(modules), scheme_count, snapshot


def validated_readiness(catalog: dict[str, object]) -> dict[str, object]:
    readiness = catalog.get("readiness")
    if not isinstance(readiness, dict) or not isinstance(readiness.get("items"), list):
        raise RuntimeError("catalog readiness registry is malformed")
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not args.reference.is_file():
        raise RuntimeError("catalog authority binary/reference is missing")
    result = subprocess.run(
        [str(binary), "--scheme-catalog=json"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    catalog = json.loads(result.stdout)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    module_count, scheme_count, snapshot = validate_catalog(catalog)
    actual = {
        "schema": catalog.get("schema"),
        "catalogVersion": catalog.get("catalogVersion"),
        "authorityFingerprintSha256": fingerprint(canonical_surface(catalog)),
        "readinessFingerprintSha256": fingerprint(validated_readiness(catalog)),
        "moduleCount": module_count,
        "schemeCount": scheme_count,
        "modules": snapshot,
    }
    if actual != reference:
        raise RuntimeError(
            "exact-binary catalog authority drifted; update the implementation, "
            "cross-language consumers, and authority reference together\n"
            + json.dumps(actual, indent=2, ensure_ascii=False)
        )
    print(
        "[catalog-authority] PASS: "
        f"{module_count} modules/{scheme_count} Ready schemes, "
        f"fingerprint={actual['authorityFingerprintSha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
