"""Authoritative identities for maintained public execution cases."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "tests/fixtures/reference/catalog/maintained_execution_inventory_v1.json"


def configured_backend(path: Path) -> str:
    return next((line.strip().split(":", 1)[1].strip().lower()
                 for line in path.read_text().splitlines()
                 if line.strip().startswith("backend:")), "cpu")


def maintained_cases(root: Path = ROOT, inventory: Path = INVENTORY) -> tuple[list[Path], list[Path]]:
    document = json.loads(inventory.read_text())
    if document.get("schema") != "nimbus.maintained-execution-inventory.v1":
        raise RuntimeError("maintained execution inventory schema drifted")
    discovered = {
        "qualified": sorted(path for path in (root / "tests/fixtures/configs/qualification").glob("*.yaml")
                            if "    mode: qualified" in path.read_text().splitlines()),
        "representatives": sorted((root / "configs/1000meterResolution").rglob("*.yaml")),
    }
    for category, paths in discovered.items():
        actual = [{"path": path.relative_to(root).as_posix(), "backend": configured_backend(path)}
                  for path in paths]
        if not actual or document.get(category) != actual:
            raise RuntimeError(f"maintained execution {category} inventory drifted; update its explicit manifest")
    return discovered["qualified"], discovered["representatives"]


def inventory_counts() -> dict[str, int]:
    qualified, representatives = maintained_cases()
    all_cases = qualified + representatives
    cpu = sum(configured_backend(path) == "cpu" for path in all_cases)
    return {"qualifiedFixtures": len(qualified), "publicRepresentatives": len(representatives),
            "catalogCases": len(all_cases), "cpuCases": cpu, "acceleratorCases": len(all_cases)-cpu}


def restart_class_scripts() -> tuple[str, ...]:
    document = json.loads(INVENTORY.read_text())
    classes = document.get("restartClasses", [])
    if (not classes or len(set(classes)) != len(classes)
            or any(not isinstance(path, str) or not path.startswith("scripts/tests/science/")
                   or ".." in Path(path).parts or not (ROOT/path).is_file() for path in classes)):
        raise RuntimeError("maintained restart class inventory is malformed")
    return tuple(classes)
