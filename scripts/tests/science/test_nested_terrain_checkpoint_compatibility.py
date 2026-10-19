#!/usr/bin/env python3
"""Continue a historical stationary mapped checkpoint with the current executable."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/reference/checkpoints"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("compatibility evidence directory must be empty")
    metadata = json.loads((FIXTURES / "stationary_mapped_terrain_schema1.json").read_text())
    packed = (FIXTURES / "stationary_mapped_terrain_schema1.nimbus.gz").read_bytes()
    raw = gzip.decompress(packed)
    config = ROOT / metadata["configuration"]
    if sha(packed) != metadata["compressed_sha256"] or sha(raw) != metadata["checkpoint_sha256"]:
        raise RuntimeError("historical checkpoint identity changed")
    if sha(config.read_bytes()) != metadata["configuration_sha256"]:
        raise RuntimeError("historical checkpoint configuration changed")
    with tempfile.TemporaryDirectory(prefix="nimbus-terrain-checkpoint-") as temporary:
        checkpoint = Path(temporary) / "historical.nimbus"
        checkpoint.write_bytes(raw)
        lifecycle = dict(metadata["lifecycle"], checkpoint_file=checkpoint.name)
        checkpoint.with_suffix(".nimbus.json").write_text(json.dumps(lifecycle) + "\n")
        command = [str(args.binary.resolve()), "--headless", "--config", str(config), f"--resume={checkpoint}",
                   "--duration=2", "--write-every=2", "--checkpoint-every=2", "--threads=1",
                   f"--outdir={output / 'continued'}", "--log-profile=normal", "--no-timing"]
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "continuation.log").write_text(result.stdout)
    if result.returncode or "[CHECKPOINT] resumed step=2, time_s=1," not in result.stdout:
        raise RuntimeError(f"historical mapped continuation failed:\n{result.stdout[-6000:]}")
    endpoint = output / "continued/checkpoints/checkpoint_step_000000004.nimbus"
    if not endpoint.is_file():
        raise RuntimeError("historical mapped continuation omitted its new checkpoint")
    (output / "summary.json").write_text(json.dumps({
        "passed": True, "binary_sha256": sha(args.binary.read_bytes()),
        "historical_checkpoint_sha256": metadata["checkpoint_sha256"],
        "continued_checkpoint_sha256": sha(endpoint.read_bytes()),
        "old_active_terrain_schema_version": 1, "continued_time_s": 2,
    }, indent=2) + "\n")
    print("PASS: historical stationary mapped checkpoint continues with the current terrain sampler")


if __name__ == "__main__":
    main()
