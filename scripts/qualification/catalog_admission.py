"""Reviewed admission baselines and bounded-memory tuple verification."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "tests/fixtures/reference/catalog/generated_admission_baseline_v1.json"
SOUNDING_PATHS = (
    ROOT / "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt",
    ROOT / "tests/fixtures/configs/qualification/soundings/catalog_cm1_profile.txt",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline():
    value = json.loads(BASELINE_PATH.read_text())
    if value.get("schema") != "nimbus.generated-admission-baseline.v1" or value.get("version") != 1:
        raise RuntimeError("generated admission baseline schema drifted")
    generated, admission, preflight = (value[name] for name in ("generated", "admission", "preflight"))
    if (generated["summary"]["accepted"] != admission["summary"]["input"]
            or admission["summary"]["input"] != preflight["summary"]["input"]
            or admission["inputTupleSha256"] != generated["acceptedTupleSha256"]
            or preflight["inputTupleSha256"] != generated["acceptedTupleSha256"]
            or admission["summary"]["admitted"] != preflight["summary"]["scienceAdmitted"]
            or preflight["summary"]["preflightAdmitted"] != value["yamlReload"]["summary"]["inputTuples"]):
        raise RuntimeError("generated admission baseline stages disagree")
    return value


def input_identity_args():
    generated = load_baseline()["generated"]
    return ["--expected-count", str(generated["summary"]["accepted"]),
            "--expected-sha256", generated["acceptedTupleSha256"]]


def baseline_records(record):
    return {"admissionBaseline": record(BASELINE_PATH), "admissionPipeline": record(Path(__file__))}


def verify_baseline_records(value, record):
    for key, expected in baseline_records(record).items():
        if value.get(key) != expected:
            raise RuntimeError(f"{key} identity drifted")


def sounding_records(record):
    return [record(path) for path in SOUNDING_PATHS]


def verify_sounding_records(value, record):
    if value.get("soundingInputs") != sounding_records(record):
        raise RuntimeError("sounding input identities drifted")


def verify_provenance_path(value, label, qualifier):
    paths = {
        "source": ROOT / "tests/compute/generated_backend_tuple_catalog_main.cpp",
        "generatedSource": ROOT / "tests/compute/generated_backend_tuple_catalog_main.cpp",
        "nativeSource": ROOT / "tests/compute/native_admission_batch_main.cpp",
        "runtimeAdmissionSource": ROOT / "src/core/runtime/nimbus_sim.cpp",
        "canonicalBase": ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml",
        "qualifier": qualifier.resolve(),
    }
    if label in paths and value["path"] != paths[label].relative_to(ROOT).as_posix():
        raise RuntimeError(f"{label} source path drifted")


@contextmanager
def native_stream(generated_tool, native_tool, arguments=(), *, signed=False):
    processes = []
    paths = (generated_tool, native_tool, BASELINE_PATH, Path(__file__))
    identities = {path: sha256_file(path) for path in paths}
    with tempfile.TemporaryFile() as producer_errors, tempfile.TemporaryFile() as consumer_errors:
        try:
            producer = subprocess.Popen(
                [str(generated_tool), "--tuples-with-signatures" if signed else "--tuples"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=producer_errors)
            processes.append(producer)
            consumer = subprocess.Popen([str(native_tool), *arguments, *input_identity_args()],
                                        cwd=ROOT, stdin=producer.stdout, stdout=subprocess.PIPE,
                                        stderr=consumer_errors)
            processes.append(consumer)
            producer.stdout.close()
            yield consumer.stdout
            consumer.stdout.close()
            consumer_status, producer_status = consumer.wait(), producer.wait()
            if producer_status or consumer_status:
                producer_errors.seek(0)
                consumer_errors.seek(0)
                raise RuntimeError(f"native admission pipeline failed ({producer_status}, {consumer_status}): "
                                   + producer_errors.read().decode(errors="replace")
                                   + consumer_errors.read().decode(errors="replace"))
            for path, expected in identities.items():
                if sha256_file(path) != expected:
                    raise RuntimeError("native admission pipeline inputs changed during execution")
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stdout is not None:
                    process.stdout.close()


def pipeline_bytes(generated_tool, native_tool, arguments=()):
    with native_stream(generated_tool, native_tool, arguments) as stream:
        return stream.read()


def tuple_partition(text):
    fields = text.split("|", 3)
    if len(fields) < 3 or fields[0] not in ("grid=c_grid", "grid=collocated"):
        raise RuntimeError("canonical tuple grid is invalid")
    if fields[1] not in ("backend=cpu", "backend=gpu") or fields[2] not in ("qualified=true", "qualified=false"):
        raise RuntimeError("canonical tuple partition is invalid")
    return (fields[0][5:] + "." + fields[1][8:] + "."
            + ("qualified" if fields[2] == "qualified=true" else "exploratory"))


def stream_identity(stream, expected_count, expected_digest, expected_partitions):
    digest, seen = hashlib.sha256(), set()
    partitions = Counter({key: 0 for key in expected_partitions})
    for raw in stream:
        if not raw.endswith(b"\n") or raw == b"\n":
            raise RuntimeError("admitted tuple stream contains a malformed line")
        identity = hashlib.sha256(raw).digest()
        if identity in seen:
            raise RuntimeError("admitted tuple stream contains duplicate tuples")
        seen.add(identity)
        digest.update(raw)
        partition = tuple_partition(raw[:-1].decode("ascii"))
        if partition not in partitions:
            raise RuntimeError("admitted tuple stream has an unknown partition")
        partitions[partition] += 1
    if len(seen) != expected_count or dict(partitions) != expected_partitions:
        raise RuntimeError("admitted tuple stream cardinality or partitions drifted")
    if digest.hexdigest() != expected_digest:
        raise RuntimeError("admitted tuple stream SHA-256 drifted")
    return digest.hexdigest(), dict(partitions)


def parse_tuple(text):
    result = {}
    for field in text.split("|"):
        key, separator, value = field.partition("=")
        if not separator or not key or not value or key in result:
            raise RuntimeError("canonical tuple field is malformed")
        result[key] = value
    return result


def select_witnesses(raw):
    stream = io.BytesIO(raw) if isinstance(raw, bytes) else raw
    ownership, witnesses, selected_signatures = {}, {}, {}
    seen, digest = set(), hashlib.sha256()
    for line in stream:
        if not line.endswith(b"\n"):
            raise RuntimeError("signed admitted tuple line lacks its terminator")
        digest.update(line)
        signature_raw, separator, tuple_raw = line[:-1].partition(b"\t")
        if not separator or not signature_raw or not tuple_raw:
            raise RuntimeError("signed admitted tuple line is malformed")
        identity = hashlib.sha256(tuple_raw).digest()
        if identity in seen:
            raise RuntimeError("signed admitted tuple stream contains duplicate tuples")
        seen.add(identity)
        signature, text = signature_raw.decode("ascii"), tuple_raw.decode("ascii")
        values = parse_tuple(text)
        partition = (values["grid"], values["backend"], values["qualified"])
        selected = signature not in ownership
        ownership.setdefault(signature, text)
        for field, value in values.items():
            if field not in {"grid", "backend", "qualified"}:
                key = (*partition, field, value)
                selected = selected or key not in witnesses
                witnesses.setdefault(key, text)
        if selected:
            selected_signatures[text] = signature
    selected = [(selected_signatures[text], text) for text in sorted(selected_signatures)]
    selected_digest, partitions = hashlib.sha256(), Counter()
    for signature, text in selected:
        selected_digest.update(f"{signature}\t{text}\n".encode())
        values = parse_tuple(text)
        partitions[".".join((values["grid"], values["backend"], values["qualified"]))] += 1
    summary = {"inputTuples": len(seen), "ownershipSignatures": len(ownership),
               "axisRequirements": len(witnesses), "representatives": len(selected),
               "representativeSha256": selected_digest.hexdigest(),
               "partitions": dict(sorted(partitions.items()))}
    return selected, summary, digest.hexdigest()
