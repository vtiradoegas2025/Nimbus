import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import {
  preparedDatasetRegistrySnapshot,
  resolvePreparedDataset,
  retainPreparedDataset,
  retainRemotePreparedDataset,
} from "../../../../ui/src/lib/vtk/vtkPreparedDatasetRegistry.ts";
import type { PreparedVtkImage } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else { failed += 1; failures.push(`FAIL ${label}`); }
}

const field = {
  shape: [3, 3, 3] as [number, number, number],
  data: Float32Array.from({ length: 27 }, (_, index) => index),
};
const prepared = prepareVtkRequest({
  kind: "scalar",
  field,
  grid: { drM: 1, dyM: 1, dzM: 1 },
});
const first = retainPreparedDataset(prepared, "scalar");
const second = retainPreparedDataset(prepared, "scalar");
check(first.handle === second.handle, "equivalent consumers share one opaque handle");
check(Object.isFrozen(first.handle), "handle metadata is immutable");
check(Object.isFrozen(first.handle.metadata), "prepared metadata projection is immutable");
check(!("values" in first.handle.metadata), "opaque metadata excludes the resident lattice");
check(first.handle.bytes === prepared.values.byteLength, "handle reports exact owned bytes");
check(resolvePreparedDataset(first.handle) === prepared, "retained handle resolves its dataset");
check(preparedDatasetRegistrySnapshot().handles === 1, "registry owns one dataset entry");
check(preparedDatasetRegistrySnapshot().references === 2, "registry counts independent leases");
check(first.active && second.active, "new leases are active");
first.release();
check(!first.active && second.active, "lease activity is independent of shared backing");
first.release();
check(preparedDatasetRegistrySnapshot().references === 1, "lease release is idempotent");
check(resolvePreparedDataset(second.handle) === prepared, "one release preserves another consumer");
second.release();
check(!second.active, "final release invalidates the lease");
check(preparedDatasetRegistrySnapshot().handles === 0, "last release evicts the registry entry");
check(
  (() => { try { resolvePreparedDataset(second.handle); return false; } catch { return true; } })(),
  "released handles fail closed",
);

const remoteBacking = {};
let remoteReleases = 0;
const remoteOptions = {
  kind: "scalar" as const,
  bytes: first.handle.bytes,
  metadata: first.handle.metadata,
  onLastRelease: () => { remoteReleases += 1; },
};
const remoteFirst = retainRemotePreparedDataset<PreparedVtkImage>(remoteBacking, remoteOptions);
const remoteSecond = retainRemotePreparedDataset<PreparedVtkImage>(remoteBacking, remoteOptions);
check(remoteFirst.handle === remoteSecond.handle, "remote consumers share one worker handle");
check(remoteFirst.handle.residency === "worker", "remote handle discloses worker residency");
check(preparedDatasetRegistrySnapshot().workerHandles === 1, "registry separates remote handle count");
check(preparedDatasetRegistrySnapshot().workerBytes === first.handle.bytes, "registry reports remote logical bytes");
check(
  (() => { try { resolvePreparedDataset(remoteFirst.handle); return false; } catch { return true; } })(),
  "worker-resident handle cannot be resolved on main",
);
remoteFirst.release();
check(remoteReleases === 0, "one remote lease preserves worker backing");
remoteSecond.release();
check(remoteReleases === 1, "last remote lease releases worker backing once");
check(preparedDatasetRegistrySnapshot().handles === 0, "remote handle leaves no registry entry");

console.log(`\nvtkPreparedDatasetRegistry tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
