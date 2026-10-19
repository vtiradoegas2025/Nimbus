import { createHash } from "node:crypto";
import {
  buildVisualizationCapacityQualification,
  REQUIRED_CAPACITY_PROFILES,
} from "./build_visualization_capacity_qualification.ts";
import { parseVisualizationCapacityQualification } from "../../../ui/src/lib/core/visualizationCapacityPolicy.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

const profile = (name: string, capacity: number) => ({
  name,
  status: "passed",
  durationSeconds: 1900,
  admittedCapacityBytes: capacity,
  systemRamBytes: 16 * 1024 ** 3,
  maximumWholeApplicationRssBytes: 2 * 1024 ** 3,
  maximumTrackedCommittedBytes: 400 * 1024 ** 2,
  maximumObservedMinusAccountedBytes: 900 * 1024 ** 2,
  packagedMemoryCoverage: "whole-application",
  gpuResidencyValidated: true,
  opaqueWorkerHeadroomValidated: true,
});
const matrix = {
  schema: "nimbus.studio.visualization-capacity-campaign-matrix",
  schemaVersion: 1,
  status: "passed",
  appVersion: "0.1.0",
  targetOs: "macos",
  targetArch: "aarch64",
  profiles: REQUIRED_CAPACITY_PROFILES.map((name, index) => profile(name, (index + 2) * 512 * 1024 ** 2)),
};
const bytes = Buffer.from(JSON.stringify(matrix));
const artifact = buildVisualizationCapacityQualification(bytes);
check(artifact.qualifiedCapacityBytes === 1024 ** 3,
  "qualification uses the minimum capacity that every required campaign admitted");
check(artifact.campaignDurationSeconds === 1900
  && artifact.evidenceSha256 === createHash("sha256").update(bytes).digest("hex"),
"qualification binds the shortest duration and exact campaign matrix bytes");
check(parseVisualizationCapacityQualification(artifact)?.targetOs === "macos",
  "release output passes the runtime parser");

let missingRejected = false;
try {
  buildVisualizationCapacityQualification(Buffer.from(JSON.stringify({
    ...matrix,
    profiles: matrix.profiles.slice(1),
  })));
} catch { missingRejected = true; }
check(missingRejected, "a missing required campaign cannot produce an artifact");

let shortRejected = false;
try {
  buildVisualizationCapacityQualification(Buffer.from(JSON.stringify({
    ...matrix,
    profiles: matrix.profiles.map((entry, index) => index === 0
      ? { ...entry, durationSeconds: 1799 } : entry),
  })));
} catch { shortRejected = true; }
check(shortRejected, "a short campaign cannot produce an artifact");

let boundedRejected = false;
try {
  buildVisualizationCapacityQualification(Buffer.from(JSON.stringify({
    schema: "nimbus.studio.packaged-visualization-matrix", schemaVersion: 2,
    status: "passed", scope: "bounded-packaged-lifecycle", memoryCampaign: "pending",
  })));
} catch { boundedRejected = true; }
check(boundedRejected, "bounded lifecycle success cannot promote aggregate memory capacity");

console.log(`\nvisualization capacity qualification tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
