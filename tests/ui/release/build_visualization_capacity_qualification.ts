import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES } from "../../../ui/src/lib/core/visualizationCapacityPolicy.ts";

export const REQUIRED_CAPACITY_PROFILES = [
  "beast-xl-native",
  "four-pane-mixed",
  "memory-pressure-recovery",
] as const;

interface CapacityCampaignProfile {
  name: string;
  status: "passed";
  durationSeconds: number;
  admittedCapacityBytes: number;
  systemRamBytes: number;
  maximumWholeApplicationRssBytes: number;
  maximumTrackedCommittedBytes: number;
  maximumObservedMinusAccountedBytes: number;
  packagedMemoryCoverage: "whole-application";
  gpuResidencyValidated: true;
  opaqueWorkerHeadroomValidated: true;
}

interface CapacityCampaignMatrix {
  schema: "nimbus.studio.visualization-capacity-campaign-matrix";
  schemaVersion: 1;
  status: "passed";
  appVersion: string;
  targetOs: "macos" | "linux" | "windows";
  targetArch: "aarch64" | "x86_64";
  profiles: CapacityCampaignProfile[];
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return Number(value);
}

function parseMatrix(value: unknown): CapacityCampaignMatrix {
  if (!value || typeof value !== "object") throw new Error("capacity campaign matrix must be an object");
  const matrix = value as Partial<CapacityCampaignMatrix>;
  if (matrix.schema !== "nimbus.studio.visualization-capacity-campaign-matrix"
      || matrix.schemaVersion !== 1 || matrix.status !== "passed"
      || typeof matrix.appVersion !== "string" || matrix.appVersion.trim() === ""
      || !["macos", "linux", "windows"].includes(String(matrix.targetOs))
      || !["aarch64", "x86_64"].includes(String(matrix.targetArch))
      || !Array.isArray(matrix.profiles)) {
    throw new Error("capacity campaign matrix header is invalid");
  }
  const names = new Set<string>();
  for (const [index, profile] of matrix.profiles.entries()) {
    if (!profile || profile.status !== "passed" || typeof profile.name !== "string"
        || !profile.name.trim() || names.has(profile.name)
        || profile.packagedMemoryCoverage !== "whole-application"
        || profile.gpuResidencyValidated !== true
        || profile.opaqueWorkerHeadroomValidated !== true) {
      throw new Error(`capacity campaign profile ${index} is invalid`);
    }
    names.add(profile.name);
    const duration = positiveInteger(profile.durationSeconds, `${profile.name}.durationSeconds`);
    const capacity = positiveInteger(profile.admittedCapacityBytes, `${profile.name}.admittedCapacityBytes`);
    const ram = positiveInteger(profile.systemRamBytes, `${profile.name}.systemRamBytes`);
    const rss = positiveInteger(profile.maximumWholeApplicationRssBytes, `${profile.name}.maximumWholeApplicationRssBytes`);
    const tracked = positiveInteger(profile.maximumTrackedCommittedBytes, `${profile.name}.maximumTrackedCommittedBytes`);
    const opaque = positiveInteger(profile.maximumObservedMinusAccountedBytes, `${profile.name}.maximumObservedMinusAccountedBytes`);
    if (duration < 30 * 60 || capacity < BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES
        || capacity >= ram || rss > ram || tracked > capacity || opaque >= ram) {
      throw new Error(`capacity campaign profile ${profile.name} is outside its measured envelope`);
    }
  }
  for (const required of REQUIRED_CAPACITY_PROFILES) {
    if (!names.has(required)) throw new Error(`capacity campaign matrix is missing ${required}`);
  }
  return matrix as CapacityCampaignMatrix;
}

export function buildVisualizationCapacityQualification(
  bytes: Uint8Array,
) {
  const matrix = parseMatrix(JSON.parse(Buffer.from(bytes).toString("utf8")));
  return {
    schema: "nimbus.studio.visualization-capacity-qualification",
    schemaVersion: 2,
    qualifiedCapacityBytes: Math.min(...matrix.profiles.map((profile) => profile.admittedCapacityBytes)),
    minimumSystemRamBytes: Math.min(...matrix.profiles.map((profile) => profile.systemRamBytes)),
    packagedMemoryCoverage: "whole-application",
    gpuResidencyValidated: true,
    opaqueWorkerHeadroomValidated: true,
    profiles: matrix.profiles.map((profile) => profile.name).sort(),
    targetOs: matrix.targetOs,
    targetArch: matrix.targetArch,
    appVersion: matrix.appVersion,
    campaignDurationSeconds: Math.min(...matrix.profiles.map((profile) => profile.durationSeconds)),
    evidenceSha256: createHash("sha256").update(bytes).digest("hex"),
  } as const;
}

function main(args: string[]): void {
  const [input, output] = args;
  if (!input || !output) {
    throw new Error("usage: build_visualization_capacity_qualification.ts CAMPAIGN_MATRIX OUTPUT_JSON");
  }
  const artifact = buildVisualizationCapacityQualification(readFileSync(input));
  mkdirSync(dirname(output), { recursive: true });
  const temporary = `${output}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(artifact, null, 2)}\n`, { flag: "wx" });
  renameSync(temporary, output);
  console.log(JSON.stringify({
    status: "passed",
    target: `${artifact.targetOs}-${artifact.targetArch}`,
    qualifiedCapacityBytes: artifact.qualifiedCapacityBytes,
    evidenceSha256: artifact.evidenceSha256,
  }, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}
