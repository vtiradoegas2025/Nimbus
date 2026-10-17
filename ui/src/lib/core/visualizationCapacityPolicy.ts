import type { VisualizationHardwareFacts } from "./visualizationResourceGovernor";

/** Exact aggregate commitment measured by the maintained one-pane mounted
 * 256×256×160 isosurface→streamline browser gate on every qualified engine. */
export const BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES = 422_363_312;

export interface VisualizationCapacityQualification {
  schema: "nimbus.studio.visualization-capacity-qualification";
  schemaVersion: 2;
  qualifiedCapacityBytes: number;
  minimumSystemRamBytes: number;
  packagedMemoryCoverage: "host-only" | "parent-descendants" | "whole-application";
  gpuResidencyValidated: boolean;
  opaqueWorkerHeadroomValidated: boolean;
  profiles: readonly string[];
  targetOs: "macos" | "linux" | "windows";
  targetArch: "aarch64" | "x86_64";
  appVersion: string;
  campaignDurationSeconds: number;
  evidenceSha256: string;
}

export interface VisualizationCapacityDecision {
  mode: "observe-only" | "enforced";
  capacityBytes: number | null;
  observedFloorBytes: number;
  blockers: readonly string[];
  qualificationProfiles: readonly string[];
}

let activeQualification: VisualizationCapacityQualification | null = null;

export function parseVisualizationCapacityQualification(
  value: unknown,
): VisualizationCapacityQualification | null {
  if (value == null) return null;
  if (!value || typeof value !== "object") {
    throw new Error("Visualization capacity qualification must be an object");
  }
  const candidate = value as Record<string, unknown>;
  const profiles = candidate.profiles;
  if (candidate.schema !== "nimbus.studio.visualization-capacity-qualification"
      || candidate.schemaVersion !== 2
      || !positiveSafeInteger(candidate.qualifiedCapacityBytes as number)
      || !positiveSafeInteger(candidate.minimumSystemRamBytes as number)
      || Number(candidate.qualifiedCapacityBytes) >= Number(candidate.minimumSystemRamBytes)
      || candidate.packagedMemoryCoverage !== "whole-application"
      || candidate.gpuResidencyValidated !== true
      || candidate.opaqueWorkerHeadroomValidated !== true
      || !Array.isArray(profiles) || profiles.length === 0
      || profiles.some((profile) => typeof profile !== "string" || profile.trim() === "")
      || !["macos", "linux", "windows"].includes(String(candidate.targetOs))
      || !["aarch64", "x86_64"].includes(String(candidate.targetArch))
      || typeof candidate.appVersion !== "string" || candidate.appVersion.trim() === ""
      || !Number.isSafeInteger(candidate.campaignDurationSeconds)
      || Number(candidate.campaignDurationSeconds) < 30 * 60
      || typeof candidate.evidenceSha256 !== "string"
      || !/^[0-9a-f]{64}$/.test(candidate.evidenceSha256)) {
    throw new Error("Visualization capacity qualification is invalid or incomplete");
  }
  return Object.freeze({
    ...(candidate as unknown as VisualizationCapacityQualification),
    profiles: Object.freeze([...(profiles as string[])]),
  });
}

export function installVisualizationCapacityQualification(
  qualification: VisualizationCapacityQualification | null,
): void {
  activeQualification = qualification;
}

export function getVisualizationCapacityQualification(): VisualizationCapacityQualification | null {
  return activeQualification;
}

function positiveSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

export function resolveVisualizationCapacity(
  hardware: VisualizationHardwareFacts,
  qualification: VisualizationCapacityQualification | null = null,
): VisualizationCapacityDecision {
  const blockers: string[] = [];
  if (!qualification) {
    blockers.push(
      "packaged whole-application RSS is not qualified",
      "GPU residency is not qualified",
      "opaque worker/native scratch headroom is not qualified",
    );
  } else {
    if (qualification.schema !== "nimbus.studio.visualization-capacity-qualification"
        || qualification.schemaVersion !== 2) {
      blockers.push("visualization capacity qualification schema is unsupported");
    }
    if (!positiveSafeInteger(qualification.qualifiedCapacityBytes)
        || qualification.qualifiedCapacityBytes < BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES) {
      blockers.push("qualified capacity is below the observed one-pane Beast commitment");
    }
    if (!positiveSafeInteger(qualification.minimumSystemRamBytes)) {
      blockers.push("qualified minimum system RAM is invalid");
    }
    if (qualification.packagedMemoryCoverage !== "whole-application") {
      blockers.push("packaged whole-application RSS is not qualified");
    }
    if (!qualification.gpuResidencyValidated) blockers.push("GPU residency is not qualified");
    if (!qualification.opaqueWorkerHeadroomValidated) {
      blockers.push("opaque worker/native scratch headroom is not qualified");
    }
    if (qualification.profiles.length === 0) blockers.push("qualification names no runtime profiles");
    if (hardware.systemRamBytes == null) {
      blockers.push("system RAM is unavailable for capacity admission");
    } else if (positiveSafeInteger(qualification.minimumSystemRamBytes)
        && hardware.systemRamBytes < qualification.minimumSystemRamBytes) {
      blockers.push("system RAM is below the qualified hardware envelope");
    }
  }

  return Object.freeze({
    mode: blockers.length === 0 ? "enforced" : "observe-only",
    capacityBytes: blockers.length === 0 ? qualification!.qualifiedCapacityBytes : null,
    observedFloorBytes: BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES,
    blockers: Object.freeze(blockers),
    qualificationProfiles: Object.freeze([...(qualification?.profiles ?? [])]),
  });
}
