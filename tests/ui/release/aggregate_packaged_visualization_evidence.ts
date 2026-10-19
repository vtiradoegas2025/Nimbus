import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import {
  validatePackagedVisualizationEvidence,
  type EvidenceExpectation,
} from "./validate_packaged_visualization_evidence.ts";
import { validatePackagedStartupEvidence } from "./validate_packaged_startup_evidence.ts";

export const PACKAGED_VISUALIZATION_MATRIX = [
  { file: "packaged-visualization-macos-arm64.json", targetOs: "macos", targetArch: "aarch64" },
  { file: "packaged-visualization-macos-x64.json", targetOs: "macos", targetArch: "x86_64" },
  { file: "packaged-visualization-linux-arm64.json", targetOs: "linux", targetArch: "aarch64" },
  { file: "packaged-visualization-linux-x64.json", targetOs: "linux", targetArch: "x86_64" },
  { file: "packaged-visualization-windows-arm64.json", targetOs: "windows", targetArch: "aarch64" },
  { file: "packaged-visualization-windows-x64.json", targetOs: "windows", targetArch: "x86_64" },
] as const;

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
}

export function aggregatePackagedVisualizationEvidence(inputDirectory: string, allowStartupOnly: readonly string[] = []) {
  const knownProfiles = new Set<string>(PACKAGED_VISUALIZATION_MATRIX.map(profile => `${profile.targetOs}/${profile.targetArch}`));
  if (new Set(allowStartupOnly).size !== allowStartupOnly.length || allowStartupOnly.some(profile => !knownProfiles.has(profile))) {
    throw new Error("startup-only allowances must name unique maintained profiles");
  }
  const pending: Array<{ file: string; sha256: string; targetOs: string; targetArch: string;
    appVersion: string; scope: string; reason: string }> = [];
  const entries = PACKAGED_VISUALIZATION_MATRIX.flatMap((profile) => {
    const path = join(inputDirectory, profile.file);
    if (!existsSync(path) && allowStartupOnly.includes(`${profile.targetOs}/${profile.targetArch}`)) {
      const startupPath = join(inputDirectory, profile.file.replace("packaged-visualization-", "packaged-startup-"));
      const bytes = readFileSync(startupPath);
      const report = validatePackagedStartupEvidence(JSON.parse(bytes.toString("utf8")), {
        mode: "release", targetOs: profile.targetOs, targetArch: profile.targetArch,
      });
      const runtime = report.packagedRuntime as { appVersion: string };
      pending.push({ file: basename(startupPath), sha256: createHash("sha256").update(bytes).digest("hex"),
        targetOs: profile.targetOs, targetArch: profile.targetArch, appVersion: runtime.appVersion,
        scope: "build-and-open", reason: "Native 3D graphics validation remains pending; startup evidence cannot satisfy it." });
      return [];
    }
    const bytes = readFileSync(path);
    const expected: EvidenceExpectation = {
      profile: "production",
      mode: "release",
      targetOs: profile.targetOs,
      targetArch: profile.targetArch,
    };
    const report = validatePackagedVisualizationEvidence(JSON.parse(bytes.toString("utf8")), expected);
    const runtime = report.packagedRuntime as Record<string, unknown>;
    const resources = report.finalResources as Record<string, unknown>;
    const reconciliation = report.memoryReconciliation as Record<string, unknown>;
    return [{
      file: basename(path),
      sha256: createHash("sha256").update(bytes).digest("hex"),
      targetOs: profile.targetOs,
      targetArch: profile.targetArch,
      appVersion: runtime.appVersion,
      profile: report.profile,
      shape: report.shape,
      scalarBytes: report.scalarBytes,
      vectorBytes: report.vectorBytes,
      checks: (report.checks as unknown[]).length,
      elapsedMs: finiteNumber(report.elapsedMs, `${profile.file}.elapsedMs`),
      maximumWatchdogDelayMs: finiteNumber(
        report.maximumWatchdogDelayMs, `${profile.file}.maximumWatchdogDelayMs`,
      ),
      maximumWorkerWatchdogDelayMs: finiteNumber(report.maximumWorkerWatchdogDelayMs, `${profile.file}.maximumWorkerWatchdogDelayMs`),
      highWaterBytes: finiteNumber(resources.highWaterBytes, `${profile.file}.highWaterBytes`),
      processTreePeakBytes: finiteNumber(
        reconciliation.processTreePeakBytes, `${profile.file}.processTreePeakBytes`,
      ),
      processTreeDeltaBytes: finiteNumber(
        reconciliation.processTreeDeltaBytes, `${profile.file}.processTreeDeltaBytes`,
      ),
      maximumMemorySamplingMs: finiteNumber(
        reconciliation.maximumSamplingMs, `${profile.file}.maximumSamplingMs`,
      ),
      observedMinusAccountedBytes: finiteNumber(
        reconciliation.observedMinusAccountedBytes, `${profile.file}.observedMinusAccountedBytes`,
      ),
    }];
  });
  const versions = new Set([...entries, ...pending].map((entry) => entry.appVersion));
  if (versions.size !== 1) throw new Error("packaged visualization matrix mixes app versions");
  return {
    schema: "nimbus.studio.packaged-visualization-matrix",
    schemaVersion: 4,
    scope: "bounded-packaged-lifecycle",
    memoryCampaign: "pending",
    renderPerformanceCampaign: "pending",
    status: pending.length === 0 ? "passed" : "incomplete",
    profile: "production",
    appVersion: [...versions][0],
    entries,
    pending,
  } as const;
}

function main(args: string[]): void {
  const [inputDirectory, output, ...options] = args;
  if (!inputDirectory || !output) {
    throw new Error("usage: aggregate_packaged_visualization_evidence.ts INPUT_DIR OUTPUT_JSON");
  }
  const allowStartupOnly: string[] = [];
  for (let index = 0; index < options.length; index += 2) {
    if (options[index] !== "--allow-startup-only" || !options[index + 1]) throw new Error("invalid startup-only allowance");
    allowStartupOnly.push(options[index + 1]);
  }
  const aggregate = aggregatePackagedVisualizationEvidence(inputDirectory, allowStartupOnly);
  mkdirSync(dirname(output), { recursive: true });
  const temporary = `${output}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(aggregate, null, 2)}\n`, { flag: "wx" });
  renameSync(temporary, output);
  console.log(JSON.stringify({
    status: aggregate.status,
    profiles: aggregate.entries.length,
    pending: aggregate.pending,
    appVersion: aggregate.appVersion,
    maximumWatchdogDelayMs: aggregate.entries.length > 0
      ? Math.max(...aggregate.entries.map((entry) => entry.maximumWatchdogDelayMs)) : null,
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
