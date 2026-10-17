import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import {
  validatePackagedVisualizationEvidence,
  type EvidenceExpectation,
} from "./validate_packaged_visualization_evidence.ts";

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

export function aggregatePackagedVisualizationEvidence(inputDirectory: string) {
  const entries = PACKAGED_VISUALIZATION_MATRIX.map((profile) => {
    const path = join(inputDirectory, profile.file);
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
    return {
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
    };
  });
  const versions = new Set(entries.map((entry) => entry.appVersion));
  if (versions.size !== 1) throw new Error("packaged visualization matrix mixes app versions");
  return {
    schema: "nimbus.studio.packaged-visualization-matrix",
    schemaVersion: 1,
    status: "passed",
    profile: "production",
    appVersion: entries[0]?.appVersion,
    entries,
  } as const;
}

function main(args: string[]): void {
  const [inputDirectory, output] = args;
  if (!inputDirectory || !output) {
    throw new Error("usage: aggregate_packaged_visualization_evidence.ts INPUT_DIR OUTPUT_JSON");
  }
  const aggregate = aggregatePackagedVisualizationEvidence(inputDirectory);
  mkdirSync(dirname(output), { recursive: true });
  const temporary = `${output}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(aggregate, null, 2)}\n`, { flag: "wx" });
  renameSync(temporary, output);
  console.log(JSON.stringify({
    status: aggregate.status,
    profiles: aggregate.entries.length,
    appVersion: aggregate.appVersion,
    maximumWatchdogDelayMs: Math.max(...aggregate.entries.map((entry) => entry.maximumWatchdogDelayMs)),
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
