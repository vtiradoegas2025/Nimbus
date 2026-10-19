import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

export const STARTUP_CHECKS = ["native-runtime", "native-bridge", "simulator-discovery",
  "live-catalog", "rendered-shell", "navigation", "no-startup-errors"] as const;

export function validatePackagedStartupEvidence(value: unknown, expected: {
  mode: "debug" | "release"; targetOs: string; targetArch: string;
}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("startup report must be an object");
  const report = value as Record<string, unknown>;
  if (report.schema !== "nimbus.studio.packaged-startup-check" || report.schemaVersion !== 1
      || report.profile !== "startup" || report.scope !== "build-and-open" || report.status !== "passed"
      || report.graphicsValidation !== "pending" || report.scienceQualification !== "pending") {
    throw new Error(`not a passed build-and-open report: ${String(report.error ?? report.status)}`);
  }
  const runtime = report.packagedRuntime as Record<string, unknown> | undefined;
  if (!runtime || runtime.buildMode !== expected.mode || runtime.targetOs !== expected.targetOs
      || runtime.targetArch !== expected.targetArch || typeof runtime.appVersion !== "string" || !runtime.appVersion) {
    throw new Error("startup runtime identity does not match the requested native build");
  }
  if (typeof report.elapsedMs !== "number" || !Number.isFinite(report.elapsedMs) || report.elapsedMs < 0) {
    throw new Error("startup elapsed time must be finite and nonnegative");
  }
  const checks = report.checks as Array<{ name?: unknown; passed?: unknown }> | undefined;
  if (!Array.isArray(checks) || checks.length !== STARTUP_CHECKS.length
      || checks.some(check => !check || check.passed !== true)
      || new Set(checks.map(check => check.name)).size !== checks.length
      || STARTUP_CHECKS.some(name => !checks.some(check => check.name === name))) {
    throw new Error("startup evidence contains missing, duplicate or failed checks");
  }
  return report;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  try {
    const [path, mode, targetOs, targetArch] = process.argv.slice(2);
    if (!path || (mode !== "debug" && mode !== "release") || !targetOs || !targetArch) {
      throw new Error("usage: validate_packaged_startup_evidence.ts REPORT MODE OS ARCH");
    }
    const report = validatePackagedStartupEvidence(JSON.parse(readFileSync(path, "utf8")), { mode, targetOs, targetArch });
    console.log(JSON.stringify({ status: report.status, scope: report.scope, graphicsValidation: report.graphicsValidation }));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
