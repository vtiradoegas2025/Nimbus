import {
  preflightConfigWithBinary,
  type ConfigPreflightRequest,
  type ConfigPreflightReport,
} from "./configPreflight";
import type { RemoteComputeTarget } from "@/lib/platform/remoteCompute";

interface QualificationCase {
  name: string;
  expectedAcceptance: boolean;
  request: ConfigPreflightRequest;
}

export async function qualifyPackagedConfigPreflight(
  input: unknown,
  preflight: (request: ConfigPreflightRequest, target?: RemoteComputeTarget) => Promise<ConfigPreflightReport> =
    (request, target) => preflightConfigWithBinary(request, undefined, target),
) {
  const batch = input as { schema?: unknown; binarySha256?: unknown; cases?: unknown;
    executionTarget?: RemoteComputeTarget | null } | null;
  if (!batch || batch.schema !== "nimbus.studio.packaged-config-qualification.v1" ||
      typeof batch.binarySha256 !== "string" || !/^[a-f0-9]{64}$/.test(batch.binarySha256) ||
      !Array.isArray(batch.cases) || batch.cases.length === 0 || batch.cases.length > 128) {
    throw new Error("Packaged configuration qualification requires a bounded batch and binary SHA-256");
  }
  const binaryFingerprint = `sha256:${batch.binarySha256}`;
  const names = new Set<string>();
  const cases = batch.cases as QualificationCase[];
  for (const test of cases) {
    if (!test || typeof test.name !== "string" || !test.name.trim() || names.has(test.name) ||
        typeof test.expectedAcceptance !== "boolean" || test.request?.source?.kind !== "exact_document") {
      throw new Error("Packaged configuration qualification requires unique exact-document cases");
    }
    names.add(test.name);
  }
  const checks = [];
  const reports = [];
  let catalogFingerprint: string | undefined;
  let canonicalPath: string | undefined;
  for (const test of cases) {
    const report = await preflight(test.request, batch.executionTarget ?? undefined);
    const expectedState = test.expectedAcceptance ? "accepted" : "rejected";
    if (report.compatible !== test.expectedAcceptance || report.nativeAdmission.state !== expectedState ||
        report.executable.binaryFingerprint !== binaryFingerprint ||
        report.nativeAdmission.validatedBinarySha256 !== binaryFingerprint ||
        (catalogFingerprint !== undefined && report.executable.catalogFingerprint !== catalogFingerprint) ||
        (canonicalPath !== undefined && report.executable.canonicalPath !== canonicalPath)) {
      throw new Error(`Packaged configuration/native agreement failed for ${test.name}`);
    }
    catalogFingerprint = report.executable.catalogFingerprint;
    canonicalPath = report.executable.canonicalPath;
    reports.push({ name: test.name, expectedAcceptance: test.expectedAcceptance, report });
    checks.push({ name: `packaged-config-${test.name}`, passed: true, detail: expectedState });
  }
  return { schema: batch.schema, binarySha256: batch.binarySha256, catalogFingerprint, canonicalPath, checks, reports };
}
