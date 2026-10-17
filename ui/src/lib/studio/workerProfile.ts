import type { HardwareInfo } from "@/lib/core/types";

export const WORKER_BENCHMARK = "npy-load-scan-v1";
const STORAGE_KEY = "nimbus-studio:worker-profile:v1";
export interface LoadingWorkerTrial {
  workers: number;
  medianMs: number;
  throughputMibS: number;
}
export interface LoadingWorkerProfile {
  schema: "nimbus.studio.worker-profile";
  schemaVersion: 1;
  benchmark: typeof WORKER_BENCHMARK;
  hardware: HardwareInfo;
  recommendedWorkers: number;
  trials: LoadingWorkerTrial[];
  bytesPerTrial: number;
  elapsedMs: number;
}
export function hardwareProfileIdentity(hardware: HardwareInfo): string {
  return JSON.stringify([hardware.cpuBrand, hardware.coreCount, hardware.ramBytes]);
}
export function parseWorkerProfile(value: unknown): LoadingWorkerProfile | null {
  if (!value || typeof value !== "object") return null;
  const p = value as LoadingWorkerProfile;
  if (p.schema !== "nimbus.studio.worker-profile" || p.schemaVersion !== 1 || p.benchmark !== WORKER_BENCHMARK
      || !p.hardware || typeof p.hardware !== "object"
      || p.hardware.cpuBrand !== null && typeof p.hardware.cpuBrand !== "string"
      || p.hardware.coreCount !== null && (!Number.isSafeInteger(p.hardware.coreCount) || p.hardware.coreCount <= 0)
      || p.hardware.ramBytes !== null && (!Number.isSafeInteger(p.hardware.ramBytes) || p.hardware.ramBytes <= 0)
      || !Number.isSafeInteger(p.recommendedWorkers) || p.recommendedWorkers < 1 || p.recommendedWorkers > 16
      || !Number.isSafeInteger(p.bytesPerTrial) || p.bytesPerTrial <= 0
      || !Number.isFinite(p.elapsedMs) || p.elapsedMs <= 0
      || !Array.isArray(p.trials) || p.trials.length === 0 || p.trials.length > 6) return null;
  const seen = new Set<number>();
  for (const t of p.trials) {
    if (!t || !Number.isSafeInteger(t.workers) || t.workers < 1 || t.workers > 16 || seen.has(t.workers)
        || !Number.isFinite(t.medianMs) || t.medianMs <= 0
        || !Number.isFinite(t.throughputMibS) || t.throughputMibS <= 0) return null;
    seen.add(t.workers);
  }
  if (!seen.has(1) || !seen.has(p.recommendedWorkers)) return null;
  if (p.hardware.coreCount != null && p.trials.some(trial => trial.workers > p.hardware.coreCount!)) return null;
  const best = Math.min(...p.trials.map(trial => trial.medianMs));
  const recommended = Math.min(...p.trials.filter(trial => trial.medianMs <= best / 0.95).map(trial => trial.workers));
  if (p.recommendedWorkers !== recommended) return null;
  return p;
}
export function loadWorkerProfile(hardware: HardwareInfo): LoadingWorkerProfile | null {
  try {
    const profile = parseWorkerProfile(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null"));
    return profile && hardwareProfileIdentity(profile.hardware) === hardwareProfileIdentity(hardware) ? profile : null;
  } catch { return null; }
}
export function saveWorkerProfile(profile: LoadingWorkerProfile): boolean {
  if (!parseWorkerProfile(profile)) return false;
  try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(profile)); return true; }
  catch { return false; }
}
