import type { VisualizationHardwareFacts } from "./visualizationResourceGovernor";
import { setFieldCacheLimit } from "./fieldCache";
import { fieldLoadScheduler } from "./loadScheduler";
import { visualizationStagingQueue } from "./visualizationStagingQueue";
import { NATIVE_WINDOW_IO_BYTES } from "./nativeWindowBudget";

const MIB = 1024 ** 2;
export interface PlaybackResourceBudget {
  fieldCacheBytes: number;
  fieldLoadBytes: number;
  maxConcurrentLoads: number;
  stagingBytes: number;
}

export function resolvePlaybackResourceBudget(
  hardware: VisualizationHardwareFacts,
  preferences: { loadingWorkers?: number; ioBytes?: number } = {},
): PlaybackResourceBudget {
  const ram = hardware.source === "tauri" && Number.isSafeInteger(hardware.systemRamBytes)
    && hardware.systemRamBytes! > 0 ? hardware.systemRamBytes! : null;
  const cores = Number.isSafeInteger(hardware.logicalCores) && hardware.logicalCores! > 0
    ? hardware.logicalCores! : null;
  const fieldCacheBytes = ram == null ? 256 * MIB
    : Math.min(2 * 1024 * MIB, Math.max(32 * MIB, Math.floor(ram / 64)));
  const hardwareLoadBytes = ram == null ? 384 * MIB
    : Math.min(1024 * MIB, Math.max(64 * MIB, Math.floor(ram / 64)));
  const fieldLoadBytes = Number.isSafeInteger(preferences.ioBytes) && preferences.ioBytes! > 0
    ? preferences.ioBytes! : Math.min(NATIVE_WINDOW_IO_BYTES, hardwareLoadBytes);
  const coreLimit = cores == null ? 3 : Math.max(1, Math.floor(cores / 4));
  const byteLimit = Math.max(1, Math.floor(hardwareLoadBytes / (128 * MIB)));
  return {
    fieldCacheBytes,
    fieldLoadBytes,
    maxConcurrentLoads: Number.isSafeInteger(preferences.loadingWorkers) && preferences.loadingWorkers! > 0
      ? Math.min(64, cores ?? 64, preferences.loadingWorkers!)
      : Math.min(ram == null ? 3 : 8, coreLimit, byteLimit),
    stagingBytes: ram == null ? 96 * MIB : fieldLoadBytes,
  };
}

export function configurePlaybackResources(
  hardware: VisualizationHardwareFacts,
  preferences: { loadingWorkers?: number; ioBytes?: number } = {},
): PlaybackResourceBudget {
  const budget = resolvePlaybackResourceBudget(hardware, preferences);
  setFieldCacheLimit(budget.fieldCacheBytes);
  visualizationStagingQueue.setMaximumBytes(budget.stagingBytes);
  fieldLoadScheduler.setLimits(budget.maxConcurrentLoads, budget.fieldLoadBytes);
  return budget;
}
