/* Static regressions for event-efficient playback and camera attachment loops. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(label: string, condition: boolean): void {
  if (condition) passed++;
  else {
    failed++;
    failures.push(label);
  }
}

const studioPath = fileURLToPath(new URL("../../../../ui/src/components/studio/PlotStudio.tsx", import.meta.url));
const studio = readFileSync(studioPath, "utf8");
const playbackStart = studio.indexOf("// Auto-advance loop");
const playbackEnd = studio.indexOf("// Warm the upcoming steps", playbackStart);
const playback = studio.slice(playbackStart, playbackEnd);
check("playback implementation block is found", playbackStart >= 0 && playbackEnd > playbackStart);
check("playback does not use a fixed interval", !playback.includes("setInterval"));
check("playback uses adaptive timeout scheduling", playback.includes("setTimeout(checkGate"));
check("frame reports can wake the playback gate", studio.includes("playbackGateWakeRef.current?.()"));
check("playback retry delay has a bounded backoff", studio.includes("PLAYBACK_GATE_MAX_RETRY_MS"));
check("project autosave pauses while frame playback is active", studio.includes("loading || playing) return"));
check("project updates do not write storage inside React state updaters", !studio.includes("setProjectState((current) =>"));
check("recent dataset updates do not write storage inside React state updaters", !studio.includes("setRecentDatasets((current) =>"));

const plotCommonPath = fileURLToPath(new URL("../../../../ui/src/components/plots/common.tsx", import.meta.url));
const plotCommon = readFileSync(plotCommonPath, "utf8");
const fieldPrefetchStart = plotCommon.indexOf("if (cancelled || !loadedSuccessfully || !prefetchEnabled) return");
const fieldPrefetchEnd = plotCommon.indexOf("return () => {", fieldPrefetchStart);
const fieldPrefetch = plotCommon.slice(fieldPrefetchStart, fieldPrefetchEnd);
check("3-D field prefetch block is found", fieldPrefetchStart >= 0 && fieldPrefetchEnd > fieldPrefetchStart);
check("3-D field prefetch is cancellable", fieldPrefetch.includes("createPrefetchController()"));
check("3-D field prefetch uses background scheduler priority", fieldPrefetch.includes('priority: "background"'));
check("3-D field prefetch releases its cancellation controller", fieldPrefetch.includes("releasePrefetchController(prefetchController)"));

for (const [label, relativePath] of [
  ["time-series", "../../../../ui/src/components/plots/TimeSeriesPlot.tsx"],
  ["time-height", "../../../../ui/src/components/plots/TimeHeightPlot.tsx"],
] as const) {
  const source = readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), "utf8");
  check(`${label} retains completed live-run reductions`, source.includes("reductionCacheRef"));
  check(`${label} requests only uncached frames`, source.includes("const outstanding = requests.filter"));
}

const cameraPath = fileURLToPath(new URL("../../../../ui/src/lib/vtk/useCameraOrientation.ts", import.meta.url));
const camera = readFileSync(cameraPath, "utf8");
const bindStart = camera.indexOf("const bind = () =>");
const bindEnd = camera.indexOf("return () =>", bindStart);
const binding = camera.slice(bindStart, bindEnd);
check("camera binding implementation block is found", bindStart >= 0 && bindEnd > bindStart);
check("camera binding no longer polls with animation frames", !binding.includes("requestAnimationFrame(bind)"));
check("camera binding retries with a timeout", binding.includes("setTimeout(bind"));
check("camera retry delay is capped", binding.includes("CAMERA_BIND_MAX_DELAY_MS"));

console.log(`\nruntime-scheduler tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
