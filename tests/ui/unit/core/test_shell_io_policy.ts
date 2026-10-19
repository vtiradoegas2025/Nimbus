/* Static regressions for shell-level reads/writes that must stay off hot paths. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

let passed = 0;
const check = (condition: boolean, label: string) => {
  if (!condition) throw new Error(label);
  passed += 1;
};
const source = (relative: string) =>
  readFileSync(fileURLToPath(new URL(`../../../../ui/${relative}`, import.meta.url)), "utf8");

const app = source("src/App.tsx");
const hooks = source("src/lib/core/hooks.ts");
const settings = source("src/components/settings/Settings.tsx");
const scope = source("src/lib/plot/threeDControlScope.ts");

check(
  app.includes('useRuns(activePage === "runs")'),
  "durable run-library scans are gated by the visible Runs page",
);
check(
  app.includes("<Settings simulator={simulator} />"),
  "Settings consumes the shell's shared simulator probe",
);
check(
  !settings.includes("useSimulator()"),
  "Settings does not create a duplicate native simulator probe",
);
check(
  hooks.includes('document.visibilityState !== "hidden"'),
  "run polling pauses while the document is hidden",
);

const adjustStart = hooks.indexOf("const adjust = useCallback(");
const adjustEnd = hooks.indexOf("return [width, adjust]", adjustStart);
check(adjustStart >= 0 && adjustEnd > adjustStart, "resize persistence block is found");
check(
  !hooks.slice(adjustStart, adjustEnd).includes("localStorage.setItem"),
  "pointer-move resize updates do not synchronously write localStorage",
);
check(
  hooks.includes("persistencePendingRef") && hooks.includes("setTimeout(() =>"),
  "resize persistence is coalesced and flushed",
);
check(
  hooks.includes('addEventListener("pagehide", flushPendingWidth)'),
  "a page navigation flushes the final coalesced width",
);
check(
  scope.includes("try {") && scope.includes('localStorage.getItem(SCOPE_STORAGE_KEY)'),
  "3-D control scope tolerates blocked localStorage reads",
);

console.log(`shell I/O policy: ${passed} checks passed`);
