/*
 * What KIND of thing the mount cycle strands.
 *
 * `leak_probe.mjs` says how many bytes per mount and which plot type; it cannot
 * say what they are. These counters can: `Nodes` and `JSEventListeners` rising
 * in step with the cycle count means detached DOM and listeners, which is a
 * different bug and a different fix from a retained typed array.
 *
 * Chrome only reports a live `JSHeapUsedSize` with --enable-precise-memory-info;
 * without it the value is quantized and rate-limited, and every read in a short
 * run returns the same cached bucket. That is not a detail — it is the reason
 * the soak's own retention check could not have been measuring anything.
 *
 *   npm run preview:browser-test &
 *   node ../tests/ui/leak_metrics.mjs [cycles] [plot]
 */
import { chromium } from "playwright";

const BASE = process.env.NIMBUS_PROBE_URL ?? "http://127.0.0.1:4178";
const CYCLES = Number(process.argv[2] ?? "20");
const PLOT = process.argv[3] ?? "volume";
const WARMUP = 6;

const WATCHED = [
  "Nodes", "JSEventListeners", "Documents", "LayoutObjects", "Resources",
  "ContextLifecycleStateObservers", "DetachedScriptStates", "JSHeapUsedSize",
];

const browser = await chromium.launch({
  args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu-sandbox",
    "--js-flags=--expose-gc", "--enable-precise-memory-info"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const cdp = await page.context().newCDPSession(page);
await cdp.send("Performance.enable");

await page.goto(`${BASE}/harness/index.html`);
await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

async function metrics() {
  await page.evaluate(async () => {
    window.__nimbusVtkHarness?.clearFieldCache();
    globalThis.gc();
    await new Promise((r) => setTimeout(r, 60));
    globalThis.gc();
  });
  const { metrics: values } = await cdp.send("Performance.getMetrics");
  return Object.fromEntries(values.map((m) => [m.name, m.value]));
}

async function cycle(step) {
  await page.evaluate(([p, s]) => {
    window.__nimbusVtkHarness?.setPlot(p);
    window.__nimbusVtkHarness?.setStep(s);
  }, [PLOT, step]);
  await page.waitForFunction((p) =>
    document.querySelector(".harness-shell")?.getAttribute("data-plot") === p, PLOT);
  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(true));
  await page.waitForFunction(() => window.__nimbusVtkHarness?.snapshot().workers === 0);
  await page.evaluate(() => window.__nimbusVtkHarness?.capture());
  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.waitForFunction(() => window.__nimbusVtkHarness?.snapshot().canvases === 0);
}

const series = [];
for (let i = 0; i < CYCLES; i += 1) {
  await cycle(i % 20);
  series.push(await metrics());
}

const measured = series.slice(WARMUP);
const first = measured[0];
const last = measured[measured.length - 1];
const n = measured.length - 1;
console.log(`${PLOT}: ${n} measured cycles after ${WARMUP} warm-up\n`);
console.log("metric".padEnd(34) + "first".padStart(12) + "last".padStart(12) + "per cycle".padStart(12));
for (const name of WATCHED) {
  if (first[name] === undefined) continue;
  const perCycle = (last[name] - first[name]) / n;
  console.log(
    name.padEnd(34)
    + String(Math.round(first[name])).padStart(12)
    + String(Math.round(last[name])).padStart(12)
    + perCycle.toFixed(1).padStart(12),
  );
}
await browser.close();
