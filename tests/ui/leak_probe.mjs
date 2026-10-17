/*
 * Per-mount retention probe.
 *
 * The soak reports one number for a thirty-minute run: how much the page still
 * holds at the end. That is the right alarm and a useless diagnosis — it cannot
 * say which of seventeen plot types is responsible, and re-running it to bisect
 * costs half an hour a guess.
 *
 * The leak signature the soak detects is per-MOUNT, so it can be reproduced in
 * seconds: mount and unmount one plot type repeatedly, force a collection, and
 * read the heap. A plot that releases everything returns to the same number; a
 * plot that strands a buffer per scene climbs by a fixed amount per cycle, and
 * the slope names the culprit and its size.
 *
 * Run against a server that is already up:
 *   npm run preview:browser-test &
 *   node ../tests/ui/leak_probe.mjs [cycles]
 */
import { chromium } from "playwright";

const BASE = process.env.NIMBUS_PROBE_URL ?? "http://127.0.0.1:4178";
const CYCLES = Number(process.argv[2] ?? "24");
/* Discarded before the slope is fit. The first mounts of any plot type pay for
 * its lazy chunk, its shader compiles and its colormap tables — real one-time
 * costs that would otherwise be charged to every later cycle as a slope. */
const WARMUP_CYCLES = Number(process.env.NIMBUS_PROBE_WARMUP ?? "6");

async function settledHeap(page) {
  return page.evaluate(async () => {
    const collect = globalThis.gc;
    window.__nimbusVtkHarness?.clearFieldCache();
    collect();
    await new Promise((r) => setTimeout(r, 60));
    collect();
    return performance.memory.usedJSHeapSize;
  });
}

async function mountCycle(page, plot, step) {
  await page.evaluate(([p, s]) => {
    window.__nimbusVtkHarness?.setPlot(p);
    window.__nimbusVtkHarness?.setStep(s);
  }, [plot, step]);
  await page.waitForFunction((p) =>
    document.querySelector(".harness-shell")?.getAttribute("data-plot") === p, plot);
  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(true));
  await page.waitForFunction(() => window.__nimbusVtkHarness?.snapshot().workers === 0);
  await page.evaluate(() => window.__nimbusVtkHarness?.capture());
  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.waitForFunction(() => window.__nimbusVtkHarness?.snapshot().canvases === 0);
}

/** Least-squares slope, in bytes per cycle. */
function slope(values) {
  const n = values.length;
  const meanX = (n - 1) / 2;
  const meanY = values.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i += 1) {
    num += (i - meanX) * (values[i] - meanY);
    den += (i - meanX) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

const mib = (bytes) => (bytes / (1024 * 1024)).toFixed(1);

const browser = await chromium.launch({
  args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu-sandbox",
    "--js-flags=--expose-gc",
    /*
     * NIMBUS_PROBE_PRECISE=0 drops the flag, which is worth doing once to see
     * what its absence looks like: every plot type and every cycle reports a
     * byte-identical figure, because Chrome quantizes performance.memory and
     * refreshes it at most every ~20 minutes. That reads exactly like "nothing
     * leaks" and is why the soak now asserts its own readings actually move.
     */
    ...(process.env.NIMBUS_PROBE_PRECISE === "0" ? [] : ["--enable-precise-memory-info"])],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto(`${BASE}/harness/index.html`);
await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

const plotTypes = await page.evaluate(() => [...window.__nimbusVtkHarness.plotTypes]);
const only = process.env.NIMBUS_PROBE_PLOT;
const targets = only ? plotTypes.filter((p) => p === only) : plotTypes;

console.log(`probing ${targets.length} plot types, ${CYCLES} cycles each `
  + `(${WARMUP_CYCLES} discarded as warm-up)\n`);
const results = [];
for (const plot of targets) {
  const heap = [];
  for (let cycle = 0; cycle < CYCLES; cycle += 1) {
    await mountCycle(page, plot, cycle % 20);
    heap.push(await settledHeap(page));
  }
  const measured = heap.slice(WARMUP_CYCLES);
  const perCycle = slope(measured);
  results.push({ plot, perCycle, first: measured[0], last: measured[measured.length - 1] });
  console.log(
    `${plot.padEnd(22)} ${(perCycle / 1024).toFixed(0).padStart(8)} KiB/cycle   `
    + `${mib(measured[0])} -> ${mib(measured[measured.length - 1])} MiB`,
  );
}

console.log("\nworst first:");
for (const r of [...results].sort((a, b) => b.perCycle - a.perCycle).slice(0, 6)) {
  console.log(`  ${r.plot.padEnd(22)} ${(r.perCycle / 1024).toFixed(0).padStart(8)} KiB/cycle`);
}
await browser.close();
