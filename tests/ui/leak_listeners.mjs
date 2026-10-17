/*
 * Which listener is left behind by a mount cycle.
 *
 * `leak_metrics.mjs` shows `JSEventListeners` rising by exactly one per mount
 * while node count stays flat — one listener per scene, on something that
 * outlives the scene. A listener is a GC root for everything its closure
 * captured, so one per mount on a long-lived target retains one whole vtk graph
 * per mount, which is the shape of the bytes the probe measures.
 *
 * This censuses the listeners on window and document by type and source
 * location, before and after a batch of cycles, and prints what grew. The
 * source location is the point: "one more resize listener" is a symptom, and
 * the file and line that added it is the bug.
 *
 *   npm run preview:browser-test &
 *   node ../tests/ui/leak_listeners.mjs [cycles] [plot]
 */
import { chromium } from "playwright";

const BASE = process.env.NIMBUS_PROBE_URL ?? "http://127.0.0.1:4178";
const CYCLES = Number(process.argv[2] ?? "8");
const PLOT = process.argv[3] ?? "volume";

const browser = await chromium.launch({
  args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu-sandbox",
    "--js-flags=--expose-gc", "--enable-precise-memory-info"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const cdp = await page.context().newCDPSession(page);
await cdp.send("Runtime.enable");
await cdp.send("Debugger.enable");

/* scriptId -> url, so a listener can be reported at a file and line rather than
 * as an opaque handler identity. */
const scriptUrls = new Map();
cdp.on("Debugger.scriptParsed", (event) => scriptUrls.set(event.scriptId, event.url));

await page.goto(`${BASE}/harness/index.html`);
await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

async function censusOf(expression) {
  const { result } = await cdp.send("Runtime.evaluate", { expression });
  const { listeners } = await cdp.send("DOMDebugger.getEventListeners", {
    objectId: result.objectId,
  });
  const counts = new Map();
  for (const listener of listeners) {
    const url = scriptUrls.get(listener.scriptId) ?? "?";
    const key = `${expression}  ${listener.type}  ${url.split("/").pop()}:${listener.lineNumber + 1}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  await cdp.send("Runtime.releaseObject", { objectId: result.objectId });
  return counts;
}

async function census() {
  const merged = new Map();
  for (const target of ["window", "document", "document.documentElement"]) {
    for (const [key, value] of await censusOf(target)) merged.set(key, value);
  }
  return merged;
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
  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.waitForFunction(() => window.__nimbusVtkHarness?.snapshot().canvases === 0);
}

// Warm up first: the first mounts legitimately install listeners that persist.
for (let i = 0; i < 4; i += 1) await cycle(i);
const before = await census();
for (let i = 0; i < CYCLES; i += 1) await cycle(i);
const after = await census();

console.log(`${PLOT}: listener census across ${CYCLES} mount cycles\n`);
let grew = false;
for (const [key, count] of after) {
  const was = before.get(key) ?? 0;
  if (count > was) {
    grew = true;
    console.log(`  +${count - was}  (${was} -> ${count})  ${key}`);
  }
}
if (!grew) console.log("  no listener on window/document grew — look at per-canvas targets");
await browser.close();
