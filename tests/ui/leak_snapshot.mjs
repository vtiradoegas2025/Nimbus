/*
 * Which class of object a mount cycle strands.
 *
 * The last resort, and the only tool that names the thing rather than its
 * symptoms. `leak_metrics.mjs` can say "not DOM, not listeners, ~30 KiB of JS
 * per mount"; only the heap itself can say what those bytes are.
 *
 * Takes a V8 heap snapshot before and after a batch of cycles and diffs the
 * per-constructor counts. A class whose instance count grows by exactly the
 * number of cycles is being allocated once per mount and never released — the
 * count is what identifies it, and the self-size total says whether it is the
 * thing holding the bytes or merely a handle onto it.
 *
 *   npm run preview:browser-test &
 *   node ../tests/ui/leak_snapshot.mjs [cycles] [plot]
 */
import { chromium } from "playwright";

const BASE = process.env.NIMBUS_PROBE_URL ?? "http://127.0.0.1:4178";
const CYCLES = Number(process.argv[2] ?? "10");
const PLOT = process.argv[3] ?? "volume";

const browser = await chromium.launch({
  args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu-sandbox",
    "--js-flags=--expose-gc", "--enable-precise-memory-info"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const cdp = await page.context().newCDPSession(page);
await cdp.send("HeapProfiler.enable");
await page.goto(`${BASE}/harness/index.html`);
await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

async function snapshot() {
  await page.evaluate(async () => {
    window.__nimbusVtkHarness?.clearFieldCache();
    globalThis.gc();
    await new Promise((r) => setTimeout(r, 60));
    globalThis.gc();
  });
  let raw = "";
  const collect = (event) => { raw += event.chunk; };
  cdp.on("HeapProfiler.addHeapSnapshotChunk", collect);
  await cdp.send("HeapProfiler.takeHeapSnapshot", { reportProgress: false });
  cdp.off("HeapProfiler.addHeapSnapshotChunk", collect);

  const parsed = JSON.parse(raw);
  const { node_fields: fields, node_types: types } = parsed.snapshot.meta;
  const stride = fields.length;
  const typeIndex = fields.indexOf("type");
  const nameIndex = fields.indexOf("name");
  const sizeIndex = fields.indexOf("self_size");
  const typeNames = types[typeIndex];

  const byClass = new Map();
  for (let i = 0; i < parsed.nodes.length; i += stride) {
    const kind = typeNames[parsed.nodes[i + typeIndex]];
    /* Every node type, because the bytes are usually not where the name is: a
     * typed array's self_size is the handle, and its contents live in a
     * separate "system / ArrayBufferData" node with no constructor name. */
    const name = parsed.strings[parsed.nodes[i + nameIndex]] || "(anonymous)";
    const key = `${kind} ${name}`;
    const entry = byClass.get(key) ?? { count: 0, bytes: 0 };
    entry.count += 1;
    entry.bytes += parsed.nodes[i + sizeIndex];
    byClass.set(key, entry);
  }
  return byClass;
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

for (let i = 0; i < 5; i += 1) await cycle(i);
const before = await snapshot();
for (let i = 0; i < CYCLES; i += 1) await cycle(i);
const after = await snapshot();

const grown = [];
for (const [key, entry] of after) {
  const was = before.get(key) ?? { count: 0, bytes: 0 };
  const dCount = entry.count - was.count;
  const dBytes = entry.bytes - was.bytes;
  if (dCount >= CYCLES * 0.8 || dBytes > 8 * 1024) {
    grown.push({ key, dCount, dBytes, perCycle: dCount / CYCLES });
  }
}
grown.sort((a, b) => b.dBytes - a.dBytes);

console.log(`${PLOT}: ${CYCLES} cycles between snapshots\n`);
console.log("delta count".padStart(12) + "per cycle".padStart(11) + "delta bytes".padStart(13) + "  class");
for (const row of grown.slice(0, 25)) {
  console.log(
    String(row.dCount).padStart(12)
    + row.perCycle.toFixed(1).padStart(11)
    + String(row.dBytes).padStart(13)
    + "  " + row.key,
  );
}
if (grown.length === 0) console.log("  nothing grew by the thresholds");
await browser.close();
