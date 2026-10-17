import { expect, test, type Page } from "@playwright/test";

type Trace = { x?: number[]; y?: (number | null)[]; z?: (number | null)[][]; name?: string };
type PlotHost = HTMLElement & { data: Trace[]; layout: { title?: { text?: string } } };

test.beforeEach(() => { test.setTimeout(45_000); });

async function open(page: Page, plot: string) {
  await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}`);
  await page.waitForFunction(() => Boolean(window.__nimbusRegularHarness));
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption(plot === "swath" ? "column_max_w" : "w");
  await waitLoaded(page);
}

async function waitLoaded(page: Page) {
  await page.waitForFunction(() => {
    if (document.querySelector(".plot-overlay--error")) return true;
    const host = document.querySelector(".js-plotly-plot") as PlotHost | null;
    return (host?.data?.length ?? 0) > 0
      && ![...document.querySelectorAll(".plot-overlay")].some(overlay => /Loading|Accumulating|recomputing/i.test(overlay.textContent ?? ""));
  }, undefined, { timeout: 20_000 });
  expect(await page.locator(".plot-overlay--error").allTextContents()).toEqual([]);
}

async function data(page: Page): Promise<Trace[]> {
  return page.evaluate(() => JSON.parse(JSON.stringify((document.querySelector(".js-plotly-plot") as PlotHost | null)?.data ?? [],
    (_key, value: unknown) => ArrayBuffer.isView(value) && !(value instanceof DataView)
      ? Array.from(value as unknown as ArrayLike<number>) : value)));
}

async function reduceCalls(page: Page, command = "reduce_field") {
  return page.evaluate(command => window.__nimbusRegularHarness!.snapshot().calls
    .filter(call => call.command === command && call.field === "w").length, command);
}

test("series preserves all four reductions, missing-count gaps and exact incremental source reuse", async ({ page }) => {
  await open(page, "series");
  const source = Array.from({ length: 12 }, (_, index) => index + 1);
  const stats = { max: 12, min: 1, mean: 6.5, rms: Math.sqrt(source.reduce((sum, value) => sum + value * value, 0) / source.length) };
  for (const [reducer, expected] of Object.entries(stats)) {
    await page.locator(".control-group", { hasText: "Reducer" }).getByRole("button", { name: reducer, exact: true }).click();
    await expect.poll(async () => (await data(page))[0].y![0]).toBeCloseTo(expected, 10);
  }
  await expect.poll(() => reduceCalls(page)).toBe(3);
  await page.locator(".control-group", { hasText: "Reducer" }).getByRole("button", { name: "max", exact: true }).click();
  await page.evaluate(() => window.__nimbusRegularHarness!.setFrames([0, 1, 2, 3]));
  await expect.poll(async () => (await data(page))[0].y).toEqual([12, 22, 32, 42]);
  expect(await reduceCalls(page)).toBe(4);
  await page.evaluate(() => window.__nimbusRegularHarness!.setFrames([1, 2, 3]));
  await expect.poll(async () => (await data(page))[0].y).toEqual([22, 32, 42]);
  expect(await reduceCalls(page)).toBe(4);
  await page.evaluate(() => window.__nimbusRegularHarness!.revise(2, -100));
  await expect.poll(async () => (await data(page))[0].y).toEqual([22, -68, 42]);
  expect(await reduceCalls(page)).toBe(5);
  await page.evaluate(() => window.__nimbusRegularHarness!.setMissing(2, true));
  await page.locator(".control-group", { hasText: "Reducer" }).getByRole("button", { name: "# >", exact: true }).click();
  await expect.poll(async () => (await data(page))[0].y).toEqual([12, null, 12]);
});

test("time-height source values and coordinates survive style edits, rolling discovery and source replacement", async ({ page }) => {
  await open(page, "height");
  await expect.poll(async () => (await data(page))[0].z).toEqual([[11, 21, 31], [12, 22, 32]]);
  const initial = await reduceCalls(page, "reduce_field_axis");
  await page.evaluate(() => { (window as unknown as { regularZ: unknown }).regularZ = (document.querySelector(".js-plotly-plot") as PlotHost).data[0].z; });
  await page.locator(".control-group", { hasText: "Colormap" }).locator("select").selectOption("Greys");
  await expect.poll(() => page.evaluate(() => (document.querySelector(".js-plotly-plot") as PlotHost).data[0].z
    === (window as unknown as { regularZ: unknown }).regularZ)).toBe(true);
  expect(await reduceCalls(page, "reduce_field_axis")).toBe(initial);
  await page.evaluate(() => window.__nimbusRegularHarness!.setFrames([1, 2, 3]));
  await expect.poll(async () => (await data(page))[0].z).toEqual([[21, 31, 41], [22, 32, 42]]);
  expect(await reduceCalls(page, "reduce_field_axis")).toBe(initial + 1);
  await page.evaluate(() => window.__nimbusRegularHarness!.revise(2, -100));
  await expect.poll(async () => (await data(page))[0].z).toEqual([[21, -69, 41], [22, -68, 42]]);
  expect(await reduceCalls(page, "reduce_field_axis")).toBe(initial + 2);
});

test("swath replaces revised extrema and removes unavailable frames instead of accumulating stale source versions", async ({ page }) => {
  const workerUrls: string[] = [];
  const errors: string[] = [];
  page.on("request", request => { if (request.url().includes("swathAccumulator.worker")) workerUrls.push(request.url()); });
  page.on("pageerror", error => { errors.push(error.message); });
  await open(page, "swath");
  await expect.poll(async () => (await data(page))[0].z).toEqual([[22, 24], [26, 28], [30, 32]]);
  await page.evaluate(() => {
    (window as unknown as { previousSwath: Element | null }).previousSwath = document.querySelector(".js-plotly-plot");
    window.__nimbusRegularHarness!.setDelay("column_max_w", 200);
    window.__nimbusRegularHarness!.revise(2, -100);
  });
  await expect(page.locator(".plot-overlay").filter({ hasText: "Displaying previous completed swath" })).toBeVisible();
  expect((await data(page))[0].z).toEqual([[22, 24], [26, 28], [30, 32]]);
  await expect.poll(async () => (await data(page))[0].z).toEqual([[12, 14], [16, 18], [20, 22]]);
  expect(await page.evaluate(() => document.querySelector(".js-plotly-plot")
    === (window as unknown as { previousSwath: Element | null }).previousSwath)).toBe(true);
  await page.evaluate(() => window.__nimbusRegularHarness!.setDelay("column_max_w", 0));
  await page.evaluate(() => window.__nimbusRegularHarness!.setFrames([0, 2]));
  await expect.poll(async () => (await data(page))[0].z).toEqual([[2, 4], [6, 8], [10, 12]]);
  await expect(page.locator(".plot-overlay", { hasText: "exact max over 2 frames" })).toBeVisible();
  expect(workerUrls.length).toBeGreaterThanOrEqual(3);
  expect(new Set(workerUrls).size).toBe(workerUrls.length);
  expect(workerUrls.every(url => new URL(url).searchParams.has("generation"))).toBe(true);
  expect(errors).toEqual([]);
});

test("hodograph preserves invalid-level gaps, finite supported bounds and unknown unit labels", async ({ page }) => {
  await page.goto("/harness/index.html?mode=regular-dataflow&regularPlot=hodograph");
  await page.waitForFunction(() => Boolean(window.__nimbusRegularHarness));
  await waitLoaded(page);
  await page.evaluate(() => {
    window.__nimbusRegularHarness!.setSample(0, "u", 0, Infinity);
    window.__nimbusRegularHarness!.setSample(0, "v", 0, 10000);
  });
  await expect.poll(async () => (await data(page))[0].x).toEqual([null, 2]);
  await expect.poll(async () => (await data(page))[0].y).toEqual([null, -3]);
  const range = await page.evaluate(() => ((document.querySelector(".js-plotly-plot") as unknown as {
    layout: { xaxis: { range: number[] } } }).layout.xaxis.range));
  expect(range).toEqual([-3.3, 3.3]);
  await page.evaluate(() => {
    window.__nimbusRegularHarness!.setComponentUnit(0, "u", "");
    window.__nimbusRegularHarness!.setComponentUnit(0, "v", "");
  });
  await expect.poll(() => page.evaluate(() => (document.querySelector(".js-plotly-plot") as unknown as {
    layout?: { xaxis?: { title?: { text?: string } } } } | null)?.layout?.xaxis?.title?.text)).toBe("u");
});

for (const plot of ["vertical", "radial"]) test(`${plot} retains invalid source cells as profile gaps`, async ({ page }) => {
  await open(page, plot);
  await page.evaluate(() => window.__nimbusRegularHarness!.setSample(0, "w", 0, Infinity));
  await expect.poll(async () => (await data(page))[0][plot === "vertical" ? "x" : "y"]?.[0]).toBe(null);
});

for (const plot of ["series", "height", "swath"]) test(`${plot} combines equivalent unit spellings and rejects a physical unit mismatch`, async ({ page }) => {
  await open(page, plot);
  await page.evaluate(() => window.__nimbusRegularHarness!.setUnit(1, "m s-1"));
  await waitLoaded(page);
  await page.evaluate(() => window.__nimbusRegularHarness!.setUnit(1, "knots"));
  await expect(page.locator(".plot-overlay--error")).toContainText("different units");
});
