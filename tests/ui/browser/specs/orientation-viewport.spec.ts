import { expect, test, type Page } from "@playwright/test";

const LEVELS = 160;
type Graph = HTMLElement & {
  data: Array<{ type: string; z: number[][]; x: number[]; y: number[] }>;
  _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[]; scaleanchor?: string | false } };
};

async function plane(page: Page) {
  return page.locator(".js-plotly-plot").evaluate(element => {
    const graph = element as Graph;
    const trace = graph.data.find(item => item.type === "heatmap")!;
    return { x: [...trace.x], y: [...trace.y], z: trace.z.map(row => [...row]),
      yRange: [...graph._fullLayout.yaxis.range], scaleanchor: graph._fullLayout.yaxis.scaleanchor ?? false };
  });
}

async function zoom(page: Page) {
  await page.locator(".plot-controls").getByRole("button", { name: "Box zoom", exact: true }).click();
  const box = (await page.locator(".nsewdrag").boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.55);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.9, { steps: 5 });
  await page.mouse.up();
}

for (const plot of ["slice", "composite"] as const) {
  test(`${plot} releases plan-view zoom and retains every native height when switching vertical axes`, async ({ page }) => {
    test.setTimeout(60_000);
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&regularNz=${LEVELS}`);
    await page.getByRole("combobox", { name: plot === "slice" ? "Field" : "Fill field", exact: true }).selectOption("w");
    await page.getByRole("button", { name: "z", exact: true }).click();
    await expect(page.locator(".ytitle")).toHaveText("x (km)");
    const source = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().source);
    await zoom(page);
    await expect.poll(async () => (await plane(page)).yRange[1]).toBeLessThan(4);
    for (const axis of ["y", "x"] as const) {
      await page.getByRole("button", { name: axis, exact: true }).click();
      await expect(page.locator(".ytitle")).toHaveText("z (km)");
      const columns = axis === "y" ? 3 : 2;
      const expected = Array.from({ length: LEVELS }, (_, z) =>
        Array.from({ length: columns }, (_, column) => source[(axis === "y" ? column * 2 : column) * LEVELS + z]));
      await expect.poll(async () => (await plane(page)).z).toEqual(expected);
      const section = await plane(page);
      expect(section.y).toEqual(Array.from({ length: LEVELS }, (_, level) => 0.25 + level * 0.5));
      expect(section.yRange[0]).toBeLessThanOrEqual(section.y[0]);
      expect(section.yRange[1]).toBeGreaterThanOrEqual(section.y.at(-1)!);
      expect(section.scaleanchor).toBe(false);
      if (axis === "y") {
        await zoom(page);
        await expect.poll(async () => (await plane(page)).yRange[1]).toBeLessThan(section.y.at(-1)! * 0.8);
        await expect.poll(async () => (await plane(page)).z.length).toBeLessThan(LEVELS);
        const zoomed = await plane(page);
        await page.evaluate(() => window.__nimbusRegularHarness!.setStep(1));
        await expect.poll(async () => (await plane(page)).z).toEqual(zoomed.z.map(row => row.map(value => value + 10)));
        expect((await plane(page)).yRange).toEqual(zoomed.yRange);
        await page.evaluate(() => window.__nimbusRegularHarness!.setStep(0));
        await expect.poll(async () => (await plane(page)).z).toEqual(zoomed.z);
      }
    }
    expect(await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().source)).toEqual(source);
    const requests = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().calls
      .filter(call => call.field === "w" && call.command === "read_field_window").map(call => call.request));
    expect(requests.some(request => request?.extent[0] === 1 && request.extent[1] === 2 && request.extent[2] === LEVELS)).toBe(true);
    expect(requests.some(request => request?.extent[0] === 3 && request.extent[1] === 1 && request.extent[2] === LEVELS)).toBe(true);
  });
}
