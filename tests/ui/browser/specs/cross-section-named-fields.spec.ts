import { expect, test, type Page } from "@playwright/test";

const LEVELS = 160;
const FIELDS = ["cloud_condensate", "p", "theta_prime"] as const;
type Graph = HTMLElement & {
  data: Array<{ type: string; z: number[][]; x: number[]; y: number[]; zsmooth: false | string }>;
  _fullLayout: { yaxis: { range: number[] } };
};

async function plane(page: Page) {
  return page.locator(".js-plotly-plot").evaluate(element => {
    const graph = element as Graph;
    const trace = graph.data.find(item => item.type === "heatmap")!;
    return { x: [...trace.x], y: [...trace.y], z: trace.z.map(row => [...row]),
      zsmooth: trace.zsmooth, yRange: [...graph._fullLayout.yaxis.range] };
  });
}

for (const plot of ["slice", "composite"] as const) {
  test(`${plot} keeps complete X/Y sections and Z plans for cloud condensate, p, and theta_prime`, async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&regularNz=${LEVELS}&namedSectionFields=1`);
    const fieldPicker = page.getByRole("combobox", { name: plot === "slice" ? "Field" : "Fill field", exact: true });
    for (const field of FIELDS) {
      await fieldPicker.selectOption(field);
      const source = await page.evaluate(name => window.__nimbusRegularHarness!.sourceFor(name), field);
      const plan = Array.from({ length: 3 }, (_, x) => Array.from({ length: 2 }, (_, y) => source[(x * 2 + y) * LEVELS]));
      await expect.poll(async () => (await plane(page)).z).toEqual(plan);
      for (const axis of ["y", "x"] as const) {
        await page.getByRole("button", { name: axis, exact: true }).click();
        await expect(page.locator(".ytitle")).toHaveText("z (km)");
        const columns = axis === "y" ? 3 : 2;
        const expected = Array.from({ length: LEVELS }, (_, z) =>
          Array.from({ length: columns }, (_, column) => source[(axis === "y" ? column * 2 : column) * LEVELS + z]));
        await expect.poll(async () => (await plane(page)).z).toEqual(expected);
        const section = await plane(page);
        expect(section.z).toHaveLength(LEVELS);
        expect(section.z[0]).toEqual(expected[0]);
        expect(section.z.at(-1)).toEqual(expected.at(-1));
        expect(section.y).toEqual(Array.from({ length: LEVELS }, (_, level) => 0.25 + level * 0.5));
        expect(section.yRange[0]).toBeLessThanOrEqual(0.25);
        expect(section.yRange[1]).toBeGreaterThanOrEqual(79.75);
        expect(section.zsmooth).toBe(false);
      }
      await page.getByRole("button", { name: "z", exact: true }).click();
      await expect.poll(async () => (await plane(page)).z).toEqual(plan);
      const restored = await plane(page);
      expect(restored.x).toEqual([1, 3]);
      expect(restored.y).toEqual([0.5, 1.5, 2.5]);
      expect(restored.zsmooth).toBe(false);
      expect(await page.evaluate(name => window.__nimbusRegularHarness!.sourceFor(name), field)).toEqual(source);
      const requests = await page.evaluate(name => window.__nimbusRegularHarness!.snapshot().calls
        .filter(call => call.field === name && call.command === "read_field_window").map(call => call.request), field);
      expect(requests.some(request => request?.extent[0] === 1 && request.extent[1] === 2 && request.extent[2] === LEVELS)).toBe(true);
      expect(requests.some(request => request?.extent[0] === 3 && request.extent[1] === 1 && request.extent[2] === LEVELS)).toBe(true);
      expect(requests.every(request => request?.stride.every(stride => stride === 1))).toBe(true);
    }
  });
}
