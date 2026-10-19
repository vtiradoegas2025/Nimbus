import { expect, test, type Page } from "@playwright/test";

test.beforeEach(({}, info) => info.setTimeout(60_000));
async function inspect(page: Page) {
  return page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { data: Array<{ name: string; x: number[]; y: number[]; customdata?: number[][] }>; layout: { xaxis: { title: { text: string } }; yaxis: { title: { text: string } } } }>(".js-plotly-plot");
    if (!graph) return null;
    const values = graph.data.find(trace => trace.name === "vector values");
    return { components: values?.customdata?.map(row => Array.from(row)), x: values?.x, y: values?.y,
      xTitle: graph.layout.xaxis.title.text, yTitle: graph.layout.yaxis.title.text };
  });
}
async function open(page: Page) {
  await page.goto("/harness/index.html?mode=regular-dataflow&regularPlot=vector&section=1");
  await page.getByRole("button", { name: "Every", exact: true }).click();
  await expect.poll(async () => (await inspect(page))?.components?.length).toBe(6);
  await page.waitForFunction(() => window.__nimbusRegularHarness?.snapshot().frameReports.at(-1)?.loading === false);
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 1 km · index 0");
}

test("vector sections keep completed plane coordinates while exact indices load the correct in-plane components", async ({ page }) => {
  await open(page);
  const before = await inspect(page);
  expect(before?.xTitle).toBe("x (km)");
  expect(before?.yTitle).toBe("z (km)");
  expect(before?.components?.map(values => values.slice(0, 2))).toEqual([[2, 1], [2, 5], [2, 9], [2, 2], [2, 6], [2, 10]]);
  await page.evaluate(() => window.__nimbusRegularHarness!.setDelay("w", 900));
  const jump = page.getByRole("spinbutton", { name: "Go to y index" });
  await jump.fill("1");
  await expect(page.getByRole("slider", { name: "Selected y index" })).toHaveValue("0");
  await jump.press("Enter");
  await expect(page.getByRole("status")).toContainText("Displaying index 0");
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 1 km · index 0");
  expect(await inspect(page)).toEqual(before);
  await expect(page.getByRole("slider", { name: "Selected y index" })).toHaveAttribute("max", "1");
  await expect.poll(async () => (await inspect(page))?.components?.map(values => values.slice(0, 2)))
    .toEqual([[2, 3], [2, 7], [2, 11], [2, 4], [2, 8], [2, 12]]);
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 3 km · index 1");
  await page.getByRole("button", { name: "x", exact: true }).click();
  await expect.poll(async () => (await inspect(page))?.xTitle).toBe("y (km)");
  await expect(page.locator(".ytitle")).toHaveText("z (km)");
  const xJump = page.getByRole("spinbutton", { name: "Go to x index" });
  await expect(xJump).toHaveAttribute("max", "2");
  await xJump.fill("2");
  await xJump.press("Enter");
  await expect.poll(async () => (await inspect(page))?.components?.map(values => values.slice(0, 2)))
    .toEqual([[-3, 9], [-3, 11], [-3, 10], [-3, 12]]);
  await expect(page.getByLabel("Displayed section position")).toContainText("x = 2.5 km · index 2");
});

test("vector frame reports advance only after both components are rendered together", async ({ page }) => {
  await open(page);
  const before = await inspect(page);
  await page.evaluate(() => { window.__nimbusRegularHarness!.setDelay("w", 1000); window.__nimbusRegularHarness!.setStep(1); });
  await expect.poll(() => page.evaluate(() => window.__nimbusRegularHarness!.snapshot().frameReports.at(-1)))
    .toMatchObject({ requestedStepIndex: 1, renderedStepIndex: 0, loading: true });
  expect(await inspect(page)).toEqual(before);
  await expect.poll(() => page.evaluate(() => window.__nimbusRegularHarness!.snapshot().frameReports.at(-1)))
    .toMatchObject({ requestedStepIndex: 1, renderedStepIndex: 1, renderedPublicationGeneration: "regular-1-0", loading: false });
  expect((await inspect(page))?.components?.map(values => values.slice(0, 2)))
    .toEqual([[2, 11], [2, 15], [2, 19], [2, 12], [2, 16], [2, 20]]);
});

test("terrain vector sections disclose physical height and preserve source index bounds", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio&terrain=1");
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Vector field", exact: true }).click();
  await expect(page.getByText("physical-volume terrain→Cartesian vector remap")).toBeVisible();
  await expect(page.locator(".ytitle")).toHaveText("z (km)");
  await expect(page.getByLabel("Displayed section position")).toContainText("x–z section");
  await expect(page.getByRole("spinbutton", { name: "Go to y index" })).toHaveAttribute("max", "35");
});
