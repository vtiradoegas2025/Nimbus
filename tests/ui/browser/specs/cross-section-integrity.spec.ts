import { expect, test, type Page } from "@playwright/test";

type Graph = HTMLElement & { data: Array<{ type: string; z: number[][]; x: number[]; y: number[]; zmin: number; zmax: number }> };
async function values(page: Page) {
  return page.evaluate(() => JSON.parse(JSON.stringify((document.querySelector(".js-plotly-plot") as Graph | null)?.data ?? [])));
}
async function openComposite(page: Page) {
  await page.goto("/harness/index.html?mode=regular-dataflow&regularPlot=composite&section=1");
  await page.waitForFunction(() => window.__nimbusRegularHarness?.snapshot().frameReports.at(-1)?.loading === false);
  await expect(page.locator(".ytitle")).toHaveText("z (km)");
  await expect(page.locator(".xtitle")).toHaveText("x (km)");
}

for (const plot of ["slice", "composite"] as const) {
  test(`${plot} switches from a longer source axis without disabling section controls`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&section=1`);
    await page.waitForFunction(() => window.__nimbusRegularHarness?.snapshot().frameReports.at(-1)?.loading === false);
    await page.getByRole("button", { name: "x", exact: true }).click();
    const xIndex = page.getByRole("spinbutton", { name: "Go to x index" });
    await xIndex.fill("2");
    await xIndex.press("Enter");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[9, 11], [10, 12]]);
    await page.getByRole("button", { name: "y", exact: true }).click();
    const yIndex = page.getByRole("spinbutton", { name: "Go to y index" });
    await expect(yIndex).toBeEnabled();
    await expect(yIndex).toHaveValue("1");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[3, 7, 11], [4, 8, 12]]);
    await expect(page.getByLabel("Displayed section position")).toContainText("y = 3 km · index 1");
    await yIndex.fill("0");
    await yIndex.press("Enter");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[1, 5, 9], [2, 6, 10]]);
  });

  test(`${plot} reapplying a preset restores its plane after a manual index edit`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&section=1`);
    const preset = { id: "fixture-section-top", label: "Top section", description: "Exact saved section",
      category: "cross-sections" as const, plotType: plot, field: "w", sliceAxis: 1 as const, sliceIndexHint: "top" as const };
    await page.waitForFunction(() => Boolean(window.__nimbusRegularHarness));
    await page.evaluate(value => window.__nimbusRegularHarness!.setPreset(value), preset);
    const index = page.getByRole("spinbutton", { name: "Go to y index" });
    await expect(index).toHaveValue("1");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[3, 7, 11], [4, 8, 12]]);
    await index.fill("0");
    await index.press("Enter");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[1, 5, 9], [2, 6, 10]]);
    await page.evaluate(value => window.__nimbusRegularHarness!.setPreset(value), preset);
    await expect(index).toHaveValue("1");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[3, 7, 11], [4, 8, 12]]);
    await page.getByRole("button", { name: "x", exact: true }).click();
    await expect(page.getByRole("spinbutton", { name: "Go to x index" })).toHaveAttribute("max", "2");
    await page.evaluate(value => window.__nimbusRegularHarness!.setPreset({ ...value,
      id: "fixture-section-mid", sliceIndexHint: "mid" }), preset);
    await expect(index).toHaveValue("0");
    await expect.poll(async () => (await values(page))[0]?.z).toEqual([[1, 5, 9], [2, 6, 10]]);
  });
}

test("scalar section position stays committed while exact index jumps load both vertical orientations", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/harness/index.html?mode=regular-dataflow&regularPlot=slice&section=1");
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  await page.waitForFunction(() => window.__nimbusRegularHarness?.snapshot().frameReports.at(-1)?.loading === false);
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 1 km · index 0");
  expect((await values(page))[0].z).toEqual([[1, 5, 9], [2, 6, 10]]);
  await page.evaluate(() => window.__nimbusRegularHarness!.setDelay("w", 700));
  const jump = page.getByRole("spinbutton", { name: "Go to y index" });
  await jump.fill("1");
  await expect(page.getByRole("slider", { name: "Selected y index" })).toHaveValue("0");
  await jump.press("Enter");
  await expect(page.getByRole("status")).toContainText("Displaying index 0");
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 1 km · index 0");
  expect((await values(page))[0].z).toEqual([[1, 5, 9], [2, 6, 10]]);
  await expect.poll(async () => (await values(page))[0].z).toEqual([[3, 7, 11], [4, 8, 12]]);
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 3 km · index 1");
  await page.getByRole("button", { name: "x", exact: true }).click();
  await expect(page.locator(".xtitle")).toHaveText("y (km)");
  await expect(page.locator(".ytitle")).toHaveText("z (km)");
  const xJump = page.getByRole("spinbutton", { name: "Go to x index" });
  await expect(xJump).toHaveAttribute("max", "2");
  await xJump.fill("2");
  await xJump.press("Enter");
  await expect.poll(async () => (await values(page))[0].z).toEqual([[9, 11], [10, 12]]);
  await expect(page.getByLabel("Displayed section position")).toContainText("x = 2.5 km · index 2");
  await xJump.fill("0");
  await xJump.press("Escape");
  await expect(xJump).toHaveValue("2");
  expect(await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().source)).toEqual(Array.from({ length: 12 }, (_, index) => index + 1));
});

test("composite sections retain the completed plane and index bounds while a replacement loads", async ({ page }) => {
  await openComposite(page);
  const before = await values(page);
  expect(before[0].z).toEqual([[1, 5, 9], [2, 6, 10]]);
  expect(before[0].x).toEqual([0.5, 1.5, 2.5]);
  expect(before[0].y).toEqual([0.25, 0.75]);
  const slider = page.getByRole("slider", { name: "Selected y index" });
  await expect(slider).toHaveAttribute("max", "1");
  await page.evaluate(() => window.__nimbusRegularHarness!.setDelay("w", 900));
  await slider.fill("1");
  await expect(page.getByRole("status")).toContainText("Displaying index 0");
  await expect(slider).toHaveAttribute("max", "1");
  await expect(slider).toBeEnabled();
  expect((await values(page))[0].z).toEqual(before[0].z);
  expect((await values(page))[0].zmin).toBe(before[0].zmin);
  expect((await values(page))[0].zmax).toBe(before[0].zmax);
  await expect.poll(async () => (await values(page))[0].z).toEqual([[3, 7, 11], [4, 8, 12]]);
  await expect(page.getByRole("status")).toHaveCount(0);
  expect(await slider.inputValue()).toBe("1");
  await expect(page.getByLabel("Displayed section position")).toContainText("y = 3 km · index 1");
  expect(await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().source))
    .toEqual(Array.from({ length: 12 }, (_, index) => index + 1));
  const jump = page.getByRole("spinbutton", { name: "Go to y index" });
  await jump.fill("0");
  expect(await slider.inputValue()).toBe("1");
  await jump.press("Enter");
  await expect(slider).toHaveValue("0");
  await expect.poll(async () => (await values(page))[0].z).toEqual(before[0].z);
  await jump.fill("1");
  await jump.press("Escape");
  await expect(jump).toHaveValue("0");
});

test("composite advances its exact frame report only after every enabled layer is drawn", async ({ page }) => {
  await openComposite(page);
  await page.getByRole("checkbox", { name: "Enable Contour A" }).check();
  await page.getByRole("combobox", { name: "Contour A field" }).selectOption("u");
  await page.getByRole("textbox", { name: "Contour A levels" }).fill("2");
  await expect.poll(async () => (await values(page)).length).toBe(2);
  const before = await values(page);
  await page.evaluate(() => {
    window.__nimbusRegularHarness!.setDelay("u", 1200);
    window.__nimbusRegularHarness!.setStep(1);
  });
  await expect(page.getByRole("status")).toContainText("loading selected");
  expect((await values(page))[0].z).toEqual(before[0].z);
  const pending = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().frameReports.at(-1));
  expect(pending).toMatchObject({ requestedStepIndex: 1, renderedStepIndex: 0, loading: true });
  await expect.poll(async () => page.evaluate(() => window.__nimbusRegularHarness!.snapshot().frameReports.at(-1)))
    .toMatchObject({ requestedStepIndex: 1, renderedStepIndex: 1, loading: false, renderedPublicationGeneration: "regular-1-0" });
  const after = await values(page);
  expect(after[0].z).toEqual([[11, 15, 19], [12, 16, 20]]);
  expect(after[0].zmin).toBe(-20);
  expect(after[0].zmax).toBe(20);
  expect(after).toHaveLength(2);
});

test("terrain composites use the same bounded physical-height lattice as scalar sections", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio&terrain=1");
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  await expect(page.getByText("physical-volume terrain→Cartesian vertical remap")).toBeVisible();
  await expect(page.locator(".ytitle")).toHaveText("z (km)");
  const scalar = (await values(page))[0];
  await page.getByRole("button", { name: "Composite", exact: true }).click();
  await page.getByRole("combobox", { name: "Fill field", exact: true }).selectOption("w");
  await page.getByRole("checkbox", { name: "Enable Contour A" }).uncheck();
  await page.getByRole("checkbox", { name: "Enable Contour B" }).uncheck();
  await expect(page.getByText("Physical-volume terrain → Cartesian vertical remap")).toBeVisible();
  await expect(page.locator(".ytitle")).toHaveText("z (km)");
  const composite = (await values(page))[0];
  expect(composite.z).toEqual(scalar.z);
  expect(composite.x).toEqual(scalar.x);
  expect(composite.y).toEqual(scalar.y);
});

test("invalid contour levels preserve the complete figure and show an actionable error", async ({ page }) => {
  await openComposite(page);
  await page.getByRole("checkbox", { name: "Enable Contour A" }).check();
  await page.getByRole("combobox", { name: "Contour A field" }).selectOption("w");
  const levels = page.getByRole("textbox", { name: "Contour A levels" });
  await levels.fill("5");
  await expect.poll(async () => (await values(page)).length).toBe(2);
  const before = await values(page);
  await levels.fill("1,,5");
  await expect(levels).toHaveAttribute("aria-invalid", "true");
  await expect(page.getByRole("alert")).toContainText("Enter finite values");
  expect(await values(page)).toEqual(before);
  await levels.fill("3:7:2");
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect.poll(async () => (await values(page)).length).toBe(4);
});
