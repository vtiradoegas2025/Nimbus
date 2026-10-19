import { expect, test, type Page } from "@playwright/test";

const EXAMPLE = "nimbus-example://manufactured-supercell-v1";
type PlotGraph = HTMLElement & {
  data: Array<{ name?: string; type?: string; x?: number[]; y?: number[]; z?: number[][]; customdata?: number[] }>;
  _fullLayout: { title: { text: string } };
};

async function openExample(page: Page) {
  page.setDefaultTimeout(15_000);
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate((uri) => window.__nimbusStudioHarness!.setRun(uri, "completed"), EXAMPLE);
  await expect(page.getByRole("heading", { name: EXAMPLE, exact: true })).toBeVisible();
}

async function traces(page: Page) {
  return page.evaluate(() => (document.querySelector(".plot-canvas") as PlotGraph | null)?.data ?? []);
}

function plotErrors(page: Page) {
  return page.locator(".plot-overlay--error, .plot-error, .error-boundary");
}

test("packaged Skew-T renders derived temperature and dewpoint at an offset column", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openExample(page);
  await page.getByRole("button", { name: "Soundings", exact: true }).click();
  await page.getByRole("button", { name: "Skew-T", exact: true }).click();
  await expect.poll(async () => (await traces(page))?.find((trace) => trace.name === "T")?.customdata?.length).toBe(20);
  const before = (await traces(page)).find((trace) => trace.name === "T")!.customdata!;
  await page.getByRole("slider", { name: /^x index/ }).fill("10");
  await page.getByRole("slider", { name: /^y index/ }).fill("5");
  await expect.poll(async () => (await traces(page)).find((trace) => trace.name === "T")?.customdata).not.toEqual(before);
  await expect(page.locator(".gtitle")).toContainText("x=10");
  await expect(page.locator(".gtitle")).toContainText("y=5");
  const plotted = await traces(page);
  for (const name of ["T", "Td"]) {
    const values = plotted.find((trace) => trace.name === name)!.customdata!;
    expect(values.length).toBe(20);
    expect(values.every(Number.isFinite)).toBe(true);
  }
  await expect(plotErrors(page)).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("packaged hodograph reports the selected physical column", async ({ page }) => {
  await openExample(page);
  await page.getByRole("button", { name: "Soundings", exact: true }).click();
  await page.getByRole("button", { name: "Hodograph", exact: true }).click();
  await expect(page.locator(".plot-canvas")).toBeVisible();
  await page.getByRole("slider", { name: /^x index/ }).fill("10");
  await page.getByRole("slider", { name: /^y index/ }).fill("5");
  await expect(page.locator(".gtitle")).toHaveText("Hodograph @ (x=5.25 km, y=2.75 km)");
  const curve = (await traces(page)).find((trace) => trace.name === "hodograph")!;
  expect(curve.x).toHaveLength(20);
  expect(curve.y).toHaveLength(20);
  await expect(plotErrors(page)).toHaveCount(0);
});

for (const plot of ["Slice", "Composite"] as const) {
  test(`packaged ${plot} preserves physical coordinates after a viewport crop`, async ({ page }) => {
    await openExample(page);
    await page.getByRole("button", { name: "Plan views", exact: true }).click();
    await page.getByRole("button", { name: plot, exact: true }).click();
    const field = page.getByRole("group", { name: plot === "Slice" ? "Field" : "Fill field", exact: true }).getByRole("combobox");
    await field.selectOption("w");
    await expect(page.locator(".gtitle")).toContainText("w — z-slice");
    const initial = (await traces(page)).find((trace) => trace.type === "heatmap")!;
    const box = await page.locator(".plot-canvas .nsewdrag").first().boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.move(box!.x + box!.width * 0.3, box!.y + box!.height * 0.3);
    await page.mouse.down();
    await page.mouse.move(box!.x + box!.width * 0.65, box!.y + box!.height * 0.65, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => (await traces(page)).find((trace) => trace.type === "heatmap")?.z?.flat().length ?? Infinity)
      .toBeLessThan(initial.z!.flat().length);
    const cropped = (await traces(page)).find((trace) => trace.type === "heatmap")!;
    expect(cropped.x![0]).toBeGreaterThan(initial.x![0]);
    expect(cropped.y![0]).toBeGreaterThan(initial.y![0]);
    expect(cropped.x!.at(-1)).toBeLessThanOrEqual(initial.x!.at(-1)!);
    expect(cropped.y!.at(-1)).toBeLessThanOrEqual(initial.y!.at(-1)!);
    await field.selectOption("temperature");
    await expect(page.locator(".gtitle")).toContainText("temperature");
    await expect(plotErrors(page)).toHaveCount(0);
  });
}
