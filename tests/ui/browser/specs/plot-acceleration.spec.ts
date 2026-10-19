import { expect, test } from "@playwright/test";

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

test("swath accumulation runs off-thread and publishes a raster result", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "Swaths", exact: true }).click();
  await page.getByRole("button", { name: "Swath", exact: true }).click();

  await expect(page.locator(".control-group", { hasText: "Frames" }))
    .toContainText("3 accumulated", { timeout: 30_000 });
  await expect(page.locator('.plot-canvas[data-render-backend="raster"]')).toBeVisible();
  await expect(page.getByText(/native horizontal cells · exact max over 3 frames/)).toBeVisible();
  await expect(page.locator(".plot-error")).toHaveCount(0);
});

test("native-fidelity Q isosurface extracts in an isolated worker", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  const dataGroup = page.locator(".plot3d-control-tab", { hasText: "Data" }).first();
  if (await dataGroup.getAttribute("aria-expanded") !== "true") await dataGroup.click();
  await page.locator(".control-group", { hasText: "Field" }).first()
    .getByRole("combobox").selectOption("q_criterion");
  await expect(page.locator(".plot3d-badge__sub")).toContainText("Q-criterion", { timeout: 30_000 });
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0, { timeout: 30_000 });
});

test("2D Plotly chrome and scientific keys stay legible and non-redundant", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();

  const canvas = page.locator(".plot-canvas").first();
  await expect(canvas).toBeVisible({ timeout: 30_000 });
  await expect(canvas).toHaveAttribute("role", "region");
  await expect(canvas).toHaveAttribute(
    "aria-label",
    /^Interactive scientific plot(?:: .+)?$/,
  );
  // The region role keeps Plotly's modebar controls in the accessibility
  // tree; role="img" made all of these interactive descendants presentational.
  await expect(canvas.getByRole("button").first()).toBeVisible();
  const textChrome = await canvas.evaluate((element) => {
    const title = element.querySelector<SVGTextElement>(".gtitle");
    const tick = element.querySelector<SVGTextElement>(".xtick text");
    return {
      token: getComputedStyle(document.documentElement).getPropertyValue("--plot-text").trim(),
      title: title ? getComputedStyle(title).fill : "",
      tick: tick ? getComputedStyle(tick).fill : "",
    };
  });
  expect(textChrome.token).toBe("#a3a8b3");
  expect(textChrome.title).toBe("rgb(163, 168, 179)");
  expect(textChrome.tick).toBe("rgb(163, 168, 179)");

  await page.evaluate(() => {
    document.documentElement.dataset.theme = "light";
    window.dispatchEvent(new CustomEvent("nimbus-studio:appearance-applied"));
  });
  await expect.poll(() => canvas.evaluate((element) => {
    const title = element.querySelector<SVGTextElement>(".gtitle");
    return title ? getComputedStyle(title).fill : "";
  })).toBe("rgb(77, 88, 100)");
  await page.evaluate(() => {
    document.documentElement.dataset.theme = "dark";
    window.dispatchEvent(new CustomEvent("nimbus-studio:appearance-applied"));
  });

  await page.locator(".control-group", { hasText: "Field" }).first()
    .getByRole("combobox").selectOption("uniform_scalar");
  await expect(page.locator(".annotation-text")).toContainText(
    "Uniform field · uniform_scalar = 0",
    { timeout: 30_000 },
  );

  await page.getByRole("button", { name: "Vector field", exact: true }).click();
  await expect(page.locator(".plot-canvas").first()).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".legendtext")).toHaveCount(0);
  await expect(page.locator(".annotation-text")).toContainText("m s-1");

  await page.getByRole("button", { name: "Composite", exact: true }).click();
  const contourA = page.locator(".control-group", { hasText: "Contour A" });
  await contourA.locator('input[type="text"]').fill("-10");
  await expect(page.locator(".legendtext")).toContainText("reflectivity_dbz = -10 dBZ", {
    timeout: 30_000,
  });
});

async function scalarFill(page: import('@playwright/test').Page) {
  return page.locator('.plot-canvas').first().evaluate((element) => {
    const graph = element as HTMLElement & { data?: Array<{
      type: string; z?: number[][]; zmin?: number; zmax?: number;
      colorbar?: { title?: { text?: string } };
    }> };
    const trace = graph.data?.find((candidate) => candidate.type === 'heatmap');
    if (!trace?.z) return null;
    const values = trace.z.flat().filter(Number.isFinite);
    return {
      label: trace.colorbar?.title?.text ?? '',
      values: trace.z,
      minimum: Math.min(...values),
      maximum: Math.max(...values),
      lower: trace.zmin,
      upper: trace.zmax,
    };
  });
}

for (const plot of ['Slice', 'Composite'] as const) {
  test(`${plot} automatic cross-section fallback resets the scalar color policy`, async ({ page }) => {
    await page.goto('/harness/index.html?mode=studio');
    await page.addStyleTag({ content: APP_SHELL });
    await page.getByRole('button', { name: 'Plan views', exact: true }).click();
    await page.getByRole('button', { name: plot, exact: true }).click();
    const field = page.getByRole('group', { name: plot === 'Composite' ? 'Fill field' : 'Field', exact: true }).getByRole('combobox');
    const palette = page.getByRole('group', { name: plot === 'Composite' ? 'Fill colormap' : 'Colormap', exact: true }).getByRole('combobox');
    await field.selectOption('composite_reflectivity');
    await expect.poll(async () => (await scalarFill(page))?.label).toContain('composite_reflectivity');
    await expect(palette).toHaveValue('dBZ');
    await page.getByRole('button', { name: 'Cross-sections', exact: true }).click();
    await expect(field).toHaveValue('w');
    await expect(palette).toHaveValue('RdBu');
    await expect.poll(async () => (await scalarFill(page))?.label).toContain('w (');
    const automatic = (await scalarFill(page))!;
    expect(automatic.maximum).toBeGreaterThan(automatic.minimum);
    expect(automatic.lower).toBeCloseTo(-automatic.upper!, 6);
    expect(automatic.upper).toBeCloseTo(Math.max(Math.abs(automatic.minimum), Math.abs(automatic.maximum)), 6);

    await palette.selectOption('Viridis');
    await page.getByRole('button', { name: 'Plan views', exact: true }).click();
    await expect(palette).toHaveValue('Viridis');
    await page.getByRole('button', { name: 'Cross-sections', exact: true }).click();
    await expect(palette).toHaveValue('Viridis');
    await field.selectOption('u');
    await field.selectOption('w');
    await expect.poll(async () => (await scalarFill(page))?.label).toContain('w (');
    expect((await scalarFill(page))?.values).toEqual(automatic.values);
  });

  test(`${plot} reset colors preserves data and explains uniform or compressed ranges`, async ({ page }) => {
    await page.goto('/harness/index.html?mode=studio');
    await page.addStyleTag({ content: APP_SHELL });
    await page.getByRole('button', { name: 'Cross-sections', exact: true }).click();
    await page.getByRole('button', { name: plot, exact: true }).click();
    const field = page.getByRole('group', { name: plot === 'Composite' ? 'Fill field' : 'Field', exact: true }).getByRole('combobox');
    const palette = page.getByRole('group', { name: plot === 'Composite' ? 'Fill colormap' : 'Colormap', exact: true }).getByRole('combobox');
    await field.selectOption('w');
    await expect.poll(async () => (await scalarFill(page))?.label).toContain('w (');
    const original = (await scalarFill(page))!;
    await palette.selectOption('dBZ');
    await expect(page.locator('.annotation-text')).toContainText('Visible range uses');
    await page.getByRole('button', { name: 'Reset colors', exact: true }).click();
    await expect(palette).toHaveValue('RdBu');
    await expect.poll(async () => (await scalarFill(page))?.upper).toBeCloseTo(Math.max(Math.abs(original.minimum), Math.abs(original.maximum)), 6);
    expect((await scalarFill(page))?.values).toEqual(original.values);

    await field.selectOption('uniform_scalar');
    await expect(page.locator('.annotation-text')).toContainText('Uniform field · uniform_scalar = 0');
    expect((await scalarFill(page))?.values.flat().every((value) => value === 0)).toBe(true);
    await field.selectOption('reflectivity_dbz');
    await expect.poll(async () => (await scalarFill(page))?.label).toContain('reflectivity_dbz');
    await page.getByRole('button', { name: 'Reset colors', exact: true }).click();
    await expect(palette).toHaveValue('dBZ');
    await expect.poll(async () => (await scalarFill(page))?.lower).toBe(-30);
    await expect.poll(async () => (await scalarFill(page))?.upper).toBe(75);
  });
}

for (const plot of ['Slice', 'Composite'] as const) {
  test(`${plot} reset colors refits a retained range without changing the viewport`, async ({ page }) => {
    await page.goto('/harness/index.html?mode=studio');
    await page.addStyleTag({ content: APP_SHELL });
    await page.getByRole('button', { name: 'Cross-sections', exact: true }).click();
    await page.getByRole('button', { name: plot, exact: true }).click();
    await expect.poll(async () => (await scalarFill(page))?.label).toContain('w (');
    const full = (await scalarFill(page))!;
    const box = await page.locator('.plot-canvas .nsewdrag').first().boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.move(box!.x + box!.width * 0.35, box!.y + box!.height * 0.3);
    await page.mouse.down();
    await page.mouse.move(box!.x + box!.width * 0.6, box!.y + box!.height * 0.65, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => (await scalarFill(page))?.values.flat().length ?? Infinity)
      .toBeLessThan(full.values.flat().length);
    const zoomed = (await scalarFill(page))!;
    const bounds = () => page.locator('.plot-canvas').first().evaluate((element) => {
      const graph = element as HTMLElement & { _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[] } } };
      return [graph._fullLayout.xaxis.range, graph._fullLayout.yaxis.range];
    });
    const before = await bounds();
    expect(zoomed.upper).toBe(full.upper);
    expect(Math.max(Math.abs(zoomed.minimum), Math.abs(zoomed.maximum))).toBeLessThan(full.upper!);
    await page.getByRole('button', { name: 'Reset colors', exact: true }).click();
    await expect.poll(async () => (await scalarFill(page))?.upper)
      .toBeCloseTo(Math.max(Math.abs(zoomed.minimum), Math.abs(zoomed.maximum)), 6);
    expect((await scalarFill(page))?.values).toEqual(zoomed.values);
    expect(await bounds()).toEqual(before);
  });
}

test('reset colors releases preset bounds and a fresh preset restores them', async ({ page }) => {
  await page.goto('/harness/index.html?mode=studio');
  await page.addStyleTag({ content: APP_SHELL });
  const preset = page.getByRole('button', { name: "Cold pool (θ')", exact: true });
  await preset.click();
  await expect.poll(async () => (await scalarFill(page))?.upper).toBe(4);
  const selected = (await scalarFill(page))!;
  await page.getByRole('button', { name: 'Reset colors', exact: true }).click();
  await expect.poll(async () => (await scalarFill(page))?.upper)
    .toBeCloseTo(Math.max(Math.abs(selected.minimum), Math.abs(selected.maximum)), 6);
  expect((await scalarFill(page))?.values).toEqual(selected.values);
  await preset.click();
  await expect.poll(async () => (await scalarFill(page))?.lower).toBe(-4);
  await expect.poll(async () => (await scalarFill(page))?.upper).toBe(4);
});
