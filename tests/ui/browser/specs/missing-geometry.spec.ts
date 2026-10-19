import { test, expect } from "@playwright/test";
import { writeFile } from "node:fs/promises";

for (const mode of ["monolithic", "bricked", "terrain"] as const) test(`${mode} missing-data surfaces and paths preserve usable regions`, async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`/harness/index.html?missingGeometry=1${mode === "bricked" ? "&derivedBricks=1" : mode === "terrain" ? "&terrainBricks=1" : ""}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  for (const plot of ["isosurface", "streamlines"] as const) {
    await page.evaluate(plot => window.__nimbusVtkHarness!.setPlot(plot), plot);
    if (plot === "isosurface") {
      const tab = page.locator(".plot3d-control-tab", { hasText: "Data" });
      if (await tab.count() && await tab.getAttribute("aria-expanded") !== "true") await tab.click();
      await page.locator('select[aria-label="Field"]').selectOption("w");
    }
    for (const step of [0, 1]) {
      await page.evaluate(step => window.__nimbusVtkHarness!.setStep(step), step);
      await expect.poll(async () => page.evaluate(({ plot, step }) => {
        const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
        const error = document.querySelector(".volume-render__overlay--error")?.textContent;
        return error ?? (scene?.dataset.framePlot === plot && scene.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(step));
      }, { plot, step }), { timeout: 60_000 }).toBe(true);
      await expect(page.locator(".plot3d-badge__sub")).toContainText("missing regions");
      const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      expect(capture).not.toBeNull();
      await info.attach(`${plot}-${step}`, { body: JSON.stringify(capture!.evidence), contentType: "application/json" });
      expect(capture!.evidence.descriptor.frameIndex).toBe(step);
      expect(capture!.evidence.descriptor.parameters?.missingSupport).toBe(true);
      expect(capture!.evidence.layers.slice(0, 2).some(layer => layer.visible && (layer.pointCount ?? 0) > 0 && (layer.cellCount ?? 0) > 0), JSON.stringify(capture!.evidence)).toBe(true);
      if (plot === "streamlines") expect(capture!.evidence.layers.some(layer => layer.visible && layer.scalarVisibility && (layer.pointCount ?? 0) > 100)).toBe(true);
      await writeFile(info.outputPath(`${plot}-${step}.png`), Buffer.from(capture!.pngDataUri.split(",")[1], "base64"));
    }
  }
  await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes)).toBe(0);
  expect(errors).toEqual([]);
});

for (const plot of ["streamlines", "isosurface"] as const) test(`an empty ${plot} frame clears old geometry and recovers when valid data returns`, async ({ page }, info) => {
  await page.goto("/harness/index.html?missingGeometry=empty-next&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(plot => { window.__nimbusVtkHarness!.setPlot(plot); window.__nimbusVtkHarness!.setMounted(true); }, plot);
  if (plot === "isosurface") {
    const tab = page.locator(".plot3d-control-tab", { hasText: "Data" });
    if (await tab.getAttribute("aria-expanded") !== "true") await tab.click();
    await page.locator('select[aria-label="Field"]').selectOption("w");
  }
  for (const step of [0, 1, 0]) {
    await page.evaluate(step => window.__nimbusVtkHarness!.setStep(step), step);
    await page.waitForFunction(({ step, plot }) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      return scene?.dataset.framePlot === plot && scene.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(step);
    }, { step, plot }, { timeout: 30_000 });
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    expect(capture).not.toBeNull();
    const paths = capture!.evidence.layers.slice(0, plot === "streamlines" ? 4 : 2);
    if (step === 1) {
      expect(paths.every(path => !path.visible && path.pointCount === 0)).toBe(true);
      if (plot === "streamlines") {
        await expect(page.locator(".volume-render__overlay--muted")).toContainText("No paths from the current seeds");
        await expect(page.locator(".plot3d-colorbar__title", { hasText: "Speed" })).toHaveCount(0);
      }
    } else expect(paths.some(path => path.visible && (path.pointCount ?? 0) > 0)).toBe(true);
    await info.attach(`frame-${step}`, { body: JSON.stringify(capture!.evidence), contentType: "application/json" });
  }
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
});

test("a rejected brick worker load is visible and a later frame can recover", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  const worker = "**/brickedContour.worker*.js*";
  await page.route(worker, route => route.abort("failed"));
  await page.goto("/harness/index.html?missingGeometry=1&derivedBricks=1&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setPlot("isosurface"); window.__nimbusVtkHarness!.setMounted(true); });
  await expect(page.getByRole("toolbar", { name: "Isosurface controls" })).toBeVisible();
  const data = page.locator(".plot3d-control-tab", { hasText: "Data" });
  if (await data.getAttribute("aria-expanded") !== "true") await data.click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  await expect(page.locator(".volume-render__overlay--error")).toContainText("Field failed:");
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-brick-error", /.+/);
  await expect(page.getByText("Pick a 3D scalar field above.", { exact: true })).toHaveCount(0);
  await page.unroute(worker);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.framePlot === "isosurface" && scene.dataset.frameLoading === "false" && scene.dataset.renderedStep === "1";
  }, undefined, { timeout: 30_000 });
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(capture?.evidence.descriptor.frameIndex).toBe(1);
  expect(capture?.evidence.layers.some(layer => layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);
  expect(errors).toEqual([]);
});
