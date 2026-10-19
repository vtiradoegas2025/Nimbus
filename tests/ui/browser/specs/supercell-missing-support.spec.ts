import { test, expect, type Page } from "@playwright/test";

async function ready(page: Page, step = 0) {
  await page.waitForFunction(step => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.framePlot === "supercell" && scene.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === String(step);
  }, step, { timeout: 60_000 });
}
async function toggle(page: Page, tab: string, group: string) {
  await page.locator(".plot3d-control-tab", { hasText: tab }).click();
  await page.getByRole("group", { name: group, exact: true }).getByRole("button").first().click();
  await ready(page);
}
for (const mode of ["1", "empty-next"]) test(`Supercell preserves valid geometry and recovers from missing support: ${mode}`, async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`/harness/index.html?missingGeometry=${mode}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  await toggle(page, "Flow", "Updraft / downdraft");
  await toggle(page, "Vortex", "Vortex core");
  await page.getByRole("group", { name: "Tornadic condensation", exact: true }).getByRole("button").first().click();
  await ready(page);
  for (const step of [0, 1, 0]) {
    await page.evaluate(step => window.__nimbusVtkHarness!.setStep(step), step);
    await ready(page, step);
    const captured = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    const surfaces = captured!.evidence.layers.filter(layer => layer.kind === "actor").slice(0, 5);
    if (mode === "empty-next" && step === 1) {
      expect(surfaces.every(layer => !layer.visible && layer.pointCount === 0)).toBe(true);
    } else {
      expect(surfaces.slice(0, 2).some(layer => layer.visible && (layer.pointCount ?? 0) > 0)).toBe(true);
      expect(surfaces.slice(2).some(layer => layer.visible && (layer.pointCount ?? 0) > 0)).toBe(true);
    }
    if (mode === "1") {
      const support = await page.evaluate(() => window.__nimbusVtkHarness!.inspectSurfaceGap(2800, 3325));
      expect(support.triangles).toBeGreaterThan(0);
      expect(support.crossings).toBe(0);
      expect(support.nonFinite).toBe(0);
      await expect(page.locator(".plot3d-badge__sub")).toContainText("missing regions");
      await info.attach(`gap-${step}`, { body: JSON.stringify(support), contentType: "application/json" });
    }
    const volumes = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeSupport());
    expect(volumes.every(volume => volume.finite)).toBe(true);
    if (mode === "1" && step === 1) {
      expect(volumes.find(volume => volume.index === 2)?.missing).toBe(4608);
    }
    await info.attach(`frame-${step}`, { body: JSON.stringify({ evidence: captured!.evidence, volumes }), contentType: "application/json" });
  }
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes)).toBe(0);
  expect(errors).toEqual([]);
});

test("Presentation cloud keeps the missing-support mask after display refinement", async ({ page }, info) => {
  await page.goto("/harness/index.html?missingGeometry=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setCoarse(true); window.__nimbusVtkHarness!.setPlot("supercell"); });
  await ready(page);
  const volumes = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeSupport());
  expect(volumes.length).toBeGreaterThan(0);
  expect(volumes.every(volume => volume.finite)).toBe(true);
  expect(volumes.some(volume => volume.missing > 0)).toBe(true);
  await expect(page.locator(".plot3d-badge__sub")).toContainText("missing regions");
  await info.attach("presentation-support", { body: JSON.stringify(volumes), contentType: "application/json" });
});
