import { test, expect } from "@playwright/test";

test("wind-informed cloud preview uses bounded reads and keeps exact endpoints and diagnostics", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const counters = { compiles: 0 };
    Object.assign(window, { windShaderCounters: counters });
    const compile = WebGL2RenderingContext.prototype.compileShader;
    WebGL2RenderingContext.prototype.compileShader = function (shader) { counters.compiles++; compile.call(this, shader); };
  });
  await page.goto("/harness/index.html?fidelity=presentation");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
  const ready = (step: number) => page.waitForFunction(step => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(step);
  }, step);
  await ready(0);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1)); await ready(1);
  const endpoint = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(0)); await ready(0);
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "Wind-informed", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().ipc.startedJobs)).toBe(before.ipc.startedJobs + 3);
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().ipc.inFlightJobs)).toBe(0);
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
  await page.waitForTimeout(100);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const motion = capture!.evidence.descriptor.parameters!.cloudMotion as { alignment: string; displacementM: number[]; timeS: number };
  expect(motion.alignment).toBe("wind");
  expect(motion.displacementM).toHaveLength(3);
  expect(motion.displacementM.every(Number.isFinite)).toBe(true);
  expect(Math.abs(motion.displacementM[2])).toBeGreaterThan(0);
  expect(capture!.evidence.descriptor.frameIndex).toBeNull();
  const compilesBefore = await page.evaluate(() => (window as unknown as { windShaderCounters: { compiles: number } }).windShaderCounters.compiles);
  const during = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.waitForTimeout(150);
  const later = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(later.ipc.startedJobs).toBe(during.ipc.startedJobs);
  expect(later.texture3DUploads).toBe(during.texture3DUploads);
  expect(await page.evaluate(() => (window as unknown as { windShaderCounters: { compiles: number } }).windShaderCounters.compiles)).toBe(compilesBefore);
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(false, true));
  await ready(1);
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints)).toEqual(endpoint);
  await expect(page.getByRole("button", { name: "Wind-informed", exact: true })).toBeDisabled();
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(false, false));
  await page.locator(".plot3d-control-tab", { hasText: "Precip" }).click();
  await page.getByRole("group", { name: "Precipitation", exact: true }).getByRole("button").first().click();
  await ready(1);
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await expect(page.getByRole("button", { name: "Wind-informed", exact: true })).toBeDisabled();
  expect((await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence.descriptor.frameIndex).toBe(1);
  await info.attach("wind-motion", { body: JSON.stringify({ motion, before, during, later }), contentType: "application/json" });
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().preparedRegistry.handles)).toBe(0);
  await page.evaluate(() => window.__nimbusVtkHarness!.clearFieldCache());
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
  expect(errors).toEqual([]);
});

for (const [scenario, reason] of [["missing", "Insufficient cloud-supported wind samples"], ["fast", "Cloud displacement exceeds"], ["read-error", "Wind point sampling unavailable"]]) {
  test(`unusable ${scenario} winds retain exact cloud frames`, async ({ page }) => {
    await page.goto(`/harness/index.html?fidelity=presentation&windMotion=${scenario}`);
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
    await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
    await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
    await page.getByRole("button", { name: "Wind-informed", exact: true }).click();
    await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
    await page.waitForTimeout(200);
    await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
    await page.waitForFunction(() => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === "1";
    });
    await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-cloud-motion-active", "false");
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    const motion = capture!.evidence.descriptor.parameters!.cloudMotion as { reason: string; alignment: string | null; timeS: number };
    expect(capture!.evidence.descriptor.frameIndex).toBe(1);
    expect(motion.reason).toContain(reason);
    expect(motion.alignment).toBeNull();
    expect(motion.timeS).toBe(30);
    await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  });
}

for (const [tab, group] of [["Flow", "Updraft / downdraft"], ["Vortex", "Vortex core"]]) {
  test(`${tab} diagnostics interrupt inferred motion and commit one exact scene time`, async ({ page }) => {
    await page.goto("/harness/index.html?fidelity=presentation");
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
    const ready = () => page.waitForFunction(() => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      return scene?.dataset.frameLoading === "false";
    });
    await ready();
    await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
    await page.getByRole("button", { name: "Wind-informed", exact: true }).click();
    await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
    await page.waitForTimeout(100);
    await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
    await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
    await page.evaluate(() => window.__nimbusVtkHarness!.setFieldReadDelay(100));
    await page.locator(".plot3d-control-tab", { hasText: tab }).click();
    await page.getByRole("group", { name: group, exact: true }).getByRole("button").first().click();
    await ready();
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    const motion = capture!.evidence.descriptor.parameters!.cloudMotion as { active: boolean; timeS: number; alignment: string | null };
    expect(capture!.evidence.descriptor.frameIndex).toBe(1);
    expect(motion.active).toBe(false); expect(motion.timeS).toBe(30); expect(motion.alignment).toBeNull();
    expect(capture!.evidence.layers.some(layer => layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);
    await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
    await expect(page.getByRole("button", { name: "Wind-informed", exact: true })).toBeDisabled();
  });
}
