import { test, expect } from "@playwright/test";
import { SCENE_SETUP_TIMEOUT_MS, withSceneErrors } from "../support/sceneBudgets";

for (const memory of [8, 48]) test(`volume zoom honors ${memory} GiB source-budget policy`, async ({ page }, info) => {
  // Initial data, field replacement and native promotion are separate full-quality preparations.
  test.setTimeout(3 * SCENE_SETUP_TIMEOUT_MS + 60_000);
  const phases: Array<{ name: string; elapsedMs: number; readiness: unknown }> = [];
  const ready = async (name: string, field: string | null = null, native: boolean | null = null, full = false) => {
    const started = Date.now();
    const result = await page.waitForFunction(({ field, native, full }) => {
      const error = document.querySelector(".volume-render__overlay--error");
      if (error) throw new Error(error.textContent ?? "Volume preparation failed");
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset;
      const hosts = Array.from(document.querySelectorAll<HTMLElement>(".plot3d-overlay-host"), host => host.dataset);
      const appearance = window.__nimbusVtkHarness?.inspectAppearance();
      const settled = scene?.framePlot === "volume" && scene.frameLoading === "false" && scene.renderedStep === "0"
        && hosts.length === 2 && hosts.every(host => host.volumeMaterialCommitted === "true" && host.volumeQualityPreview === "false")
        && appearance?.descriptor.frameIndex === 0 && appearance.descriptor.plot === "volume"
        && (field == null || appearance.descriptor.field === field)
        && (native == null || hosts[0].volumeZoomNative === String(native))
        && (!full || appearance.layers.some(layer => layer.visible && layer.pointCount === 256 * 256 * 160));
      return settled ? { frame: scene.renderedStep, field: appearance.descriptor.field,
        native: hosts[0].volumeZoomNative, stride: appearance.descriptor.parameters?.stride,
        visiblePoints: appearance.layers.filter(layer => layer.visible).map(layer => layer.pointCount),
        materialCommitted: hosts.map(host => host.volumeMaterialCommitted),
        qualityPreview: hosts.map(host => host.volumeQualityPreview) } : false;
    }, { field, native, full }, { timeout: SCENE_SETUP_TIMEOUT_MS });
    try { phases.push({ name, elapsedMs: Date.now() - started, readiness: await result.jsonValue() }); }
    finally { await result.dispose(); }
  };
  try {
    await withSceneErrors(page, async () => {
      await page.goto(`/harness/index.html?large=1&panes=2&volumeBricks=off&volumeRamGiB=${memory}`);
      await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
      await ready("initial");
      const host = page.locator(".plot3d-overlay-host").first();
      const tab = page.locator(".plot3d-control-tab", { hasText: "Volume" }).first();
      if (await tab.getAttribute("aria-expanded") !== "true") await tab.click();
      await page.locator('select[aria-label="Field"]').first().selectOption("w");
      await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([80000, 80000, 80000], [12800, 12800, 8000], false));
      await ready("far-field", "w", false);
      await expect(host).toHaveAttribute("data-volume-zoom-native", "false");
      await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([16000, 16000, 11000], [12800, 12800, 8000], false));
      await page.waitForTimeout(600);
      await ready("near-field", "w", memory === 48, memory === 48);
      await expect(host).toHaveAttribute("data-volume-zoom-native", memory === 48 ? "true" : "false");
      const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      expect(capture).not.toBeNull();
      const parameters = capture!.evidence.descriptor.parameters!;
      expect(capture!.evidence.descriptor.field).toBe("w");
      expect(capture!.evidence.descriptor.frameIndex).toBe(0);
      if (memory === 48) {
        expect(parameters.stride).toEqual([1, 1, 1]);
        expect(capture!.evidence.layers.some(layer => layer.visible && layer.pointCount === 256 * 256 * 160)).toBe(true);
      } else expect((parameters.stride as number[]).some(stride => stride > 1)).toBe(true);
      const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads);
      await page.waitForTimeout(500);
      expect(await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads)).toBe(before);
      await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
      await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
    });
  } finally {
    await info.attach("volume-zoom-readiness", { body: JSON.stringify(phases, null, 2), contentType: "application/json" });
  }
});
