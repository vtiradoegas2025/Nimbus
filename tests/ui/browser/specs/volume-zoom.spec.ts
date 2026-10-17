import { test, expect } from "@playwright/test";

for (const memory of [8,48]) test(`volume zoom honors ${memory} GiB source-budget policy`, async ({ page }) => {
  test.setTimeout(120000);
  await page.goto(`/harness/index.html?large=1&panes=2&volumeBricks=off&volumeRamGiB=${memory}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  const host = page.locator(".plot3d-overlay-host").first();
  const tab = page.locator(".plot3d-control-tab", { hasText: "Volume" }).first();
  if (await tab.getAttribute("aria-expanded") !== "true") await tab.click();
  await page.locator('select[aria-label="Field"]').first().selectOption("w");
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([80000,80000,80000],[12800,12800,8000],false));
  await expect(host).toHaveAttribute("data-volume-zoom-native","false");
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([16000,16000,11000],[12800,12800,8000],false));
  await page.waitForTimeout(600);
  await expect(host).toHaveAttribute("data-volume-zoom-native", memory === 48 ? "true" : "false");
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  if (memory === 48) await expect.poll(async () => {
    const rendered = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    return rendered?.evidence.layers.some(layer => layer.visible && layer.pointCount === 256*256*160);
  }, { timeout: 30000 }).toBe(true);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const parameters = capture!.evidence.descriptor.parameters!;
  if (memory === 48) {
    expect(parameters.stride).toEqual([1,1,1]);
    expect(capture!.evidence.layers.some(layer => layer.visible && layer.pointCount === 256*256*160)).toBe(true);
  } else expect((parameters.stride as number[]).some(stride => stride > 1)).toBe(true);
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads);
  await page.waitForTimeout(500);
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads)).toBe(before);
  await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
});
