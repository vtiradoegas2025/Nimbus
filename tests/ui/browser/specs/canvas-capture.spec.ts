import { expect, test } from "@playwright/test";

for (const plot of ["volume", "supercell"] as const) {
  test(`owned ${plot} snapshots preserve upstream pixels in the same render pass`, async ({ page }) => {
    await page.goto("/harness/index.html");
    await page.waitForFunction(() => !!window.__nimbusVtkHarness);
    await page.evaluate(plot => {
      window.__nimbusVtkHarness!.setSize(480, 320);
      window.__nimbusVtkHarness!.setPlot(plot);
    }, plot);
    await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
    await expect(page.locator(".volume-render__canvas canvas")).toBeVisible();
    await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    const result = await page.evaluate(() => window.__nimbusVtkHarness!.compareCanvasCapturePaths());
    expect(result.bytes).toBeGreaterThan(1000);
    expect(result.equal).toBe(true);
  });
}
