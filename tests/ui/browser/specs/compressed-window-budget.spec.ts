import { test, expect } from "@playwright/test";

test("compressed plot metadata admits decoder work without changing samples or per-frame wind reads", async ({ page }) => {
  await page.goto("/harness/index.html?compressedPlotBudget=1&fidelity=presentation");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "Wind-informed", exact: true }).click();
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
  await page.getByText("Waiting for the next saved cloud frame", { exact: true }).waitFor();
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  const readRequests = () => page.evaluate(() => (window as unknown as { compressedWindowRequests: { field: string; aggregation: string; stride: number[]; extent: number[]; maxSourceBytes: number; maxOutputBytes: number }[] }).compressedWindowRequests);
  const requests = await readRequests();
  const winds = requests.filter(request => request.aggregation === "point_sample");
  expect(winds.map(request => request.field).sort()).toEqual(["u", "u", "v", "v", "w", "w"]);
  expect(winds.every(request => request.maxSourceBytes === 32 * 1024 * 1024 && request.maxOutputBytes === 131072)).toBe(true);
  const clouds = requests.filter(request => request.field === "cloud_condensate");
  expect(clouds.length).toBeGreaterThan(0);
  expect(clouds.every(request => request.stride.every(value => value === 1)
    && request.maxSourceBytes > request.extent.reduce((count, size) => count * size, 4))).toBe(true);
  await page.waitForTimeout(150);
  expect((await readRequests()).length).toBe(requests.length);
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes)).toBe(0);
});
