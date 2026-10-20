import { test, expect } from "@playwright/test";
import { SCENE_SETUP_TIMEOUT_MS, withSceneErrors } from "../support/sceneBudgets";

test("volume preparation tolerates a delayed cold worker module", async ({ page }) => {
  let loads = 0;
  await page.route(/\/vtkPreparation\.worker[^/]*\.(?:js|ts)(?:\?|$)/, async route => {
    loads += 1;
    await new Promise(resolve => setTimeout(resolve, 4_000));
    await route.continue();
  });
  await page.goto("/harness/index.html?volumeBricks=off");
  await withSceneErrors(page, () => page.waitForFunction(() => {
    const error = document.querySelector(".volume-render__overlay--error");
    if (error) throw new Error(error.textContent ?? "Delayed module preparation failed");
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset;
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    return scene?.frameLoading === "false" && scene.renderedStep === "0"
      && host?.volumeMaterialCommitted === "true";
  }, undefined, { timeout: 45_000 }));
  expect(loads).toBeGreaterThan(0);
  await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
});

for (const recovery of ["replacement", "budget change"] as const) test(`a failed required volume context recovers through ${recovery}`, async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("/harness/index.html?large=1&volumeBricks=off&volumeRamGiB=48&volumeContextReadFailure=w");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const host = page.locator(".plot3d-overlay-host").first();
  const scene = page.getByTestId("scene");
  await withSceneErrors(page, () => page.waitForFunction(() => {
    const error = document.querySelector(".volume-render__overlay--error");
    if (error) throw new Error(error.textContent ?? "Initial volume preparation failed");
    return document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false";
  }, undefined, { timeout: SCENE_SETUP_TIMEOUT_MS }));
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([18000,18000,12000],[16000,16000,9000],false));
  await expect(host).toHaveAttribute("data-volume-zoom-native", "true");
  const volumeTab = page.locator(".plot3d-control-tab", { hasText: "Volume" }).first();
  if (await volumeTab.getAttribute("aria-expanded") !== "true") await volumeTab.click();
  const field = page.getByRole("combobox", { name: "Field", exact: true });
  await field.selectOption("w");
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([18000,18000,12000],[16000,16000,9000],false));
  await expect(host).toHaveAttribute("data-volume-zoom-native", "true");
  await expect(host).not.toHaveAttribute("data-scalar-transform", "loading", { timeout: 45_000 });
  await expect(page.locator(".volume-render__overlay--error")).toContainText(
    "required reduced context read exceeds budget", { timeout: 45_000 });
  await expect(scene).toHaveAttribute("data-rendered-step", "");
  await expect(field).toBeEnabled();
  const camera = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))?.evidence.camera;
  expect(camera).toBeDefined();
  if (recovery === "replacement") await field.selectOption("vorticity_z");
  else await page.evaluate(() => {
    window.__nimbusVolumeBudgetFixture!.allowContextRead();
    window.__nimbusVolumeBudgetFixture!.publishIoLimit(1024 * 1024 ** 2);
  });
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await expect(host).toHaveAttribute("data-volume-material-committed", "true", { timeout: 45_000 });
  if (recovery === "replacement") await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([18000,18000,12000],[16000,16000,9000],false));
  await expect(host).toHaveAttribute("data-volume-zoom-native", "true");
  await expect(host).toHaveAttribute("data-volume-context-ready", "true", { timeout: 45_000 });
  await expect(host).toHaveAttribute("data-volume-material-committed", "true", { timeout: 45_000 });
  await expect(scene).toHaveAttribute("data-frame-loading", "false");
  if (recovery === "budget change") {
    await expect(field).toHaveValue("w");
    const after = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))?.evidence.camera;
    expect(after).toBeDefined();
    expect(after?.position).toEqual(camera?.position);
    expect(after?.focalPoint).toEqual(camera?.focalPoint);
  }
  await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
});
