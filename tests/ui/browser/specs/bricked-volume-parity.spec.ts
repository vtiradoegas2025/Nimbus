import { expect, test, type Page } from "@playwright/test";

async function captureVolume(
  page: Page,
  monolithic: boolean,
  extraQuery = "",
  field = "dense_signed",
  datasetQuery = "derivedBricks=1",
  fixedCamera = false,
) {
  const suffix = `${monolithic ? "&volumeBricks=off" : ""}${extraQuery}`;
  await page.goto(`/harness/index.html?${datasetQuery}${suffix}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("volume"));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption(field);
  await expect.poll(async () => JSON.stringify(await page.evaluate((expected) => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.volumeMaterialCommitted === "true"
        && host.dataset.volumeQualityPreview === "false"
        && host.dataset.volumeRepresentation === expected,
      representation: host?.dataset.volumeRepresentation,
      error: document.querySelector(".volume-render__overlay--error")?.textContent ?? null,
    };
  }, monolithic ? "full-domain" : "progressive-bricks")), {
    timeout: 120_000,
  }).toContain('"ready":true');
  if (fixedCamera) await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([17937.5, -8937.5, 9035], [2937.5, 3062.5, 1035], false));
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  return capture!;
}

async function releaseVolume(page: Page) {
  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === snapshot?.backgroundLoad.workers
      && snapshot?.visualization.residentBytes === 0
      && snapshot?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
}

test("stretched volume bricks share the monolithic physical lattice despite nominal spacing", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const bricked = await captureVolume(page, false, "&stretchedBricks=1", "dense_signed", "derivedBricks=1", true);
  const monolithic = await captureVolume(page, true, "&stretchedBricks=1", "dense_signed", "derivedBricks=1", true);
  expect(bricked.evidence.camera.position).toEqual(monolithic.evidence.camera.position);
  const metrics = await page.evaluate(([reference, candidate]) => window.__nimbusVtkHarness!.compareAppearance(reference, candidate), [monolithic.pngDataUri, bricked.pngDataUri] as const);
  await info.attach("stretched-volume-comparison", { body: JSON.stringify({ metrics, bricked: bricked.evidence, monolithic: monolithic.evidence }), contentType: "application/json" });
  expect(metrics.rgbRmse).toBeLessThan(0.004);
  expect(metrics.meanAbsoluteError).toBeLessThan(0.001);
  expect(errors).toEqual([]);
  await releaseVolume(page);
});

test("native volume bricks preserve monolithic scalar composition and image structure", async ({ page }) => {
  test.setTimeout(8 * 60_000);
  const bricked = await captureVolume(page, false);
  const monolithic = await captureVolume(page, true);
  expect(bricked.evidence.descriptor.parameters?.scalarCompositionFingerprint)
    .toBe(monolithic.evidence.descriptor.parameters?.scalarCompositionFingerprint);

  const metrics = await page.evaluate(
    ([reference, candidate]) => window.__nimbusVtkHarness?.compareAppearance(reference, candidate) ?? null,
    [monolithic.pngDataUri, bricked.pngDataUri] as const,
  );
  expect(metrics).not.toBeNull();
  expect(metrics!.rgbRmse).toBeLessThanOrEqual(0.08);
  expect(metrics!.meanAbsoluteError).toBeLessThanOrEqual(0.02);
  expect(metrics!.changedPixelFraction).toBeLessThanOrEqual(0.12);
  expect(metrics!.edgeRmse).toBeLessThanOrEqual(0.1);

  await releaseVolume(page);
});

test("durable coarse volume bricks preserve fine-level image structure", async ({ page }, info) => {
  test.setTimeout(8 * 60_000);
  const fine = await captureVolume(page, false, "", "dense_signed", "derivedBricks=1", true);
  const coarse = await captureVolume(page, false, "&brickLod=1", "dense_signed", "derivedBricks=1", true);
  expect(coarse.evidence.camera.position).toEqual(fine.evidence.camera.position);
  expect(fine.evidence.descriptor.parameters?.brickHierarchyLevel).toBe(0);
  expect(coarse.evidence.descriptor.parameters?.brickHierarchyLevel).toBe(1);
  const metrics = await page.evaluate(
    ([reference, candidate]) => window.__nimbusVtkHarness?.compareAppearance(reference, candidate) ?? null,
    [fine.pngDataUri, coarse.pngDataUri] as const,
  );
  expect(metrics).not.toBeNull();
  await info.attach("coarse-comparison", { body: JSON.stringify({ metrics, fine: fine.evidence, coarse: coarse.evidence }), contentType: "application/json" });
  expect(metrics!.rgbRmse).toBeLessThanOrEqual(0.02);
  expect(metrics!.meanAbsoluteError).toBeLessThanOrEqual(0.005);
  expect(metrics!.changedPixelFraction).toBeLessThanOrEqual(0.03);
  expect(metrics!.edgeRmse).toBeLessThanOrEqual(0.01);
  await releaseVolume(page);
});

test("background-stratified bricks share one global anomaly profile", async ({ page }) => {
  test.setTimeout(8 * 60_000);
  const bricked = await captureVolume(page, false, "", "temperature");
  const monolithic = await captureVolume(page, true, "", "temperature");
  expect(bricked.evidence.descriptor.parameters?.scalarTransform)
    .toBe("horizontal-level-anomaly");
  expect(bricked.evidence.descriptor.parameters?.scalarCompositionQuantizedFingerprint)
    .toBe(monolithic.evidence.descriptor.parameters?.scalarCompositionQuantizedFingerprint);
  const metrics = await page.evaluate(
    ([reference, candidate]) => window.__nimbusVtkHarness?.compareAppearance(reference, candidate) ?? null,
    [monolithic.pngDataUri, bricked.pngDataUri] as const,
  );
  expect(metrics).not.toBeNull();
  expect(metrics!.rgbRmse).toBeLessThanOrEqual(0.08);
  expect(metrics!.meanAbsoluteError).toBeLessThanOrEqual(0.02);
  expect(metrics!.changedPixelFraction).toBeLessThanOrEqual(0.12);
  expect(metrics!.edgeRmse).toBeLessThanOrEqual(0.1);
  await releaseVolume(page);
});

test("terrain background bricks share the monolithic physical-height anomaly", async ({ page }) => {
  test.setTimeout(8 * 60_000);
  const bricked = await captureVolume(page, false, "", "temperature", "terrainBricks=1");
  const monolithic = await captureVolume(page, true, "", "temperature", "terrainBricks=1");
  expect(bricked.evidence.descriptor.parameters?.scalarTransform)
    .toBe("horizontal-level-anomaly");
  expect(bricked.evidence.descriptor.parameters?.scalarCompositionQuantizedFingerprint)
    .toBe(monolithic.evidence.descriptor.parameters?.scalarCompositionQuantizedFingerprint);
  const metrics = await page.evaluate(
    ([reference, candidate]) => window.__nimbusVtkHarness?.compareAppearance(reference, candidate) ?? null,
    [monolithic.pngDataUri, bricked.pngDataUri] as const,
  );
  expect(metrics).not.toBeNull();
  expect(metrics!.rgbRmse).toBeLessThanOrEqual(0.08);
  expect(metrics!.meanAbsoluteError).toBeLessThanOrEqual(0.02);
  expect(metrics!.changedPixelFraction).toBeLessThanOrEqual(0.12);
  expect(metrics!.edgeRmse).toBeLessThanOrEqual(0.1);
  await releaseVolume(page);
});
