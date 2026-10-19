import { expect, test, type Page } from "@playwright/test";

async function captureStreamlines(page: Page, monolithic: boolean) {
  const suffix = monolithic ? "&monolithicVectors=1" : "";
  await page.goto(`/harness/index.html?derivedBricks=1${suffix}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("streamlines"));
  await expect.poll(async () => JSON.stringify(await page.evaluate((expected) => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.streamlineLoading === "false"
        && host.dataset.streamlineRepresentation === expected
        && host.dataset.streamlineGeometryCommitted === "true",
      representation: host?.dataset.streamlineRepresentation,
      error: document.querySelector(".volume-render__overlay--error")?.textContent ?? null,
    };
  }, monolithic ? "bounded-monolithic" : "trajectory-native-bricks")), {
    timeout: 120_000,
  }).toContain('"ready":true');
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  return capture!;
}

test("trajectory bricks match the full-domain streamline geometry and image", async ({ page }, info) => {
  test.setTimeout(8 * 60_000);
  const bricked = await captureStreamlines(page, false);
  const monolithic = await captureStreamlines(page, true);

  expect(bricked.evidence.descriptor.parameters?.seedFingerprint)
    .toBe(monolithic.evidence.descriptor.parameters?.seedFingerprint);
  expect(bricked.evidence.descriptor.parameters?.integrationStep)
    .toBe(monolithic.evidence.descriptor.parameters?.integrationStep);
  expect(bricked.evidence.descriptor.parameters?.geometryFingerprint)
    .toBe(monolithic.evidence.descriptor.parameters?.geometryFingerprint);
  expect(bricked.evidence.descriptor.parameters?.integratedPoints)
    .toBe(monolithic.evidence.descriptor.parameters?.integratedPoints);
  const visibleLayers = (capture: typeof bricked) => capture.evidence.layers
    .filter((layer) => layer.visible)
    .map((layer) => ({
      kind: layer.kind,
      pointCount: layer.pointCount,
      cellCount: layer.cellCount,
    }));
  expect(visibleLayers(bricked)).toEqual(visibleLayers(monolithic));

  const metrics = await page.evaluate(
    ([reference, candidate]) => window.__nimbusVtkHarness?.compareAppearance(reference, candidate) ?? null,
    [monolithic.pngDataUri, bricked.pngDataUri] as const,
  );
  await info.attach("parity", {body: JSON.stringify({bricked:bricked.evidence,monolithic:monolithic.evidence,metrics}),contentType:"application/json"});
  expect(metrics).not.toBeNull();
  expect(metrics!.rgbRmse).toBeLessThanOrEqual(0.015);
  expect(metrics!.meanAbsoluteError).toBeLessThanOrEqual(0.004);
  expect(metrics!.changedPixelFraction).toBeLessThanOrEqual(0.035);
  expect(metrics!.edgeRmse).toBeLessThanOrEqual(0.015);

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
});
