import { expect, test } from "@playwright/test";

async function waitForBrickedStreamlines(page: import("@playwright/test").Page, level: number) {
  await expect.poll(async () => JSON.stringify(await page.evaluate((expectedLevel) => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.streamlineLoading === "false"
        && host.dataset.streamlineRepresentation === "trajectory-native-bricks"
        && host.dataset.vectorBrickLevel === String(expectedLevel)
        && host.dataset.streamlineGeometryCommitted === "true"
        && Number(host.dataset.logicalGpuUploadBytes ?? 0) > 0
        && Number(host.dataset.streamlineIntegratedPoints ?? 0) > 0,
      representation: host?.dataset.streamlineRepresentation,
      level: host?.dataset.vectorBrickLevel,
      bricks: host?.dataset.vectorBrickCount,
      total: host?.dataset.vectorBrickTotal,
      integrated: host?.dataset.streamlineIntegratedPoints,
      gpu: host?.dataset.logicalGpuUploadBytes,
      error: document.querySelector(".volume-render__overlay--error")?.textContent ?? null,
    };
  }, level)), { timeout: 120_000 }).toContain('"ready":true');
}

test("streamlines expand native vector bricks when trajectories reach a boundary", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("streamlines"));
  await waitForBrickedStreamlines(page, 0);
  const host = page.locator(".plot3d-overlay-host");
  expect(Number(await host.getAttribute("data-vector-brick-count"))).toBeGreaterThan(1);
  expect(Number(await host.getAttribute("data-vector-brick-iteration"))).toBeGreaterThan(0);
  const readySnapshot = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(readySnapshot.visualization.byStage.gpu.residentBytes).toBeGreaterThan(0);

  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.representation).toBe("trajectory-native-bricks");
  expect(capture?.evidence.descriptor.parameters?.brickHierarchyLevel).toBe(0);
  expect(Number(capture?.evidence.descriptor.parameters?.logicalGpuUploadBytes)).toBeGreaterThan(0);
  expect(capture?.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.getByRole("button", { name: /Tracing/ }).click();
  await page.locator('.plot3d-control-panel input[type="range"]').nth(1)
    .evaluate((element: HTMLInputElement) => {
      element.value = String(Math.min(Number(element.max), Number(element.value) + 50));
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForTimeout(50);
  await waitForBrickedStreamlines(page, 0);
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);

  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === snapshot?.backgroundLoad.workers
      && snapshot?.visualization.residentBytes === 0
      && snapshot?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("streamline trajectory expansion remains inside its admitted brick budget", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("streamlines"));
  await waitForBrickedStreamlines(page, 0);
  const host = page.locator(".plot3d-overlay-host");
  expect(Number(await host.getAttribute("data-vector-brick-count"))).toBe(2);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.brickHierarchyLevel).toBe(0);

  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === snapshot?.backgroundLoad.workers
      && snapshot?.visualization.residentBytes === 0
      && snapshot?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("resolved-feature seeds discover a hierarchy-ranked brick without a full-field scan", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("streamlines"));
  await waitForBrickedStreamlines(page, 0);
  await page.getByRole("button", { name: /Tracing/ }).click();
  await page.getByRole("button", { name: "Resolved features" }).click();
  await waitForBrickedStreamlines(page, 0);
  const host = page.locator(".plot3d-overlay-host");
  const selected = Number(await host.getAttribute("data-vector-brick-count"));
  const total = Number(await host.getAttribute("data-vector-brick-total"));
  expect(selected).toBeGreaterThan(0);
  expect(selected).toBeLessThan(total);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.seedMode).toBe("feature");
  expect(capture?.evidence.descriptor.parameters?.representation).toBe("trajectory-native-bricks");

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

test("terrain-following vector bricks remap physical volumes before trajectory integration", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?terrainBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("streamlines"));
  await waitForBrickedStreamlines(page, 0);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.representation).toBe("trajectory-native-bricks");
  expect(capture?.evidence.descriptor.parameters?.preparedShape).toEqual(expect.any(Array));
  expect(capture?.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);

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

test("vector brick streamlines reject superseded playback generations", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("streamlines"));
  const waitForFrame = async (step: number) => expect.poll(async () => JSON.stringify(
    await page.evaluate((expectedStep) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const snapshot = window.__nimbusVtkHarness?.snapshot();
      return {
        ready: scene?.dataset.renderedStep === String(expectedStep)
          && scene.dataset.frameLoading === "false"
          && host?.dataset.vectorBrickLevel === "0"
          && host.dataset.streamlineGeometryCommitted === "true"
          && snapshot?.ipc.inFlightJobs === 0,
        rendered: scene?.dataset.renderedStep,
        level: host?.dataset.vectorBrickLevel,
        integrated: host?.dataset.streamlineIntegratedPoints,
      };
    }, step),
  ), { timeout: 120_000 }).toContain('"ready":true');
  await waitForFrame(0);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(20));
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
  await page.waitForTimeout(5);
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(2));
  await waitForFrame(2);

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setFieldReadDelay(0);
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
