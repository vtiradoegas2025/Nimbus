import { expect, test } from "@playwright/test";

test("ZFP-safe hierarchy drives progressive native-brick isosurface extraction", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await page.waitForFunction(() =>
    document.querySelector(".harness-shell")?.getAttribute("data-plot") === "isosurface");

  await page.getByRole("button", { name: /Data/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("cloud_condensate");
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return scene?.dataset.framePlot === "isosurface"
      && scene.dataset.frameLoading === "false"
      && host?.dataset.contourRepresentation === "progressive-bricks"
      && snapshot?.ipc.inFlightJobs === 0
      && document.querySelector(".volume-render__overlay--error") == null;
  }, undefined, { timeout: 4 * 60_000 });

  await expect(page.locator(".plot3d-badge__sub")).toContainText("progressive native bricks");
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  if (!capture) return;
  expect(capture.evidence.descriptor.frameIndex).toBe(0);
  expect(capture.evidence.descriptor.parameters?.renderedPublicationGeneration).toEqual(expect.any(String));
  const parameters = capture.evidence.descriptor.parameters ?? {};
  expect(parameters.representation).toBe("progressive-native-bricks");
  expect(Number(parameters.selectedBrickCount)).toBeGreaterThan(0);
  expect(Number(parameters.selectedBrickCount)).toBeLessThan(80);
  expect(Number(parameters.culledBrickCount)).toBeGreaterThan(0);
  expect(Number(parameters.brickSourceBytes)).toBeLessThan(40 * 1024 * 1024);
  expect(Number(parameters.logicalGpuUploadBytes)).toBeGreaterThan(0);
  expect(Number(await page.locator(".plot3d-overlay-host")
    .getAttribute("data-logical-gpu-upload-bytes"))).toBeGreaterThan(0);
  const surface = capture.evidence.layers.find((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0);
  expect(surface?.cellCount).toBeGreaterThan(0);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  const residentBeforeEdit = Number(parameters.residentBrickCount);
  expect(residentBeforeEdit).toBeGreaterThanOrEqual(Number(parameters.selectedBrickCount));
  await page.getByRole("button", { name: /Surface/ }).click();
  const threshold = page.locator('.plot3d-control-panel input[type="range"]').first();
  await threshold.evaluate((element: HTMLInputElement) => {
    const current = Number(element.value);
    element.value = String(Math.min(Number(element.max), current * 1.5));
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")
      ?.dataset.contourLoading === "true");
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false"
      && host?.dataset.contourLoading === "false";
  }, undefined, { timeout: 120_000 });
  const afterEditCapture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(afterEditCapture).not.toBeNull();
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);
  expect(Number(afterEditCapture!.evidence.descriptor.parameters?.residentBrickCount))
    .toBe(residentBeforeEdit);
  expect(afterEditCapture!.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);

  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  await page.waitForFunction(() =>
    (window.__nimbusVtkHarness?.snapshot().visualization.byStage["decoded-field"].residentBytes
      ?? Number.POSITIVE_INFINITY) < 40 * 1024 * 1024);
  const settled = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(settled.visualization.byStage["decoded-field"].residentBytes).toBeLessThan(
    40 * 1024 * 1024,
  );
  expect(settled.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(settled.visualization.byStage.gpu.residentBytes).toBeGreaterThan(0);
  expect(settled.visualization.prediction.underpredictions).toBe(0);
  expect(pageErrors).toEqual([]);

  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === snapshot?.backgroundLoad.workers
      && snapshot?.preparedRegistry.handles === 0
      && snapshot?.visualization.residentBytes === 0
      && snapshot?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("nested coarse/fine isosurfaces admit and stitch the complete hierarchy atomically", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&nestedVolumes=2");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await page.getByRole("button", { name: /Data/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("w");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.contourLoading === "false"
        && host.dataset.nestedContourLayers === "3"
        && host.dataset.nestedContourAtomicAdmission === "true"
        && host.dataset.nestedContourComposition === "worker-streamed"
        && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
      layers: host?.dataset.nestedContourLayers,
      transitions: host?.dataset.nestedContourTransitionTriangles,
      atomic: host?.dataset.nestedContourAtomicAdmission,
      planned: host?.dataset.nestedContourPlannedLayers,
      enabled: host?.dataset.nestedContourEnabled,
      contourReady: host?.dataset.contourReady,
      threshold: host?.dataset.contourThreshold,
      committed: host?.dataset.contourCommittedThreshold,
      thresholdField: host?.dataset.contourThresholdField,
      field: host?.dataset.contourField,
      scalarEligible: host?.dataset.scalarBrickEligible,
      error: host?.dataset.brickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');

  const host = page.locator(".plot3d-overlay-host");
  await expect(host).toHaveAttribute("data-nested-contour-layers", "3");
  await expect(host).toHaveAttribute("data-nested-contour-atomic-admission", "true");
  await expect(host).toHaveAttribute("data-nested-contour-composition", "worker-streamed");
  expect(Number(await host.getAttribute("data-nested-contour-transition-triangles"))).toBeGreaterThan(0);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.nestedLayerCount).toBe(3);
  expect(capture?.evidence.descriptor.parameters?.nestedAtomicAdmission).toBe(true);
  expect(capture?.evidence.descriptor.parameters?.nestedComposition).toBe("worker-streamed");
  expect(Number(capture?.evidence.descriptor.parameters?.nestedTransitionTriangles)).toBeGreaterThan(0);

  const firstBounds = await host.getAttribute("data-nested-contour-child-bounds-lower");
  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setFieldReadDelay(20);
    window.__nimbusVtkHarness?.setStep(1);
  });
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.contourLoading === "true");
  const duringTransition = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(duringTransition?.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return {
      ready: scene?.dataset.renderedStep === "1"
        && scene.dataset.frameLoading === "false"
        && host?.dataset.contourLoading === "false"
        && host.dataset.nestedContourLayers === "3"
        && host.dataset.nestedContourAtomicAdmission === "true"
        && Number(host.dataset.nestedContourTransitionTriangles ?? 0) > 0
        && snapshot?.visualization.reservedBytes === 0,
      rendered: scene?.dataset.renderedStep,
      loading: host?.dataset.contourLoading,
      error: host?.dataset.brickError,
      reserved: snapshot?.visualization.reservedBytes,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');
  expect(await host.getAttribute("data-nested-contour-child-bounds-lower")).not.toBe(firstBounds);

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setFieldReadDelay(0);
    window.__nimbusVtkHarness?.setMounted(false);
  });
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === snapshot?.backgroundLoad.workers
      && snapshot?.visualization.residentBytes === 0
      && snapshot.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("signed vorticity pair shares one resident union brick set", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?large=1&bricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return {
      frameLoading: scene?.dataset.frameLoading,
      representation: host?.dataset.contourRepresentation,
      contourLoading: host?.dataset.contourLoading,
      contourReady: host?.dataset.contourReady,
      scalarEligible: host?.dataset.scalarBrickEligible,
      diagnosticEligible: host?.dataset.diagnosticBrickEligible,
      declaredBrickFields: host?.dataset.declaredBrickFields,
      threshold: host?.dataset.contourThreshold,
      committedThreshold: host?.dataset.contourCommittedThreshold,
      paired: host?.dataset.contourPaired,
      residentBricks: Number(host?.dataset.brickResidentCount ?? 0),
      rawPrimary: Number(host?.dataset.brickRawPrimaryTriangles ?? 0),
      rawOpposite: Number(host?.dataset.brickRawOppositeTriangles ?? 0),
      mergedPrimary: Number(host?.dataset.brickMergedPrimaryTriangles ?? 0),
      mergedOpposite: Number(host?.dataset.brickMergedOppositeTriangles ?? 0),
      legends: document.querySelectorAll(".plot3d-discrete-legend__item").length,
      error: document.querySelector(".volume-render__overlay--error")?.textContent ?? null,
      workers: snapshot?.workers,
      inFlightJobs: snapshot?.ipc.inFlightJobs,
      workerResident: snapshot?.visualization.byStage["worker-resident"].residentBytes,
      reserved: snapshot?.visualization.reservedBytes,
    };
  })), { timeout: 120_000 }).toContain('"legends":2');

  const first = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(first).not.toBeNull();
  if (!first) return;
  const parameters = first.evidence.descriptor.parameters ?? {};
  expect(parameters.representation).toBe("progressive-native-bricks");
  expect(parameters.signedPair).toBe(true);
  expect(Number(parameters.residentBrickCount)).toBeGreaterThan(0);
  const visibleSurfaces = first.evidence.layers.filter((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0);
  expect(visibleSurfaces.length).toBeGreaterThanOrEqual(2);
  await expect(page.locator(".plot3d-discrete-legend__item")).toHaveCount(2);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(beforeEdit.visualization.byStage["decoded-field"].residentBytes).toBeLessThan(
    40 * 1024 * 1024,
  );
  await page.getByRole("button", { name: /Surface/ }).click();
  await page.locator('.plot3d-control-panel input[type="range"]').first()
    .evaluate((element: HTMLInputElement) => {
      const current = Number(element.value);
      element.value = String(Math.min(Number(element.max), current * 1.25));
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")
      ?.dataset.contourLoading === "true");
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false" && host?.dataset.contourLoading === "false";
  });
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);

  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === snapshot?.backgroundLoad.workers
      && snapshot?.preparedRegistry.handles === 0
      && snapshot?.visualization.residentBytes === 0
      && snapshot?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("Q criterion derives inside compatible resident u-v-w bricks", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await page.getByRole("button", { name: /Data/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("q_criterion");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.contourRepresentation === "progressive-bricks"
        && host.dataset.contourLoading === "false"
        && Number(host.dataset.brickRawPrimaryTriangles ?? 0) > 0
        && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
      contourReady: host?.dataset.contourReady,
      scalarEligible: host?.dataset.scalarBrickEligible,
      diagnosticEligible: host?.dataset.diagnosticBrickEligible,
      declaredBrickFields: host?.dataset.declaredBrickFields,
      threshold: host?.dataset.contourThreshold,
      committedThreshold: host?.dataset.contourCommittedThreshold,
      residentBricks: host?.dataset.brickResidentCount,
      rawTriangles: host?.dataset.brickRawPrimaryTriangles,
      mergedTriangles: host?.dataset.brickMergedPrimaryTriangles,
      brickError: host?.dataset.brickError,
      brickLoading: host?.dataset.brickLoading,
      brickPlanCount: host?.dataset.brickPlanCount,
      error: document.querySelector(".volume-render__overlay--error")?.textContent ?? null,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');

  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  if (!capture) return;
  const parameters = capture.evidence.descriptor.parameters ?? {};
  expect(capture.evidence.descriptor.field).toBe("q_criterion");
  expect(parameters.representation).toBe("progressive-native-bricks");
  expect(Number(parameters.selectedBrickCount)).toBe(8);
  expect(Number(parameters.residentBrickCount)).toBe(8);
  expect(Number((parameters.rawBrickTriangles as { primary?: number })?.primary)).toBeGreaterThan(0);
  expect(capture.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.getByRole("button", { name: /Surface/ }).click();
  await page.locator('.plot3d-control-panel input[type="range"]').first()
    .evaluate((element: HTMLInputElement) => {
      const current = Number(element.value);
      element.value = String(Math.min(Number(element.max), current * 1.2));
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")
      ?.dataset.contourLoading === "true");
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false" && host?.dataset.contourLoading === "false";
  });
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);

  for (const diagnosticField of ["lambda2_criterion", "swirling_strength"] as const) {
    const fieldSelect = page.locator('select[aria-label="Field"]');
    if (!await fieldSelect.isVisible()) await page.getByRole("button", { name: /Data/ }).click();
    await fieldSelect.selectOption(diagnosticField);
    await expect.poll(async () => JSON.stringify(await page.evaluate((expectedField) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      return {
        ready: scene?.dataset.frameLoading === "false"
          && host?.dataset.contourRepresentation === "progressive-bricks"
          && host.dataset.contourLoading === "false"
          && Number(host.dataset.brickRawPrimaryTriangles ?? 0) > 0
          && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
        field: document.querySelector<HTMLSelectElement>('select[aria-label="Field"]')?.value,
        expectedField,
        residentBricks: host?.dataset.brickResidentCount,
        rawTriangles: host?.dataset.brickRawPrimaryTriangles,
        mergedTriangles: host?.dataset.brickMergedPrimaryTriangles,
        brickError: host?.dataset.brickError,
      };
    }, diagnosticField)), { timeout: 120_000 }).toContain('"ready":true');
    const diagnosticCapture = await page.evaluate(() =>
      window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(diagnosticCapture?.evidence.descriptor.field).toBe(diagnosticField);
    expect(diagnosticCapture?.evidence.descriptor.parameters?.representation)
      .toBe("progressive-native-bricks");
    expect(Number(diagnosticCapture?.evidence.descriptor.parameters?.residentBrickCount)).toBe(8);
    expect(diagnosticCapture?.evidence.layers.some((layer) =>
      layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);
  }

  await page.evaluate(() => window.__nimbusVtkHarness?.setMounted(false));
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot ? {
      released: snapshot.workers === snapshot.backgroundLoad.workers
        && snapshot.visualization.residentBytes === 0
        && snapshot.visualization.reservedBytes === 0,
      workers: snapshot.workers,
      backgroundWorkers: snapshot.backgroundLoad.workers,
      visualization: snapshot.visualization,
    } : null;
  })), { timeout: 30_000 }).toContain('"released":true');
});

test("durable level-one velocity bricks bound the diagnostic working set", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await page.getByRole("button", { name: /Data/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("q_criterion");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.contourLoading === "false"
        && host.dataset.brickHierarchyLevel === "1"
        && Number(host.dataset.brickRawPrimaryTriangles ?? 0) > 0
        && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
      level: host?.dataset.brickHierarchyLevel,
      residentBricks: host?.dataset.brickResidentCount,
      error: host?.dataset.brickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');

  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  const parameters = capture!.evidence.descriptor.parameters ?? {};
  expect(parameters.brickHierarchyLevel).toBe(1);
  expect(Number(parameters.selectedBrickCount)).toBe(1);
  expect(Number(parameters.residentBrickCount)).toBe(1);
  expect(Number(parameters.brickSourceBytes)).toBeLessThan(200_000);
  expect(capture!.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.getByRole("button", { name: /Surface/ }).click();
  await page.locator('.plot3d-control-panel input[type="range"]').first()
    .evaluate((element: HTMLInputElement) => {
      element.value = String(Math.min(Number(element.max), Number(element.value) * 1.15));
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false" && host?.dataset.contourLoading === "false";
  });
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

test("durable diagnostic bricks replace generations atomically during playback", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await page.getByRole("button", { name: /Data/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("q_criterion");
  const waitForFrame = async (step: number) => expect.poll(async () => JSON.stringify(
    await page.evaluate((expectedStep) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const snapshot = window.__nimbusVtkHarness?.snapshot();
      return {
        ready: scene?.dataset.renderedStep === String(expectedStep)
          && scene.dataset.frameLoading === "false"
          && host?.dataset.contourLoading === "false"
          && host.dataset.brickHierarchyLevel === "1"
          && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0
          && snapshot?.ipc.inFlightJobs === 0
          && snapshot.visualization.reservedBytes === 0,
        renderedStep: scene?.dataset.renderedStep,
        level: host?.dataset.brickHierarchyLevel,
        residentBricks: host?.dataset.brickResidentCount,
        error: host?.dataset.brickError,
      };
    }, step),
  ), { timeout: 120_000 }).toContain('"ready":true');

  await waitForFrame(0);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(25));
  const stable = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  const stableWorkerBytes = stable.visualization.byStage["worker-resident"].residentBytes;
  for (let cycle = 0; cycle < 3; cycle += 1) {
    await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
    await page.waitForTimeout(5);
    await page.evaluate(() => window.__nimbusVtkHarness?.setStep(2));
    await waitForFrame(2);
    if (cycle === 0) {
      const presented = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      expect(presented?.evidence.descriptor.frameIndex).toBe(2);
      expect(presented?.evidence.descriptor.parameters?.renderedPublicationGeneration).toEqual(expect.any(String));
    }
    const newest = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(newest.visualization.byStage["worker-resident"].residentBytes).toBe(stableWorkerBytes);
    expect(newest.workers).toBe(stable.workers);
    await page.evaluate(() => window.__nimbusVtkHarness?.setStep(0));
    await waitForFrame(0);
    const returned = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(returned.visualization.byStage["worker-resident"].residentBytes).toBe(stableWorkerBytes);
    expect(returned.workers).toBe(stable.workers);
  }
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.field).toBe("q_criterion");
  expect(capture?.evidence.descriptor.parameters?.brickHierarchyLevel).toBe(1);

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

test("terrain-following scalar bricks contour exact physical centroids without seams", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?terrainBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.contourLoading === "false"
        && host.dataset.contourRepresentation === "progressive-bricks"
        && host.dataset.brickGeometry === "terrain-physical-centroids"
        && host.dataset.brickHierarchyLevel === "0"
        && Number(host.dataset.brickResidentCount ?? 0) > 1
        && Number(host.dataset.brickRawPrimaryTriangles ?? 0) > 0
        && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
      geometry: host?.dataset.brickGeometry,
      level: host?.dataset.brickHierarchyLevel,
      selected: host?.dataset.brickPlanCount,
      resident: host?.dataset.brickResidentCount,
      raw: host?.dataset.brickRawPrimaryTriangles,
      merged: host?.dataset.brickMergedPrimaryTriangles,
      error: host?.dataset.brickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');

  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  const parameters = capture!.evidence.descriptor.parameters ?? {};
  expect(parameters.brickGeometry).toBe("terrain-physical-centroids");
  expect(parameters.brickHierarchyLevel).toBe(0);
  expect(Number(parameters.selectedBrickCount)).toBeGreaterThan(1);
  expect(capture!.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.getByRole("button", { name: /Surface/ }).click();
  await page.locator('.plot3d-control-panel input[type="range"]').first()
    .evaluate((element: HTMLInputElement) => {
      element.value = String(Math.min(Number(element.max), Number(element.value) * 1.1));
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.contourLoading === "true");
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false" && host?.dataset.contourLoading === "false";
  });
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);

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

test("terrain Q criterion derives from native physical-centroid gradients", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?terrainBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await page.getByRole("button", { name: /Data/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("q_criterion");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.contourLoading === "false"
        && host.dataset.diagnosticBrickEligible === "true"
        && host.dataset.brickGeometry === "terrain-physical-centroids"
        && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
      eligible: host?.dataset.diagnosticBrickEligible,
      geometry: host?.dataset.brickGeometry,
      triangles: host?.dataset.brickMergedPrimaryTriangles,
      error: host?.dataset.brickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.field).toBe("q_criterion");
  expect(capture?.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0)).toBe(true);
  for (const diagnostic of ["lambda2_criterion", "swirling_strength"] as const) {
    const select = page.locator('select[aria-label="Field"]');
    if (!await select.isVisible()) await page.getByRole("button", { name: /Data/ }).click();
    await select.selectOption(diagnostic);
    await expect.poll(async () => JSON.stringify(await page.evaluate((field) => {
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      return {
        ready: document.querySelector<HTMLSelectElement>('select[aria-label="Field"]')?.value === field
          && host?.dataset.contourLoading === "false"
          && host.dataset.brickGeometry === "terrain-physical-centroids"
          && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
        field,
        triangles: host?.dataset.brickMergedPrimaryTriangles,
        error: host?.dataset.brickError,
      };
    }, diagnostic)), { timeout: 120_000 }).toContain('"ready":true');
  }
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

test("heavy terrain selects a durable physical-geometry LOD", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?terrainBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("isosurface"));
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.contourLoading === "false"
        && host.dataset.brickGeometry === "terrain-physical-centroids"
        && host.dataset.brickHierarchyLevel === "1"
        && Number(host.dataset.brickResidentCount ?? 0) === 1
        && Number(host.dataset.brickMergedPrimaryTriangles ?? 0) > 0,
      geometry: host?.dataset.brickGeometry,
      level: host?.dataset.brickHierarchyLevel,
      resident: host?.dataset.brickResidentCount,
      raw: host?.dataset.brickRawPrimaryTriangles,
      merged: host?.dataset.brickMergedPrimaryTriangles,
      error: host?.dataset.brickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');

  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  const parameters = capture!.evidence.descriptor.parameters ?? {};
  expect(parameters.brickGeometry).toBe("terrain-physical-centroids");
  expect(parameters.brickHierarchyLevel).toBe(1);
  expect(Number(parameters.selectedBrickCount)).toBe(1);
  expect(Number(parameters.brickSourceBytes)).toBeLessThan(2 * 1024 * 1024);
  expect(capture!.evidence.layers.some((layer) =>
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
