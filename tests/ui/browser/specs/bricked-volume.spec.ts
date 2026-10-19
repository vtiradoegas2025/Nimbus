import { expect, test } from "@playwright/test";

test("volume hierarchy skips transparent bricks and reuses prepared GPU residency", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?large=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("cloud_condensate");
  await expect.poll(async () => JSON.stringify(await page.evaluate(async () => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.volumeRepresentation === "progressive-bricks"
        && host.dataset.volumeMaterialCommitted === "true"
        && Number(host.dataset.volumeBrickCount ?? 0) > 0
        && (Number(host.dataset.volumeTransparentBricks ?? 0) > 0
          || Number(host.dataset.volumeBoundaryOnlyBricks ?? 0) > 0),
      representation: host?.dataset.volumeRepresentation,
      level: host?.dataset.volumeBrickLevel,
      bricks: host?.dataset.volumeBrickCount,
      transparent: host?.dataset.volumeTransparentBricks,
      material: host?.dataset.volumeMaterialCommitted,
      error: host?.dataset.volumeBrickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');

  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  const parameters = capture!.evidence.descriptor.parameters ?? {};
  expect(parameters.representation).toBe("progressive-native-bricks");
  expect(Number(parameters.selectedBrickCount)).toBeGreaterThan(0);
  expect(Number(parameters.transparentBrickCount) + Number(parameters.boundaryOnlyBrickCount ?? 0)).toBeGreaterThan(0);
  expect(Number(parameters.logicalGpuUploadBytes)).toBeLessThan(256 * 256 * 160 * 4);
  expect(capture!.evidence.layers.filter((layer) =>
    layer.kind === "volume" && layer.visible).length).toBeGreaterThan(0);

  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.locator('.plot3d-control-panel input[type="range"]').first()
    .evaluate((element: HTMLInputElement) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(
        element, String(Math.min(Number(element.max), Number(element.value) * 1.2)),
      );
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForTimeout(50);
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")
      ?.dataset.volumeMaterialCommitted === "true");
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);
  expect(afterEdit.texture3DUploads).toBe(beforeEdit.texture3DUploads);
  expect(afterEdit.visualization.byStage.gpu.residentBytes).toBeGreaterThan(0);
  const densityCapture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(Number(densityCapture!.evidence.descriptor.parameters?.opacityScale))
    .toBeGreaterThan(Number(capture!.evidence.descriptor.parameters?.opacityScale));
  await page.getByRole("button", { name: "Viridis", exact: true }).click();
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")
    ?.dataset.volumeMaterialCommitted === "true");
  const paletteCapture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(paletteCapture!.evidence.descriptor.parameters?.colormap).toBe("Viridis");
  expect(paletteCapture!.evidence.descriptor.parameters?.scalarCompositionFingerprint)
    .toBe(capture!.evidence.descriptor.parameters?.scalarCompositionFingerprint);
  const afterPalette = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterPalette.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);
  expect(afterPalette.texture3DUploads).toBe(beforeEdit.texture3DUploads);

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

test("volume hierarchy selects a durable coarse level under a constrained pane budget", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("dense_signed");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.volumeRepresentation === "progressive-bricks"
        && host.dataset.volumeMaterialCommitted === "true"
        && host.dataset.volumeBrickLevel === "1"
        && Number(host.dataset.volumeBrickCount ?? 0) > 0,
      representation: host?.dataset.volumeRepresentation,
      level: host?.dataset.volumeBrickLevel,
      bricks: host?.dataset.volumeBrickCount,
      transparent: host?.dataset.volumeTransparentBricks,
      error: host?.dataset.volumeBrickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.representation)
    .toBe("progressive-native-bricks");
  expect(capture?.evidence.descriptor.parameters?.brickHierarchyLevel).toBe(1);
  const snapshot = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(snapshot.visualization.byStage.gpu.residentBytes).toBeGreaterThan(0);

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const state = window.__nimbusVtkHarness?.snapshot();
    return state ? {
      released: state.workers === state.backgroundLoad.workers
        && state.visualization.residentBytes === 0
        && state.visualization.reservedBytes === 0,
      workers: state.workers,
      workerKinds: state.workerKinds,
      backgroundWorkers: state.backgroundLoad.workers,
      visualization: state.visualization,
    } : null;
  })), { timeout: 15_000 }).toContain('"released":true');
});

test("terrain-following volume bricks compose physical-volume remaps", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?terrainBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("w");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.volumeRepresentation === "progressive-bricks"
        && host.dataset.volumeMaterialCommitted === "true"
        && Number(host.dataset.volumeBrickCount ?? 0) > 0,
      representation: host?.dataset.volumeRepresentation,
      level: host?.dataset.volumeBrickLevel,
      bricks: host?.dataset.volumeBrickCount,
      error: host?.dataset.volumeBrickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture?.evidence.descriptor.parameters?.representation)
    .toBe("progressive-native-bricks");
  expect(capture?.evidence.layers.filter((layer) =>
    layer.kind === "volume" && layer.visible).length).toBeGreaterThan(0);

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

test("heavy terrain volume selects durable physical geometry with scalar-only GPU accounting", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?terrainBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("w");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.volumeMaterialCommitted === "true"
        && host.dataset.volumeBrickLevel === "1"
        && Number(host.dataset.volumeBrickCount ?? 0) > 0,
      level: host?.dataset.volumeBrickLevel,
      error: host?.dataset.volumeBrickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');
  const snapshot = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(snapshot.visualization.byStage.gpu.residentBytes).toBeGreaterThan(0);
  expect(snapshot.visualization.byStage.gpu.residentBytes).toBeLessThanOrEqual(250_000 * 4);

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await page.waitForFunction(() => {
    const state = window.__nimbusVtkHarness?.snapshot();
    return state?.workers === state?.backgroundLoad.workers
      && state?.visualization.residentBytes === 0
      && state?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("global anomaly brick batches reuse material state and reject superseded frames", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("temperature");
  const waitForFrame = async (step: number) => expect.poll(async () => JSON.stringify(
    await page.evaluate((expectedStep) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const capture = window.__nimbusVtkHarness?.snapshot();
      return {
        ready: scene?.dataset.renderedStep === String(expectedStep)
          && scene.dataset.frameLoading === "false"
          && host?.dataset.volumeRepresentation === "progressive-bricks"
          && host.dataset.volumeMaterialCommitted === "true"
          && capture?.ipc.inFlightJobs === 0,
        rendered: scene?.dataset.renderedStep,
        error: host?.dataset.volumeBrickError,
      };
    }, step),
  ), { timeout: 120_000 }).toContain('"ready":true');
  await waitForFrame(0);
  const beforeEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.locator('.plot3d-control-panel input[type="range"]').first()
    .evaluate((element: HTMLInputElement) => {
      element.value = String(Math.min(Number(element.max), Number(element.value) * 1.1));
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  await page.waitForFunction(() =>
    document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.volumeMaterialCommitted === "true");
  const afterEdit = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(afterEdit.ipc.startedJobs).toBe(beforeEdit.ipc.startedJobs);

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
    const state = window.__nimbusVtkHarness?.snapshot();
    return state?.workers === state?.backgroundLoad.workers
      && state?.visualization.residentBytes === 0
      && state?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("camera interaction retains one volume generation and shaded lighting", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("dense_signed");
  const waitForLevel = async (level: number) => expect.poll(async () => JSON.stringify(
    await page.evaluate((expectedLevel) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      return {
        ready: scene?.dataset.frameLoading === "false"
          && host?.dataset.volumeMaterialCommitted === "true"
          && host.dataset.volumeBrickLevel === String(expectedLevel),
        level: host?.dataset.volumeBrickLevel,
        error: host?.dataset.volumeBrickError,
      };
    }, level),
  ), { timeout: 120_000 }).toContain('"ready":true');
  await waitForLevel(0);
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-volume-quality-preview", "false");
  const fine = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  const settledLighting = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeLighting());
  expect(settledLighting.length).toBeGreaterThan(0);
  expect(settledLighting.every(layer => layer.shade && !layer.flatPreview)).toBe(true);
  const uploadsBeforeCamera = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(20));
  const canvas = page.locator(".volume-render__canvas canvas");
  const bounds = await canvas.boundingBox();
  if (!bounds) throw new Error("Volume canvas bounds unavailable");
  await page.mouse.move(bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 0.58, bounds.y + bounds.height * 0.54, { steps: 4 });
  await waitForLevel(0);
  const host = page.locator(".plot3d-overlay-host");
  await expect(host).toHaveAttribute("data-volume-quality-preview", "true");
  const preview = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(preview?.evidence.layers.filter((layer) =>
    layer.kind === "volume" && layer.visible).every((layer) => layer.shade === true)).toBe(true);
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeLighting())).toEqual(settledLighting);
  expect(Number(await host.getAttribute("data-volume-lod-retained-transitions"))).toBe(0);
  await page.mouse.up();
  await waitForLevel(0);
  await expect(host).toHaveAttribute("data-volume-quality-preview", "false");
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(0));
  const refined = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(refined?.evidence.descriptor.parameters?.interactionQuality).toBe("settled");
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads))
    .toBe(uploadsBeforeCamera);
  expect(refined?.evidence.layers.some((layer) =>
    layer.kind === "volume" && layer.visible && layer.shade === true)).toBe(true);
  expect(refined?.evidence.descriptor.parameters?.scalarCompositionFingerprint)
    .toBe(fine?.evidence.descriptor.parameters?.scalarCompositionFingerprint);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(50));
  for (let cycle = 0; cycle < 3; cycle += 1) {
    await page.mouse.down();
    await page.mouse.move(
      bounds.x + bounds.width * (0.54 + cycle * 0.01),
      bounds.y + bounds.height * 0.52,
      { steps: 2 },
    );
    await page.waitForTimeout(5);
    await page.mouse.up();
  }
  await waitForLevel(0);
  const settled = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(settled.ipc.inFlightJobs).toBe(0);
  expect(settled.visualization.reservedBytes).toBe(0);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(0));

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await page.waitForFunction(() => {
    const state = window.__nimbusVtkHarness?.snapshot();
    return state?.workers === state?.backgroundLoad.workers
      && state?.visualization.residentBytes === 0
      && state?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("nested temperature volumes fit the transformed anomaly range", async ({ page }) => {
  await page.goto("/harness/index.html?derivedBricks=1&nestedVolumes=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("temperature");
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false"
    && document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.volumeNestedLayers === "2"
    && document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.volumeMaterialCommitted === "true");
  const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const parameters = capture!.evidence.descriptor.parameters!;
  expect(parameters.scalarTransform).toBe("horizontal-level-anomaly");
  const range = parameters.colorRange as number[];
  expect(range[0]).toBeLessThan(0);
  expect(range[1]).toBeGreaterThan(0);
  expect(Math.max(...range.map(Math.abs))).toBeLessThan(100);
});

test("nested parent and moving child volumes commit as one masked generation", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&nestedVolumes=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("dense_signed");
  const waitForNested = async (step: number) => expect.poll(async () => JSON.stringify(
    await page.evaluate((expectedStep) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      return {
        ready: scene?.dataset.renderedStep === String(expectedStep)
          && scene.dataset.frameLoading === "false"
          && host?.dataset.volumeMaterialCommitted === "true"
          && host.dataset.volumeNestedLayers === "2"
          && Number(host.dataset.volumeParentBrickCount ?? 0) > 0
          && Number(host.dataset.volumeBrickCount ?? 0)
            > Number(host.dataset.volumeParentBrickCount ?? 0),
        rendered: scene?.dataset.renderedStep,
        layers: host?.dataset.volumeNestedLayers,
        bricks: host?.dataset.volumeBrickCount,
        parent: host?.dataset.volumeParentBrickCount,
        bounds: host?.dataset.volumeChildBoundsLower,
        error: host?.dataset.volumeBrickError,
      };
    }, step),
  ), { timeout: 120_000 }).toContain('"ready":true');
  await waitForNested(0);
  const host = page.locator(".plot3d-overlay-host");
  const firstBounds = await host.getAttribute("data-volume-child-bounds-lower");
  const firstCapture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(firstCapture?.evidence.layers.filter((layer) =>
    layer.kind === "volume" && layer.visible).length).toBeGreaterThan(8);

  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(20));
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
  await waitForNested(1);
  expect(await host.getAttribute("data-volume-child-bounds-lower")).not.toBe(firstBounds);

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setFieldReadDelay(0);
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const state = window.__nimbusVtkHarness?.snapshot();
    return state ? {
      released: state.workers === state.backgroundLoad.workers
        && state.visualization.residentBytes === 0
        && state.visualization.reservedBytes === 0,
      workers: state.workers,
      backgroundWorkers: state.backgroundLoad.workers,
      workerKinds: state.workerKinds,
      visualization: state.visualization,
    } : null;
  })), { timeout: 60_000 }).toContain('"released":true');
});

test("arbitrary-depth nested volume chain commits root child and grandchild together", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&nestedVolumes=2");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("dense_signed");
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return {
      ready: scene?.dataset.frameLoading === "false"
        && host?.dataset.volumeMaterialCommitted === "true"
        && host.dataset.volumeNestedLayers === "3"
        && Number(host.dataset.volumeBrickCount ?? 0) > 16,
      layers: host?.dataset.volumeNestedLayers,
      bricks: host?.dataset.volumeBrickCount,
      error: host?.dataset.volumeBrickError,
    };
  })), { timeout: 120_000 }).toContain('"ready":true');
  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await expect.poll(async () => JSON.stringify(await page.evaluate(() => {
    const state = window.__nimbusVtkHarness?.snapshot();
    return state ? {
      released: state.workers === state.backgroundLoad.workers
        && state.visualization.residentBytes === 0
        && state.visualization.reservedBytes === 0,
    } : null;
  })), { timeout: 60_000 }).toContain('"released":true');
});

test("progressive volume bricks replace playback generations atomically", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  await page.goto("/harness/index.html?derivedBricks=1&brickLod=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("dense_signed");
  const waitForFrame = async (step: number) => expect.poll(async () => JSON.stringify(
    await page.evaluate((expectedStep) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const snapshot = window.__nimbusVtkHarness?.snapshot();
      return {
        ready: scene?.dataset.renderedStep === String(expectedStep)
          && scene.dataset.frameLoading === "false"
          && host?.dataset.volumeRepresentation === "progressive-bricks"
          && host.dataset.volumeMaterialCommitted === "true"
          && host.dataset.volumeBrickLevel === "1"
          && snapshot?.ipc.inFlightJobs === 0
          && snapshot.visualization.reservedBytes === 0,
        rendered: scene?.dataset.renderedStep,
        level: host?.dataset.volumeBrickLevel,
        bricks: host?.dataset.volumeBrickCount,
        error: host?.dataset.volumeBrickError,
      };
    }, step),
  ), { timeout: 120_000 }).toContain('"ready":true');

  await waitForFrame(0);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(20));
  const baseline = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  const gpuBytes = baseline.visualization.byStage.gpu.residentBytes;
  for (let cycle = 0; cycle < 2; cycle += 1) {
    await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
    await page.waitForTimeout(5);
    await page.evaluate(() => window.__nimbusVtkHarness?.setStep(2));
    await waitForFrame(2);
    const newest = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(newest.visualization.byStage.gpu.residentBytes).toBe(gpuBytes);
    expect(newest.workers).toBe(baseline.workers);
    await page.evaluate(() => window.__nimbusVtkHarness?.setStep(0));
    await waitForFrame(0);
  }

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setFieldReadDelay(0);
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await page.waitForFunction(() => {
    const state = window.__nimbusVtkHarness?.snapshot();
    return state?.workers === state?.backgroundLoad.workers
      && state?.visualization.residentBytes === 0
      && state?.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("nested ownership retains raw parent samples and shares clipped texture inputs", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/harness/index.html?derivedBricks=1&nestedVolumes=2");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.locator('select[aria-label="Field"]').selectOption("temperature");
  await page.getByRole("button", { name: "Raw field", exact: true }).click();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  for (const step of [0, 1]) {
    if (step) await page.evaluate((index) => window.__nimbusVtkHarness!.setStep(index), step);
    await page.waitForFunction((index) => {
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      return scene?.dataset.renderedStep === String(index) && scene.dataset.frameLoading === "false"
        && host?.dataset.volumeMaterialCommitted === "true" && host.dataset.volumeQualityPreview === "false"
        && host.dataset.volumeNestedLayers === "3";
    }, step);
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    expect(capture?.evidence.descriptor.parameters?.scalarTransform).toBe("none");
    expect(capture?.evidence.descriptor.parameters?.volumeOrdering).toBe("spatial-partition");
    const inputs = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
    expect(inputs.minimum).toBeGreaterThan(200);
    expect(inputs.zeros).toBe(0);
    expect(inputs.volumes).toBeGreaterThan(inputs.images);
    expect(capture?.evidence.descriptor.parameters?.logicalGpuUploadBytes).toBe(inputs.bytes);
    const uploads = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads);
    await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([18000, -12000, 9000], [2900, 3000, 1000], false));
    await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([-18000, 12000, 9000], [2900, 3000, 1000], false));
    expect(await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads)).toBe(uploads);
  }
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await page.waitForFunction(() => window.__nimbusVtkHarness!.snapshot().visualization.residentBytes === 0);
  expect(errors).toEqual([]);
});
