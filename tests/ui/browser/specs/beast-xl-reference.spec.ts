import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import type { VtkAppearanceImageMetrics } from "../../../../ui/src/lib/vtk/vtkAppearanceComparison";
import type {
  VtkAppearanceCameraEvidence,
  VtkAppearanceLayerEvidence,
} from "../../../../ui/src/lib/vtk/vtkAppearanceEvidence";

const runDir = process.env.NIMBUS_BEAST_XL_REFERENCE_RUN;
const referenceManifest = process.env.NIMBUS_BEAST_XL_REFERENCE_MANIFEST;
const referenceAblation = process.env.NIMBUS_VOLUME_ABLATION ?? "baseline";
const updateReference = process.env.NIMBUS_UPDATE_BEAST_REFERENCE === "1";
const screenDiagnostic = process.env.NIMBUS_VOLUME_SCREEN_DIAGNOSTIC === "1";
const interactionBenchmark = process.env.NIMBUS_VOLUME_INTERACTION_BENCHMARK === "1";
test.skip(!runDir, "set NIMBUS_BEAST_XL_REFERENCE_RUN to the preserved Beast XL run");

interface VisualSignal {
  meanLuma: number;
  lumaDeviation: number;
  edgeMean: number;
  structurePixelFraction: number;
  structureBounds: [number, number, number, number] | null;
}

test("Beast XL step 24 screen-space volume diagnostic", async ({ page }, testInfo) => {
  test.skip(!screenDiagnostic, "set NIMBUS_VOLUME_SCREEN_DIAGNOSTIC=1 for controlled screen evidence");
  test.setTimeout(10 * 60_000);
  if (!runDir) return;
  await page.goto(`/harness/index.html?realRunBridge=beast-xl&volumeAblation=${referenceAblation}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(3));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return scene?.dataset.framePlot === "volume"
      && scene.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === "24"
      && host?.dataset.volumeMaterialCommitted === "true"
      && host.dataset.volumeHierarchySource === "transient"
      && snapshot?.ipc.inFlightJobs === 0;
  }, undefined, { timeout: 8 * 60_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
  const opacity = page.locator('.volume-render__slider[type="range"]').first();
  await opacity.evaluate((element) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    setter?.call(input, "2.5");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForTimeout(500);
  const pngPath = testInfo.outputPath(`beast-xl-step24-${referenceAblation}.png`);
  await page.locator(".plot3d-overlay-host").screenshot({ path: pngPath });
  await testInfo.attach(`beast-xl-step24-${referenceAblation}`, {
    path: pngPath,
    contentType: "image/png",
  });
});

test("Beast XL transient-brick interaction benchmark", async ({ page }, testInfo) => {
  test.skip(!interactionBenchmark, "set NIMBUS_VOLUME_INTERACTION_BENCHMARK=1 for interaction timing");
  test.setTimeout(10 * 60_000);
  await page.goto(`/harness/index.html?realRunBridge=beast-xl&volumeAblation=${referenceAblation}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(3));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === "24"
      && host?.dataset.volumeMaterialCommitted === "true"
      && host.dataset.volumeHierarchySource === "transient";
  }, undefined, { timeout: 8 * 60_000 });
  const canvas = page.locator(".volume-render__canvas canvas");
  const bounds = await canvas.boundingBox();
  if (!bounds) throw new Error("Beast interaction benchmark has no canvas bounds");
  await page.mouse.move(bounds.x + bounds.width * 0.4, bounds.y + bounds.height * 0.48);
  await page.mouse.down();
  const moveDurationsMs: number[] = [];
  for (let step = 1; step <= 10; step += 1) {
    const started = performance.now();
    await page.mouse.move(
      bounds.x + bounds.width * (0.4 + step * 0.018),
      bounds.y + bounds.height * (0.48 + Math.sin(step * 0.7) * 0.025),
    );
    moveDurationsMs.push(performance.now() - started);
  }
  const releaseStarted = performance.now();
  await page.mouse.up();
  await expect(page.locator(".plot3d-overlay-host"))
    .toHaveAttribute("data-volume-quality-preview", "false", { timeout: 30_000 });
  const settleMs = performance.now() - releaseStarted;
  const sorted = [...moveDurationsMs].sort((left, right) => left - right);
  const evidence = {
    ablation: referenceAblation,
    moveDurationsMs,
    medianMoveMs: sorted[Math.floor(sorted.length / 2)],
    maximumMoveMs: Math.max(...moveDurationsMs),
    settleMs,
  };
  await testInfo.attach(`beast-interaction-${referenceAblation}`, {
    body: Buffer.from(`${JSON.stringify(evidence, null, 2)}\n`),
    contentType: "application/json",
  });
  console.log(`BEAST_INTERACTION ${JSON.stringify(evidence)}`);
});

async function visualSignal(page: Page, dataUri: string): Promise<VisualSignal> {
  return page.evaluate(async (source) => {
    const image = new Image();
    image.src = source;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Beast reference 2D evidence context unavailable");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const luma = new Float64Array(canvas.width * canvas.height);
    let sum = 0;
    let squares = 0;
    let structureCount = 0;
    let minX = canvas.width;
    let maxX = -1;
    let minY = canvas.height;
    let maxY = -1;
    for (let pixel = 0; pixel < luma.length; pixel += 1) {
      const offset = pixel * 4;
      const value = pixels[offset]! * 0.2126
        + pixels[offset + 1]! * 0.7152
        + pixels[offset + 2]! * 0.0722;
      luma[pixel] = value;
      sum += value;
      squares += value * value;
      const red = pixels[offset]!;
      const green = pixels[offset + 1]!;
      const blue = pixels[offset + 2]!;
      if (value < 80 || red <= blue * 1.25 || red <= green * 0.9) continue;
      structureCount += 1;
      const x = pixel % canvas.width;
      const y = Math.floor(pixel / canvas.width);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    let edgeSum = 0;
    let edgeCount = 0;
    for (let y = 0; y < canvas.height - 1; y += 1) {
      for (let x = 0; x < canvas.width - 1; x += 1) {
        const index = y * canvas.width + x;
        edgeSum += Math.abs(luma[index + 1]! - luma[index]!)
          + Math.abs(luma[index + canvas.width]! - luma[index]!);
        edgeCount += 2;
      }
    }
    const count = luma.length;
    const meanLuma = sum / count;
    return {
      meanLuma,
      lumaDeviation: Math.sqrt(Math.max(0, squares / count - meanLuma * meanLuma)),
      edgeMean: edgeSum / edgeCount,
      structurePixelFraction: structureCount / count,
      structureBounds: structureCount === 0 ? null : [
        minX / canvas.width,
        (maxX + 1) / canvas.width,
        minY / canvas.height,
        (maxY + 1) / canvas.height,
      ],
    };
  }, dataUri);
}

test("Beast XL step 23 vorticity commits uncropped native structure with repeatable evidence", async ({ page }, testInfo) => {
  test.setTimeout(10 * 60_000);
  if (!runDir) return;
  const stepDir = join(runDir, "step_000023");
  const manifestPath = join(stepDir, "manifest.json");
  const fieldPath = join(stepDir, "vorticity_magnitude.npy");
  expect(existsSync(manifestPath), "Beast reference manifest").toBe(true);
  expect(existsSync(fieldPath), "Beast reference field").toBe(true);
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
    step_index: number;
    simulation_time_s: number;
    grid: { nr: number; nth: number; nz: number; dr_m: number; dz_m: number };
  };
  expect(manifest.step_index).toBe(23);
  expect(manifest.simulation_time_s).toBeCloseTo(1380.21, 2);
  expect([manifest.grid.nr, manifest.grid.nth, manifest.grid.nz]).toEqual([512, 512, 160]);
  expect(manifest.grid.dr_m).toBe(100);
  expect(manifest.grid.dz_m).toBe(100);
  expect(statSync(fieldPath).size).toBeGreaterThan(160 * 1024 * 1024);
  expect(referenceManifest, "set NIMBUS_BEAST_XL_REFERENCE_MANIFEST to the explicit preserved baseline").toBeTruthy();
  const reference = JSON.parse(readFileSync(
    referenceManifest!,
    "utf8",
  )) as {
    representation: {
      contextShape: number[];
      contextStride: number[];
      detailShape: number[];
      detailIndexLower: number[];
      detailIndexUpper: number[];
      detailStride: number[];
    };
    transfer: { range: number[]; opacityPoints: Array<[number, number]> };
    renderer: { maximumCameraDistanceToFocusDiagonalRatio: number };
    candidateEvidence: Record<string, {
      pngSha256: string;
      metadataFingerprint: string;
      visualSignal?: VisualSignal;
    }>;
    visualTolerance?: {
      meanLuma: number;
      lumaDeviation: number;
      edgeMean: number;
      structurePixelFraction: number;
      structureBounds: number;
    };
    tolerance: { geometric: { status: string } };
  };
  expect(reference.tolerance.geometric.status).toBe("pass");

  await page.goto(`/harness/index.html?realRunBridge=beast-xl&volumeAblation=${referenceAblation}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return scene?.dataset.framePlot === "volume"
      && scene.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === "23"
      && (host?.dataset.volumeRepresentation === "hybrid"
        ? host.dataset.volumeContextReady === "true"
        : host?.dataset.volumeRepresentation === "full-domain"
          && host.dataset.volumeReducedContextOnly === "false")
      && host?.dataset.volumeMaterialCommitted === "true"
      && document.querySelector(".volume-render__quality-badge") == null
      && snapshot?.ipc.inFlightJobs === 0
      && snapshot?.workers === 0;
  }, undefined, { timeout: 8 * 60_000 });

  await expect(page.locator(".plot3d-badge__sub")).toContainText(/native|stride-1 ROI/);
  await expect(page.locator(".plot3d-colorbar__title")).toContainText("vorticity_magnitude");
  const first = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  const second = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(first).not.toBeNull();
  expect(second).not.toBeNull();
  if (!first || !second) return;
  expect(first.evidence.descriptor.datasetId).toBe("/__nimbus_browser_fixture__/beast-xl-reference");
  expect(first.evidence.descriptor.frameIndex).toBe(23);
  expect(first.evidence.descriptor.field).toBe("vorticity_magnitude");
  expect(first.evidence.metadataFingerprint).toBe(second.evidence.metadataFingerprint);
  const visibleVolumes = first.evidence.layers.filter((layer) =>
    layer.kind === "volume" && layer.visible
  );
  const parameters = first.evidence.descriptor.parameters ?? {};
  const detailShape = parameters.sampledShape;
  expect(Array.isArray(detailShape) && detailShape.length === 3).toBe(true);
  const detailCells = Array.isArray(detailShape)
    ? detailShape.reduce((product, count) => product * Number(count), 1)
    : 0;
  const nativeFullDomain = parameters.representation === "full-domain"
    && Array.isArray(parameters.stride)
    && parameters.stride.every((value: unknown) => value === 1);
  if (nativeFullDomain) {
    expect(visibleVolumes).toHaveLength(1);
    expect(detailShape).toEqual([512, 512, 160]);
    expect(detailCells).toBe(41_943_040);
    expect(parameters.detailIndexLower).toBeNull();
    expect(parameters.detailIndexUpper).toBeNull();
    expect(parameters.reducedContextOnly).toBe(false);
  } else {
    expect(visibleVolumes).toHaveLength(2);
    expect(visibleVolumes.some((layer) =>
      layer.pointCount === reference.representation.contextShape.reduce(
        (product, count) => product * count,
        1,
      ))).toBe(true);
    expect(detailShape).toEqual(reference.representation.detailShape);
    expect(parameters.detailIndexLower).toEqual(reference.representation.detailIndexLower);
    expect(parameters.detailIndexUpper).toEqual(reference.representation.detailIndexUpper);
    expect(parameters.overviewStride).toEqual(reference.representation.contextStride);
    expect(detailCells).toBeLessThanOrEqual(16_000_000);
    expect(parameters.stride).toEqual(reference.representation.detailStride);
    expect(
      Array.isArray(parameters.detailIndexUpper)
        && Number(parameters.detailIndexUpper[2]) >= 155,
      JSON.stringify(parameters),
    ).toBe(true);
  }
  expect(visibleVolumes.some((layer) => layer.pointCount === detailCells)).toBe(true);
  expect(Array.isArray(parameters.colorRange)).toBe(true);
  expect(Array.isArray(parameters.opacityPoints)).toBe(true);
  const supportBounds = parameters.cameraFocusBounds;
  expect(Array.isArray(supportBounds) && supportBounds.length === 6).toBe(true);
  const cameraPosition = first.evidence.camera.position;
  const cameraFocalPoint = first.evidence.camera.focalPoint;
  expect(cameraPosition).not.toBeNull();
  expect(cameraFocalPoint).not.toBeNull();
  if (Array.isArray(supportBounds) && supportBounds.length === 6
      && cameraPosition && cameraFocalPoint) {
    const expectedFocus = [
      0.5 * (Number(supportBounds[0]) + Number(supportBounds[1])),
      0.5 * (Number(supportBounds[2]) + Number(supportBounds[3])),
      0.5 * (Number(supportBounds[4]) + Number(supportBounds[5])),
    ];
    expectedFocus.forEach((coordinate, axis) =>
      expect(cameraFocalPoint[axis]).toBeCloseTo(coordinate, 6));
    const cameraDistance = Math.hypot(
      ...cameraPosition.map((coordinate, axis) =>
        coordinate - cameraFocalPoint[axis]),
    );
    const supportDiagonal = Math.hypot(
      Number(supportBounds[1]) - Number(supportBounds[0]),
      Number(supportBounds[3]) - Number(supportBounds[2]),
      Number(supportBounds[5]) - Number(supportBounds[4]),
    );
    expect(cameraDistance / supportDiagonal).toBeLessThanOrEqual(
      reference.renderer.maximumCameraDistanceToFocusDiagonalRatio,
    );
  }

  const prefix = "data:image/png;base64,";
  expect(first.pngDataUri.startsWith(prefix)).toBe(true);
  const png = Buffer.from(first.pngDataUri.slice(prefix.length), "base64");
  const pngSha256 = createHash("sha256").update(png).digest("hex");
  const signal = await visualSignal(page, first.pngDataUri);
  expect(pngSha256)
    .toBe(createHash("sha256").update(Buffer.from(second.pngDataUri.slice(prefix.length), "base64")).digest("hex"));
  const referenceLane = {
    "chromium-swiftshader": "chromiumSwiftShader",
    "chromium-lavapipe": "chromiumLavapipe",
    "webkit-standard": "webkitStandard",
    "webkit-retina": "webkitRetina",
  }[testInfo.project.name];
  expect(referenceLane).toBeTruthy();
  if (referenceLane && !updateReference) {
    const expected = reference.candidateEvidence[referenceLane];
    expect(first.evidence.metadataFingerprint).toBe(expected?.metadataFingerprint);
    expect(expected?.visualSignal).toBeDefined();
    expect(reference.visualTolerance).toBeDefined();
    if (expected?.visualSignal && reference.visualTolerance) {
      for (const metric of ["meanLuma", "lumaDeviation", "edgeMean", "structurePixelFraction"] as const) {
        expect(Math.abs(signal[metric] - expected.visualSignal[metric]), metric)
          .toBeLessThanOrEqual(reference.visualTolerance[metric]);
      }
      expect(signal.structureBounds).not.toBeNull();
      expect(expected.visualSignal.structureBounds).not.toBeNull();
      if (signal.structureBounds && expected.visualSignal.structureBounds) {
        signal.structureBounds.forEach((coordinate, index) =>
          expect(
            Math.abs(coordinate - expected.visualSignal!.structureBounds![index]),
            `structureBounds[${index}]`,
          ).toBeLessThanOrEqual(reference.visualTolerance!.structureBounds));
      }
    }
  }
  const pngPath = testInfo.outputPath("beast-xl-step23-vorticity.png");
  const evidencePath = testInfo.outputPath("beast-xl-step23-vorticity.appearance.json");
  const manifestOutputPath = testInfo.outputPath("beast-xl-step23-vorticity.reference.json");
  await Promise.all([
    writeFile(pngPath, png),
    writeFile(evidencePath, `${JSON.stringify(first.evidence, null, 2)}\n`, "utf8"),
    writeFile(
      manifestOutputPath,
      `${JSON.stringify({
        schema: "nimbus.studio.vtk-reference-candidate",
        schemaVersion: 1,
        runId: "1781996355874-beast_xl",
        stepIndex: 23,
        simulationTimeS: manifest.simulation_time_s,
        field: "vorticity_magnitude",
        sourceShape: [512, 512, 160],
        sourceSpacingM: [100, 100, 100],
        representation: `${String(parameters.representation)}:${referenceAblation}`,
        pngSha256,
        metadataFingerprint: first.evidence.metadataFingerprint,
        visualSignal: signal,
      }, null, 2)}\n`,
      "utf8",
    ),
  ]);
  await testInfo.attach("beast-xl-vorticity", { path: pngPath, contentType: "image/png" });
  await testInfo.attach("beast-xl-vorticity-evidence", {
    path: evidencePath,
    contentType: "application/json",
  });
  await testInfo.attach("beast-xl-vorticity-reference-candidate", {
    path: manifestOutputPath,
    contentType: "application/json",
  });
});

test("Beast XL broad-support frames never expose an unsupported native-detail box", async ({ page }, testInfo) => {
  test.setTimeout(12 * 60_000);
  if (!runDir) return;
  await page.goto("/harness/index.html?realRunBridge=beast-xl");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

  for (const frame of [
    { slot: 1, stepIndex: 2 },
    { slot: 2, stepIndex: 5 },
    { slot: 3, stepIndex: 24 },
  ]) {
    await page.evaluate((slot) => window.__nimbusVtkHarness?.setStep(slot), frame.slot);
    await page.waitForFunction((stepIndex) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const snapshot = window.__nimbusVtkHarness?.snapshot();
      return scene?.dataset.framePlot === "volume"
        && scene.dataset.frameLoading === "false"
        && scene.dataset.renderedStep === String(stepIndex)
        && host?.dataset.volumeMaterialCommitted === "true"
        && snapshot?.ipc.inFlightJobs === 0
        && snapshot?.workers === 0;
    }, frame.stepIndex, { timeout: 8 * 60_000 });

    // Let the frame-local locator either promote an admissible structural ROI
    // or settle on complete reduced context before collecting evidence.
    await page.waitForTimeout(750);
    const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(capture, `step ${frame.stepIndex} capture`).not.toBeNull();
    if (!capture) continue;
    expect(capture.evidence.descriptor.frameIndex).toBe(frame.stepIndex);
    const parameters = capture.evidence.descriptor.parameters ?? {};
    const representation = String(parameters.representation ?? "");
    if (representation === "hybrid-overview-native-detail") {
      const support = parameters.structuralSupportBounds as number[] | undefined;
      const lower = parameters.detailIndexLower as number[] | undefined;
      const upper = parameters.detailIndexUpper as number[] | undefined;
      expect(support).toHaveLength(6);
      expect(lower).toHaveLength(3);
      expect(upper).toHaveLength(3);
      if (support?.length === 6 && lower?.length === 3 && upper?.length === 3) {
        const origin = [-25_550, -25_550, 50];
        for (let axis = 0; axis < 3; axis += 1) {
          expect(origin[axis] + lower[axis] * 100).toBeLessThanOrEqual(support[axis * 2]);
          expect(origin[axis] + upper[axis] * 100).toBeGreaterThanOrEqual(support[axis * 2 + 1]);
        }
      }
    } else {
      expect(representation).toBe("full-domain");
      const stride = parameters.stride as number[] | undefined;
      expect(stride).toHaveLength(3);
      const native = stride?.every((value) => value === 1) === true;
      expect(parameters.reducedContextOnly).toBe(!native);
      expect(Number(parameters.effectiveOpacityScale))
        .toBeLessThanOrEqual(native ? 1 : 0.08);
    }
    const prefix = "data:image/png;base64,";
    expect(capture.pngDataUri.startsWith(prefix)).toBe(true);
    const pngPath = testInfo.outputPath(`beast-xl-step${frame.stepIndex}-vorticity.png`);
    await writeFile(pngPath, Buffer.from(capture.pngDataUri.slice(prefix.length), "base64"));
    await testInfo.attach(`beast-xl-step${frame.stepIndex}-vorticity`, {
      path: pngPath,
      contentType: "image/png",
    });
  }
});

test("legacy Beast XL volume keeps transient native bricks during camera interaction", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  if (!runDir) return;
  await page.goto("/harness/index.html?realRunBridge=beast-xl");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return scene?.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === "23"
      && host?.dataset.volumeRepresentation === "progressive-bricks"
      && host.dataset.volumeMaterialCommitted === "true"
      && Number(host.dataset.volumeBrickCount ?? 0) > 0
      && snapshot?.ipc.inFlightJobs === 0;
  }, undefined, { timeout: 8 * 60_000 });

  const host = page.locator(".plot3d-overlay-host");
  const settledBrickCount = Number(await host.getAttribute("data-volume-brick-count"));
  expect(settledBrickCount).toBeGreaterThan(0);
  const canvas = page.locator(".volume-render__canvas canvas");
  const bounds = await canvas.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move(bounds!.x + bounds!.width * 0.42, bounds!.y + bounds!.height * 0.5);
  await page.mouse.down();
  for (let step = 1; step <= 6; step += 1) {
    await page.mouse.move(
      bounds!.x + bounds!.width * (0.42 + 0.22 * step / 6),
      bounds!.y + bounds!.height * (0.5 + 0.08 * Math.sin(step * 0.7)),
    );
  }
  await expect(host).toHaveAttribute("data-volume-quality-preview", "true");
  await expect(host).not.toHaveAttribute("data-volume-interaction-proxy-active", "true");
  expect(Number(await host.getAttribute("data-volume-brick-count"))).toBe(settledBrickCount);
  await page.mouse.up();
  await expect(page.locator(".plot3d-control-notice")).toBeHidden({ timeout: 2_000 });
  await expect(host).toHaveAttribute("data-volume-quality-preview", "false");
  await expect(host).not.toHaveAttribute("data-volume-interaction-proxy-active", "true");
  expect(Number(await host.getAttribute("data-volume-brick-count"))).toBe(settledBrickCount);
  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setMounted(false);
    window.__nimbusVtkHarness?.clearFieldCache();
  });
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.visualization.residentBytes === 0
      && snapshot.visualization.reservedBytes === 0;
  }, undefined, { timeout: 30_000 });
});

test("legacy Beast XL frame loading retains settled native bricks without exposing the proxy", async ({ page }, testInfo) => {
  test.setTimeout(12 * 60_000);
  if (!runDir) return;
  await page.goto("/harness/index.html?realRunBridge=beast-xl");
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === "23"
      && Number(host?.dataset.volumeBrickCount ?? 0) > 0
      && host?.dataset.volumeInteractionProxyActive !== "true";
  }, undefined, { timeout: 8 * 60_000 });
  const host = page.locator(".plot3d-overlay-host");
  const retainedBrickCount = Number(await host.getAttribute("data-volume-brick-count"));
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
  await expect.poll(async () => page.locator('[data-testid="scene"]').getAttribute("data-frame-loading"), {
    timeout: 10_000,
  }).toBe("true");
  await expect(host).not.toHaveAttribute("data-volume-interaction-proxy-active", "true");
  expect(Number(await host.getAttribute("data-volume-brick-count"))).toBe(retainedBrickCount);
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const currentHost = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false"
      && scene.dataset.renderedStep === "2"
      && Number(currentHost?.dataset.volumeBrickCount ?? 0) > 0
      && currentHost?.dataset.volumeInteractionProxyActive !== "true"
      && currentHost?.dataset.volumeBrickError === "";
  }, undefined, { timeout: 8 * 60_000 });
  const screenshotPath = testInfo.outputPath("beast-xl-step2-retained-native-bricks.png");
  await host.screenshot({ path: screenshotPath });
  await testInfo.attach("beast-xl-step2-retained-native-bricks", {
    path: screenshotPath,
    contentType: "image/png",
  });
});

test("Beast XL vorticity one-factor appearance ablations emit comparable evidence", async ({ page }, testInfo) => {
  test.setTimeout(15 * 60_000);
  if (!runDir) return;
  const variants = [
    "baseline",
    "context-off",
    "context-018",
    "detail-075",
    "flat-detail",
    "opacity-250m",
    "no-native-detail",
    "overview-camera",
  ] as const;
  const captures: Array<{
    name: typeof variants[number];
    pngDataUri: string;
    evidence: NonNullable<Awaited<ReturnType<NonNullable<Window["__nimbusVtkHarness"]>["captureAppearance"]>>>["evidence"];
  }> = [];

  for (const name of variants) {
    await page.goto(`/harness/index.html?realRunBridge=beast-xl&volumeAblation=${name}`);
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    await page.waitForFunction((variant) => {
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      const snapshot = window.__nimbusVtkHarness?.snapshot();
      const expectedRepresentation = variant === "no-native-detail" ? "full-domain" : "hybrid";
      return scene?.dataset.framePlot === "volume"
        && scene.dataset.frameLoading === "false"
        && scene.dataset.renderedStep === "23"
        && host?.dataset.volumeRepresentation === expectedRepresentation
        && (expectedRepresentation !== "hybrid" || host.dataset.volumeContextReady === "true")
        && host?.dataset.volumeMaterialCommitted === "true"
        && document.querySelector(".volume-render__quality-badge") == null
        && snapshot?.ipc.inFlightJobs === 0
        && snapshot?.workers === 0;
    }, name, { timeout: 8 * 60_000 });
    const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(capture, `${name} capture`).not.toBeNull();
    if (!capture) continue;
    expect(capture.evidence.descriptor.parameters?.qualificationOverrides)
      .toEqual(expect.any(Object));
    captures.push({ name, ...capture });
  }

  const baseline = captures.find((capture) => capture.name === "baseline");
  expect(baseline).toBeDefined();
  if (!baseline) return;
  const matrix: Array<{
    name: typeof variants[number];
    pngSha256: string;
    metadataFingerprint: string;
    camera: VtkAppearanceCameraEvidence;
    layers: readonly VtkAppearanceLayerEvidence[];
    metricsVersusBaseline: VtkAppearanceImageMetrics | null;
  }> = [];
  for (const capture of captures) {
    const metrics = await page.evaluate(
      ([reference, candidate]) => window.__nimbusVtkHarness?.compareAppearance(
        reference,
        candidate,
      ) ?? null,
      [baseline.pngDataUri, capture.pngDataUri] as const,
    );
    expect(metrics).not.toBeNull();
    const prefix = "data:image/png;base64,";
    const png = Buffer.from(capture.pngDataUri.slice(prefix.length), "base64");
    const pngPath = testInfo.outputPath(`beast-xl-${capture.name}.png`);
    const evidencePath = testInfo.outputPath(`beast-xl-${capture.name}.appearance.json`);
    await writeFile(pngPath, png);
    await writeFile(evidencePath, `${JSON.stringify(capture.evidence, null, 2)}\n`, "utf8");
    await testInfo.attach(`beast-xl-${capture.name}`, { path: pngPath, contentType: "image/png" });
    matrix.push({
      name: capture.name,
      pngSha256: createHash("sha256").update(png).digest("hex"),
      metadataFingerprint: capture.evidence.metadataFingerprint,
      camera: capture.evidence.camera,
      layers: capture.evidence.layers,
      metricsVersusBaseline: metrics,
    });
  }
  const noNative = captures.find((capture) => capture.name === "no-native-detail");
  expect(noNative?.evidence.layers.filter((layer) => layer.kind === "volume" && layer.visible))
    .toHaveLength(1);
  const metricFor = (name: typeof variants[number]) => matrix.find((entry) => entry.name === name)
    ?.metricsVersusBaseline;
  expect(metricFor("context-018")?.rgbRmse).toBeGreaterThan(0.04);
  expect(metricFor("opacity-250m")?.edgeRmse).toBeGreaterThan(0.01);
  expect(metricFor("no-native-detail")?.edgeRmse).toBeGreaterThan(0.05);
  expect(metricFor("overview-camera")?.changedPixelFraction).toBeGreaterThan(0.15);
  const matrixPath = testInfo.outputPath("beast-xl-vorticity-ablation-matrix.json");
  await writeFile(matrixPath, `${JSON.stringify({
    schema: "nimbus.studio.vtk-appearance-ablation",
    schemaVersion: 1,
    runId: "1781996355874-beast_xl",
    stepIndex: 23,
    field: "vorticity_magnitude",
    baseline: "hybrid-overview-native-detail",
    variants: matrix,
  }, null, 2)}\n`, "utf8");
  await testInfo.attach("beast-xl-vorticity-ablation-matrix", {
    path: matrixPath,
    contentType: "application/json",
  });
});
