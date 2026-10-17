import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";
import type { HarnessPlotType } from "../harness/main";
import { ANALYTIC_FIDELITY_FIELD_NAMES, createAnalyticFidelityFixture } from "../../../../ui/src/lib/vtk/analyticFidelityFixtures";

const APPEARANCE_PLOTS: readonly HarnessPlotType[] = [
  "volume",
  "isosurface",
  "streamlines",
  "supercell",
  "trajectories",
];

async function settle(page: Page, plot: HarnessPlotType, timeout = 120_000): Promise<void> {
  await page.evaluate((value) => window.__nimbusVtkHarness?.setPlot(value), plot);
  await page.waitForFunction((expected) =>
    document.querySelector(".harness-shell")?.getAttribute("data-plot") === expected,
  plot);
  if (plot === "trajectories") {
    await page.waitForFunction(() =>
      /^\d+ parcel paths/.test(document.querySelector(".volume-render__overlay--muted")?.textContent ?? "")
      && !document.querySelector(".volume-render__overlay:not(.volume-render__overlay--muted)")
      && document.querySelector(".volume-render__canvas canvas") != null,
    );
    return;
  }
  await page.waitForFunction((expected) => {
    const harness = window.__nimbusVtkHarness;
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    if (!harness || !scene) return false;
    if (document.querySelector(".volume-render__overlay--error")) return true;
    const resources = harness.snapshot();
    return scene.dataset.framePlot === expected
      && scene.dataset.frameLoading === "false"
      && document.querySelector(".volume-render__quality-badge") == null
      && document.querySelector(".plot3d-control-notice.is-visible") == null
      && (expected !== "volume"
        || document.querySelector(".plot3d-overlay-host")
          ?.getAttribute("data-volume-material-committed") === "true")
      && resources.ipc.inFlightJobs === 0;
  }, plot, { timeout });
  expect(await page.locator(".volume-render__overlay--error").allTextContents()).toEqual([]);
}

function pngBytes(dataUri: string): Buffer {
  const prefix = "data:image/png;base64,";
  if (!dataUri.startsWith(prefix)) throw new Error("appearance capture returned a non-PNG payload");
  return Buffer.from(dataUri.slice(prefix.length), "base64");
}

function differencePaths(first: unknown, second: unknown, path = "root"): string[] {
  if (Object.is(first, second)) return [];
  if (first == null || second == null || typeof first !== "object" || typeof second !== "object") {
    return [`${path}: ${JSON.stringify(first)} != ${JSON.stringify(second)}`];
  }
  const firstRecord = first as Record<string, unknown>;
  const secondRecord = second as Record<string, unknown>;
  const keys = new Set([...Object.keys(firstRecord), ...Object.keys(secondRecord)]);
  return [...keys].flatMap((key) => differencePaths(firstRecord[key], secondRecord[key], `${path}.${key}`));
}

test("all native 3D renderers produce deterministic PNG and scene evidence", async ({ page }, testInfo) => {
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setSize(960, 620));

  for (const plot of APPEARANCE_PLOTS) {
    await settle(page, plot);
    const first = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    const second = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(first, `${plot} appearance capture`).not.toBeNull();
    expect(second, `${plot} repeat appearance capture`).not.toBeNull();
    if (!first || !second) continue;

    expect(first.evidence.schema).toBe("nimbus.studio.vtk-appearance");
    expect(first.evidence.schemaVersion).toBe(1);
    expect(first.evidence.descriptor.plot).toBe(plot);
    expect(first.evidence.viewport.cssWidth).toBeGreaterThan(0);
    expect(first.evidence.viewport.cssHeight).toBeGreaterThan(0);
    expect(first.evidence.viewport.backingWidth).toBeGreaterThan(0);
    expect(first.evidence.viewport.backingHeight).toBeGreaterThan(0);
    expect(first.evidence.camera.position).not.toBeNull();
    expect(first.evidence.camera.focalPoint).not.toBeNull();
    expect(first.evidence.layers.length).toBeGreaterThan(0);
    if (plot === "streamlines") {
      const paths = first.evidence.layers.filter((layer) => layer.kind === "actor").slice(0, 2);
      expect(paths).toHaveLength(2);
      for (const path of paths) {
        expect(path.visible).toBe(true);
        expect(path.scalarVisibility, "magnitude-colored paths must use the displayed speed ramp").toBe(true);
        expect(path.pointCount).toBeGreaterThan(0);
        expect(path.cellCount).toBeGreaterThan(0);
      }
    }
    expect(
      first.evidence.metadataFingerprint,
      differencePaths(first.evidence, second.evidence).join("\n"),
    ).toBe(second.evidence.metadataFingerprint);

    const firstPng = pngBytes(first.pngDataUri);
    const secondPng = pngBytes(second.pngDataUri);
    expect(firstPng.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
    expect(createHash("sha256").update(firstPng).digest("hex"))
      .toBe(createHash("sha256").update(secondPng).digest("hex"));

    const pngPath = testInfo.outputPath(`${plot}.png`);
    const evidencePath = testInfo.outputPath(`${plot}.appearance.json`);
    await writeFile(pngPath, firstPng);
    await writeFile(evidencePath, `${JSON.stringify(first.evidence, null, 2)}\n`, "utf8");
    await testInfo.attach(`${plot}-appearance`, { path: pngPath, contentType: "image/png" });
    await testInfo.attach(`${plot}-evidence`, { path: evidencePath, contentType: "application/json" });
  }
});

test("analytic volume fixtures capture the transfer and ray state that governs fidelity", async ({ page }, testInfo) => {
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setSize(960, 620));
  await settle(page, "volume");
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
  const fieldSelect = page.locator(".control-group").filter({
    has: page.locator(".control-group__label", { hasText: "Field" }),
  }).locator("select").first();

  for (const fieldName of ANALYTIC_FIDELITY_FIELD_NAMES) {
    await fieldSelect.selectOption(fieldName);
    if (fieldName === "analytic_smooth_gradient") {
      await page.getByRole("button", { name: "Raw field", exact: true }).click();
    }
    await expect(page.locator(".plot3d-colorbar__title")).toContainText(fieldName);
    await page.waitForFunction((expected) => {
      const harness = window.__nimbusVtkHarness;
      return harness?.snapshot().ipc.inFlightJobs === 0
        && document.querySelector(".volume-render__quality-badge") == null
        && document.querySelector(".plot3d-control-notice.is-visible") == null
        && document.querySelector(".volume-render__overlay--error") == null
        && document.querySelector(".plot3d-overlay-host")
          ?.getAttribute("data-volume-material-committed") === "true"
        && document.querySelector(".plot3d-colorbar__title")?.textContent?.includes(expected);
    }, fieldName, { timeout: 120_000 });

    const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(capture, `${fieldName} appearance capture`).not.toBeNull();
    if (!capture) continue;
    expect(capture.evidence.descriptor.field).toBe(fieldName);
    expect(capture.evidence.descriptor.parameters?.interactionQuality).toBe("settled");
    if (fieldName === "analytic_smooth_gradient") {
      expect(capture.evidence.descriptor.parameters?.scalarTransform).toBe("none");
    }
    const visibleVolumes = capture.evidence.layers.filter((layer) =>
      layer.kind === "volume" && layer.visible
    );
    expect(visibleVolumes.length, `${fieldName} visible volume layers`).toBeGreaterThan(0);
    expect(visibleVolumes.every((layer) => (layer.sampleDistance ?? 0) > 0)).toBe(true);
    expect(visibleVolumes.every((layer) => (layer.maximumSamplesPerRay ?? 0) > 0)).toBe(true);
    const expectedFlatLighting = fieldName === "analytic_smooth_gradient"
      || fieldName === "analytic_signed_couplet"
      || fieldName === "analytic_stratified_anomaly";
    expect(expectedFlatLighting
      ? visibleVolumes.every((layer) => layer.shade === false)
      : visibleVolumes.some((layer) => layer.shade === true)).toBe(true);
    expect(capture.evidence.descriptor.parameters?.normalBasis)
      .toBe(expectedFlatLighting ? "unlit" : "scalar-gradient");
    if (fieldName === "analytic_helical_tube") {
      const colorRange = capture.evidence.descriptor.parameters?.colorRange;
      const opacityMaximum = capture.evidence.descriptor.parameters?.opacityMaximum;
      expect(Array.isArray(colorRange)).toBe(true);
      expect(typeof opacityMaximum).toBe("number");
      if (Array.isArray(colorRange) && typeof opacityMaximum === "number") {
        expect(opacityMaximum).toBeGreaterThan(Number(colorRange[1]));
      }
      expect(visibleVolumes.some((layer) =>
        layer.shade === true && layer.ambient === 0.6 && layer.diffuse === 0.4
      )).toBe(true);
    }

    const pngPath = testInfo.outputPath(`${fieldName}.png`);
    const evidencePath = testInfo.outputPath(`${fieldName}.appearance.json`);
    await writeFile(pngPath, pngBytes(capture.pngDataUri));
    await writeFile(evidencePath, `${JSON.stringify(capture.evidence, null, 2)}\n`, "utf8");
    await testInfo.attach(fieldName, { path: pngPath, contentType: "image/png" });
    await testInfo.attach(`${fieldName}-evidence`, {
      path: evidencePath,
      contentType: "application/json",
    });
  }
});

test("all native 3D renderers settle and capture under concurrent worker load", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => {
    window.__nimbusVtkHarness?.setSize(960, 620);
    window.__nimbusVtkHarness?.startBackgroundLoad(2);
  });
  await page.waitForFunction(() => {
    const load = window.__nimbusVtkHarness?.snapshot().backgroundLoad;
    return load?.workers === 2 && load.iterations >= 8;
  });

  const evidence: Array<{
    plot: HarnessPlotType;
    fingerprint: string;
    pngSha256: string;
    workers: number;
    iterations: number;
    maxLongTaskMs: number;
  }> = [];
  for (const plot of APPEARANCE_PLOTS) {
    await settle(page, plot);
    await expect(page.locator(".volume-render__quality-badge")).toHaveCount(0);
    const first = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    const second = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(first, `${plot} loaded capture`).not.toBeNull();
    expect(second, `${plot} repeat loaded capture`).not.toBeNull();
    if (!first || !second) continue;
    expect(first.evidence.metadataFingerprint).toBe(second.evidence.metadataFingerprint);
    const firstPng = pngBytes(first.pngDataUri);
    const firstHash = createHash("sha256").update(firstPng).digest("hex");
    expect(firstHash).toBe(createHash("sha256").update(pngBytes(second.pngDataUri)).digest("hex"));
    const snapshot = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(snapshot.backgroundLoad.workers).toBe(2);
    expect(snapshot.workers).toBeGreaterThanOrEqual(2);
    evidence.push({
      plot,
      fingerprint: first.evidence.metadataFingerprint,
      pngSha256: firstHash,
      workers: snapshot.workers,
      iterations: snapshot.backgroundLoad.iterations,
      maxLongTaskMs: snapshot.maxLongTaskMs,
    });
  }
  expect(evidence).toHaveLength(APPEARANCE_PLOTS.length);
  expect(evidence.at(-1)!.iterations).toBeGreaterThan(evidence[0]!.iterations);
  await page.evaluate(() => {
    window.__nimbusVtkHarness?.stopBackgroundLoad();
    window.__nimbusVtkHarness?.setMounted(false);
  });
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.backgroundLoad.workers === 0 && snapshot.workers === 0;
  }, undefined, { timeout: 45_000 });
  expect(pageErrors).toEqual([]);

  const evidencePath = testInfo.outputPath("concurrent-background-load.appearance.json");
  await writeFile(evidencePath, `${JSON.stringify({
    schema: "nimbus.studio.vtk-concurrent-appearance",
    schemaVersion: 1,
    backgroundWorkers: 2,
    renderers: evidence,
  }, null, 2)}\n`, "utf8");
  await testInfo.attach("concurrent-background-load-appearance", {
    path: evidencePath,
    contentType: "application/json",
  });
});

test("split 3D panes remain deterministic under concurrent worker load", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?panes=2");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.startBackgroundLoad(2));
  await page.waitForFunction(() =>
    (window.__nimbusVtkHarness?.snapshot().backgroundLoad.iterations ?? 0) >= 8
  );

  const evidence: Array<{ plot: HarnessPlotType; fingerprint: string; canvases: number }> = [];
  for (const plot of APPEARANCE_PLOTS) {
    await settle(page, plot);
    const snapshot = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(snapshot.paneCount).toBe(2);
    expect(snapshot.canvasCount).toBe(2);
    expect(snapshot.backgroundLoad.workers).toBe(2);
    await expect(page.locator(".volume-render__quality-badge")).toHaveCount(0);
    const first = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    const second = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    expect(first).not.toBeNull();
    expect(second).not.toBeNull();
    if (!first || !second) continue;
    expect(first.evidence.metadataFingerprint).toBe(second.evidence.metadataFingerprint);
    expect(createHash("sha256").update(pngBytes(first.pngDataUri)).digest("hex"))
      .toBe(createHash("sha256").update(pngBytes(second.pngDataUri)).digest("hex"));
    evidence.push({ plot, fingerprint: first.evidence.metadataFingerprint, canvases: 2 });
  }

  await page.evaluate(() => {
    window.__nimbusVtkHarness?.stopBackgroundLoad();
    window.__nimbusVtkHarness?.setMounted(false);
  });
  await page.waitForFunction(() => {
    const snapshot = window.__nimbusVtkHarness?.snapshot();
    return snapshot?.workers === 0
      && snapshot.canvasCount === 0
      && snapshot.visualization.reservedBytes === 0;
  }, undefined, { timeout: 45_000 });
  expect(evidence).toHaveLength(APPEARANCE_PLOTS.length);
  expect(pageErrors).toEqual([]);
  const evidencePath = testInfo.outputPath("split-concurrent-background-load.appearance.json");
  await writeFile(evidencePath, `${JSON.stringify({
    schema: "nimbus.studio.vtk-split-concurrent-appearance",
    schemaVersion: 1,
    panes: 2,
    backgroundWorkers: 2,
    renderers: evidence,
  }, null, 2)}\n`, "utf8");
  await testInfo.attach("split-concurrent-background-load-appearance", {
    path: evidencePath,
    contentType: "application/json",
  });
});

test("thermodynamic volumes switch atomically between structure and raw science values", async ({ page }) => {
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await settle(page, "volume");
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
  const fieldGroup = page.locator(".control-group").filter({
    has: page.locator(".control-group__label", { hasText: "Field" }),
  });
  await fieldGroup.locator("select").selectOption("analytic_stratified_anomaly");
  const modeGroup = page.locator(".control-group").filter({
    has: page.locator(".control-group__label", { hasText: "Volume view" }),
  });
  await expect(modeGroup.getByRole("button", { name: "Structure" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute(
    "data-scalar-transform",
    "horizontal-level-anomaly",
  );
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute(
    "data-volume-material-committed",
    "true",
  );
  await expect(page.locator(".plot3d-badge__sub")).toContainText("horizontal-level anomaly");
  await expect(page.locator(".plot3d-colorbar__title")).toContainText("level anomaly");
  const structure = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(structure?.evidence.descriptor.parameters?.scalarTransform)
    .toBe("horizontal-level-anomaly");
  expect(structure?.evidence.descriptor.parameters?.effectiveOpacityClass)
    .toBe("signed-perturbation");

  await modeGroup.getByRole("button", { name: "Raw field" }).click();
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-scalar-transform", "none");
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute(
    "data-volume-material-committed",
    "true",
  );
  await expect(page.locator(".plot3d-badge__sub")).toContainText("background stratified opacity");
  await expect(page.locator(".plot3d-badge__sub")).toContainText("flat lit");
  await expect(page.locator(".plot3d-colorbar__title")).not.toContainText("level anomaly");
  const raw = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(raw?.evidence.descriptor.parameters?.scalarTransform).toBe("none");
  expect(raw?.evidence.descriptor.parameters?.effectiveOpacityClass).toBe("background-stratified");
  expect(raw?.evidence.metadataFingerprint).not.toBe(structure?.evidence.metadataFingerprint);
});

test("analytic volume textures preserve source axes and horizontal anomaly values", async ({ page }, testInfo) => {
  await page.goto("/harness/index.html?volumeBricks=off");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await settle(page, "volume");
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
  const picker = page.getByRole("combobox", { name: "Field", exact: true });
  const shape: [number, number, number] = [48, 36, 24];
  const evidence: Array<{ field: string; mode: string; checked: number; maximumError: number; shape: number[]; origin: number[]; spacing: number[]; nonFinite: number }> = [];

  for (const field of ["analytic_smooth_gradient", "analytic_signed_couplet", "analytic_stratified_anomaly"] as const) {
    await picker.selectOption(field);
    const source = createAnalyticFidelityFixture(field, shape).field.data;
    const means = Array.from({ length: shape[2] }, (_, z) => {
      let sum = 0;
      for (let x = 0; x < shape[0]; x++) for (let y = 0; y < shape[1]; y++) {
        sum += source[(x * shape[1] + y) * shape[2] + z];
      }
      return sum / (shape[0] * shape[1]);
    });
    for (const mode of field === "analytic_signed_couplet" ? ["Raw field"] : ["Raw field", "Structure"]) {
      if (field !== "analytic_signed_couplet") await page.getByRole("button", { name: mode, exact: true }).click();
      await settle(page, "volume");
      const images = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeScalarImages());
      expect(images).toHaveLength(1);
      const image = images[0];
      expect(image.shape).toEqual(shape);
      expect(image.origin).toEqual([0, 0, 0]);
      expect(image.spacing).toEqual([125, 175, 90]);
      expect(image.values).toHaveLength(source.length);
      let maximumError = 0;
      let nonFinite = 0;
      for (let z = 0; z < shape[2]; z++) for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[0]; x++) {
        const expected = source[(x * shape[1] + y) * shape[2] + z] - (mode === "Structure" ? means[z] : 0);
        const actual = image.values[x + shape[0] * (y + shape[1] * z)];
        if (!Number.isFinite(actual)) nonFinite++;
        maximumError = Math.max(maximumError, Math.abs(actual - expected));
      }
      expect(nonFinite).toBe(0);
      expect(maximumError, `${field} ${mode}: C-order source to x-fast texture`).toBeLessThan(1e-6);
      evidence.push({ field, mode, checked: source.length, maximumError, shape: image.shape, origin: image.origin, spacing: image.spacing, nonFinite });
    }
  }
  const path = testInfo.outputPath("analytic-volume-source-reference.json");
  await writeFile(path, JSON.stringify(evidence, null, 2));
  await testInfo.attach("analytic-volume-source-reference", { path, contentType: "application/json" });
});

for (const plot of ["volume", "isosurface"] as const) test(`${plot} reloads an edited expression at the same saved frame without resetting the camera`, async ({ page }, info) => {
  test.setTimeout(info.project.name.includes("swiftshader") ? 90_000 : 45_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?definitionProbe=1&volumeBricks=off");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await settle(page, plot, 15_000);
  await page.locator(".plot3d-control-tab", { hasText: plot === "volume" ? "Volume" : "Data" }).click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("expression_probe");
  await settle(page, plot, 15_000);
  const before = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
  const texture = plot === "volume" ? await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeScalarImages()) : null;
  await page.evaluate(() => window.__nimbusVtkHarness!.setFieldReadDelay(400));
  await page.evaluate(() => window.__nimbusVtkHarness!.clearFieldCache());
  await page.evaluate(() => window.__nimbusVtkHarness!.setExpressionMultiplier(3));
  await expect(page.locator('[data-testid="scene"]')).toHaveAttribute("data-frame-loading", "true");
  const retained = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
  expect(retained.evidence.camera).toEqual(before.evidence.camera);
  if (texture) expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeScalarImages())).toEqual(texture);
  try {
    await settle(page, plot, 15_000);
  } catch (error) {
    const diagnostics = await page.evaluate(() => ({ snapshot: window.__nimbusVtkHarness!.snapshot(),
      scene: { ...document.querySelector<HTMLElement>('[data-testid="scene"]')!.dataset },
      volume: { ...document.querySelector<HTMLElement>(".plot3d-overlay-host")!.dataset },
      overlays: Array.from(document.querySelectorAll(".volume-render__overlay")).map(element => element.textContent) }));
    await writeFile(info.outputPath("source-definition-failure.json"), JSON.stringify({ errors, diagnostics }, null, 2));
    throw error;
  }
  const after = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
  expect(after.evidence.descriptor.frameIndex).toBe(before.evidence.descriptor.frameIndex);
  expect(after.evidence.descriptor.field).toBe("expression_probe");
  expect(after.evidence.camera).toEqual(before.evidence.camera);
  if (texture) {
    const updated = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeScalarImages());
    expect(updated[0].values).not.toEqual(texture[0].values);
    const maximumError = Math.max(...updated[0].values.map((value, index) => Math.abs(value - texture[0].values[index] * 1.5)));
    expect(maximumError).toBeLessThan(4e-6);
  } else expect(after.evidence.layers).not.toEqual(before.evidence.layers);
  const path = info.outputPath("source-definition-reload.json");
  await writeFile(path, JSON.stringify({ plot, before: before.evidence, pending: retained.evidence, after: after.evidence }, null, 2));
  await info.attach("source-definition-reload", { path, contentType: "application/json" });
  expect(errors).toEqual([]);
});
