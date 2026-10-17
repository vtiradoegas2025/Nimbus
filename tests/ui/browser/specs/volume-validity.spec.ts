import { test, expect } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test("exact GPU texture tiling preserves every logical scalar mask and boundary interpolation", async ({ page }, info) => {
  test.setTimeout(90_000);
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const rows = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeTexturePacking());
  await info.attach("logical-texture-packing", { body: JSON.stringify(rows), contentType: "application/json" });
  for (const row of rows) {
    expect(row.nativeShaderHasPacking).toBe(false);
    expect(row.nativeHardwareShaderHasTexelFetch).toBe(false);
    expect(row.layout.textureShape.every(value => value <= row.admittedMaximum)).toBe(true);
    expect(row.scalarMismatches).toBe(0);
    expect(row.maskMismatches).toBe(0);
    expect(row.unchangedSource).toBe(true);
    expect(row.unchangedMask).toBe(true);
    expect(row.checkedScalarCells).toBe(row.layout.sourceCells);
    expect(row.textureMinimum).toBe(row.minimum);
    expect(row.gpuBytes).toBe(row.layout.textureCells * 4);
    expect(row.maskGpuBytes).toBe(row.layout.textureCells);
    if (row.layout.tiled) expect(row.seams.length).toBeGreaterThan(0);
    expect(row.glError).toBe(0);
    for (const sample of row.interpolation) {
      expect(Math.abs(sample.scalar - sample.expectedScalar)).toBeLessThanOrEqual(row.interpolationTolerance + (row.hardwareInterpolation ? 2*Math.max(Math.abs(row.minimum),1)/256 : 0));
      expect(Math.abs(sample.mask - sample.expectedMask)).toBeLessThanOrEqual(row.maskInterpolationTolerance);
      expect(sample.nearestBits).toBe(sample.expectedNearestBits);
      expect(sample.valid).toBe(sample.expectedValid);
    }
  }
  expect(rows.some(row => row.sourceShape[2] > row.actualMaximum)).toBe(true);
  expect(rows.some(row => !row.layout.tiled && row.shaderMode === "production-manual")).toBe(true);
});

test("ordinary Supercell native columns tile before upload and survive saved-frame context restoration", async ({ page }, info) => {
  test.setTimeout(300_000);
  const diagnostics: unknown[] = [];
  page.on("console", message => {
    if (message.text().startsWith("atlas-restore-diagnostic:")) {
      try { diagnostics.push(JSON.parse(message.text().slice("atlas-restore-diagnostic:".length))); } catch { /* Only bounded structured diagnostic records are retained. */ }
    }
  });
  page.on("pageerror", error => diagnostics.push({ pageerror: error.message }));
  await page.addInitScript(() => {
    const receipt = { lost: 0, restored: 0 };
    Object.assign(window, { atlasContextReceipt: receipt });
    const record = (event: string) => console.log("atlas-restore-diagnostic:" + JSON.stringify({
      event, at: performance.now(), visibility: document.visibilityState, receipt: { ...receipt },
    }));
    document.addEventListener("webglcontextlost", () => { receipt.lost += 1; record("lost"); }, true);
    document.addEventListener("webglcontextrestored", () => { receipt.restored += 1; record("restored"); }, true);
    const get = WebGL2RenderingContext.prototype.getParameter;
    WebGL2RenderingContext.prototype.getParameter = function(parameter: number) {
      return parameter === this.MAX_3D_TEXTURE_SIZE ? 8 : get.call(this, parameter);
    };
  });
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(512, 384));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  const settle = (expectedStep: number) => page.waitForFunction(expected => {
    const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return data?.scenePresentationPending === "false" && data.committedSceneGeneration === data.requestedSceneGeneration
      && scene?.dataset.renderedStep === String(expected);
  }, expectedStep, { timeout: 60_000 }).catch(async (error: unknown) => {
    await info.attach("atlas-restore-stream", { body: JSON.stringify(diagnostics), contentType: "application/json" });
    throw error;
  });
  await settle(0);
  const before = await page.evaluate(async () => ({ images: window.__nimbusVtkHarness!.inspectVolumeScalarImages(),
    capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence }));
  expect(before.images.some(image => Math.max(...image.shape) > 8)).toBe(true);
  expect(before.images.every(image => image.values.length <= 512)).toBe(true);
  expect(before.capture.descriptor.frameIndex).toBe(0);
  const losses = await page.evaluate(() => (window as unknown as { atlasContextReceipt: { lost: number } }).atlasContextReceipt.lost);
  await page.evaluate(() => {
    let samples = 1;
    const record = () => {
      const dataset = { ...document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset };
      console.log("atlas-restore-diagnostic:" + JSON.stringify({ at: performance.now(), visibility: document.visibilityState, dataset,
        receipt: (window as unknown as { atlasContextReceipt: { lost: number; restored: number } }).atlasContextReceipt,
        resources: window.__nimbusVtkHarness!.snapshot() }));
    };
    record();
    const timer = setInterval(() => { record(); if (++samples === 8) clearInterval(timer); }, 500);
  });
  const supported = await page.evaluate(() => window.__nimbusVtkHarness!.loseContext());
  expect(supported).toBe(true);
  await expect.poll(() => page.evaluate(() => (window as unknown as { atlasContextReceipt: { lost: number } }).atlasContextReceipt.lost)).toBeGreaterThan(losses);
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.restoreContext())).toBe(true);
  await settle(0);
  await expect(page.getByText("3D graphics context lost", { exact: false })).not.toBeVisible();
  const restored = await page.evaluate(async () => ({ images: window.__nimbusVtkHarness!.inspectVolumeScalarImages(),
    capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence }));
  expect(restored.images).toEqual(before.images);
  expect(restored.capture.descriptor.frameIndex).toBe(0);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await settle(1);
  const advanced = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(advanced!.evidence.descriptor.frameIndex).toBe(1);
  await info.attach("native-column-context-restoration", { body: JSON.stringify({ before, restored, advanced: advanced!.evidence }), contentType: "application/json" });
  await info.attach("atlas-restore-stream", { body: JSON.stringify(diagnostics), contentType: "application/json" });
});

test("invalid viewer texture capacity rejects before any field payload read", async ({ page }) => {
  await page.addInitScript(() => {
    const get = WebGL2RenderingContext.prototype.getParameter;
    WebGL2RenderingContext.prototype.getParameter = function(parameter: number) {
      return parameter === this.MAX_3D_TEXTURE_SIZE ? 1 : get.call(this, parameter);
    };
  });
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await expect(page.locator(".volume-render__overlay--error")).toContainText("valid viewer 3D texture limit");
  const snapshot = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(snapshot.ipc.startedJobs).toBe(0);
});

test("mounted playback screen adaptation preserves scalar textures ray spacing and full resolution captures", async ({ page }, info) => {
  test.setTimeout(60_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=playback-screen");
  await page.waitForFunction(() => Boolean(window.__nimbusPlaybackScreenHarness)
    && document.querySelector("canvas")?.width === 256);
  const initialCapture = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.capture());
  const initial = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.snapshot());
  expect(initial.distance).toBe(1);
  expect(initial.values).toEqual(initial.source);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.setPlayback(true));
  await page.waitForFunction(() => window.__nimbusPlaybackScreenHarness!.snapshot().controller.playing);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.completeGpuSamples(40, 9));
  const reduced = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.snapshot());
  expect(reduced.distance).toBe(1.5);
  expect(reduced.controller.measuredGpuMs).toBe(40);
  expect(reduced.controller.completedSamples).toBeGreaterThanOrEqual(9);
  const exported = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.capture());
  expect(exported.heldDistance).toBe(1);
  expect(exported.restoredDistance).toBe(1.5);
  expect(exported.png).toBe(initialCapture.png);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.setPlayback(true, true));
  await page.waitForFunction(() => !window.__nimbusPlaybackScreenHarness!.snapshot().controller.playing);
  const recording = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.snapshot());
  expect(recording.distance).toBe(1);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.setPlayback(true));
  await page.waitForFunction(() => window.__nimbusPlaybackScreenHarness!.snapshot().controller.playing);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.completeGpuSamples(40, 3));
  expect((await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.snapshot())).distance).toBe(1.25);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.setPlayback(false));
  await page.waitForFunction(() => !window.__nimbusPlaybackScreenHarness!.snapshot().controller.playing);
  const paused = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.snapshot());
  expect(paused.distance).toBe(1);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.setPlayback(true));
  await page.waitForFunction(() => window.__nimbusPlaybackScreenHarness!.snapshot().controller.playing);
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.completeGpuSamples(40, 3));
  await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.setTimingUnavailable());
  const unavailable = await page.evaluate(() => window.__nimbusPlaybackScreenHarness!.snapshot());
  expect(unavailable.distance).toBe(1);
  expect(unavailable.controller.timing).toBe("unavailable");
  const stages = [initial, reduced, recording, paused, unavailable];
  for (const stage of stages) {
    expect(stage.values).toEqual(initial.source);
    expect(stage.validity).toEqual(initial.validity);
    expect(stage.sameArray).toBe(true);
    expect(stage.sameMtime).toBe(true);
    expect(stage.dimensions).toEqual(initial.dimensions);
    expect(stage.spacing).toEqual(initial.spacing);
    expect(stage.origin).toEqual(initial.origin);
    expect(stage.sampleDistance).toBe(initial.sampleDistance);
    expect(stage.opacityDistance).toBe(initial.opacityDistance);
    expect(stage.cameraPosition).toEqual(initial.cameraPosition);
    expect(stage.cameraFocus).toEqual(initial.cameraFocus);
    expect(stage.texture3DUploads).toBe(initial.texture3DUploads);
    expect(stage.glError).toBe(0);
  }
  const path = info.outputPath("playback-screen-resolution.json");
  await writeFile(path, JSON.stringify(stages, null, 2));
  await info.attach("playback-screen-resolution", { path, contentType: "application/json" });
  await writeFile(info.outputPath("full-resolution-export.png"), Buffer.from(exported.png.split(",")[1], "base64"));
  expect(errors).toEqual([]);
});

test("mounted published overviews share bounded reads preserve publications and reject invalid hierarchies", async ({ page }, info) => {
  test.setTimeout(60_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=published-overview");
  await page.waitForFunction(() => Boolean(window.__nimbusPublishedOverviewHarness));
  const waitComplete = async (frame: number, level: number) => {
    await page.waitForFunction(({ frame, level }) => {
      const value = window.__nimbusPublishedOverviewHarness!.snapshot();
      return value.frame === frame && value.consumers.length === 2 && value.consumers.every(consumer =>
        !consumer.loading && !consumer.planning && !consumer.error && consumer.level === level
        && consumer.loadedStepIndex === frame && consumer.requestedStepIndex === frame
        && consumer.loadedPublicationGeneration === value.publication
        && consumer.requestedPublicationGeneration === value.publication && consumer.cells > 0);
    }, { frame, level }, { timeout: 15_000 });
    return page.evaluate(() => window.__nimbusPublishedOverviewHarness!.snapshot());
  };
  const rows: unknown[] = [];
  const initial = await waitComplete(0, 1);
  rows.push({ stage: "two-mounted-consumers", ...initial });
  expect(initial.consumers.map(value => value.shape)).toEqual([[48, 36, 24], [48, 36, 24]]);
  expect(initial.calls.filter(value => value.command === "read_field_brick_hierarchy")).toHaveLength(1);
  expect(initial.calls.filter(value => value.command === "read_field_brick_window")).toHaveLength(1);
  expect(initial.calls.filter(value => value.command === "read_field_window" || value.command === "read_field_bytes")).toHaveLength(0);
  const compare = async (stage: string, level: number) => {
    const value = await page.evaluate(() => window.__nimbusPublishedOverviewHarness!.compareOrdinaryRead());
    rows.push({ stage, ...value });
    expect(value.level).toBe(level);
    expect(value.cells).toBeLessThanOrEqual(50_000);
    expect(value.comparedValues).toBe(value.cells * 2);
    expect(value.mismatches).toEqual([0, 0]);
    expect(value.geometryMatches).toEqual([true, true]);
    expect(value.sameConsumerBacking).toBe(true);
    expect(value.independentReference).toBe(true);
    expect(value.geometry?.sourceShape).toEqual([96, 72, 48]);
    expect(value.geometry?.origin).toEqual([0, 0, 0]);
    expect(value.geometry?.extent).toEqual([96, 72, 48]);
    expect(value.geometry?.sourceFormat).toBe(level === 1 ? "npy_lod" : "npy");
    return value;
  };
  await compare("independent-published-reference", 1);
  const beforeWarm = await page.evaluate(() => window.__nimbusPublishedOverviewHarness!.snapshot());
  await page.evaluate(() => {
    for (let index = 0; index < 12; index++) window.__nimbusPublishedOverviewHarness!.rerender();
  });
  await page.waitForFunction(() => window.__nimbusPublishedOverviewHarness!.snapshot().revision === 12);
  const afterWarm = await waitComplete(0, 1);
  expect(afterWarm.calls).toEqual(beforeWarm.calls);
  rows.push({ stage: "warm-props-no-reads", ...afterWarm });
  await page.evaluate(() => {
    window.__nimbusPublishedOverviewHarness!.setMetadataDelay(150);
    window.__nimbusPublishedOverviewHarness!.setFrame(1);
  });
  await page.waitForFunction(() => {
    const value = window.__nimbusPublishedOverviewHarness!.snapshot();
    return value.frame === 1 && value.consumers.length === 2 && value.consumers.every(consumer =>
      consumer.loading && consumer.planning && consumer.requestedStepIndex === 1
      && consumer.requestedPublicationGeneration === value.publication && consumer.loadedStepIndex !== 1);
  }, undefined, { timeout: 5_000 });
  rows.push({ stage: "new-publication-planning", ...await page.evaluate(() => window.__nimbusPublishedOverviewHarness!.snapshot()) });
  const next = await waitComplete(1, 1);
  expect(next.calls.filter(value => value.command === "read_field_brick_hierarchy" && value.frame === 1)).toHaveLength(1);
  expect(next.calls.filter(value => value.command === "read_field_brick_window" && value.frame === 1)).toHaveLength(1);
  await compare("independent-next-frame-reference", 1);
  await page.evaluate(() => {
    window.__nimbusPublishedOverviewHarness!.setMetadataDelay(0);
    window.__nimbusPublishedOverviewHarness!.setMode("native-only");
  });
  const raw = await waitComplete(1, 0);
  rows.push({ stage: "valid-native-only-fallback", ...raw });
  expect(raw.consumers.every(value => value.disclosure?.startsWith("bounded raw cell-mean overview"))).toBe(true);
  expect(raw.calls.filter(value => value.command === "read_field_bytes")).toHaveLength(0);
  expect(raw.calls.filter(value => value.command === "read_field_window").length).toBeGreaterThan(0);
  const rawReference = await compare("independent-bounded-raw-reference", 0);
  expect(rawReference.windows.every(window => window.aggregation === "cell_mean")).toBe(true);
  const beforeInvalid = await page.evaluate(() => window.__nimbusPublishedOverviewHarness!.snapshot());
  for (const mode of ["changed-publication", "invalid-shape"] as const) {
    await page.evaluate(mode => window.__nimbusPublishedOverviewHarness!.setMode(mode), mode);
    await page.waitForFunction(mode => {
      const value = window.__nimbusPublishedOverviewHarness!.snapshot();
      return value.mode === mode && value.consumers.length === 2 && value.consumers.every(consumer =>
        !consumer.loading && !consumer.planning && consumer.error?.includes("hierarchy failed validation")
        && consumer.level === null && consumer.cells === 0 && consumer.loadedStepIndex === null);
    }, mode, { timeout: 10_000 });
    const failed = await page.evaluate(() => window.__nimbusPublishedOverviewHarness!.snapshot());
    rows.push({ stage: mode, ...failed });
    expect(failed.calls.filter(value => value.command !== "read_field_brick_hierarchy"))
      .toEqual(beforeInvalid.calls.filter(value => value.command !== "read_field_brick_hierarchy"));
    expect(failed.calls.filter(value => value.command === "read_field_brick_hierarchy").length)
      .toBe(beforeInvalid.calls.filter(value => value.command === "read_field_brick_hierarchy").length + (mode === "changed-publication" ? 1 : 2));
  }
  const path = info.outputPath("mounted-published-overview.json");
  await writeFile(path, JSON.stringify(rows, null, 2));
  await info.attach("mounted-published-overview", { path, contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("missing volume support stays transparent while valid zero and signed values remain visible", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const rows = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeValidity());
  for (const row of rows) {
    expect(row.glError, row.mode).toBe(0);
    expect(row.repeatedUploads, row.mode).toBe(0);
    expect(row.validityBytes).toBe(row.mode === "valid-zero" ? 0 : 4096);
    for (const [left, right] of row.pixels) {
      expect(Math.abs(left[0] - (row.mode === "valid-zero" ? right[0] : 0)), row.mode).toBeLessThanOrEqual(1);
      if (row.mode === "all-missing") expect(right[0]).toBeLessThanOrEqual(1);
      else expect(right[0]).toBeGreaterThan(150);
      expect(Math.max(left[1], left[2], right[1], right[2])).toBeLessThanOrEqual(1);
    }
    expect(row.pixels[0]).toEqual(row.pixels[2]);
    await writeFile(info.outputPath(`${row.mode}.png`), Buffer.from(row.captures[0].split(",")[1], "base64"));
  }
  await info.attach("validity-results", { body: JSON.stringify(rows.map(({ captures: _, ...row }) => row)), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("unchanged volume spans reuse GPU textures while reshaped validity and changed scalars update correctly", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const stages = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeArrayReuse());
  const stage = (name: string) => stages.find(value => value.name === name)!;
  for (const value of stages) {
    expect(value.glError, value.name).toBe(0);
    await writeFile(info.outputPath(`${value.name}.png`), Buffer.from(value.png.split(",")[1], "base64"));
  }
  for (const name of ["same-preparation", "same-span-wrapper", "origin-and-spacing", "warm-repeat"]) {
    expect(stage(name).uploads, name).toBe(0);
    expect(stage(name).sameScalarArray, name).toBe(true);
    expect(stage(name).sameScalarMtime, name).toBe(true);
  }
  expect(stage("origin-and-spacing").spacing).toEqual([2, 1, 2]);
  expect(stage("origin-and-spacing").origin).toEqual([11, 0.5, 1]);
  for (const name of ["same-buffer-reshaped", "changed-scalars"]) {
    expect(stage(name).uploads, name).toBeGreaterThan(0);
    expect(stage(name).sameScalarArray, name).toBe(false);
  }
  expect(stage("changed-mask").sameScalarArray).toBe(true);
  expect(stage("changed-mask").sameScalarMtime).toBe(true);
  expect(stage("changed-mask").uploads).toBeGreaterThan(0);
  expect(stage("mask-removed").uploads).toBe(0);
  const expectSignal = (name: string, visible: readonly boolean[], channel: number) => {
    stage(name).pixels.forEach((pixel, index) => {
      if (visible[index]) expect(pixel[channel], `${name} pixel ${index}`).toBeGreaterThan(150);
      else expect(pixel[channel], `${name} pixel ${index}`).toBeLessThanOrEqual(1);
      for (const other of [0, 1, 2].filter(value => value !== channel)) expect(pixel[other], `${name} pixel ${index}`).toBeLessThanOrEqual(1);
    });
  };
  expectSignal("initial", [false, false, true, true], 0);
  expectSignal("same-buffer-reshaped", [false, true, false, true], 0);
  expectSignal("changed-scalars", [false, true, false, true], 2);
  expectSignal("changed-mask", [true, true, true, true], 2);
  expectSignal("mask-removed", [true, true, true, true], 2);
  const path = info.outputPath("volume-array-reuse.json");
  await writeFile(path, JSON.stringify(stages.map(({ png: _, ...value }) => value), null, 2));
  await info.attach("volume-array-reuse", { path, contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("float32 scalar textures preserve small gradients with outliers and absent hardware float filtering", async ({ page }, info) => {
  test.setTimeout(60_000);
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const rows = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeScalarPrecision());
  const path = info.outputPath("scalar-texture-precision.json");
  await writeFile(path, JSON.stringify(rows, null, 2));
  await info.attach("scalar-texture-precision", { path, contentType: "application/json" });
  for (const row of rows.filter(value => value.mode.startsWith("float32"))) {
    expect(row.glError, `${row.mode} ${row.offset}`).toBe(0);
    expect(row.internalFormat).toBe(0x822e);
    expect(row.allocatedBytes).toBe(row.expectedScalarBytes);
    for (const sample of row.samples.filter(value => value.texelCenter)) expect(sample.actual).toBe(sample.stored);
    const arithmeticTolerance = row.offset >= 280 ? 0.0002 : 0.00001;
    const nativeInterpolationTolerance = row.mode === "float32-native" ? (0.01 + 0.007 + 0.003) / 256 : 0;
    expect(row.maximumError, `${row.mode} ${row.offset}`).toBeLessThanOrEqual(arithmeticTolerance + nativeInterpolationTolerance);
  }
  expect(rows.some(value => value.mode === "upstream-without-float-linear" && value.offset === 1000 && value.maximumError > 0.01)).toBe(true);
});

test("manufactured volumes preserve angular anisotropic sampling and one-to-two field optical composition", async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const rows = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeAngularSampling());
  const path = info.outputPath("angular-volume-sampling.json");
  await writeFile(path, JSON.stringify(rows.map(({ captures, ...value }) => ({ ...value,
    captures: captures.map(({ png: _, ...capture }) => capture) })), null, 2));
  await info.attach("angular-volume-sampling", { path, contentType: "application/json" });
  for (const row of rows) {
    const name = `${row.field}-${row.shape.join("x")}-${row.azimuth}-${row.elevation}`;
    expect(row.glError, name).toBe(0);
    expect(row.warmUploads, name).toBe(0);
    expect(row.oneRmse, name).toBeLessThan(0.01);
    expect(row.mixtureRmse, name).toBeLessThan(0.015);
    if (row.field === "affine" && row.elevation === 0) expect(Math.max(...row.captures[0].pixels) - Math.min(...row.captures[0].pixels)).toBeLessThan(0.004);
    if (row.field !== "masked-affine") expect(row.captures[0].pixels[1], name).toBeGreaterThan(0.2);
  }
});
