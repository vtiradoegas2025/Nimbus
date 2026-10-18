import { test, expect } from "@playwright/test";
import { advanceCloudMotionClock, armCloudMotionClock, installCloudMotionClock, resumeCloudMotionClock, waitForCloudMotionFrame } from "../support/cloudMotionClock";

test("cloud interpolation preserves endpoints, reports intermediate time, and reuses textures", async ({ page }, info) => {
  test.setTimeout(120_000);
  await installCloudMotionClock(page);
  const errors: string[] = [];
  const readDiagnostics: unknown[] = [];
  page.on("console", message => {
    const text = message.text();
    if (text.startsWith("NIMBUS_FIELD_READ_DIAGNOSTIC ") && readDiagnostics.length < 6) {
      readDiagnostics.push(JSON.parse(text.slice("NIMBUS_FIELD_READ_DIAGNOSTIC ".length)));
    }
  });
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const pending = new Set<WebGLSync>();
    const counters = { maximum: 0, pending: 0 };
    Object.assign(window, { motionFences: counters });
    const create = WebGL2RenderingContext.prototype.fenceSync;
    const remove = WebGL2RenderingContext.prototype.deleteSync;
    WebGL2RenderingContext.prototype.fenceSync = function (...args) {
      const sync = create.apply(this, args);
      if (sync) pending.add(sync);
      counters.pending = pending.size; counters.maximum = Math.max(counters.maximum, pending.size);
      return sync;
    };
    WebGL2RenderingContext.prototype.deleteSync = function (sync) {
      if (sync) pending.delete(sync);
      counters.pending = pending.size;
      remove.call(this, sync);
    };
  });
  await page.goto("/harness/index.html?fidelity=presentation");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(512, 384));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  const ready = (step: number) => page.waitForFunction(step => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(step);
  }, step, { timeout: 30_000 });
  await ready(0);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1)); await ready(1);
  const assertExactTime = async (timeS: number) => {
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    const motion = capture!.evidence.descriptor.parameters!.cloudMotion as { active: boolean; fraction: number; timeS: number };
    expect(motion.active).toBe(false); expect(motion.fraction).toBe(1); expect(motion.timeS).toBe(timeS);
  };
  const endpoint = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(0)); await ready(0);
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "Interpolate cloud", exact: true }).click();
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.evaluate(() => {
    const started = performance.now();
    let previous = started, count = 0;
    const target = window as typeof window & { __nimbusMotionDiagnosticTimer?: number };
    const sample = () => {
      const now = performance.now();
      const snapshot = window.__nimbusVtkHarness!.snapshot();
      const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
      console.info("NIMBUS_FIELD_READ_DIAGNOSTIC " + JSON.stringify({ elapsedMs: now - started, callbackGapMs: now - previous,
        requestedStep: snapshot.step, ipc: snapshot.ipc, cache: snapshot.fieldCache, scheduler: snapshot.fieldScheduler,
        prefetches: snapshot.fieldPrefetches, visualization: snapshot.visualization,
        prepared: snapshot.preparedRegistry, workers: snapshot.workers, workerPosts: snapshot.workerPosts,
        cloudLoad: data?.cloudLoadState, cloudPreparation: data?.cloudPreparationState, admission: data?.sceneGpuAdmission,
        presentationPending: data?.scenePresentationPending, motionActive: data?.cloudMotionActive,
      }));
      previous = now;
      if (++count >= 6) { clearInterval(target.__nimbusMotionDiagnosticTimer); delete target.__nimbusMotionDiagnosticTimer; }
    };
    target.__nimbusMotionDiagnosticTimer = window.setInterval(sample, 500);
    sample();
  });
  try {
    await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
    await page.waitForTimeout(100);
    await armCloudMotionClock(page);
    await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
    await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true", undefined, { timeout: 30_000 });
    await waitForCloudMotionFrame(page);
  } catch (error) {
    await info.attach("failed-cloud-field-read", { body: JSON.stringify(readDiagnostics, null, 2), contentType: "application/json" });
    throw error;
  }
  await page.evaluate(() => {
    const target = window as typeof window & { __nimbusMotionDiagnosticTimer?: number };
    clearInterval(target.__nimbusMotionDiagnosticTimer); delete target.__nimbusMotionDiagnosticTimer;
  });
  const first = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  const a = first!.evidence.descriptor.parameters!.cloudMotion as { fraction: number; timeS: number };
  await advanceCloudMotionClock(page, 180);
  await page.waitForFunction((fraction) => Number(document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionFraction) > fraction, a.fraction);
  const second = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const after = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(first!.evidence.descriptor.frameIndex).toBeNull();
  expect(second!.evidence.descriptor.frameIndex).toBeNull();
  const b = second!.evidence.descriptor.parameters!.cloudMotion as { fraction: number; timeS: number };
  expect(b.fraction).toBeGreaterThan(a.fraction);
  expect(b.timeS).toBeGreaterThan(a.timeS);
  expect(b.timeS).toBeLessThan(30);
  expect(after.texture3DUploads).toBe(before.texture3DUploads);
  await resumeCloudMotionClock(page);
  await ready(1);
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints)).toEqual(endpoint);
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(2));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(false, true));
  await ready(2);
  await resumeCloudMotionClock(page);
  await assertExactTime(60);
  expect(await page.locator(".plot3d-overlay-host").getAttribute("data-cloud-motion-active")).toBe("false");
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
  await page.waitForTimeout(100);
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(3));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.captureSavedFrame())).toMatch(/^data:image\/png/);
  await ready(3);
  await resumeCloudMotionClock(page);
  await assertExactTime(90);
  expect(await page.locator(".plot3d-overlay-host").getAttribute("data-cloud-motion-active")).toBe("false");
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(0)); await ready(0);
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  await page.locator(".plot3d-control-tab", { hasText: "Precip" }).click();
  await page.getByRole("group", { name: "Precipitation", exact: true }).getByRole("button").first().click();
  await ready(1);
  await resumeCloudMotionClock(page);
  expect(await page.locator(".plot3d-overlay-host").getAttribute("data-cloud-motion-active")).toBe("false");
  await page.getByRole("group", { name: "Precipitation", exact: true }).getByRole("button").first().click();
  await ready(1);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(0)); await ready(0);
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  const cameraBefore = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence.camera;
  const canvas = (await page.locator(".volume-render__canvas canvas").boundingBox())!;
  await page.mouse.move(canvas.x + canvas.width * 0.5, canvas.y + canvas.height * 0.5);
  await page.mouse.down();
  await resumeCloudMotionClock(page);
  await page.mouse.move(canvas.x + canvas.width * 0.5 + 20, canvas.y + canvas.height * 0.5, { steps: 4 });
  await page.mouse.up(); await ready(1);
  await assertExactTime(30);
  const cameraAfter = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence.camera;
  const direction = (camera: typeof cameraBefore) => camera.position!.map((value, axis) => value - camera.focalPoint![axis]);
  const left = direction(cameraBefore), right = direction(cameraAfter);
  const cosine = left.reduce((sum, value, axis) => sum + value * right[axis], 0) / (Math.hypot(...left) * Math.hypot(...right));
  expect(Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI).toBeLessThan(20);
  await info.attach("interpolation", { body: JSON.stringify({ a, b, before, after }), contentType: "application/json" });
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes)).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { motionFences: { maximum: number; pending: number } }).motionFences)).toEqual({ maximum: 1, pending: 0 });
  expect(errors).toEqual([]);
});

test("graphics-context loss disables interpolation and restores exact saved frames", async ({ page }) => {
  await installCloudMotionClock(page);
  await page.goto("/harness/index.html?fidelity=presentation");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(512, 384));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  const ready = (step: number) => page.waitForFunction(step => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(step);
  }, step, { timeout: 30_000 });
  await ready(0);
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "Interpolate cloud", exact: true }).click();
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 2500));
  await page.waitForTimeout(100);
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionActive === "true");
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.loseContext())).toBe(true);
  await expect(page.getByText("3D graphics context lost", { exact: false })).toBeVisible();
  await ready(1);
  await resumeCloudMotionClock(page);
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await expect(page.getByRole("button", { name: "Saved frames", exact: true })).toHaveClass(/pill--active/);
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(2)); await ready(2);
  expect(await page.locator(".plot3d-overlay-host").getAttribute("data-cloud-motion-active")).toBe("false");
  const captured = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(captured!.evidence.descriptor.frameIndex).toBe(2);
  const motion = captured!.evidence.descriptor.parameters!.cloudMotion as { active: boolean; timeS: number; toStep: number };
  expect(motion.active).toBe(false); expect(motion.timeS).toBe(60); expect(motion.toStep).toBe(2);
});
