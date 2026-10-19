import { test, expect } from "@playwright/test";
import { advanceCloudMotionClock, armCloudMotionClock, installCloudMotionClock, resumeCloudMotionClock, waitForCloudMotionFrame } from "../support/cloudMotionClock";

for (const mode of ["grid", "terrain", "nested", "terrain-nested"]) test(`${mode} clouds follow Earth winds across saved grid changes`, async ({ page }, info) => {
  test.setTimeout(mode.endsWith("nested") ? 300_000 : 180_000);
  await installCloudMotionClock(page);
  const activationOptions = mode.endsWith("nested") ? { timeout: 90_000 } : undefined;
  const errors: string[] = [];
  const diagnostics: unknown[] = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.text().startsWith("moving-cloud-diagnostic:")) {
      diagnostics.push(JSON.parse(message.text().slice("moving-cloud-diagnostic:".length)));
    }
  });
  await page.addInitScript(() => {
    const receipt = { armed: false, records: 0, phase: "", operations: 0, maximumBlockingMs: 0 };
    Object.assign(window, { movingCloudDiagnosticReceipt: receipt });
    const prototype = WebGL2RenderingContext.prototype as unknown as Record<string, (...args: unknown[]) => unknown>;
    for (const method of ["compileShader", "linkProgram", "drawArrays", "getShaderParameter", "getProgramParameter",
      "getUniformLocation", "getAttribLocation", "getParameter", "getError", "fenceSync", "clientWaitSync", "flush",
      "finish", "readPixels", "checkFramebufferStatus", "getBufferSubData", "getQueryParameter"]) {
      const original = prototype[method];
      let calls = 0;
      prototype[method] = function(...args: unknown[]) {
        if (!receipt.armed) return Reflect.apply(original, this, args);
        const started = performance.now();
        const sequence = ++receipt.operations;
        const first = ++calls === 1;
        const record = (event: string) => {
          if (receipt.records++ < 64) console.log("moving-cloud-diagnostic:" + JSON.stringify({
            phase: receipt.phase, event, method, sequence, at: performance.now(),
            elapsedMs: performance.now() - started, maximumBlockingMs: receipt.maximumBlockingMs,
            argument: args.find(value => typeof value === "number" || typeof value === "string"),
          }));
        };
        if (first) record("start");
        try { return Reflect.apply(original, this, args); }
        finally {
          const elapsed = performance.now() - started;
          receipt.maximumBlockingMs = Math.max(receipt.maximumBlockingMs, elapsed);
          if (first || elapsed >= 10) record("end");
        }
      };
    }
  });
  await page.goto(`/harness/index.html?fidelity=presentation&movingCloud=${mode}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "Wind-informed", exact: true }).click();
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 4000));
  await expect(page.getByText("Waiting for the next saved cloud frame", { exact: true })).toBeVisible();
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await page.evaluate(() => {
    const receipt = (window as unknown as { movingCloudDiagnosticReceipt: { armed: boolean; phase: string } }).movingCloudDiagnosticReceipt;
    receipt.phase = "first-transition"; receipt.armed = true;
    let samples = 1;
    const record = () => console.log("moving-cloud-diagnostic:" + JSON.stringify({
      at: performance.now(), visibility: document.visibilityState, receipt: { ...receipt },
      dataset: { ...document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset },
      resources: window.__nimbusVtkHarness!.snapshot(),
    }));
    record();
    const timer = setInterval(() => { record(); if (++samples === 8) clearInterval(timer); }, 500);
  });
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  try { await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-cloud-motion-active", "true", activationOptions); }
  catch (error) {
    await info.attach("moving-cloud-activation-stream", { body: JSON.stringify({ mode, errors, diagnostics }), contentType: "application/json" });
    throw error;
  }
  await waitForCloudMotionFrame(page);
  if (mode.endsWith("nested")) await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-output-level", "child_0000");
  const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const motion = capture!.evidence.descriptor.parameters!.cloudMotion as { fraction: number; displacementM: number[]; fromStep: number; toStep: number; alignment: string; fromSources: { role: string; runDir: string; spacingM: number[] }[]; toSources: { role: string; runDir: string; spacingM: number[] }[] };
  expect(motion.alignment).toBe("wind");
  expect(motion.fromStep).toBe(0); expect(motion.toStep).toBe(1);
  expect(motion.fromSources[0].spacingM[0]).toBe(125);
  expect(motion.toSources[0].spacingM[0]).toBe(mode.endsWith("nested") ? 62.5 : 125);
  if (mode.endsWith("nested")) {
    expect(motion.fromSources[0].runDir).toBe("/__nimbus_browser_fixture__");
    expect(motion.toSources[0].runDir).toBe("/__nimbus_browser_fixture__/moving-child");
    const winds = await page.evaluate(() => (window as unknown as { movingWindReads: { runDir: string; step: number; field: string }[] }).movingWindReads.filter(read => read.step === 1));
    expect(winds.map(read => read.field).sort()).toEqual(["u","v","w"]);
    expect(winds.every(read => read.runDir === "/__nimbus_browser_fixture__/moving-child")).toBe(true);
    expect(motion.toSources.find(source => source.role === "context")?.runDir).toBe("/__nimbus_browser_fixture__");
  }
  for (const [axis, value] of [60,30,6].entries()) expect(motion.displacementM[axis]).toBeCloseTo(value, 4);
  expect(capture!.evidence.descriptor.frameIndex).toBeNull();
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await advanceCloudMotionClock(page, 150);
  await page.waitForFunction((fraction) => Number(document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.cloudMotionFraction) > fraction, motion.fraction);
  const during = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  if (during.texture3DUploads !== before.texture3DUploads) console.log(JSON.stringify({ mode, before, during,
    host: await page.locator(".plot3d-overlay-host").evaluate(el => ({ ... (el as HTMLElement).dataset })),
    laterMotion: (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))?.evidence.descriptor.parameters?.cloudMotion }));
  expect(during.texture3DUploads).toBe(before.texture3DUploads);
  expect(during.ipc.startedJobs).toBe(before.ipc.startedJobs);
  await info.attach("motion", { body: JSON.stringify({ motion, before, during }), contentType: "application/json" });
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(false, true));
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-cloud-motion-active", "false");
  await page.waitForFunction((step) => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(step);
  }, 1);
  await resumeCloudMotionClock(page);
  if (mode.endsWith("nested")) {
    await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true, false, 4000));
    await expect(page.getByText("Waiting for the next saved cloud frame", { exact: true })).toBeVisible();
    await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
    await armCloudMotionClock(page);
    await page.evaluate(() => window.__nimbusVtkHarness!.setStep(3));
    await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-cloud-motion-active", "true", activationOptions);
    await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-output-level", "root");
    const reverse = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence.descriptor.parameters!.cloudMotion as { displacementM: number[] };
    for (const [axis,value] of [120,60,12].entries()) expect(reverse.displacementM[axis]).toBeCloseTo(value,4);
    await resumeCloudMotionClock(page);
  }
  await page.locator(".plot3d-control-tab", { hasText: "Flow" }).click();
  await page.getByRole("group", { name: "Updraft / downdraft", exact: true }).getByRole("button").first().click();
  await page.evaluate(() => { window.__nimbusVtkHarness!.setPlayback(true,false); window.__nimbusVtkHarness!.setStep(4); });
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === "4";
  });
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-cloud-motion-active", "false");
  expect((await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence.descriptor.frameIndex).toBe(4);
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await page.evaluate(() => window.__nimbusVtkHarness!.clearFieldCache());
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
  await info.attach("moving-cloud-activation-stream", { body: JSON.stringify({ mode, errors, diagnostics }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

for (const mode of ["grid", "terrain", "nested", "terrain-nested"]) test(`${mode} centroid preview preserves Earth registration`, async ({ page }, info) => {
  if (mode.endsWith("nested")) test.setTimeout(300_000);
  await installCloudMotionClock(page);
  await page.goto(`/harness/index.html?fidelity=presentation&movingCloud=${mode}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "Interpolate cloud", exact: true }).click();
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(true,false,4000));
  await expect(page.getByText("Waiting for the next saved cloud frame", { exact: true })).toBeVisible();
  const inspectTransition = () => page.evaluate(() => ({
    appearance: window.__nimbusVtkHarness!.inspectAppearance(),
    clock: window.__nimbusCloudMotionClock.state(),
    scene: { ...document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset },
    host: { ...document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset },
    resources: window.__nimbusVtkHarness!.snapshot(),
  }));
  // Enabling interpolation also admits parent context; its stale waiting label
  // alone does not establish that frame zero owns the complete source pair.
  await page.waitForFunction(nested => {
    const error = document.querySelector(".volume-render__overlay--error");
    if (error) throw new Error(error.textContent ?? "Centroid source preparation failed");
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset;
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    const evidence = window.__nimbusVtkHarness!.inspectAppearance();
    const motion = evidence?.descriptor.parameters?.cloudMotion as {
      active: boolean; reason: string | null; toStep: number | null;
      toSources: Array<{ role: string; step: number; publicationGeneration: string }>;
    } | undefined;
    const generation = evidence?.descriptor.parameters?.renderedPublicationGeneration;
    return scene?.framePlot === "supercell" && scene.frameLoading === "false" && scene.renderedStep === "0"
      && host?.sceneFrameCommitPending === "false" && host.scenePresentationPending === "false"
      && host.committedSceneGeneration != null && host.committedSceneGeneration === host.requestedSceneGeneration
      && evidence?.descriptor.frameIndex === 0 && motion?.active === false && motion.toStep === 0
      && motion.reason === "Waiting for the next saved cloud frame"
      && motion.toSources.some(source => source.role === "cloud")
      && (!nested || motion.toSources.some(source => source.role === "context"))
      && motion.toSources.every(source => source.step === 0 && source.publicationGeneration === generation);
  }, mode.endsWith("nested"), { timeout: 90_000 });
  await info.attach("centroid-before-transition", { body: JSON.stringify(await inspectTransition()), contentType: "application/json" });
  await armCloudMotionClock(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  try {
    await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-cloud-motion-active","true",
      mode.endsWith("nested") ? { timeout: 90_000 } : undefined);
  } catch (error) {
    const state = await inspectTransition().catch(diagnosticError => ({ diagnosticError: String(diagnosticError) }));
    await info.attach("centroid-transition-failure", { body: JSON.stringify(state), contentType: "application/json" });
    throw error;
  }
  const motion = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence.descriptor.parameters!.cloudMotion as { alignment: string; displacementM: number[] };
  expect(motion.alignment).toBe("cloud-centroid");
  expect(Math.abs(motion.displacementM[0] - 60)).toBeLessThan(5);
  expect(Math.abs(motion.displacementM[1] - 30)).toBeLessThan(5);
  expect(motion.displacementM[2]).toBe(0);
  await resumeCloudMotionClock(page);
  await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes)).toBe(0);
});
