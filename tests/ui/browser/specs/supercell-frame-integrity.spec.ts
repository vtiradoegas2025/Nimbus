import { test, expect, type Page } from "@playwright/test";

async function ready(page: Page) {
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.framePlot === "supercell" && scene.dataset.frameLoading === "false"
      && document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.scenePresentationPending === "false";
  }, undefined, { timeout: 90_000 });
}

async function toggle(page: Page, tab: string, group: string) {
  await page.locator(".plot3d-control-tab", { hasText: tab }).click();
  await page.getByRole("group", { name: group, exact: true }).getByRole("button").first().click();
  await ready(page);
}

async function fullQuality(page: Page) {
  await expect(page.locator(".plot3d-control-notice")).toHaveAttribute("aria-hidden", "true");
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}

test("joint supercell layers keep a common scale through persisted settings and live scale edits", async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem("nimbus-studio:3d-local:supercell:scene-scale:v1", JSON.stringify({ xy: 2, z: 4 }));
  });
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  await toggle(page, "Precip", "Precipitation");
  const original = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
  await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  for (const [xy, z] of [[1, 1], [.5, 3], [2, 4], [1, 1]]) {
    for (const [label, value] of [["Horizontal scene scale", xy], ["Vertical scene scale", z]] as const) {
      await page.getByRole("slider", { name: label, exact: true }).evaluate((element, value) => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(element, String(value));
        element.dispatchEvent(new Event("input", { bubbles: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
      }, value);
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      await expect(page.locator(".error-boundary")).toHaveCount(0);
    }
    await ready(page);
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    expect(capture?.evidence.descriptor.parameters?.jointVolumeLayers).toBe(2);
    expect((await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs())).fingerprints).toEqual(original.fingerprints);
  }
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await ready(page);
  expect(errors).toEqual([]);
});

test("precipitation species reentry preserves suspended cloud values and releases resources", async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  const cloud = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints[0]);
  await toggle(page, "Precip", "Precipitation");
  for (const mode of ["total", "qr", "qg", "qh", "qr", "total", "qg"]) {
    await page.locator("#supercell-controls-precip select").selectOption(mode);
    await ready(page);
    const inputs = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
    expect(inputs.fingerprints[0], `${mode} must not alter suspended condensate`).toBe(cloud);
    expect(inputs.volumes).toBe(2);
    await info.attach(mode, { body: JSON.stringify(inputs), contentType: "application/json" });
  }
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes)).toBe(0);
  expect(errors).toEqual([]);
});

test("forced redraw retains the complete previous frame while diagnostic extraction is delayed", async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const send = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (this: Worker, ...args: Parameters<Worker["postMessage"]>) {
      if (args[0]?.kind === "extract") setTimeout(() => send.apply(this, args), 1200);
      else send.apply(this, args);
    } as Worker["postMessage"];
  });
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  await toggle(page, "Flow", "Updraft / downdraft");
  await toggle(page, "Vortex", "Vortex core");
  await toggle(page, "Radar", "Radar floor");
  const beforeCapture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(1));
  await page.waitForFunction(() => {
    const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    return data?.sceneFrameCommitPending === "true" && data.scenePresentationPending === "true"
      && (data.flowContourLoading === "true" || data.vortexContourLoading === "true");
  }, undefined, { timeout: 30_000 });
  const pending = await page.evaluate(async () => {
    const capture = await window.__nimbusVtkHarness!.captureAppearance();
    return { capture: capture!.evidence, png: capture!.pngDataUri, inputs: window.__nimbusVtkHarness!.inspectVolumeInputs() };
  });
  expect(pending.inputs).toEqual(before);
  expect(pending.capture.layers).toEqual(beforeCapture!.evidence.layers);
  const pixels = await page.evaluate(([a, b]) => window.__nimbusVtkHarness!.compareAppearance(a, b),
    [beforeCapture!.pngDataUri, pending.png]);
  expect(pixels.rgbRmse).toBeLessThan(0.00001);
  await ready(page);
  const after = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
  expect(after.fingerprints).not.toEqual(before.fingerprints);
  await info.attach("frame-commit", { body: JSON.stringify({ before, pending, after }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("same-frame threshold replacements retain all committed fields until the complete scene is ready", async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.addInitScript(() => {
    const send = Worker.prototype.postMessage;
    const state = { delay: false };
    Object.assign(window, { sceneThresholdDelay: state });
    Worker.prototype.postMessage = function(this: Worker, ...args: Parameters<Worker["postMessage"]>) {
      if (state.delay && (args[0]?.kind === "extract" || args[0]?.kind === "extract-contours")) {
        setTimeout(() => send.apply(this, args), 1200);
      } else send.apply(this, args);
    } as Worker["postMessage"];
  });
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  await toggle(page, "Precip", "Precipitation");
  await toggle(page, "Flow", "Updraft / downdraft");
  await toggle(page, "Vortex", "Vortex core");
  await toggle(page, "Radar", "Radar floor");
  await fullQuality(page);
  const before = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs(),
    disclosure: document.querySelector(".plot3d-badge")?.textContent,
    generation: document.querySelector<HTMLElement>(".plot3d-overlay-host")!.dataset.committedSceneGeneration }));
  await page.evaluate(() => { (window as unknown as { sceneThresholdDelay: { delay: boolean } }).sceneThresholdDelay.delay = true; });
  await page.locator(".plot3d-control-tab", { hasText: "Flow" }).click();
  await page.getByRole("group", { name: "|w| threshold", exact: true }).getByRole("slider").fill("12");
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-scene-presentation-pending", "true");
  await fullQuality(page);
  const pending = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs(),
    disclosure: document.querySelector(".plot3d-badge")?.textContent,
    generation: document.querySelector<HTMLElement>(".plot3d-overlay-host")!.dataset.committedSceneGeneration }));
  expect(pending.scalars).toEqual(before.scalars);
  expect(pending.capture.layers).toEqual(before.capture.layers);
  expect(pending.capture.camera).toEqual(before.capture.camera);
  expect(pending.generation).toBe(before.generation);
  expect(pending.disclosure).toBe(before.disclosure);
  await ready(page);
  await expect(page.locator(".plot3d-overlay-host")).not.toHaveAttribute("data-committed-scene-generation", before.generation!);
  await info.attach("same-frame-transaction", { body: JSON.stringify({ before, pending }), contentType: "application/json" });
});

test("required-field failures preserve the complete scene and a later scrub commits only the latest frame", async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  await toggle(page, "Precip", "Precipitation");
  await toggle(page, "Flow", "Updraft / downdraft");
  await toggle(page, "Vortex", "Vortex core");
  await toggle(page, "Radar", "Radar floor");
  const before = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs() }));
  await page.evaluate(() => {
    window.__nimbusVtkHarness!.clearFieldCache();
    const internal = (window as unknown as { __TAURI_INTERNALS__: { invoke: (command: string, args?: Record<string, unknown>, options?: unknown) => Promise<unknown> } }).__TAURI_INTERNALS__;
    const invoke = internal.invoke;
    internal.invoke = async (command, args, options) => {
      if (command === "read_field_window" && args?.fieldName === "w" && args.stepIndex === 1) {
        throw { code: "io", message: "Injected required vertical velocity read failure" };
      }
      return invoke(command, args, options);
    };
    window.__nimbusVtkHarness!.setStep(1);
  });
  await expect(page.locator(".volume-render__overlay--error")).toContainText("Injected required vertical velocity read failure", { timeout: 30_000 });
  const failed = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs() }));
  expect(failed.scalars).toEqual(before.scalars);
  expect(failed.capture.layers).toEqual(before.capture.layers);
  expect(failed.capture.descriptor.frameIndex).toBe(before.capture.descriptor.frameIndex);
  expect(failed.capture.descriptor.parameters!.floorRequestIdentity).toBe(before.capture.descriptor.parameters!.floorRequestIdentity);
  expect(failed.capture.descriptor.parameters!.floorScalarRange).toEqual(before.capture.descriptor.parameters!.floorScalarRange);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(2));
  await ready(page);
  await expect(page.locator('[data-testid="scene"]')).toHaveAttribute("data-rendered-step", "2");
  const recovered = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(recovered!.evidence.descriptor.frameIndex).toBe(2);
  expect(recovered!.evidence.descriptor.parameters!.floorSourceField).toBe(before.capture.descriptor.parameters!.floorSourceField);
  expect(recovered!.evidence.descriptor.parameters!.floorRequestIdentity).toEqual(expect.stringContaining("|2|browser-fixture-2|"));
  await info.attach("required-failure-transaction", { body: JSON.stringify({ before, failed, recovered: recovered!.evidence }), contentType: "application/json" });
});

test("same-frame floor definitions replace committed source identity and values together", async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const send = Worker.prototype.postMessage;
    const held: Array<() => void> = [];
    const state = { delay: false, fail: false, held: 0, release: () => {
      state.delay = false;
      held.splice(0).forEach(send => send());
    } };
    Object.assign(window, { sceneDefinitionDelay: state });
    Worker.prototype.postMessage = function(this: Worker, ...args: Parameters<Worker["postMessage"]>) {
      if (state.fail && args[0]?.definition?.id === "theta_prime") {
        throw new Error("Injected floor definition evaluation failure");
      }
      if (state.delay && args[0]?.definition?.id === "theta_prime") {
        state.held += 1;
        held.push(() => send.apply(this, args));
      } else send.apply(this, args);
    } as Worker["postMessage"];
  });
  await page.goto("/harness/index.html?floorDefinitionProbe=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await ready(page);
  await toggle(page, "Precip", "Precipitation");
  await toggle(page, "Flow", "Updraft / downdraft");
  await toggle(page, "Vortex", "Vortex core");
  await toggle(page, "Radar", "Radar floor");
  await page.getByRole("group", { name: "Floor field", exact: true }).getByRole("combobox").selectOption("theta_prime");
  await ready(page);
  await fullQuality(page);
  const before = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs(),
    generation: document.querySelector<HTMLElement>(".plot3d-overlay-host")!.dataset.committedSceneGeneration }));
  const beforeParameters = before.capture.descriptor.parameters!;
  expect(beforeParameters.floorSourceField).toBe("theta_prime");
  expect(beforeParameters.floorRequestIdentity).toEqual(expect.stringContaining('"gain",2'));
  const beforeRange = beforeParameters.floorScalarRange as number[];
  const beforeDisplayRange = beforeParameters.floorDisplayRange as number[];
  expect(beforeRange.every(Number.isFinite)).toBe(true);
  expect(beforeDisplayRange.every(Number.isFinite)).toBe(true);
  expect(beforeRange[1]! - beforeRange[0]!).toBeGreaterThan(0);
  await page.evaluate(() => {
    (window as unknown as { sceneDefinitionDelay: { delay: boolean } }).sceneDefinitionDelay.delay = true;
    window.__nimbusVtkHarness!.setExpressionMultiplier(3);
  });
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-scene-presentation-pending", "true");
  await page.waitForFunction(() => (window as unknown as { sceneDefinitionDelay: { held: number } }).sceneDefinitionDelay.held > 0);
  const pending = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs(),
    generation: document.querySelector<HTMLElement>(".plot3d-overlay-host")!.dataset.committedSceneGeneration }));
  expect(pending.scalars).toEqual(before.scalars);
  expect(pending.capture.layers).toEqual(before.capture.layers);
  expect(pending.capture.camera).toEqual(before.capture.camera);
  expect(pending.capture.descriptor.parameters!.floorRequestIdentity).toBe(beforeParameters.floorRequestIdentity);
  expect(pending.capture.descriptor.parameters!.floorScalarRange).toEqual(beforeRange);
  expect(pending.capture.descriptor.parameters!.floorDisplayRange).toEqual(beforeDisplayRange);
  expect(pending.generation).toBe(before.generation);
  await page.evaluate(() => (window as unknown as { sceneDefinitionDelay: { release: () => void } }).sceneDefinitionDelay.release());
  await ready(page);
  const after = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs(),
    generation: document.querySelector<HTMLElement>(".plot3d-overlay-host")!.dataset.committedSceneGeneration }));
  const afterParameters = after.capture.descriptor.parameters!;
  expect(afterParameters.floorRequestIdentity).not.toBe(beforeParameters.floorRequestIdentity);
  expect(afterParameters.floorRequestIdentity).toEqual(expect.stringContaining('"gain",3'));
  const afterRange = afterParameters.floorScalarRange as number[];
  const afterDisplayRange = afterParameters.floorDisplayRange as number[];
  for (let axis = 0; axis < 2; axis += 1) {
    expect(afterRange[axis]).toBeCloseTo(beforeRange[axis]! * 1.5, 6);
    expect(afterDisplayRange[axis]).toBeCloseTo(beforeDisplayRange[axis]! * 1.5, 6);
  }
  expect(after.scalars).toEqual(before.scalars);
  expect(after.generation).not.toBe(before.generation);
  expect(after.capture.descriptor.frameIndex).toBe(before.capture.descriptor.frameIndex);
  await page.evaluate(() => {
    (window as unknown as { sceneDefinitionDelay: { fail: boolean } }).sceneDefinitionDelay.fail = true;
    window.__nimbusVtkHarness!.setExpressionMultiplier(4);
  });
  await expect(page.locator(".volume-render__overlay--error")).toContainText("Injected floor definition evaluation failure", { timeout: 30_000 });
  const failed = await page.evaluate(async () => ({ capture: (await window.__nimbusVtkHarness!.captureAppearance())!.evidence,
    scalars: window.__nimbusVtkHarness!.inspectVolumeInputs() }));
  expect(failed.scalars).toEqual(after.scalars);
  expect(failed.capture.layers).toEqual(after.capture.layers);
  expect(failed.capture.camera).toEqual(after.capture.camera);
  expect(failed.capture.descriptor.parameters!.floorRequestIdentity).toBe(afterParameters.floorRequestIdentity);
  expect(failed.capture.descriptor.parameters!.floorScalarRange).toEqual(afterRange);
  expect(failed.capture.descriptor.parameters!.floorDisplayRange).toEqual(afterDisplayRange);
  await page.evaluate(() => {
    (window as unknown as { sceneDefinitionDelay: { fail: boolean } }).sceneDefinitionDelay.fail = false;
    window.__nimbusVtkHarness!.setExpressionMultiplier(3);
  });
  await ready(page);
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-committed-scene-generation", after.generation!);
  await info.attach("source-definition-transaction", { body: JSON.stringify({ before, pending, after, failed }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("missing enabled layer sources preserve the previous scene and permit explicit recovery", async ({ page }, info) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => {
    const internal = (window as unknown as { __TAURI_INTERNALS__: { invoke: (command: string, args?: Record<string, unknown>, options?: unknown) => Promise<unknown> } }).__TAURI_INTERNALS__;
    const invoke = internal.invoke;
    internal.invoke = async (command, args, options) => {
      const value = await invoke(command, args, options);
      if (command !== "list_step_dirs" || args?.runDir !== "/__nimbus_browser_fixture__/missing-owner-members" || !Array.isArray(value)) return value;
      return value.map((step: { index: number; fields: string[]; fieldUnits?: Record<string, string> }) => ({ ...step,
        fields: [...step.fields.filter(field => !(step.index === 1 && field === "w")
          && !(step.index === 2 && (field === "vorticity_z" || field === "p_prime"))
          && !(step.index === 3 && field === "qr")), ...(step.index === 2 ? ["p_prime"] : [])],
        fieldUnits: { ...step.fieldUnits, p_prime: "Pa" },
      }));
    };
    window.__nimbusStudioHarness!.setRun("/__nimbus_browser_fixture__/missing-owner-members", "completed");
  });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Rendered supercell", exact: true }).click();
  const scene = page.locator(".plot3d-overlay-host");
  const studioReady = async () => {
    await page.waitForFunction(() => {
      const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
      return data?.scenePresentationPending === "false"
        && data.committedSceneGeneration != null && data.committedSceneGeneration === data.requestedSceneGeneration;
    }, undefined, { timeout: 90_000 });
  };
  const layer = async (tab: string, group: string) => {
    const controlTab = page.locator(".plot3d-control-tab", { hasText: tab });
    if (await controlTab.getAttribute("aria-expanded") !== "true") await controlTab.click();
    const button = page.getByRole("group", { name: group, exact: true }).getByRole("button").first();
    await expect(button).toBeEnabled();
    await button.click();
    await studioReady();
  };
  const capture = () => page.evaluate(async () => {
    const capture = (await window.__nimbusStudioHarness!.capturePaneAppearances())[0];
    if (!capture) throw new Error("Missing committed Supercell capture");
    const scalars = window.__nimbusStudioHarness!.inspectPrimaryVolumeScalarImages().map(image => {
      const words = new Uint32Array(new Float32Array(image.values).buffer);
      let hash = 2166136261;
      for (const value of words) hash = Math.imul(hash ^ value, 16777619);
      return { shape: image.shape, origin: image.origin, spacing: image.spacing, hash: hash >>> 0 };
    });
    return { evidence: capture.evidence, scalars, disclosure: document.querySelector(".plot3d-badge")?.textContent };
  });
  await studioReady();
  await layer("Precip", "Precipitation");
  await page.getByRole("group", { name: "Precip field", exact: true }).getByRole("combobox").selectOption("qr");
  await studioReady();
  await layer("Flow", "Updraft / downdraft");
  await layer("Vortex", "Vortex core");
  await layer("Radar", "Radar floor");
  await fullQuality(page);
  let committed = await capture();
  expect(committed.evidence.descriptor.frameIndex).toBe(0);
  const audits: unknown[] = [];
  const cases = [
    { index: 1, error: "Flow unavailable", tab: "Flow", group: "Updraft / downdraft" },
    { index: 2, error: "Vortex core unavailable", tab: "Vortex", group: "Vortex core" },
    { index: 3, error: "does not publish selected qr", tab: "Precip", group: "Precipitation" },
  ];
  for (const missing of cases) {
    await page.getByRole("slider", { name: "Selected frame", exact: true }).fill(String(missing.index));
    await expect(page.locator(".volume-render__overlay--error")).toContainText(missing.error, { timeout: 30_000 });
    await expect(scene).toHaveAttribute("data-scene-presentation-pending", "true");
    const failed = await capture();
    expect(failed.scalars).toEqual(committed.scalars);
    expect(failed.disclosure).toBe(committed.disclosure);
    expect(failed.evidence.layers).toEqual(committed.evidence.layers);
    expect(failed.evidence.camera).toEqual(committed.evidence.camera);
    expect(failed.evidence.descriptor.frameIndex).toBe(committed.evidence.descriptor.frameIndex);
    expect(failed.evidence.descriptor.parameters!.floorRequestIdentity).toBe(committed.evidence.descriptor.parameters!.floorRequestIdentity);
    if (missing.index === 2) {
      await page.locator(".plot3d-control-tab", { hasText: "Vortex" }).click();
      const selector = page.getByRole("group", { name: "Core field", exact: true }).getByRole("combobox");
      await expect(selector).toHaveValue("vorticity_z");
      await expect(selector.locator('option[value="vorticity_z"]')).toBeDisabled();
      await expect(selector.locator('option[value="p_prime"]')).toBeEnabled();
      await layer(missing.tab, missing.group);
    } else if (missing.index === 3) {
      await page.locator(".plot3d-control-tab", { hasText: "Precip" }).click();
      const selector = page.getByRole("group", { name: "Precip field", exact: true }).getByRole("combobox");
      await expect(selector).toHaveValue("qr");
      await expect(selector.locator('option[value="qr"]')).toBeDisabled();
      await selector.selectOption("qg");
      await studioReady();
    } else await layer(missing.tab, missing.group);
    const recovered = await capture();
    expect(recovered.evidence.descriptor.frameIndex).toBe(missing.index);
    expect(recovered.evidence.descriptor.parameters![missing.index === 1 ? "showUpdraft" : missing.index === 2 ? "showVortexCore" : "precipMode"])
      .toBe(missing.index === 3 ? "qg" : false);
    audits.push({ missing, before: committed, failed, recovered });
    committed = recovered;
  }
  await info.attach("missing-owner-recovery", { body: JSON.stringify(audits), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("Scientific saved-frame playback excludes temporal previews and their wind reads", async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Rendered supercell", exact: true }).click();
  const scene = page.locator(".plot3d-overlay-host");
  const settled = () => page.waitForFunction(() => {
    const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    return data?.scenePresentationPending === "false"
      && data.committedSceneGeneration === data.requestedSceneGeneration;
  }, undefined, { timeout: 90_000 });
  const fidelity = async (mode: "Scientific" | "Presentation") => {
    const menu = page.locator(".plot-studio__view-menu");
    if (!(await menu.evaluate(element => (element as HTMLDetailsElement).open))) await menu.locator("summary").click();
    await menu.getByRole("group", { name: "3D rendering fidelity", exact: true }).getByRole("button", { name: mode, exact: true }).click();
    if (await menu.evaluate(element => (element as HTMLDetailsElement).open)) await menu.locator("summary").click();
    await settled();
  };
  await settled();
  await fidelity("Scientific");
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  const motion = page.getByRole("group", { name: "Cloud motion", exact: true });
  await expect(motion.getByRole("button", { name: "Interpolate cloud", exact: true })).toBeDisabled();
  await expect(motion.getByRole("button", { name: "Wind-informed", exact: true })).toBeDisabled();
  await expect(motion).toContainText("temporal previews infer data and require Presentation mode");
  await expect(scene).toHaveAttribute("data-cloud-motion-enabled", "false");
  await expect(scene).toHaveAttribute("data-cloud-wind-requested", "false");
  await fidelity("Presentation");
  await expect(motion.getByRole("button", { name: "Wind-informed", exact: true })).toBeEnabled();
  await motion.getByRole("button", { name: "Wind-informed", exact: true }).click();
  await expect(scene).toHaveAttribute("data-cloud-motion-enabled", "true");
  await expect(scene).toHaveAttribute("data-cloud-wind-preview-enabled", "true");
  await settled();
  await fidelity("Scientific");
  await expect(scene).toHaveAttribute("data-cloud-motion-enabled", "false");
  await expect(scene).toHaveAttribute("data-cloud-wind-preview-enabled", "false");
  await expect(scene).toHaveAttribute("data-cloud-wind-requested", "false");
  await expect(motion.getByRole("button", { name: "Saved frames", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.evaluate(() => {
    window.__nimbusStudioHarness!.clearFieldCache();
    const state = { fields: [] as string[] };
    Object.assign(window, { scientificReadAudit: state });
    const internal = (window as unknown as { __TAURI_INTERNALS__: { invoke: (command: string, args?: Record<string, unknown>, options?: unknown) => Promise<unknown> } }).__TAURI_INTERNALS__;
    const invoke = internal.invoke;
    internal.invoke = async (command, args, options) => {
      if (command === "read_field_window" || command === "read_field") state.fields.push(String(args?.fieldName));
      return invoke(command, args, options);
    };
  });
  const frame = page.getByRole("slider", { name: "Selected frame", exact: true });
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await expect.poll(async () => Number(await frame.inputValue()), { timeout: 20_000 }).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  const selectedFrame = Number(await frame.inputValue());
  await settled();
  const actual = await page.evaluate(async () => ({
    fields: (window as unknown as { scientificReadAudit: { fields: string[] } }).scientificReadAudit.fields,
    capture: (await window.__nimbusStudioHarness!.capturePaneAppearances())[0]!.evidence,
  }));
  expect(actual.fields.some(field => ["u", "v", "w", "rho"].includes(field))).toBe(false);
  expect(actual.fields).toContain("cloud_condensate");
  expect(actual.capture.descriptor.frameIndex).toBe(selectedFrame);
  expect(actual.capture.descriptor.fidelityMode).toBe("scientific");
  expect(actual.capture.descriptor.parameters!.cloudInterpolationEnabled).toBe(false);
  await fidelity("Presentation");
  await expect(scene).toHaveAttribute("data-cloud-motion-enabled", "false");
  await expect(motion.getByRole("button", { name: "Saved frames", exact: true })).toHaveAttribute("aria-pressed", "true");
  await info.attach("scientific-temporal-contract", { body: JSON.stringify(actual), contentType: "application/json" });
  expect(errors).toEqual([]);
});
