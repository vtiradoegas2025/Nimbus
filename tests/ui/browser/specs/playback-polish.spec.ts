import { expect, test, type Page } from "@playwright/test";

async function ready(page: Page, step: number) {
  await page.waitForFunction(value => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(value);
  }, step, { timeout: 60000 });
}

for (const plot of ["supercell", "streamlines"] as const) {
  test(`${plot} playback retains reference actors and control geometry`, async ({ page }, info) => {
    test.setTimeout(plot === "supercell" ? 240_000 : 120_000);
    await page.addInitScript(() => {
      const state = { extractions: 0 };
      Object.assign(window, { playbackWork: state });
      const send = Worker.prototype.postMessage;
      Worker.prototype.postMessage = function (this: Worker, ...args: Parameters<Worker["postMessage"]>) {
        if (args[0]?.kind === "extract") state.extractions++;
        return send.apply(this, args);
      } as Worker["postMessage"];
    });
    await page.goto("/harness/index.html");
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    await page.evaluate(value => window.__nimbusVtkHarness!.setPlot(value), plot);
    await ready(page, 0);
    await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
    await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.inspectReferences().length)).toBe(2);
    await page.evaluate(() => {
      const baseline = window.__nimbusVtkHarness!.inspectReferences().map(value => value.id);
      const controls = () => Array.from(document.querySelectorAll('.control-group__body input,.control-group__body select')).map(element => {
        const box = element.getBoundingClientRect();
        return [box.x, box.y, box.width, box.height].map(value => Math.round(value));
      });
      const state = { active: true, missing: 0, loadingFlashes: 0, samples: 0, controls: [] as number[][][] };
      Object.assign(window, { playbackAudit: state });
      const tick = () => {
        if (!state.active) return;
        if (JSON.stringify(window.__nimbusVtkHarness!.inspectReferences().map(value => value.id)) !== JSON.stringify(baseline)) state.missing++;
        if (document.body.textContent?.includes("Loading storm fields")) state.loadingFlashes++;
        state.controls.push(controls()); state.samples++;
        requestAnimationFrame(tick);
      };
      window.__nimbusVtkHarness!.setPlayback(true);
      window.__nimbusVtkHarness!.setFieldReadDelay(150);
      requestAnimationFrame(tick);
    });
    const times = [];
    for (const step of [1, 2]) {
      const start = Date.now();
      await page.evaluate(value => window.__nimbusVtkHarness!.setStep(value), step);
      await ready(page, step);
      times.push(Date.now() - start);
    }
    const result = await page.evaluate(() => {
      const state = (window as unknown as { playbackAudit: { active: boolean; missing: number; loadingFlashes: number; samples: number; controls: number[][][] } }).playbackAudit;
      state.active = false;
      return { ...state, extractions: (window as unknown as { playbackWork: { extractions: number } }).playbackWork.extractions };
    });
    expect(result.samples).toBeGreaterThan(0);
    expect(result.missing).toBe(0);
    expect(result.loadingFlashes).toBe(0);
    expect(new Set(result.controls.map(value => JSON.stringify(value))).size).toBe(1);
    if (plot === "streamlines") expect(result.extractions).toBeLessThanOrEqual(4);
    await info.attach("playback", { body: JSON.stringify({ result, times }), contentType: "application/json" });
    await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
    await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes)).toBe(0);
  });
}

test("every selectable isosurface field renders or explains a constant field", async ({ page }, info) => {
  test.setTimeout(150000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setPlot("isosurface"); window.__nimbusVtkHarness!.setStep(1); });
  await ready(page, 1);
  const data = page.locator(".plot3d-control-tab", { hasText: "Data" });
  const surface = page.locator(".plot3d-control-tab", { hasText: "Surface" });
  await data.click();
  const fields = await page.getByRole("combobox", { name: "Field", exact: true }).locator("option").evaluateAll(options => options.map(option => (option as HTMLOptionElement).value).filter(Boolean));
  const evidence = [];
  for (const field of fields) {
    if (await data.getAttribute("aria-expanded") !== "true") await data.click();
    await page.getByRole("combobox", { name: "Field", exact: true }).selectOption(field);
    await ready(page, 1);
    await surface.click();
    const fit = page.getByRole("button", { name: "Fit threshold", exact: true });
    if (field === "uniform_scalar") {
      await expect(fit).toBeDisabled();
      await expect(page.getByText("Field is uniform (min = max) — no isosurface to extract.")).toBeVisible();
      continue;
    }
    await expect(fit).toBeEnabled();
    await fit.click();
    await expect.poll(async () => {
      const c = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      return c?.evidence.descriptor.field === field && c.evidence.layers.some(layer => layer.visible && (layer.pointCount ?? 0) > 100 && (layer.cellCount ?? 0) > 0);
    }, { timeout: 30000 }).toBe(true);
    evidence.push((await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence);
  }
  expect(fields.length).toBeGreaterThan(20);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await info.attach("fields", { body: JSON.stringify(evidence), contentType: "application/json" });
});

async function studio(page: Page) {
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("nimbus-example://manufactured-supercell-v1", "completed"));
  await expect(page.locator(".plot-canvas .gtitle")).toBeVisible();
}

for (const field of ["cloud_condensate", "analytic_smooth_gradient"]) {
  test(`${field} acquires a useful isovalue after uniform startup`, async ({ page }) => {
    await page.goto(`/harness/index.html?uniformAtStart=${field}`);
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("isosurface"));
    await ready(page, 0);
    await page.locator(".plot3d-control-tab", { hasText: "Data" }).click();
    await page.getByRole("combobox", { name: "Field", exact: true }).selectOption(field);
    await expect(page.getByText("Field is uniform (min = max) — no isosurface to extract.")).toBeVisible();
    await page.evaluate(() => { window.__nimbusVtkHarness!.setPlayback(true); window.__nimbusVtkHarness!.setStep(1); });
    await expect.poll(async () => {
      const c = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      return c?.evidence.descriptor.frameIndex === 1 && c.evidence.layers.some(layer => layer.visible && (layer.pointCount ?? 0) > 100 && (layer.cellCount ?? 0) > 0);
    }, { timeout: 30000 }).toBe(true);
    await expect(page.getByText(/Threshold .* is outside this frame/)).toHaveCount(0);
  });
}

test("every bricked isosurface field bootstraps its threshold", async ({ page }, info) => {
  test.setTimeout(180000);
  await page.goto("/harness/index.html?derivedBricks=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("isosurface"));
  await ready(page, 0);
  await page.locator(".plot3d-control-tab", { hasText: "Data" }).click();
  const picker = page.getByRole("combobox", { name: "Field", exact: true });
  const fields = await picker.locator("option").evaluateAll(options => options.map(option => (option as HTMLOptionElement).value).filter(Boolean));
  const evidence = [];
  for (const field of fields) {
    await picker.selectOption(field);
    await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-contour-threshold-field", field, { timeout: 30000 });
    await expect.poll(async () => {
      const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      return capture?.evidence.descriptor.field === field && capture.evidence.layers.some(layer => layer.visible && (layer.pointCount ?? 0) > 100 && (layer.cellCount ?? 0) > 0);
    }, { timeout: 30000 }).toBe(true);
    evidence.push((await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!.evidence);
  }
  expect(fields).toContain("temperature");
  expect(fields).toContain("dense_signed");
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await info.attach("bricked-fields", { body: JSON.stringify(evidence), contentType: "application/json" });
});

test("heatmap zoom and optional smoothing preserve native values and viewport", async ({ page }, info) => {
  await page.setViewportSize({ width: 2560, height: 1440 });
  await studio(page);
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  const canvas = page.locator(".plot-canvas").first();
  await expect.poll(() => canvas.evaluate(el => (el as unknown as { data: { z?: number[][] }[] }).data?.[0]?.z?.length ?? 0)).toBeGreaterThan(1);
  const drag = (await canvas.locator(".nsewdrag").boundingBox())!;
  await page.mouse.move(drag.x + drag.width * 0.4, drag.y + drag.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(drag.x + drag.width * 0.6, drag.y + drag.height * 0.6, { steps: 5 });
  await page.mouse.up();
  await page.waitForTimeout(500);
  const inspect = () => canvas.evaluate(el => {
    const graph = el as unknown as { data: { type: string; z?: number[][]; zmin?: number; zmax?: number }[]; _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[] } } };
    const trace = graph.data.find(t => t.type === "heatmap")!;
    return { z: trace.z, min: trace.zmin, max: trace.zmax, x: graph._fullLayout.xaxis.range, y: graph._fullLayout.yaxis.range };
  });
  const before = await inspect();
  await page.getByRole("button", { name: "Smooth heatmap shading (display only; values unchanged)", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-heatmap-display", "smooth");
  await expect.poll(() => canvas.evaluate(el => (el as unknown as { _fullData: { zsmooth: string }[] })._fullData[0].zsmooth)).toBe("fast");
  expect(await inspect()).toEqual(before);
  await info.attach("smooth-display", { body: await page.screenshot(), contentType: "image/png" });
  await page.getByRole("button", { name: "Restore original shading (smooth display active)", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-heatmap-display", "native");
  expect(await inspect()).toEqual(before);
});

for (const plot of ["Slice", "Vector field", "Composite", "Vertical", "Skew-T", "Hodograph"]) {
  test(`${plot} controls remain stable through frame loading and fit a narrow pane`, async ({ page }, info) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await studio(page);
    if (["Vertical", "Skew-T", "Hodograph"].includes(plot)) await page.getByRole("button", { name: "Soundings", exact: true }).click();
    await page.getByRole("button", { name: plot, exact: true }).click();
    await expect(page.locator(".plot-canvas .gtitle")).toBeVisible();
    await page.evaluate(() => window.__nimbusStudioHarness!.setFieldReadDelay(350));
    const widths = [];
    for (const frame of [1, 2]) {
      await page.getByRole("slider", { name: "Selected frame", exact: true }).fill(String(frame));
      for (let sample = 0; sample < 5; sample++) {
        await page.waitForTimeout(100);
        widths.push(await page.locator(".plot-controls").first().evaluate(el => {
          const panel = el.getBoundingClientRect();
          const controls = Array.from(el.querySelectorAll("input,select,button")).filter(c => c.getClientRects().length > 0);
          return { geometry: controls.map(c => { const r = c.getBoundingClientRect(); return [r.x,r.y,r.width,r.height].map(Math.round); }), overflow: controls.filter(c => { const r=c.getBoundingClientRect(); return r.left<panel.left-1 || r.right>panel.right+1; }).map(c=>c.getAttribute("aria-label") ?? c.textContent) };
        }));
      }
    }
    await info.attach("controls", { body: JSON.stringify(widths), contentType: "application/json" });
    expect(widths.every(value => value.overflow.length === 0)).toBe(true);
    expect(new Set(widths.map(value => JSON.stringify(value.geometry))).size).toBe(1);
    if (["Vertical", "Skew-T", "Hodograph"].includes(plot)) {
      const canvas = page.locator(".plot-canvas");
      await expect(canvas.locator(".scatterlayer path.js-line").first()).toBeVisible();
      const drag = (await canvas.locator(".nsewdrag").boundingBox())!;
      await page.mouse.move(drag.x+drag.width*.3,drag.y+drag.height*.3);await page.mouse.down();
      await page.mouse.move(drag.x+drag.width*.7,drag.y+drag.height*.7,{steps:5});await page.mouse.up();
      await expect(canvas.locator(".scatterlayer path.js-line").first()).toBeVisible();
      await expect(canvas.locator(".main-svg").first()).toBeVisible();
    }
  });
}
