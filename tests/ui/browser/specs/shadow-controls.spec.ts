import { test, expect } from "@playwright/test";

for (const plot of ["volume", "isosurface", "streamlines", "trajectories", "supercell"] as const) {
  test(`${plot} shadow depth and tint preserve saved data and reset`, async ({ page }, info) => {
    test.setTimeout(90000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("/harness/index.html");
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    await page.evaluate(plot => {
      if (plot === "supercell") window.__nimbusVtkHarness!.setSize(512, 384);
      window.__nimbusVtkHarness!.setPlot(plot); window.__nimbusVtkHarness!.setStep(1);
    }, plot);
    const ready = async () => {
      let previousViewport = "";
      let lastEvidence: unknown;
      await expect.poll(async () => {
        const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
        lastEvidence = capture?.evidence;
        const viewport = JSON.stringify(capture?.evidence.viewport);
        const stable = viewport === previousViewport;
        previousViewport = viewport;
        const parameters = capture?.evidence.descriptor.parameters;
        const fullQuality = plot === "volume" ? parameters?.interactionQuality === "settled"
          : plot === "supercell" ? parameters?.shadowSampleFactor === 4 : true;
        return stable && fullQuality && capture?.evidence.descriptor.plot === plot && capture.evidence.layers.some(layer =>
          layer.visible && (layer.pointCount ?? 0) > 100 && (layer.cellCount ?? 0) > 0);
      }, { timeout: 30000, intervals: [250] }).toBe(true).catch(async (error: unknown) => {
        await info.attach("readiness-evidence", { body: JSON.stringify(lastEvidence ?? null), contentType: "application/json" });
        throw error;
      });
    };
    await ready();
    if (plot === "volume") {
      const volume = page.locator(".plot3d-control-tab", { hasText: "Volume" });
      if (await volume.getAttribute("aria-expanded") !== "true") await volume.click();
      await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("cloud_condensate");
      await ready();
    }
    const scene = page.locator(".plot3d-control-tab", { hasText: "Scene" });
    const expanded = page.locator('.plot3d-control-tab[aria-expanded="true"]');
    if (await expanded.count()) await expanded.click();
    await ready();
    const baseline = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
    const rig = await page.evaluate(() => window.__nimbusVtkHarness!.inspectLighting());
    expect(rig).toHaveLength(3);
    expect(rig.every(light => light.type === "SceneLight")).toBe(true);
    if (plot === "streamlines" || plot === "trajectories") {
      const surfaces = await page.evaluate(() => window.__nimbusVtkHarness!.inspectSurfaceNormals());
      const tubes = surfaces.filter(surface => surface.points > 100 && surface.strips > 0);
      expect(tubes.length).toBeGreaterThan(0);
      expect(tubes.every(surface => surface.normals === surface.points)).toBe(true);
    }
    const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    const fields = plot === "volume" || plot === "supercell"
      ? await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints) : null;
    await scene.click();
    await page.getByRole("slider", { name: "Shadow depth", exact: true }).fill("1.6");
    await scene.click();
    await ready();
    const depthOnly = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
    const depthPixels = await page.evaluate(({ baseline, depth }) => window.__nimbusVtkHarness!.compareAppearance(baseline, depth), { baseline: baseline.pngDataUri, depth: depthOnly.pngDataUri });
    expect(depthPixels.meanAbsoluteError).toBeGreaterThan(0);
    await scene.click();
    await page.getByRole("combobox", { name: "Shadow tint mode" }).selectOption("custom");
    await page.getByLabel("Shadow fill tint", { exact: true }).fill("#4477ff");
    await scene.click();
    await ready();
    const changed = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
    expect(changed.evidence.descriptor.parameters!.lightingControls).toEqual({ preset: "neutral", shadowDepth: 1.6, fillTint: "#4477ff" });
    expect(changed.evidence.descriptor.parameters!.colorRange).toEqual(baseline.evidence.descriptor.parameters!.colorRange);
    expect(changed.evidence.layers.map(layer => [layer.kind,layer.pointCount,layer.cellCount])).toEqual(baseline.evidence.layers.map(layer => [layer.kind,layer.pointCount,layer.cellCount]));
    expect(changed.evidence.viewport).toEqual(baseline.evidence.viewport);
    expect(changed.evidence.camera).toEqual(baseline.evidence.camera);
    const changedPixels = await page.evaluate(({ baseline, changed }) => window.__nimbusVtkHarness!.compareAppearance(baseline, changed), { baseline: baseline.pngDataUri, changed: changed.pngDataUri });
    expect(changedPixels.meanAbsoluteError).toBeGreaterThan(0);
    const tintPixels = await page.evaluate(({ depth, changed }) => window.__nimbusVtkHarness!.compareAppearance(depth, changed), { depth: depthOnly.pngDataUri, changed: changed.pngDataUri });
    expect(tintPixels.meanAbsoluteError).toBeGreaterThan(0);
    expect(changed.evidence.layers.some((layer,index) => layer.visible && (layer.pointCount ?? 0) > 100
      && layer.ambient != null && baseline.evidence.layers[index].ambient != null
      && layer.ambient < baseline.evidence.layers[index].ambient!)).toBe(true);
    if (fields) expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs().fingerprints)).toEqual(fields);
    const after = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(after.ipc.startedJobs).toBe(before.ipc.startedJobs);
    expect(after.texture3DUploads).toBe(before.texture3DUploads);
    await scene.click();
    await page.getByRole("button", { name: "Reset lighting", exact: true }).click();
    await scene.click();
    await ready();
    const reset = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
    expect(reset.evidence.descriptor.parameters!.lightingControls).toEqual({ preset: "neutral", shadowDepth: 1, fillTint: null });
    expect(reset.evidence.viewport).toEqual(baseline.evidence.viewport);
    expect(reset.evidence.camera).toEqual(baseline.evidence.camera);
    const resetPixels = await page.evaluate(({ baseline, reset }) => window.__nimbusVtkHarness!.compareAppearance(baseline, reset), { baseline: baseline.pngDataUri, reset: reset.pngDataUri });
    await info.attach("lighting-controls", { body: JSON.stringify({ before: baseline.evidence, changed: changed.evidence, after: reset.evidence, depthPixels, tintPixels, changedPixels, resetPixels }), contentType: "application/json" });
    expect(resetPixels.meanAbsoluteError).toBe(0);
    expect(errors).toEqual([]);
  });
}

test("shadow settings persist locally and exact-color mode bypasses them", async ({ page }) => {
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await page.getByRole("slider", { name: "Shadow depth" }).fill("1.4");
  await page.getByRole("combobox", { name: "Shadow tint mode" }).selectOption("custom");
  await page.getByLabel("Shadow fill tint", { exact: true }).fill("#ff8866");
  await page.reload();
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); });
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await expect(page.getByRole("slider", { name: "Shadow depth" })).toHaveValue("1.4");
  await expect(page.getByLabel("Shadow fill tint", { exact: true })).toHaveValue("#ff8866");
  const lighting = page.locator(".control-group", { has: page.locator(".control-group__label", { hasText: "3D lighting" }) }).locator("select");
  await lighting.selectOption("flat");
  await expect(page.getByRole("slider", { name: "Shadow depth" })).toBeDisabled();
  await expect(page.getByRole("combobox", { name: "Shadow tint mode" })).toBeDisabled();
  const flat = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
  expect(flat.evidence.layers.filter(layer => layer.kind === "volume" && layer.visible).every(layer => layer.shade === false)).toBe(true);
});

test("unlit scalar views explain inactive shadow controls", async ({ page }) => {
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const volume = page.locator(".plot3d-control-tab", { hasText: "Volume" });
  if (await volume.getAttribute("aria-expanded") !== "true") await volume.click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await expect(page.getByRole("slider", { name: "Shadow depth" })).toBeDisabled();
  await expect(page.getByRole("combobox", { name: "Shadow tint mode" })).toBeDisabled();
  await expect(page.getByText("This field is unlit; shadow controls do not apply.")).toBeVisible();
});

test("shadow settings share across 3D views only when requested", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Rendered supercell", exact: true }).click();
  await page.getByRole("button", { name: "All 3D", exact: true }).click();
  await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await page.getByRole("slider", { name: "Shadow depth" }).fill("1.7");
  await page.getByRole("combobox", { name: "Shadow tint mode" }).selectOption("custom");
  await page.getByLabel("Shadow fill tint", { exact: true }).fill("#ccddee");
  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await page.getByRole("button", { name: "All 3D", exact: true }).click();
  const scene = page.locator(".plot3d-control-tab", { hasText: "Scene" });
  if (await scene.getAttribute("aria-expanded") !== "true") await scene.click();
  await expect(page.getByRole("slider", { name: "Shadow depth" })).toHaveValue("1.7");
  await expect(page.getByLabel("Shadow fill tint", { exact: true })).toHaveValue("#ccddee");
});
