import { expect, test } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test("ordinary volume export includes the displayed colorbar and preserves its signed palette", async ({ page }, info) => {
  test.setTimeout(90_000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.getByRole("button", { name: /Volume/ }).click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  await page.getByRole("button", { name: "Cool→Warm", exact: true }).click();
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await expect(page.locator(".plot3d-colorbar")).toBeVisible();
  const result = await page.evaluate(async () => {
    const before = window.__nimbusVtkHarness!.inspectVolumeInputs();
    const png = await window.__nimbusVtkHarness!.captureSavedFrame();
    const image = new Image();
    await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = reject; image.src = png; });
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0);
    const host = document.querySelector(".volume-render__canvas")!.getBoundingClientRect();
    const bar = document.querySelector(".plot3d-colorbar__bar")!.getBoundingClientRect();
    const scale = Math.min(canvas.width / host.width, canvas.height / host.height);
    const left = (canvas.width - host.width * scale) / 2;
    const top = (canvas.height - host.height * scale) / 2;
    const sample = (fraction: number) => Array.from(ctx.getImageData(
      Math.floor(left + (bar.x - host.x + bar.width / 2) * scale),
      Math.floor(top + (bar.y - host.y + fraction * bar.height) * scale), 1, 1).data);
    return { png, high: sample(0.05), low: sample(0.95), width: canvas.width, height: canvas.height,
      before, after: window.__nimbusVtkHarness!.inspectVolumeInputs() };
  });
  await writeFile(info.outputPath("volume-export-with-legends.png"), Buffer.from(result.png.split(",")[1], "base64"));
  expect(result.high[0]).toBeGreaterThan(result.high[2] + 60);
  expect(result.low[2]).toBeGreaterThan(result.low[0] + 60);
  expect(result.after).toEqual(result.before);
  expect(result.width).toBe(960);
  expect(result.height).toBe(620);
});

test("ordinary supercell export restores full quality for every field and includes discrete legends during held preview", async ({ page }, info) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); window.__nimbusVtkHarness!.setStep(1); });
  const ready = async () => page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.framePlot === "supercell" && scene.dataset.frameLoading === "false"
      && document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.scenePresentationPending === "false";
  }, undefined, { timeout: 45_000 });
  await ready();
  for (const [tab, group] of [["Precip", "Precipitation"], ["Vortex", "Vortex core"]]) {
    await page.locator(".plot3d-control-tab", { hasText: tab }).click();
    await page.getByRole("group", { name: group, exact: true }).getByRole("button").first().click();
    await ready();
  }
  await page.locator('.plot3d-control-tab[aria-expanded="true"]').click();
  await expect(page.locator(".plot3d-control-notice")).toHaveAttribute("aria-hidden", "true");
  await page.waitForFunction(() => {
    const canvas = document.querySelector<HTMLCanvasElement>(".volume-render__canvas canvas");
    if (!canvas) return false;
    const bounds = canvas.getBoundingClientRect();
    return bounds.width > 0 && Math.abs(canvas.height - bounds.height * canvas.width / bounds.width) < 2;
  });
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(page.locator(".plot3d-discrete-legend__title", { hasText: "Precipitation volume" })).toBeVisible();
  await expect(page.locator(".plot3d-discrete-legend__title", { hasText: "vorticity_z vortex hierarchy" })).toBeVisible();
  const baseline = await page.evaluate(async () => ({ png: await window.__nimbusVtkHarness!.captureSavedFrame(),
    draw: window.__nimbusVtkHarness!.inspectLastSavedFrame(),
    inputs: window.__nimbusVtkHarness!.inspectVolumeInputs(), snapshot: window.__nimbusVtkHarness!.snapshot(),
    overlays: window.__nimbusVtkHarness!.inspectExportOverlays() }));
  expect(baseline.inputs.fingerprints.length).toBeGreaterThanOrEqual(2);
  const paintTexts = baseline.overlays?.overlays.flatMap(overlay => overlay.paint.filter(item => item.kind === "text").map(item => item.text)) ?? [];
  expect(paintTexts).toContain("Precipitation volume");
  expect(paintTexts).toContain("vorticity_z vortex hierarchy");
  const canvas = page.locator(".volume-render__canvas");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  let candidate: string;
  try {
    await expect(page.locator(".plot3d-control-notice")).toHaveClass(/is-visible/);
    await expect(page.locator(".plot3d-control-notice")).toContainText("Preview while moving");
    candidate = await page.evaluate(() => window.__nimbusVtkHarness!.captureSavedFrame());
    await expect(page.locator(".plot3d-control-notice")).toHaveClass(/is-visible/);
  } finally { await page.mouse.up(); }
  const result = await page.evaluate(async ({ reference, captured }) => {
    const image = new Image();
    await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = reject; image.src = captured; });
    const surface = document.createElement("canvas"); surface.width = image.naturalWidth; surface.height = image.naturalHeight;
    const context = surface.getContext("2d")!; context.drawImage(image, 0, 0);
    const host = document.querySelector(".volume-render__canvas")!.getBoundingClientRect();
    const scale = Math.min(surface.width / host.width, surface.height / host.height);
    const left = (surface.width - host.width * scale) / 2, top = (surface.height - host.height * scale) / 2;
    const swatches = Array.from(document.querySelectorAll(".plot3d-discrete-legend__swatch"), swatch => {
      const rect = swatch.getBoundingClientRect();
      const paint = document.createElement("canvas"); paint.width = 1; paint.height = 1;
      const ctx = paint.getContext("2d")!; ctx.fillStyle = getComputedStyle(swatch).backgroundColor; ctx.fillRect(0, 0, 1, 1);
      return { expected: Array.from(ctx.getImageData(0, 0, 1, 1).data), actual: Array.from(context.getImageData(
        Math.floor(left + (rect.x - host.x + rect.width / 2) * scale),
        Math.floor(top + (rect.y - host.y + rect.height / 2) * scale), 1, 1).data) };
    });
    return { swatches, pixels: await window.__nimbusVtkHarness!.compareAppearance(reference, captured),
      draw: window.__nimbusVtkHarness!.inspectLastSavedFrame(),
      inputs: window.__nimbusVtkHarness!.inspectVolumeInputs(), snapshot: window.__nimbusVtkHarness!.snapshot() };
  }, { reference: baseline.png, captured: candidate! });
  await writeFile(info.outputPath("supercell-export-settled.png"), Buffer.from(baseline.png.split(",")[1], "base64"));
  await writeFile(info.outputPath("supercell-export-held-preview.png"), Buffer.from(candidate!.split(",")[1], "base64"));
  await info.attach("supercell-export-metadata.json", { body: JSON.stringify({ baseline: { inputs: baseline.inputs, overlays: baseline.overlays, draw: baseline.draw }, result }, null, 2), contentType: "application/json" });
  expect(result.inputs).toEqual(baseline.inputs);
  expect(result.snapshot.texture3DUploads).toBe(baseline.snapshot.texture3DUploads);
  expect(result.snapshot.ipc.startedJobs).toBe(baseline.snapshot.ipc.startedJobs);
  expect(result.swatches.length).toBeGreaterThanOrEqual(2);
  for (const swatch of result.swatches) for (let channel = 0; channel < 3; channel++) expect(Math.abs(swatch.actual[channel] - swatch.expected[channel])).toBeLessThanOrEqual(8);
  expect(result.pixels.meanAbsoluteError).toBe(0);
  expect(errors).toEqual([]);
});
