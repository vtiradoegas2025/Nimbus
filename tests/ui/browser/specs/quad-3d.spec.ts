import { test, expect, type Page } from "@playwright/test";
import { writeFile } from "node:fs/promises";

declare global { interface Window { quadShaderCompiles: number } }

async function openQuad(page: Page) {
  await page.addInitScript(() => {
    const counters = window;
    counters.quadShaderCompiles = 0;
    for (const type of [WebGLRenderingContext, WebGL2RenderingContext]) {
      const compile = type.prototype.compileShader;
      type.prototype.compileShader = function(shader) { counters.quadShaderCompiles++; compile.call(this, shader); };
    }
  });
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("nimbus-example://manufactured-supercell-v1", "completed"));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__panes-options").getByRole("button", { name: "Quad", exact: true }).click();
}

async function renderedFrames(page: Page) {
  return page.evaluate(async () => (await window.__nimbusStudioHarness!.capturePaneAppearances())
    .map((capture) => capture?.evidence.descriptor.frameIndex ?? null));
}

for (const mixed of [false, true]) test(`${mixed ? "mixed 3D" : "four volumes"} quad preserves displayed frames and releases companion contexts`, async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.setViewportSize(info.project.name.includes("retina") ? { width: 1512, height: 982 } : { width: 2560, height: 1440 });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openQuad(page);
  const pickers = page.getByRole("combobox", { name: "Companion pane plot type" });
  for (let index = 0; index < 3; index++) await pickers.nth(index).selectOption(mixed ? ["isosurface", "streamlines", "volume"][index] : "volume");
  await expect.poll(() => renderedFrames(page), { timeout: 60_000 }).toEqual([0, 0, 0, 0]);
  await expect(page.locator(".volume-render__overlay--error, .error-boundary")).toHaveCount(0);
  const before = await page.evaluate(() => window.__nimbusStudioHarness!.resourceSnapshot());
  const first = page.locator(".plot-studio__pane--primary");
  if (!mixed) {
    const compiles = await page.evaluate(() => window.quadShaderCompiles);
    const box = (await first.locator(".volume-render__canvas canvas").boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5, { steps: 5 });
    await page.mouse.up();
    await expect(page.locator('[data-volume-quality-preview="true"]')).toHaveCount(0);
    expect(await page.evaluate(() => window.quadShaderCompiles)).toBe(compiles);
  }
  await first.getByRole("button", { name: "Controls", exact: true }).click();
  await first.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await first.getByRole("button", { name: "E", exact: true }).click();
  await first.getByRole("button", { name: "Hide controls", exact: true }).click();
  await expect.poll(async () => page.evaluate(async () => {
    const captures = await window.__nimbusStudioHarness!.capturePaneAppearances();
    const authority = captures[0]?.evidence.camera;
    return captures.every((capture) => JSON.stringify(capture?.evidence.camera.position) === JSON.stringify(authority?.position)
      && JSON.stringify(capture?.evidence.camera.focalPoint) === JSON.stringify(authority?.focalPoint));
  })).toBe(true);
  await page.evaluate(() => window.__nimbusStudioHarness!.setFieldReadDelay(350));
  await page.getByRole("slider", { name: "Selected frame", exact: true }).fill("1");
  await expect.poll(() => renderedFrames(page), { timeout: 60_000 }).toEqual([1, 1, 1, 1]);
  const captures = await page.evaluate(() => window.__nimbusStudioHarness!.capturePaneAppearances());
  for (const [index, capture] of captures.entries()) {
    expect(capture).not.toBeNull();
    if (capture?.evidence.descriptor.plot === "streamlines") {
      const paths = capture.evidence.layers.filter(layer => layer.kind === "actor").slice(0, 2);
      expect(paths).toHaveLength(2);
      expect(paths.every(path => path.visible && path.scalarVisibility)).toBe(true);
    }
    await writeFile(info.outputPath(`pane-${index}.png`), Buffer.from(capture!.pngDataUri.split(",")[1], "base64"));
  }
  await page.screenshot({ path: info.outputPath("quad.png") });
  if (mixed) {
    const streamlines = page.locator(".plot-studio__pane").nth(2);
    await streamlines.getByRole("button", { name: "Controls", exact: true }).click();
    await streamlines.locator(".plot3d-control-tab", { hasText: "Field" }).click();
    for (const label of ["Uniform", "Magnitude"]) {
      await streamlines.getByRole("button", { name: label, exact: true }).click();
      await expect.poll(async () => page.evaluate(async () => {
        const capture = (await window.__nimbusStudioHarness!.capturePaneAppearances())[2];
        return capture?.evidence.layers.filter(layer => layer.kind === "actor").slice(0, 2).map(layer => layer.scalarVisibility);
      })).toEqual([label === "Magnitude", label === "Magnitude"]);
      await expect(streamlines.locator(".plot3d-colorbar__title", { hasText: "Speed" })).toHaveCount(label === "Magnitude" ? 1 : 0);
    }
    await streamlines.getByRole("button", { name: "Hide controls", exact: true }).click();
  }
  const geometry = await page.locator(".volume-render__canvas canvas").evaluateAll((elements) => elements.map((element) => {
    const canvas = element as HTMLCanvasElement, box = canvas.getBoundingClientRect();
    return { width: box.width, height: box.height, backing: [canvas.width, canvas.height] };
  }));
  expect(geometry).toHaveLength(4);
  for (const box of geometry) { expect(box.width).toBeGreaterThan(200); expect(box.height).toBeGreaterThan(150); }
  const bar = (await page.locator(".plot-studio__frame-bar").boundingBox())!;
  expect(bar.y + bar.height).toBeLessThanOrEqual(page.viewportSize()!.height + 1);
  const after = await page.evaluate(() => window.__nimbusStudioHarness!.resourceSnapshot());
  expect(after.contextLosses).toBe(0);
  await info.attach("quad-resources", { body: JSON.stringify({ before, after, geometry }), contentType: "application/json" });
  await page.getByRole("button", { name: "Exit split", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__nimbusStudioHarness!.resourceSnapshot().webglContexts)).toBe(1);
  expect(errors).toEqual([]);
});

test("volume capture reports the displayed frame while a replacement is loading", async ({ page }) => {
  await openQuad(page);
  await expect.poll(() => renderedFrames(page), { timeout: 60_000 }).toEqual([0, 0, 0, 0]);
  await page.evaluate(() => window.__nimbusStudioHarness!.setFieldReadDelay(3000));
  // The native fixture supports delayed reads; switch all panes onto it first.
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("/__nimbus_browser_fixture__/live", "running"));
  await expect.poll(() => renderedFrames(page), { timeout: 60_000 }).toEqual([0, 0, 0, 0]);
  await page.evaluate(() => window.__nimbusStudioHarness!.clearFieldCache());
  await page.getByRole("slider", { name: "Selected frame", exact: true }).fill("1");
  const frames = await page.evaluate(async () => (await window.__nimbusStudioHarness!.capturePaneAppearances()).map(capture => ({
    rendered: capture?.evidence.descriptor.frameIndex,
    requested: capture?.evidence.descriptor.parameters?.requestedFrameIndex,
  })));
  expect(frames.every(frame => (frame.rendered === 0 || frame.rendered == null) && frame.requested === 1), JSON.stringify(frames)).toBe(true);
  await expect.poll(() => renderedFrames(page), { timeout: 60_000 }).toEqual([1, 1, 1, 1]);
});
