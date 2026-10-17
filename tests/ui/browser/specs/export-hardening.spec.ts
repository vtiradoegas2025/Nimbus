import { expect, test } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test("MP4 start waits for each encoder probe and never reuses stale success", async ({ page }) => {
  await page.goto("/harness/index.html?mode=recorder&recorderProbe=held");
  const start = page.getByRole("button", { name: "Start recording", exact: true });
  await page.getByRole("radio", { name: "MP4", exact: true }).click();
  await expect(start).toBeDisabled();
  await expect(page.getByRole("status")).toContainText("Checking the MP4 encoder");
  await page.waitForFunction(() => window.__nimbusRecorderHarness!.snapshot().nativeEncoding.probePending);
  await page.evaluate(() => window.__nimbusRecorderHarness!.releaseProbe());
  await expect(start).toBeEnabled();
  await page.getByRole("radio", { name: "GIF", exact: true }).click();
  await page.getByRole("radio", { name: "MP4", exact: true }).click();
  await expect(start).toBeDisabled();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.evaluate(() => window.__nimbusRecorderHarness!.releaseProbe());
  expect((await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot())).nativeEncoding.prepared).toBe(0);
});

test("portable 3D viewpoints restore the camera, preserve PNGs and work offline", async ({ page, context }, info) => {
  test.setTimeout(150_000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => !!window.__nimbusVtkHarness);
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(480, 320); window.__nimbusVtkHarness!.setPlot("volume"); });
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  await expect(page.locator(".plot3d-colorbar")).toBeVisible();
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const before = await page.evaluate(async () => { await window.__nimbusVtkHarness!.captureSavedFrame(); return window.__nimbusVtkHarness!.inspectLastSavedFrame(); });
  const html = await page.evaluate(() => window.__nimbusVtkHarness!.capturePortableViewpoints(12));
  const after = await page.evaluate(async () => { await window.__nimbusVtkHarness!.captureSavedFrame(); return window.__nimbusVtkHarness!.inspectLastSavedFrame(); });
  expect(after?.camera).toEqual(before?.camera);
  await writeFile(info.outputPath("portable-volume.html"), html);
  const viewer = await context.newPage();
  const requests: string[] = []; viewer.on("request", request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
  await viewer.setContent(html);
  await expect(viewer.getByRole("heading", { level: 1 })).toHaveText("Portable storm fixture");
  const metadata = JSON.parse(/<script id="nimbus-viewpoints" type="application\/json">(.*?)<\/script>/s.exec(html)![1]);
  expect(metadata.width).toBeGreaterThan(400);
  expect(metadata.width).toBeLessThanOrEqual(480);
  await expect(viewer.getByRole("img")).toHaveJSProperty("naturalWidth", metadata.width);
  await expect(viewer.locator("#position")).toHaveText("1 / 12 · 0°");
  const first = await viewer.getByRole("img").getAttribute("src");
  await viewer.getByRole("button", { name: "Next viewpoint", exact: true }).click();
  await expect(viewer.locator("#position")).toHaveText("2 / 12 · 30°");
  expect(await viewer.getByRole("img").getAttribute("src")).not.toEqual(first);
  await viewer.locator("#viewport").press("ArrowLeft");
  expect(await viewer.getByRole("img").getAttribute("src")).toEqual(first);
  expect(requests).toEqual([]);
  await viewer.close();
  const cancelled = await page.evaluate(async () => {
    try { await window.__nimbusVtkHarness!.capturePortableViewpoints(12, 2); return false; }
    catch (error) { return error instanceof DOMException && error.name === "AbortError"; }
  });
  expect(cancelled).toBe(true);
  const restored = await page.evaluate(async () => { await window.__nimbusVtkHarness!.captureSavedFrame(); return window.__nimbusVtkHarness!.inspectLastSavedFrame(); });
  expect(restored?.camera).toEqual(before?.camera);
});

test("cancelling GIF encoding terminates its worker and writes no partial export", async ({ page }) => {
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    const probe = { pending: false, terminated: 0 };
    (window as unknown as { gifProbe: typeof probe }).gifProbe = probe;
    window.Worker = class extends OriginalWorker {
      gif: boolean;
      constructor(url: string | URL, options?: WorkerOptions) { super(url, options); this.gif = options?.name === "nimbus-gif-encoder"; }
      postMessage(message: unknown, transfer?: Transferable[] | StructuredSerializeOptions): void {
        if (this.gif && (message as { kind: string }).kind === "frame") { probe.pending = true; return; }
        super.postMessage(message, transfer as Transferable[]);
      }
      terminate(): void { if (this.gif) probe.terminated++; super.terminate(); }
    };
  });
  await page.goto("/harness/index.html?mode=recorder");
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { gifProbe: { pending: boolean } }).gifProbe.pending);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => (window as unknown as { gifProbe: { terminated: number } }).gifProbe.terminated)).toBe(1);
  expect((await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot())).savedGif.byteLength).toBe(0);
});

test("Studio recorder offers portable export for a complete saved 3D frame", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => !!window.__nimbusStudioHarness);
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("/__nimbus_browser_fixture__", "completed"));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  const record = page.getByRole("button", { name: "Rec", exact: true });
  await expect(record).toBeEnabled();
  await record.click();
  await page.getByRole("radio", { name: "Interactive viewpoints", exact: true }).click();
  await page.getByRole("combobox", { name: "Viewpoints", exact: true }).selectOption("12");
  await page.getByRole("button", { name: "Export viewpoints", exact: true }).click();
  await expect(page.locator(".movie-recorder__done")).toContainText(".html", { timeout: 90_000 });
  const path = (await page.locator(".movie-recorder__done").innerText()).replace("Saved to ", "").trim();
  const html = await page.evaluate(path => window.__nimbusStudioHarness!.readExportFile(path), path);
  expect(html).toContain("nimbus.portable-viewpoints");
  const metadata = JSON.parse(/<script id="nimbus-viewpoints" type="application\/json">(.*?)<\/script>/s.exec(html!)![1]);
  expect(metadata.views).toHaveLength(12);
  expect(metadata.source.stepIndex).toBe(0);
  expect(metadata.scene.descriptor.plot).toBe("volume");
});

test("portable supercell viewpoints retain every prepared field without rebuilding source arrays", async ({ page }, info) => {
  test.setTimeout(150_000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => !!window.__nimbusVtkHarness);
  await page.evaluate(() => { window.__nimbusVtkHarness!.setSize(512, 384); window.__nimbusVtkHarness!.setPlot("supercell"); window.__nimbusVtkHarness!.setStep(1); });
  const ready = () => page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false"
    && document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.scenePresentationPending === "false");
  await ready();
  await page.locator(".plot3d-control-tab", { hasText: "Precip" }).click();
  await page.getByRole("group", { name: "Precipitation", exact: true }).getByRole("button").first().click();
  await ready();
  await page.locator('.plot3d-control-tab[aria-expanded="true"]').click();
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
  expect(before.fingerprints.length).toBeGreaterThanOrEqual(2);
  const html = await page.evaluate(() => window.__nimbusVtkHarness!.capturePortableViewpoints(12));
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs())).toEqual(before);
  const exportDocument = JSON.parse(/<script id="nimbus-viewpoints" type="application\/json">(.*?)<\/script>/s.exec(html)![1]);
  expect(exportDocument.scene.descriptor.plot).toBe("supercell");
  expect(exportDocument.views).toHaveLength(12);
  expect(exportDocument.views.every((view: { png: string }) => view.png.startsWith("data:image/png;base64,"))).toBe(true);
  await writeFile(info.outputPath("portable-supercell.html"), html);
});
