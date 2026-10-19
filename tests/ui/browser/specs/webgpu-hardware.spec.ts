import { expect, test, type Browser, type Page, type TestInfo } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

// Hardware-only lane (not part of the portable default matrix):
// NIMBUS_WEBGPU_HARDWARE=1 NIMBUS_WEBGPU_EXECUTABLE="/path/to/Chromium" \
//   npx playwright test ../tests/ui/browser/specs/webgpu-hardware.spec.ts \
//   --project=chromium-webgpu-hardware

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

for (const lighting of [
  { preset: "neutral", shadowDepth: 1, fillTint: null },
  { preset: "neutral", shadowDepth: 1.6, fillTint: "#4477ff" },
  { preset: "flat", shadowDepth: 1.6, fillTint: "#4477ff" },
]) test(`surface material and tube topology remain visible across hardware backends (${lighting.preset}, ${lighting.shadowDepth})`, async ({ page }, testInfo) => {
  const captures = new Map<string, NonNullable<Awaited<ReturnType<NonNullable<Window["__nimbusVtkHarness"]>["captureAppearance"]>>>>();
  for (const backend of ["webgl", "webgpu"] as const) {
    await page.goto("/harness/index.html?mounted=0");
    await page.evaluate((value) => localStorage.setItem("nimbus-studio:vtk-backend:v1", value), backend);
    await page.evaluate(value => localStorage.setItem("nimbus-studio:scene-appearance:v1", JSON.stringify({
      backgroundMode: "dark", customBackground: "#202734", lighting: value.preset,
      shadowDepth: value.shadowDepth, fillTint: value.fillTint,
    })), lighting);
    await page.goto("/harness/index.html");
    await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
    for (const plot of ["isosurface", "streamlines", "trajectories"] as const) {
      await page.evaluate((value) => {
        window.__nimbusVtkHarness!.setPlot(value);
        window.__nimbusVtkHarness!.setStep(1);
      }, plot);
      await page.waitForFunction(([plot, backend]) => {
        const harness = window.__nimbusVtkHarness;
        const host = document.querySelector(".volume-render__canvas");
        const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
        return harness && harness.snapshot().ipc.inFlightJobs === 0
          && host?.getAttribute("data-vtk-active-api") === (backend === "webgpu" ? "vtkWebGPURenderWindow" : "vtkOpenGLRenderWindow")
          && (plot === "trajectories"
            ? document.querySelector(".volume-render__overlay--muted")?.textContent?.includes("parcel paths")
            : scene?.dataset.framePlot === plot && scene.dataset.frameLoading === "false");
      }, [plot, backend] as const, { timeout: 120_000 });
      await expect.poll(async () => {
        const value = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
        return value?.evidence.descriptor.plot === plot && value.evidence.layers.some(layer =>
          layer.visible && layer.kind === "actor" && (layer.pointCount ?? 0) > 100 && (layer.cellCount ?? 0) > 0);
      }, { timeout: 30000 }).toBe(true);
      const rig = await page.evaluate(() => window.__nimbusVtkHarness!.inspectLighting());
      expect(rig).toHaveLength(3);
      expect(rig.every(light => light.type === "SceneLight")).toBe(true);
      if (plot !== "isosurface") {
        const surfaces = await page.evaluate(() => window.__nimbusVtkHarness!.inspectSurfaceNormals());
        const tubes = surfaces.filter(surface => surface.points > 100 && surface.strips > 0);
        expect(tubes.length).toBeGreaterThan(0);
        expect(tubes.every(surface => surface.normals === surface.points)).toBe(true);
      }
      const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
      expect(capture).not.toBeNull();
      if (!capture) continue;
      expect(capture.evidence.descriptor.parameters?.lightingControls).toEqual(lighting);
      captures.set(`${plot}:${backend}`, capture);
      if (plot === "streamlines") {
        for (const layer of capture.evidence.layers.filter((layer) => layer.kind === "actor").slice(0, 2)) {
          expect(layer.visible).toBe(true);
          expect(layer.cellCount).toBeGreaterThan(0);
        }
      }
      await testInfo.attach(`${plot}-${backend}`, {
        body: Buffer.from(capture.pngDataUri.split(",")[1], "base64"), contentType: "image/png",
      });
    }
  }
  for (const plot of ["isosurface", "streamlines", "trajectories"]) {
    const reference = captures.get(`${plot}:webgl`)!;
    const candidate = captures.get(`${plot}:webgpu`)!;
    expect(candidate.evidence.camera).toEqual(reference.evidence.camera);
    expect(candidate.evidence.layers).toEqual(reference.evidence.layers);
    const comparison = await page.evaluate(([left, right]) =>
      window.__nimbusVtkHarness?.compareAppearance(left, right),
    [reference.pngDataUri, candidate.pngDataUri] as const);
    await testInfo.attach(`${plot}-backend-comparison`, {
      body: JSON.stringify(comparison), contentType: "application/json",
    });
    expect(comparison?.rgbRmse).toBeLessThan(0.02);
    expect(comparison?.meanAbsoluteError).toBeLessThan(0.003);
  }
});

interface PixelSignal {
  meanLuma: number;
  lumaDeviation: number;
}

async function canvasEvidence(page: Page, testInfo: TestInfo, name: string): Promise<PixelSignal> {
  const canvas = page.locator(".volume-render__canvas canvas").first();
  await expect(canvas).toBeVisible();
  const png = await canvas.screenshot();
  const evidenceDirectory = resolve("build/qualification/webgpu-hardware");
  await mkdir(evidenceDirectory, { recursive: true });
  await writeFile(resolve(evidenceDirectory, `${name}.png`), png);
  await testInfo.attach(`${name}.png`, { body: png, contentType: "image/png" });
  return page.evaluate(async (base64) => {
    const image = new Image();
    image.src = `data:image/png;base64,${base64}`;
    await image.decode();
    const scratch = document.createElement("canvas");
    scratch.width = image.naturalWidth;
    scratch.height = image.naturalHeight;
    const context = scratch.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("2D evidence context unavailable");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, scratch.width, scratch.height).data;
    let sum = 0;
    let sumSquares = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const luma = pixels[index]! * 0.2126
        + pixels[index + 1]! * 0.7152
        + pixels[index + 2]! * 0.0722;
      sum += luma;
      sumSquares += luma * luma;
    }
    const count = pixels.length / 4;
    const meanLuma = sum / count;
    return {
      meanLuma,
      lumaDeviation: Math.sqrt(Math.max(0, sumSquares / count - meanLuma * meanLuma)),
    };
  }, png.toString("base64"));
}

function assertRendered(signal: PixelSignal, label: string): void {
  expect(
    signal.meanLuma > 18 || signal.lumaDeviation > 5,
    `${label} did not contain rendered scene pixels: ${JSON.stringify(signal)}`,
  ).toBeTruthy();
}

async function hardwareEvidence(browser: Browser, page: Page): Promise<unknown> {
  const session = await browser.newBrowserCDPSession();
  const system = await session.send("SystemInfo.getInfo") as {
    gpu?: {
      devices?: Array<{ vendorString?: string; deviceString?: string; vendorId?: number; deviceId?: number }>;
      auxAttributes?: Record<string, unknown>;
      featureStatus?: Record<string, string>;
    };
  };
  await session.detach();
  const browserGpuText = JSON.stringify(system.gpu ?? {});
  expect(browserGpuText).not.toMatch(/swiftshader|llvmpipe|software raster/i);

  const device = await page.evaluate(async () => Promise.race([
    (async () => {
    const gpu = (navigator as Navigator & {
      gpu?: {
        requestAdapter: (options?: {
          powerPreference?: "high-performance";
          forceFallbackAdapter?: boolean;
        }) => Promise<{
          info?: Record<string, string>;
          isFallbackAdapter?: boolean;
          requestDevice: () => Promise<{
            createBuffer: (descriptor: { size: number; usage: number }) => {
              mapAsync: (mode: number) => Promise<void>;
              getMappedRange: () => ArrayBuffer;
              unmap: () => void;
              destroy: () => void;
            };
            createCommandEncoder: () => {
              copyBufferToBuffer: (
                source: unknown,
                sourceOffset: number,
                destination: unknown,
                destinationOffset: number,
                size: number,
              ) => void;
              finish: () => unknown;
            };
            queue: {
              writeBuffer: (buffer: unknown, offset: number, data: Float32Array) => void;
              submit: (commands: unknown[]) => void;
              onSubmittedWorkDone: () => Promise<void>;
            };
            destroy?: () => void;
          }>;
        } | null>;
      };
    }).gpu;
    if (!gpu) throw new Error("navigator.gpu is absent");
    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });
    if (!adapter) throw new Error("hardware WebGPU adapter request returned null");
    if (adapter.isFallbackAdapter === true) throw new Error("WebGPU returned a fallback adapter");
    const activeDevice = await adapter.requestDevice();
    const input = new Float32Array(new SharedArrayBuffer(16));
    input.set([11, 22, 33, 44]);
    const source = activeDevice.createBuffer({ size: 16, usage: 12 });
    const readback = activeDevice.createBuffer({ size: 16, usage: 9 });
    activeDevice.queue.writeBuffer(source, 0, input);
    const encoder = activeDevice.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, 16);
    activeDevice.queue.submit([encoder.finish()]);
    await activeDevice.queue.onSubmittedWorkDone();
    await readback.mapAsync(1);
    const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    readback.unmap();
    source.destroy();
    readback.destroy();
      return {
        adapterInfo: adapter.info ?? {},
        values,
        crossOriginIsolated,
        brave: "brave" in navigator,
      };
    })(),
    new Promise<never>((_, reject) => setTimeout(
      () => reject(new Error("hardware WebGPU write/readback timed out after 30 seconds")),
      30_000,
    )),
  ]));
  expect(device.crossOriginIsolated).toBe(true);
  expect(device.values).toEqual([11, 22, 33, 44]);
  expect(JSON.stringify(device.adapterInfo)).not.toMatch(/swiftshader|llvmpipe|software/i);
  return { cdp: system.gpu, webgpu: device };
}

async function openIsosurface(page: Page): Promise<void> {
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });
  const sceneTab = page.locator(".plot3d-control-tab", { hasText: "Scene" }).first();
  await sceneTab.click();
  await expect(sceneTab).toHaveAttribute("aria-expanded", "true");
}

function backendControl(page: Page) {
  return page.locator(".control-group").filter({
    has: page.locator(".control-group__label", { hasText: "3D renderer" }),
  }).first();
}

async function waitForBackend(page: Page, backend: "webgl" | "webgpu"): Promise<void> {
  const expectedApi = backend === "webgpu" ? "vtkWebGPURenderWindow" : "vtkOpenGLRenderWindow";
  await expect(page.locator(".volume-render__canvas")).toHaveAttribute(
    "data-vtk-active-api",
    expectedApi,
    { timeout: 45_000 },
  );
  await expect(page.locator(".volume-render__canvas-wrap")).toHaveAttribute(
    "data-contour-loading",
    "false",
    { timeout: 45_000 },
  );
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await page.evaluate(() => new Promise<void>((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}

test("hardware WebGPU renders and survives WebGL to WebGPU to WebGL reconstruction", async ({
  browser,
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    window.localStorage.setItem("nimbus-studio:vtk-backend:v1", "webgl");
  });
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await openIsosurface(page);
  const control = backendControl(page);
  const select = control.locator("select");

  await select.selectOption("webgl");
  await expect(control).toContainText("WebGL active");
  await waitForBackend(page, "webgl");
  const before = await canvasEvidence(page, testInfo, "before-webgl");
  assertRendered(before, "initial WebGL frame");

  await select.selectOption("webgpu");
  await expect(control).toContainText("WebGPU active", { timeout: 45_000 });
  await waitForBackend(page, "webgpu");
  const webgpu = await canvasEvidence(page, testInfo, "hardware-webgpu");
  assertRendered(webgpu, "hardware WebGPU frame");
  const hardware = await hardwareEvidence(browser, page);

  await select.selectOption("webgl");
  await expect(control).toContainText("WebGL active");
  await waitForBackend(page, "webgl");
  const after = await canvasEvidence(page, testInfo, "after-webgl");
  assertRendered(after, "restored WebGL frame");

  const handoff = await page.evaluate(() => window.__NIMBUS_VTK_HANDOFF_METRICS__ ?? null);
  const qualification = { hardware, before, webgpu, after, handoff };
  await testInfo.attach("webgpu-qualification.json", {
    body: JSON.stringify(qualification, null, 2),
    contentType: "application/json",
  });
  await mkdir(resolve("build/qualification/webgpu-hardware"), { recursive: true });
  await writeFile(
    resolve("build/qualification/webgpu-hardware/evidence.json"),
    JSON.stringify(qualification, null, 2),
  );
  expect(handoff).not.toBeNull();
  expect(handoff!.byStage.preparation.logicalBytes).toBeGreaterThan(0);
  expect(handoff!.byStage.preparation.copiedBytes)
    .toBeLessThanOrEqual(handoff!.byStage.preparation.logicalBytes);
  expect(handoff!.sourceRetainedDatasets).toBeGreaterThan(0);
  expect(handoff!.downgradeReasons).not.toContain(
    "Brave disabled SAB geometry after a qualified large-grid VTK worker crash",
  );
  expect(errors).toEqual([]);
});
