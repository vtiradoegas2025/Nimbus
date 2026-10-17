import { expect, test, type Locator, type Page } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { createAnalyticFidelityFixture } from "../../../../ui/src/lib/vtk/analyticFidelityFixtures";
import {
  DBZ_COLORSCALE,
  DBZ_PHYSICAL_STOPS,
  DBZ_RANGE,
  DBZ_VTK_STOPS_VOLUME,
  cssColorbarGradient,
} from "../../../../ui/src/lib/plot/plotColormaps";

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openStudio(page: Page, query = "mode=studio"): Promise<void> {
  await page.goto(`/harness/index.html?${query}`, { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await expect(page.locator(".plot-canvas").first()).toBeVisible({ timeout: 30_000 });
}

test("moving-grid axis provenance follows the selected saved frame", async ({ page }) => {
  await openStudio(page, "mode=studio&movingAxes=1");
  await expect(page.getByText("Axis coordinates are grid-relative", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Axis coordinates unverified", { exact: true })).toHaveCount(0);
  const frame = page.getByRole("slider", { name: "Selected frame" });
  await frame.focus();
  await frame.press("Home");
  await frame.press("ArrowRight");
  await expect(frame).toHaveValue("1");
  await expect(page.getByText("Axis coordinates unverified", { exact: true })).toBeVisible();
  await frame.press("ArrowRight");
  await expect(frame).toHaveValue("2");
  await expect(page.getByText("Axis coordinates unverified", { exact: true })).toHaveCount(0);
});

async function sampleVerticalRamp(
  page: Page,
  locator: Locator,
  fractions: number[],
  targets: ReadonlyArray<readonly number[]>,
): Promise<number[][]> {
  const png = await locator.screenshot();
  return page.evaluate(async ({ dataUrl, fractions: positions, targets: expected }) => {
    const image = new Image();
    image.src = dataUrl;
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Could not decode colorbar screenshot"));
    });
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas 2D unavailable");
    context.drawImage(image, 0, 0);
    return positions.map((fraction, index) => {
      const y = Math.max(0, Math.min(
        canvas.height - 1,
        Math.round((1 - fraction) * (canvas.height - 1)),
      ));
      // A physical stop can fall between screenshot pixels on a compact ramp.
      const pixels = [-1, 0, 1].map(offset => [...context.getImageData(Math.floor(canvas.width / 2), Math.max(0, Math.min(canvas.height - 1, y + offset)), 1, 1).data.slice(0, 3)]);
      const distance = (pixel: number[]) => Math.max(...pixel.map((value, channel) => Math.abs(value - expected[index][channel])));
      return pixels.sort((a, b) => distance(a) - distance(b))[0];
    });
  }, { dataUrl: `data:image/png;base64,${png.toString("base64")}`, fractions, targets });
}

function expectRgbClose(actual: number[], expected: readonly number[], tolerance = 30): void {
  expect(actual).toHaveLength(3);
  actual.forEach((channel, index) => {
    expect(
      Math.abs(channel - expected[index]),
      `rendered ${actual.join(",")} versus expected ${expected.map(Math.round).join(",")}`,
    ).toBeLessThanOrEqual(tolerance);
  });
}

async function minimumExportedRgbDistances(
  page: Page,
  png: Buffer,
  expected: ReadonlyArray<readonly number[]>,
): Promise<number[]> {
  return page.evaluate(async ({ dataUrl, expected: targets }) => {
    const image = new Image();
    image.src = dataUrl;
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Could not decode exported PNG"));
    });
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas 2D unavailable");
    context.drawImage(image, 0, 0);
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const distances = targets.map(() => Number.POSITIVE_INFINITY);
    for (let offset = 0; offset < data.length; offset += 4) {
      if (data[offset + 3] === 0) continue;
      for (let target = 0; target < targets.length; target += 1) {
        const color = targets[target];
        const distance = Math.max(
          Math.abs(data[offset] - color[0]),
          Math.abs(data[offset + 1] - color[1]),
          Math.abs(data[offset + 2] - color[2]),
        );
        if (distance < distances[target]) distances[target] = distance;
      }
    }
    return distances;
  }, { dataUrl: `data:image/png;base64,${png.toString("base64")}`, expected });
}

test("slice index replacement retains controls and the coherent plane", async ({ page }) => {
  await openStudio(page);
  const field = page.locator("select[aria-label='Field']").first();
  await field.selectOption("w");
  const indexGroup = page.locator(".control-group", { hasText: /^Index \(/ }).first();
  await expect(indexGroup).toBeVisible();
  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(700));
  await indexGroup.locator('input[type="range"]').fill("2");

  await expect(indexGroup).toBeVisible();
  await expect(page.locator(".plot-canvas").first()).toBeVisible();
  await expect(page.getByText(/loading selected .* index 2/)).toBeVisible();
  await expect(page.getByText(/loading selected .* index 2/)).toHaveCount(0, { timeout: 30_000 });
  await expect(indexGroup).toBeVisible();
  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(0));
});

test("composite index interaction retains its layer controls and plane", async ({ page }) => {
  await openStudio(page);
  await page.getByRole("button", { name: "Composite", exact: true }).click();
  await expect(page.locator(".plot-canvas").first())
    .toHaveAttribute("aria-label", /contours/, { timeout: 30_000 });
  const indexGroup = page.locator(".control-group", { hasText: /^Index \(/ }).first();
  const contourA = page.locator(".control-group", { hasText: "Contour A" });
  await expect(indexGroup).toBeVisible({ timeout: 30_000 });
  await expect(contourA).toBeVisible();
  const slider = indexGroup.locator('input[type="range"]');
  await expect.poll(async () => Number(await slider.getAttribute("max")))
    .toBeGreaterThan(0);
  await expect(slider).toBeEnabled();
  const box = await slider.boundingBox();
  expect(box).not.toBeNull();
  await slider.click({ position: { x: box!.width * 0.35, y: box!.height / 2 } });
  await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(0);
  const selected = await slider.inputValue();
  await expect(indexGroup).toBeVisible();
  await expect(contourA).toBeVisible();
  await expect(page.locator(".plot-canvas").first()).toBeVisible();
  await expect(indexGroup).toContainText(`Index (z = ${selected})`);
});

test("plane relayout requests a disclosed source viewport", async ({ page }) => {
  await openStudio(page);
  const graph = page.locator(".plot-canvas").first();
  await graph.evaluate((element) => {
    (element as HTMLDivElement & {
      emit?: (name: string, payload: Record<string, unknown>) => void;
    }).emit?.("plotly_relayout", {
      "xaxis.range[0]": -2,
      "xaxis.range[1]": 2,
      "yaxis.range[0]": -2,
      "yaxis.range[1]": 2,
    });
  });
  await expect(page.getByText(/viewport native cells/)).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Reset view", exact: true }).click();
  await expect(page.getByText(/viewport native cells/)).toHaveCount(0);
});

test("vertical scalar and vector sections put height upward", async ({ page }) => {
  await openStudio(page);
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();
  const scalar = page.locator(".plot-canvas").first();
  await expect(scalar.locator(".xtitle")).toContainText("x (km)", { timeout: 30_000 });
  await expect(scalar.locator(".ytitle")).toContainText("z (km)");

  await page.getByRole("button", { name: "Composite", exact: true }).click();
  const composite = page.locator(".plot-canvas").first();
  await expect(composite.locator(".xtitle")).toContainText("x (km)", { timeout: 30_000 });
  await expect(composite.locator(".ytitle")).toContainText("z (km)");

  await page.getByRole("button", { name: "Vector field", exact: true }).click();
  const vector = page.locator(".plot-canvas").first();
  await expect(vector.locator(".xtitle")).toContainText("x (km)", { timeout: 30_000 });
  await expect(vector.locator(".ytitle")).toContainText("z (km)");
  await expect(page.locator(".control-group", { hasText: "Vector field" }).locator("select"))
    .toContainText("u →, w ↑");
});

test("analytic plan and vertical sections preserve source values on the physical axes", async ({ page }, info) => {
  await openStudio(page);
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("analytic_signed_couplet");
  const shape: [number, number, number] = [48, 36, 24];
  const source = createAnalyticFidelityFixture("analytic_signed_couplet", shape).field.data;
  const graph = page.locator(".plot-canvas").first();
  const evidence: Array<{ axis: string; rows: number; columns: number; maximumError: number }> = [];
  for (const axis of ["z", "y", "x"] as const) {
    await page.getByRole("button", { name: axis === "z" ? "Plan views" : "Cross-sections", exact: true }).click();
    await page.getByRole("button", { name: "Slice", exact: true }).click();
    await page.locator(".control-group", { hasText: "Slice axis" }).getByRole("button", { name: axis, exact: true }).click();
    const index = axis === "z" ? 7 : 12;
    const rail = page.locator(".control-group", { hasText: /^Index \(/ }).locator('input[type="range"]');
    await rail.fill(String(index));
    const rows = axis === "z" ? shape[0] : shape[2];
    const columns = axis === "y" ? shape[0] : shape[1];
    await expect.poll(() => graph.evaluate(element => {
      const trace = (element as HTMLElement & { data: Array<{ z: number[][] }> }).data?.[0];
      return [trace?.z?.length, trace?.z?.[0]?.length];
    })).toEqual([rows, columns]);
    const expected = Array.from({ length: rows }, (_, row) => Array.from({ length: columns }, (_, column) => {
      const x = axis === "x" ? index : axis === "z" ? row : column;
      const y = axis === "y" ? index : column;
      const z = axis === "z" ? index : row;
      return source[(x * shape[1] + y) * shape[2] + z];
    }));
    await expect.poll(() => graph.evaluate((element, values) => {
      const z = (element as HTMLElement & { data: Array<{ z: number[][] }> }).data?.[0]?.z;
      if (!z || z.length !== values.length || z[0].length !== values[0].length) return Number.POSITIVE_INFINITY;
      return Math.max(...z.flatMap((row, r) => row.map((value, c) => Math.abs(value - values[r][c]))));
    }, expected), { timeout: 20_000 }).toBeLessThan(1e-7);
    const trace = await graph.evaluate(element => (element as HTMLElement & { data: Array<{ x: number[]; y: number[]; z: number[][] }> }).data[0]);
    expect(trace.x).toEqual(Array.from({ length: columns }, (_, position) => position * (axis === "y" ? 125 : 175) / 1000));
    expect(trace.y).toEqual(Array.from({ length: rows }, (_, position) => position * (axis === "z" ? 125 : 90) / 1000));
    let maximumError = 0;
    for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
      const x = axis === "x" ? index : axis === "z" ? row : column;
      const y = axis === "y" ? index : column;
      const z = axis === "z" ? index : row;
      const actual = trace.z[row][column];
      if (!Number.isFinite(actual)) maximumError = Number.POSITIVE_INFINITY;
      maximumError = Math.max(maximumError, Math.abs(actual - source[(x * shape[1] + y) * shape[2] + z]));
    }
    expect(maximumError, `${axis} plane source reference`).toBeLessThan(1e-7);
    evidence.push({ axis, rows, columns, maximumError });
  }
  const path = info.outputPath("analytic-plane-source-reference.json");
  await writeFile(path, JSON.stringify(evidence, null, 2));
  await info.attach("analytic-plane-source-reference", { path, contentType: "application/json" });
});

test("composite contour diagnostics belong to the layer control", async ({ page }) => {
  await openStudio(page);
  await page.getByRole("button", { name: "Composite", exact: true }).click();
  const contour = page.locator(".control-group", { hasText: "Contour A" });
  await contour.locator('input[type="text"]').fill("9999");
  await expect(contour.locator(".composite-contour__status"))
    .toContainText("outside the displayed data range", { timeout: 30_000 });
  await expect(page.locator(".annotation-text", { hasText: "outside the displayed data range" }))
    .toHaveCount(0);
  await expect(page.getByRole("button", { name: "View fill in 3D", exact: true })).toBeVisible();
});

test("composite controls reserve warning height and keep the index rail stable", async ({ page }) => {
  await page.setViewportSize({ width: 1_900, height: 1_050 });
  await openStudio(page);
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Composite", exact: true }).click();

  const controls = page.locator(".plot-controls--composite");
  const plotBody = page.locator(".plot-body").first();
  const contours = page.locator(".composite-contour-control");
  await contours.nth(0).locator('input[type="text"]').fill("9999");
  await contours.nth(1).locator('input[type="text"]').fill("9999");
  await expect(page.locator(".composite-contour__status")).toHaveCount(2, { timeout: 30_000 });

  const layout = await page.evaluate(() => {
    const box = (selector: string) => {
      const rect = document.querySelector<HTMLElement>(selector)?.getBoundingClientRect();
      return rect ? { top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left } : null;
    };
    const warnings = [...document.querySelectorAll<HTMLElement>(".composite-contour__status")]
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return { top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left };
      });
    return {
      controls: box(".plot-controls--composite"),
      primary: box(".composite-controls__row--primary"),
      secondary: box(".composite-controls__row--secondary"),
      plot: box(".plot-body"),
      warnings,
    };
  });
  expect(layout.controls).not.toBeNull();
  expect(layout.primary).not.toBeNull();
  expect(layout.secondary).not.toBeNull();
  expect(layout.plot).not.toBeNull();
  expect(layout.warnings).toHaveLength(2);
  expect(layout.primary!.bottom).toBeLessThanOrEqual(layout.secondary!.top);
  for (const warning of layout.warnings) {
    expect(warning.bottom).toBeLessThanOrEqual(layout.controls!.bottom);
    expect(warning.bottom).toBeLessThanOrEqual(layout.plot!.top);
  }

  const index = page.locator(".composite-index-control");
  const slider = index.locator('input[type="range"]');
  const initial = await slider.boundingBox();
  expect(initial).not.toBeNull();
  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(700));
  await slider.click({ position: { x: initial!.width * 0.7, y: initial!.height / 2 } });
  await expect(index).toBeVisible();
  const loading = await slider.boundingBox();
  expect(loading).not.toBeNull();
  expect(Math.abs(loading!.width - initial!.width)).toBeLessThanOrEqual(1);

  await page.locator(".control-group", { hasText: "Slice axis" })
    .getByRole("button", { name: "y", exact: true }).click();
  const anotherAxis = await slider.boundingBox();
  expect(anotherAxis).not.toBeNull();
  expect(Math.abs(anotherAxis!.width - initial!.width)).toBeLessThanOrEqual(1);
  await expect(controls).toBeVisible();
  await expect(plotBody).toBeVisible();
  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(0));
});

test("terrain-following scalar and vector sections use physical remap references", async ({ page }, testInfo) => {
  await openStudio(page, "mode=studio&terrain=1");
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();
  await expect(page.getByText("physical-volume terrain→Cartesian vertical remap"))
    .toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".plot-canvas .xtitle")).toContainText("x (km)");
  await expect(page.locator(".plot-canvas .ytitle")).toContainText("z (km)");
  const scalarPath = testInfo.outputPath("terrain-scalar-section.png");
  await page.locator(".plot-body").screenshot({ path: scalarPath });
  await testInfo.attach("terrain-scalar-section", { path: scalarPath, contentType: "image/png" });

  await page.getByRole("button", { name: "Vector field", exact: true }).click();
  await expect(page.getByText("physical-volume terrain→Cartesian vector remap"))
    .toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".plot-canvas .xtitle")).toContainText("x (km)");
  await expect(page.locator(".plot-canvas .ytitle")).toContainText("z (km)");
  const vectorPath = testInfo.outputPath("terrain-vector-section.png");
  await page.locator(".plot-body").screenshot({ path: vectorPath });
  await testInfo.attach("terrain-vector-section", { path: vectorPath, contentType: "image/png" });
});

test("stretched vertical coordinates remain exact and height-up", async ({ page }, testInfo) => {
  await openStudio(page);
  await page.evaluate(() => window.__nimbusStudioHarness?.setRun(
    "/__nimbus_browser_fixture__/stretched", "completed",
  ));
  await expect(page.locator(".plot-studio__title-row h2"))
    .toHaveText("/__nimbus_browser_fixture__/stretched");
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();
  const graph = page.locator(".plot-canvas").first();
  await expect(graph.locator(".ytitle")).toContainText("z (km)", { timeout: 30_000 });
  await expect.poll(() => graph.evaluate((element) => {
    const y = (element as HTMLDivElement & { data?: Array<{ y?: number[] }> }).data?.[0]?.y ?? [];
    return y.at(-1) ?? null;
  }), { timeout: 30_000 }).toBe(3.027);
  const displayedZ = await graph.evaluate((element) => {
    const traces = (element as HTMLDivElement & { data?: Array<{ y?: number[] }> }).data;
    return traces?.[0]?.y ?? [];
  });
  const expectedZ = Array.from({ length: 24 }, (_, index) =>
    Math.round(35 * index + 4.2 * index * index) / 1000);
  expect(displayedZ).toEqual(expectedZ);
  const path = testInfo.outputPath("stretched-coordinate-section.png");
  await page.locator(".plot-body").screenshot({ path });
  await testInfo.attach("stretched-coordinate-section", { path, contentType: "image/png" });
});

test("above-cap plane refines from reduced overview to native viewport", async ({ page }, testInfo) => {
  test.setTimeout(2 * 60_000);
  await openStudio(page);
  await page.evaluate(() => window.__nimbusStudioHarness?.setRun(
    "/__nimbus_browser_fixture__/plane-large", "completed",
  ));
  await expect(page.locator(".plot-studio__title-row h2"))
    .toHaveText("/__nimbus_browser_fixture__/plane-large");
  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();
  await expect(page.getByText(/bounded linear-Z mean 2×1×1/))
    .toBeVisible({ timeout: 60_000 });
  const graph = page.locator(".plot-canvas").first();
  const overview = await graph.evaluate((element) => {
    const trace = (element as HTMLDivElement & {
      data?: Array<{ x?: number[]; y?: number[] }>;
    }).data?.[0];
    const x = trace?.x ?? [];
    const y = trace?.y ?? [];
    return {
      xSamples: x.length,
      xSpacing: x.length > 1 ? x[1] - x[0] : null,
      ySamples: y.length,
      ySpacing: y.length > 1 ? y[1] - y[0] : null,
    };
  });
  expect(overview.xSamples).toBe(2_300);
  expect(overview.xSpacing).toBeCloseTo(0.1, 8);
  expect(overview.ySamples).toBe(1_150);
  expect(overview.ySpacing).toBeCloseTo(0.2, 8);
  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(300));
  await graph.evaluate((element) => {
    (element as HTMLDivElement & {
      emit?: (name: string, payload: Record<string, unknown>) => void;
    }).emit?.("plotly_relayout", {
      "xaxis.range": [100, 110],
      "yaxis.range": [100, 110],
    });
  });
  await page.waitForTimeout(75);
  await expect(page.getByText(/viewport native cells/)).toHaveCount(0);
  await expect(page.getByText(/viewport native cells/)).toBeVisible({ timeout: 60_000 });
  const detail = await graph.evaluate((element) => {
    const trace = (element as HTMLDivElement & {
      data?: Array<{ x?: number[]; y?: number[] }>;
    }).data?.[0];
    const x = trace?.x ?? [];
    const y = trace?.y ?? [];
    return {
      xSamples: x.length,
      xSpacing: x.length > 1 ? x[1] - x[0] : null,
      ySamples: y.length,
      ySpacing: y.length > 1 ? y[1] - y[0] : null,
    };
  });
  expect(detail.xSamples).toBeGreaterThanOrEqual(100);
  expect(detail.xSamples).toBeLessThan(110);
  expect(detail.ySamples).toBeGreaterThanOrEqual(100);
  expect(detail.ySamples).toBeLessThan(110);
  expect(detail.xSpacing).toBeCloseTo(0.1, 8);
  expect(detail.ySpacing).toBeCloseTo(0.1, 8);
  const path = testInfo.outputPath("large-plane-native-viewport.png");
  await page.locator(".plot-body").screenshot({ path });
  await testInfo.attach("large-plane-native-viewport", { path, contentType: "image/png" });
});

test("rendered dBZ plane and volume legends use their canonical physical ramps", async ({ page }, testInfo) => {
  await openStudio(page);
  await page.locator("select[aria-label='Field']").first().selectOption("reflectivity_dbz");
  const graph = page.locator(".plot-canvas").first();
  await expect(graph).toHaveAttribute("aria-label", /reflectivity_dbz/, { timeout: 30_000 });
  const trace = await graph.evaluate((element) => {
    const first = (element as HTMLDivElement & {
      data?: Array<{ colorscale?: Array<[number, string]>; zmin?: number; zmax?: number }>;
    }).data?.[0];
    return { colorscale: first?.colorscale ?? [], zmin: first?.zmin, zmax: first?.zmax };
  });
  expect(trace.colorscale).toEqual(DBZ_COLORSCALE);
  expect([trace.zmin, trace.zmax]).toEqual([...DBZ_RANGE]);
  const plotlyStops = DBZ_PHYSICAL_STOPS.filter((stop) => [0, 30, 50, 70].includes(stop.scalar));
  const plotlyPixels = await sampleVerticalRamp(
    page,
    graph.locator(".colorbar .cbfill"),
    plotlyStops.map((stop) => (stop.scalar - DBZ_RANGE[0]) / (DBZ_RANGE[1] - DBZ_RANGE[0])),
    plotlyStops.map(stop => stop.rgb),
  );
  plotlyPixels.forEach((pixel, index) => expectRgbClose(pixel, plotlyStops[index].rgb));
  const downloadStarted = page.waitForEvent("download");
  await graph.getByRole("button", { name: "Download plot as a PNG", exact: true }).click();
  const download = await downloadStarted;
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();
  const exportedDistances = await minimumExportedRgbDistances(
    page,
    await readFile(downloadPath!),
    plotlyStops.map((stop) => stop.rgb),
  );
  exportedDistances.forEach((distance) => expect(distance).toBeLessThanOrEqual(30));
  const planePath = testInfo.outputPath("dbz-canonical-plane.png");
  await page.locator(".plot-body").screenshot({ path: planePath });
  await testInfo.attach("dbz-canonical-plane", { path: planePath, contentType: "image/png" });

  await page.getByRole("button", { name: "View in 3D", exact: true }).click();
  const companion = page.locator(".plot-studio__pane--companion").first();
  await expect(companion.locator(".plot3d-colorbar__title"))
    .toContainText("reflectivity_dbz (dBZ)", { timeout: 30_000 });
  const style = await companion.locator(".plot3d-colorbar__bar").getAttribute("style");
  expect(style?.replaceAll(" ", "")).toContain(
    cssColorbarGradient("dBZ volume").replaceAll(" ", ""),
  );
  const volumeStops = DBZ_VTK_STOPS_VOLUME.filter((stop) => [0, 30, 50, 70].includes(stop.scalar));
  const volumePixels = await sampleVerticalRamp(
    page,
    companion.locator(".plot3d-colorbar__bar").first(),
    volumeStops.map((stop) => (stop.scalar - DBZ_RANGE[0]) / (DBZ_RANGE[1] - DBZ_RANGE[0])),
    volumeStops.map(stop => [stop.r * 255, stop.g * 255, stop.b * 255]),
  );
  volumePixels.forEach((pixel, index) => expectRgbClose(pixel, [
    volumeStops[index].r * 255,
    volumeStops[index].g * 255,
    volumeStops[index].b * 255,
  ]));
  const volumePath = testInfo.outputPath("dbz-canonical-volume.png");
  await companion.locator(".plot-body, .volume-render").first().screenshot({ path: volumePath });
  await testInfo.attach("dbz-canonical-volume", { path: volumePath, contentType: "image/png" });
  const vtkCapture = await page.evaluate(() => window.__nimbusStudioHarness?.captureActivePlot());
  expect(vtkCapture).toMatch(/^data:image\/png;base64,/);
  await testInfo.attach("dbz-vtk-export", {
    body: Buffer.from(vtkCapture!.slice(vtkCapture!.indexOf(",") + 1), "base64"),
    contentType: "image/png",
  });
});
