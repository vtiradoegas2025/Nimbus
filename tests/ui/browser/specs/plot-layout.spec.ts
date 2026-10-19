import { expect, test, type Page } from "@playwright/test";

test("failed reductions do not count timeline placeholders as loaded", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio");
  await page.getByRole("button", { name: "Time series", exact: true }).click();
  // The native fixture rejects reduction IPC, providing a real failure path.
  await expect(page.locator(".plot-overlay--error")).toContainText("reduce_field");
  await expect(page.getByRole("group", { name: "Frames", exact: true })).toContainText("0 / 3 loaded");
  await page.getByRole("button", { name: "Time-height", exact: true }).click();
  await expect(page.locator(".plot-overlay--error")).toContainText("reduce_field_axis");
  await expect(page.getByRole("group", { name: "Frames", exact: true })).toContainText("0 / 3 loaded");
});

test("packaged time-height plots keep physical levels and sample-time hover values", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("nimbus-example://manufactured-supercell-v1", "completed"));
  await expect(page.getByRole("heading", { name: "nimbus-example://manufactured-supercell-v1", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Time series", exact: true }).click();
  await page.getByRole("button", { name: "Time-height", exact: true }).click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  const canvas = page.locator(".plot-canvas");
  await expect.poll(() => canvas.evaluate((element) => {
    const graph = element as HTMLElement & { data: Array<{ x: number[]; y: number[]; z: number[][]; customdata: number[][] }> };
    return graph.data?.[0]?.z?.[0]?.length ?? 0;
  })).toBe(8);
  const trace = await canvas.evaluate((element) => (element as HTMLElement & { data: Array<{ x: number[]; y: number[]; z: number[][]; customdata: number[][] }> }).data[0]);
  expect(trace.y).toHaveLength(20);
  await expect(page.getByRole("group", { name: "Frames", exact: true })).toContainText("8 / 8 loaded");
  expect(trace.x).toHaveLength(9);
  expect(trace.customdata[0]).toHaveLength(8);
  expect(trace.customdata[0].every((time, index, values) => index === 0 || time > values[index - 1])).toBe(true);
  await expect(canvas.locator(".ytitle")).toContainText("km");
  await expect(page.locator(".plot-error, .plot-overlay--error, .error-boundary")).toHaveCount(0);
});

test("time-series reducer changes refit the value axis after zoom", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("nimbus-example://manufactured-supercell-v1", "completed"));
  await expect(page.getByRole("heading", { name: "nimbus-example://manufactured-supercell-v1", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Time series", exact: true }).click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  const canvas = page.locator(".plot-canvas");
  await expect(canvas.locator(".gtitle")).toBeVisible();
  await page.getByRole("button", { name: "mean", exact: true }).click();
  await expect(canvas.locator(".gtitle")).toContainText("mean(");
  const drag = await canvas.locator(".nsewdrag").boundingBox();
  expect(drag).not.toBeNull();
  await page.mouse.move(drag!.x + drag!.width * 0.2, drag!.y + drag!.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(drag!.x + drag!.width * 0.8, drag!.y + drag!.height * 0.8, { steps: 5 });
  await page.mouse.up();
  await page.getByRole("button", { name: "# >", exact: true }).click();
  await expect.poll(() => canvas.evaluate((element) => {
    const graph = element as HTMLElement & { data: Array<{ y: Array<number | null> }>; _fullLayout: { yaxis: { range: number[] } } };
    const [low, high] = graph._fullLayout.yaxis.range;
    const values = graph.data[0].y.filter((value): value is number => value != null);
    return values.length >= 2 && values.every((value) => value >= low && value <= high);
  })).toBe(true);
});

test("comfortable plot text grows without clipping labels in a compact window", async ({ page }) => {
  await openPlot(page, "Vertical", "dark");
  await page.evaluate(() => {
    document.documentElement.dataset.textSize = "comfortable";
    window.dispatchEvent(new CustomEvent("nimbus-studio:appearance-applied"));
  });
  const canvas = page.locator(".plot-canvas");
  await expect.poll(() => canvas.evaluate((element) =>
    (element as HTMLElement & { _fullLayout: { font: { size: number } } })._fullLayout.font.size)).toBe(13);
  await expect.poll(() => canvas.evaluate((element) => {
    const body = element.closest(".plot-body")!.getBoundingClientRect();
    return [".xtitle", ".ytitle", ".gtitle"].every((selector) => {
      const box = element.querySelector(selector)?.getBoundingClientRect();
      return box != null && box.left >= body.left - 1 && box.right <= body.right + 1 && box.bottom <= body.bottom + 1;
    });
  })).toBe(true);
});

for (const plot of ["Vertical", "Skew-T", "Hodograph"]) {
  test(`${plot} grows on a taller monitor and returns to laptop size without remounting`, async ({ page }, info) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/harness/index.html?mode=studio");
    await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
    await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
    await page.evaluate(() => window.__nimbusStudioHarness!.setRun("nimbus-example://manufactured-supercell-v1", "completed"));
    await expect(page.getByRole("heading", { name: "nimbus-example://manufactured-supercell-v1", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Soundings", exact: true }).click();
    await page.getByRole("button", { name: plot, exact: true }).click();
    const canvas = page.locator(".plot-canvas");
    await expect(canvas.locator(".gtitle")).toBeVisible();
    const original = await canvas.elementHandle();
    const laptopWidth = await canvas.evaluate((element) => element.clientWidth);
    expect(laptopWidth).toBeLessThanOrEqual(680);

    for (const height of [1320, 1440]) {
      await page.setViewportSize({ width: 2560, height });
      await expect.poll(() => canvas.evaluate((element) => element.clientWidth)).toBeGreaterThan(laptopWidth + 200);
      await expect.poll(() => canvas.evaluate((element) => {
        const layout = (element as HTMLElement & { _fullLayout: { width: number } })._fullLayout;
        return Math.abs(layout.width - element.clientWidth);
      })).toBeLessThanOrEqual(1);
      expect(await canvas.evaluate((element, before) => element === before, original)).toBe(true);
      if (plot === "Hodograph") {
        const ratio = await canvas.evaluate((element) => {
          type Axis = { _length: number; range: [number, number] };
          const { xaxis: x, yaxis: y } = (element as HTMLElement & { _fullLayout: { xaxis: Axis; yaxis: Axis } })._fullLayout;
          return (x._length / Math.abs(x.range[1] - x.range[0])) / (y._length / Math.abs(y.range[1] - y.range[0]));
        });
        expect(ratio).toBeCloseTo(1, 5);
      }
      const bar = await page.locator(".plot-studio__frame-bar").boundingBox();
      expect(bar!.y + bar!.height).toBeLessThanOrEqual(height + 1);
      await page.screenshot({ path: info.outputPath(`sounding-${height}.png`) });
    }
    await page.setViewportSize({ width: 1440, height: 900 });
    await expect.poll(() => canvas.evaluate((element) => element.clientWidth)).toBe(laptopWidth);
    expect(await canvas.evaluate((element, before) => element === before, original)).toBe(true);
  });
}

async function openPlot(page: Page, plot: "Composite" | "Vertical", theme: "dark" | "light") {
  await page.setViewportSize({ width: 1100, height: 720 });
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.evaluate((mode) => {
    document.documentElement.dataset.theme = mode;
    window.dispatchEvent(new CustomEvent("nimbus-studio:appearance-applied"));
  }, theme);
  await page.getByRole("button", { name: plot === "Composite" ? "Cross-sections" : "Soundings", exact: true }).click();
  await page.getByRole("button", { name: plot, exact: true }).click();
  await expect(page.locator(".plot-canvas .gtitle")).toBeVisible();
  await page.waitForFunction(() => {
    const graph = document.querySelector(".plot-canvas") as (HTMLElement & {
      _fullLayout?: { width: number; height: number };
    }) | null;
    return graph?._fullLayout && Math.abs(graph._fullLayout.width - graph.clientWidth) < 1
      && Math.abs(graph._fullLayout.height - graph.clientHeight) < 1;
  });
}

for (const plot of ["Composite", "Vertical"] as const) {
  for (const theme of ["dark", "light"] as const) {
    test(`${plot} keeps plot labels and toolbar clear in a compact ${theme} window`, async ({ page }, info) => {
      await openPlot(page, plot, theme);
      const bounds = await page.locator(".plot-canvas").evaluate((element) => {
        const box = (node: Element | null) => {
          if (!node) throw new Error("Missing plot chrome");
          const rect = node.getBoundingClientRect();
          return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
        };
        return {
          body: box(element.closest(".plot-body")),
          title: box(element.querySelector(".gtitle")),
          toolbar: box(element.querySelector(".modebar")),
          xTitle: box(element.querySelector(".xtitle")),
          yTitle: box(element.querySelector(".ytitle")),
        };
      });
      for (const label of [bounds.title, bounds.xTitle, bounds.yTitle]) {
        expect(label.left).toBeGreaterThanOrEqual(bounds.body.left - 1);
        expect(label.right).toBeLessThanOrEqual(bounds.body.right + 1);
        expect(label.top).toBeGreaterThanOrEqual(bounds.body.top - 1);
        expect(label.bottom).toBeLessThanOrEqual(bounds.body.bottom + 1);
      }
      expect(bounds.toolbar.bottom).toBeLessThanOrEqual(bounds.title.top);
      expect(bounds.body.bottom).toBeLessThanOrEqual(720);
      await page.screenshot({ path: info.outputPath("plot.png") });
      if (plot === "Composite") {
        const controls = page.locator(".plot-controls--composite");
        const geometry = await controls.evaluate((element) => ({
          width: element.clientWidth, contentWidth: element.scrollWidth,
        }));
        expect(geometry.contentWidth).toBeLessThanOrEqual(geometry.width + 1);
        for (const name of ["Contour A", "Contour B"]) {
          const field = page.getByRole("group", { name, exact: true }).getByRole("combobox").first();
          await field.scrollIntoViewIfNeeded();
          const hit = await field.evaluate((element) => {
            const rect = element.getBoundingClientRect();
            return document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2) === element;
          });
          expect(hit).toBe(true);
        }
      }
    });
  }
}

test("plot toolbar exposes visible keyboard focus and keyboard actions", async ({ page, browserName }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openPlot(page, "Vertical", "dark");
  const toolbar = page.locator(".plot-canvas .modebar");
  await toolbar.getByRole("button").first().focus();
  await page.keyboard.press(browserName === "webkit" && process.platform === "darwin" ? "Alt+Tab" : "Tab");
  const focus = await page.evaluate(() => {
    const active = document.activeElement as HTMLElement;
    const style = getComputedStyle(active);
    return { inToolbar: Boolean(active.closest(".modebar")), width: parseFloat(style.outlineWidth), style: style.outlineStyle };
  });
  expect(focus.inToolbar).toBe(true);
  expect(focus.width).toBeGreaterThanOrEqual(2);
  expect(focus.style).not.toBe("none");
  await toolbar.getByRole("button", { name: "Pan", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect.poll(() => page.locator(".plot-canvas").evaluate((element) =>
    (element as HTMLElement & { _fullLayout: { dragmode: string } })._fullLayout.dragmode)).toBe("pan");
});

test("animated plot sizing settles the latest geometry without delaying appearance or losing zoom", async ({ page }, info) => {
  await openPlot(page, "Vertical", "dark");
  const canvas = page.locator(".plot-canvas");
  const original = await canvas.elementHandle();
  const drag = (await canvas.locator(".nsewdrag").boundingBox())!;
  await page.mouse.move(drag.x + drag.width * 0.3, drag.y + drag.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(drag.x + drag.width * 0.7, drag.y + drag.height * 0.7, { steps: 5 });
  await page.mouse.up();
  const zoom = await canvas.evaluate(element => {
    const layout = (element as HTMLElement & { _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[] } } })._fullLayout;
    return { x: [...layout.xaxis.range], y: [...layout.yaxis.range] };
  });
  await canvas.evaluate(async element => {
    const graph = element as HTMLElement;
    for (let frame = 0; frame < 36; frame++) {
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      graph.style.width = `${430 + frame * 5}px`;
      graph.style.height = `${360 + frame * 3}px`;
    }
    graph.style.width = "640px";
    graph.style.height = "500px";
    document.documentElement.dataset.textSize = "comfortable";
    window.dispatchEvent(new CustomEvent("nimbus-studio:appearance-applied"));
  });
  const started = Date.now();
  await expect.poll(() => canvas.evaluate(element => {
    const layout = (element as HTMLElement & { _fullLayout: { width: number; height: number; font: { size: number } } })._fullLayout;
    const box = element.getBoundingClientRect();
    return { geometryError: Math.max(Math.abs(layout.width - box.width), Math.abs(layout.height - box.height)), font: layout.font.size };
  }), { timeout: 2_000 }).toEqual({ geometryError: 0, font: 13 });
  expect(await canvas.evaluate((element, before) => element === before, original)).toBe(true);
  const finalZoom = await canvas.evaluate(element => {
    const layout = (element as HTMLElement & { _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[] } } })._fullLayout;
    return { x: layout.xaxis.range, y: layout.yaxis.range };
  });
  expect(finalZoom).toEqual(zoom);
  await info.attach("animated-resize", { body: JSON.stringify({ resizeFrames: 36, settleMs: Date.now() - started, finalZoom }), contentType: "application/json" });
});

test("dense line backing pixels follow viewer density and resize without changing samples or zoom", async ({ browser, baseURL }, info) => {
  test.setTimeout(90_000);
  const rows: unknown[] = [];
  for (const density of [1, 2, 3]) {
    const context = await browser.newContext({ baseURL, viewport: { width: 640, height: 480 }, deviceScaleFactor: density });
    try {
      const page = await context.newPage();
      await page.goto("/harness/index.html?mode=plot-resolution");
      const plot = page.locator(".plot-canvas");
      await expect(plot).toHaveAttribute("data-render-backend", "webgl");
      const geometry = () => plot.evaluate(element => {
        const graph = element as HTMLElement & { _context: { plotGlPixelRatio: number }; data: Array<{ x: number[]; y: number[] }>; _fullLayout: { xaxis: { range: number[] } } };
        return { viewer: window.devicePixelRatio, configured: graph._context.plotGlPixelRatio,
          canvases: Array.from(element.querySelectorAll(".gl-container canvas"), canvas => {
            const pixels = canvas as HTMLCanvasElement, rect = canvas.getBoundingClientRect();
            return { width: pixels.width, height: pixels.height, cssWidth: rect.width, cssHeight: rect.height };
          }).filter(canvas => canvas.cssWidth > 0 && canvas.cssHeight > 0),
          samplesExact: graph.data[0].x.length === 6000 && graph.data[0].x.every((value, index) => value === index / 100)
            && graph.data[0].y.every((value, index) => value === Math.sin(index / 100)),
          range: graph._fullLayout.xaxis.range };
      });
      await expect.poll(async () => (await geometry()).canvases.length).toBeGreaterThan(0);
      const assertDensity = async (expected: number) => {
        await expect.poll(async () => (await geometry()).configured).toBe(expected);
        await expect.poll(async () => (await geometry()).canvases.every(canvas =>
          Math.abs(canvas.width / canvas.cssWidth - expected) < 0.02 && Math.abs(canvas.height / canvas.cssHeight - expected) < 0.02)).toBe(true);
        expect((await geometry()).samplesExact).toBe(true);
      };
      await assertDensity(density);
      await page.getByRole("button", { name: "Zoom in", exact: true }).click();
      const zoom = (await geometry()).range;
      await page.setViewportSize({ width: 740, height: 520 });
      await assertDensity(density);
      expect((await geometry()).range).toEqual(zoom);
      await page.screenshot({ path: info.outputPath(`dense-line-dpr-${density}.png`) });
      rows.push(await geometry());
      if (density === 3) {
        await page.evaluate(() => { Object.defineProperty(window, "devicePixelRatio", { configurable: true, value: 1 }); window.dispatchEvent(new Event("resize")); });
        await assertDensity(1);
        expect((await geometry()).range).toEqual(zoom);
        rows.push({ simulatedDisplayDensityChange: await geometry() });
      }
    } finally { await context.close(); }
  }
  await info.attach("dense-line-density.json", { body: JSON.stringify(rows, null, 2), contentType: "application/json" });
});
