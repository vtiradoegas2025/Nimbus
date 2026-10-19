import { expect, test, type Page } from "@playwright/test";

/*
 * The Python scripting tab, against a real CPython 3.12 compiled to WASM.
 *
 * Why these run a real interpreter
 * --------------------------------
 * The tab is ~2,500 lines that had no end-to-end coverage, because booting
 * Python meant a CDN download on every spec. The browser-test build now serves
 * a vendored copy of the same pinned release, so this is affordable.
 *
 * A stand-in interpreter was considered and rejected. The claims worth testing
 * here -- that a traceback names the line the user is looking at, that one cell
 * can see what another bound, that `source="comparison"` reaches other data --
 * are all claims about what CPython does with what we hand it. A double would
 * answer them with our own assumptions, which is how a suite certifies a bug.
 *
 * Boot costs seconds even locally (wasm compile, numpy, matplotlib) and every
 * test pays it once, hence the timeout.
 */

/*
 * Per-test, via testInfo: `test.setTimeout` is only callable from inside a
 * test body, and at file scope it throws before a single spec runs.
 */
test.beforeEach(({}, testInfo) => {
  testInfo.setTimeout(120_000);
});

const HARNESS = "/harness/python.html";
const TAB = ".python-plot";

test("Python coordinates preserve exact physical axes and never label index geometry as metres", async ({ page }) => {
  await openHarness(page);
  for (const geometry of ["index", "translated", "cylindrical"] as const) {
    const assertions = geometry === "index" ? [
      "assert coordinate_units == ('index', 'index', 'index')",
      "assert x_m.size == y_m.size == z_m.size == 0",
      "assert np.isnan(scalar_origin_m).all()",
      "assert cube.x.values.tolist() == [0., 1.] and cube.x.attrs['units'] == 'index'",
    ] : geometry === "translated" ? [
      "assert coordinate_units == ('m', 'm', 'm')",
      "assert x_m.tolist() == [100., 115.] and y_m.tolist() == [200., 225.] and z_m.tolist() == [300., 345.]",
      "assert scalar_origin_m.tolist() == [100., 200., 300.]",
      "assert cube.x.values.tolist() == [100., 115.] and cube.z.values.tolist() == [300., 345.]",
    ] : [
      "assert coordinate_units == ('m', 'rad', 'm')",
      "assert x_m.tolist() == [100., 110.] and y_m.size == 0",
      "assert theta_rad.tolist() == [200., 200.5]",
      "assert cube.theta.attrs['units'] == 'rad' and cube.theta.values.tolist() == [200., 200.5]",
    ];
    const result = await page.evaluate(({ geometry, assertions }) => window.__nimbusPythonFixture.executeProvider(
      ["cube = dataset()", ...assertions, "result = w.ravel()[:2]"].join("\n"), false, geometry,
    ), { geometry, assertions });
    expect(result).toEqual({ shape: [2], values: [1, 2] });
  }
});

test("large Python heatmaps refine native values on zoom while retaining the color scale and export", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, "import numpy as np\nresult = np.arange(1_440_000, dtype=np.float32).reshape(1200, 1200)");
  await runWholeScript(page);
  await showResult(page);
  const plot = page.locator(".plot-canvas");
  await expect(page.locator(".python-plot__display-note")).toContainText("Zooming refines");
  await page.evaluate(() => window.__nimbusPythonFixture.relayoutArray({ "xaxis.range": [510, 520], "yaxis.range": [610, 620] }));
  await expect.poll(() => plot.evaluate(element => {
    const graph = element as HTMLElement & { data: Array<{ x: number[]; y: number[]; z: number[][]; zmin: number; zmax: number }>;
      _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[] } } };
    const trace = graph.data[0];
    return { x: trace.x, y: trace.y, exact: trace.z.every((row, y) => row.every((v, x) => v === trace.y[y] * 1200 + trace.x[x])),
      min: trace.zmin, max: trace.zmax, xRange: graph._fullLayout.xaxis.range, yRange: graph._fullLayout.yaxis.range };
  })).toEqual({ x: Array.from({ length: 13 }, (_, i) => i + 509), y: Array.from({ length: 13 }, (_, i) => i + 609),
    exact: true, min: 0, max: 1_439_999, xRange: [510, 520], yRange: [610, 620] });
  const path = "/__nimbus_browser_fixture__/zoom-source.npy";
  await chooseSavePath(page, path);
  await page.getByRole("button", { name: "Save .npy", exact: true }).click();
  await expect.poll(() => page.evaluate(path => window.__nimbusPythonFixture.readFileBytes(path)?.byteLength ?? 0, path)).toBeGreaterThan(1_440_000 * 4);
  await page.evaluate(() => window.__nimbusPythonFixture.relayoutArray({ "xaxis.autorange": true, "yaxis.autorange": true }));
  await expect(page.locator(".python-plot__display-note")).toContainText("Zooming refines");
});

test("retained Python arrays stay stable while editing and captured figures release their owner", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, "import numpy as np\nresult = np.arange(300_000, dtype=np.float32).reshape(500, 600)");
  await runWholeScript(page);
  await showResult(page);
  const plot = page.locator(".plot-canvas");
  await expect(plot).toHaveCount(1);
  await plot.evaluate(element => {
    const graph = element as HTMLElement & { data: { z: number[][] }[] };
    Object.assign(window, { retainedPythonTrace: graph.data[0].z });
  });
  await setEditorText(page, "import numpy as np\nresult = np.arange(300_000, dtype=np.float32).reshape(500, 600)\n# editing does not rerun the array");
  await expect(page.locator(".python-plot__stale")).toBeVisible();
  await expect.poll(() => plot.evaluate(element => {
    const graph = element as HTMLElement & { data: { z: number[][] }[] };
    return graph.data[0].z === (window as unknown as { retainedPythonTrace: number[][] }).retainedPythonTrace;
  })).toBe(true);
  await setEditorText(page, "import matplotlib.pyplot as plt\nimport weakref\nresult = plt.figure()\nreference = weakref.ref(result)");
  await runWholeScript(page);
  await setEditorText(page, "del result\nimport gc\ngc.collect()\nprint('figure released', reference() is None)");
  await runWholeScript(page);
  await expect(stdout(page)).toContainText("figure released True");
});

test("Python providers publish the declared result and reject failed or mismatched dtype outputs", async ({ page }) => {
  await openHarness(page);
  const result = await page.evaluate(() => window.__nimbusPythonFixture.executeProvider(
    "display(np.array([99, 99], dtype=np.int32))\nresult = np.array([1.25, 2.5], dtype=np.float32)",
  ));
  expect(result).toEqual({ shape: [2], values: [1.25, 2.5] });
  for (const source of [
    "result = np.array([1, 2], dtype=np.int32)",
    "result = np.array([1, 1 + 1e-12], dtype=np.float64)",
    "display(np.array([1, 2], dtype=np.float32))\nraise ValueError('failed provider')",
  ]) {
    const error = await page.evaluate(async code => {
      try { await window.__nimbusPythonFixture.executeProvider(code); return null; }
      catch (error) { return error instanceof Error ? error.message : String(error); }
    }, source);
    expect(error).toMatch(/dtype float32|raised an exception/);
  }
  const terrain = await page.evaluate(() => window.__nimbusPythonFixture.executeProvider([
    "cube = dataset()",
    "assert cube.w.attrs['units'] == 'm/s'",
    "assert cube.x.values.tolist() == [20., 30.]",
    "assert cube.y.values.tolist() == [15., 25.]",
    "assert cube.z.values.tolist() == [2, 3]",
    "assert cube.z.attrs['units'] == 'model_level'",
    "assert z_m.size == 0 and np.isnan(scalar_origin_m[2])",
    "assert step_index == 42 and frame_index is None and simulation_time_s == 300",
    "result = cube.w.values.ravel()[:2]",
  ].join("\n"), true));
  expect(terrain).toEqual({ shape: [2], values: [1, 2] });
});

test("ordinary Python provider reads preserve values and provenance after same-version input changes and warm reuse", async ({ page }, info) => {
  await openHarness(page);
  const result = await page.evaluate(() => window.__nimbusPythonFixture.verifyProviderCache());
  expect(result.uMatches).toBe(true);
  expect(result.vMatches).toBe(true);
  expect(result.warmMatches).toBe(true);
  expect(result.distinctInputs).toBe(true);
  expect(result.warmReused).toBe(true);
  expect(result.warmInputIdentity).toBe(result.expectedIdentity);
  await info.attach("ordinary-provider-cache", { body: JSON.stringify(result), contentType: "application/json" });
});

test("large CSV export runs in its worker without detaching results and can be cancelled", async ({ page }, info) => {
  await page.addInitScript(() => {
    const original = Worker.prototype.postMessage;
    const state = { calls: 0, transferredInputs: 0, held: false, source: null as Float64Array | null };
    Object.assign(window, { pythonExportAudit: state });
    Worker.prototype.postMessage = function(this: Worker, ...args: Parameters<Worker["postMessage"]>) {
      const request = args[0] as { format?: string; data?: Float64Array };
      if (request?.format === "csv") {
        state.calls++;
        state.source = request.data ?? null;
        if (Array.isArray(args[1])) state.transferredInputs += args[1].length;
        if (state.held) return;
      }
      return original.apply(this, args);
    } as Worker["postMessage"];
  });
  await openHarness(page);
  await setEditorText(page, "import numpy as np\nresult = np.linspace(-1.123, 1.987, 1_000_000, dtype=np.float64).reshape(1000, 1000)");
  await runWholeScript(page);
  await showResult(page);
  const path = "/__nimbus_browser_fixture__/large-worker.csv";
  await chooseSavePath(page, path);
  await page.getByRole("button", { name: "Save CSV", exact: true }).click();
  await expect.poll(() => page.evaluate(value => {
    const bytes = window.__nimbusPythonFixture.readFileBytes(value);
    if (!bytes) return null;
    const lines = new TextDecoder().decode(bytes).trim().split("\n");
    return { rows: lines.length, first: Number(lines[0].split(",")[0]), last: Number(lines.at(-1)!.split(",").at(-1)) };
  }, path)).toEqual({ rows: 1000, first: -1.123, last: 1.987 });
  const audit = await page.evaluate(() => {
    const state = (window as unknown as { pythonExportAudit: { calls: number; transferredInputs: number; source: Float64Array } }).pythonExportAudit;
    return { calls: state.calls, transferredInputs: state.transferredInputs, retainedBytes: state.source.byteLength };
  });
  expect(audit).toEqual({ calls: 1, transferredInputs: 0, retainedBytes: 8_000_000 });
  await info.attach("numeric-export-worker", { body: JSON.stringify(audit), contentType: "application/json" });
  await page.evaluate(() => { (window as unknown as { pythonExportAudit: { held: boolean } }).pythonExportAudit.held = true; });
  const cancelledPath = "/__nimbus_browser_fixture__/cancelled-worker.csv";
  await chooseSavePath(page, cancelledPath);
  await page.getByRole("button", { name: "Save CSV", exact: true }).click();
  await expect(page.getByRole("button", { name: "Cancel export", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Cancel export", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Array export cancelled.");
  expect(await diskBytes(page, cancelledPath)).toBeNull();
});

test("numeric outputs preserve dtype and precision through plots and exports", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    "import numpy as np",
    "a = np.array([16777216, 16777217], dtype=np.int64)",
    "display(a, label='integers')",
    "a[:] = 0",
    "display(np.array([1., 1. + 1e-9], dtype=np.float64), label='doubles')",
    "display(np.array([9007199254740993], dtype=np.int64), label='large integer')",
  ].join("\n"));
  await runWholeScript(page);
  await showResult(page);
  await expect(page.locator(".plot-canvas")).toHaveCount(2);
  const values = await page.locator(".plot-canvas").evaluateAll((elements) => elements.map((element) =>
    (element as HTMLElement & { data: Array<{ y: number[] }> }).data[0].y));
  expect(values).toEqual([[16777216, 16777217], [1, 1 + 1e-9]]);
  await expect(page.locator(".python-plot__empty")).toContainText("exact integer range");
  const target = "/__nimbus_browser_fixture__/precision.npy";
  await chooseSavePath(page, target);
  await page.getByRole("button", { name: "Save .npy", exact: true }).last().click();
  await expect.poll(() => diskBytes(page, target)).not.toBeNull();
  const bytes = Uint8Array.from((await diskBytes(page, target))!);
  const view = new DataView(bytes.buffer);
  const offset = 10 + view.getUint16(8, true);
  expect(new TextDecoder().decode(bytes.subarray(10, offset))).toContain("'<i8'");
  expect(view.getBigInt64(offset, true)).toBe(9007199254740993n);
  await chooseSavePath(page, "/__nimbus_browser_fixture__/precision.csv");
  await page.getByRole("button", { name: "Save CSV", exact: true }).last().click();
  await expect.poll(() => diskContents(page, "/__nimbus_browser_fixture__/precision.csv")).toBe("9007199254740993\n");
});

test("figure capture honors DPI, fits previews, exports SVG, and refuses excessive raster size", async ({ page }) => {
  await page.setViewportSize({ width: 2560, height: 1320 });
  await openHarness(page);
  await setEditorText(page, [
    "import matplotlib.pyplot as plt",
    "fig, ax = plt.subplots(figsize=(6,4), dpi=300)",
    "ax.plot([0,1], [0,1])",
    "display(fig)",
    "plt.close(fig)",
    "large = plt.figure(figsize=(32,32), dpi=300)",
    "try:",
    "    display(large)",
    "except ValueError as error:",
    "    print(error)",
    "finally:",
    "    plt.close(large)",
  ].join("\n"));
  await runWholeScript(page);
  await showResult(page);
  const figure = page.locator(".python-plot__fig");
  await expect(figure).toBeVisible();
  const size = await figure.evaluate((element: HTMLImageElement) => ({
    native: element.naturalWidth, width: element.getBoundingClientRect().width, height: element.getBoundingClientRect().height,
  }));
  await expect(figure).toHaveAttribute("src", /^data:image\/svg\+xml/);
  expect(size.width).toBeGreaterThan(300);
  expect(size.height).toBeLessThanOrEqual(1320 * 0.7 + 1);
  const pngTarget = "/__nimbus_browser_fixture__/figure.png";
  await chooseSavePath(page, pngTarget);
  await page.getByRole("button", { name: "Save PNG", exact: true }).click();
  await expect.poll(() => diskBytes(page, pngTarget)).not.toBeNull();
  const pngWidth = await page.evaluate(async (bytes) => {
    const url = URL.createObjectURL(new Blob([new Uint8Array(bytes!)], { type: "image/png" }));
    try {
      const image = new Image(); image.src = url; await image.decode(); return image.naturalWidth;
    } finally { URL.revokeObjectURL(url); }
  }, await diskBytes(page, pngTarget));
  expect(pngWidth).toBeGreaterThan(1500);
  const target = "/__nimbus_browser_fixture__/figure.svg";
  await chooseSavePath(page, target);
  await page.getByRole("button", { name: "Save SVG", exact: true }).click();
  await expect.poll(() => diskContents(page, target)).toContain("<svg");
  await showConsole(page);
  await expect(stdout(page)).toContainText("16 megapixel");
  await expect(stderr(page)).toHaveCount(0);
});

test("narrow Python gallery leaves execution and file controls reachable", async ({ page }) => {
  await page.setViewportSize({ width: 640, height: 700 });
  await openHarness(page);
  await page.getByRole("button", { name: "Gallery", exact: true }).click();
  for (const name of ["Run script", "Restart", "Save as…", "Starter"]) {
    const control = page.getByRole("button", { name, exact: true });
    expect(await control.evaluate((element) => {
      const rect = element.getBoundingClientRect();
      return element.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2));
    }), name).toBe(true);
  }
  await page.getByTitle("Hide gallery", { exact: true }).click();
  await expect(page.locator(".plugin-gallery")).toHaveCount(0);
});

/**
 * Replace the editor's contents in one insertion.
 *
 * Typing key-by-key would be corrupted by the editor's own auto-indent: every
 * newline inside a block gains leading whitespace, so a script typed that way
 * is not the script being asserted on. `insertText` delivers the whole string
 * as a single input event, which CodeMirror applies verbatim.
 */
async function setEditorText(page: Page, text: string): Promise<void> {
  await page.locator(".cm-content").click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(text);
}

/** Runs finished so far, read from the tab's own counter. */
async function runsCompleted(page: Page): Promise<number> {
  return Number((await page.locator(TAB).getAttribute("data-run-count")) ?? "0");
}

/**
 * Wait for exactly one more run than `previous` to have finished.
 *
 * Waiting for "not running" instead would be a race: that is also true in the
 * moment between the click and the state update, so a fast run is indeterminate
 * and a slow one only passes by luck of timing.
 */
async function expectRunAfter(page: Page, previous: number): Promise<void> {
  await expect(page.locator(TAB))
    .toHaveAttribute("data-run-count", String(previous + 1), { timeout: 90_000 });
}

/**
 * Run the whole script and wait for it.
 *
 * The header's Run script button, not the cell bar's "Run all": the cell bar renders
 * run controls only for a document that HAS multiple cells, so a single-cell
 * script has no "Run all" to click. "Run script" is an exact accessible name, which
 * "Run cell" and "Run all" do not collide with.
 */
async function runWholeScript(page: Page): Promise<void> {
  const before = await runsCompleted(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();
  await expectRunAfter(page, before);
}

/** Run just the cell the caret is in, and wait for it. */
async function runCurrentCell(page: Page): Promise<void> {
  const before = await runsCompleted(page);
  await page.getByRole("button", { name: "Run cell", exact: true }).click();
  await expectRunAfter(page, before);
}

/**
 * Bring the stdout/stderr pane to the front.
 *
 * Needed after every run, not just the first: a run that ends cleanly switches
 * the tab to Result, and one that writes to stderr switches it to the console.
 * So a spec asserting on printed output has to re-select the pane each time,
 * or it asserts against a console that is no longer mounted.
 */
async function showConsole(page: Page): Promise<void> {
  await page.getByRole("tab", { name: "stdout / stderr" }).click();
}

async function showResult(page: Page): Promise<void> {
  await page.getByRole("tab", { name: "Result", exact: true }).click();
}

async function openHarness(page: Page): Promise<void> {
  await page.goto(HARNESS);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();
}

/** The unscoped editor recovery store — the one used when no project is open. */
const UNSCOPED_STORE = "nimbus-studio:python-workspace:v1";
const storeFor = (projectId: string) => `${UNSCOPED_STORE}:${projectId}`;

/**
 * Point the harness at a project and wait for the tab strip to agree.
 *
 * Waiting on the strip's own scope attribute, not on the select's value: the
 * select updates first, and asserting on it would pass in the window before the
 * buffers have been swapped — which is precisely the window the bugs live in.
 */
async function selectProject(page: Page, projectId: string | null): Promise<void> {
  await page.getByLabel("Harness project").selectOption(projectId ?? "");
  await expect(page.locator(".python-plot__tabs"))
    .toHaveAttribute("data-script-scope", projectId ?? "none");
}

const stdout = (page: Page) => page.locator(".python-plot__console-stdout");
const stderr = (page: Page) => page.locator(".python-plot__console-stderr");

/**
 * The run store, read directly.
 *
 * The panel's results outlive the panel, so a spec that unmounts it has no DOM
 * left to wait on — and sleeping for "long enough" instead is how a test
 * becomes a coin toss. This is the only way to wait on a run nothing is
 * watching.
 */
async function runStore(page: Page): Promise<{ running: boolean; completedRuns: number }> {
  return page.evaluate(() => (
    window as unknown as {
      __nimbusPythonSession: (
        projectId: string | null,
        paneId: string,
      ) => { running: boolean; completedRuns: number };
    }
  ).__nimbusPythonSession(null, "primary"));
}

/* ---------- The virtual disk ---------- */

/**
 * Contents that exist ONLY on disk.
 *
 * Distinctive on purpose: the point of asserting on it is to tell a buffer that
 * was re-read from the file apart from one whose text rode along in
 * localStorage, and text either side could plausibly hold proves neither.
 */
const ON_DISK_ANALYSIS = "# read back from the real file\nvalue = 41 + 1\n";
/* One line of it. CodeMirror renders each line in its own element with no
 * newline between them, so the whole string never appears as contiguous text
 * in the DOM and matching on it would fail against a correct editor. */
const ON_DISK_MARKER = "read back from the real file";

declare global {
  interface Window {
    __nimbusPythonFixture: {
      writeFile: (path: string, contents: string) => void;
      writeBinaryFile: (path: string, contents: readonly number[]) => void;
      readFile: (path: string) => string | null;
      readFileBytes: (path: string) => Uint8Array | null;
      setSaveDialogPath: (path: string | null | undefined) => void;
      setOpenDialogPath: (path: string | null | undefined) => void;
      pluginsDir: string;
      executeProvider: (source: string, terrain?: boolean, geometry?: "index" | "translated" | "cylindrical") => Promise<{ values: number[]; shape: readonly number[] }>;
      verifyProviderCache: () => Promise<{
        shape: number[]; uMatches: boolean; vMatches: boolean; warmMatches: boolean; distinctInputs: boolean; warmReused: boolean;
        warmInputIdentity: string | undefined; expectedIdentity: string;
      }>;
      holdScriptIo: (command: "read_config" | "write_config", path: string) => void;
      releaseScriptIo: () => void;
      scriptIoWaiting: () => boolean;
      relayoutArray: (event: Record<string, unknown>) => Promise<void>;
    };
  }
}

/** Put a file on the fixture's disk. The reader is not permissive — a path
 *  nothing has written reports not_found, as the Rust command does. */
async function seedFile(page: Page, path: string, contents: string): Promise<void> {
  await page.evaluate(
    ([target, body]) => window.__nimbusPythonFixture.writeFile(target, body),
    [path, contents] as const,
  );
}

async function seedBinaryFile(page: Page, path: string, contents: string): Promise<void> {
  await page.evaluate(
    ([target, bytes]) => window.__nimbusPythonFixture.writeBinaryFile(target, bytes),
    [path, [...new TextEncoder().encode(contents)]] as const,
  );
}

/** Read what actually landed on disk, rather than what the UI says landed. */
async function diskContents(page: Page, path: string): Promise<string | null> {
  return page.evaluate(
    ([target]) => window.__nimbusPythonFixture.readFile(target),
    [path] as const,
  );
}

/**
 * The exact bytes on disk, for the exports that are not text.
 *
 * Returned as a plain array because a Uint8Array does not survive Playwright's
 * evaluate boundary as one — it arrives as an object with numeric keys, which
 * compares equal to almost nothing and fails in a way that reads as the export
 * having been wrong.
 */
async function diskBytes(page: Page, path: string): Promise<number[] | null> {
  return page.evaluate(
    ([target]) => {
      const bytes = window.__nimbusPythonFixture.readFileBytes(target);
      return bytes ? Array.from(bytes) : null;
    },
    [path] as const,
  );
}

/** Decide what the next Save dialog returns. `null` is the user cancelling. */
async function chooseSavePath(page: Page, path: string | null): Promise<void> {
  await page.evaluate(
    ([target]) => window.__nimbusPythonFixture.setSaveDialogPath(target),
    [path] as [string | null],
  );
}

async function pluginsDir(page: Page): Promise<string> {
  return page.evaluate(() => window.__nimbusPythonFixture.pluginsDir);
}

/**
 * One entry in the plugin gallery.
 *
 * Scoped to the list, because an open tab for the same file contributes a
 * "Close <name>" button and an unscoped role query matches both. Naming the
 * list is also what keeps this asserting on the gallery rather than on any
 * control that happens to mention the file.
 */
function galleryEntry(page: Page, name: string) {
  return page.locator(".plugin-gallery__list").getByRole("button", { name: new RegExp(name) });
}

test("boots a real interpreter and runs the starter script end to end", async ({ page }) => {
  await openHarness(page);
  await runWholeScript(page);
  await showConsole(page);

  /*
   * One assertion covering the whole chain: the interpreter booted from the
   * vendored files, numpy was preloaded, the bounded frame loader returned a
   * real field plus coordinate metadata, the grid globals resolved, and
   * print() was captured. Any link failing leaves this text absent.
   *
   * "source: open frame" is the half that matters most here — the starter
   * falls back to an explicitly labelled plotting check when no field can be
   * loaded, and a broken field bridge must not masquerade as a data figure.
   */
  await expect(stdout(page)).toContainText("source: open frame");
  await expect(stdout(page)).toContainText("nearest-cell display · no interpolation");
  await expect(stderr(page)).toHaveCount(0);

  /*
   * The fidelity-first starter deliberately emits one honest figure rather
   * than a gallery of derived structures. Multi-output behavior is exercised
   * separately below.
   */
  await page.getByRole("tab", { name: "Result", exact: true }).click();
  await expect(page.locator(".python-plot__empty")).toHaveCount(0);
  expect(await page.locator(".python-plot__output").count()).toBeGreaterThan(0);
});

test("an unanswered Python bootstrap recovers before user code and preserves the resulting namespace", async ({ page }) => {
  await page.addInitScript(() => {
    const audit = { workers: [] as string[], droppedBoots: 0, terminated: 0, userCalls: [] as string[] };
    Object.assign(window, { pythonBootAudit: audit });
    const OriginalWorker = window.Worker;
    window.Worker = new Proxy(OriginalWorker, {
      construct(target, args) {
        const worker = Reflect.construct(target, args) as Worker;
        const url = String(args[0]);
        if (!url.includes("pyodideWorker")) return worker;
        const hold = audit.workers.length === 0;
        audit.workers.push(url);
        const post = worker.postMessage.bind(worker);
        worker.postMessage = ((message: { kind?: string; op?: string; id?: number; args?: unknown[] }, options?: unknown) => {
          if (hold && message.kind === "request" && message.op === "boot") { audit.droppedBoots++; return; }
          if (message.kind === "request" && message.op === "runCell") audit.userCalls.push(String(message.args?.[0]));
          Reflect.apply(post, worker, options === undefined ? [message] : [message, options]);
        }) as typeof worker.postMessage;
        const terminate = worker.terminate.bind(worker);
        worker.terminate = () => { audit.terminated++; terminate(); };
        return worker;
      },
    });
  });
  await openHarness(page);
  await setEditorText(page, "answer = 41\nprint('startup recovery ran once')");
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("startup recovery ran once");
  await setEditorText(page, "answer += 1\nprint(answer)");
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("42");
  const audit = await page.evaluate(() => (window as unknown as { pythonBootAudit: {
    workers: string[]; droppedBoots: number; terminated: number; userCalls: string[];
  } }).pythonBootAudit);
  expect(audit.workers).toHaveLength(2);
  expect(new Set(audit.workers).size).toBe(2);
  expect(audit.droppedBoots).toBe(1);
  expect(audit.terminated).toBe(1);
  expect(audit.userCalls).toEqual(["answer = 41\nprint('startup recovery ran once')", "answer += 1\nprint(answer)"]);
});

test("print-only output opens the console, stays with its script, and clears without resetting variables", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, 'kept_marker = 42\nprint("owned-console", kept_marker)');
  await runWholeScript(page);

  const consoleTab = page.getByRole("tab", { name: "stdout / stderr" });
  await expect(consoleTab).toHaveAttribute("aria-selected", "true");
  await expect(stdout(page)).toContainText("owned-console 42");
  await expect(consoleTab).toContainText("stdout / stderr");

  await page.getByRole("button", { name: "New scratch buffer" }).click();
  await expect(page.getByRole("tab", { name: "Scratch 2", exact: true }))
    .toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".python-plot__output-pane")).not.toContainText("owned-console");
  await expect(page.getByRole("button", { name: "Clear output" })).toBeDisabled();

  await page.locator(".python-plot__buffer-name").first().click();
  await expect(stdout(page)).toContainText("owned-console 42");
  await page.getByRole("button", { name: "Clear output" }).click();
  await expect(page.locator(".python-plot__output-pane")).not.toContainText("owned-console");
  await expect(page.getByRole("button", { name: "Clear output" })).toBeDisabled();

  await setEditorText(page, 'print("namespace-survived", kept_marker)');
  await runWholeScript(page);
  await expect(stdout(page)).toContainText("namespace-survived 42");
});

test("starter replacement, script tabs, and editor search preserve deliberate keyboard control", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, "# unsaved surgery marker\n");

  page.once("dialog", (dialog) => void dialog.dismiss());
  await page.getByRole("button", { name: "Starter", exact: true }).click();
  await expect(page.locator(".cm-content")).toContainText("unsaved surgery marker");

  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Starter", exact: true }).click();
  await expect(page.locator(".cm-content")).toContainText("Fidelity-first starter");

  await page.getByRole("button", { name: "New scratch buffer" }).click();
  const second = page.getByRole("tab", { name: "Scratch 2", exact: true });
  await expect(second).toHaveAttribute("tabindex", "0");
  await second.focus();
  await second.press("ArrowLeft");
  const first = page.getByRole("tab", { name: "Scratch", exact: true });
  await expect(first).toBeFocused();
  await expect(first).toHaveAttribute("aria-selected", "true");

  await page.getByRole("button", { name: "Close Scratch", exact: true }).click();
  await expect(second).toBeFocused();
  await expect(second).toHaveAttribute("tabindex", "0");

  await page.locator(".cm-content").click();
  await page.keyboard.press("ControlOrMeta+f");
  const search = page.locator(".cm-search input").first();
  await expect(search).toBeVisible();
  await expect(search).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(search).toHaveCount(0);
});

/*
 * The starter draws something before there is anything to draw.
 *
 * Run used to return silently with no frame open — no output, no error, no way
 * to tell a dead button from a slow one. It is also the state a user is in the
 * first time they open the tab, so it is the worst possible moment for the
 * panel to look broken. The starter carries an explicitly non-meteorological
 * plotting check for exactly this case, and labels it so it cannot be mistaken
 * for simulation output.
 */
test("the starter script renders with no dataset open at all", async ({ page }) => {
  await page.goto(`${HARNESS}?frames=none`);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();

  await runWholeScript(page);
  await showConsole(page);

  await expect(stdout(page)).toContainText("NO SIMULATION FIELD WAS USED");
  await expect(stderr(page)).toHaveCount(0);

  await page.getByRole("tab", { name: "Result", exact: true }).click();
  expect(await page.locator(".python-plot__output").count()).toBeGreaterThan(0);
  await expect(page.locator(".python-plot__empty")).toHaveCount(0);
});

test("a cell runs alone, and cells share one namespace", async ({ page }) => {
  await openHarness(page);
  await setEditorText(
    page,
    ["# %% one", "alpha = 41 + 1", "", "# %% two", 'print("alpha is", alpha)'].join("\n"),
  );

  // Put the caret in the second cell by clicking its text.
  await page.locator(".cm-content").getByText("alpha is").click();
  await expect(page.locator(".python-plot__cell-position")).toContainText("two");

  /*
   * Running the second cell must NOT run the first. Were "Run cell" quietly
   * executing the whole document this would succeed, so the NameError is the
   * assertion: alpha was never bound.
   */
  await runCurrentCell(page);
  await showConsole(page);
  await expect(stderr(page)).toContainText("NameError");

  // Running everything binds alpha, and the second cell reads it.
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("alpha is 42");

  // Now that the namespace holds alpha, the cell that failed a moment ago
  // succeeds alone -- which is the point of sharing the namespace.
  await page.locator(".cm-content").getByText("alpha is").click();
  await runCurrentCell(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("alpha is 42");
  await expect(stderr(page)).toHaveCount(0);
});

test("each cell keeps its own output", async ({ page }) => {
  await openHarness(page);
  /*
   * display() rather than print(): the two panes are scoped differently on
   * purpose. Rendered output belongs to a cell and follows the caret, which is
   * the property under test here; the console belongs to a RUN and is asserted
   * in the tests above.
   */
  await setEditorText(
    page,
    ["# %% one", 'display("marker-one")', "# %% two", 'display("marker-two")'].join("\n"),
  );
  const pane = page.locator(".python-plot__output-pane");

  await page.locator(".cm-content").getByText("marker-one").click();
  await runCurrentCell(page);
  await showResult(page);
  await expect(pane).toContainText("marker-one");

  await page.locator(".cm-content").getByText("marker-two").click();
  await runCurrentCell(page);
  await showResult(page);
  await expect(pane).toContainText("marker-two");
  await expect(pane).not.toContainText("marker-one");

  /*
   * Back to the first cell, WITHOUT re-running: its output is still there.
   * This is the assertion the previous single-result model could not satisfy --
   * running the second cell used to discard the first's figure outright.
   */
  await page.locator(".cm-content").getByText("marker-one").click();
  await showResult(page);
  await expect(pane).toContainText("marker-one");
  await expect(pane).not.toContainText("marker-two");

  // An output that still matches its cell is not flagged...
  await expect(page.locator(".python-plot__stale")).toHaveCount(0);

  // ...and one whose cell has since been edited is. The caret lands in the
  // second cell after the rewrite, which is the cell that changed.
  await setEditorText(
    page,
    ["# %% one", 'display("marker-one")', "# %% two", 'display("marker-two EDITED")'].join("\n"),
  );
  await expect(page.locator(".python-plot__stale")).toBeVisible();

  /*
   * Run All keeps every cell's output too, rather than collapsing the document
   * into one result: afterwards each cell still answers for itself.
   */
  await runWholeScript(page);
  await expect(page.locator(".python-plot__stale")).toHaveCount(0);
  await page.locator(".cm-content").getByText("marker-one").click();
  await showResult(page);
  await expect(pane).toContainText("marker-one");
  await expect(pane).not.toContainText("EDITED");
});

test("a traceback names the line the user is looking at", async ({ page }) => {
  await openHarness(page);
  /*
   * The raise sits on file line 6, 1-based. Running only its cell hands the
   * interpreter a fragment starting at line 5, and an unpadded fragment would
   * report line 2 -- pointing at a blank line four above the mistake. This is
   * what the padding in pythonCells.ts exists for, and only the interpreter
   * that produces the traceback can settle it.
   */
  await setEditorText(page, [
    "# %% setup", // 1
    "value = 1", // 2
    "", // 3
    "", // 4
    "# %% boom", // 5
    'raise ValueError("marker-xyz")', // 6
  ].join("\n"));

  await page.locator(".cm-content").getByText("marker-xyz").click();
  await runCurrentCell(page);
  await showConsole(page);

  await expect(stderr(page)).toContainText("marker-xyz");
  await expect(stderr(page)).toContainText("line 6");
  // Named explicitly: line 2 is where the raise lands with no padding, so its
  // absence is what shows the padding is doing something.
  await expect(stderr(page)).not.toContainText("line 2");
});

test("source= reaches a second dataset, and an unknown one says which exist", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    'a = await load_field("w", frame=0)',
    'b = await load_field("w", frame=0, source="comparison")',
    'print("differs", a.shape != b.shape)',
    'print("open", sorted(datasets))',
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);

  /*
   * The comparison dataset is the fixture's coarse lattice: a different sampler
   * at a different shape. Asserting the two DIFFER is what separates "source=
   * was accepted" from "source= reached other data" -- an argument that is
   * parsed and then ignored would satisfy the weaker check.
   */
  await expect(stdout(page)).toContainText("differs True");
  await expect(stdout(page)).toContainText("comparison");

  await setEditorText(page, 'await load_field("w", source="nope")');
  await runWholeScript(page);
  await expect(stderr(page)).toContainText("unknown source");
  // The message must list what IS open, or it tells the user they are wrong
  // without telling them what would be right.
  await expect(stderr(page)).toContainText("comparison");
});

test("stride and box are applied by the reader with exact crop-then-sample semantics", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    'coarse = await load_field("w", stride=(3, 4, 5))',
    'point = await load_field("w", box=((3, 4), (4, 5), (5, 6)))',
    'cropped = await load_field("w", stride=(3, 4, 5), box=((3, 20), (4, 25), (5, 20)))',
    'meta = field_metadata("w")',
    'print("coarse shape", coarse.shape)',
    'print("cropped shape", cropped.shape)',
    'print("same source cell", bool(np.isclose(coarse[1, 1, 1], point[0, 0, 0])))',
    'print("crop starts exactly", bool(np.isclose(cropped[0, 0, 0], point[0, 0, 0])))',
    'print("coords start", [axis["values"][0] for axis in meta["coordinates"]])',
    'print("coords step", [axis["values"][1] - axis["values"][0] for axis in meta["coordinates"]])',
    'print("fidelity", meta["sourceShape"], meta["outputShape"], meta["stride"], meta["sampling"], meta["dtype"], meta["sourceFormat"])',
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);

  await expect(stdout(page)).toContainText("coarse shape (16, 9, 5)");
  await expect(stdout(page)).toContainText("cropped shape (6, 6, 3)");
  await expect(stdout(page)).toContainText("same source cell True");
  await expect(stdout(page)).toContainText("crop starts exactly True");
  await expect(stdout(page)).toContainText("coords start [375, 700, 450]");
  await expect(stdout(page)).toContainText("coords step [375, 700, 450]");
  await expect(stdout(page)).toContainText("fidelity [48, 36, 24] [6, 6, 3] [3, 4, 5] point_sample float32 npy");
  await expect(stderr(page)).toHaveCount(0);
});

test("raw terrain reads budget geometry without forcing a vertical remap", async ({ page }) => {
  await page.goto(`${HARNESS}?terrain=1`);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();
  await setEditorText(page, [
    "import numpy as np",
    "raw = await load_field('w')",
    "sampled = await load_field('w', stride=(2,3,2))",
    "assert np.array_equal(raw[::2,::3,::2], sampled)",
    "print('raw terrain', raw.shape, sampled.shape)",
    "print('remap', field_metadata('w')['coordinateResampling'])",
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("raw terrain (48, 36, 24) (24, 12, 12)");
  await expect(stdout(page)).toContainText("remap none");
  await expect(stderr(page)).toHaveCount(0);
});

test("terrain fields are conservatively remapped onto physical z before Python receives them", async ({ page }) => {
  await page.goto(`${HARNESS}?terrain=1`);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();
  await setEditorText(page, [
    'remapped = await load_field("w", stride=(2, 3, 2), terrain_remap=True)',
    'meta = field_metadata("w")',
    'print("shape", remapped.shape)',
    'print("source terrain", meta["sourceTerrainFollowing"])',
    'print("resampling", meta["coordinateResampling"])',
    'print("z units", meta["coordinates"][2]["unit"])',
    'print("z monotonic", bool(np.all(np.diff(meta["coordinates"][2]["values"]) > 0)))',
    'dbz = await load_field("reflectivity_dbz", stride=(2, 3, 2), terrain_remap=True)',
    'dbz_meta = field_metadata("reflectivity_dbz")',
    'print("w aggregation", meta["aggregation"], meta["weighting"])',
    'print("dbz aggregation", dbz_meta["aggregation"])',
    'print("terrain fidelity", dbz_meta["sourceShape"], dbz_meta["outputShape"], dbz_meta["stride"], dbz_meta["sourceFormat"])',
  ].join("\n"));

  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("shape (24, 12, 12)");
  await expect(stdout(page)).toContainText("source terrain True");
  await expect(stdout(page)).toContainText("resampling conservative-terrain-to-cartesian");
  await expect(stdout(page)).toContainText("z units m");
  await expect(stdout(page)).toContainText("z monotonic True");
  await expect(stdout(page)).toContainText("w aggregation cell_mean physical_cell_volume");
  await expect(stdout(page)).toContainText("dbz aggregation linear_reflectivity_mean");
  await expect(stdout(page)).toContainText("terrain fidelity [48, 36, 24] [24, 12, 12] [2, 3, 2] npy");
  await expect(stderr(page)).toHaveCount(0);
});

test("load_dataset keeps exact sampled coordinates and per-variable fidelity provenance", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    'ds = await load_dataset(("w", "u"), stride=(3, 4, 5), box=((3, 20), (4, 25), (5, 20)))',
    'print("dataset sizes", dict(ds.sizes))',
    'print("dataset starts", [int(ds.coords[axis][0]) for axis in ("x", "y", "z")])',
    'print("dataset steps", [int(ds.coords[axis][1] - ds.coords[axis][0]) for axis in ("x", "y", "z")])',
    'print("variable fidelity", ds.w.attrs["nimbus_sampling"], ds.w.attrs["nimbus_source_shape"], ds.w.attrs["nimbus_output_shape"], ds.w.attrs["nimbus_stride"], ds.w.attrs["nimbus_source_format"])',
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);

  await expect(stdout(page)).toContainText("dataset sizes {'x': 6, 'y': 6, 'z': 3}");
  await expect(stdout(page)).toContainText("dataset starts [375, 700, 450]");
  await expect(stdout(page)).toContainText("dataset steps [375, 700, 450]");
  await expect(stdout(page)).toContainText("variable fidelity point_sample [48, 36, 24] [6, 6, 3] [3, 4, 5] npy");
  await expect(stderr(page)).toHaveCount(0);
});

test("output appears while the cell is still running", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    "for i in range(4):",
    '    print("tick", i)',
    "    await tick(0.6)",
    'print("done")',
  ].join("\n"));

  // The pane has to be showing before the run starts: the tab only switches to
  // it once a run has finished.
  await showConsole(page);
  const before = await runsCompleted(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();

  /*
   * Both halves matter. The banner proves the run is still in flight, and the
   * first tick proves output crossed the boundary before the cell returned --
   * the banner alone would also show for a run that streamed nothing.
   */
  await expect(page.locator(".python-plot__console-live")).toBeVisible();
  await expect(stdout(page)).toContainText("tick 0");

  await expectRunAfter(page, before);
  // Re-selected because the clean finish moved the tab to Result -- without
  // this the "no live banner" check below would pass on an unmounted pane.
  await showConsole(page);
  await expect(page.locator(".python-plot__console-live")).toHaveCount(0);
  await expect(stdout(page)).toContainText("done");
});

test("a traceback frame jumps the caret to that line", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    "# %% first", // 1
    "value = 1", // 2
    "", // 3
    "# %% second", // 4
    'raise ValueError("jump-marker")', // 5
  ].join("\n"));

  await runWholeScript(page);
  await showConsole(page);

  // Park the caret in the FIRST cell, so a caret that ends up in the second
  // can only have been moved by the click.
  await page.locator(".cm-content").getByText("value = 1").click();
  await expect(page.locator(".python-plot__cell-position")).toContainText("first");

  /*
   * Only the user's own frame is a button. The stdlib frames above it are
   * plain text, so this locator finding exactly one thing is itself part of
   * the assertion.
   */
  const frame = page.locator(".python-plot__tb-link");
  await expect(frame).toHaveCount(1);
  await expect(frame).toContainText("line 5");
  await frame.click();

  await expect(page.locator(".python-plot__cell-position")).toContainText("second");
});

test("the namespace panel shows the user's bindings and hides the helpers", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, "zephyr_total = 7\nzephyr_name = 'storm'");
  await runWholeScript(page);

  await page.getByRole("button", { name: "Variables" }).click();
  const panel = page.locator(".python-plot__packages");
  await expect(panel).toContainText("zephyr_total");
  await expect(panel).toContainText("zephyr_name");

  /*
   * The helpers this module injects are excluded. Without that the panel lists
   * two dozen names the user never bound, and the two they did bind are lost
   * among them -- which defeats the point of a panel that answers "what have
   * I got".
   */
  await expect(panel).not.toContainText("load_field");
  await expect(panel).not.toContainText("save_path");
});

test("completions come from the live namespace, not a static list", async ({ page }) => {
  await openHarness(page);

  /*
   * Bind a name whose attributes no shipped table could know, then complete
   * against it. `zephyr_marker` exists only because this cell ran, so an
   * offered `gust_speed` proves the popup is reading the real interpreter.
   */
  await setEditorText(
    page,
    ["class Zephyr:", "    def gust_speed(self):", "        return 1", "", "zephyr_marker = Zephyr()"].join("\n"),
  );
  await runWholeScript(page);

  /*
   * Completed on the bare dot rather than on a typed prefix. Both assertions
   * below depend on that: with `gu` typed, CodeMirror's own prefix filter would
   * remove every dunder on its own, so "no __class__" would hold whether or not
   * this code suppressed anything -- an assertion that cannot fail. On the bare
   * dot the full attribute list is in play and the suppression is load-bearing.
   *
   * Typed as a keystroke, not a document replacement: the popup opens on input
   * events, and `insertText` of a whole buffer is not one.
   */
  await setEditorText(page, "zephyr_marker");
  await page.keyboard.type(".");

  const popup = page.locator(".cm-tooltip-autocomplete");
  await expect(popup).toBeVisible({ timeout: 15_000 });
  await expect(popup).toContainText("gust_speed");

  /*
   * Dunders stay suppressed until asked for. Without that, `gust_speed` sits
   * below a screenful of `__reduce_ex__` and the feature is worse than absent.
   */
  await expect(popup).not.toContainText("__class__");
});

test("a package can be installed by name from the packages panel", async ({ page }) => {
  await openHarness(page);
  await page.getByRole("button", { name: "Packages" }).click();

  const input = page.getByLabel("Install a package by name");
  await input.fill("micropip");
  // Scoped to the install row: catalog entries whose availability is micropip
  // carry their own "Install" button, so an unscoped lookup is ambiguous.
  await page.locator(".python-plot__install-row")
    .getByRole("button", { name: "Install", exact: true })
    .click();
  /*
   * Disabled, then enabled again. Waiting only for "enabled" would be the same
   * race the run counter exists to avoid: the input is enabled before the click
   * too, so that assertion passes without the install having happened.
   */
  await expect(input).toBeDisabled();
  await expect(input).toBeEnabled({ timeout: 90_000 });
  await expect(page.locator(".python-plot__package-error")).toHaveCount(0);

  /*
   * Verified by importing it, not by reading our own badge: the badge is drawn
   * from the same state the install path just wrote, so it would agree with an
   * install that never happened.
   */
  await setEditorText(page, ["import micropip", 'print("micropip", micropip.__name__)'].join("\n"));
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("micropip micropip");
});

test("scripts belong to the project they were written in", async ({ page }) => {
  await openHarness(page);
  await expect(page.locator(".python-plot__tabs"))
    .toHaveAttribute("data-script-scope", "none");

  await selectProject(page, "project_alpha");
  await setEditorText(page, "# alpha hook echo analysis\n");
  await expect(page.locator(".cm-content")).toContainText("alpha hook echo analysis");

  /*
   * The switch, which is the whole feature. Beta must not be looking at
   * alpha's editor.
   */
  await selectProject(page, "project_beta");
  await expect(page.locator(".cm-content")).not.toContainText("alpha hook echo analysis");
  await setEditorText(page, "# beta cold pool analysis\n");

  await selectProject(page, "project_alpha");
  await expect(page.locator(".cm-content")).toContainText("alpha hook echo analysis");
  await expect(page.locator(".cm-content")).not.toContainText("beta cold pool analysis");

  /*
   * The stores themselves, not just what is on screen.
   *
   * A workspace swap has two writes racing: the debounced save of the OUTGOING
   * buffers and the load of the incoming ones. Get the order wrong and alpha's
   * text is written under beta's key — the screen still looks right on the way
   * out, and the damage only surfaces the next time beta is opened. Reading
   * both stores is the only way to see it now rather than then.
   */
  const stores = await page.evaluate(([alphaKey, betaKey, unscopedKey]) => ({
    alpha: window.localStorage.getItem(alphaKey) ?? "",
    beta: window.localStorage.getItem(betaKey) ?? "",
    unscoped: window.localStorage.getItem(unscopedKey) ?? "",
  }), [storeFor("project_alpha"), storeFor("project_beta"), UNSCOPED_STORE]);

  expect(stores.alpha).toContain("alpha hook echo analysis");
  expect(stores.alpha).not.toContain("beta cold pool analysis");
  expect(stores.beta).toContain("beta cold pool analysis");
  expect(stores.beta).not.toContain("alpha hook echo analysis");
  // Neither project's work leaks into the no-project workspace.
  expect(stores.unscoped).not.toContain("alpha hook echo analysis");
  expect(stores.unscoped).not.toContain("beta cold pool analysis");
});

test("typing during a delayed Python save preserves exact disk bytes and dirty recovery", async ({ page }) => {
  await openHarness(page);
  const target = `${await pluginsDir(page)}/delayed-save.py`;
  const written = "print('written snapshot')\n";
  const newer = "print('newer unsaved edit')\n";
  await setEditorText(page, written);
  await chooseSavePath(page, target);
  await page.evaluate(path => window.__nimbusPythonFixture.holdScriptIo("write_config", path), target);
  await page.getByRole("button", { name: "Save…", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__nimbusPythonFixture.scriptIoWaiting()), { timeout: 10_000 }).toBe(true);
  await setEditorText(page, newer);
  await page.evaluate(() => window.__nimbusPythonFixture.releaseScriptIo());
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved delayed-save.py");
  expect(await diskContents(page, target)).toBe(written);
  await expect(page.locator(".python-plot__buffer-tab--active .python-plot__buffer-dirty")).toHaveCount(1);
  await expect(page.locator(".cm-content")).toContainText("newer unsaved edit");
  await expect.poll(() => page.evaluate(key => window.localStorage.getItem(key), UNSCOPED_STORE))
    .toContain("newer unsaved edit");
});

test("a delayed script open cannot insert an outgoing project's file into another workspace", async ({ page }) => {
  await openHarness(page);
  await selectProject(page, "project_alpha");
  const target = `${await pluginsDir(page)}/outgoing-only.py`;
  await seedFile(page, target, "print('belongs only to alpha')\n");
  await page.evaluate(path => {
    window.__nimbusPythonFixture.setOpenDialogPath(path);
    window.__nimbusPythonFixture.holdScriptIo("read_config", path);
  }, target);
  await page.getByRole("button", { name: "Open…", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__nimbusPythonFixture.scriptIoWaiting()), { timeout: 10_000 }).toBe(true);
  await selectProject(page, "project_beta");
  await setEditorText(page, "# beta remains its own scratch\n");
  await page.evaluate(() => window.__nimbusPythonFixture.releaseScriptIo());
  await expect.poll(() => page.evaluate(() => window.__nimbusPythonFixture.scriptIoWaiting())).toBe(false);
  await expect(page.locator(".python-plot__buffer-tab", { hasText: "outgoing-only.py" })).toHaveCount(0);
  await expect(page.locator(".cm-content")).toContainText("beta remains its own scratch");
});

test("completed Python results survive leaving and returning to their project", async ({ page }) => {
  await openHarness(page);
  await selectProject(page, "project_alpha");
  await setEditorText(page, "print('alpha completed result')\nresult = 17.25\n");
  await runWholeScript(page);
  await selectProject(page, "project_beta");
  await setEditorText(page, "print('beta completed result')\nresult = 91.5\n");
  await runWholeScript(page);
  await selectProject(page, "project_alpha");
  await expect(page.locator(TAB)).toHaveAttribute("data-run-count", "1");
  await showConsole(page);
  await expect(stdout(page)).toContainText("alpha completed result");
  await expect(stdout(page)).not.toContainText("beta completed result");
  await selectProject(page, "project_beta");
  await expect(page.locator(TAB)).toHaveAttribute("data-run-count", "1");
  await showConsole(page);
  await expect(stdout(page)).toContainText("beta completed result");
});

test("a project's first workspace opens the files you had open, re-read from disk", async ({ page }) => {
  // Load once to clear storage, then plant a workspace and reload keeping it.
  await page.goto(HARNESS);
  await page.evaluate(([key]) => {
    window.localStorage.setItem(key, JSON.stringify({
      schema: "nimbus.studio.python-workspace",
      schemaVersion: 1,
      activeId: "buf_9",
      buffers: [
        { id: "buf_9", name: "Scratch", path: null, code: "# unsaved scratch marker\n" },
        { id: "buf_3", name: "analysis.py", path: "/w/analysis.py" },
      ],
    }));
  }, [UNSCOPED_STORE]);
  /*
   * Put the file the workspace refers to on the virtual disk. The reader
   * reports `not_found` for anything nothing has written, exactly as the Rust
   * command does, so a spec that wants a file to exist has to say so.
   */
  await seedFile(page, "/w/analysis.py", ON_DISK_ANALYSIS);
  await page.goto(`${HARNESS}?keep=1`);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();

  // Baseline: with no project open, both buffers are there.
  await expect(page.locator(".python-plot__buffer-name")).toHaveCount(2);
  await expect(page.locator(".cm-content")).toContainText("unsaved scratch marker");

  await selectProject(page, "project_alpha");

  /*
   * The file follows; the unsaved scratch does not. Copying scratch text would
   * fork it into two independently editable copies of one unsaved script with
   * no way to tell them apart — worse than leaving it in the store it was
   * written in, where it is still reachable.
   */
  await expect(page.locator(".python-plot__buffer-name")).toHaveCount(1);
  await expect(page.locator(".python-plot__buffer-name")).toHaveText(/analysis\.py/);
  await expect(page.locator(".cm-content")).not.toContainText("unsaved scratch marker");

  /*
   * And the text is the file's, fetched on mount. The contents exist only on
   * the virtual disk and never went into the planted workspace, so this
   * distinguishes "re-read from disk" from "text carried across in storage" —
   * and it renders without a tab click, which is the state every seeded
   * project opens in.
   */
  await expect(page.locator(".cm-content")).toContainText(ON_DISK_MARKER);

  /*
   * Seeding wrote paths, not text: a project store must not grow by the size of
   * the files open in it.
   *
   * Polled, unlike the store reads in the test above. Those follow a project
   * SWITCH, which flushes synchronously so the bytes are already there. Nothing
   * is switched away from here, so the only thing that writes this store is the
   * ordinary debounce — reading it straight after the last UI assertion is a
   * race the fast path loses.
   */
  let seededStore = "";
  await expect.poll(async () => {
    seededStore = await page.evaluate(
      ([key]) => window.localStorage.getItem(key) ?? "",
      [storeFor("project_alpha")],
    );
    return seededStore;
  }, { timeout: 10_000 }).not.toBe("");
  expect(seededStore).toContain("/w/analysis.py");
  expect(seededStore).not.toContain(ON_DISK_MARKER);
});

/*
 * The reason the interpreter moved off the UI thread.
 *
 * Two symptoms, one cause. A cell held the page's only thread, so `while True:
 * pass` froze the whole application; and Pyodide cancels by having some OTHER
 * thread write a byte the interpreter polls, so the Stop button's click handler
 * could not run precisely when the user needed it. Stop was enabled,
 * documented, and incapable of stopping anything.
 *
 * Both halves are asserted here, against a loop that never yields. Nothing
 * below can pass if the interpreter is sharing a thread with the UI.
 */
test("Stop interrupts a running cell, and the UI stays live while it spins", async ({ page }) => {
  await openHarness(page);

  /*
   * Cross-origin isolation is the precondition for the whole mechanism: without
   * it SharedArrayBuffer is unavailable, the worker gets a private copy of the
   * sentinel, and the UI disables Stop rather than offering a dead control.
   * Asserted first so a dropped COOP/COEP header fails here with its own name
   * instead of surfacing as a mystery timeout below.
   */
  expect(await page.evaluate(() => window.crossOriginIsolated)).toBe(true);

  // Boot on a trivial cell. Clicking Stop while the runtime is still loading
  // would test nothing: there is no Python running yet to interrupt.
  // The binding is what tells the two halves of Stop apart further down: an
  // interrupt leaves it standing, and the forceful half cannot.
  await setEditorText(page, 'kept = 41\nprint("booted")');
  await runWholeScript(page);

  await setEditorText(page, ['print("spinning")', "while True:", "    pass"].join("\n"));
  await showConsole(page);

  const before = await runsCompleted(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();

  // The loop never awaits, so this chunk can only have been painted by a thread
  // that is not the one executing it.
  await expect(stdout(page)).toContainText("spinning");

  /*
   * The freeze half. This click is ordinary React state, so it completes only
   * if the main thread is still turning while Python spins — which is exactly
   * what was impossible before. Asserted on the active-tab class rather than on
   * the click resolving: Playwright's click resolves against the DOM whether or
   * not the handler ever ran.
   */
  await showResult(page);
  await expect(page.getByRole("tab", { name: "Result", exact: true }))
    .toHaveClass(/python-plot__tab--active/);
  await showConsole(page);

  const stop = page.getByRole("button", { name: "Stop", exact: true });
  await expect(stop).toBeEnabled();
  await stop.click();

  // The cancel half: a loop with no exit condition ends anyway, and says why.
  await expectRunAfter(page, before);
  await showConsole(page);
  await expect(stderr(page)).toContainText("KeyboardInterrupt");

  /*
   * And the session survives it. An interrupt that left the interpreter unusable
   * would trade a frozen app for a dead one, and the sentinel is cleared after
   * every run precisely so a late click cannot kill the next cell.
   *
   * `kept` is the part that distinguishes this from the forceful path below:
   * the polite route costs nothing, and a Stop that quietly emptied the
   * namespace every time would be the wrong trade to make silently.
   */
  await setEditorText(page, 'print("still alive", kept)');
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("still alive 41");
});

/*
 * The other half of Stop, and the half that was missing.
 *
 * A cell only honours KeyboardInterrupt where Python bytecode is running. Real
 * scripts spend their time somewhere else — a numpy kernel, a matplotlib
 * render, a wheel download — and there the interrupt byte is set, polled by
 * nobody, and the cell runs on. Stop was that first move alone, so pressing it
 * did nothing; and Restart, which would have worked, was disabled for the
 * duration of the very run the user was trying to escape.
 *
 * The loop below stands in for all of that by catching the interrupt it is
 * sent, which is indistinguishable from never seeing it. Nothing here waits on
 * the interpreter's cooperation: the escalation is on a timer, and terminating
 * the worker does not need the thread to agree.
 */
const UNINTERRUPTIBLE = [
  'print("ignoring interrupts")',
  "while True:",
  "    try:",
  "        while True:",
  "            pass",
  "    except KeyboardInterrupt:",
  "        pass",
].join("\n");

test("Stop force-stops a cell that ignores its interrupt, and says what that cost", async ({ page }) => {
  await openHarness(page);

  await setEditorText(page, 'kept = 41\nprint("booted")');
  await runWholeScript(page);

  await setEditorText(page, UNINTERRUPTIBLE);
  await showConsole(page);

  const before = await runsCompleted(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();
  await expect(stdout(page)).toContainText("ignoring interrupts");

  /*
   * Asserted while the cell is still spinning, because this is exactly when it
   * used to be refused. Restart is the only control that always ends a run, so
   * disabling it during a run withheld it for the one case it exists to serve.
   */
  await expect(page.getByRole("button", { name: "Restart", exact: true })).toBeEnabled();

  await page.getByRole("button", { name: "Stop", exact: true }).click();

  /*
   * The button renames itself to the harsher thing the next press would do,
   * within the grace period it is counting down. Checked with a timeout inside
   * that window: after it elapses the escalation has fired on its own and the
   * label is back to Stop, which is the assertion below.
   */
  await expect(page.getByRole("button", { name: "Force stop", exact: true }))
    .toBeVisible({ timeout: 1_500 });

  // Nobody pressed anything else. The timer is the whole point: a user who
  // pressed Stop should not have to work out that it did not take.
  await expectRunAfter(page, before);
  await expect(page.locator(".python-plot__io-status")).toContainText("Force-stopped");

  /*
   * And it really was replaced, rather than reported as replaced. `kept` was
   * bound in the first run and cannot survive a new interpreter — this is the
   * assertion that separates "the worker was terminated" from "the UI said so".
   */
  await setEditorText(page, "print(kept)");
  await runWholeScript(page);
  await showConsole(page);
  await expect(stderr(page)).toContainText("NameError");

  // The tab is usable again on the fresh interpreter, not merely unblocked.
  await setEditorText(page, 'print("new session")');
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("new session");
});

/*
 * A run outlives the panel that started it.
 *
 * Studio routes with a switch (App.tsx PageView), so leaving the Plots page
 * unmounts the whole studio — while the interpreter, a module-level worker,
 * carries on. When the results were React state, the cell finished and handed
 * its output to a component that no longer existed; coming back mounted a fresh
 * panel with nothing in it and no sign a run had ever happened. Every symptom
 * of "clicking off killed my run" except the killing.
 *
 * The checkbox is the harness's stand-in for that navigation: it unmounts the
 * panel outright, which is the part of the router that matters here.
 */
test("a run started before navigating away is waiting when you come back", async ({ page }) => {
  await openHarness(page);

  // Boot first, so the wait below is the cell's own and not the interpreter's.
  await setEditorText(page, 'print("booted")');
  await runWholeScript(page);

  const before = await runsCompleted(page);
  await setEditorText(page, [
    'print("started")',
    "await tick(3.0)",
    'print("finished while away")',
    "display(41 + 1)",
  ].join("\n"));
  await showConsole(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();
  await expect(stdout(page)).toContainText("started");

  // Leave. The panel goes; the cell does not.
  const mounted = page.getByLabel("Panel mounted");
  await mounted.uncheck();
  await expect(page.locator("[data-harness-unmounted]")).toBeVisible();
  await expect(page.locator(TAB)).toHaveCount(0);

  // Come back to a panel that never saw the run start.
  await mounted.check();
  await expect(page.locator(TAB)).toBeVisible();

  /*
   * The run is either still going or already done, depending on how long the
   * round trip took — both are correct, and both are the point: the result
   * arrives. Waiting on the counter covers the two without racing them.
   */
  await expectRunAfter(page, before);
  await showConsole(page);
  await expect(stdout(page)).toContainText("finished while away");

  // And the output it produced is on screen, not merely counted.
  await page.getByRole("tab", { name: "Result", exact: true }).click();
  await expect(page.locator(".python-plot__empty")).toHaveCount(0);
});

/*
 * The same promise, for a run that finishes with nobody there at all.
 *
 * Separate from the test above because it fails for a different reason. There,
 * the run lands after the panel is back and the result is simply written to a
 * mounted component. Here it lands while nothing is mounted, so the store is
 * the only thing that receives it — and coming back must not be what discards
 * it. The effect that clears stored output when the dataset changes also runs
 * on mount, and for a while remounting was indistinguishable from opening a
 * different run: the figure was there, and looking at it deleted it.
 */
test("a run that finishes while the panel is gone is not discarded by coming back", async ({ page }) => {
  await openHarness(page);

  await setEditorText(page, 'print("booted")');
  await runWholeScript(page);
  const before = await runsCompleted(page);

  await setEditorText(page, ['print("ran while away")', "display(41 + 1)"].join("\n"));
  await showConsole(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();

  /*
   * Leave immediately, and wait for the run to finish with nothing mounted.
   *
   * Waited on the store rather than on a clock: a fixed sleep decides the
   * outcome by how loaded the machine is, which made an earlier version of this
   * pass and fail on the same code.
   */
  const mounted = page.getByLabel("Panel mounted");
  await mounted.uncheck();
  await expect(page.locator(TAB)).toHaveCount(0);
  await expect.poll(
    async () => (await runStore(page)).completedRuns,
    { message: "the run should finish while the panel is unmounted", timeout: 90_000 },
  ).toBe(before + 1);
  expect((await runStore(page)).running).toBe(false);

  await mounted.check();
  await expect(page.locator(TAB)).toBeVisible();

  // Already finished, so this must be satisfied by what was restored rather
  // than by anything that happens next.
  await expect(page.locator(TAB))
    .toHaveAttribute("data-run-count", String(before + 1));
  await showConsole(page);
  await expect(stdout(page)).toContainText("ran while away");
  await page.getByRole("tab", { name: "Result", exact: true }).click();
  await expect(page.locator(".python-plot__empty")).toHaveCount(0);
});

/*
 * The seam between the panes moves.
 *
 * It was frozen at 45% of the pane, which is the wrong answer for both of the
 * things people do here: a wide figure wants the output side, a long script
 * wants the editor. The width is stored, so the check is not only that the drag
 * moved something but that the pane is still that size after a reload.
 */
test("the editor/output split can be dragged, and stays where it is put", async ({ page }) => {
  await openHarness(page);

  const editor = page.locator(".python-plot__editor-pane");
  const handle = page.getByRole("separator", { name: "Resize the code editor" });
  await expect(handle).toBeVisible();

  /*
   * The grip is re-measured before each drag rather than reused. It moves with
   * the seam it draws, so a second press at the first drag's coordinates lands
   * on the output pane instead — a drag that does nothing, and an assertion
   * that passes or fails on where the handle happened to end up.
   */
  async function dragBy(deltaX: number): Promise<void> {
    const grip = (await handle.boundingBox())!;
    const y = grip.y + grip.height / 2;
    await page.mouse.move(grip.x + grip.width / 2, y);
    await page.mouse.down();
    await page.mouse.move(grip.x + grip.width / 2 + deltaX, y, { steps: 8 });
    await page.mouse.up();
  }

  const start = (await editor.boundingBox())!;
  await dragBy(-120);
  expect((await editor.boundingBox())!.width).toBeLessThan(start.width - 80);

  await dragBy(160);
  expect((await editor.boundingBox())!.width).toBeGreaterThan(start.width);

  /*
   * Neither pane can be dragged out of existence: the output track keeps a hard
   * minimum, so a drag far past the right edge stops rather than closing the
   * pane the user is dragging in order to see.
   */
  await dragBy(4000);
  expect((await page.locator(".python-plot__output-pane").boundingBox())!.width)
    .toBeGreaterThan(200);

  await dragBy(-4000);
  expect((await editor.boundingBox())!.width).toBeGreaterThan(200);

  /*
   * And it stays put. `keep=1` leaves the harness's stores alone, which is what
   * makes this a test of persistence rather than of the default.
   */
  await dragBy(-90);
  const settled = (await editor.boundingBox())!.width;
  await page.goto(`${HARNESS}?keep=1`);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();
  expect(Math.abs((await editor.boundingBox())!.width - settled)).toBeLessThan(4);
});

/*
 * Save, and what the gallery makes of it.
 *
 * Three claims that only mean something end to end, because each one is a
 * different subsystem agreeing with the others: the editor's dirty tracking,
 * the text write, and the directory listing that the gallery renders. The
 * fixture's disk enforces the same extension allowlist and the same
 * not-found as `src-tauri/src/commands/config.rs`, so nothing here passes on
 * a path the shipped app would refuse.
 */
test("saving a script puts it in the gallery, and opening it back reads the file", async ({ page }) => {
  await openHarness(page);
  const dir = await pluginsDir(page);
  const target = `${dir}/hook_echo.py`;

  await setEditorText(page, "# hook echo study\nradius_km = 12\n");
  /*
   * A scratch buffer has no path, so the control says `Save…` and routes
   * through the dialog. The ellipsis is the promise that it will ask, and it
   * disappears below once there is a file to write to — so the two names are
   * asserted separately rather than matched loosely.
   */
  await chooseSavePath(page, target);
  await page.getByRole("button", { name: "Save…", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved hook_echo.py");

  // The bytes, from the disk rather than from the banner. A status line is the
  // UI's claim; this is whether the claim is true.
  expect(await diskContents(page, target)).toContain("radius_km = 12");

  /*
   * The tab adopts the path, so the NEXT save writes straight back to it. The
   * dialog is armed to cancel: if the second save consulted it anyway the
   * write would not happen and the assertion below would fail — which is the
   * point, since every save being a Save As is the bug this behaviour fixed.
   */
  await expect(page.locator(".python-plot__buffer-name")).toHaveText(/hook_echo\.py/);
  await chooseSavePath(page, null);
  await setEditorText(page, "# hook echo study\nradius_km = 30\n");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved hook_echo.py");
  expect(await diskContents(page, target)).toContain("radius_km = 30");

  // And the gallery lists it, because the gallery is a listing of that
  // directory rather than a record the save path also has to remember to keep.
  await page.getByRole("button", { name: "Gallery" }).click();
  await expect(galleryEntry(page, "hook_echo\\.py")).toBeVisible();
});

test("a gallery entry opens as its own tab, with the text that is on disk", async ({ page }) => {
  await openHarness(page);
  const dir = await pluginsDir(page);
  await seedFile(page, `${dir}/vorticity_budget.py`, "# stretching vs tilting\nterm = 'stretching'\n");

  await page.getByRole("button", { name: "Gallery" }).click();
  await galleryEntry(page, "vorticity_budget\\.py").click();

  // Its own tab, alongside the scratch buffer rather than on top of it. Opening
  // a gallery entry used to overwrite whatever was in the editor.
  await expect(page.locator(".python-plot__buffer-name")).toHaveCount(2);
  await expect(page.locator(".cm-content")).toContainText("stretching vs tilting");
});

test("cancelling the save dialog writes nothing and leaves the buffer unsaved", async ({ page }) => {
  await openHarness(page);
  const dir = await pluginsDir(page);

  await setEditorText(page, "# not going to disk\n");
  await chooseSavePath(page, null);
  await page.getByRole("button", { name: "Save…", exact: true }).click();

  /*
   * Nothing written, nothing claimed. A "Saved" banner after a cancel would be
   * the worst possible outcome: the user believes their work is on disk and
   * stops treating the editor as the only copy of it.
   */
  expect(await diskContents(page, `${dir}/nimbus_snippet.py`)).toBeNull();
  await expect(page.locator(".python-plot__io-status")).toHaveCount(0);
  // Still dirty, so the tab still says so.
  await expect(page.locator(".python-plot__buffer-tab--active")).toContainText("•");
});

/*
 * Two Python tabs side by side.
 *
 * Studio offers `python` as a companion pane type, so this is reachable today.
 * Both panes shared one recovery store, and each one's debounced write rebuilt
 * the store from its own buffers — so whichever pane you were not looking at
 * lost its work, silently, on a timer. Nothing about that is observable from
 * inside one instance, which is why it survived until there were two.
 */
test("two Python panes keep separate scripts instead of overwriting each other", async ({ page }) => {
  await page.goto(`${HARNESS}?panes=2`);
  const panes = page.locator("[data-harness-pane]");
  await expect(panes).toHaveCount(2);

  const left = panes.filter({ has: page.locator('[data-script-pane="primary"]') });
  const right = panes.filter({ has: page.locator('[data-script-pane="companion-0"]') });
  await expect(left).toHaveCount(1);
  await expect(right).toHaveCount(1);

  await left.locator(".cm-content").click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText("# left pane script\n");
  await right.locator(".cm-content").click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText("# right pane script\n");

  // Neither pane shows the other's text. Before the split this was already
  // false in memory, since each pane held its own React state -- the loss
  // happened on the way to storage, and on the way back.
  await expect(left.locator(".cm-content")).toContainText("left pane script");
  await expect(right.locator(".cm-content")).toContainText("right pane script");

  /*
   * The part that actually broke. Both writes are debounced, so the surviving
   * store used to be whichever pane's timer fired last; polling until both
   * stores exist and then asserting on each is what distinguishes "two stores"
   * from "one store written twice".
   */
  await expect.poll(async () => page.evaluate(
    ([primaryKey, companionKey]) => ({
      primary: window.localStorage.getItem(primaryKey) ?? "",
      companion: window.localStorage.getItem(companionKey) ?? "",
    }),
    [UNSCOPED_STORE, `${UNSCOPED_STORE}|pane:companion-0`] as const,
  ), { timeout: 10_000 }).toEqual({
    primary: expect.stringContaining("left pane script"),
    companion: expect.stringContaining("right pane script"),
  });

  // And they survive a reload as two, which is the whole reason the store is
  // written at all.
  await page.goto(`${HARNESS}?panes=2&keep=1`);
  await expect(left.locator(".cm-content")).toContainText("left pane script");
  await expect(right.locator(".cm-content")).toContainText("right pane script");
});

test("simultaneous Python panes isolate namespace, streams, and Restart", async ({ page }) => {
  await page.goto(`${HARNESS}?panes=2`);
  const left = page.locator('[data-harness-pane="primary"]');
  const right = page.locator('[data-harness-pane="companion-0"]');
  await expect(left.locator(".python-plot__cell-bar")).toBeVisible();
  await expect(right.locator(".python-plot__cell-bar")).toBeVisible();

  const setPaneCode = async (pane: typeof left, source: string) => {
    await pane.locator(".cm-content").click();
    await page.keyboard.press("ControlOrMeta+a");
    await page.keyboard.insertText(source);
  };
  const runCount = async (pane: typeof left) => Number(
    (await pane.locator(TAB).getAttribute("data-run-count")) ?? "0",
  );

  await setPaneCode(left, [
    "import time",
    "left_marker = 11",
    "print('LEFT-BEGIN', flush=True)",
    "time.sleep(0.25)",
    "print('LEFT-END', left_marker, 'right_marker' in globals())",
  ].join("\n"));
  await setPaneCode(right, [
    "import time",
    "right_marker = 22",
    "print('RIGHT-BEGIN', flush=True)",
    "time.sleep(0.25)",
    "print('RIGHT-END', right_marker, 'left_marker' in globals())",
  ].join("\n"));

  const [leftBefore, rightBefore] = await Promise.all([runCount(left), runCount(right)]);
  await left.getByRole("button", { name: "Run script", exact: true }).click();
  await right.getByRole("button", { name: "Run script", exact: true }).click();
  await expect(left.locator(TAB)).toHaveAttribute("data-run-count", String(leftBefore + 1), {
    timeout: 90_000,
  });
  await expect(right.locator(TAB)).toHaveAttribute("data-run-count", String(rightBefore + 1), {
    timeout: 90_000,
  });

  await left.getByRole("tab", { name: "stdout / stderr" }).click();
  await right.getByRole("tab", { name: "stdout / stderr" }).click();
  const leftStdout = left.locator(".python-plot__console-stdout");
  const rightStdout = right.locator(".python-plot__console-stdout");
  await expect(leftStdout).toContainText("LEFT-END 11 False");
  await expect(leftStdout).not.toContainText("RIGHT-");
  await expect(rightStdout).toContainText("RIGHT-END 22 False");
  await expect(rightStdout).not.toContainText("LEFT-");

  // Restarting the idle left interpreter while the right pane is executing
  // must neither terminate the right worker nor erase its namespace.
  await setPaneCode(right, [
    "import time",
    "print('RIGHT-WAIT', flush=True)",
    "time.sleep(0.8)",
    "print('RIGHT-SURVIVED', right_marker)",
  ].join("\n"));
  const rightLongBefore = await runCount(right);
  await right.getByRole("button", { name: "Run script", exact: true }).click();
  await expect(right.locator(TAB)).toHaveAttribute("data-running", "1");
  page.once("dialog", (dialog) => dialog.accept());
  await left.getByRole("button", { name: "Restart", exact: true }).click();
  await expect(right.locator(TAB)).toHaveAttribute("data-run-count", String(rightLongBefore + 1), {
    timeout: 90_000,
  });
  await right.getByRole("tab", { name: "stdout / stderr" }).click();
  await expect(rightStdout).toContainText("RIGHT-SURVIVED 22");

  await setPaneCode(left, "print('LEFT-RESET', 'left_marker' in globals())");
  const leftResetBefore = await runCount(left);
  await left.getByRole("button", { name: "Run script", exact: true }).click();
  await expect(left.locator(TAB)).toHaveAttribute("data-run-count", String(leftResetBefore + 1), {
    timeout: 90_000,
  });
  await left.getByRole("tab", { name: "stdout / stderr" }).click();
  await expect(leftStdout).toContainText("LEFT-RESET False");
});

test("Open data, files dictionary, produced-file save, and Release work end to end", async ({ page }) => {
  await openHarness(page);
  const dir = await pluginsDir(page);
  const input = `${dir}/observations.npy`;
  await seedBinaryFile(page, input, "radar sample bytes");
  await page.evaluate((path) => window.__nimbusPythonFixture.setOpenDialogPath(path), input);

  await page.getByRole("button", { name: "Open data", exact: true }).click();
  await expect(page.getByTitle("Release observations.npy from the interpreter")).toBeVisible({
    timeout: 90_000,
  });

  await setEditorText(page, [
    "print(open(files['observations.npy'], 'rb').read().decode())",
    "with open(save_path('analysis.txt'), 'w') as report:",
    "    _ = report.write('python report')",
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("radar sample bytes");

  await expect(page.getByText("analysis.txt", { exact: true })).toBeVisible();
  await chooseSavePath(page, `${dir}/analysis-export.txt`);
  await page.getByTitle("Save analysis.txt").click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved analysis-export.txt");
  expect(await diskContents(page, `${dir}/analysis-export.txt`)).toBe("python report");

  await page.getByTitle("Release observations.npy from the interpreter").click();
  await expect(page.getByTitle("Release observations.npy from the interpreter")).toHaveCount(0);
});

/*
 * A file that is no longer there.
 *
 * The recovery store deliberately keeps a path instead of the text for any
 * buffer that is clean, so every restored file-backed tab depends on the file
 * still existing. Deleting it outside Studio is ordinary — a moved directory, a
 * different checkout — and until the fixture's reader started reporting
 * not_found the way the real command does, this branch could not be reached
 * from a test at all.
 */
test("a buffer whose file has gone says so, and is restored by retyping it", async ({ page }) => {
  await page.goto(HARNESS);
  await page.addInitScript(([key]) => {
    window.localStorage.setItem(key, JSON.stringify({
      schema: "nimbus.studio.python-workspace",
      schemaVersion: 1,
      activeId: "buf_1",
      buffers: [{ id: "buf_1", name: "vanished.py", path: "/w/vanished.py" }],
    }));
  }, [UNSCOPED_STORE]);
  // Deliberately NOT seeded onto the disk: this is the file that is gone.
  await page.goto(`${HARNESS}?keep=1`);
  await expect(page.locator(".python-plot__cell-bar")).toBeVisible();

  // Named, with the reason. An empty editor under a real filename would read
  // as an empty file rather than as one nobody could fetch.
  await expect(page.locator(".python-plot__io-status")).toContainText("Could not reopen vanished.py");
  await expect(page.locator(".python-plot__io-status")).toContainText("not found");

  /*
   * And Save will not write this buffer back over the path it names. The tab
   * knows it never loaded, so the control says it will ask — an ellipsis on a
   * buffer that HAS a path, which is the whole signal.
   */
  await expect(page.getByRole("button", { name: "Save…", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save", exact: true })).toHaveCount(0);

  /*
   * Typing is what releases the path. The buffer now holds text the user meant
   * for that file rather than emptiness nobody chose, so the control drops its
   * ellipsis and commits to writing there — which recreates the file where it
   * used to be, the outcome someone whose checkout moved actually wants.
   */
  await setEditorText(page, "# retyped from memory\n");
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeEnabled();

  /*
   * The dialog is armed to CANCEL. If Save consulted it, nothing would be
   * written and the assertions below would fail — so this is what distinguishes
   * "wrote to the path it had" from "asked, and happened to be given it back".
   */
  await chooseSavePath(page, null);
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved vanished.py");
  expect(await diskContents(page, "/w/vanished.py")).toContain("retyped from memory");
});

/*
 * Replacing the interpreter.
 *
 * Two things made this necessary and one made it possible. WebAssembly memory
 * only grows, so a session that has held a large array once carries the reading
 * forever; and an interpreter that reaches what the browser will grant aborts
 * rather than raising, which left the tab in a state no button could fix. The
 * fix is to terminate the thread — which only became an option when the
 * interpreter stopped sharing a thread with the UI.
 */
test("Restart discards the namespace and leaves a working interpreter", async ({ page }) => {
  await openHarness(page);
  page.on("dialog", (dialog) => void dialog.accept());

  await setEditorText(page, 'marker = "survivor"\nprint(marker)');
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("survivor");

  // The name is in the namespace panel, which is how the user sees session
  // state at all.
  await page.getByRole("button", { name: "Variables" }).click();
  await expect(page.locator(".python-plot__packages")).toContainText("marker");

  await page.getByRole("button", { name: "Restart", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Interpreter restarted");

  /*
   * Gone from the panel, and gone from Python. The panel alone would not
   * settle it — it is cleared by the restart handler either way, so a restart
   * that had not actually replaced anything would still look right.
   */
  await expect(page.locator(".python-plot__packages")).not.toContainText("marker");
  await setEditorText(page, "print(marker)");
  await runWholeScript(page);
  await showConsole(page);
  await expect(stderr(page)).toContainText("NameError");

  // And the replacement is a working interpreter, not a wedged one.
  await setEditorText(page, 'import numpy as np\nprint("alive", int(np.array([2, 3]).sum()))');
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("alive 5");
});

/*
 * The budget on `load_series`.
 *
 * The one call that multiplies a field by a frame count, and so the usual way
 * a session ends: overrunning the heap aborts the interpreter instead of
 * raising, and there is nothing to catch. Refusing early costs a subset;
 * not refusing costs the namespace.
 *
 * Provoked with a repeated frame list rather than a large run, because the
 * fixture's grid is small and the arithmetic under test is `bytes × count` —
 * which does not care whether the indices differ.
 */
test("load_series refuses a request that would end the session, and says what fits", async ({ page }) => {
  await openHarness(page);

  await setEditorText(page, [
    "try:",
    "    await load_series('w', frames=[0] * 20000)",
    "except MemoryError as err:",
    "    print('refused:', err)",
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);

  // MemoryError, not ValueError: the caller can tell "too big for this
  // interpreter" from "you passed nonsense".
  await expect(stdout(page)).toContainText("refused:");
  // Both sizes and the remedy. A refusal that does not say how much would fit
  // leaves the user bisecting.
  await expect(stdout(page)).toContainText("MiB limit");
  await expect(stdout(page)).toContainText("frames of this field fit");

  // And it does not false-positive: an ordinary request still works.
  await setEditorText(page, [
    "series = await load_series('w', frames=[0, 1])",
    "print('shape', series.shape)",
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("shape (2,");
});

test("load_series preserves requested frame order and every value without broadcasting changing grids", async ({ page }) => {
  await openHarness(page);
  await setEditorText(page, [
    "import numpy as np",
    "targets = [2, 0, 2, 1]",
    "expected = np.stack([await load_field('w', frame=frame) for frame in targets])",
    "series = await load_series('w', frames=targets)",
    "print('series exact', np.array_equal(series, expected, equal_nan=True), series.dtype)",
    "print('ordered times', frame_times_s[targets].tolist())",
    "original_loader = load_field",
    "shape_calls = 0",
    "async def changing_shape(*args, **kwargs):",
    "    global shape_calls",
    "    shape_calls += 1",
    "    return np.zeros((2 if shape_calls == 1 else 1, 2), dtype=np.float32)",
    "load_field = changing_shape",
    "try:",
    "    await load_series('w', frames=[0, 1])",
    "except ValueError as error:",
    "    print('shape rejected', error)",
    "finally:",
    "    load_field = original_loader",
  ].join("\n"));
  await runWholeScript(page);
  await showConsole(page);
  await expect(stdout(page)).toContainText("series exact True float32");
  await expect(stdout(page)).toContainText("ordered times [60.0, 0.0, 60.0, 30.0]");
  await expect(stdout(page)).toContainText("shape rejected load_series fields must have the same shape");
});

/*
 * Bringing your own script in.
 *
 * A script written elsewhere arrives with its imports already at the top, and
 * the interpreter used to discover them the only way it could: by failing on
 * one. `detectMissingModule` reads a single name out of a traceback, so four
 * new imports meant four failed runs and four clicks, each revealing exactly
 * one more thing to fetch. Pyodide can read the imports out of the source and
 * fetch them together, which is the difference between pasting an analysis in
 * and working up to it.
 *
 * Three packages, none of them loaded at boot, all vendored for this.
 */
test("a script's imports are all resolved by the first run, not one failure at a time", async ({ page }) => {
  await openHarness(page);

  await setEditorText(page, [
    "import micropip",
    "import idna",
    "import sortedcontainers",
    "print('imported', micropip.__name__, idna.__name__, sortedcontainers.__name__)",
  ].join("\n"));

  const before = await runsCompleted(page);
  await page.getByRole("button", { name: "Run script", exact: true }).click();
  await expectRunAfter(page, before);
  await showConsole(page);

  // One run. Three packages.
  await expect(stdout(page)).toContainText("imported micropip idna sortedcontainers");
  await expect(stderr(page)).toHaveCount(0);

  /*
   * And no banner. Its absence is the assertion: the banner is what the old
   * flow produced here, three times over, and a run that merely succeeded
   * could still have shown it on the way.
   */
  await expect(page.getByRole("button", { name: /^Load / })).toHaveCount(0);
  // Exactly one run happened — no silent retry loop behind the scenes.
  expect(await runsCompleted(page)).toBe(before + 1);
});

/*
 * Getting the numbers out.
 *
 * `display(arr)` drew a heatmap and stopped there, so the one output a script
 * exists to produce was the one output that could not leave — unless its author
 * had known in advance to write `np.save(save_path(...))` instead of displaying
 * it. The check that matters is not that a file appears but that the values in
 * it are the ones Python computed.
 */
test("a displayed array can be saved as .npy and as CSV", async ({ page }) => {
  await openHarness(page);
  const dir = await pluginsDir(page);

  await setEditorText(page, [
    "import numpy as np",
    "display(np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32), label='table')",
  ].join("\n"));
  await runWholeScript(page);
  await showResult(page);

  // CSV, because its contents can be read back here and compared exactly.
  await chooseSavePath(page, `${dir}/table.csv`);
  await page.getByRole("button", { name: "Save CSV", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved table.csv");
  /*
   * C order and float32 precision, both. A row-major reading of the array the
   * cell built is "1.5,2.5,3.5" first — a transposed writer would put
   * "1.5,4.5" there and every other assertion in this test would still pass.
   */
  expect(await diskContents(page, `${dir}/table.csv`)).toBe("1.5,2.5,3.5\n4.5,5.5,6.5\n");

  // And .npy, which is the format that keeps the shape.
  await chooseSavePath(page, `${dir}/table.npy`);
  await page.getByRole("button", { name: "Save .npy", exact: true }).click();
  await expect(page.locator(".python-plot__io-status")).toContainText("Saved table.npy");

  /*
   * Checked as bytes, because with .npy the format IS the claim. The magic
   * makes it an NPY file at all, and the header ending on a 64-byte boundary is
   * what a memory-mapped reader needs — numpy reads a mis-padded file without
   * complaint, so getting that wrong produces something that opens fine right
   * up until someone opens it the other way.
   *
   * The values round-trip through the app's own `parseNpy` in the node suite,
   * which is a stronger check than restating the writer's arithmetic here.
   */
  const npyBytes = await diskBytes(page, `${dir}/table.npy`);
  expect(npyBytes).not.toBeNull();
  expect(npyBytes!.slice(0, 6)).toEqual([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);
  expect(npyBytes![6]).toBe(1);
  const headerLength = npyBytes![8] | (npyBytes![9] << 8);
  expect((10 + headerLength) % 64).toBe(0);
  // Header, then exactly six float32 and nothing else.
  expect(npyBytes!.length).toBe(10 + headerLength + 6 * 4);
});

/*
 * An array too big or too many-dimensioned to draw is still the result the
 * script produced. It used to render a sentence explaining that it could not be
 * plotted, and offer nothing.
 */
test("an array that cannot be plotted can still be exported", async ({ page }) => {
  await openHarness(page);

  await setEditorText(page, [
    "import numpy as np",
    "display(np.zeros((4, 5, 6), dtype=np.float32), label='cube')",
  ].join("\n"));
  await runWholeScript(page);
  await showResult(page);

  await expect(page.locator(".python-plot__empty")).toContainText("4×5×6 array");
  await expect(page.getByRole("button", { name: "Save .npy", exact: true })).toBeEnabled();
  // CSV is not offered at all for rank 3: a table cannot carry the third
  // dimension, and writing one anyway would lose it silently.
  await expect(page.getByRole("button", { name: "Save CSV", exact: true })).toHaveCount(0);
});

test("large line and heatmap results use disclosed renderer LOD without losing export", async ({ page }) => {
  await openHarness(page);
  const editorSeparator = page.getByRole("separator", { name: "Resize the code editor" });
  await expect(editorSeparator).toHaveAttribute("aria-valuemin", "260");
  await expect(editorSeparator).toHaveAttribute("aria-valuemax", "1100");
  await expect(editorSeparator).toHaveAttribute("aria-valuenow", /\d+/);
  await setEditorText(page, [
    "import numpy as np",
    "line = np.zeros(100_001, dtype=np.float32)",
    "line[12_345] = -80",
    "line[67_890] = 120",
    "line[80_000] = np.nan",
    "display(line, label='large line')",
    "display(np.arange(1_100_000, dtype=np.float32).reshape(1100, 1000), label='large heatmap')",
  ].join("\n"));
  await runWholeScript(page);
  await showResult(page);

  await expect(page.locator(".python-plot__display-note")).toHaveCount(2);
  await expect(page.locator(".python-plot__display-note").first()).toContainText("endpoint/extrema/missing-data");
  await expect(page.locator(".python-plot__display-note").last()).toContainText("point-samples");
  await expect(page.locator(".python-plot__display-note").last()).toContainText("1,100×1,000");
  await expect(page.locator(".python-plot__display-note").last()).toContainText("exports remain full resolution");
  await expect(page.getByRole("button", { name: "Save .npy", exact: true })).toHaveCount(2);
  await expect(page.getByRole("region", { name: /^Interactive scientific plot/ })).toHaveCount(2);
});
