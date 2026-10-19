import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import type {
  HarnessPlotType,
  HarnessSnapshot,
} from "../harness/main";

const mountCycles = Number(process.env.NIMBUS_VTK_MOUNT_CYCLES ?? "100");
const soakMinutes = Number(process.env.NIMBUS_SOAK_MINUTES ?? "0");
const soakMs = soakMinutes > 0 ? soakMinutes * 60_000 : 10_000;

/*
 * Time the soak needs AFTER its loop: unmount, drain in-flight IPC, two settled
 * heap reads, fifteen assertions, and the metrics write. The loop runs for
 * exactly soakMs, so the test's budget has to be soakMs plus this -- a fixed
 * ceiling equal to soakMs is a guaranteed timeout no matter what it measures.
 */
const SOAK_TAIL_BUDGET_MS = 5 * 60_000;

/**
 * What the page retains once the deliberate caches are dropped and a
 * collection has run. Null when the browser will not give an answer worth
 * asserting on.
 *
 * Two corrections over reading `usedJSHeapSize` directly, and the check is
 * worthless without either:
 *
 *   - Collect first. The raw counter climbs between collections, so comparing
 *     two arbitrary samples measures where in the GC cycle each landed. WebKit
 *     exposes neither a collection hook nor the counter, so it gets null and
 *     the caller skips the check rather than reporting a sawtooth as a plateau.
 *
 *   - Drop the field cache. It is an LRU with a 512 MB cap, and a soak that
 *     cycles twenty steps across every plot type fills it on purpose. Leaving
 *     it in place means the check fails when the cache is doing its job, which
 *     is the opposite of what it is for -- and is exactly how this assertion
 *     read as a leak on a run that had none.
 */
async function settledHeapBytes(page: Page): Promise<number | null> {
  return page.evaluate(async () => {
    const collect = (globalThis as { gc?: () => void }).gc;
    const memory = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
    if (typeof collect !== "function" || !memory) return null;
    window.__nimbusVtkHarness?.clearFieldCache();
    // Twice, with a turn of the event loop between: the first pass drops the
    // unreachable graph, the second collects what that pass made unreachable
    // (detached DOM freed by finalizers, typed arrays behind released views).
    collect();
    await new Promise((resolve) => setTimeout(resolve, 50));
    collect();
    return memory.usedJSHeapSize;
  });
}

async function waitForCanvas(page: Page, present: boolean): Promise<void> {
  await page.waitForFunction(
    (expected) => Boolean(document.querySelector(".harness-scene canvas")) === expected,
    present,
  );
}

async function waitForWorkers(page: Page): Promise<void> {
  try {
    await page.waitForFunction(() => {
      const harness = window.__nimbusVtkHarness;
      if (!harness) return false;
      const state = harness.snapshot();
      const activeLayer = document.querySelector([
        '[data-contour-loading="true"]',
        '[data-streamline-loading="true"]',
        '[data-flow-contour-loading="true"]',
        '[data-flow-preparation-loading="true"]',
        '[data-flow-extraction-loading="true"]',
        '[data-vortex-contour-loading="true"]',
        '[data-vortex-preparation-loading="true"]',
        '[data-condensate-preparation-loading="true"]',
        '[data-vortex-extraction-loading="true"]',
      ].join(","));
      return state.ipc.inFlightJobs === 0
        && state.visualization.reservedBytes === 0
        && activeLayer == null;
    }, undefined, { timeout: 180_000 });
  } catch {
    const state = await snapshot(page);
    throw new Error(`VTK worker jobs did not settle: ${JSON.stringify({
      workers: state.workers,
      workerKinds: state.workerKinds,
      preparedRegistry: state.preparedRegistry,
      extractionPool: state.extractionPool,
      contourProgress: state.contourProgress,
      visualization: state.visualization,
      ipc: state.ipc,
    })}`);
  }
}

async function waitForNoWorkers(page: Page): Promise<void> {
  try {
    await page.waitForFunction(() =>
      window.__nimbusVtkHarness?.snapshot().workers === 0,
    undefined, { timeout: 45_000 });
  } catch {
    const state = await snapshot(page);
    throw new Error(`VTK workers did not retire: ${JSON.stringify({
      workers: state.workers,
      workerKinds: state.workerKinds,
      preparedRegistry: state.preparedRegistry,
      extractionPool: state.extractionPool,
      contourProgress: state.contourProgress,
      visualization: state.visualization,
    })}`);
  }
}

/** Wait for scientific pixels, not for deliberately warm worker processes.
 * Streamline and contour workers retain one bounded dataset for 15 seconds so
 * playback can reuse their module and lattice; requiring zero here turns each
 * frame assertion into an idle-retirement test and hides actual display races.
 * Trajectories span the timeline rather than reporting one selected frame. */
async function waitForDisplayedFrame(
  page: Page,
  plot: HarnessPlotType,
  step: number,
): Promise<void> {
  if (plot === "trajectories") {
    await waitForWorkers(page);
    return;
  }
  await page.waitForFunction(
    ([expectedPlot, expectedStep]) => {
      const scene = document.querySelector<HTMLElement>(".harness-scene");
      return scene?.dataset.framePlot === expectedPlot
        && scene.dataset.frameLoading === "false"
        && scene.dataset.renderedStep === String(expectedStep);
    },
    [plot, step] as const,
    { timeout: 90_000 },
  );
}

/** Canvases and WebGL contexts are released synchronously on unmount. The
 *  shared VTK preparation coordinator may briefly retain one idle worker for
 *  its bounded burst-reuse window; waitForWorkers proves that it ultimately
 *  retires. A field read already inside platform I/O cannot be force-cancelled
 *  (see the contract note in lib/core/loadScheduler.ts) -- it drains and its
 *  result is then discarded as superseded. Wait for that drain before
 *  asserting on the IPC counters, so teardown still catches a read *started*
 *  after unmount without flaking on one legitimately already dispatched. */
async function waitForIdleIpc(page: Page): Promise<void> {
  await page.waitForFunction(() =>
    window.__nimbusVtkHarness?.snapshot().ipc.inFlightJobs === 0,
  );
}

async function snapshot(page: Page): Promise<HarnessSnapshot> {
  return page.evaluate(() => {
    if (!window.__nimbusVtkHarness) throw new Error("VTK harness unavailable");
    return window.__nimbusVtkHarness.snapshot();
  });
}

async function choosePlotTool(page: Page, name: string): Promise<void> {
  const actions = page.getByRole("navigation", { name: "Dataset and workspace actions" });
  const menu = actions.locator(".plot-studio__tools-menu");
  if (await menu.getAttribute("open") === null) await menu.locator("summary").click();
  await menu.getByRole("button", { name, exact: true }).click();
}

async function setPlot(page: Page, plot: HarnessPlotType): Promise<void> {
  await page.evaluate((next) => window.__nimbusVtkHarness?.setPlot(next), plot);
  await page.waitForFunction((next) =>
    document.querySelector(".harness-shell")?.getAttribute("data-plot") === next,
  plot);
}

async function setCoarse(page: Page, coarse: boolean): Promise<void> {
  await page.evaluate((next) => window.__nimbusVtkHarness?.setCoarse(next), coarse);
  await page.waitForFunction((next) =>
    document.querySelector(".harness-shell")?.getAttribute("data-coarse") === String(next),
  coarse);
}

async function setStep(page: Page, step: number): Promise<void> {
  await page.evaluate((next) => window.__nimbusVtkHarness?.setStep(next), step);
  await page.waitForFunction((next) =>
    document.querySelector(".harness-shell")?.getAttribute("data-step") === String(next),
  step);
}

async function mount(page: Page, mounted: boolean): Promise<void> {
  await page.evaluate((next) => window.__nimbusVtkHarness?.setMounted(next), mounted);
  await page.waitForFunction((next) =>
    document.querySelector(".harness-shell")?.getAttribute("data-mounted") === String(next),
  mounted);
  await waitForCanvas(page, mounted);
}

async function writeMetrics(testInfo: TestInfo, payload: unknown): Promise<void> {
  const path = testInfo.outputPath("vtk-lifecycle-metrics.json");
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  await testInfo.attach("vtk-lifecycle-metrics", {
    path,
    contentType: "application/json",
  });
}

function monotonicGrowth(values: readonly number[]): boolean {
  if (values.length < 4) return false;
  let increases = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[i - 1]) increases += 1;
  }
  return increases === values.length - 1;
}

test("Plot Studio empty state is solver-independent and capability truthful", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&empty=1");

  await expect(page.getByRole("heading", { name: "Open data" })).toBeVisible();
  await expect(page.getByText(/Analysis works independently of simulator setup/)).toBeVisible();
  await expect(page.getByRole("button", { name: /Open Nimbus run/ })).toBeEnabled();
  await expect(page.getByText("NetCDF data", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open files…" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Open folder…" })).toBeEnabled();
  await expect(page.getByText("Open one or more atmospheric netCDF files. Compatible model metadata is recognized automatically.")).toBeVisible();
  await expect(page.getByText(/NPY, ZFP, and netCDF histories can publish checksummed LODs/)).toBeVisible();
  await expect(page.getByText(/Imported netCDF uses native spatial windows and ROI reduction/)).toBeVisible();
  await expect(page.getByText("Saved comparison", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open workspace…" })).toBeEnabled();
  await expect(page.getByRole("region", { name: "Projects" })).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("guided CM1 handoff revalidates durable evidence before loading both comparison sources", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("nimbus:cm1-guided-handoff:v2", JSON.stringify({
      schema: "nimbus.studio.cm1-guided-handoff",
      schemaVersion: 2,
      id: "cm1-guided-browser",
      updatedAt: "2026-07-20T00:00:00.000Z",
      cm1Source: {
        schema: "nimbus.studio.dataset-source",
        schemaVersion: 1,
        kind: "cm1_netcdf",
        sourceModel: "cm1",
        nativeFormat: "netcdf",
        locator: { kind: "local_path", path: "/__nimbus_browser_fixture__/cm1out.nc" },
        displayName: "CM1 guided browser source",
        readOnly: true,
      },
      evidence: {
        historyFileCount: 1,
        soundingSourcePath: "/__nimbus_browser_fixture__/input_sounding",
        projectedLevelCount: 72,
        mappingCounts: { ready: 9, needsDecision: 3, different: 2, unsupported: 4 },
        mappingBlockers: ["3 mapping decisions remain"],
        mappingBlockerTotal: 1,
        mappingBlockersTruncated: false,
        reviewedAt: "2026-07-20T00:00:00.000Z",
      },
      smoke: {
        soundingPath: "/__nimbus_browser_fixture__/projected-sounding.txt",
        configPath: "/__nimbus_browser_fixture__/smoke.yaml",
        yaml: "schema: nimbus.v1\n",
        configurationDigest: "fnv1a64:f2b2b37c105d34a9",
        durationS: 30,
        outputIntervalS: 10,
        catalogFingerprint: "fnv1a64:cm1fixture",
        executablePath: "/__nimbus_browser_fixture__/bin/nimbus_sim",
        initializationExecutablePath: "/__nimbus_browser_fixture__/bin/nimbus_sim",
        stagedAt: "2026-07-20T00:00:00.000Z",
        consumedAt: "2026-07-20T00:00:01.000Z",
      },
      run: {
        status: "complete",
        runId: "cm1-guided-run",
        outputDir: "/__nimbus_browser_fixture__/live",
        pendingReason: null,
      },
      comparisonOpenedAt: null,
    }));
  });
  await page.goto("/harness/index.html?mode=studio&empty=1");

  await page.getByRole("button", { name: "Verify and compare" }).click();
  await expect(page.locator(".plot-studio__dataset-identity h2")).toHaveText("CM1 guided browser source");
  await expect(page.getByText("CM1 and Nimbus outputs verified. Review the alignment report before opening synchronized differences.")).toBeVisible();
  await expect(page.getByText("/__nimbus_browser_fixture__/live", { exact: true })).toBeVisible();
});

test("Plot Studio keeps analysis controls reachable at compact width and reduced motion", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.setViewportSize({ width: 520, height: 900 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/harness/index.html?mode=studio&cm1=1");

  const studio = page.locator(".plot-studio");
  const actions = page.getByRole("navigation", { name: "Dataset and workspace actions" });
  await expect(actions).toBeVisible();
  await expect(actions.getByRole("button", { name: "Open files…" })).toBeVisible();
  await expect(actions.getByText("Compare…", { exact: true })).toBeVisible();
  await expect(actions.getByText("Tools", { exact: true })).toBeVisible();
  await actions.locator(".plot-studio__tools-menu summary").click();
  await expect(actions.getByRole("button", { name: "Projects", exact: true })).toBeVisible();
  await expect(actions.getByRole("button", { name: "External providers", exact: true })).toBeVisible();

  const bounds = await studio.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(bounds.scrollWidth).toBeLessThanOrEqual(bounds.clientWidth + 1);

  const tools = actions.locator(".plot-studio__tools-menu summary");
  await tools.focus();
  await page.keyboard.press("Tab");
  await page.keyboard.press("Shift+Tab");
  await expect(tools).toBeFocused();
  const focusAndMotion = await tools.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      outlineStyle: style.outlineStyle,
      outlineWidth: style.outlineWidth,
      transitionDuration: style.transitionDuration,
    };
  });
  expect(focusAndMotion.outlineStyle).toBe("solid");
  expect(Number.parseFloat(focusAndMotion.outlineWidth)).toBeGreaterThanOrEqual(2);
  await actions.locator(".plot-studio__tools-menu summary").click();
  const transitionSeconds = focusAndMotion.transitionDuration
    .split(",")
    .map((duration) => duration.trim().endsWith("ms")
      ? Number.parseFloat(duration) / 1000
      : Number.parseFloat(duration));
  expect(Math.max(...transitionSeconds)).toBeLessThanOrEqual(0.00001);

  const categories = page.getByRole("group", { name: "Plot categories" });
  await expect(categories.getByRole("button", { name: "Plan views", exact: true })).toHaveAttribute("aria-pressed", "false");
  await expect(categories.getByRole("button", { name: "Radar", exact: true })).toHaveAttribute("aria-pressed", "true");
  const scripting = categories.getByRole("button", { name: "Scripting", exact: true });
  await scripting.scrollIntoViewIfNeeded();
  await expect(scripting).toBeVisible();

  const frame = page.getByRole("slider", { name: "Selected frame" });
  await expect(frame).toHaveAttribute("aria-valuetext", /Frame 1 of \d+/);
  await expect(page.getByRole("button", { name: "Play frames" })).toBeVisible();

  const compactScreenshot = testInfo.outputPath("plot-studio-compact.png");
  await page.screenshot({ path: compactScreenshot, fullPage: false });
  await testInfo.attach("plot-studio-compact", { path: compactScreenshot, contentType: "image/png" });

  await page.setViewportSize({ width: 1280, height: 900 });
  await expect(actions).toBeVisible();
  const desktopBounds = await studio.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(desktopBounds.scrollWidth).toBeLessThanOrEqual(desktopBounds.clientWidth + 1);
  const desktopTitle = page.locator(".plot-studio__dataset-identity h2");
  const desktopTitleBounds = await desktopTitle.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(desktopTitleBounds.scrollWidth).toBeLessThanOrEqual(desktopTitleBounds.clientWidth + 1);
  const desktopScreenshot = testInfo.outputPath("plot-studio-desktop.png");
  await page.screenshot({ path: desktopScreenshot, fullPage: false });
  await testInfo.attach("plot-studio-desktop", { path: desktopScreenshot, contentType: "image/png" });
  expect(pageErrors).toEqual([]);
});

test("CM1 history opens on a qualified familiar view and keeps the full catalog reachable", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&cm1=1");

  await choosePlotTool(page, "Model-oriented views");
  await expect(page.getByText("CM1 familiar start", { exact: true })).toBeVisible();
  const familiar = page.locator(".plot-studio__familiar");
  await expect(familiar.getByRole("button", { name: "Reflectivity", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Radar", exact: true })).toHaveClass(/plot-category--active/);
  await expect(page.getByRole("button", { name: "Surface dBZ", exact: true })).toHaveClass(/plot-preset-pill--active/);
  const undoView = page.getByRole("button", { name: "Undo plot view selection" });
  const redoView = page.getByRole("button", { name: "Redo plot view selection" });
  await expect(undoView).toBeDisabled();
  await expect(redoView).toBeDisabled();

  await familiar.getByRole("button", { name: "Cold pool", exact: true }).click();
  await expect(familiar.getByRole("button", { name: "Cold pool", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Plan views", exact: true })).toHaveClass(/plot-category--active/);
  await expect(page.getByRole("button", { name: "Cold pool \(θ'\)", exact: true })).toHaveClass(/plot-preset-pill--active/);

  await familiar.getByRole("button", { name: "Skew-T", exact: true }).click();
  await expect(familiar.getByRole("button", { name: "Skew-T", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Soundings", exact: true })).toHaveClass(/plot-category--active/);
  await expect(page.locator(".plot-studio__type-bar").getByRole("button", { name: "Skew-T", exact: true })).toHaveClass(/plot-type--active/);

  await undoView.click();
  await expect(familiar.getByRole("button", { name: "Cold pool", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(redoView).toBeEnabled();
  await redoView.click();
  await expect(familiar.getByRole("button", { name: "Skew-T", exact: true })).toHaveAttribute("aria-pressed", "true");

  await page.keyboard.press("Control+z");
  await expect(familiar.getByRole("button", { name: "Cold pool", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("slider", { name: "Selected frame" }).focus();
  await page.keyboard.press("Control+Shift+z");
  await expect(familiar.getByRole("button", { name: "Cold pool", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.locator("body").click({ position: { x: 1, y: 1 } });
  await page.keyboard.press("Control+Shift+z");
  await expect(familiar.getByRole("button", { name: "Skew-T", exact: true })).toHaveAttribute("aria-pressed", "true");
  expect(pageErrors).toEqual([]);
});

test("source-neutral derived registry exposes assumptions and portable provenance", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&cm1=1");

  await choosePlotTool(page, "Field registry");
  const ledger = page.locator(".derived-ledger");
  await expect(ledger).toBeVisible();
  await expect(ledger.locator("summary")).toContainText("5 available now");
  await ledger.locator("summary").click();
  const table = ledger.getByRole("table", { name: "Source-neutral derived field availability" });
  await expect(table).toContainText("Horizontal wind speed");
  await expect(table).toContainText("sqrt(u^2 + v^2)");
  await expect(table).toContainText("Bolton (1980) Eq. 15/43");
  await expect(table.getByText("Studio-derived", { exact: true })).toHaveCount(5);

  const downloadPromise = page.waitForEvent("download");
  await ledger.getByRole("button", { name: "Export registry evidence" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("CM1-browser-supercell-derived-fields.json");
  const reportPath = await download.path();
  expect(reportPath).not.toBeNull();
  const reportText = await readFile(reportPath!, "utf8");
  expect(reportText).toContain('"schema": "nimbus.studio.derived-field-catalog-report"');
  expect(reportText).toContain('"schema": "nimbus.studio.derived-field-provenance"');
  expect(reportText).toContain('"cacheIdentity"');
  expect(reportText).toContain('"horizontal_wind_speed"');

  await page.goto("/harness/index.html?mode=studio");
  await choosePlotTool(page, "Field registry");
  const nimbusLedger = page.locator(".derived-ledger");
  await expect(nimbusLedger.locator("summary")).toContainText("1 available now");
  await nimbusLedger.locator("summary").click();
  await expect(nimbusLedger.getByText("Horizontal wind speed", { exact: true })).toBeVisible();
  await expect(nimbusLedger.getByText("Studio-derived", { exact: true })).toHaveCount(1);
  await expect(nimbusLedger.getByText("missing theta + p", { exact: true })).toBeVisible();
  expect(pageErrors).toEqual([]);
});

test("Studio restores recent data, evaluates persisted expressions, and exposes resource budgets", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio");
  await expect(page.getByText("Nimbus history", { exact: true })).toBeVisible();

  await choosePlotTool(page, "Recent data");
  const recent = page.getByRole("region", { name: "Recent datasets" });
  await expect(recent).toContainText("Nimbus run");
  await expect(recent).toContainText("/__nimbus_browser_fixture__/live");
  await recent.getByRole("button", { name: "Pin", exact: true }).click();
  await expect(recent).toContainText("Pinned");
  await recent.getByRole("button", { name: "Close recent datasets" }).click();

  await choosePlotTool(page, "Derived expressions");
  const expressions = page.getByRole("region", { name: "Expression fields" });
  await expressions.getByLabel("Field ID").fill("wind_3d");
  await expressions.getByLabel("Display label").fill("Three-dimensional wind speed");
  await expressions.getByLabel("Output unit").fill("m/s");
  await expressions.getByLabel("Dependencies").fill("u, v, w");
  await expressions.getByLabel("Formula").fill("sqrt(u^2 + v^2 + w^2)");
  await expect(expressions.getByText("Definition qualified", { exact: true })).toBeVisible();
  await expressions.getByRole("button", { name: "Add definition" }).click();
  await expect(expressions.getByText("wind_3d [m/s]", { exact: true })).toBeVisible();

  await choosePlotTool(page, "Data health");
  const health = page.getByRole("region", { name: "Dataset health report" });
  await health.getByRole("combobox").selectOption("wind_3d");
  await health.getByRole("button", { name: "Scan exact values" }).click();
  await expect(health.getByText("1/1 frames", { exact: true })).toBeVisible();
  await expect(health.getByLabel("Exact field health results")).toContainText("41,472 / 41,472");
  await health.getByRole("button", { name: "Close", exact: true }).click();

  await choosePlotTool(page, "Resource use");
  const resources = page.getByRole("region", { name: "Memory & requests" });
  await expect(resources).toContainText("256 MiB");
  await expect(resources).toContainText("384 MiB");
  await expect(resources).toContainText("NPY and classic-netCDF window/LOD reads stop cooperatively");
  await resources.getByRole("button", { name: "Release parsed cache" }).click();
  await expect(page.getByText(/Released \d+ parsed field/)).toBeVisible();

  await choosePlotTool(page, "Projects");
  const projects = page.getByRole("region", { name: "Projects" });
  await projects.getByRole("button", { name: "New from current view" }).click();
  await expect(projects.getByText("Current", { exact: true })).toBeVisible();
  await expect(projects).toContainText("/__nimbus_browser_fixture__/live");
  const projectDownloadPromise = page.waitForEvent("download");
  await projects.getByRole("button", { name: "Publish recipe", exact: true }).click();
  const projectDownload = await projectDownloadPromise;
  const projectPath = await projectDownload.path();
  expect(projectPath).not.toBeNull();
  const projectText = await readFile(projectPath!, "utf8");
  expect(projectText).toContain('"schema": "nimbus.studio.project-publication"');
  expect(projectText).toContain('"fieldData": "references_only"');
  await projects.locator("input[type='file']").setInputFiles(projectPath!);
  await expect(page.getByText(/Imported .* as a new project/)).toBeVisible();
  await expect(projects.locator(".project-navigator__row")).toHaveCount(2);

  await choosePlotTool(page, "External providers");
  const providers = page.getByRole("region", { name: "Derived-field providers" });
  await expect(providers).toContainText("imported manifests start disabled");
  await expect(providers).toContainText("Native plugins remain unavailable");

  await page.reload();
  await choosePlotTool(page, "Projects");
  const restoredProjects = page.getByRole("region", { name: "Projects" });
  await expect(restoredProjects).toContainText("/__nimbus_browser_fixture__/live");
  await restoredProjects.getByRole("button", { name: "Open", exact: true }).first().click();
  await choosePlotTool(page, "Projects");
  await expect(page.getByRole("region", { name: "Projects" }).getByText("Current", { exact: true })).toBeVisible();
  const restoredExpressions = page.getByRole("region", { name: "Expression fields" });
  if (!await restoredExpressions.isVisible()) await choosePlotTool(page, "Derived expressions");
  await expect(restoredExpressions.getByText("wind_3d [m/s]", { exact: true })).toBeVisible();
  await page.goto("/harness/index.html?mode=studio&empty=1");
  const emptyRecent = page.getByRole("region", { name: "Recent datasets" });
  await expect(emptyRecent).toContainText("Pinned");
  await emptyRecent.getByRole("button", { name: "Open", exact: true }).click();
  await expect(page.getByText("Nimbus history", { exact: true })).toBeVisible();
  expect(pageErrors).toEqual([]);
});

test("CM1 data health separates metadata evidence from exact value scans", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&cm1=1");
  await choosePlotTool(page, "Data health");
  const health = page.getByRole("region", { name: "Dataset health report" });
  await expect(health).toBeVisible();
  const structure = health.getByLabel("Dataset structure health");
  await expect(structure).toContainText("Physical");
  await expect(structure).toContainText("3/3 exact coordinate sets");
  await expect(structure).toContainText("Qualified");
  await expect(structure).toContainText("Every published field has a declared unit");
  await expect(health.getByRole("combobox")).toHaveValue("reflectivity_dbz");
  await health.getByLabel("All frames").check();
  await health.getByRole("button", { name: "Scan exact values" }).click();
  await expect(health.getByText("3/3 frames", { exact: true })).toBeVisible();
  const results = health.getByLabel("Exact field health results");
  await expect(results).toContainText("124,416 / 124,416");
  await expect(results).toContainText("Complete finite coverage");
  await expect(results).toContainText("3 / 3");
  const downloadPromise = page.waitForEvent("download");
  await results.getByRole("button", { name: "Export health report" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("CM1-browser-supercell-reflectivity_dbz-health.json");
  const reportPath = await download.path();
  expect(reportPath).not.toBeNull();
  const reportText = await readFile(reportPath!, "utf8");
  expect(reportText).toContain('"schema": "nimbus.studio.data-health"');
  expect(reportText).toContain('"requestedScope": "all_available_frames"');
  expect(reportText).toContain('"elementCount": 124416');
  await health.getByRole("button", { name: "Close" }).click();
  await expect(health).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("CM1 investigation workspace links transect probe time notes and exact restore", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&cm1=1");

  await choosePlotTool(page, "Investigation workspace");
  const workspace = page.getByRole("region", { name: "Linked transect probe and annotation workspace" });
  await expect(workspace).toBeVisible();
  await expect(workspace.getByRole("heading", { name: "Linked investigation workspace" })).toBeVisible();
  await expect(workspace).toContainText("Bilinear x/y interpolation · native z levels");
  await page.waitForFunction(() => {
    const plan = document.querySelector<HTMLElement & { data?: unknown[] }>(".investigation__plan-canvas");
    const transect = document.querySelector<HTMLElement & { data?: unknown[] }>(".investigation__transect-canvas");
    const profile = document.querySelector<HTMLElement & { data?: unknown[] }>(".investigation__profile-canvas");
    return plan?.data?.length === 3 && transect?.data?.length === 2 && profile?.data?.length === 1;
  });

  await workspace.getByRole("button", { name: "Place B", exact: true }).click();
  await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { emit?: (event: string, payload: Record<string, unknown>) => void }>(".investigation__plan-canvas");
    graph?.emit?.("plotly_click", { points: [{ x: 4.0, y: 4.5 }] });
  });
  await expect(workspace).toContainText("B x 4.00, y 4.50 km");

  await workspace.getByRole("button", { name: "Place probe", exact: true }).click();
  await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { emit?: (event: string, payload: Record<string, unknown>) => void }>(".investigation__plan-canvas");
    graph?.emit?.("plotly_click", { points: [{ x: 3.3, y: 3.1 }] });
  });
  await expect(workspace).toContainText(/Probe x \d+\.\d{2}, y \d+\.\d{2} km/);

  await workspace.getByLabel("Annotation text").fill("Rear-flank notch becomes coherent");
  await workspace.getByRole("button", { name: "Add at current probe" }).click();
  await expect(workspace.getByLabel("Workspace annotations").getByText("Rear-flank notch becomes coherent", { exact: true })).toBeVisible();
  await expect(workspace.getByText("Saved locally", { exact: true })).toBeVisible();

  await workspace.getByRole("button", { name: "Build probe history" }).click();
  await expect(workspace.locator(".investigation__history-canvas")).toBeVisible();
  await page.waitForFunction(() => {
    const history = document.querySelector<HTMLElement & { data?: Array<{ x?: number[]; y?: number[] }> }>(".investigation__history-canvas");
    return history?.data?.length === 1 && history.data[0]?.x?.length === 3 && history.data[0]?.y?.length === 3;
  });

  await workspace.getByLabel("Linked investigation time").fill("1");
  await expect(workspace).toContainText("Time 30 s");
  await expect(workspace.locator(".investigation__history-canvas")).toBeVisible();
  await expect(workspace.getByText("Saved locally", { exact: true })).toBeVisible();

  await workspace.getByRole("button", { name: "Publication", exact: true }).click();
  const publication = workspace.getByRole("region", { name: "Investigation figure publication" });
  await expect(publication).toContainText("Figure and provenance, inseparable");
  await publication.getByLabel("Publication figure title").fill("CM1 rear-flank investigation");
  const packageDownload = page.waitForEvent("download");
  await publication.getByRole("button", { name: "Export figure package" }).click();
  const figurePackage = await packageDownload;
  expect(figurePackage.suggestedFilename()).toBe("CM1-rear-flank-investigation.nimbus-figure.zip");
  const figurePackagePath = await figurePackage.path();
  expect(figurePackagePath).not.toBeNull();
  const packageBytes = await readFile(figurePackagePath!);
  expect(packageBytes.subarray(0, 4).toString("hex")).toBe("504b0304");
  expect(packageBytes.toString("utf8")).toContain("nimbus.studio.investigation-publication");
  expect(packageBytes.toString("utf8")).toContain('"headlessBatchRendering": false');
  await expect(publication).toContainText("4-panel figure packaged with verified provenance");

  await workspace.getByRole("button", { name: "Place B", exact: true }).click();
  await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { emit?: (event: string, payload: Record<string, unknown>) => void }>(".investigation__plan-canvas");
    graph?.emit?.("plotly_click", { points: [{ x: 2.0, y: 2.0 }] });
  });
  await expect(workspace).toContainText("B x 2.00, y 2.00 km");
  await workspace.locator("input[type='file'][accept*='.zip']").setInputFiles(figurePackagePath!);
  await expect(publication).toContainText("Verified package restored against the exact open dataset");
  await expect(workspace).toContainText("B x 4.00, y 4.50 km");

  await page.reload();
  await choosePlotTool(page, "Investigation workspace");
  const restored = page.getByRole("region", { name: "Linked transect probe and annotation workspace" });
  await expect(restored.getByLabel("Workspace annotations").getByText("Rear-flank notch becomes coherent", { exact: true })).toBeVisible();
  await expect(restored).toContainText("B x 4.00, y 4.50 km");
  await expect(restored).toContainText("Time 30 s");
  expect(pageErrors).toEqual([]);
});

test("CM1 versus Nimbus comparison produces qualified synchronized differences", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&cm1=1&compare=1");

  const report = page.getByRole("region", { name: "Dataset alignment report" });
  await expect(report).toBeVisible();
  await expect(report).toContainText("CM1 browser supercell");
  await expect(report).toContainText("Nimbus browser control");
  await expect(report.getByText("Matched subset", { exact: true })).toBeVisible();
  await expect(report.getByText("3 matched", { exact: true })).toBeVisible();
  await expect(report.getByText("48 × 36 × 24", { exact: true })).toBeVisible();
  await expect(report).toContainText("difference-ready");
  await expect(report).toContainText("synchronized A, B, and A−B plan slices");
  await expect(page.getByText("A−B", { exact: true })).toHaveCount(0);

  await report.getByText("Alignment evidence", { exact: true }).click();
  await expect(report).toContainText("Native grid-staggering labels differ");
  await report.getByRole("button", { name: "Open A/B/difference slices" }).click();
  const comparison = page.getByRole("region", { name: "Synchronized A and B slices" });
  await expect(comparison).toBeVisible();
  await expect(report.getByRole("button", { name: "A/B/difference slices open" })).toHaveAttribute("aria-pressed", "true");
  await expect(comparison.getByRole("combobox").first()).toHaveValue("reflectivity_dbz");
  await expect(comparison).toContainText("A t = 0.000 s");
  await expect(comparison).toContainText("B t = 0.000 s");
  await expect(comparison.getByText("Mean A−B", { exact: true })).toBeVisible();
  await expect(comparison.getByText("RMSE", { exact: true })).toBeVisible();
  await expect(comparison.getByText("1,728", { exact: true })).toBeVisible();
  await comparison.getByRole("button", { name: "Full grid", exact: true }).click();
  await expect(comparison.getByText("Full valid grid", { exact: true })).toBeVisible();
  await expect(comparison.getByText("41,472", { exact: true })).toBeVisible();
  await comparison.getByRole("button", { name: "Zoomed region", exact: true }).click();
  await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { emit?: (event: string, payload: Record<string, number>) => void }>(".comparison-slice__canvas");
    graph?.emit?.("plotly_relayout", {
      "xaxis.range[0]": 0,
      "xaxis.range[1]": 0.5,
      "yaxis.range[0]": 0,
      "yaxis.range[1]": 0.35,
    });
  });
  await expect(comparison.getByLabel("A minus B statistics").getByText("Zoomed region", { exact: true })).toBeVisible();
  await expect(comparison.getByText("9", { exact: true })).toBeVisible();
  await comparison.getByRole("button", { name: "Full grid", exact: true }).click();

  await comparison.getByRole("button", { name: "Build series", exact: true }).click();
  const metricSeries = page.getByRole("region", { name: "Matched-time comparison metrics" });
  await expect(metricSeries).toBeVisible();
  await expect(metricSeries.getByText("3/3 frames", { exact: true })).toBeVisible();
  await expect(metricSeries).toContainText("Time-mean RMSE");
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & { data?: Array<{ x?: number[] }> }>(".comparison-series__canvas");
    return graph?.data?.length === 4 && graph.data.every((trace) => trace.x?.length === 3);
  });

  await comparison.getByRole("button", { name: "Open vectors", exact: true }).click();
  const vectorComparison = page.getByRole("region", { name: "Synchronized vector comparison" });
  await expect(vectorComparison).toBeVisible();
  await expect(vectorComparison).toContainText("Velocity: v → · u ↑");
  await expect(vectorComparison).toContainText("shared A/B arrow normalization");
  await expect(vectorComparison).toContainText("Vector RMSE");
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & {
      data?: Array<{ coloraxis?: string; type?: string }>;
      _fullLayout?: { xaxis2?: { matches?: string }; xaxis3?: { matches?: string } };
    }>(".comparison-vector__canvas");
    // Each panel has a heatmap, visible arrow shafts, and transparent marker
    // targets carrying exact vector-component hover values.
    return graph?.data?.length === 9
      && graph.data.filter((trace) => trace.type === "heatmap").length === 3
      && graph.data[0]?.coloraxis === "coloraxis"
      && graph.data[3]?.coloraxis === "coloraxis"
      && graph.data[6]?.coloraxis === "coloraxis2"
      && graph._fullLayout?.xaxis2?.matches === "x"
      && graph._fullLayout?.xaxis3?.matches === "x";
  });

  await vectorComparison.getByRole("button", { name: "Volume region", exact: true }).click();
  await expect(vectorComparison.locator(".comparison-vector__metrics").getByText("41,472", { exact: true })).toBeVisible();
  await vectorComparison.getByLabel("z upper index").fill("1");
  await expect(vectorComparison.locator(".comparison-vector__metrics").getByText("3,456", { exact: true })).toBeVisible();
  await vectorComparison.getByRole("button", { name: "Build vector series", exact: true }).click();
  const vectorSeries = page.getByRole("region", { name: "Matched-time vector metrics" });
  await expect(vectorSeries).toContainText("3/3 frames");
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & { data?: unknown[] }>(".comparison-vector-series__canvas");
    return graph?.data?.length === 5;
  });

  await comparison.getByRole("button", { name: "Open 3-D", exact: true }).click();
  const volumeComparison = page.getByRole("region", { name: "Synchronized three-dimensional comparison" });
  await expect(volumeComparison).toBeVisible();
  await expect(volumeComparison).toContainText(
    "41,472 volume-mean blocks from 41,472 selected cells",
  );
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & { data?: Array<{ type?: string }> }>(".comparison-volume__canvas");
    return graph?.data?.length === 3 && graph.data.every((trace) => trace.type === "volume");
  });
  await volumeComparison.getByLabel("3D z upper index").fill("1");
  await expect(volumeComparison).toContainText("3,456 selected cells");
  await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { emit?: (event: string, payload: Record<string, unknown>) => void }>(".comparison-volume__canvas");
    graph?.emit?.("plotly_relayout", { "scene.camera": { eye: { x: 1.1, y: 1.2, z: 0.9 } } });
  });
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & { _fullLayout?: { scene2?: { camera?: { eye?: { x?: number } } }; scene3?: { camera?: { eye?: { x?: number } } } } }>(".comparison-volume__canvas");
    return graph?._fullLayout?.scene2?.camera?.eye?.x === 1.1 && graph._fullLayout?.scene3?.camera?.eye?.x === 1.1;
  });

  await comparison.getByRole("button", { name: "Open composite", exact: true }).click();
  const compositeComparison = page.getByRole("region", { name: "Synchronized storm composite comparison" });
  await expect(compositeComparison).toBeVisible();
  await expect(compositeComparison).toContainText("Reflectivity shading");
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & { data?: Array<{ type?: string }> }>(".comparison-composite__canvas");
    return graph?.data?.length === 9
      && graph.data.filter((trace) => trace.type === "heatmap").length === 3
      && graph.data.filter((trace) => trace.type === "contour").length === 6;
  });

  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & {
      data?: Array<{ coloraxis?: string; z?: number[][] }>;
      _fullLayout?: {
        xaxis2?: { matches?: string };
        yaxis2?: { matches?: string };
        xaxis3?: { matches?: string };
        yaxis3?: { matches?: string };
      };
    }>(".comparison-slice__canvas");
    return graph?.data?.length === 3
      && graph.data[0]?.coloraxis === "coloraxis"
      && graph.data[1]?.coloraxis === "coloraxis"
      && graph.data[2]?.coloraxis === "coloraxis2"
      && graph.data[2]?.z?.every((row) => row.every((value) => value === 0))
      && graph._fullLayout?.xaxis2?.matches === "x"
      && graph._fullLayout?.yaxis2?.matches === "y"
      && graph._fullLayout?.xaxis3?.matches === "x"
      && graph._fullLayout?.yaxis3?.matches === "y";
  });
  const frameSlider = page.locator(".plot-studio__frame-bar input[type='range']");
  await frameSlider.fill("1");
  await expect(comparison).toContainText("A t = 30.000 s");
  await expect(comparison).toContainText("B t = 30.000 s");

  await comparison.getByRole("combobox").first().selectOption("w");
  await comparison.locator(".pill-group").filter({ hasText: "x" }).getByRole("button", { name: "x", exact: true }).click();
  await comparison.locator("input[type='range']").fill("2");
  await expect(comparison.getByText("view recipe retained", { exact: true })).toBeVisible();
  await page.reload();
  const reloadedReport = page.getByRole("region", { name: "Dataset alignment report" });
  await reloadedReport.getByRole("button", { name: "Open A/B/difference slices" }).click();
  await expect(comparison.getByRole("combobox").first()).toHaveValue("w");
  await expect(comparison.getByRole("button", { name: "Hide series", exact: true })).toBeVisible();
  await expect(comparison.getByRole("button", { name: "Hide vectors", exact: true })).toBeVisible();
  await expect(comparison.getByRole("button", { name: "Hide 3-D", exact: true })).toBeVisible();
  await expect(comparison.getByRole("button", { name: "Hide composite", exact: true })).toBeVisible();
  await expect(comparison.locator("input[type='range']")).toHaveValue("2");
  await expect(comparison.getByText("Full valid grid", { exact: true })).toBeVisible();

  const downloadPromise = page.waitForEvent("download");
  await reloadedReport.getByRole("button", { name: "Export workspace", exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("nimbus-comparison-workspace.json");
  const workspacePath = await download.path();
  expect(workspacePath).not.toBeNull();
  const workspaceText = await readFile(workspacePath!, "utf8");
  expect(workspaceText).toContain("nimbus.studio.comparison-publication-workspace");
  expect(workspaceText).toContain("nimbus.studio.dataset-alignment");

  await page.goto("/harness/index.html?mode=studio&empty=1");
  await expect(page.getByRole("region", { name: "Dataset alignment report" })).toBeVisible();
  await expect(page.getByText("Last comparison pair reopened from local source references.")).toBeVisible();

  await page.evaluate(() => window.localStorage.clear());
  await page.goto("/harness/index.html?mode=studio&empty=1");
  await expect(page.getByRole("heading", { name: "Open data" })).toBeVisible();
  await page.locator("input[type='file'][accept*='json']").first().setInputFiles(workspacePath!);
  const importedReport = page.getByRole("region", { name: "Dataset alignment report" });
  await expect(importedReport).toBeVisible();
  await expect(page.getByText(/Imported .*alignment is recomputed/)).toBeVisible();
  await expect(comparison).toHaveCount(0);
  await expect(importedReport.getByRole("button", { name: "Open A/B/difference slices" })).toBeEnabled();

  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await expect(comparison).toHaveCount(0);
  await report.getByRole("button", { name: "Close comparison" }).click();
  await expect(report).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("qualified ensemble produces mean spread probability and spaghetti evidence", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&cm1=1&ensemble=1");

  const report = page.getByRole("region", { name: "Ensemble alignment report" });
  await expect(report).toBeVisible();
  await expect(report).toContainText("3 atmospheric histories");
  await expect(report).toContainText("CM1 browser supercell");
  await expect(report).toContainText("Nimbus ensemble 01");
  await expect(report).toContainText("Nimbus ensemble 02");
  await expect(report.getByText("Matched subset", { exact: true })).toBeVisible();
  await expect(report).toContainText("48 × 36 × 24");
  await expect(report).toContainText("physical frames across every member");

  await report.getByRole("button", { name: "Open ensemble board" }).click();
  const board = page.getByRole("region", { name: "Ensemble mean spread probability and spaghetti analysis" });
  await expect(board).toBeVisible();
  await expect(report.getByRole("button", { name: "Ensemble board open" })).toHaveAttribute("aria-pressed", "true");
  await expect(board).toContainText("Strict common finite sample at every cell");
  await expect(board.getByLabel("Ensemble aggregate statistics")).toContainText("1,728 / 1,728");
  await expect(board.getByLabel("Ensemble aggregate statistics")).toContainText("Mean spread 4.000");
  await page.waitForFunction(() => {
    const graph = document.querySelector<HTMLElement & {
      data?: Array<{ type?: string }>;
      _fullLayout?: { xaxis2?: { matches?: string }; xaxis4?: { matches?: string } };
    }>(".ensemble-analysis__canvas");
    return graph?.data?.length === 6
      && graph.data.filter((trace) => trace.type === "heatmap").length === 3
      && graph.data.filter((trace) => trace.type === "contour").length === 3
      && graph._fullLayout?.xaxis2?.matches === "x"
      && graph._fullLayout?.xaxis4?.matches === "x";
  });

  await expect.poll(() => page.evaluate(() => window.__nimbusStudioHarness?.snapshot().ipc.inFlightJobs)).toBe(0);
  const originalEnsemble = await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { data: { z: ArrayLike<ArrayLike<number>> }[] }>(".ensemble-analysis__canvas")!;
    const flatten = (values: ArrayLike<ArrayLike<number>>) => Array.from(values, row => Array.from(row)).flat();
    return { mean: flatten(graph.data[0].z), spread: flatten(graph.data[1].z), jobs: window.__nimbusStudioHarness!.snapshot().ipc.startedJobs };
  });
  const threshold = board.getByLabel("Ensemble exceedance threshold");
  await threshold.fill("-7.25");
  await expect(board).toContainText("Probability > -7.25");
  const updatedEnsemble = await page.evaluate(() => {
    const graph = document.querySelector<HTMLElement & { data: { z: ArrayLike<ArrayLike<number>> }[] }>(".ensemble-analysis__canvas")!;
    const flatten = (values: ArrayLike<ArrayLike<number>>) => Array.from(values, row => Array.from(row)).flat();
    return { mean: flatten(graph.data[0].z), spread: flatten(graph.data[1].z), probability: flatten(graph.data[2].z),
      jobs: window.__nimbusStudioHarness!.snapshot().ipc.startedJobs };
  });
  expect(updatedEnsemble.mean).toEqual(originalEnsemble.mean);
  expect(updatedEnsemble.spread).toEqual(originalEnsemble.spread);
  expect(updatedEnsemble.jobs).toBe(originalEnsemble.jobs);
  expect(updatedEnsemble.probability.some(value => value > 0)).toBe(true);
  updatedEnsemble.probability.forEach((value, index) => {
    const mean = originalEnsemble.mean[index];
    const expected = [mean - 4, mean, mean + 4].filter(member => member > -7.25).length * 100 / 3;
    expect(Math.abs(value - expected)).toBeLessThan(1e-4);
  });
  await threshold.fill("");
  await expect(threshold).toHaveAttribute("aria-invalid", "true");
  await expect(board).toContainText("previous valid threshold of -7.25");
  await threshold.fill("45");
  await expect(board).toContainText("Probability > 45");
  const frameSlider = page.locator(".plot-studio__frame-bar input[type='range']");
  await frameSlider.fill("1");
  await expect(board).toContainText("30.000");
  await expect(board.getByLabel("Ensemble aggregate statistics")).toContainText("1,728 / 1,728");
  expect(pageErrors).toEqual([]);
});

test("CM1 compatibility report gates a reviewed sounding project", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=compatibility");

  const disclosure = page.getByText("CM1 → Nimbus compatibility", { exact: true });
  await expect(disclosure).toBeVisible();
  await expect(page.getByText("Draft blocked", { exact: true })).toBeVisible();
  await disclosure.click();
  await expect(page.getByText("Compatibility evidence, not runnable configuration")).toBeVisible();
  await expect(page.getByText("param0.nx", { exact: true })).toBeVisible();
  await expect(page.getByText("grid.nx", { exact: true })).toBeVisible();
  const compatibilityLedger = page.getByLabel("CM1 compatibility disposition");
  await expect(compatibilityLedger.getByText("param0.ptype", { exact: true })).toBeVisible();
  await expect(compatibilityLedger.getByText("param0.mystery_control", { exact: true })).toBeVisible();
  await page.getByText("Mechanical YAML preview", { exact: true }).click();
  const draft = page.locator(".cm1-draft")
    .filter({ hasText: "Mechanical YAML preview" })
    .locator(".cm1-draft__source pre");
  await expect(draft).toContainText("NON-RUNNABLE");
  await expect(draft).toContainText("nx: 96");
  await expect(draft).not.toContainText("microphysics:");
  await expect(page.getByText("4 included · 3 omitted", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Copy partial YAML" })).toBeEnabled();
  await page.getByText("Optional compatibility projection", { exact: true }).click();
  const sounding = page.locator(".cm1-sounding-projection .cm1-draft__source pre");
  await expect(sounding).toContainText("TMV_SHARPY_PROFILE_V1");
  await expect(sounding).toContainText("levels\t2");
  await expect(page.getByText("Hydrostatic projection · not an exact CM1 model state")).toBeVisible();
  await expect(page.getByRole("button", { name: "Copy sounding text" })).toBeEnabled();
  const saveProject = page.getByRole("button", { name: "Stage 30 s Nimbus smoke test" });
  await expect(saveProject).toBeDisabled();
  await page.getByRole("checkbox", { name: "I reviewed the projected levels, caveats, and wind behavior." }).check();
  await expect(saveProject).toBeEnabled();
  await saveProject.click();
  const saveStatus = page.locator(".cm1-sounding-projection__save-status");
  await expect(saveStatus).toContainText("Handoff staged");
  await expect(saveStatus).toContainText("fnv1a64:cm1fixture");
  await expect(saveStatus).toContainText("no-step initialization");

  const screenshotPath = testInfo.outputPath("cm1-compatibility-report.png");
  await page.locator(".cm1-compatibility").screenshot({ path: screenshotPath });
  await testInfo.attach("cm1-compatibility-report", {
    path: screenshotPath,
    contentType: "image/png",
  });

  await page.getByRole("button", { name: /^Different / }).click();
  await expect(compatibilityLedger.getByText("param0.ptype", { exact: true })).toBeVisible();
  await expect(page.getByText("param0.nx", { exact: true })).toBeHidden();
  await expect(page.getByRole("button", { name: /^Different / })).toHaveAttribute("aria-pressed", "true");
  expect(pageErrors).toEqual([]);
});

test("coarse supercell uses bounded disclosed reconstruction", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setCoarse(page, true);
  await setPlot(page, "supercell");
  await waitForWorkers(page);
  await expect(page.locator(".volume-render__quality-badge")).toBeHidden();

  for (const disclosure of ["linear 3×3×2 reconstruction", "Cloud detail: source", "native samples",
    "linear display reconstruction", "aggregate suspended condensate", "single-pass volume",
    "physical opacity per 1 km", "subtle volume edge"]) {
    await expect(page.locator(".plot3d-badge__sub")).toContainText(disclosure);
  }
  await expect(page.locator(".plot3d-badge__mode")).toContainText("Presentation");

  await page.locator(".plot3d-control-tab", { hasText: "Vortex" }).click();
  const vortexToggle = page.locator(".control-group", { hasText: "Vortex core" })
    .getByRole("button").first();
  await vortexToggle.click();
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "storm-detail source",
    { timeout: 30_000 },
  );
  await expect(page.locator(".plot3d-badge__sub")).toContainText("native cells");
  await expect(page.locator(".control-group", { hasText: "Vortex structure" })
    .getByRole("button", { name: "Core + meso" })).toHaveClass(/pill--active/);
  await expect(page.locator(".control-group", { hasText: "Meso envelope" }))
    .toContainText(/s⁻¹|No distinct/);
  await waitForWorkers(page);

  /*
   * The ground plane is now the shared radar-floor layer: it loads its own 2D
   * hyperslab at native resolution rather than reducing this scene's
   * voxel-capped volume, and it defaults to the SURFACE reduction rather than
   * the column maximum, matching the reference imagery it exists to reproduce.
   */
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
  const radarFloorControl = page.locator(".control-group", { hasText: "Radar floor" });
  await radarFloorControl.getByRole("button").click();
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-colorbar__title")).toHaveCount(0);
  await expect(radarFloorControl).toContainText("no echo");
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "registered lowest valid level plane · dbz",
  );

  // Column maximum stays reachable — it is the better view when the hook-echo
  // signature is aloft.
  const reductionControl = page.locator(".control-group", { hasText: "Floor reduction" });
  await reductionControl.getByRole("button", { name: "Column max" }).click();
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "registered column maximum plane · dbz",
  );
  await reductionControl.getByRole("button", { name: "Surface" }).click();
  await waitForWorkers(page);

  const planeFieldControl = page.locator(".control-group", { hasText: "Floor field" });
  await planeFieldControl.getByRole("combobox").selectOption("theta_prime");
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-colorbar__title")).toContainText("theta_prime floor (K)");
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "registered lowest valid level plane · symmetric diverging",
  );
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.capture())).toBe(true);
  expect((await snapshot(page)).workers).toBeLessThanOrEqual(3);
  expect(pageErrors).toEqual([]);

  const screenshotPath = testInfo.outputPath("coarse-supercell.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: screenshotPath });
  await testInfo.attach("coarse-supercell", {
    path: screenshotPath,
    contentType: "image/png",
  });
});

test("vorticity scenes expose signed structures and resolved-feature seeds", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

  await setPlot(page, "volume");
  await waitForWorkers(page);
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
  await expect(page.locator(".pill--active", { hasText: "Cool→Warm" })).toBeVisible();
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "signed diverging volume · structural deadband · opacity per 500 m",
  );
  await expect(page.locator(".volume-render__quality-badge")).toBeHidden();
  const volumePath = testInfo.outputPath("signed-vorticity-volume.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: volumePath });
  await testInfo.attach("signed-vorticity-volume", {
    path: volumePath,
    contentType: "image/png",
  });

  await setPlot(page, "isosurface");
  await waitForWorkers(page);
  await expect(page.getByText("Signed ± pair", { exact: true })).toBeVisible();
  await expect(page.locator(".plot3d-badge__sub")).toContainText("signed ± resolved surfaces");
  await expect(page.locator(".plot3d-discrete-legend__item")).toHaveCount(2);
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
  const signedPath = testInfo.outputPath("signed-vorticity-surfaces.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: signedPath });
  await testInfo.attach("signed-vorticity-surfaces", {
    path: signedPath,
    contentType: "image/png",
  });

  await page.locator(".plot3d-control-tab", { hasText: "Data" }).click();
  const isoFieldControl = page.locator(".control-group", { hasText: "Field" }).first();
  await isoFieldControl.getByRole("combobox").selectOption("okubo_weiss");
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "rotation-dominated Okubo-Weiss surface · W < 0",
  );
  await expect(page.locator(".plot3d-discrete-legend__title")).toContainText(
    "Resolved horizontal rotation core",
  );
  await expect(page.locator(".plot3d-discrete-legend__item")).toContainText("W ≤ -");
  const rotationPath = testInfo.outputPath("okubo-weiss-rotation-cores.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: rotationPath });
  await testInfo.attach("okubo-weiss-rotation-cores", {
    path: rotationPath,
    contentType: "image/png",
  });

  await isoFieldControl.getByRole("combobox").selectOption("q_criterion");
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-badge__sub")).toContainText(
    "full 3D Q-criterion surface · Q > 0 · worker-derived from u/v/w",
  );
  await expect(page.locator(".plot3d-discrete-legend__title")).toContainText(
    "Resolved 3D rotation-dominated core",
  );
  await expect(page.locator(".plot3d-discrete-legend__item")).toContainText("Q ≥");
  await expect(page.locator('[data-contour-loading="false"]')).toBeVisible({ timeout: 30_000 });
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
  const qPath = testInfo.outputPath("q-criterion-cores.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: qPath });
  await testInfo.attach("q-criterion-cores", {
    path: qPath,
    contentType: "image/png",
  });

  await setPlot(page, "streamlines");
  await page.locator(".plot3d-control-tab", { hasText: "Field" }).click();
  await page.getByText("Vorticity", { exact: true }).click();
  await waitForWorkers(page);
  await page.locator(".plot3d-control-tab", { hasText: "Tracing" }).click();
  await expect(page.getByText("Resolved features", { exact: true })).toBeVisible();
  await expect(page.locator(".plot3d-badge__sub")).toContainText("resolved-feature seeds");
  await expect(page.locator('[data-streamline-loading="false"]')).toBeVisible({ timeout: 90_000 });
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
  // The preceding Q-contour worker retains its shared lattice until its bounded idle retirement.
  await expect.poll(async () => (await snapshot(page))
    .visualization.byStage["prepared-main"].residentBytes, { timeout: 30_000 }).toBe(0);
  const streamlineResources = await snapshot(page);
  expect(streamlineResources.preparedRegistry.mainHandles).toBe(0);
  expect(streamlineResources.preparedRegistry.workerHandles).toBeGreaterThan(0);
  expect(streamlineResources.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(streamlineResources.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThan(0);
  expect(streamlineResources.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(
    streamlineResources.visualization.byStage["worker-scratch"].reservedHighWaterBytes,
  ).toBeGreaterThan(0);
  const tubesPath = testInfo.outputPath("feature-seeded-vortex-lines.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: tubesPath });
  await testInfo.attach("feature-seeded-vortex-lines", {
    path: tubesPath,
    contentType: "image/png",
  });

  /*
   * Vortex-ring seeding: the tornado-scale view.
   *
   * Uniform and ranked seeding both spread over the storm; a ring concentrates
   * every seed around the NEAR-SURFACE rotation maximum, which is what draws a
   * tornadic vortex rather than the midlevel mesocyclone. The badge must state
   * the peak rotation and its sense, because a ring drawn around a weak feature
   * looks identical to one drawn around a strong one.
   */
  const seedPlacement = page.locator(".control-group", { hasText: "Seed placement" });
  await seedPlacement.getByRole("button", { name: "Vortex ring" }).click();
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-badge__sub")).toContainText("vortex-ring seeds");
  await expect(page.locator(".plot3d-badge__sub")).toContainText(/\|zeta\| [0-9.]+ s\^-1/);
  await expect(page.locator(".plot3d-badge__sub")).toContainText(/cyclonic|anticyclonic/);
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
  const ringPath = testInfo.outputPath("vortex-ring-seeded-lines.png");
  await page.locator('[data-testid="scene"]').screenshot({ path: ringPath });
  await testInfo.attach("vortex-ring-seeded-lines", {
    path: ringPath,
    contentType: "image/png",
  });

  /*
   * The honesty property: velocity has no rotation maximum to ring, so the ring
   * option is not offered there. A vortex must never be implied by a control
   * that has nothing to centre on.
   */
  await page.locator(".plot3d-control-tab", { hasText: "Field" }).click();
  await page.getByText("Velocity", { exact: true }).click();
  await waitForWorkers(page);
  await page.locator(".plot3d-control-tab", { hasText: "Tracing" }).click();
  await expect(seedPlacement.getByRole("button", { name: "Vortex ring" })).toHaveCount(0);

  expect(pageErrors).toEqual([]);
});

test("Beast-sized prepared lattices extract off-thread without UI caps", async ({ page }, testInfo) => {
  test.setTimeout(6 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await mount(page, false);
  await waitForNoWorkers(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());

  const result = await page.evaluate(() =>
    window.__nimbusVtkHarness!.qualifyProductionGeometry()
  );
  expect(result.scalarInputBytes).toBe(40 * 1024 * 1024);
  expect(result.preparedScalarBytes).toBe(40 * 1024 * 1024);
  expect(result.workerOwnedPreparedScalarBytes).toBe(40 * 1024 * 1024);
  expect(result.combinedContourPairBytes).toBe(2 * 140 * 140 * 100 * 4);
  expect(result.vectorInputBytes).toBe(120 * 1024 * 1024);
  expect(result.workerOwnedPreparedVectorBytes).toBe(120 * 1024 * 1024);
  expect(result.contourTriangles).toBeGreaterThan(0);
  expect(result.contourBytes).toBeGreaterThan(0);
  expect(result.contourKnownScratchBytes).toBeGreaterThan(0);
  expect(result.streamlineIntegratedPoints).toBeGreaterThan(0);
  expect(result.streamlineKnownScratchBytes).toBeGreaterThan(0);
  expect(result.contourElapsedMs).toBeLessThan(180_000);
  expect(result.preparationElapsedMs).toBeLessThan(180_000);
  expect(result.workerOwnedContourPreparationElapsedMs).toBeLessThan(180_000);
  expect(result.combinedContourPairElapsedMs).toBeLessThan(180_000);
  expect(result.streamlineElapsedMs).toBeLessThan(180_000);
  expect(result.workerOwnedPreparationElapsedMs).toBeLessThan(180_000);
  expect(result.maximumWatchdogDelayMs).toBeLessThan(500);
  await waitForNoWorkers(page);
  const released = await snapshot(page);
  expect(released.workers).toBe(0);
  expect(released.preparedRegistry.handles).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.production-geometry-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    ...result,
  });
});

test("large bounded fields traverse mounted isosurface and streamline scenes", async ({ page }, testInfo) => {
  test.setTimeout(6 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.addInitScript(() => localStorage.setItem("nimbus-studio:vtk-backend:v1", "webgl"));
  await page.goto("/harness/index.html?large=1&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

  await setPlot(page, "isosurface");
  await mount(page, true);
  await waitForDisplayedFrame(page, "isosurface", 0);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await expect.poll(async () => (await snapshot(page)).gpuLogicalBufferBytes).toBeGreaterThan(0);
  const isosurface = await snapshot(page);
  expect(isosurface.large).toBe(true);
  expect(isosurface.fixtureArrayBufferBytes).toBe(40 * 1024 * 1024);
  expect(isosurface.visualization.byStage["decoded-field"].residentBytes).toBeGreaterThanOrEqual(
    40 * 1024 * 1024,
  );
  expect(isosurface.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(isosurface.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThanOrEqual(
    40 * 1024 * 1024,
  );
  expect(isosurface.preparedRegistry.mainHandles).toBe(0);
  expect(isosurface.preparedRegistry.workerHandles).toBe(1);
  expect(isosurface.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(isosurface.gpuLogicalBufferBytes).toBeGreaterThan(0);
  expect(isosurface.visualization.prediction.underpredictions).toBe(0);

  await setPlot(page, "streamlines");
  try {
    await waitForDisplayedFrame(page, "streamlines", 0);
  } catch (error) {
    await writeFile(testInfo.outputPath("streamlines-timeout.json"), JSON.stringify(await snapshot(page), null, 2));
    throw error;
  }
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const streamlines = await snapshot(page);
  expect(streamlines.visualization.byStage["decoded-field"].residentBytes).toBeGreaterThanOrEqual(
    120 * 1024 * 1024,
  );
  expect(streamlines.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(streamlines.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThanOrEqual(
    120 * 1024 * 1024,
  );
  expect(streamlines.preparedRegistry.mainHandles).toBe(0);
  expect(streamlines.preparedRegistry.workerHandles).toBe(1);
  expect(streamlines.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(streamlines.gpuLogicalBufferBytes).toBeGreaterThan(0);
  expect(streamlines.visualization.prediction.underpredictions).toBe(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(released.gpuLogicalBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.large-mounted-geometry-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    isosurface: isosurface.visualization,
    streamlines: streamlines.visualization,
    released: released.visualization,
    logicalGpu: {
      isosurfaceBuffers: isosurface.gpuLogicalBufferBytes,
      isosurfaceTextures: isosurface.gpuLogicalTextureBytes,
      streamlinesBuffers: streamlines.gpuLogicalBufferBytes,
      streamlinesTextures: streamlines.gpuLogicalTextureBytes,
      released: released.gpuLogicalBytes,
    },
  });
});

test("two large 3D panes share source residency and release aggregate resources", async ({ page }, testInfo) => {
  test.setTimeout(8 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1&panes=2&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

  await setPlot(page, "isosurface");
  await mount(page, true);
  await waitForDisplayedFrame(page, "isosurface", 0);
  await expect(page.locator(".harness-scene canvas")).toHaveCount(2);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const isosurface = await snapshot(page);
  expect(isosurface.paneCount).toBe(2);
  expect(isosurface.canvasCount).toBe(2);
  expect(isosurface.visualization.byStage["decoded-field"].residentBytes)
    .toBeGreaterThanOrEqual(40 * 1024 * 1024);
  expect(isosurface.visualization.byStage["decoded-field"].residentBytes)
    .toBeLessThan(80 * 1024 * 1024);
  expect(isosurface.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(isosurface.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThan(0);
  expect(isosurface.visualization.byStage["worker-resident"].residentBytes)
    .toBeLessThan(50 * 1024 * 1024);
  expect(isosurface.preparedRegistry.mainHandles).toBe(0);
  expect(isosurface.preparedRegistry.workerHandles).toBe(2);
  expect(isosurface.preparedRegistry.references).toBe(4);
  expect(isosurface.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(isosurface.gpuLogicalBufferBytes).toBeGreaterThan(0);
  expect(isosurface.visualization.prediction.underpredictions).toBe(0);

  await setPlot(page, "streamlines");
  await waitForDisplayedFrame(page, "streamlines", 0);
  await expect(page.locator(".harness-scene canvas")).toHaveCount(2);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const streamlines = await snapshot(page);
  expect(streamlines.visualization.byStage["decoded-field"].residentBytes)
    .toBeGreaterThanOrEqual(100 * 1024 * 1024);
  expect(streamlines.visualization.byStage["decoded-field"].residentBytes)
    .toBeLessThan(160 * 1024 * 1024);
  expect(streamlines.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(streamlines.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThan(0);
  expect(streamlines.visualization.byStage["worker-resident"].residentBytes)
    .toBeLessThan(120 * 1024 * 1024);
  expect(streamlines.preparedRegistry.mainHandles).toBe(0);
  expect(streamlines.preparedRegistry.workerHandles).toBe(1);
  expect(streamlines.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(streamlines.gpuLogicalBufferBytes).toBeGreaterThan(0);
  expect(streamlines.visualization.prediction.underpredictions).toBe(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.canvasCount).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(released.gpuLogicalBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.large-two-pane-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    paneCount: 2,
    isosurface: {
      visualization: isosurface.visualization,
      logicalGpuBytes: isosurface.gpuLogicalBytes,
    },
    streamlines: {
      visualization: streamlines.visualization,
      logicalGpuBytes: streamlines.gpuLogicalBytes,
    },
    released: {
      visualization: released.visualization,
      logicalGpuBytes: released.gpuLogicalBytes,
    },
  });
});

test("four large streamline panes remain bounded and failure-atomic", async ({ page }, testInfo) => {
  test.setTimeout(8 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1&panes=4&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));

  await setPlot(page, "streamlines");
  await mount(page, true);
  await waitForDisplayedFrame(page, "streamlines", 0);
  await expect(page.locator(".harness-scene canvas")).toHaveCount(4);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const mounted = await snapshot(page);
  expect(mounted.paneCount).toBe(4);
  expect(mounted.canvasCount).toBe(4);
  expect(mounted.preparedRegistry.handles).toBe(1);
  expect(mounted.preparedRegistry.references).toBe(4);
  expect(mounted.preparedRegistry.mainHandles).toBe(0);
  expect(mounted.preparedRegistry.workerHandles).toBe(1);
  expect(mounted.visualization.byStage["decoded-field"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["decoded-field"].residentBytes)
    .toBeLessThan(120 * 1024 * 1024);
  expect(mounted.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(mounted.visualization.prediction.samples).toBe(1);
  expect(mounted.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["worker-resident"].residentBytes)
    .toBeLessThan(40 * 1024 * 1024);
  expect(mounted.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(mounted.gpuLogicalBufferBytes).toBeGreaterThan(0);
  expect(mounted.visualization.prediction.underpredictions).toBe(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.canvasCount).toBe(0);
  expect(released.preparedRegistry.handles).toBe(0);
  expect(released.preparedRegistry.references).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(released.gpuLogicalBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.large-four-pane-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    paneCount: 4,
    mounted: {
      visualization: mounted.visualization,
      logicalGpuBytes: mounted.gpuLogicalBytes,
    },
    released: {
      visualization: released.visualization,
      logicalGpuBytes: released.gpuLogicalBytes,
    },
  });
});

test("four large signed-isosurface panes expose aggregate contour ownership", async ({ page }, testInfo) => {
  test.setTimeout(8 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1&panes=4&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "isosurface");
  await mount(page, true);
  await waitForDisplayedFrame(page, "isosurface", 0);
  await expect(page.locator(".harness-scene canvas")).toHaveCount(4);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const mounted = await snapshot(page);
  expect(mounted.preparedRegistry.handles).toBe(2);
  expect(mounted.preparedRegistry.references).toBe(8);
  expect(mounted.preparedRegistry.mainHandles).toBe(0);
  expect(mounted.preparedRegistry.workerHandles).toBe(2);
  expect(mounted.visualization.byStage["decoded-field"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(mounted.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["worker-resident"].residentBytes)
    .toBeLessThan(30 * 1024 * 1024);
  expect(mounted.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["retained-geometry"].residentBytes)
    .toBeLessThan(3 * 1024 * 1024);
  expect(mounted.visualization.prediction.underpredictions).toBe(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.preparedRegistry.handles).toBe(0);
  expect(released.preparedRegistry.references).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(released.gpuLogicalBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.large-four-pane-contour-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    mounted: mounted.visualization,
    logicalGpuBytes: mounted.gpuLogicalBytes,
    released: released.visualization,
  });
});

test("two large supercell panes sustain cloud flow vortex and condensate layers", async ({ page }, testInfo) => {
  test.setTimeout(10 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1&panes=2&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "supercell");
  await mount(page, true);
  try { await waitForDisplayedFrame(page, "supercell", 0); }
  catch (error) {
    const state = await page.evaluate(() => ({
      scene: { ...document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset },
      panes: Array.from(document.querySelectorAll<HTMLElement>(".plot3d-overlay-host"), pane => ({ ...pane.dataset })),
      appearance: window.__nimbusVtkHarness!.inspectAppearance(),
      resources: window.__nimbusVtkHarness!.snapshot(),
    })).catch(diagnosticError => ({ diagnosticError: String(diagnosticError) }));
    await testInfo.attach("initial-large-pane-state", { body: JSON.stringify(state), contentType: "application/json" });
    throw error;
  }

  const panes = page.locator(".harness-pane");
  await expect(panes).toHaveCount(2);
  for (let paneIndex = 0; paneIndex < 2; paneIndex += 1) {
    const pane = panes.nth(paneIndex);
    await pane.locator(".plot3d-control-tab", { hasText: "Flow" }).click();
    await pane.locator(".control-group", { hasText: "Updraft / downdraft" })
      .getByRole("button").first().click();
    await pane.locator(".control-group", { hasText: "|w| threshold" })
      .locator('input[type="range"]').fill("2");
    await pane.locator(".plot3d-control-tab", { hasText: "Vortex" }).click();
    await pane.locator(".control-group", { hasText: "Vortex core" })
      .getByRole("button").first().click();
    const condensationToggle = pane.locator(".control-group", { hasText: "Tornadic condensation" })
      .getByRole("button").first();
    await expect(condensationToggle).toBeEnabled({ timeout: 30_000 });
    await condensationToggle.click();
  }

  const scenes = page.locator(".volume-render__canvas-wrap");
  await expect(scenes).toHaveCount(2);
  await expect(page.locator('.volume-render__canvas-wrap[data-vortex-core-enabled="true"]')).toHaveCount(2);
  await expect(page.locator('.volume-render__canvas-wrap[data-tornadic-condensation-enabled="true"]')).toHaveCount(2);
  const waitForLayer = (attribute: string, timeout: number) => page.waitForFunction((name) => {
    const errors = [...document.querySelectorAll<HTMLElement>(".volume-render__overlay--error")]
      .filter((element) => element.getClientRects().length > 0)
      .map((element) => element.textContent);
    if (errors.length > 0) throw new Error(`Supercell layer failed: ${errors.join("; ")}`);
    return document.querySelectorAll(`.volume-render__canvas-wrap[${name}="false"]`).length === 2;
  }, attribute, { timeout });
  await waitForLayer("data-flow-preparation-loading", 90_000);
  await waitForLayer("data-flow-extraction-loading", 90_000);
  await waitForLayer("data-flow-contour-loading", 180_000);
  await waitForLayer("data-vortex-preparation-loading", 90_000);
  await waitForLayer("data-condensate-preparation-loading", 90_000);
  await waitForLayer("data-vortex-extraction-loading", 30_000);
  await waitForLayer("data-vortex-contour-loading", 180_000);
  await waitForLayer("data-scene-frame-commit-pending", 180_000);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await expect(page.locator(".harness-scene canvas")).toHaveCount(2);
  const mounted = await snapshot(page);
  const budgetPath = testInfo.outputPath("supercell-resource-budget.json");
  await writeFile(budgetPath, JSON.stringify(mounted.visualization, null, 2));
  await testInfo.attach("supercell-resource-budget", { path: budgetPath, contentType: "application/json" });
  expect(mounted.preparedRegistry.handles).toBeGreaterThan(0);
  // Identical preparations share only while their worker requests overlap.
  expect(mounted.preparedRegistry.references).toBeGreaterThanOrEqual(mounted.preparedRegistry.handles);
  expect(mounted.visualization.byStage["decoded-field"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["prepared-main"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["worker-resident"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["worker-resident"].residentBytes)
    .toBeLessThan(64 * 1024 * 1024);
  expect(mounted.visualization.byStage["retained-geometry"].residentBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["retained-geometry"].residentBytes)
    .toBeLessThan(32 * 1024 * 1024);
  expect(mounted.visualization.highWaterBytes).toBeLessThan(256 * 1024 * 1024);
  expect(mounted.visualization.prediction.underpredictions).toBe(0);
  expect(mounted.gpuLogicalTextureBytes).toBeGreaterThan(0);
  expect(mounted.gpuLogicalBufferBytes).toBeGreaterThan(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.preparedRegistry.handles).toBe(0);
  expect(released.preparedRegistry.references).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(released.gpuLogicalBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.large-two-pane-supercell-layers-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    mounted: mounted.visualization,
    logicalGpu: {
      buffers: mounted.gpuLogicalBufferBytes,
      textures: mounted.gpuLogicalTextureBytes,
    },
    released: released.visualization,
  });
});

test("playback bounds idle extraction services across prepared generations", async ({ page }) => {
  test.setTimeout(4 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "isosurface");
  await mount(page, true);

  for (let step = 0; step < 10; step += 1) {
    await setStep(page, step);
    await waitForDisplayedFrame(page, "isosurface", step);
    const frame = await snapshot(page);
    expect(frame.extractionPool.warmServices)
      .toBeLessThanOrEqual(frame.extractionPool.maximumWarmServices);
  }
  const cycled = await snapshot(page);
  expect(cycled.extractionPool.maximumWarmServices).toBe(3);
  // Worker-owned contour preparation coalesces these generations before the
  // legacy extraction pool needs to evict them; zero evictions is valid as
  // long as the warm-service and worker ceilings remain bounded.
  expect(cycled.workers).toBeLessThanOrEqual(4);

  await mount(page, false);
  await waitForNoWorkers(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.extractionPool.warmServices).toBe(0);
  expect(released.preparedRegistry.handles).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(pageErrors).toEqual([]);
});

test("large volume reports logical GPU texture allocation and releases it", async ({ page }, testInfo) => {
  test.setTimeout(4 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "volume");
  await mount(page, true);
  await waitForDisplayedFrame(page, "volume", 0);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const mounted = await snapshot(page);
  expect(mounted.gpuLogicalTextureBytes).toBeGreaterThan(0);
  expect(mounted.visualization.byStage["prepared-main"].residentBytes)
    .toBeGreaterThanOrEqual(40 * 1024 * 1024);
  expect(mounted.visualization.byStage.gpu.residentBytes)
    .toBeGreaterThanOrEqual(40 * 1024 * 1024);
  expect(Number(await page.locator(".plot3d-overlay-host")
    .getAttribute("data-logical-gpu-upload-bytes"))).toBeGreaterThanOrEqual(40 * 1024 * 1024);

  await mount(page, false);
  await waitForNoWorkers(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.gpuLogicalTextureBytes).toBe(0);
  expect(released.gpuLogicalBufferBytes).toBe(0);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
  expect(pageErrors).toEqual([]);

  await writeMetrics(testInfo, {
    schema: "nimbus.studio.large-volume-logical-gpu-qualification",
    schemaVersion: 1,
    browser: testInfo.project.name,
    mounted: {
      logicalBufferBytes: mounted.gpuLogicalBufferBytes,
      logicalTextureBytes: mounted.gpuLogicalTextureBytes,
      visualization: mounted.visualization,
    },
    released: {
      logicalBytes: released.gpuLogicalBytes,
      visualization: released.visualization,
    },
  });
});

test("trajectory tubes admit conservative then exact logical GPU ownership", async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "trajectories");
  await mount(page, true);
  await expect.poll(async () => Number(await page.locator(".plot3d-overlay-host")
    .getAttribute("data-logical-gpu-upload-bytes")), { timeout: 60_000 }).toBeGreaterThan(0);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  const mounted = await snapshot(page);
  expect(mounted.visualization.byStage.gpu.residentBytes).toBeGreaterThan(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.visualization.byStage.gpu.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
});

test("large field I/O and extraction cancel atomically and recover", async ({ page }) => {
  test.setTimeout(6 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?large=1&mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "isosurface");

  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(1_000));
  await mount(page, true);
  await page.waitForFunction(() =>
    (window.__nimbusVtkHarness?.snapshot().visualization.byStage["native-read"].reservedBytes ?? 0) > 0
  );
  await mount(page, false);
  await page.evaluate(() => window.__nimbusVtkHarness?.setFieldReadDelay(0));
  await waitForIdleIpc(page);
  await waitForNoWorkers(page);
  await page.waitForFunction(() =>
    window.__nimbusVtkHarness?.snapshot().visualization.reservedBytes === 0
  );
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const cancelledRead = await snapshot(page);
  expect(cancelledRead.visualization.residentBytes).toBe(0);
  expect(cancelledRead.visualization.reservedBytes).toBe(0);

  await mount(page, true);
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-contour-loading="true"]');
    const workerBytes = window.__nimbusVtkHarness?.snapshot()
      .visualization.byStage["worker-resident"].residentBytes ?? 0;
    return Boolean(scene && workerBytes > 0);
  });
  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  await page.waitForFunction(() =>
    window.__nimbusVtkHarness?.snapshot().visualization.reservedBytes === 0
  );
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const cancelledExtraction = await snapshot(page);
  expect(cancelledExtraction.visualization.residentBytes).toBe(0);
  expect(cancelledExtraction.visualization.reservedBytes).toBe(0);

  await mount(page, true);
  await waitForDisplayedFrame(page, "isosurface", 0);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  expect((await snapshot(page)).visualization.byStage["retained-geometry"].residentBytes)
    .toBeGreaterThan(0);
  expect(pageErrors).toEqual([]);

  await mount(page, false);
  await waitForNoWorkers(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.clearFieldCache());
  const released = await snapshot(page);
  expect(released.visualization.residentBytes).toBe(0);
  expect(released.visualization.reservedBytes).toBe(0);
});

test("paired fine/coarse supercells remain distinct across time and display scale", async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "supercell");
  await waitForDisplayedFrame(page, "supercell", 0);
  await waitForWorkers(page);

  const scene = page.locator('[data-testid="scene"]');
  await expect(page.locator(".plot3d-badge__mode")).toContainText("Scientific");
  await expect(page.locator(".volume-render__quality-badge")).toBeHidden();
  const finePath = testInfo.outputPath("paired-fine-step-0.png");
  const finePng = await scene.screenshot({ path: finePath });
  const fine = await snapshot(page);

  await setCoarse(page, true);
  await waitForWorkers(page);
  await expect(page.locator(".plot3d-badge__mode")).toContainText("Presentation");
  await expect(page.locator(".plot3d-badge__sub")).toContainText("linear 3×3×2 reconstruction");
  for (const disclosure of ["Cloud detail: source", "native samples", "linear display reconstruction"]) {
    await expect(page.locator(".plot3d-badge__sub")).toContainText(disclosure);
  }
  await expect(page.locator(".volume-render__quality-badge")).toBeHidden();
  const coarsePath = testInfo.outputPath("paired-coarse-step-0.png");
  const coarsePng = await scene.screenshot({ path: coarsePath });
  const coarse = await snapshot(page);

  await setStep(page, 8);
  await waitForDisplayedFrame(page, "supercell", 8);
  await waitForWorkers(page);
  for (const disclosure of ["Cloud detail: source", "native samples", "linear display reconstruction"]) {
    await expect(page.locator(".plot3d-badge__sub")).toContainText(disclosure);
  }
  await expect(page.locator(".volume-render__quality-badge")).toBeHidden();
  const temporalPath = testInfo.outputPath("paired-coarse-step-8.png");
  const temporalPng = await scene.screenshot({ path: temporalPath });
  await waitForNoWorkers(page);
  const temporal = await snapshot(page);

  const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
  expect(finePng.byteLength).toBeGreaterThan(20_000);
  expect(coarsePng.byteLength).toBeGreaterThan(20_000);
  expect(temporalPng.byteLength).toBeGreaterThan(20_000);
  expect(hash(coarsePng)).not.toBe(hash(finePng));
  expect(hash(temporalPng)).not.toBe(hash(coarsePng));

  const display = await page.evaluate(() => {
    const canvas = document.querySelector<HTMLCanvasElement>(".harness-scene canvas");
    const rect = canvas?.getBoundingClientRect();
    return {
      dpr: window.devicePixelRatio,
      cssWidth: rect?.width ?? 0,
      cssHeight: rect?.height ?? 0,
    };
  });
  expect(temporal.canvasWidth).toBeCloseTo(display.cssWidth * display.dpr, -1);
  expect(temporal.canvasHeight).toBeCloseTo(display.cssHeight * display.dpr, -1);
  expect(fine.canvasWidth).toBe(coarse.canvasWidth);
  expect(fine.canvasHeight).toBe(coarse.canvasHeight);
  expect(temporal.workers).toBe(0);
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
  expect(pageErrors).toEqual([]);

  for (const [name, path] of [
    ["paired-fine-step-0", finePath],
    ["paired-coarse-step-0", coarsePath],
    ["paired-coarse-step-8", temporalPath],
  ] as const) {
    await testInfo.attach(name, { path, contentType: "image/png" });
  }
});

test("lost VTK scene rebuilds when the driver omits the restore event", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "volume");
  await waitForWorkers(page);

  const beforeLoss = await snapshot(page);
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.loseContext())).toBe(true);
  const contextOverlay = page.locator(".volume-render__overlay--error", {
    hasText: "3D graphics context lost",
  });
  await expect(contextOverlay).toBeVisible();

  // Deliberately do not call restoreContext(). After the bounded grace period,
  // Nimbus must abandon the dead canvas, prepare the same immutable frame on a
  // fresh lifecycle generation, and clear the overlay only after real pixels
  // have rendered. This covers drivers that stop emitting webglcontextrestored
  // after sustained context churn.
  await expect(contextOverlay).toBeHidden({ timeout: 20_000 });
  await waitForWorkers(page);
  const recovered = await snapshot(page);
  expect(recovered.contextLosses).toBe(beforeLoss.contextLosses + 1);
  expect(recovered.contextRestores).toBe(beforeLoss.contextRestores);
  expect(recovered.canvases).toBe(1);
  expect(recovered.webglContexts).toBe(1);
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.capture())).toBe(true);
  expect(pageErrors).toEqual([]);
});

test("rapid isosurface supersession terminates stale worker geometry", async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await setPlot(page, "isosurface");
  await waitForWorkers(page);
  await waitForDisplayedFrame(page, "isosurface", (await snapshot(page)).step);
  const maxWatchdogDelay = await page.evaluate(async () => {
    let maximum = 0;
    let previous = performance.now();
    let running = true;
    const watch = (now: number) => {
      maximum = Math.max(maximum, now - previous);
      previous = now;
      if (running) requestAnimationFrame(watch);
    };
    requestAnimationFrame(watch);
    for (let index = 0; index < 60; index += 1) {
      window.__nimbusVtkHarness?.setStep(index % 20);
      await new Promise((resolve) => setTimeout(resolve, 8));
    }
    await new Promise((resolve) => setTimeout(resolve, 350));
    running = false;
    return maximum;
  });
  expect(maxWatchdogDelay).toBeLessThan(250);
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  const final = await snapshot(page);
  expect(final.visualization.reservedBytes).toBe(0);
  expect(final.workers).toBe(0);
});

test("production VTK scenes survive lifecycle, field, context, resize, and contour stress", async ({ page }, testInfo) => {
  // Five plot types, 500 default remounts and 100 complete frames accumulate
  // software-renderer work beyond 15 minutes. Frame and watchdog bounds remain separate.
  test.setTimeout(25 * 60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await waitForCanvas(page, true);
  await waitForWorkers(page);

  const plotTypes = await page.evaluate(() =>
    [...(window.__nimbusVtkHarness?.plotTypes ?? [])],
  );
  expect(plotTypes).toHaveLength(5);
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.capture())).toBe(true);

  const lifecycle: HarnessSnapshot[] = [];
  for (const plot of plotTypes) {
    await setPlot(page, plot);
    for (let cycle = 0; cycle < mountCycles; cycle += 1) {
      await mount(page, false);
      const released = await snapshot(page);
      expect(released.canvases, `${plot} cycle ${cycle} canvas leak`).toBe(0);
      expect(released.webglContexts, `${plot} cycle ${cycle} context leak`).toBe(0);
      expect(
        released.workers,
        `${plot} cycle ${cycle} exceeded the single warm preparation-worker boundary`,
      ).toBeLessThanOrEqual(1);
      await mount(page, true);
    }
    await waitForWorkers(page);
    lifecycle.push(await snapshot(page));
  }

  for (const plot of plotTypes) {
    await setPlot(page, plot);
    for (let step = 0; step < 20; step += 1) {
      await page.evaluate((next) => window.__nimbusVtkHarness?.setStep(next), step);
      await page.waitForFunction((next) =>
        document.querySelector(".harness-shell")?.getAttribute("data-step") === String(next),
      step);
      await waitForDisplayedFrame(page, plot, step);
      expect(
        (await snapshot(page)).workers,
        `${plot} step ${step} exceeded the bounded active-scene worker budget`,
      ).toBeLessThanOrEqual(3);
      expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);
    }
  }

  await setPlot(page, "volume");
  await waitForWorkers(page);
  const beforeLoss = await snapshot(page);
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.loseContext())).toBe(true);
  const contextOverlay = page.locator(".volume-render__overlay--error", {
    hasText: "3D graphics context lost",
  });
  await expect(contextOverlay).toBeVisible();
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.restoreContext())).toBe(true);
  await expect(contextOverlay).toBeHidden();
  await waitForWorkers(page);
  const afterRestore = await snapshot(page);
  expect(afterRestore.contextLosses).toBe(beforeLoss.contextLosses + 1);
  expect(afterRestore.contextRestores).toBe(beforeLoss.contextRestores + 1);
  expect(afterRestore.canvases).toBe(1);
  expect(afterRestore.webglContexts).toBe(1);
  expect(await page.evaluate(() => window.__nimbusVtkHarness?.capture())).toBe(true);

  const beforeResize = await snapshot(page);
  await page.evaluate(() => window.__nimbusVtkHarness?.resizeStorm());
  await page.waitForFunction(() => {
    const canvas = document.querySelector<HTMLCanvasElement>(".harness-scene canvas");
    return Boolean(canvas && canvas.width > 0 && canvas.height > 0);
  });
  await page.waitForTimeout(250);
  const afterResize = await snapshot(page);
  expect(afterResize.canvasMutations - beforeResize.canvasMutations).toBeLessThanOrEqual(8);
  expect(afterResize.canvases).toBe(1);

  // Rapid re-render must not stall the main thread. Frame stepping drives the
  // contour rebuild here; it replaces an equivalent threshold sweep that ran
  // against the removed JSON-spec isosurface widget.
  await setPlot(page, "isosurface");
  await waitForWorkers(page);
  await waitForDisplayedFrame(page, "isosurface", (await snapshot(page)).step);
  const maxWatchdogDelay = await page.evaluate(async () => {
    let maxDelay = 0;
    let previous = performance.now();
    let running = true;
    const watch = (now: number) => {
      maxDelay = Math.max(maxDelay, now - previous);
      previous = now;
      if (running) requestAnimationFrame(watch);
    };
    requestAnimationFrame(watch);
    for (let i = 0; i < 60; i += 1) {
      window.__nimbusVtkHarness?.setStep(i % 20);
      await new Promise((resolve) => setTimeout(resolve, 8));
    }
    await new Promise((resolve) => setTimeout(resolve, 350));
    running = false;
    return maxDelay;
  });
  expect(maxWatchdogDelay).toBeLessThan(250);
  expect(await page.locator(".volume-render__overlay--error").count()).toBe(0);

  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  const final = await snapshot(page);
  expect(final.canvases).toBe(0);
  expect(final.webglContexts).toBe(0);
  expect(final.gpuLogicalBytes).toBe(0);
  expect(final.workers).toBe(0);
  expect(final.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(final.visualization.byStage["worker-resident"].residentBytes).toBe(0);
  expect(final.visualization.byStage["retained-geometry"].residentBytes).toBe(0);
  expect(final.visualization.reservedBytes).toBe(0);
  expect(final.visualization.residentBytes)
    .toBe(final.visualization.byStage["decoded-field"].residentBytes);
  expect(final.ipc.maxInFlightJobs).toBeLessThanOrEqual(3);
  expect(final.ipc.inFlightJobs).toBe(0);
  expect(final.ipc.inFlightBytes).toBe(0);
  expect(final.ipc.failures, final.ipc.failureMessages.join("\n")).toBe(0);
  expect(final.captureFailures).toBe(0);
  expect(final.contextLosses).toBe(afterRestore.contextLosses);
  expect(pageErrors, "uncaught browser/worker errors").toEqual([]);

  await writeMetrics(testInfo, {
    evidence: "software-browser",
    browser: testInfo.project.name,
    mountCyclesPerPlot: mountCycles,
    plots: plotTypes,
    lifecycle,
    contextRecovery: { beforeLoss, afterRestore },
    resize: { beforeResize, afterResize },
    contourWatchdogMaxMs: maxWatchdogDelay,
    pageErrors,
    final,
    unsupportedInBrowserLane: {
      nativeTauriWkWebView: null,
      processRssBytes: null,
      gpuTimeMs: null,
      simulatorContention: null,
    },
  });
});

test("playback, plot switching, and capture resources plateau", async ({ page }, testInfo) => {
  test.setTimeout(soakMs + SOAK_TAIL_BUDGET_MS);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await waitForCanvas(page, true);

  const plotTypes = await page.evaluate(() =>
    [...(window.__nimbusVtkHarness?.plotTypes ?? [])],
  );
  const samples: HarnessSnapshot[] = [];
  const started = Date.now();
  let iteration = 0;
  // The default ten-second lane is a smoke, not a soak. On software rendering
  // one cold iteration can consume almost that whole budget, so also require a
  // minimum sample count before evaluating a plateau. The 30-minute lane is
  // duration-dominated and this floor has no effect there.
  const minimumSamples = 6;
  // Taken once the run is warm, and compared against the same measurement at
  // the end. Both are post-collection reads of the same quantity, which is
  // what makes the ratio between them mean "still holding" rather than
  // "collected more recently".
  let baselineHeap: number | null = null;
  while (Date.now() - started < soakMs || samples.length < minimumSamples) {
    const plot = plotTypes[iteration % plotTypes.length];
    await setPlot(page, plot);
    await page.evaluate((step) => window.__nimbusVtkHarness?.setStep(step), iteration % 20);
    if (iteration % 5 === 0) {
      await mount(page, false);
      await mount(page, true);
    }
    await waitForWorkers(page);
    expect(await page.evaluate(() => window.__nimbusVtkHarness?.capture())).toBe(true);
    samples.push(await snapshot(page));
    iteration += 1;
    // A third of the way in: past first-load caches and lazy chunk evaluation,
    // which would otherwise be charged to the app as growth.
    if (baselineHeap === null && Date.now() - started >= soakMs / 3) {
      baselineHeap = await settledHeapBytes(page);
    }
  }

  await mount(page, false);
  await waitForNoWorkers(page);
  await waitForIdleIpc(page);
  const final = await snapshot(page);
  const warm = samples.slice(Math.floor(samples.length / 3));
  expect(warm.length).toBeGreaterThan(3);
  expect(Math.max(...warm.map((sample) => sample.canvases))).toBeLessThanOrEqual(1);
  expect(Math.max(...warm.map((sample) => sample.webglContexts))).toBeLessThanOrEqual(1);
  expect(Math.max(...warm.map((sample) => sample.workers))).toBeLessThanOrEqual(3);
  expect(monotonicGrowth(warm.map((sample) => sample.webglContexts))).toBe(false);
  expect(monotonicGrowth(warm.map((sample) => sample.textures))).toBe(false);
  expect(monotonicGrowth(warm.map((sample) => sample.buffers))).toBe(false);
  expect(final.canvases).toBe(0);
  expect(final.webglContexts).toBe(0);
  expect(final.gpuLogicalBytes).toBe(0);
  expect(final.workers).toBe(0);
  expect(final.visualization.byStage["prepared-main"].residentBytes).toBe(0);
  expect(final.visualization.byStage["worker-resident"].residentBytes).toBe(0);
  expect(final.visualization.byStage["retained-geometry"].residentBytes).toBe(0);
  expect(final.visualization.reservedBytes).toBe(0);
  expect(final.visualization.residentBytes)
    .toBe(final.visualization.byStage["decoded-field"].residentBytes);
  expect(final.ipc.inFlightJobs).toBe(0);
  expect(final.captureFailures).toBe(0);

  /*
   * Retention, measured against a collected baseline rather than a raw sample.
   *
   * The raw per-sample counter is kept in `samples` for diagnosis, but it is
   * not what gets asserted: it sawtooths, and a point-to-point comparison of
   * it fails or passes according to when the last collection ran. The check
   * below compares two forced-collection reads of the same quantity, so a
   * failure means the page is still holding what it took, which is the thing
   * this soak exists to detect.
   *
   * `heapChecked` is recorded so a run that skipped this -- WebKit, which has
   * no collection hook -- reports the gap instead of reading as a pass.
   */
  /*
   * How much of the end-of-run heap was cache. Recorded, not asserted: it is
   * the number that explains a retention failure, and its absence is why the
   * earlier version of this check pointed at a leak that was a warm LRU.
   * Taken by clearing, which is what the settle below would do anyway.
   */
  const fieldCacheBytesAtEnd = await page.evaluate(
    () => window.__nimbusVtkHarness?.clearFieldCache() ?? null,
  );
  const finalHeap = await settledHeapBytes(page);
  const heapChecked = soakMinutes >= 30 && baselineHeap !== null && finalHeap !== null;
  const heap = warm.map((sample) => sample.jsHeapBytes).filter((value): value is number => value !== null);
  /*
   * Whether the heap counter is a measurement at all.
   *
   * Chrome quantizes `performance.memory` and refreshes it at most every ~20
   * minutes unless launched with --enable-precise-memory-info. Without that
   * flag this soak took 1146 samples over half an hour and saw exactly TWO
   * distinct values -- so the retention check was comparing a cached bucket to
   * itself, and passed or failed on whether the one refresh happened to land
   * between the baseline and the final read. It found a 97 MB leak by luck.
   *
   * A live counter moves constantly. Asserting that it did is the only thing
   * standing between a dropped launch flag and a soak that reports green
   * forever, and it costs nothing.
   */
  const distinctHeapReadings = new Set(heap).size;

  /*
   * Written BEFORE the assertions, deliberately.
   *
   * A thirty-minute run that fails its last assertion is the run whose numbers
   * are most worth having, and an assertion that throws first takes the whole
   * record with it -- which is exactly what happened twice while this check was
   * being sorted out, costing a re-run each time just to see the inputs.
   */
  await writeMetrics(testInfo, {
    evidence: testInfo.project.name.startsWith("webkit") ? "webkit-browser-soak" : "software-browser-soak",
    browser: testInfo.project.name,
    configuredMinutes: soakMinutes,
    contentionProfile: process.env.NIMBUS_CONTENTION_PROFILE ?? "simulator-off",
    elapsedMs: Date.now() - started,
    iterations: iteration,
    samples,
    final,
    plateau: {
      warmupSamplesDiscarded: samples.length - warm.length,
      strictHeapPlateauApplied: soakMinutes >= 30 && heap.length >= 6,
      // The retention check is the one that can be absent without the run
      // looking any different, so what it did is recorded rather than implied.
      // A null baseline or final means the browser exposes no collection hook.
      heapRetentionChecked: heapChecked,
      baselineHeapBytes: baselineHeap,
      finalHeapBytes: finalHeap,
      fieldCacheBytesAtEnd,
      /** Two of these means the counter is a cached bucket — see above. */
      distinctHeapReadings,
      heapSamples: heap.length,
    },
    unsupportedInBrowserLane: {
      nativeTauriWkWebView: null,
      processRssBytes: null,
      gpuTimeMs: null,
      simulatorContention: null,
    },
  });

  if (heapChecked) {
    /*
     * Asserted BEFORE the retention ratio, because it decides whether that
     * ratio means anything. A quantized counter yields a handful of values
     * across a thousand samples; a live one moves on nearly every read. Ten is
     * far above the two a cached bucket produces and far below the hundreds a
     * real measurement does, so it fails only for the reason it exists.
     */
    expect(distinctHeapReadings,
      "performance.memory looks quantized — is --enable-precise-memory-info still set?")
      .toBeGreaterThan(10);
    expect(finalHeap!).toBeLessThanOrEqual(baselineHeap! * 1.2);
  }
  // Monotonic growth is a leak signature the raw counter CAN carry: a sawtooth
  // is not monotonic, so a strictly rising series means collection never
  // reclaimed anything across the whole warm window.
  if (soakMinutes >= 30 && heap.length >= 6) {
    expect(monotonicGrowth(heap)).toBe(false);
  }
});

test("mounted Studio tails complete publications and keeps the last requested run", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=studio");
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));

  await expect(page.locator(".plot-studio__meta")).toContainText("3 steps");
  await expect(page.locator(".plot-studio__in-progress")).toContainText(
    "1 incomplete or invalid frame hidden",
  );
  await expect(page.locator(".plot-studio__meta")).toContainText("showing t = 0.0 s");

  const slider = page.locator(".plot-studio__frame-bar input[type=range]");
  await slider.fill("1");
  await expect(page.locator(".plot-studio__frame-meta")).toContainText("selected 2 / 3");
  await page.evaluate(() => window.__nimbusStudioHarness?.publishLiveThrough(5));
  await expect(page.locator(".plot-studio__meta")).toContainText("5 steps", { timeout: 6_000 });
  await expect(page.locator(".plot-studio__frame-meta")).toContainText("selected 2 / 5");
  await expect(page.locator(".plot-studio__meta")).toContainText("showing t = 30.0 s");

  await slider.fill("4");
  await expect(page.locator(".plot-studio__frame-meta")).toContainText("selected 5 / 5");
  await page.evaluate(() => window.__nimbusStudioHarness?.publishLiveThrough(6));
  await expect(page.locator(".plot-studio__meta")).toContainText("6 steps", { timeout: 6_000 });
  await expect(page.locator(".plot-studio__frame-meta")).toContainText("selected 6 / 6");
  await expect(page.locator(".plot-studio__meta")).toContainText("showing t = 150.0 s");

  await page.evaluate(() => {
    window.__nimbusStudioHarness?.setListDelay(
      "/__nimbus_browser_fixture__/slow",
      700,
    );
    window.__nimbusStudioHarness?.setRun(
      "/__nimbus_browser_fixture__/slow",
      "completed",
    );
  });
  await expect(page.locator(".plot-studio__title-row h2")).toHaveText(
    "/__nimbus_browser_fixture__/slow",
  );
  await page.waitForTimeout(50);
  await page.evaluate(() => window.__nimbusStudioHarness?.setRun(
    "/__nimbus_browser_fixture__/fast",
    "completed",
  ));
  await expect(page.locator(".plot-studio__title-row h2")).toHaveText(
    "/__nimbus_browser_fixture__/fast",
  );
  await expect(page.locator(".plot-studio__meta")).toContainText("4 steps");
  await page.waitForTimeout(800);
  await expect(page.locator(".plot-studio__title-row h2")).toHaveText(
    "/__nimbus_browser_fixture__/fast",
  );
  await expect(page.locator(".plot-studio__meta")).toContainText("4 steps");

  await page.evaluate(() => window.__nimbusStudioHarness?.setRun(
    "/__nimbus_browser_fixture__/legacy",
    "completed",
  ));
  await expect(page.locator(".plot-studio__title-row h2")).toHaveText(
    "/__nimbus_browser_fixture__/legacy",
  );
  // An unmarked run now opens like any other: its frames load and nothing
  // distinguishes it in the chrome.
  await expect(page.locator(".plot-studio__meta")).toContainText("2 steps");
  await expect(page.locator(".plot-studio__in-progress")).toHaveCount(0);

  const fixture = await page.evaluate(() => window.__nimbusStudioHarness?.snapshot());
  expect(fixture?.listCalls["/__nimbus_browser_fixture__/live"]).toBeGreaterThanOrEqual(3);
  expect(fixture?.listCalls["/__nimbus_browser_fixture__/slow"]).toBe(1);
  expect(fixture?.listCalls["/__nimbus_browser_fixture__/fast"]).toBe(1);
  expect(fixture?.ipc.failures).toBe(0);
  expect(pageErrors, "uncaught mounted-Studio errors").toEqual([]);
});

test("mounted Plot Studio restores recorder focus to the Rec opener", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));

  const record = page.getByRole("button", { name: "Rec", exact: true });
  const play = page.getByRole("button", { name: "Play frames", exact: true });
  await expect(record).toBeEnabled({ timeout: 30_000 });
  await record.click();

  const dialog = page.getByRole("dialog", { name: "Record animation" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("radio", { name: "GIF" }).press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(record).toBeFocused();
  await expect(play).not.toBeFocused();
});

test("mounted recorder captures exact frames and writes a complete GIF", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=recorder");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await expect(page.getByRole("dialog", { name: "Record animation" })).toBeVisible();

  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page.locator(".movie-recorder__done")).toContainText(
    "Saved to /__nimbus_browser_fixture__/recording.gif",
  );

  const result = await page.evaluate(() => window.__nimbusRecorderHarness?.snapshot());
  expect(result?.capturedSteps).toEqual([0, 1, 2]);
  expect(result?.savedGif.writes).toBe(1);
  expect(result?.savedGif.path).toBe("/__nimbus_browser_fixture__/recording.gif");
  expect(result?.savedGif.signature).toBe("GIF89a");
  expect(result?.savedGif.byteLength ?? 0).toBeGreaterThan(1_000);
  expect(result?.ipc.failures).toBe(0);
  expect(pageErrors, "uncaught mounted-recorder errors").toEqual([]);
});

test("mounted recorder exports only the current saved frame as native-size PNG", async ({ page }) => {
  test.setTimeout(45_000);
  await page.goto("/harness/index.html?mode=recorder&recorderStep=2&recorderRetina=1");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await page.getByRole("radio", { name: "PNG", exact: true }).click();
  await expect(page.locator(".movie-recorder__hint")).toContainText("Primary plot only · 800 × 480 output pixels");
  await expect(page.getByLabel("Start step")).toHaveCount(0);
  await page.getByRole("button", { name: "Save current frame", exact: true }).click();
  await expect(page.locator(".movie-recorder__done")).toContainText("nimbus-recorder-fixture.png");
  const result = await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot());
  expect(result.capturedSteps).toEqual([2]);
  const png = Buffer.from(result.savedPng);
  expect(Array.from(png.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  expect(png.readUInt32BE(16)).toBe(800);
  expect(png.readUInt32BE(20)).toBe(480);
  expect(result.savedGif.writes).toBe(1);
  expect(result.ipc.failures).toBe(0);
});

test("mounted recorder rejects identity changes during asynchronous capture", async ({ page }) => {
  test.setTimeout(45_000);
  await page.goto("/harness/index.html?mode=recorder&recorderCapture=held");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusRecorderHarness!.snapshot().capturePending);
  await page.evaluate(() => window.__nimbusRecorderHarness!.releaseCapture(true));
  await expect(page.locator(".movie-recorder__error")).toContainText("displayed frame changed during capture");
  const result = await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot());
  expect(result.capturedSteps).toEqual([0]);
  expect(result.savedGif.writes).toBe(0);
  expect(result.savedPng).toEqual([]);
  expect(result.ipc.failures).toBe(0);
});

test("mounted recorder cancellation discards a pending capture without writing", async ({ page }) => {
  test.setTimeout(45_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=recorder&recorderCapture=held");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusRecorderHarness!.snapshot().capturePending);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Record animation" })).toHaveCount(0);
  await page.evaluate(() => window.__nimbusRecorderHarness!.releaseCapture());
  await page.getByRole("button", { name: "Open recorder" }).click();
  await expect(page.getByRole("button", { name: "Start recording", exact: true })).toBeEnabled();
  const result = await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot());
  expect(result.capturedSteps).toEqual([0]);
  expect(result.savedGif.writes).toBe(0);
  expect(result.ipc.failures).toBe(0);
  expect(errors).toEqual([]);
});

for (const resize of ["viewport", "backing"] as const) {
  test(`mounted recorder rejects ${resize} changes during asynchronous capture`, async ({ page }) => {
    test.setTimeout(45_000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("/harness/index.html?mode=recorder&recorderCapture=held&recorderRetina=1");
    await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
    await page.getByRole("radio", { name: "PNG", exact: true }).click();
    await page.getByRole("button", { name: "Save current frame", exact: true }).click();
    await page.waitForFunction(() => window.__nimbusRecorderHarness!.snapshot().capturePending);
    await page.locator(".recorder-fixture__canvas").evaluate((element, mode) => {
      const canvas = element as HTMLCanvasElement;
      if (mode === "viewport") canvas.style.width = "320px";
      else canvas.width += 2;
    }, resize);
    await page.evaluate(() => window.__nimbusRecorderHarness!.releaseCapture());
    await expect(page.locator(".movie-recorder__error")).toContainText("plot viewport changed during capture");
    const result = await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot());
    expect(result.capturedSteps).toEqual([0]);
    expect(result.savedGif.writes).toBe(0);
    expect(result.savedPng).toEqual([]);
    expect(result.nativeEncoding.activeSessions).toBe(0);
    expect(result.ipc.failures).toBe(0);
    expect(errors).toEqual([]);
  });
}

test("native MP4 staging streams every exact saved frame with the requested fixed cadence", async ({ page }) => {
  test.setTimeout(45_000);
  await page.goto("/harness/index.html?mode=recorder");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await page.getByRole("radio", { name: "MP4", exact: true }).click();
  await page.getByRole("combobox", { name: /^FPS/ }).selectOption("30");
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await expect(page.locator(".movie-recorder__done")).toContainText("nimbus-recorder-fixture.mp4");
  const result = await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot());
  expect(result.capturedSteps).toEqual([0, 1, 2]);
  expect(result.nativeEncoding.prepared).toBe(1);
  expect(result.nativeEncoding.appended).toEqual([0, 1, 2].map(index => ({ index, width: 400, height: 240 })));
  expect(result.nativeEncoding.encoded).toEqual([{ fps: 30, frames: 3, outputPath: "/__nimbus_browser_fixture__/plugins/nimbus-recorder-fixture.mp4" }]);
  expect(result.nativeEncoding.activeSessions).toBe(0);
  expect(result.ipc.failures).toBe(0);
});

test("native MP4 cancellation cleans a session created after the dialog closed", async ({ page }) => {
  test.setTimeout(45_000);
  await page.goto("/harness/index.html?mode=recorder&recorderPreparation=held");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await page.getByRole("radio", { name: "MP4", exact: true }).click();
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusRecorderHarness!.snapshot().nativeEncoding.preparationPending);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Record animation" })).toHaveCount(0);
  await page.evaluate(() => window.__nimbusRecorderHarness!.releasePreparation());
  await expect.poll(async () => page.evaluate(() => window.__nimbusRecorderHarness!.snapshot().nativeEncoding.cleaned)).toBe(1);
  const result = await page.evaluate(() => window.__nimbusRecorderHarness!.snapshot());
  expect(result.capturedSteps).toEqual([]);
  expect(result.nativeEncoding.appended).toEqual([]);
  expect(result.nativeEncoding.encoded).toEqual([]);
  expect(result.nativeEncoding.activeSessions).toBe(0);
  expect(result.ipc.failures).toBe(0);
});

test("recorder modal traps focus, uses roving radios, and restores its trigger", async ({ page }) => {
  await page.goto("/harness/index.html?mode=recorder&recorderClosed=1");
  const trigger = page.getByRole("button", { name: "Open recorder" });
  await trigger.focus();
  await trigger.click();

  const dialog = page.getByRole("dialog", { name: "Record animation" });
  await expect(dialog).toBeVisible();
  const gif = dialog.getByRole("radio", { name: "GIF" });
  const mp4 = dialog.getByRole("radio", { name: "MP4" });
  await expect(gif).toBeFocused();
  await expect(gif).toHaveAttribute("tabindex", "0");
  await expect(mp4).toHaveAttribute("tabindex", "-1");

  await gif.press("ArrowRight");
  await expect(mp4).toBeFocused();
  await expect(mp4).toBeChecked();
  await expect(mp4).toHaveAttribute("tabindex", "0");
  await expect(gif).toHaveAttribute("tabindex", "-1");

  await mp4.press("ArrowLeft");
  await expect(gif).toBeFocused();
  const start = dialog.getByRole("button", { name: "Start recording" });
  await gif.press("Shift+Tab");
  await expect(start).toBeFocused();
  await start.press("Tab");
  await expect(gif).toBeFocused();

  await gif.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();

  // The explicit Safari pointer fallback must not disturb the ordinary
  // keyboard-invoker path.
  await trigger.press("Enter");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("radio", { name: "GIF" })).toBeFocused();
  await dialog.getByRole("radio", { name: "GIF" }).press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
});

test("mounted recorder can sample a long source sequence without weakening frame readiness", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=recorder");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));

  await page.getByLabel("Source frames").selectOption("2");
  await expect(page.locator(".movie-recorder__hint")).toContainText("2 exported frames");
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page.locator(".movie-recorder__done")).toContainText(
    "Saved to /__nimbus_browser_fixture__/recording.gif",
  );

  const result = await page.evaluate(() => window.__nimbusRecorderHarness?.snapshot());
  expect(result?.capturedSteps).toEqual([0, 2]);
  expect(result?.savedGif.signature).toBe("GIF89a");
  expect(result?.ipc.failures).toBe(0);
  expect(pageErrors, "uncaught strided-recorder errors").toEqual([]);
});

test("mounted recorder visibly rejects a stale frame without writing", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/index.html?mode=recorder&recorderFailure=stale");
  await page.waitForFunction(() => Boolean(window.__nimbusRecorderHarness));
  await page.getByRole("button", { name: "Start recording" }).click();

  const progress = page.getByRole("progressbar", { name: "Capturing frames…" });
  await expect(progress).toHaveAttribute("aria-valuemin", "0");
  await expect(progress).toHaveAttribute("aria-valuemax", "100");
  await expect(progress).toHaveAttribute("aria-valuenow", /\d+/);
  const progressStyle = await progress.locator(".movie-recorder__progress-fill").evaluate((element) => ({
    transform: (element as HTMLElement).style.transform,
    width: (element as HTMLElement).style.width,
  }));
  expect(progressStyle.transform).toMatch(/^scaleX\(/);
  expect(progressStyle.width).toBe("");

  await expect(page.locator(".movie-recorder__error")).toContainText(
    "recording stopped rather than capture a stale or partial frame",
  );
  const result = await page.evaluate(() => window.__nimbusRecorderHarness?.snapshot());
  expect(result?.capturedSteps).toEqual([0]);
  expect(result?.savedGif.writes).toBe(0);
  expect(result?.savedGif.byteLength).toBe(0);
  expect(result?.ipc.failures).toBe(0);
  expect(pageErrors, "uncaught stale-recorder errors").toEqual([]);
});
