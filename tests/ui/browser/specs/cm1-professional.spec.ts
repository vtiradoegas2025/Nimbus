import { expect, test, type Page } from "@playwright/test";

async function openCm1Workbook(page: Page) {
  const report = page.locator(".cm1-compatibility");
  await report.locator(":scope > summary").click();
  const workbook = page.locator(".cm1-workbook");
  await workbook.locator(":scope > summary").click();
  return { report, workbook };
}

async function expectNoHorizontalOverflow(page: Page, includeHarness = true) {
  const overflow = await page.evaluate(() => {
    const harness = document.querySelector<HTMLElement>(".cm1-qualification-harness");
    return {
      document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      harness: harness ? harness.scrollWidth - harness.clientWidth : 0,
    };
  });
  expect(overflow.document).toBeLessThanOrEqual(1);
  if (includeHarness) expect(overflow.harness).toBeLessThanOrEqual(1);
}

async function stageCompletedHandoff(page: Page) {
  await page.addInitScript(() => {
    const now = "2026-07-20T00:00:00.000Z";
    const longOutput = `/runs/${"qualified_output_segment_".repeat(24)}/run-1`;
    const yaml = "schema: nimbus.v1\ntime:\n  run_time: 30\n  output_interval: 10\n";
    let hash = 0xcbf29ce484222325n;
    for (const byte of new TextEncoder().encode(yaml)) { hash ^= BigInt(byte); hash = BigInt.asUintN(64, hash * 0x100000001b3n); }
    localStorage.setItem("nimbus:cm1-guided-handoff:v2", JSON.stringify({
      schema: "nimbus.studio.cm1-guided-handoff",
      schemaVersion: 2,
      id: "cm1-handoff-professional",
      updatedAt: now,
      cm1Source: {
        schema: "nimbus.studio.dataset-source",
        schemaVersion: 1,
        kind: "cm1_netcdf",
        sourceModel: "cm1",
        nativeFormat: "netcdf",
        locator: { kind: "local_paths", path: "/case/cm1out_1.nc", paths: ["/case/cm1out_1.nc", "/case/cm1out_2.nc"] },
        displayName: "2 netCDF history files",
        readOnly: true,
      },
      evidence: {
        historyFileCount: 2,
        soundingSourcePath: "/case/input_sounding",
        projectedLevelCount: 72,
        mappingCounts: { ready: 9, needsDecision: 3, different: 2, unsupported: 4 },
        mappingBlockers: ["3 mapping decisions remain"],
        mappingBlockerTotal: 13,
        mappingBlockersTruncated: true,
        reviewedAt: now,
      },
      smoke: {
        soundingPath: "/work/cm1-sounding.txt",
        configPath: "/work/cm1-smoke.yaml",
        yaml,
        configurationDigest: `fnv1a64:${hash.toString(16).padStart(16, "0")}`,
        durationS: 30,
        outputIntervalS: 10,
        catalogFingerprint: "browser-mock-catalog",
        executablePath: "/bin/nimbus_sim",
        initializationExecutablePath: "/bin/nimbus_sim",
        stagedAt: now,
        consumedAt: now,
      },
      run: { status: "complete", runId: "run-1", outputDir: longOutput, pendingReason: null },
      comparisonOpenedAt: null,
    }));
  });
}

test("CM1 report and workbook remain keyboard-operable at compact width", async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 900 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/harness/cm1-report.html");

  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveClass(/cm1-compatibility__summary/);
  const focusStyle = await page.locator(":focus").evaluate((element) => {
    const style = getComputedStyle(element);
    return { style: style.outlineStyle, width: style.outlineWidth };
  });
  expect(focusStyle).toEqual({ style: "solid", width: "2px" });
  await page.keyboard.press("Enter");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveClass(/cm1-workbook__summary/);
  await page.keyboard.press("Enter");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveText("Export review JSON");

  const workbook = page.locator(".cm1-workbook");
  await expect(workbook.getByLabel("Review decision").first()).toBeVisible();
  await expect(workbook.getByText("Review record only")).toBeVisible();
  const transitionSeconds = await workbook.getByRole("button", { name: "Export review JSON" }).evaluate((element) =>
    getComputedStyle(element).transitionDuration.split(",").map((value) => Number.parseFloat(value)),
  );
  expect(Math.max(...transitionSeconds)).toBeLessThanOrEqual(0.001);
  await expectNoHorizontalOverflow(page);
});

test("native CM1 sounding stages the original file while preserving optional projection", async ({ page }) => {
  await page.goto("/harness/cm1-report.html");
  const report = page.locator(".cm1-compatibility");
  await report.locator(":scope > summary").click();

  const native = report.locator(".cm1-native-sounding");
  await expect(native.getByText(/original levels · no conversion/)).toBeVisible();
  await native.locator(":scope > summary").click();
  await expect(native.getByText("Direct θ, qv, u, and v ingestion · Nimbus EOS reconstruction")).toBeVisible();
  await expect(native.getByText(/CM1 isnd=7 selects input_sounding thermodynamics and winds/)).toBeVisible();
  await native.getByRole("checkbox").check();
  await native.getByRole("button", { name: "Stage native 30 s smoke test" }).click();
  await expect(native.getByText(/Handoff staged\. Open Configure/)).toBeVisible();

  const staged = await page.evaluate(() => JSON.parse(
    localStorage.getItem("nimbus:cm1-guided-handoff:v2") ?? "null",
  ));
  expect(staged.evidence.soundingSourceMode).toBe("native_cm1");
  expect(staged.smoke.soundingPath).toMatch(/\/input_sounding$/);
  expect(staged.smoke.yaml).toContain("NATIVE SOURCE: theta, qv, u, and v are read directly");
  expect(staged.smoke.yaml).toMatch(/hodograph:\s+type: auto/);

  await expect(report.locator(".cm1-sounding-projection").getByText("Optional compatibility projection")).toBeVisible();
});

test("a CM1 source change retires pending handoff work and requires a fresh review", async ({ page }) => {
  await page.goto("/harness/cm1-report.html");
  await page.evaluate(() => {
    const internal = (window as unknown as { __TAURI_INTERNALS__: { invoke: (command: string, args?: unknown) => Promise<unknown> } }).__TAURI_INTERNALS__;
    const invoke = internal.invoke;
    const state = { held: 0, finished: false, laterActions: 0, release: () => {} };
    Object.assign(window, { cm1PendingHandoff: state });
    internal.invoke = async (command, args) => {
      if (["validate_cm1_sounding_project", "plugin:dialog|save", "write_config"].includes(command)) state.laterActions++;
      if (command === "preflight_config") {
        state.held++;
        await new Promise<void>(resolve => { state.release = resolve; });
        const result = await invoke(command, args);
        state.finished = true;
        return result;
      }
      return invoke(command, args);
    };
  });
  const report = page.locator(".cm1-compatibility");
  await report.locator(":scope > summary").click();
  const native = report.locator(".cm1-native-sounding");
  await native.locator(":scope > summary").click();
  await native.getByRole("checkbox").check();
  await native.getByRole("button", { name: "Stage native 30 s smoke test" }).click();
  await page.waitForFunction(() => (window as unknown as { cm1PendingHandoff: { held: number } }).cm1PendingHandoff.held === 1);
  await page.evaluate(() => (window as unknown as { __nimbusCm1ReportHarness: { switchSource: () => void } }).__nimbusCm1ReportHarness.switchSource());
  await expect(native.getByRole("checkbox")).not.toBeChecked();
  await expect(native.getByRole("button", { name: "Stage native 30 s smoke test" })).toBeDisabled();
  await page.evaluate(() => (window as unknown as { cm1PendingHandoff: { release: () => void } }).cm1PendingHandoff.release());
  await page.waitForFunction(() => (window as unknown as { cm1PendingHandoff: { finished: boolean } }).cm1PendingHandoff.finished);
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())));
  expect(await page.evaluate(() => (window as unknown as { cm1PendingHandoff: { laterActions: number } }).cm1PendingHandoff.laterActions)).toBe(0);
  expect(await page.evaluate(() => localStorage.getItem("nimbus:cm1-guided-handoff:v2"))).toBeNull();
  await expect(native.getByText(/Handoff staged/)).toHaveCount(0);
});

test("CM1 review reflows at 200% effective zoom and preserves long evidence", async ({ page }) => {
  await page.setViewportSize({ width: 720, height: 900 });
  await page.goto("/harness/cm1-report.html");
  await page.evaluate(() => { document.documentElement.style.zoom = "2"; });
  const { workbook } = await openCm1Workbook(page);

  const firstDecision = workbook.getByLabel("Review decision").first();
  await firstDecision.selectOption("keep_nimbus_default");
  const note = workbook.getByLabel(/Review note/).first();
  await note.fill("reviewed_note_segment_".repeat(42));
  await expect(note).toHaveValue(/reviewed_note_segment_/);
  await expect(workbook.getByText("Exact default recorded").first()).toBeVisible();
  await expect(workbook.getByText("grid.ztop = 16000").first()).toBeVisible();
  const longEvidenceFits = await workbook.locator(".cm1-workbook__header strong").evaluate((element) =>
    element.scrollWidth <= element.clientWidth + 1,
  );
  expect(longEvidenceFits).toBe(true);
  await workbook.getByRole("button", { name: "Record decision" }).first().click();
  await expect(workbook.getByRole("button", { name: "Update decision" }).first()).toBeVisible();
  await expectNoHorizontalOverflow(page, false);
});

test("persisted complete handoff fails closed without durable evidence at compact 200% zoom", async ({ page }) => {
  await stageCompletedHandoff(page);
  await page.setViewportSize({ width: 720, height: 900 });
  await page.goto("/harness/config.html");
  await page.evaluate(() => { document.documentElement.style.zoom = "2"; });
  const guide = page.locator(".config-editor__cm1-handoff");
  await expect(guide.getByText("Reviewed sounding · short Nimbus smoke test")).toBeVisible();
  await expect(guide).toContainText("Durable Run Library scan has not found the matching output directory yet.");
  await expect(guide).not.toContainText("Durable output verified");
  const guideFits = await guide.evaluate((element) => element.scrollWidth <= element.clientWidth + 1);
  expect(guideFits).toBe(true);
  const restore = guide.getByRole("button", { name: "Restore staged baseline" });
  await restore.focus();
  await expect(restore).toBeFocused();
  const focus = await restore.evaluate((element) => getComputedStyle(element).outlineWidth);
  expect(focus).toBe("2px");
  const documentOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(documentOverflow).toBeLessThanOrEqual(1);
});

test("post-restart handoff stays pending without durable qualified frames", async ({ page }) => {
  await stageCompletedHandoff(page);
  await page.addInitScript(() => {
    const key = "nimbus:cm1-guided-handoff:v2";
    const record = JSON.parse(localStorage.getItem(key)!);
    record.run = {
      status: "pending",
      runId: "run-1",
      outputDir: record.run.outputDir,
      pendingReason: "Corrupt output manifests; durable qualification remains blocked.",
    };
    localStorage.setItem(key, JSON.stringify(record));
  });
  await page.goto("/harness/config.html");
  const guide = page.locator(".config-editor__cm1-handoff");
  await expect(guide).toContainText("Durable Run Library scan has not found the matching output directory yet.");
  await expect(guide).not.toContainText("Durable output verified");
});

test("CM1 workbook exposes coarse-pointer hit areas", async ({ browser, baseURL }) => {
  const context = await browser.newContext({
    baseURL,
    viewport: { width: 480, height: 900 },
    hasTouch: true,
    isMobile: true,
  });
  const page = await context.newPage();
  await page.goto("/harness/cm1-report.html");
  const { workbook } = await openCm1Workbook(page);
  const heights = await Promise.all([
    workbook.getByRole("button", { name: "Export review JSON" }).evaluate((element) => element.getBoundingClientRect().height),
    workbook.getByLabel("Review decision").first().evaluate((element) => element.getBoundingClientRect().height),
    workbook.locator(":scope > summary").evaluate((element) => element.getBoundingClientRect().height),
  ]);
  heights.forEach((height) => expect(height).toBeGreaterThanOrEqual(44));
  await expectNoHorizontalOverflow(page);
  await context.close();
});
