import { expect, test } from "@playwright/test";

/*
 * The archive list collapses to a readable head, but QueuePanel is the only
 * run browser in the app — a row that is neither rendered nor disclosed is
 * unreachable. These tests pin the two halves of that contract: the hidden
 * count is stated, and expanding actually reveals every run.
 *
 * Regression: the list rendered `completed.slice(0, 12)` with no expander,
 * beneath a counter reading "21 of 21 runs". Eight archived runs could only be
 * reached by guessing a search term.
 */

const COLLAPSED_COUNT = 12;
const ARCHIVED_COUNT = 20;

test("pending run actions and late results remain bound to the initiating run", async ({ page }) => {
  test.setTimeout(45_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/queue-panel.html?mode=run-actions");
  await page.getByRole("button", { name: "Abort", exact: true }).click();
  await expect(page.getByRole("button", { name: "Aborting…", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Select Action run 1", exact: true }).click();
  await expect(page.getByRole("button", { name: "Abort", exact: true })).toBeEnabled();
  await page.evaluate(() => window.__nimbusWorkflowFixture!.rejectAbort("Run 0 stop failed"));
  await expect(page.locator(".run-detail")).not.toContainText("Run 0 stop failed");
  await page.getByRole("button", { name: "Select Action run 0", exact: true }).click();
  await page.getByText("Run evidence", { exact: true }).click();
  await page.getByRole("button", { name: "Export evidence bundle", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().dialogPending);
  await page.getByRole("button", { name: "Select Action run 1", exact: true }).click();
  await page.getByText("Run evidence", { exact: true }).click();
  await expect(page.getByRole("button", { name: "Export evidence bundle", exact: true })).toBeEnabled();
  await page.evaluate(() => window.__nimbusWorkflowFixture!.resolveSaveDialog("/__workflow_fixture__/run-0-evidence.json"));
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().calls.some(value => value.command === "copy_run_provenance"));
  await expect(page.locator(".run-detail")).not.toContainText("Copy saved");
  const snapshot = await page.evaluate(() => window.__nimbusWorkflowFixture!.snapshot());
  expect(snapshot.calls.filter(value => value.command === "abort_sim").map(value => value.args.runId)).toEqual(["action-0"]);
  expect(snapshot.calls.find(value => value.command === "copy_run_provenance")?.args.runDir).toBe("/__workflow_fixture__/run-0");
  expect(errors).toEqual([]);
});

test("remote receipt recovery cancels only transfer and retries saved results without rerunning compute", async ({ page }) => {
  test.setTimeout(45_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/queue-panel.html?mode=run-actions&remote=1");
  await expect(page.getByText("Prior receive transport failed", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Retry receiving results", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().receivePending);
  await expect(page.getByRole("button", { name: "Abort", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Cancel transfer", exact: true }).click();
  await expect(page.getByText("Receiving cancelled; remote originals retained", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Retry receiving results", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().receivePending);
  await page.evaluate(() => window.__nimbusWorkflowFixture!.settleReceive("failed", "Current receive checksum mismatch"));
  await expect(page.getByText("Current receive checksum mismatch", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Retry receiving results", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().receivePending);
  await page.evaluate(() => window.__nimbusWorkflowFixture!.settleReceive("complete"));
  await expect(page.getByText("Remote results received", { exact: true })).toBeVisible();
  await expect(page.getByText(/Original computation evidence remains unchanged/)).toBeVisible();
  const calls = await page.evaluate(() => window.__nimbusWorkflowFixture!.snapshot().calls);
  expect(calls.filter(value => value.command === "receive_remote_results").map(value => value.args.runDir)).toEqual(Array(3).fill("/__workflow_fixture__/run-0"));
  expect(calls.filter(value => value.command === "cancel_remote_receive")).toHaveLength(1);
  expect(calls.filter(value => value.command === "abort_sim" || value.command === "spawn_sim")).toHaveLength(0);
  expect(errors).toEqual([]);
});

test("archive truncation discloses the hidden runs and can be expanded", async ({ page }) => {
  await page.goto("/harness/queue-panel.html");

  const rows = page.locator(".queue-list .run-item");
  // One live run above the archive section.
  await expect(rows).toHaveCount(COLLAPSED_COUNT + 1);

  const expander = page.getByRole("button", { name: `Show ${ARCHIVED_COUNT - COLLAPSED_COUNT} more` });
  await expect(expander).toBeVisible();

  await expander.click();
  await expect(rows).toHaveCount(ARCHIVED_COUNT + 1);
  // The last archived run is only reachable once expanded.
  await expect(page.getByText("cart_cgrid_archive_019")).toBeVisible();

  const collapse = page.getByRole("button", { name: `Show fewer (${COLLAPSED_COUNT} of ${ARCHIVED_COUNT})` });
  await expect(collapse).toBeVisible();
  await collapse.click();
  await expect(rows).toHaveCount(COLLAPSED_COUNT + 1);
});

test("the run counter agrees with what the panel can show", async ({ page }) => {
  await page.goto("/harness/queue-panel.html");

  // The counter describes the filter, and the expander accounts for the rest.
  await expect(page.getByText(`${ARCHIVED_COUNT + 1} of ${ARCHIVED_COUNT + 1} runs`)).toBeVisible();

  const rendered = await page.locator(".queue-list .run-item").count();
  const expanderText = await page.locator(".queue-panel__more").innerText();
  const disclosed = Number(expanderText.match(/\d+/)?.[0] ?? 0);
  expect(rendered + disclosed).toBe(ARCHIVED_COUNT + 1);
});

test("a search that narrows below the collapse point drops the expander", async ({ page }) => {
  await page.goto("/harness/queue-panel.html");
  await page.getByLabel("Find a run").fill("archive_01");

  // archive_010..archive_019 — ten rows, under the collapse threshold.
  await expect(page.locator(".queue-list .run-item")).toHaveCount(10);
  await expect(page.locator(".queue-panel__more")).toHaveCount(0);
});

test("archive selection and deletion remain separate keyboard controls", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?deletable=1");

  const row = page.locator(".run-item", { hasText: "cart_cgrid_archive_000" });
  const open = row.getByRole("button", { name: "Open cart_cgrid_archive_000 in Run Archive" });
  const beginDelete = row.getByRole("button", { name: "Remove cart_cgrid_archive_000 from the session" });

  await beginDelete.focus();
  await beginDelete.press("Enter");
  await expect(row).not.toHaveClass(/run-item--selected/);

  const confirmDelete = row.getByRole("button", { name: "Remove", exact: true });
  await expect(confirmDelete).toBeFocused();
  await row.getByRole("button", { name: "Keep", exact: true }).press("Enter");
  await expect(open).toBeFocused();
});

test("running progress is semantic and uses a compositor transform", async ({ page }) => {
  await page.goto("/harness/queue-panel.html");
  const progress = page.getByRole("progressbar", {
    name: "cart_cgrid_live simulation progress",
  });
  await expect(progress).toHaveAttribute("aria-valuemin", "0");
  await expect(progress).toHaveAttribute("aria-valuemax", "100");
  await expect(progress).toHaveAttribute("aria-valuenow", "10");
  const inlineStyle = await progress.locator(".run-item__progress-fill").evaluate((element) => ({
    transform: (element as HTMLElement).style.transform,
    width: (element as HTMLElement).style.width,
  }));
  expect(inlineStyle).toEqual({ transform: "scaleX(0.1)", width: "" });
});

test("archive delete and resize controls stay touch-sized on coarse pointers", async ({ browser, baseURL }) => {
  const context = await browser.newContext({
    hasTouch: true,
    viewport: { width: 520, height: 800 },
  });
  const page = await context.newPage();
  try {
    await page.goto(`${baseURL}/harness/queue-panel.html?deletable=1`);
    expect(await page.evaluate(() => matchMedia("(pointer: coarse)").matches)).toBe(true);
    const row = page.locator(".run-item", { hasText: "cart_cgrid_archive_000" });
    const remove = row.getByRole("button", {
      name: "Remove cart_cgrid_archive_000 from the session",
    });
    await expect(remove).toHaveCSS("opacity", "1");
    const removeBox = await remove.boundingBox();
    expect(removeBox?.width).toBeGreaterThanOrEqual(44);
    expect(removeBox?.height).toBeGreaterThanOrEqual(44);

    await remove.click();
    for (const name of ["Remove", "Keep"]) {
      const box = await row.getByRole("button", { name, exact: true }).boundingBox();
      expect(box?.height).toBeGreaterThanOrEqual(44);
    }

    await page.goto(`${baseURL}/harness/run-library.html`);
    await page.getByRole("button", { name: "Runs", exact: true }).click();
    const separatorBox = await page.getByRole("separator", {
      name: "Resize queue panel",
    }).boundingBox();
    expect(separatorBox?.width).toBeGreaterThanOrEqual(44);
  } finally {
    await context.close();
  }
});

/*
 * Failure-state distinctness. Three unrelated failures used to reach one
 * hardcoded banner that discarded the underlying message, and one of them
 * ("live") was told "Live runs remain available" -- the opposite of what had
 * just happened. Separately, "still scanning" and "there are no runs" rendered
 * the same definitive sentence, and a failed simulator probe was byte-identical
 * to "no simulator configured".
 */

const FIXTURE_ERROR_MESSAGE = "cannot delete a running run; abort it first";

test("each run-error scope names its own remedy and shows the real message", async ({ page }) => {
  const banner = page.locator(".queue-panel__library-error");

  await page.goto("/harness/queue-panel.html?error=library");
  await expect(banner).toContainText("The run library could not be rebuilt");
  await expect(banner).toContainText("Check the Runs directory in Settings");

  await page.goto("/harness/queue-panel.html?error=live");
  await expect(banner).toContainText("Live run status is unavailable");
  // The precise regression: this scope must NOT claim live runs are fine.
  await expect(banner).not.toContainText("Live runs remain available");
  await expect(banner).not.toContainText("Runs directory in Settings");

  await page.goto("/harness/queue-panel.html?error=action");
  await expect(banner).toContainText("That action was refused");

  // Every scope surfaces the backend's own text rather than swallowing it.
  for (const scope of ["library", "live", "action"]) {
    await page.goto(`/harness/queue-panel.html?error=${scope}`);
    await expect(page.locator(".queue-panel__error-detail")).toHaveText(FIXTURE_ERROR_MESSAGE);
  }
});

test("scanning and empty are different sentences", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?empty=1&loading=1");
  await expect(page.locator(".queue-panel__empty")).toHaveText("Scanning the run library…");

  await page.goto("/harness/queue-panel.html?empty=1");
  await expect(page.locator(".queue-panel__empty")).toContainText("No local runs found");
});

test("a failed simulator probe is distinguishable from no simulator configured", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?simError=1");
  const status = page.locator(".sim-status");
  await expect(status).toContainText("Simulator status unknown");
  await expect(status).toContainText("fixture transport failure");
  // Not the "go connect one" advice, which would send the user to re-enter a
  // path that may already be correct.
  await expect(status).not.toContainText("Connect a simulator in Settings to run");
  await expect(page.locator(".sim-status__dot--warn")).toHaveCount(1);

  // The unconfigured case keeps its original muted treatment and advice.
  await page.goto("/harness/queue-panel.html");
  await expect(page.locator(".sim-status")).toContainText("Simulation ready");
  await expect(page.locator(".sim-status__dot--warn")).toHaveCount(0);
});
