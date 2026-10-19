import { expect, test } from "@playwright/test";

/*
 * The native-read band reports which filesystem read paths the desktop runtime
 * supports. A null capability record used to collapse three states into one:
 * browser mode, probe in flight, and probe threw. All three rendered the
 * browser-mode copy, so a packaged desktop user whose IPC failed was told they
 * were "in browser mode" and could not issue filesystem reads -- on a panel
 * whose entire purpose is reporting boundaries accurately.
 *
 * The desktop fixture has no handler for `native_field_read_capabilities`, so
 * it rejects, which is precisely the probe-failure state.
 */

const NATIVE_BAND = ".resource-budget__native";

test("browser mode says browser mode", async ({ page }) => {
  await page.goto("/harness/resource-budget.html");

  const band = page.locator(NATIVE_BAND);
  await expect(band).toContainText("Browser mode retains the cache and scheduler ledger");
  await expect(band.locator("strong")).toHaveText("Desktop only");
  await expect(page.locator(".resource-budget__boundary--error")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "VTK ownership" })).toBeVisible();
  await expect(page.locator(".resource-budget__measure--visualization")).toContainText("0 B");
  await expect(page.getByText("Underpredictions", { exact: true })).toBeVisible();
  await expect(page.getByText("Prepared handles", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("0 / 0 refs");
  await expect(page.getByText("Handle residency", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("0 main / 0 worker");
  await expect(page.getByText("Warm extraction", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("0 / 3");
  await expect(page.getByText("Admission", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("Observe only");
  await expect(page.getByText("Evidence floor", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("403 MiB");
  await expect(page.getByText("Host RSS", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("Unavailable");
});

test("a failed desktop probe is not reported as browser mode", async ({ page }) => {
  await page.goto("/harness/resource-budget.html?desktop=1");

  const band = page.locator(NATIVE_BAND);
  const failure = page.locator(".resource-budget__boundary--error");
  await expect(failure).toHaveCount(1);

  // The precise regression: the desktop user must not be told they are in a
  // browser, nor that filesystem reads are impossible.
  await expect(band).not.toContainText("Browser mode retains the cache and scheduler ledger");
  await expect(band).not.toContainText("cannot issue filesystem reads");

  // It must say what is actually unknown, and carry the underlying message.
  await expect(failure).toContainText("did not report its native read capabilities");
  await expect(failure).toContainText("reads may still work");
  await expect(failure).toContainText("native_field_read_capabilities");
  await expect(band.locator("strong")).toHaveText("Unavailable");

  // The contract table is genuinely unknown here, so it must not be rendered.
  await expect(band.locator(".resource-budget__native-grid")).toHaveCount(0);
});

/*
 * The third state — a desktop probe that SUCCEEDS — was untested, so nothing
 * checked the table this panel exists to draw. That matters because cancellation
 * is not uniform across formats, and the sibling prose has already drifted once
 * by collapsing the netCDF variants into a single "non-interruptible" claim.
 * Reporting a cooperative path as non-interruptible tells a user a long read
 * cannot be interrupted when it can, so each row is pinned to its own mode.
 */
test("a successful desktop probe reports each format's real cancellation mode", async ({ page }) => {
  await page.goto("/harness/resource-budget.html?desktop=1&capabilities=1");

  const band = page.locator(NATIVE_BAND);
  const grid = band.locator(".resource-budget__native-grid");
  await expect(grid).toHaveCount(1);
  await expect(page.locator(".resource-budget__boundary--error")).toHaveCount(0);

  // Not browser mode, and the window budget is reported rather than a placeholder.
  await expect(band).not.toContainText("Browser mode retains the cache and scheduler ledger");
  await expect(band.locator("strong")).toHaveText("64.0 MiB cap");
  await expect(page.getByText("Host RSS", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("384 MiB");
  await expect(page.getByText("Host RSS high", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("512 MiB");
  await expect(page.getByText("Process tree RSS", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("640 MiB");
  await expect(page.getByText("Process tree high", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("768 MiB");
  await expect(page.getByText("Tree processes", { exact: true }).locator("xpath=..").locator("dd"))
    .toHaveText("3");
  await expect(page.getByText(/Process-tree coverage:/)).toContainText("host_plus_descendants");
  await expect(page.getByText(/Process-tree coverage:/)).toContainText("detached XPC or webview helpers");

  // Cancellation is per-format. A table that labelled every row the same way
  // would be wrong in exactly the way the boundary prose was.
  const rowMode = async (label: string) =>
    grid.locator("div").filter({ has: page.getByText(label, { exact: true }) }).locator("dd");
  await expect(await rowMode("NPY windows")).toHaveText("Cooperative");
  await expect(await rowMode("Classic netCDF")).toHaveText("Cooperative");
  await expect(await rowMode("ZFP windows")).toHaveText("Boundary only");
  await expect(await rowMode("netCDF4")).toHaveText("Boundary only");

  // The mixed contract must actually be mixed, not four identical labels.
  const modes = await grid.locator("dd").allTextContents();
  expect(modes).toHaveLength(4);
  expect(new Set(modes).size).toBe(2);

  // Each row carries the backend's own limitation text, not invented copy.
  await expect(band).toContainText("stop cooperatively between row reads");
  await expect(band).toContainText("without an abort hook");
});
