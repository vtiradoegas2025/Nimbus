import { expect, test } from "@playwright/test";

test("startup crashes explain the signal and keep empty plots closed", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?mode=run-actions&outcome=crash");
  await expect(page.getByRole("alert")).toContainText("Run failed without a complete saved frame");
  await expect(page.getByRole("alert")).toContainText("SIGILL");
  await expect(page.getByRole("button", { name: "View plots", exact: true })).toBeDisabled();
  await expect(page.locator(".run-detail__plot-notice")).toContainText("No complete saved frames");
  await page.getByText("Run evidence", { exact: true }).click();
  await expect(page.locator(".run-detail__provenance-body")).toContainText("Process ended on signal 4");
});

test("verified partial output remains accessible after a failed run", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?mode=run-actions&outcome=partial");
  await page.getByRole("button", { name: "View plots", exact: true }).click();
  await expect(page.getByLabel("Opened plot run")).toHaveText("action-0");
});

test("a completed process without saved frames does not open empty plots", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?mode=run-actions&outcome=empty");
  await expect(page.getByRole("button", { name: "View plots", exact: true })).toBeDisabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("remote transfer failures distinguish saved computation from received results", async ({ page }) => {
  await page.goto("/harness/queue-panel.html?mode=run-actions&remote=1");
  await expect(page.getByRole("alert").filter({ hasText: "Remote process finished" })).toContainText("does not require rerunning");
  await expect(page.getByRole("button", { name: "View plots", exact: true })).toBeDisabled();
  await expect(page.locator(".run-detail__plot-notice")).toContainText("Receive and verify");
});
