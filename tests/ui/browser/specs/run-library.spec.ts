import { expect, test } from "@playwright/test";

test("run library searches local records and navigates into Run Archive", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Runs", exact: true }).click();

  const search = page.getByLabel("Find a run");
  await expect(search).toBeVisible();
  await expect(page.getByText("3 of 3 runs")).toBeVisible();

  await search.fill("agrid repair");
  await expect(page.getByText("1 of 3 runs")).toBeVisible();
  await expect(page.getByRole("button", { name: "Open cartesian_agrid_repair in Run Archive" })).toBeVisible();
  await expect(page.locator(".queue-list").getByText("cart_cgrid_supercell")).toHaveCount(0);

  await page.getByRole("button", { name: "Open cartesian_agrid_repair in Run Archive" }).click();
  await expect(page.getByRole("button", { name: "Runs" })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".run-detail__header h2")).toHaveText("cartesian_agrid_repair");
  await page.locator(".run-detail__provenance > summary").click();
  await expect(page.getByRole("table", { name: "Requested and observed runtime controls" })).toBeVisible();
  await expect(page.getByRole("row", { name: /Compute backend vulkan cpu/ })).toBeVisible();
  await expect(page.getByRole("row", { name: /Backend fallback Allowed Active/ })).toBeVisible();
  await expect(page.getByText("/verified/inputs/wk.sounding")).toBeVisible();
  await expect(page.getByText("fnv1a64:4444444444444444")).toBeVisible();
  await expect(page.getByText("launch verified · runtime verified")).toBeVisible();
  await expect(page.getByRole("button", { name: "Export evidence bundle" })).toBeDisabled();
});

test("run library empty search explains how to recover", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await page.getByLabel("Find a run").fill("netcdf missing-case");
  await expect(page.getByText("No runs match this search.")).toBeVisible();
  await page.getByRole("button", { name: "Clear run search" }).click();
  await expect(page.getByText("3 of 3 runs")).toBeVisible();
});

test("run progress and queue resizing expose their current values", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Runs", exact: true }).click();

  const separator = page.getByRole("separator", { name: "Resize queue panel" });
  await expect(separator).toHaveAttribute("aria-valuemin", "180");
  await expect(separator).toHaveAttribute("aria-valuemax", "560");
  await expect(separator).toHaveAttribute("aria-valuenow", "240");
  await separator.focus();
  await separator.press("ArrowRight");
  await expect(separator).toHaveAttribute("aria-valuenow", "256");
  await separator.press("Home");
  await expect(separator).toHaveAttribute("aria-valuenow", "180");
  await separator.press("End");
  await expect(separator).toHaveAttribute("aria-valuenow", "560");

  await page.getByRole("button", {
    name: "Open cart_cgrid_supercell in Run Archive",
  }).click();
  const detailProgress = page.getByRole("progressbar", {
    name: "Simulation progress",
    exact: true,
  });
  await expect(detailProgress).toHaveAttribute("aria-valuenow", "70");
  const inlineStyle = await detailProgress.locator(".run-detail__progress-fill").evaluate((element) => ({
    transform: (element as HTMLElement).style.transform,
    width: (element as HTMLElement).style.width,
  }));
  expect(inlineStyle).toEqual({ transform: "scaleX(0.7)", width: "" });
});
