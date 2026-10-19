import { expect, test } from "@playwright/test";

test("manual archive navigation remains on Runs when a preferred startup case exists", async ({ page }) => {
  await page.goto("/harness/run-library.html?startupBeast=1");
  await expect(page.getByRole("button", { name: "Configure", exact: true })).toHaveAttribute("aria-current", "page");
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open beast_xl in Run Archive" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Runs", exact: true })).toHaveAttribute("aria-current", "page");
});

test("restored selected archive run survives a preferred startup case", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("nimbus-studio:navigation:v1", JSON.stringify({
    schema: "nimbus.studio.navigation", schemaVersion: 1, page: "runs", selectedRunId: "run-002",
  })));
  await page.goto("/harness/run-library.html?startupBeast=1");
  await expect(page.getByRole("button", { name: "Open beast_xl in Run Archive" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Runs", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".run-detail__header h2")).toHaveText("cartesian_agrid_repair");
});

test("primary destination and valid run context survive a Studio restart", async ({ page }) => {
  await page.goto("/harness/run-library.html");

  const projects = page.getByRole("button", { name: "Projects", exact: true });
  await expect(projects).toHaveAttribute("aria-keyshortcuts", /\+0$/);
  await projects.click();
  await expect(projects).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".queue-panel")).toHaveCount(0);
  const projectLedger = page.getByRole("region", { name: "Projects" });
  await expect(projectLedger.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
  await expect(projectLedger).toContainText("local data references");
  await expect(projectLedger.getByRole("button", { name: "Import project…" })).toBeVisible();
  await expect(projectLedger).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
  await expect(projectLedger).toHaveCSS("border-top-width", "0px");
  await expect(projectLedger).toHaveCSS("border-radius", "0px");
  await expect(projectLedger).toHaveCSS("overflow-y", "hidden");
  const projectFit = await projectLedger.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(projectFit.scrollWidth).toBeLessThanOrEqual(projectFit.clientWidth);

  await page.reload();
  await expect(page.getByRole("button", { name: "Projects", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".queue-panel")).toHaveCount(0);

  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await page.getByRole("button", { name: "Open cartesian_agrid_repair in Run Archive" }).click();
  await expect(page.getByRole("button", { name: "Runs", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".run-detail__header h2")).toHaveText("cartesian_agrid_repair");

  await page.reload();
  await expect(page.getByRole("button", { name: "Runs", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".run-detail__header h2")).toHaveText("cartesian_agrid_repair");
  await expect(page.getByRole("button", { name: "Open cartesian_agrid_repair in Run Archive" })).toHaveAttribute("aria-current", "true");
});

test("invalid durable navigation fails back to a reachable Configure page", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.evaluate(() => {
    localStorage.setItem("nimbus-studio:navigation:v1", JSON.stringify({
      schema: "nimbus.studio.navigation",
      schemaVersion: 1,
      page: "removed-page",
      selectedRunId: "run-002",
    }));
  });
  await page.reload();
  await expect(page.getByRole("button", { name: "Configure", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".queue-panel")).toHaveCount(0);
});

test("a named project restores its analysis after leaving and restarting Studio", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Projects", exact: true }).click();
  const ledger = page.getByRole("region", { name: "Projects" });
  await ledger.getByRole("button", { name: "Open included example" }).click();

  await expect(page.getByRole("button", { name: "Plots", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".plot-studio__dataset-identity h2")).toContainText("Manufactured supercell");

  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await expect(page.locator(".queue-panel")).toHaveCount(0);
  await page.getByRole("button", { name: "Plots", exact: true }).click();
  await expect(page.locator(".plot-studio__dataset-identity h2")).toContainText("Manufactured supercell");

  await page.reload();
  await expect(page.getByRole("button", { name: "Plots", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".plot-studio__dataset-identity h2")).toContainText("Manufactured supercell");
});

test("an unsaved configuration draft survives primary navigation", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  const experimentName = page.getByLabel("Experiment configuration");
  await experimentName.fill("navigation-safe-draft");
  await expect(page.getByRole("region", { name: "Configuration drafts" }).getByText("File has unsaved changes", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await expect(page.locator(".queue-panel__header h2")).toHaveText("Run Archive");
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await expect(page.getByLabel("Experiment configuration")).toHaveValue("navigation-safe-draft");

  await page.reload();
  await expect(page.getByRole("button", { name: "Configure", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.getByLabel("Experiment configuration")).toHaveValue("navigation-safe-draft");
});

test("critical workflow surfaces expose truthful controls and explained disabled states", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  const destinations = ["Configure", "Projects", "Runs", "Plots", "Settings"];
  const prohibitedCopy = /coming soon|not implemented|placeholder|planned feature/i;

  for (const destination of destinations) {
    await page.getByRole("button", { name: destination, exact: true }).click();
    await expect(page.locator("main")).toBeVisible();
    await expect(page.locator("main")).not.toContainText(prohibitedCopy);

    const unexplained = await page.locator("main button:visible:disabled").evaluateAll((buttons) =>
      buttons
        .filter((button) => !(button.getAttribute("title") ?? "").trim())
        .map((button) => (button.textContent ?? "").trim() || button.getAttribute("aria-label") || "unnamed"),
    );
    expect(unexplained, `${destination} has disabled actions without a reason`).toEqual([]);
  }
});

test("settings radio groups expose one tab stop and support arrow-key selection", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();

  const theme = page.getByRole("radiogroup", { name: "Theme" });
  const system = theme.getByRole("radio", { name: "System" });
  const light = theme.getByRole("radio", { name: "Light" });
  const dark = theme.getByRole("radio", { name: "Dark" });
  await expect(theme.getByRole("radio")).toHaveCount(3);
  await expect(system).toBeChecked();
  await expect(system).toHaveAttribute("tabindex", "0");
  await expect(light).toHaveAttribute("tabindex", "-1");
  await expect(dark).toHaveAttribute("tabindex", "-1");

  await system.focus();
  await system.press("End");
  await expect(dark).toBeFocused();
  await expect(dark).toBeChecked();
  await dark.press("ArrowRight");
  await expect(system).toBeFocused();
  await expect(system).toBeChecked();

  const plotSide = page.getByRole("radiogroup", { name: "Plot side" });
  const right = plotSide.getByRole("radio", { name: "Right" });
  const left = plotSide.getByRole("radio", { name: "Left" });
  await expect(right).toBeChecked();
  await right.press("ArrowLeft");
  await expect(left).toBeFocused();
  await expect(left).toBeChecked();
});

test("settings storage failures explain that unsaved edits require the page to stay open", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Theme" })
    .getByRole("radio", { name: "Dark" }).click();
  await page.evaluate(() => {
    const nativeSetItem = Storage.prototype.setItem;
    Storage.prototype.setItem = function setItem(key: string, value: string) {
      if (key === "nimbus-studio:settings:v1") throw new Error("fixture quota exceeded");
      nativeSetItem.call(this, key, value);
    };
  });

  await page.getByRole("button", { name: "Save settings" }).click();
  await expect(page.getByRole("alert")).toHaveText(
    /Could not save these settings\. Keep this page open and retry\./,
  );
  await expect(page.getByRole("button", { name: "Save settings" })).toBeEnabled();
});
