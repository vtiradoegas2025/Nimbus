import { expect, test } from "@playwright/test";

test("appearance is persistent, neutral, and follows the operating system live", async ({ page }, testInfo) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/harness/run-library.html");
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-appearance", "system");
  await expect(root).toHaveAttribute("data-theme", "dark");

  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();
  const darkScreenshot = testInfo.outputPath("professional-dark-settings.png");
  await page.screenshot({ path: darkScreenshot, fullPage: true });
  await testInfo.attach("professional-dark-settings", {
    path: darkScreenshot,
    contentType: "image/png",
  });
  const light = page.getByRole("radio", { name: "Light" });
  await light.click();
  await expect(light).toHaveAttribute("aria-checked", "true");
  await page.getByRole("button", { name: /Save settings/ }).click();
  await expect(root).toHaveAttribute("data-appearance", "light");
  await expect(root).toHaveAttribute("data-theme", "light");
  await page.waitForTimeout(220);
  const lightScreenshot = testInfo.outputPath("professional-light-settings.png");
  await page.screenshot({ path: lightScreenshot, fullPage: true });
  await testInfo.attach("professional-light-settings", {
    path: lightScreenshot,
    contentType: "image/png",
  });

  await page.reload();
  await expect(root).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();
  await page.getByRole("radio", { name: "System" }).click();
  await page.getByRole("button", { name: /Save settings/ }).click();
  await expect(root).toHaveAttribute("data-theme", "dark");

  await page.emulateMedia({ colorScheme: "light" });
  await expect(root).toHaveAttribute("data-appearance", "system");
  await expect(root).toHaveAttribute("data-theme", "light");

  const primaryColors = await page.getByRole("button", { name: "Saved", exact: true }).evaluate((button) => {
    const style = getComputedStyle(button);
    return { background: style.backgroundColor, foreground: style.color };
  });
  expect(primaryColors.background).not.toBe(primaryColors.foreground);
});

test("global command palette navigates safely and ignores editable targets", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  const mac = await page.evaluate(() => /Mac|iPhone|iPad|iPod/i.test(navigator.platform));
  const modifier = mac ? "Meta" : "Control";

  await page.keyboard.press(`${modifier}+k`);
  const dialog = page.getByRole("dialog", { name: "Command palette" });
  await expect(dialog).toBeVisible();
  const search = page.getByRole("combobox", { name: "Find a command" });
  await expect(search).toBeFocused();
  await search.fill("netcdf analysis");
  await expect(page.getByRole("option", { name: /Open Plot Studio/ })).toHaveCount(1);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: "Plots", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(dialog).not.toBeVisible();

  await page.keyboard.press(`${modifier}+2`);
  await expect(page.locator(".queue-panel__header h2")).toHaveText("Run Archive");
  const runSearch = page.getByRole("searchbox", { name: "Find a run" });
  await runSearch.focus();
  await page.keyboard.press(`${modifier}+1`);
  await expect(page.locator(".queue-panel__header h2")).toHaveText("Run Archive");
  await expect(runSearch).toBeFocused();
  await page.keyboard.press(`${modifier}+k`);
  await expect(dialog).not.toBeVisible();

  await page.getByRole("button", { name: /Open command palette/ }).click();
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();

  await page.keyboard.press(`${modifier}+,`);
  await expect(page.getByRole("button", { name: "Settings", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("button", { name: "Configure" })).toHaveAttribute("aria-keyshortcuts", mac ? "Meta+1" : "Control+1");
});

test("dataset commands open qualified sources and treat cancellation and errors safely", async ({ page }) => {
  await page.goto("/harness/run-library.html?datasetPicker=packaged-nimbus");
  await page.getByRole("button", { name: /Open command palette/ }).click();
  const search = page.getByRole("combobox", { name: "Find a command" });
  await search.fill("nimbus run directory");
  const openNimbus = page.getByRole("option", { name: /Open Nimbus history/ });
  await expect(openNimbus).toHaveCount(1);
  await expect(openNimbus.locator("kbd")).toHaveCount(0);
  await openNimbus.click();
  await expect(page.getByRole("button", { name: "Plots", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".plot-studio__dataset-identity h2")).toContainText("Manufactured supercell");

  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();
  await page.getByRole("button", { name: /Open command palette/ }).click();
  await page.getByRole("combobox", { name: "Find a command" }).fill("cm1 netcdf history");
  await page.getByRole("option", { name: /Open netCDF data/ }).click();
  await expect(page.getByRole("button", { name: "Settings", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("alert")).toHaveCount(0);

  await page.goto("/harness/run-library.html?datasetPicker=error");
  await page.getByRole("button", { name: /Open command palette/ }).click();
  await page.getByRole("combobox", { name: "Find a command" }).fill("cm1 netcdf history");
  await page.getByRole("option", { name: /Open netCDF data/ }).click();
  const error = page.getByRole("alert", { name: "Command error" });
  await expect(error).toContainText("Could not open netCDF data");
  await expect(error).toContainText("operating system picker is unavailable");
  await expect(page.getByRole("button", { name: "Settings", exact: true })).toHaveAttribute("aria-current", "page");
  await error.getByRole("button", { name: "Dismiss command error" }).click();
  await expect(error).toHaveCount(0);
});

test("reopening the same dataset identity explicitly refreshes Plot Studio", async ({ page }) => {
  await page.goto("/harness/run-library.html?datasetPicker=repeat-nimbus");

  const openNimbusHistory = async () => {
    await page.getByRole("button", { name: /Open command palette/ }).click();
    await page.getByRole("combobox", { name: "Find a command" }).fill("nimbus run directory");
    await page.getByRole("option", { name: /Open Nimbus history/ }).click();
  };

  await openNimbusHistory();
  const datasetHeading = page.locator(".plot-studio__dataset-identity h2");
  await expect(datasetHeading).toHaveText("Repeat-open fixture · request 1");

  await openNimbusHistory();
  await expect(datasetHeading).toHaveText("Repeat-open fixture · request 2");
});

test("CM1 command opens a qualified netCDF source through Plot Studio", async ({ page }) => {
  await page.goto("/harness/run-library.html?datasetPicker=packaged-cm1");
  await page.getByRole("button", { name: /Open command palette/ }).click();
  await page.getByRole("combobox", { name: "Find a command" }).fill("cm1 netcdf history");
  await page.getByRole("option", { name: /Open netCDF data/ }).click();

  await expect(page.getByRole("button", { name: "Plots", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".plot-studio__dataset-identity h2")).toHaveText("CM1 command-palette fixture");
  await expect(page.locator(".plot-studio__source-badge")).toHaveText("netCDF");
});
