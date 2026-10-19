import { expect, test, type Page } from "@playwright/test";

const profile = { id: "workstation-fixture", name: "Linux workstation", host: "research-node", binaryPath: "/home/research/Nimbus/bin/nimbus_sim", workDirectory: "/home/research/.nimbus/remote", assetMappings: [{ localPath: "/data/sounding.txt", remotePath: "inputs/sounding.txt" }] };
const category = (page: Page, name: string) => page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name, exact: true }).click();
test.beforeEach(async ({ page }) => {
  page.on("pageerror", error => { throw error; });
  await page.goto("/harness/config.html?mode=settings");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.settings{height:100vh}" });
});

for (const size of [{ width: 1440, height: 900 }, { width: 1100, height: 720 }, { width: 760, height: 600 }]) {
  test(`Settings categories and controls fit ${size.width}x${size.height}`, async ({ page }, info) => {
    await page.setViewportSize(size);
    for (const name of ["This computer", "Performance", "Compute nodes", "Remote rendering", "Files and defaults", "Exports", "Appearance"]) {
      await category(page, name);
      const body = page.locator(".settings__body");
      expect(await body.evaluate(el => el.scrollWidth <= el.clientWidth), name).toBe(true);
      const controls = body.locator("input:visible, select:visible, button:visible");
      for (const control of await controls.all()) {
        await control.scrollIntoViewIfNeeded();
        expect(await control.evaluate(el => {
          const rect = el.getBoundingClientRect();
          const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
          return rect.width > 0 && rect.height > 0 && (hit === el || el.contains(hit));
        }), `${name} control is reachable`).toBe(true);
      }
      if (name === "Compute nodes" || name === "Appearance" || name === "Performance") await page.screenshot({ path: info.outputPath(`${name.replaceAll(" ", "-")}.png`) });
    }
  });
}

test("app preferences survive category switches, save, reload and discard", async ({ page }) => {
  await page.getByLabel("Binary path override", { exact: true }).fill("/custom/nimbus_sim");
  await category(page, "Files and defaults");
  await page.getByLabel("Runs directory", { exact: true }).fill("/data/nimbus-results");
  await page.getByLabel("Default template", { exact: true }).selectOption("custom/rtx5090_lifecycle_50m/01_classic_supercell.yaml");
  await category(page, "Exports");
  await page.getByLabel("ffmpeg path", { exact: true }).fill("/custom/ffmpeg");
  await category(page, "Appearance");
  await page.getByRole("radio", { name: "Dark", exact: true }).click();
  await page.getByRole("combobox", { name: "Text size", exact: true }).selectOption("comfortable");
  await page.getByRole("radio", { name: "Left", exact: true }).click();
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.locator("html")).toHaveAttribute("data-text-size", "comfortable");
  await expect(page.locator("html")).toHaveAttribute("data-plot-side", "left");
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("nimbus-studio:settings:v1")!));
  expect(saved).toMatchObject({ binaryPathOverride: "/custom/nimbus_sim", runsDirectory: "/data/nimbus-results", ffmpegPathOverride: "/custom/ffmpeg", defaultTemplate: "custom/rtx5090_lifecycle_50m/01_classic_supercell.yaml" });
  await page.reload();
  await expect(page.getByLabel("Binary path override", { exact: true })).toHaveValue("/custom/nimbus_sim");
  await page.getByLabel("Binary path override", { exact: true }).fill("/unsaved");
  await page.getByRole("button", { name: "Discard changes", exact: true }).click();
  await expect(page.getByLabel("Binary path override", { exact: true })).toHaveValue("/custom/nimbus_sim");
});

test("performance preferences validate, persist and preserve edits made during a pending save", async ({ page }) => {
  await category(page, "Performance");
  const workers = page.getByLabel("Loading workers", { exact: true });
  const memory = page.getByLabel("I/O memory limit (MiB)", { exact: true });
  await workers.selectOption("3");
  await memory.fill("1024");
  await page.getByLabel("Measure automatically on a new computer").uncheck();
  await page.evaluate(() => window.__nimbusSettingsFixture.holdBudget());
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__nimbusSettingsFixture.calls().filter(call =>
    call.command === "configure_native_io_budget" && call.payload.limitMib === 1024).length)).toBe(1);
  await memory.fill("2048");
  await page.evaluate(() => window.__nimbusSettingsFixture.releaseBudget());
  await expect(page.locator(".settings-performance__status")).toContainText("1,024 MiB");
  await expect(memory).toHaveValue("2048");
  await expect(page.getByRole("button", { name: "Save settings", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  await expect(page.locator(".settings-performance__status")).toContainText("2,048 MiB");
  await memory.fill("32");
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("64 to 16384");
  await expect(page.locator(".settings-performance__status")).toContainText("2,048 MiB");
  await page.reload();
  await category(page, "Performance");
  await expect(workers).toHaveValue("3");
  await expect(memory).toHaveValue("2048");
  await expect(page.getByLabel("Measure automatically on a new computer")).not.toBeChecked();
});

test("first-start worker measurement can stop, rerun and reuse a saved recommendation", async ({ page }) => {
  await category(page, "Performance");
  await page.evaluate(() => window.__nimbusSettingsFixture.startAutomaticMeasurement());
  await expect(page.getByRole("button", { name: "Stop measurement", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop measurement", exact: true }).click();
  await expect(page.getByRole("button", { name: "Measure this computer", exact: true })).toBeEnabled();
  expect(await page.evaluate(() => localStorage.getItem("nimbus-studio:worker-profile:v1"))).toBeNull();
  await page.getByRole("button", { name: "Measure this computer", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__nimbusSettingsFixture.calls().filter(call => call.command === "profile_loading_workers").length)).toBe(2);
  await page.evaluate(() => window.__nimbusSettingsFixture.finishMeasurement());
  await expect(page.getByLabel("Loading workers", { exact: true })).toContainText("Automatic (measured: 2)");
  await expect(page.locator(".settings-performance__status")).toContainText("2 loading workers");
  await page.reload();
  await category(page, "Performance");
  await page.evaluate(() => window.__nimbusSettingsFixture.startAutomaticMeasurement());
  await page.waitForTimeout(1700);
  expect(await page.evaluate(() => window.__nimbusSettingsFixture.calls().filter(call => call.command === "profile_loading_workers").length)).toBe(0);
  await expect(page.getByLabel("Loading workers", { exact: true })).toContainText("Automatic (measured: 2)");
});

test("path checks follow the current value and discard delayed results", async ({ page }) => {
  const input = page.getByLabel("Binary path override", { exact: true });
  await input.fill("/old/simulator");
  await page.evaluate(() => window.__nimbusSettingsFixture.holdSimulator());
  await page.getByRole("button", { name: "Check simulator path", exact: true }).click();
  await expect(page.getByRole("button", { name: "Checking path…", exact: true })).toBeDisabled();
  await input.fill("/new/simulator");
  await page.evaluate(() => window.__nimbusSettingsFixture.releaseSimulator());
  await expect(page.locator(".settings-path-check").first()).not.toContainText("Simulator found");
  await page.getByRole("button", { name: "Check simulator path", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "Simulator found" })).toContainText("/new/simulator");
  await category(page, "Exports");
  await page.getByLabel("ffmpeg path", { exact: true }).fill("/tools/ffmpeg");
  await page.getByRole("button", { name: "Check ffmpeg path", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "Encoder found" })).toContainText("/tools/ffmpeg");
  expect(await page.evaluate(() => window.__nimbusSettingsFixture.calls().filter(c => c.command === "ffmpeg_detect").at(-1)?.payload)).toEqual({ overridePath: "/tools/ffmpeg" });
});

test("connection imports stay drafts until saved and checks expire when inputs change", async ({ page }) => {
  await category(page, "Compute nodes");
  await page.locator('#settings-section-compute input[type="file"]').setInputFiles({ name: "workstation.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(profile)) });
  await expect(page.getByLabel("SSH host", { exact: true })).toHaveValue(profile.host);
  expect(await page.evaluate(() => localStorage.getItem("nimbus-studio:remote-compute:v1"))).toBeNull();
  await page.getByRole("button", { name: "Check connection", exact: true }).click();
  await expect(page.locator(".settings-remote__verified")).toContainText(profile.binaryPath);
  await page.getByLabel("Remote input file 1", { exact: true }).fill("inputs/revised.txt");
  await expect(page.locator(".settings-remote__verified")).toHaveCount(0);
  await category(page, "Appearance");
  await category(page, "Compute nodes");
  await expect(page.getByLabel("Remote input file 1", { exact: true })).toHaveValue("inputs/revised.txt");
  await page.getByRole("button", { name: "Save connection", exact: true }).click();
  await page.reload();
  await category(page, "Compute nodes");
  await expect(page.getByLabel("Remote simulator path", { exact: true })).toHaveValue(profile.binaryPath);
  await expect(page.getByLabel("Remote input file 1", { exact: true })).toHaveValue("inputs/revised.txt");
});

test("remote rendering reuses saved hosts and polls only while its category is visible", async ({ page }) => {
  await page.clock.install();
  await category(page, "Compute nodes");
  await page.locator('#settings-section-compute input[type="file"]').setInputFiles({ name: "workstation.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(profile)) });
  await page.getByRole("button", { name: "Save connection", exact: true }).click();
  const polls = () => page.evaluate(() => window.__nimbusSettingsFixture.calls().filter(c => c.command === "remote_workspace_status").length);
  expect(await polls()).toBe(0);
  await category(page, "Remote rendering");
  await page.getByLabel("Copy SSH host from", { exact: true }).selectOption(profile.id);
  await expect(page.getByLabel("Workspace SSH host", { exact: true })).toHaveValue(profile.host);
  await page.getByRole("button", { name: "Save workspace", exact: true }).click();
  await page.getByRole("button", { name: "Connect workspace", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open remote desktop", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Open remote desktop", exact: true }).click();
  await category(page, "This computer");
  const before = await polls();
  await page.clock.fastForward(6500);
  expect(await polls()).toBe(before);
  await category(page, "Remote rendering");
  await expect(page.getByRole("button", { name: "Disconnect workspace", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Disconnect workspace", exact: true }).click();
  await expect(page.getByRole("button", { name: "Connect workspace", exact: true })).toBeEnabled();
  expect(await page.evaluate(() => window.__nimbusSettingsFixture.calls().filter(c => c.command === "open_remote_workspace").length)).toBe(1);
});
