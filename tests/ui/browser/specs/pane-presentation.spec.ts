import { test, expect } from "@playwright/test";

test("companion 3D playback stays available with a whole-run primary plot", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("/__nimbus_browser_fixture__", "completed"));
  await page.getByRole("button", { name: "Time series", exact: true }).click();
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__panes-options").getByRole("button", { name: "Quad", exact: true }).click();
  const slider = page.getByRole("slider", { name: "Selected frame", exact: true });
  await expect(slider).toBeVisible();
  const frames = () => page.evaluate(() => window.__nimbusStudioHarness!.inspectPaneDescriptors()
    .map(descriptor => descriptor?.frameIndex ?? null));
  await expect.poll(frames, { timeout: 60_000 }).toEqual([null, 0, 0, 0]);
  await slider.fill("1");
  await expect.poll(frames, { timeout: 60_000 }).toEqual([null, 1, 1, 1]);
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await expect(slider).toHaveValue("2");
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  await expect.poll(frames, { timeout: 60_000 }).toEqual([null, 2, 2, 2]);
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await page.getByRole("button", { name: "Exit split", exact: true }).click();
  await expect(page.locator(".plot-studio__frame-bar")).toHaveCount(0);
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__panes-options").getByRole("button", { name: "Quad", exact: true }).click();
  await expect(page.getByRole("button", { name: "Play frames", exact: true })).toBeVisible();
});

test("mixed3D panes retain one saved time while a required streamline read is held", async ({ page }, info) => {
  test.setTimeout(150_000);
  await page.setViewportSize({ width: 1920, height: 1200 });
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.setRun("/__nimbus_browser_fixture__", "completed"));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__panes-options").getByRole("button", { name: "Quad", exact: true }).click();
  await page.evaluate(() => {
    const host = window as unknown as {
      __TAURI_INTERNALS__: { invoke: (command: string, payload?: Record<string, unknown>) => Promise<unknown> };
      paneReadHold: { count: number; release: () => void };
    };
    const invoke = host.__TAURI_INTERNALS__.invoke.bind(host.__TAURI_INTERNALS__);
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    host.paneReadHold = { count: 0, release };
    host.__TAURI_INTERNALS__.invoke = async (command, payload) => {
      if (["read_field_window", "read_field_bytes"].includes(command) && Number(payload?.stepIndex) === 1 && payload?.fieldName === "u") {
        host.paneReadHold.count++;
        await gate;
      }
      return invoke(command, payload);
    };
  });
  const pickers = page.getByRole("combobox", { name: "Companion pane plot type" });
  await pickers.nth(0).selectOption("isosurface");
  await pickers.nth(1).selectOption("streamlines");
  await pickers.nth(2).selectOption("supercell");
  const frames = () => page.evaluate(() => window.__nimbusStudioHarness!.inspectPaneDescriptors()
    .map(descriptor => descriptor?.frameIndex ?? null));
  await expect.poll(frames, { timeout: 90_000 }).toEqual([0, 0, 0, 0]);
  await expect(page.locator('[data-volume-quality-preview="true"]')).toHaveCount(0);
  await expect(page.locator('.plot3d-control-notice.is-visible')).toHaveCount(0);
  const before = await page.evaluate(async () => ({ captures: await window.__nimbusStudioHarness!.capturePaneAppearances(),
    scalars: window.__nimbusStudioHarness!.inspectPrimaryVolumeScalarImages() }));
  await page.evaluate(() => window.__nimbusStudioHarness!.clearFieldCache());
  await page.getByRole("slider", { name: "Selected frame", exact: true }).fill("1");
  await page.waitForFunction(() => {
    const held = (window as unknown as { paneReadHold: { count: number } }).paneReadHold.count;
    const snapshot = window.__nimbusStudioHarness!.presentationSnapshot();
    return held > 0 && snapshot?.allowed === false && snapshot.readyPaneIds.length === 3
      && !snapshot.readyPaneIds.includes("companion-1");
  }, undefined, { timeout: 90_000 });
  const pending = await page.evaluate(async () => ({ captures: await window.__nimbusStudioHarness!.capturePaneAppearances(),
    scalars: window.__nimbusStudioHarness!.inspectPrimaryVolumeScalarImages(),
    snapshot: window.__nimbusStudioHarness!.presentationSnapshot() }));
  expect(pending.captures.map(capture => capture?.evidence.descriptor.frameIndex ?? null)).toEqual([0, 0, 0, 0]);
  expect(pending.scalars).toEqual(before.scalars);
  for (const [index, capture] of pending.captures.entries()) {
    expect(capture!.evidence.layers).toEqual(before.captures[index]!.evidence.layers);
    expect(capture!.evidence.camera).toEqual(before.captures[index]!.evidence.camera);
    expect(capture!.pngDataUri).toEqual(before.captures[index]!.pngDataUri);
  }
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await page.waitForTimeout(1200);
  await expect(page.getByRole("slider", { name: "Selected frame", exact: true })).toHaveValue("1");
  expect(await page.evaluate(() => window.__nimbusStudioHarness!.presentationSnapshot()?.allowed)).toBe(false);
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  await page.evaluate(() => (window as unknown as { paneReadHold: { release: () => void } }).paneReadHold.release());
  await expect.poll(frames, { timeout: 90_000 }).toEqual([1, 1, 1, 1]);
  const published = await page.evaluate(() => window.__nimbusStudioHarness!.inspectPaneDescriptors()
    .map(descriptor => descriptor?.parameters?.renderedPublicationGeneration));
  expect(published.every(generation => typeof generation === "string" && generation.length > 0)).toBe(true);
  expect(new Set(published).size).toBe(1);
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await expect(page.getByRole("slider", { name: "Selected frame", exact: true })).toHaveValue("2");
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  await expect.poll(frames, { timeout: 90_000 }).toEqual([2, 2, 2, 2]);
  await info.attach("saved-time-cohort", { body: JSON.stringify({ before, pending,
    final: await page.evaluate(() => window.__nimbusStudioHarness!.presentationSnapshot()) }), contentType: "application/json" });
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__panes-options").getByRole("button", { name: "Single", exact: true }).click();
  await expect.poll(frames).toEqual([2]);
});
