import { expect, test, type Locator, type Page } from "@playwright/test";

async function openTerrain(page: Page) {
  await page.goto("/harness/index.html?mode=studio&terrain=1");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.getByRole("button", { name: "Cross-sections", exact: true }).click();
  await page.getByRole("button", { name: "Slice", exact: true }).click();
  await expect(page.getByText("physical-volume terrain→Cartesian vertical remap", { exact: true })).toBeVisible();
}

async function checkNoticeLayout(body: Locator) {
  await expect.poll(() => body.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const graph = element.querySelector(".plot-canvas")?.getBoundingClientRect();
    const notices = [...element.querySelectorAll<HTMLElement>(":scope > .plot-overlay")];
    let previousBottom = graph?.bottom ?? box.top;
    return Boolean(graph && graph.height >= 120 && notices.length > 0)
      && notices.every((notice) => {
        const rect = notice.getBoundingClientRect();
        const valid = rect.top >= previousBottom - 1 && rect.bottom <= box.bottom + 1
          && rect.left >= box.left && rect.right <= box.right + 1
          && notice.scrollWidth <= notice.clientWidth + 1;
        previousBottom = rect.bottom;
        return valid;
      });
  })).toBe(true);
}

test("terrain and loading notices stay separate from each other and the toolbar", async ({ page }, info) => {
  await openTerrain(page);
  const body = page.locator(".plot-body").first();
  await checkNoticeLayout(body);
  const toolbar = body.locator(".modebar");
  await body.locator(".plot-canvas").hover();
  await expect(toolbar).toBeVisible();
  await toolbar.locator(".modebar-btn").first().click({ trial: true });
  const originalValues = await body.locator(".plot-canvas").evaluate(element => {
    const graph = element as HTMLElement & { data: { z: ArrayLike<ArrayLike<number>> }[] };
    return Array.from(graph.data[0].z, row => Array.from(row));
  });
  await page.evaluate(() => window.__nimbusStudioHarness!.setFieldReadDelay(1800));
  const sliceIndex = page.locator(".control-group", { hasText: /^Index \(/ }).first().locator('input[type="range"]');
  await expect(sliceIndex).toHaveAttribute("max", "35");
  await sliceIndex.fill("2");
  await expect(sliceIndex).toHaveValue("2");
  await expect(sliceIndex).toBeEnabled();
  await expect(body.locator(".plot-overlay")).toHaveCount(2);
  await expect(body).toContainText("Displaying index 0 · loading selected y index 2");
  expect(await body.locator(".plot-canvas").evaluate(element => {
    const graph = element as HTMLElement & { data: { z: ArrayLike<ArrayLike<number>> }[] };
    return Array.from(graph.data[0].z, row => Array.from(row));
  })).toEqual(originalValues);
  await checkNoticeLayout(body);
  await expect(page.getByText(/loading selected .* index 2/)).toHaveCount(1);
  await page.evaluate(() => window.__nimbusStudioHarness!.setFieldReadDelay(0));
  await expect(page.getByText(/loading selected .* index 2/)).toHaveCount(0);
  await expect(sliceIndex).toHaveValue("2");
  expect(await body.locator(".plot-canvas").evaluate(element => {
    const graph = element as HTMLElement & { data: { z: ArrayLike<ArrayLike<number>> }[] };
    return Array.from(graph.data[0].z, row => Array.from(row));
  })).not.toEqual(originalValues);
  await checkNoticeLayout(body);
  await body.screenshot({ path: info.outputPath("terrain-notice.png") });

  await page.getByRole("button", { name: "Vector field", exact: true }).click();
  await expect(page.getByText("physical-volume terrain→Cartesian vector remap", { exact: true })).toBeVisible();
  await checkNoticeLayout(body);
  await body.screenshot({ path: info.outputPath("vector-notice.png") });
});

test("notices wrap in a compact window and stay inside quad panes", async ({ page }, info) => {
  await page.setViewportSize({ width: 800, height: 640 });
  await openTerrain(page);
  await checkNoticeLayout(page.locator(".plot-body").first());
  await expect.poll(async () => (await page.locator(".plot-canvas").first().boundingBox())?.height ?? 0)
    .toBeGreaterThanOrEqual(208);
  await page.getByRole("button", { name: "Play frames", exact: true }).click({ trial: true });
  await page.screenshot({ path: info.outputPath("compact-notice.png") });

  await page.setViewportSize(info.project.name.includes("retina")
    ? { width: 1512, height: 982 } : { width: 2560, height: 1440 });
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__panes-options").getByRole("button", { name: "Quad", exact: true }).click();
  const pickers = page.getByRole("combobox", { name: "Companion pane plot type" });
  for (let index = 0; index < 3; index++) await pickers.nth(index).selectOption("slice");
  await expect(page.locator(".plot-body")).toHaveCount(4);
  for (const body of await page.locator(".plot-body").all()) {
    await expect(body.locator(".plot-canvas")).toBeVisible();
  }
  await checkNoticeLayout(page.locator(".plot-body").first());
  await page.screenshot({ path: info.outputPath("quad-notice.png") });
});
