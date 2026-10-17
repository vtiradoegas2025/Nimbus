import { expect, test, type Locator, type Page } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  page.on("pageerror", error => { throw error; });
  await page.goto("/harness/index.html?mode=studio");
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
});

async function reachable(locator: Locator) {
  await locator.scrollIntoViewIfNeeded();
  expect(await locator.evaluate(element => {
    const r = element.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return r.width > 0 && r.height > 0 && (hit === element || element.contains(hit));
  })).toBe(true);
}

async function tool(page: Page, name: string) {
  await page.locator(".plot-studio__tools-menu summary").click();
  await page.locator(".plot-studio__tools-popover").getByRole("button", { name, exact: true }).click();
}

for (const viewport of [{ width: 1280, height: 720 }, { width: 760, height: 500 }, { width: 420, height: 640 }]) {
  test(`Tools stay within the window and every action is reachable at ${viewport.width}x${viewport.height}`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    const trigger = page.locator(".plot-studio__tools-menu summary");
    await trigger.click();
    const popup = page.locator(".plot-studio__tools-popover");
    await expect(popup).toBeVisible();
    await expect.poll(async () => popup.evaluate(element => {
      const r = element.getBoundingClientRect();
      return r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight;
    })).toBe(true);
    const actions = popup.getByRole("button");
    for (const action of await actions.all()) await reachable(action);
    expect(await popup.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await popup.getByRole("button", { name: "Research workspace", exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: info.outputPath("tools-menu.png") });
    await page.keyboard.press("Escape");
    await expect(popup).toBeHidden();
    await expect(trigger).toBeFocused();
    await trigger.click();
    await page.locator(".plot-studio__view-menu summary").click();
    await expect(popup).toBeHidden();
    await expect(page.locator(".plot-studio__view-popover")).toBeVisible();
    await page.locator(".plot-studio__dataset-identity").click();
    await expect(page.locator(".plot-studio__view-popover")).toBeHidden();
  });
}

test("tool panels scroll together without clipping their forms or the frame bar", async ({ page }, info) => {
  await page.setViewportSize({ width: 760, height: 600 });
  await tool(page, "Derived expressions");
  const editor = page.locator(".expression-fields");
  await expect(editor).toBeVisible();
  const input = editor.locator('input').first();
  await input.fill("retained_tool_draft");
  await tool(page, "External providers");
  await expect(input).toHaveValue("retained_tool_draft");
  const providers = page.locator(".provider-console");
  await providers.locator(".provider-console__import summary").click();
  await reachable(providers.locator("textarea"));
  for (const button of await providers.getByRole("button").all()) await reachable(button);
  const body = page.locator(".plot-studio__tool-panels-body");
  expect(await body.evaluate(element => element.clientHeight)).toBeGreaterThan(200);
  expect(await body.evaluate(element => element.scrollHeight > element.clientHeight)).toBe(true);
  expect(await body.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  await reachable(page.getByRole("button", { name: "Close tools", exact: true }));
  await reachable(page.getByRole("slider", { name: "Selected frame", exact: true }));
  await page.screenshot({ path: info.outputPath("tools-panels.png") });
  await page.getByRole("button", { name: "Close tools", exact: true }).click();
  await expect(page.getByRole("region", { name: "Tools panels", exact: true })).toHaveCount(0);
  await expect(page.locator(".plot-studio__tools-menu summary")).toBeFocused();
});

test("Research opens directly from the header and fits a narrow window", async ({ page }, info) => {
  await page.setViewportSize({ width: 620, height: 640 });
  await page.getByRole("button", { name: "Research", exact: true }).click();
  const research = page.getByRole("region", { name: "Research workspace", exact: true });
  await expect(research).toBeVisible();
  await expect(page.getByRole("button", { name: "Research", exact: true })).toHaveAttribute("aria-pressed", "true");
  expect(await research.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  await reachable(research.getByRole("button", { name: "Import research recipe or package", exact: true }));
  await page.screenshot({ path: info.outputPath("research-narrow.png") });
  await research.getByRole("button", { name: "Close research workspace", exact: true }).click();
  await expect(research).toHaveCount(0);
  await expect(page.locator(".plot-studio__frame-bar")).toBeVisible();
});

test("data, project and remote tools keep their full content inside one usable panel", async ({ page }) => {
  await page.setViewportSize({ width: 760, height: 600 });
  await page.goto("/harness/index.html?mode=studio&cm1=1");
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  for (const [name, selector] of [
    ["Data health", ".data-health"], ["Source details", ".plot-studio__source-details"],
    ["Field registry", ".derived-ledger"], ["Projects", ".project-navigator"],
    ["Recent data", ".recent-datasets"], ["Remote visualization workspace", ".remote-workspace"],
    ["Resource use", ".resource-budget"],
  ]) {
    await tool(page, name);
    const panel = page.locator(selector);
    await expect(panel).toBeVisible();
    if (name === "Field registry") await panel.locator("summary").click();
    const body = page.locator(".plot-studio__tool-panels-body");
    expect(await body.evaluate(element => element.clientHeight)).toBeGreaterThan(200);
    expect(await body.evaluate(element => element.scrollWidth <= element.clientWidth), name).toBe(true);
    const content = panel.locator("button:visible, input:visible, select:visible").last();
    if (await content.count()) await reachable(content);
    await page.getByRole("button", { name: "Close tools", exact: true }).click();
    await expect(page.locator(".plot-studio__tool-panels")).toHaveCount(0);
  }
});
