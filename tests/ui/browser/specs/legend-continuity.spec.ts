import { expect, test, type Page } from "@playwright/test";

declare global { interface Window { heldLegendRead?: { held: boolean; release: () => void } } }

async function ready(page: Page, step: number) {
  await page.waitForFunction(index => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(index);
  }, step, { timeout: 60_000 });
}
async function toggle(page: Page, tab: string, group: string) {
  await page.locator(".plot3d-control-tab", { hasText: tab }).click();
  await page.getByRole("group", { name: group, exact: true }).getByRole("button").first().click();
  await ready(page, 1);
}
async function legends(page: Page) {
  return page.locator(".plot3d-overlay-stack").evaluate(element => element.innerHTML);
}
async function holdNextFrame(page: Page) {
  await page.evaluate(() => {
    const internal = (window as unknown as { __TAURI_INTERNALS__: {
      invoke: (command: string, args?: Record<string, unknown>, options?: unknown) => Promise<unknown>;
    } }).__TAURI_INTERNALS__;
    const invoke = internal.invoke;
    let release = () => {};
    const blocked = new Promise<void>(resolve => { release = resolve; });
    window.heldLegendRead = { held: false, release };
    internal.invoke = async (command, args, options) => {
      if ((command === "read_field_window" || command === "read_field_bytes") && args?.stepIndex === 7) {
        window.heldLegendRead!.held = true;
        await blocked;
      }
      return invoke(command, args, options);
    };
    window.__nimbusVtkHarness!.setStep(7);
  });
  await page.waitForFunction(() => window.heldLegendRead?.held);
}

for (const plot of ["volume", "supercell"] as const) test(`${plot} retains all committed legends during a held multi-field frame load`, async ({ page }) => {
  test.setTimeout(150_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(plot => { window.__nimbusVtkHarness!.setSize(640, 480);
    window.__nimbusVtkHarness!.setPlot(plot); window.__nimbusVtkHarness!.setStep(1); }, plot);
  await ready(page, 1);
  if (plot === "supercell") {
    await toggle(page, "Precip", "Precipitation");
    await toggle(page, "Vortex", "Vortex core");
  }
  await toggle(page, "Radar", "Radar floor");
  await page.getByRole("button", { name: "Column max", exact: true }).click();
  await ready(page, 1);
  await page.locator('.plot3d-control-tab[aria-expanded="true"]').click();
  await expect(page.locator(".plot3d-colorbar")).toHaveCount(plot === "volume" ? 2 : 1);
  if (plot === "supercell") await expect(page.locator(".plot3d-discrete-legend")).toHaveCount(2);
  const before = await legends(page);
  const scalars = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeInputs());
  await holdNextFrame(page);
  expect(await legends(page)).toBe(before);
  const pending = await page.evaluate(async () => ({ capture: await window.__nimbusVtkHarness!.captureAppearance(),
    inputs: window.__nimbusVtkHarness!.inspectVolumeInputs() }));
  expect(pending.capture!.evidence.descriptor.frameIndex).toBe(1);
  expect(pending.inputs).toEqual(scalars);
  expect(await legends(page)).toBe(before);
  await page.evaluate(() => window.heldLegendRead!.release());
  await ready(page, 7);
  await expect(page.locator(".plot3d-colorbar")).toHaveCount(plot === "volume" ? 2 : 1);
  if (plot === "supercell") await expect(page.locator(".plot3d-discrete-legend")).toHaveCount(2);
  expect(errors).toEqual([]);
});
