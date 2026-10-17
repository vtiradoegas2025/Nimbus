import { test, expect, type Page } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test.skip(!process.env.NIMBUS_BEAST_XL_REFERENCE_RUN, "requires preserved Beast XL data");
async function ready(page: Page, index: number) {
  await page.waitForFunction(index => {
    if (document.querySelector(".volume-render__overlay--error")) return true;
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(index)
      && host?.dataset.volumeMaterialCommitted === "true" && host.dataset.volumeQualityPreview === "false";
  }, index, { timeout: 180000 });
  expect(await page.locator(".volume-render__overlay--error").allTextContents()).toEqual([]);
}
async function open(page: Page) {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("/harness/index.html?realRunBridge=beast-xl&beastFields=1");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(1500, 850));
  await ready(page, 23);
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
}
test("early vorticity hides the background sheet and releases its GPU bricks without changing colors", async ({ page }, info) => {
  test.setTimeout(240000);
  await open(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(5));
  await ready(page, 1);
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([-72000, -45000, 18000], [0, 0, 8000], false));
  await ready(page, 1);
  const capture = () => page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const focused = (await capture())!;
  await page.getByRole("button", { name: "Raw field", exact: true }).click();
  await ready(page, 1);
  const raw = (await capture())!;
  expect(focused.evidence.descriptor.parameters!.colorRange).toEqual(raw.evidence.descriptor.parameters!.colorRange);
  expect(focused.evidence.descriptor.parameters!.scalarTransform).toBe("none");
  const cost = (value: typeof focused) => Number(value.evidence.descriptor.parameters!.logicalGpuUploadBytes);
  expect(cost(focused)).toBeLessThan(cost(raw) * 0.6);
  expect(Number(focused.evidence.descriptor.parameters!.backgroundCulledBricks)).toBeGreaterThan(0);
  expect(Number(focused.evidence.descriptor.parameters!.backgroundProfileGpuBytes)).toBeLessThan(65536);
  for (const [name, value] of [["focused", focused], ["raw", raw]] as const) {
    await writeFile(info.outputPath(`${name}.png`), Buffer.from(value.pngDataUri.split(",")[1], "base64"));
    await writeFile(info.outputPath(`${name}.json`), JSON.stringify(value.evidence, null, 2));
  }
  await page.getByRole("button", { name: "Structure", exact: true }).click();
  await ready(page, 1);
  const restored = (await capture())!;
  const comparison = await page.evaluate(([a, b]) => window.__nimbusVtkHarness!.compareAppearance(a, b), [focused.pngDataUri, restored.pngDataUri]);
  expect(comparison.rgbRmse).toBeLessThan(0.001);
  await info.attach("cost", { body: JSON.stringify({ focused: cost(focused), raw: cost(raw), profile: focused.evidence.descriptor.parameters!.backgroundProfileGpuBytes }), contentType: "application/json" });
});

for (const frame of [{ slot: 4, index: 0 }, { slot: 5, index: 1 }, { slot: 6, index: 3 }, { slot: 3, index: 24 }]) {
  test(`every selectable Beast volume renders or explains empty storm support at step ${frame.index}`, async ({ page }, info) => {
    test.setTimeout(600000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await open(page);
    await page.evaluate(slot => window.__nimbusVtkHarness!.setStep(slot), frame.slot);
    await ready(page, frame.index);
    const select = page.getByRole("combobox", { name: "Field", exact: true });
    const fields = await select.locator("option").evaluateAll(options => options.map(option => (option as HTMLOptionElement).value).filter(Boolean));
    expect(fields).toHaveLength(21);
    const evidence: object[] = [];
    for (const field of fields) {
      await select.selectOption(field);
      await ready(page, frame.index);
      await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([-72000, -45000, 18000], [0, 0, 8000], false));
      await ready(page, frame.index);
      const value = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
      expect(value.evidence.descriptor.frameIndex).toBe(frame.index);
      evidence.push({ field, parameters: value.evidence.descriptor.parameters, layers: value.evidence.layers.filter(layer => layer.visible).map(layer => ({ kind: layer.kind, pointCount: layer.pointCount })) });
      await writeFile(info.outputPath(`${field}.png`), Buffer.from(value.pngDataUri.split(",")[1], "base64"));
    }
    await writeFile(info.outputPath("catalog.json"), JSON.stringify(evidence, null, 2));
    expect(errors).toEqual([]);
  });
}

test("four focused volume panes commit complete frames and retire their resources", async ({ page }, info) => {
  test.setTimeout(480000);
  await page.setViewportSize({ width: 2560, height: 1440 });
  await page.goto("/harness/index.html?realRunBridge=beast-xl&beastFields=1&panes=4");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(2450, 1250));
  await ready(page, 23);
  const panes = page.locator(".volume-render");
  await expect(panes).toHaveCount(4);
  for (const [index, field] of ["vorticity_magnitude", "horizontal_wind_speed", "theta", "w"].entries()) {
    await panes.nth(index).locator(".plot3d-control-tab", { hasText: "Volume" }).click();
    await panes.nth(index).getByRole("combobox", { name: "Field", exact: true }).selectOption(field);
  }
  for (const frame of [{ slot: 5, index: 1 }, { slot: 6, index: 3 }, { slot: 3, index: 24 }]) {
    await page.evaluate(slot => window.__nimbusVtkHarness!.setStep(slot), frame.slot);
    await ready(page, frame.index);
    const committed = await page.locator(".plot3d-overlay-host").evaluateAll(hosts => hosts.every(host => (host as HTMLElement).dataset.volumeMaterialCommitted === "true"));
    expect(committed).toBe(true);
  }
  await page.screenshot({ path: info.outputPath("quad.png") });
  await page.evaluate(() => { window.__nimbusVtkHarness!.setMounted(false); window.__nimbusVtkHarness!.clearFieldCache(); });
  await expect.poll(() => page.evaluate(() => window.__nimbusVtkHarness!.snapshot().visualization.committedBytes), { timeout: 30000 }).toBe(0);
});
