import { expect, test, type Page } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test.skip(!process.env.NIMBUS_BEAST_XL_REFERENCE_RUN, "requires preserved Beast XL data");

async function ready(page: Page, frame: number) {
  await page.waitForFunction(frame => {
    if (document.querySelector(".volume-render__overlay--error")) return true;
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === String(frame)
      && host?.dataset.volumeMaterialCommitted === "true" && host.dataset.volumeQualityPreview === "false";
  }, frame, { timeout: 180000 });
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

test("Beast uniform startup cannot dim later thermodynamic fields or amplify their weak background", async ({ page }, info) => {
  test.setTimeout(480000);
  await open(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(4));
  await ready(page, 0);
  const uniform = page.locator(".volume-render__overlay", { hasText: /is uniform in this frame/ });
  await expect(uniform).toBeVisible();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("theta");
  for (const frame of [{ slot: 4, index: 0 }, { slot: 5, index: 1 }, { slot: 6, index: 3 }, { slot: 0, index: 23 }]) {
    await page.evaluate(slot => window.__nimbusVtkHarness!.setStep(slot), frame.slot);
    await ready(page, frame.index);
    await expect(uniform).toBeHidden();
    await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([-72000, -45000, 18000], [0, 0, 8000], false));
    await ready(page, frame.index);
    const capture = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
    expect(capture.evidence.layers.some(layer => layer.visible && layer.kind === "volume" && (layer.pointCount ?? 0) > 0)).toBe(true);
    const range = capture.evidence.descriptor.parameters!.colorRange as number[];
    if (frame.index === 1) expect(range[1]).toBeGreaterThan(1);
    await writeFile(info.outputPath(`theta-${frame.index}.png`), Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
    await writeFile(info.outputPath(`theta-${frame.index}.json`), JSON.stringify(capture.evidence, null, 2));
  }
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(5));
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("theta_prime");
  await ready(page, 1);
  await expect(uniform).toBeHidden();
  const prime = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
  expect((prime.evidence.descriptor.parameters!.colorRange as number[])[1]).toBeGreaterThan(1);
  await writeFile(info.outputPath("theta-prime-1.png"), Buffer.from(prime.pngDataUri.split(",")[1], "base64"));
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("vorticity_magnitude");
  await ready(page, 1);
  const vorticity = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
  await writeFile(info.outputPath("vorticity-1.png"), Buffer.from(vorticity.pngDataUri.split(",")[1], "base64"));
});

test("Beast derived wind and dewpoint use bounded native dependency reads", async ({ page }, info) => {
  test.setTimeout(480000);
  const reads: { fieldName: string; request: { maxOutputBytes: number; maxSourceBytes: number } }[] = [];
  page.on("request", request => {
    if (request.url().includes("/__nimbus_real_run/field-window")) reads.push(request.postDataJSON());
  });
  await open(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setStep(3));
  await ready(page, 24);
  for (const field of ["horizontal_wind_speed", "dewpoint"]) {
    await page.getByRole("combobox", { name: "Field", exact: true }).selectOption(field);
    await ready(page, 24);
    const capture = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
    expect(capture.evidence.layers.some(layer => layer.visible && layer.kind === "volume" && (layer.pointCount ?? 0) > 0)).toBe(true);
    await writeFile(info.outputPath(`${field}.png`), Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
    await writeFile(info.outputPath(`${field}.json`), JSON.stringify(capture.evidence, null, 2));
  }
  expect(reads.some(read => read.fieldName === "u")).toBe(true);
  expect(reads.some(read => read.fieldName === "v")).toBe(true);
  expect(reads.some(read => read.fieldName === "qv")).toBe(true);
  expect(reads.some(read => read.fieldName === "p")).toBe(true);
  expect(reads.some(read => ["horizontal_wind_speed", "dewpoint"].includes(read.fieldName))).toBe(false);
  expect(reads.every(read => read.request.maxOutputBytes > 0 && read.request.maxOutputBytes <= 67108864)).toBe(true);
  await info.attach("native-dependency-requests", { body: JSON.stringify(reads), contentType: "application/json" });
});
