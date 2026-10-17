import { expect, test, type Page } from "@playwright/test";
import { writeFile } from "node:fs/promises";

async function openField(page: Page, field: string, bricked: boolean, compact = false) {
  if (await page.evaluate(() => Boolean(window.__nimbusVtkHarness))) {
    await page.evaluate((enabled) => window.__nimbusVtkHarness!.setBrickedVolume(enabled), bricked);
  } else {
    await page.goto(`/harness/index.html?derivedBricks=1${compact ? "" : "&volumeAblation=uncompacted-bricks"}${bricked ? "" : "&volumeBricks=off"}`);
  }
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === "0"
      && document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.volumeMaterialCommitted === "true";
  }, undefined, { timeout: 60_000 });
  const tab = page.locator(".plot3d-control-tab", { hasText: "Volume" });
  if (await tab.getAttribute("aria-expanded") !== "true") await tab.click();
  await page.locator('select[aria-label="Field"]').selectOption(field);
  await page.waitForFunction((field) => {
    if (document.querySelector(".volume-render__overlay--error")) return true;
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false"
      && host?.dataset.volumeMaterialCommitted === "true"
      && host.dataset.volumeQualityPreview === "false"
      && document.querySelector(".plot3d-colorbar__title")?.textContent?.includes(field);
  }, field, { timeout: 60_000 });
  expect(await page.locator(".volume-render__overlay--error").allTextContents()).toEqual([]);
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-volume-representation", bricked ? "progressive-bricks" : "full-domain");
}

for (const compact of [false, true]) test(`${compact ? "compacted" : "separate"} volume bricks preserve sheets and tubes from both camera directions`, async ({ page }, testInfo) => {
  test.setTimeout(8 * 60_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const field of ["dense_signed", "analytic_thin_sheet", "analytic_helical_tube"]) {
    const references: Array<NonNullable<Awaited<ReturnType<NonNullable<Window["__nimbusVtkHarness"]>["captureAppearance"]>>>> = [];
    for (const bricked of [false, true]) {
      await openField(page, field, bricked, compact);
      if (!bricked) await page.evaluate(() => window.__nimbusVtkHarness!.captureScalarReference());
      else {
        const values = await page.evaluate(() => window.__nimbusVtkHarness!.compareScalarReference());
        expect(values.checked).toBeGreaterThan(0);
        expect(values.maximumError).toBe(0);
      }
      const uploads = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads);
      let view = 0;
      for (const parallel of [true, false]) for (const side of [-1, 1]) {
        await page.evaluate(({ parallel, side }) => window.__nimbusVtkHarness!.setCamera(
          [2937.5 + side * 15000, 3062.5 - side * 12000, 9035], [2937.5, 3062.5, 1035], parallel,
        ), { parallel, side });
        await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.volumeQualityPreview === "false");
        const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
        expect(capture).not.toBeNull();
        if (!capture) throw new Error("Missing volume capture");
        const label = `${field}-${bricked ? "bricks" : "whole"}-${parallel ? "parallel" : "perspective"}-${side}`;
        const pngPath = testInfo.outputPath(`${label}.png`);
        await writeFile(pngPath, Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
        await testInfo.attach(label, { path: pngPath, contentType: "image/png" });
        await writeFile(testInfo.outputPath(`${label}.json`), JSON.stringify(capture.evidence, null, 2));
        if (!bricked) references.push(capture);
        else {
          expect(Number(capture.evidence.descriptor.parameters?.selectedBrickCount)).toBeGreaterThan(compact ? 0 : 1);
          expect(capture.evidence.descriptor.parameters?.volumeHaloSamples).toBe(2);
          expect(capture.evidence.descriptor.parameters?.volumeOrdering).toBe("spatial-partition");
          expect(capture.evidence.camera.position).toEqual(references[view].evidence.camera.position);
          expect(capture.evidence.camera.focalPoint).toEqual(references[view].evidence.camera.focalPoint);
          expect(capture.evidence.camera.parallelProjection).toBe(parallel);
          const metrics = await page.evaluate(([left, right]) => window.__nimbusVtkHarness!.compareAppearance(left, right),
            [references[view].pngDataUri, capture.pngDataUri] as const);
          await testInfo.attach(`${label}-metrics`, { body: JSON.stringify(metrics), contentType: "application/json" });
          console.log(`VOLUME_BOUNDARY ${label} ${JSON.stringify(metrics)}`);
          expect(metrics.rgbRmse, label).toBeLessThan(field === "analytic_helical_tube" ? 0.002 : 0.004);
          expect(metrics.meanAbsoluteError, label).toBeLessThan(0.001);
        }
        view++;
      }
      expect(await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads)).toBe(uploads);
    }
  }
  expect(errors).toEqual([]);
});

test("volume preview restores the same optical density without uploading scalar textures", async ({ page }, testInfo) => {
  await openField(page, "dense_signed", true);
  const pose = () => page.evaluate(() => window.__nimbusVtkHarness!.setCamera(
    [17937.5, -8937.5, 9035], [2937.5, 3062.5, 1035], true,
  ));
  await pose();
  const first = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  const canvas = page.locator(".volume-render__canvas canvas");
  const box = await canvas.boundingBox();
  if (!box || !first) throw new Error("Missing volume fixture");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-volume-quality-preview", "true");
  await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  const preview = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  const targetBytes = (preview.canvasWidth ?? 0) * (preview.canvasHeight ?? 0) * 4;
  expect(preview.gpuLogicalTextureBytes - before.gpuLogicalTextureBytes,
    "volume actors must share one preview color target").toBeLessThanOrEqual(targetBytes);
  await page.mouse.up();
  await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-volume-quality-preview", "false");
  await pose();
  const last = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  if (!last) throw new Error("Missing settled volume");
  expect(last.evidence.descriptor.parameters?.interactionQuality).toBe("settled");
  const metrics = await page.evaluate(([left, right]) => window.__nimbusVtkHarness!.compareAppearance(left, right),
    [first.pngDataUri, last.pngDataUri] as const);
  await testInfo.attach("restored-optical-density", { body: JSON.stringify(metrics), contentType: "application/json" });
  expect(metrics.rgbRmse).toBeLessThan(0.001);
  expect(await page.evaluate(() => window.__nimbusVtkHarness!.snapshot().texture3DUploads)).toBe(before.texture3DUploads);
});

test.describe("shared-plane rays", () => {
  test.use({ deviceScaleFactor: 1 });
  for (const side of [-1, 1]) test(`parallel rays on an internal face have one optical owner from side ${side}`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const captures: string[] = [];
    for (const bricked of [false, true]) {
      await openField(page, "analytic_smooth_gradient", bricked);
      await page.getByRole("button", { name: "Raw field", exact: true }).click();
      await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      await page.evaluate(() => window.__nimbusVtkHarness!.setSize(961, 621));
      await page.locator('.plot3d-control-panel input[type="range"]').first().evaluate((element: HTMLInputElement) => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(element, "1");
        element.dispatchEvent(new Event("input", { bubbles: true }));
      });
      await page.waitForFunction(() => document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset.volumeQualityPreview === "false"
        && document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
      await page.evaluate((side) => window.__nimbusVtkHarness!.setCamera(
        [63.5 * 5875 / 95, 3062.5 + side * 15000, 1035], [63.5 * 5875 / 95, 3062.5, 1035], true,
      ), side);
      const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
      if (!capture) throw new Error("Missing shared-plane capture");
      expect(capture.evidence.descriptor.parameters?.scalarTransform).toBe("none");
      await writeFile(testInfo.outputPath(`${bricked ? "bricks" : "whole"}.json`), JSON.stringify(capture.evidence, null, 2));
      captures.push(capture.pngDataUri);
      await writeFile(testInfo.outputPath(`${bricked ? "bricks" : "whole"}.png`), Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
    }
    const metrics = await page.evaluate(async ([a, b]) => {
      const pixels = async (uri: string) => {
        const img = new Image(); img.src = uri; await img.decode();
        const canvas = document.createElement("canvas"); canvas.width = img.width; canvas.height = img.height;
        const ctx = canvas.getContext("2d")!; ctx.drawImage(img, 0, 0);
        return { width: img.width, height: img.height, data: ctx.getImageData(0, 0, img.width, img.height).data };
      };
      const left = await pixels(a), right = await pixels(b);
      let square = 0, count = 0, signal = 0;
      for (let y = Math.floor(left.height * 0.42); y < left.height * 0.58; y++) {
        const index = (y * left.width + Math.floor(left.width / 2)) * 4;
        for (let channel = 0; channel < 3; channel++) {
          square += ((left.data[index + channel] - right.data[index + channel]) / 255) ** 2;
          signal += left.data[index + channel] / 255; count++;
        }
      }
      return { width: left.width, height: left.height, centerRmse: Math.sqrt(square / count), signal: signal / count };
    }, captures);
    await testInfo.attach("shared-plane-metrics", { body: JSON.stringify(metrics), contentType: "application/json" });
    console.log("SHARED_PLANE", JSON.stringify(metrics));
    expect(metrics.width % 2).toBe(1);
    expect(metrics.signal).toBeGreaterThan(0.08);
    expect(metrics.centerRmse).toBeLessThan(0.003);
    expect(errors).toEqual([]);
  });
});
