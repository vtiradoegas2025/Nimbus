import { expect, test, type Page } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test.skip(!process.env.NIMBUS_BEAST_XL_REFERENCE_RUN, "requires the preserved Beast XL history");

async function ready(page: Page) {
  await page.waitForFunction(() => {
    if (document.querySelector(".volume-render__overlay--error")) return true;
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
    return document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false"
      && host?.dataset.volumeMaterialCommitted === "true" && host.dataset.volumeQualityPreview === "false";
  }, undefined, { timeout: 180_000 });
  expect(await page.locator(".volume-render__overlay--error").allTextContents()).toEqual([]);
}

async function open(page: Page, compact = true) {
  await page.goto(`/harness/index.html?realRunBridge=beast-xl&beastFields=1${compact ? "" : "&volumeAblation=uncompacted-bricks"}`);
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(1500, 850));
  await ready(page);
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
  await ready(page);
}

for (const frame of [{ slot: 0, index: 23 }, { slot: 3, index: 24 }]) test(`Beast theta_prime frame ${frame.index} preserves raw and structural appearance evidence`, async ({ page }, info) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await open(page);
  await page.evaluate(slot => window.__nimbusVtkHarness!.setStep(slot), frame.slot);
  await ready(page);
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("theta_prime");
  await ready(page);
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([85000, -1500, -3800], [0, 0, 8000], false));
  await ready(page);
  for (const mode of ["Structure", "Raw field"] as const) {
    await page.getByRole("button", { name: mode, exact: true }).click();
    await ready(page);
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    expect(capture).not.toBeNull();
    if (!capture) continue;
    expect(capture.evidence.descriptor.frameIndex).toBe(frame.index);
    expect(capture.evidence.descriptor.field).toBe("theta_prime");
    expect(capture.evidence.descriptor.parameters?.scalarTransform).toBe(mode === "Structure" ? "horizontal-level-anomaly" : "none");
    const range = capture.evidence.descriptor.parameters?.colorRange;
    expect(Array.isArray(range) && range.length === 2 && range.every(Number.isFinite)).toBe(true);
    if (Array.isArray(range)) {
      expect(Math.abs(Number(range[1])), "rare historical spikes must not suppress the maintained broad theta perturbation").toBeLessThan(10);
    }
    expect(capture.evidence.layers.some(layer => layer.visible && layer.kind === "volume" && (layer.pointCount ?? 0) > 0)).toBe(true);
    const label = mode === "Structure" ? "structure" : "raw";
    const path = info.outputPath(`theta-prime-${label}.png`);
    await writeFile(path, Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
    await writeFile(info.outputPath(`theta-prime-${label}.json`), JSON.stringify(capture.evidence, null, 2));
    await info.attach(`theta-prime-${label}`, { path, contentType: "image/png" });
  }
  expect(errors).toEqual([]);
});

test("Beast compaction preserves the image across camera-side changes and keeps orbit shading", async ({ page }, info) => {
  test.setTimeout(240_000);
  const references: string[] = [];
  for (const compact of [false, true]) {
    await open(page, compact);
    let index = 0;
    for (const side of [1, -1, 1]) {
      await page.evaluate(side => window.__nimbusVtkHarness!.setCamera([78000 * side, 33000 * side, 18000], [0, 0, 8000], false), side);
      await ready(page);
      const capture = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
      await writeFile(info.outputPath(`${compact ? "compact" : "separate"}-${index}.png`), Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
      if (!compact) references.push(capture.pngDataUri);
      else {
        const metrics = await page.evaluate(([a, b]) => window.__nimbusVtkHarness!.compareAppearance(a, b), [references[index], capture.pngDataUri]);
        await info.attach(`camera-${index}`, { body: JSON.stringify(metrics), contentType: "application/json" });
        expect(metrics.rgbRmse).toBeLessThan(0.004);
        expect(metrics.meanAbsoluteError).toBeLessThan(0.001);
      }
      index++;
    }
    const before = await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeLighting());
    expect(before.length).toBeGreaterThan(0);
    expect(before.every(value => value.shade && !value.flatPreview)).toBe(true);
    const box = (await page.locator(".volume-render__canvas canvas").boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.52);
    await expect(page.locator(".plot3d-overlay-host")).toHaveAttribute("data-volume-quality-preview", "true");
    expect(await page.evaluate(() => window.__nimbusVtkHarness!.inspectVolumeLighting())).toEqual(before);
    await page.mouse.up();
    await ready(page);
  }
});

for (const frame of [{ slot: 3, index: 24 }, { slot: 1, index: 2 }, { slot: 2, index: 5 }]) {
  test(`Beast reported fields remain usable at step ${frame.index}`, async ({ page }, info) => {
    test.setTimeout(480_000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await open(page);
    await page.evaluate(slot => window.__nimbusVtkHarness!.setStep(slot), frame.slot);
    await ready(page);
    await page.locator('.volume-render__slider[type="range"]').first().fill("2.5");
    for (const field of ["vorticity_magnitude", "reflectivity_dbz", "theta_prime", "p_prime", "qv", "p", "buoyancy", "w", "cloud_condensate", "dewpoint"]) {
      await page.getByRole("combobox", { name: "Field", exact: true }).selectOption(field);
      await ready(page);
      await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([85000, -1500, -3800], [0, 0, 8000], false));
      await ready(page);
      const capture = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
      expect(capture.evidence.descriptor.frameIndex).toBe(frame.index);
      expect(capture.evidence.layers.some(layer => layer.visible && layer.kind === "volume" && (layer.pointCount ?? 0) > 0)).toBe(true);
      if (field === "p_prime" || field === "theta_prime") expect(capture.evidence.descriptor.parameters?.scalarTransform).toBe("horizontal-level-anomaly");
      if (field === "reflectivity_dbz" && frame.index === 24) expect(capture.evidence.descriptor.parameters?.boundedOverviewFallback).toBe(true);
      await writeFile(info.outputPath(`${field}.png`), Buffer.from(capture.pngDataUri.split(",")[1], "base64"));
      await writeFile(info.outputPath(`${field}.json`), JSON.stringify(capture.evidence, null, 2));
      if (field === "p_prime" || field === "theta_prime") {
        await page.getByRole("button", { name: "Raw field", exact: true }).click();
        await ready(page);
        const raw = (await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance()))!;
        expect(raw.evidence.descriptor.parameters?.scalarTransform).toBe("none");
        await writeFile(info.outputPath(`${field}-raw.png`), Buffer.from(raw.pngDataUri.split(",")[1], "base64"));
        await page.getByRole("button", { name: "Structure", exact: true }).click();
        await ready(page);
      }
    }
    expect(errors).toEqual([]);
  });
}
