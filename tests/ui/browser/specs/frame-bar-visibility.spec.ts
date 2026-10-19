import { expect, test, type Page } from "@playwright/test";

/*
 * The frame bar is fully on screen, in every plot category, without immersive.
 *
 * The scrubber sat partly below the bottom edge of the window whenever the full
 * navigation chrome was present, and came back into view in immersive mode —
 * which is exactly the signature of a height that overflows its container
 * rather than of anything wrong with the bar itself. `.plot-studio` is
 * `overflow: hidden` with a definite `100vh` from `.app > *`, so whatever does
 * not fit is silently clipped off the bottom, and the frame bar is last in the
 * column.
 *
 * This measures the bar against the viewport rather than asserting on CSS,
 * because the defect is a composition of several individually reasonable rules
 * and any one of them could change without reintroducing it.
 */

/** The harness mounts PlotStudio bare; the app shell gives it the viewport.
 *  Mirrors `.app > *` in src/styles/App.css. */
const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openStudio(page: Page): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
}

/** How far the frame bar's bottom edge falls below the viewport, in px. */
async function clippedBelowViewport(page: Page): Promise<number> {
  const bar = page.locator(".plot-studio__frame-bar");
  await expect(bar).toBeVisible();
  const box = await bar.boundingBox();
  const viewport = page.viewportSize();
  if (!box || !viewport) throw new Error("frame bar is not laid out");
  return Math.round(box.y + box.height - viewport.height);
}

/*
 * Every per-frame category, because the amount of chrome above the canvas
 * differs between them — the picker wraps to a second row for some, and the
 * 3D categories add the fidelity row on top. The bar overflowed only once the
 * chrome was tall enough, so a single category would not have caught it.
 */
for (const category of ["Plan views", "Cross-sections", "Soundings", "3D"]) {
  test(`the frame bar is fully visible in ${category} without immersive`, async ({ page }) => {
    await openStudio(page);
    await page.getByRole("button", { name: category, exact: true }).click();
    await expect(page.locator(".plot-studio__frame-bar")).toBeVisible();

    const overflow = await clippedBelowViewport(page);
    expect(
      overflow,
      `frame bar extends ${overflow}px past the bottom of the window`,
    ).toBeLessThanOrEqual(0);
  });
}

/*
 * The scrubber is the control the user actually grabs, and a range input's
 * thumb can overflow the input's own box. Pin the interactive control, not
 * just its container.
 */
test("the scrubber control itself is fully within the window", async ({ page }) => {
  await openStudio(page);
  await page.getByRole("button", { name: "Plan views", exact: true }).click();

  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');
  await expect(slider).toBeVisible();
  const box = await slider.boundingBox();
  const viewport = page.viewportSize();
  if (!box || !viewport) throw new Error("scrubber is not laid out");

  expect(Math.round(box.y + box.height - viewport.height)).toBeLessThanOrEqual(0);
  expect(Math.round(box.y)).toBeGreaterThanOrEqual(0);
});

/*
 * Small windows are the case that actually bites, and it takes BOTH axes: a
 * narrow window wraps the picker to three rows and the header to two, and a
 * short one leaves no height to absorb them. 800x560 is the geometry the
 * original defect was measured at — the bar overhung the bottom edge by 44 px
 * there while 1280x620 was still fine, so a height-only case would not have
 * caught it.
 */
for (const [width, height] of [[1280, 620], [1024, 700], [900, 640], [800, 560]]) {
  test(`the frame bar survives a ${width}x${height} window`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await openStudio(page);
    await page.getByRole("button", { name: "Plan views", exact: true }).click();

    expect(await clippedBelowViewport(page)).toBeLessThanOrEqual(0);
  });
}

/*
 * Figures leave the app at publication resolution.
 *
 * The camera button wrote a PNG at the on-screen pixel size — around 100 DPI once placed in a
 * document, below what a journal or a projected talk needs. `scale` re-renders the vector scene at
 * the higher resolution rather than upsampling, and the added SVG button keeps type and axes as
 * paths for a typesetter. Both are asserted by downloading, because a mis-typed config silently
 * falls back to defaults rather than erroring.
 */
test("a figure exports at publication resolution, as raster and as vector", async ({ page }) => {
  test.setTimeout(90_000);
  await openStudio(page);
  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await expect(page.locator(".plot-canvas").first()).toBeVisible();

  const pngDownload = page.waitForEvent("download");
  await page.locator('[data-title="Download plot as a PNG"]').first().click();
  const png = await pngDownload;
  const pngPath = await png.path();
  const { readFileSync } = await import("node:fs");
  const bytes = readFileSync(pngPath);

  // PNG IHDR: width and height are big-endian uint32 at byte 16 and 20.
  const width = bytes.readUInt32BE(16);
  const height = bytes.readUInt32BE(20);
  const box = await page.locator(".plot-canvas").first().boundingBox();
  if (!box) throw new Error("figure not laid out");
  expect(width, "PNG is not exported above screen resolution")
    .toBeGreaterThan(Math.round(box.width) * 2);
  expect(height, "PNG is not exported above screen resolution")
    .toBeGreaterThan(Math.round(box.height) * 2);

  const svgDownload = page.waitForEvent("download");
  await page.locator('[data-title="Download as SVG (vector, for publication)"]').first().click();
  const svg = await svgDownload;
  expect(svg.suggestedFilename()).toMatch(/\.svg$/);
  const svgText = readFileSync(await svg.path(), "utf8");
  expect(svgText.startsWith("<svg"), "SVG export is not an SVG document").toBe(true);
  // Axis text must survive as text, not be flattened into the raster.
  expect(svgText).toContain("<text");
});
