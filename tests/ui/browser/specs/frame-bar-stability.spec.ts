import { expect, test, type Page } from "@playwright/test";

/*
 * The scrubber holds still while frames go past.
 *
 * The frame bar is one flex row: play, speed, "Frame", the range input, the
 * readout, record. The range input was the only flexible item in it, and the
 * readout beside it had a min-width but no maximum — so the readout's width was
 * subtracted from the scrubber track. That text changes constantly during
 * playback: the ordinal rolls 9 -> 10, the simulation time gains a digit, and
 * two status qualifiers appear and vanish as frames load ("· showing 12",
 * "· reading frame data", ~20 characters each). Every one of those changes
 * resized the track and moved the thumb under the user's cursor.
 *
 * This measures geometry rather than markup, because the defect was never
 * visible in the markup — the JSX was correct and the CSS was individually
 * reasonable. It was the composition of `flex: 1` on one item and unbounded
 * content on its neighbour that produced it.
 */

/*
 * Deliberately a 2D plan view rather than whatever the studio opens on. The
 * frame bar is plot-agnostic — it is chrome around the canvas, not part of it —
 * and driving playback against a 3D scene here would make every sample wait on
 * a full volume re-render, which under software rasterisation is minutes of
 * runtime measuring something this spec is not about.
 *
 * The live fixture publishes only its first few steps by default; the whole
 * run is published so the ordinal actually crosses a digit boundary, which is
 * one of the widths that used to move the track.
 */
async function openStudio(page: Page): Promise<number> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  // The control is installed by an effect, so it does not exist at `load`.
  // Publishing into an absent harness is a silent no-op.
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness?.publishLiveThrough(20));
  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await expect(page.locator(".plot-studio__frame-bar")).toBeVisible();

  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');
  // The live run is polled, so the step count climbs after the first render.
  await expect
    .poll(async () => Number(await slider.getAttribute("max")), { timeout: 30_000 })
    .toBeGreaterThan(9);
  return Number(await slider.getAttribute("max"));
}

/** Widths of the row's geometry, to sub-pixel precision. */
async function rowGeometry(page: Page): Promise<{ track: number; meta: number }> {
  const track = await page.locator('.plot-studio__frame-bar input[type="range"]').boundingBox();
  const meta = await page.locator(".plot-studio__frame-meta").boundingBox();
  if (!track || !meta) throw new Error("frame bar is not laid out");
  return { track: track.width, meta: meta.width };
}

const round = (value: number) => Math.round(value * 100) / 100;

test("the scrubber track does not resize as the selected frame changes", async ({ page }) => {
  const max = await openStudio(page);
  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');

  const tracks = new Set<number>();
  const metas = new Set<number>();

  // Frames chosen so the readout changes in every way that used to matter: the
  // ordinal crossing a digit boundary (9 -> 10) and the simulation time
  // gaining digits.
  for (const frame of [0, 5, 8, 9, 10, max]) {
    await slider.fill(String(frame));
    const { track, meta } = await rowGeometry(page);
    tracks.add(round(track));
    metas.add(round(meta));
  }

  expect(
    [...tracks],
    "scrubber track resized as frames changed",
  ).toHaveLength(1);
  expect([...metas], "readout resized as frames changed").toHaveLength(1);
});

test("the readout holds its width when a transient status appears", async ({ page }) => {
  await openStudio(page);
  const before = await rowGeometry(page);

  // Playback is what surfaces the status qualifiers, and it is also the case
  // the user reported: the flicker was continuous while frames advanced.
  await page.getByRole("button", { name: "Play frames" }).click();
  const during: Array<{ track: number; meta: number }> = [];
  for (let i = 0; i < 10; i++) {
    during.push(await rowGeometry(page));
    await page.waitForTimeout(100);
  }
  await page.getByRole("button", { name: "Pause frame playback" }).click();

  expect(
    [...new Set(during.map((d) => round(d.track)))],
    "scrubber track resized during playback",
  ).toHaveLength(1);
  expect([...new Set(during.map((d) => round(d.meta)))]).toHaveLength(1);
  expect(during[0].track).toBeCloseTo(before.track, 1);

  // The status is a real disclosure, so holding the layout still must not have
  // been achieved by dropping it: the element is always present to hold its
  // width, and the full reading stays available even if the box clips it.
  await expect(page.locator(".plot-studio__frame-status")).toHaveCount(1);
  expect(await page.locator(".plot-studio__frame-meta").getAttribute("title"))
    .toContain("selected");
});

test("the readout is padded to a width the run cannot change", async ({ page }) => {
  const max = await openStudio(page);
  const selection = page.locator(".plot-studio__frame-selection");
  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');

  // Frame 1 must already reserve the columns that the last frame needs, for
  // both the ordinal and the simulation time.
  await slider.fill("0");
  const first = (await selection.textContent()) ?? "";
  await slider.fill(String(max));
  const last = (await selection.textContent()) ?? "";

  expect(first.length).toBe(last.length);
  // Padding only works if the whitespace survives to layout; `white-space: pre`
  // is what preserves it, and its absence would silently undo this.
  expect(await selection.evaluate((el) => getComputedStyle(el).whiteSpace)).toBe("pre");
});

test("a slow 2D frame change stays in the chrome instead of covering the plot", async ({ page }) => {
  const max = await openStudio(page);
  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');
  await expect(page.locator(".plot-canvas").first()).toBeVisible();
  await expect(page.locator(".plot-studio__meta")).toContainText("showing t =");

  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(1_400));
  await page.evaluate(() => window.__nimbusStudioHarness?.clearFieldCache());
  const current = Number(await slider.inputValue());
  await slider.fill(String(current < max ? current + 1 : current - 1));

  // The retained figure remains usable while the compact header carries the
  // selected-vs-rendered distinction. Prefetch may already have satisfied the
  // request, in which case going directly to the rendered time is also valid.
  await expect(page.locator(".plot-studio__meta")).toContainText(/Loading step|loads|showing t =/);
  await expect(
    page.locator(".plot-body > .plot-overlay").filter({ hasText: /loading frame/i }),
  ).toHaveCount(0);
  await expect(page.locator(".plot-canvas").first()).toBeVisible();
});

test("a slow vector-frame change retains arrows without a plot loading cover", async ({ page }) => {
  const max = await openStudio(page);
  await page.getByRole("button", { name: "Vector field", exact: true }).click();
  await expect(page.locator(".plot-canvas").first()).toBeVisible({ timeout: 30_000 });

  // A stretched vector pane changes the apparent direction, so vector plots do
  // not expose the scalar plot's Fill option.
  await expect(page.locator(".plot-controls").getByRole("button", { name: "Fill", exact: true }))
    .toHaveCount(0);

  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(1_400));
  await page.evaluate(() => window.__nimbusStudioHarness?.clearFieldCache());
  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');
  const current = Number(await slider.inputValue());
  // Jump beyond the adjacent-frame prefetch window so the transition cannot
  // resolve from cache before the status assertion observes it.
  await slider.fill(String(current === max ? 0 : max));

  await expect(page.locator(".plot-studio__meta")).toContainText(/Loading step|loads/);
  await expect(
    page.locator(".plot-body > .plot-overlay").filter({ hasText: /loading|next plane/i }),
  ).toHaveCount(0);
  await expect(page.locator(".plot-canvas").first()).toBeVisible();
});

test("a slow 3D frame change stays in the chrome instead of covering the scene", async ({ page }) => {
  const max = await openStudio(page);
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".plot-studio__meta")).toContainText("showing t =", { timeout: 30_000 });

  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(1_400));
  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');
  const current = Number(await slider.inputValue());
  await slider.fill(String(current === max ? 0 : max));

  await expect(page.locator(".plot-studio__meta")).toContainText(/Loading step|loads/);
  await expect(
    page.locator(".volume-render__overlay").filter({ hasText: /loading frame/i }),
  ).toHaveCount(0);
  await expect(page.locator(".volume-render__canvas")).toBeVisible();
});

test("a slow streamline-frame change keeps direction and magnitude context visible", async ({ page }) => {
  const max = await openStudio(page);
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Streamlines", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".plot3d-colorbar__title")).toContainText("Speed", { timeout: 30_000 });
  await expect(page.locator(".plot3d-badge__sub")).toContainText("cones point with vector");

  await page.evaluate(() => window.__nimbusStudioHarness?.setFieldReadDelay(1_400));
  await page.evaluate(() => window.__nimbusStudioHarness?.clearFieldCache());
  const slider = page.locator('.plot-studio__frame-bar input[type="range"]');
  const current = Number(await slider.inputValue());
  await slider.fill(String(current === max ? 0 : max));

  await expect(page.locator(".plot-studio__meta")).toContainText(/Loading step|loads/);
  await expect(
    page.locator(".volume-render__overlay").filter({ hasText: /Loading u, v, w|Building vector grid/i }),
  ).toHaveCount(0);
  await expect(page.locator(".volume-render__canvas")).toBeVisible();
});
