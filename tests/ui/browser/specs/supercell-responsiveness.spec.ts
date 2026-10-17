import { expect, test, type Page } from "@playwright/test";

/*
 * How long the Rendered supercell scene blocks the main thread.
 *
 * The cloud edge is a volume transfer update, while vortex extraction is
 * isolated in a latest-wins worker. These timings pin both the light path and
 * the control handoff into heavy geometry work.
 *
 * These measure the gap between a click and the browser being able to service
 * the next interaction, which is what "sluggish" actually means.
 */

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

/*
 * A wall-clock budget is only meaningful against a known renderer. The chromium
 * project pins swiftshader, so its timings are comparable run to run; WebKit
 * uses whatever GPU path the host offers, and a threshold tuned here would
 * either fail on a slow host or pass on a regression.
 */
/** A display-only control must not wait on a contour rebuild. */
const DISPLAY_CONTROL_BUDGET_MS = 400;

async function openSupercell(page: Page): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Rendered supercell", exact: true }).click();
  await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });
  // Poll for the first frame to settle so we time the interaction, not the load.
  await expect(async () => {
    expect(await page.locator(".volume-render__overlay").count()).toBe(0);
    expect(Number(await page.locator(".plot3d-overlay-host")
      .getAttribute("data-logical-gpu-upload-bytes"))).toBeGreaterThan(0);
  }).toPass({ timeout: 30_000 });
}

async function openControlGroup(page: Page, label: string): Promise<void> {
  const button = page.locator(".plot3d-control-tab", { hasText: label }).first();
  if (await button.getAttribute("aria-expanded") !== "true") await button.click();
  await expect(button).toHaveAttribute("aria-expanded", "true");
}

/**
 * Wall time from the click until the main thread can run a task again.
 *
 * A long synchronous filter inside the click handler delays the following
 * rAF + task, so this captures the freeze the user feels rather than the
 * handler's own return.
 */
async function blockingTimeMs(page: Page, click: () => Promise<void>): Promise<number> {
  await page.evaluate(() => {
    const timings = window as unknown as { __t0?: number };
    delete timings.__t0;
    document.addEventListener("click", () => {
      timings.__t0 = performance.now();
    }, { capture: true, once: true });
  });
  await click();
  return page.evaluate(async () => {
    await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
    const startedAt = (window as unknown as { __t0?: number }).__t0;
    if (startedAt == null) throw new Error("the timed click did not reach the page");
    return performance.now() - startedAt;
  });
}

function edgeButton(page: Page, label: string) {
  return page.locator(".control-group", { hasText: "Cloud edge" })
    .getByRole("button", { name: label, exact: true });
}

async function canvasSignal(page: Page): Promise<{ meanLuma: number; lumaDeviation: number }> {
  const png = await page.locator(".volume-render__canvas canvas").screenshot();
  return page.evaluate(async (base64) => {
    const image = new Image();
    image.src = `data:image/png;base64,${base64}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("2D screenshot context unavailable");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0;
    let sumSquares = 0;
    const count = pixels.length / 4;
    for (let index = 0; index < pixels.length; index += 4) {
      const luma = pixels[index]! * 0.2126
        + pixels[index + 1]! * 0.7152
        + pixels[index + 2]! * 0.0722;
      sum += luma;
      sumSquares += luma * luma;
    }
    const meanLuma = sum / count;
    return {
      meanLuma,
      lumaDeviation: Math.sqrt(Math.max(0, sumSquares / count - meanLuma * meanLuma)),
    };
  }, png.toString("base64"));
}

function expectVisibleSceneSignal(
  signal: { meanLuma: number; lumaDeviation: number },
  context: string,
): void {
  // The dark scene background is ~14 luma. A black/cleared WebGL frame has
  // neither a brighter mean nor spatial contrast; legitimate close cloud and
  // surfaces have at least one, including when the camera is inside the cloud.
  expect(
    signal.meanLuma > 20 || signal.lumaDeviation > 6,
    `${context}: canvas cleared (mean ${signal.meanLuma.toFixed(1)}, deviation ${signal.lumaDeviation.toFixed(1)})`,
  ).toBeTruthy();
}

async function setLayer(page: Page, group: string, label: string, on: boolean): Promise<void> {
  await openControlGroup(page, group);
  const toggle = page.locator(".control-group", { hasText: label }).getByRole("button").first();
  test.skip(await toggle.isDisabled(), `fixture cannot enable ${label}`);
  const active = (await toggle.getAttribute("class"))?.split(/\s+/).includes("pill--active") === true;
  if (active !== on) await toggle.click();
}

/*
 * Cloud edge changes only volume colour/opacity. It must never regain a
 * marching-cubes dependency or a translucent geometry actor.
 */
test("changing the cloud edge stays on the volume-transfer path", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "timing budget is calibrated against pinned SwiftShader");
  await openSupercell(page);
  await openControlGroup(page, "Cloud");
  await expect(edgeButton(page, "Full")).toBeVisible();

  const elapsed = await blockingTimeMs(page, () => edgeButton(page, "Full").click());
  expect(
    elapsed,
    `the cloud edge control blocked for ${Math.round(elapsed)}ms`,
  ).toBeLessThan(DISPLAY_CONTROL_BUDGET_MS);
});

test("turning the cloud edge off is immediate and stays off", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "timing budget is calibrated against pinned SwiftShader");
  await openSupercell(page);
  await openControlGroup(page, "Cloud");
  const elapsed = await blockingTimeMs(page, () => edgeButton(page, "Off").click());
  expect(
    elapsed,
    `turning the edge off blocked for ${Math.round(elapsed)}ms`,
  ).toBeLessThan(DISPLAY_CONTROL_BUDGET_MS);

  // The edge mode must remain selected while the floor reshapes the same transfer.
  const floor = page.locator(".control-group", { hasText: "Cloud floor" })
    .locator("input[type=range]");
  await floor.evaluate((element) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value",
    )?.set;
    if (!setter) throw new Error("HTMLInputElement value setter unavailable");
    setter.call(input, String(Number.parseFloat(input.value) + Number.parseFloat(input.step || "0.1")));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(edgeButton(page, "Off")).toHaveClass(/pill--active/);
});

/*
 * Every control in the row has to remain operable while the scene is mounted.
 * A layer toggle that takes seconds reads as a broken button, and the user
 * clicks it again, which queues another rebuild.
 */
test("layer toggles respond without a contour stall", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "timing budget is calibrated against pinned SwiftShader");
  await openSupercell(page);
  for (const [controlGroup, layer] of [
    ["Flow", "Updraft / downdraft"],
    ["Vortex", "Vortex core"],
  ] as const) {
    await openControlGroup(page, controlGroup);
    const toggle = page.locator(".control-group", { hasText: layer })
      .getByRole("button").first();
    if (await toggle.isDisabled()) continue;
    const elapsed = await blockingTimeMs(page, () => toggle.click());
    expect(
      elapsed,
      `the ${layer} toggle blocked for ${Math.round(elapsed)}ms`,
    ).toBeLessThan(DISPLAY_CONTROL_BUDGET_MS * 4);
  }
});

test("vortex core settles and permits continued frame playback", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openSupercell(page);
  await openControlGroup(page, "Vortex");

  const toggle = page.locator(".control-group", { hasText: "Vortex core" })
    .getByRole("button").first();
  test.skip(await toggle.isDisabled(), "fixture has no vortex diagnostic");
  await toggle.click();

  const scene = page.locator(".volume-render__canvas-wrap").first();
  await expect(scene).toHaveAttribute("data-vortex-core-enabled", "true");
  await expect(scene).toHaveAttribute("data-vortex-contour-loading", "false", {
    timeout: 30_000,
  });
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });

  const thresholdReadout = page.locator(".control-group", { hasText: "Core threshold" })
    .locator(".volume-render__slider-value")
    .first();
  const thresholdBeforePlayback = (await thresholdReadout.textContent())?.split(" · ")[0].trim();
  expect(thresholdBeforePlayback).toBeTruthy();

  const frame = page.getByRole("slider", { name: "Selected frame" });
  const initial = Number(await frame.inputValue());
  await page.getByRole("button", { name: "Play frames" }).click();
  await expect(scene).toHaveAttribute("data-synchronized-presentation", "true");
  await expect(page.locator(".plot-studio__frame-status")).toHaveText("· synchronized layers");
  await expect.poll(async () => Number(await frame.inputValue()), {
    timeout: 15_000,
    message: "vortex extraction held playback on its initial frame",
  }).not.toBe(initial);
  const firstAdvance = Number(await frame.inputValue());
  await expect.poll(async () => Number(await frame.inputValue()), {
    timeout: 15_000,
    message: "vortex playback advanced once but did not continue",
  }).not.toBe(firstAdvance);
  await expect(thresholdReadout).toContainText("fixed across playback");
  const thresholdDuringPlayback = (await thresholdReadout.textContent())?.split(" · ")[0].trim();
  expect(thresholdDuringPlayback).toBe(thresholdBeforePlayback);
  await page.getByRole("button", { name: "Pause frame playback" }).click();
  await expect(scene).toHaveAttribute("data-synchronized-presentation", "false");

  await expect(scene).toHaveAttribute("data-vortex-contour-loading", "false", {
    timeout: 30_000,
  });
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
  expect(pageErrors).toEqual([]);
});

test("condensation evidence is gated by the solved vortex and exposes a true vortex focus", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openSupercell(page);
  await setLayer(page, "Vortex", "Vortex core", true);
  await openControlGroup(page, "Vortex");

  const condensation = page.locator(".control-group", { hasText: "Tornadic condensation" })
    .getByRole("button").first();
  test.skip(await condensation.isDisabled(), "fixture has no solved condensate for vortex evidence");
  await condensation.click();

  const scene = page.locator(".volume-render__canvas-wrap").first();
  await expect(scene).toHaveAttribute("data-tornadic-condensation-enabled", "true");
  await expect(scene).toHaveAttribute("data-vortex-contour-loading", "false", {
    timeout: 30_000,
  });
  await expect(page.locator(".control-group", { hasText: "Condensation classification" }))
    .toBeVisible();

  const focus = page.getByRole("button", { name: "Focus vortex" });
  await expect(focus).toBeEnabled({ timeout: 30_000 });
  await focus.click();
  expectVisibleSceneSignal(await canvasSignal(page), "focused vortex and condensation evidence");
  expect(pageErrors).toEqual([]);
});

test("flow signed-extrema surfaces settle independently", async ({ page }) => {
  await openSupercell(page);
  await setLayer(page, "Flow", "Updraft / downdraft", true);
  const scene = page.locator(".volume-render__canvas-wrap").first();
  await expect(scene).toHaveAttribute("data-diagnostic-reveal-active", "true");
  await expect(scene).toHaveAttribute("data-flow-contour-loading", "false", {
    timeout: 30_000,
  });
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
});

test("diagnostic cloud reveal is automatic and user-overridable", async ({ page }) => {
  await openSupercell(page);
  await setLayer(page, "Flow", "Updraft / downdraft", true);
  const scene = page.locator(".volume-render__canvas-wrap").first();
  await expect(scene).toHaveAttribute("data-diagnostic-reveal-active", "true");

  await openControlGroup(page, "Cloud");
  const unchanged = page.locator(".control-group", { hasText: "Internal diagnostics" })
    .getByRole("button", { name: "Cloud unchanged", exact: true });
  await unchanged.click();
  await expect(scene).toHaveAttribute("data-diagnostic-reveal-active", "false");
  await expect(unchanged).toHaveClass(/pill--active/);
});

test("heavy 3D playback offers requested cadence independently of long-run frame stride", async ({ page }) => {
  await openSupercell(page);
  const speed = page.getByRole("combobox", { name: "Playback speed" });
  const stride = page.getByRole("combobox", { name: "Frame stride" });
  await expect(speed.locator("option")).toHaveCount(7);
  await expect(speed.locator("option").nth(0)).toHaveText("1 fps");
  await expect(speed.locator("option").nth(1)).toHaveText("2 fps");
  await speed.selectOption("60");
  await expect(speed).toHaveValue("60");
  await expect(stride).toBeVisible();
  await stride.selectOption("5");
  await expect(stride).toHaveValue("5");
});

test("every precip, radar, flow, and vortex combination holds the last complete image while settling", async ({ page }) => {
  await openSupercell(page);
  const scene = page.locator(".volume-render__canvas-wrap").first();
  expectVisibleSceneSignal(await canvasSignal(page), "cloud only");

  // Gray-code order changes one asynchronous layer at a time while visiting all
  // sixteen combinations. The immediate sample exercises the hold frame; the
  // settled sample exercises the newly committed combination.
  const transitions = [
    ["Precip", "Precipitation", true, "cloud + precip"],
    ["Radar", "Radar floor", true, "cloud + precip + radar"],
    ["Precip", "Precipitation", false, "cloud + radar"],
    ["Flow", "Updraft / downdraft", true, "cloud + radar + flow"],
    ["Precip", "Precipitation", true, "cloud + precip + radar + flow"],
    ["Radar", "Radar floor", false, "cloud + precip + flow"],
    ["Precip", "Precipitation", false, "cloud + flow"],
    ["Vortex", "Vortex core", true, "cloud + flow + vortex"],
    ["Precip", "Precipitation", true, "cloud + precip + flow + vortex"],
    ["Radar", "Radar floor", true, "cloud + precip + radar + flow + vortex"],
    ["Precip", "Precipitation", false, "cloud + radar + flow + vortex"],
    ["Flow", "Updraft / downdraft", false, "cloud + radar + vortex"],
    ["Precip", "Precipitation", true, "cloud + precip + radar + vortex"],
    ["Radar", "Radar floor", false, "cloud + precip + vortex"],
    ["Precip", "Precipitation", false, "cloud + vortex"],
  ] as const;

  for (const [group, layer, enabled, label] of transitions) {
    await setLayer(page, group, layer, enabled);
    expectVisibleSceneSignal(await canvasSignal(page), `${label} hold`);
    await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
      timeout: 30_000,
    });
    expectVisibleSceneSignal(await canvasSignal(page), `${label} settled`);
  }
});

test("precip, radar, flow, and vortex retain a visible coherent scene during fast scan", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openSupercell(page);
  const scene = page.locator(".volume-render__canvas-wrap").first();
  await setLayer(page, "Precip", "Precipitation", true);
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
  await setLayer(page, "Flow", "Updraft / downdraft", true);
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
  await setLayer(page, "Vortex", "Vortex core", true);
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
  await setLayer(page, "Radar", "Radar floor", true);

  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
  await expect(scene).toHaveAttribute("data-vortex-contour-loading", "false", {
    timeout: 30_000,
  });
  await expect(scene).toHaveAttribute("data-flow-contour-loading", "false", {
    timeout: 30_000,
  });
  expectVisibleSceneSignal(await canvasSignal(page), "settled all-layer frame");

  await page.getByRole("combobox", { name: "Playback speed" }).selectOption("2");
  await page.getByRole("combobox", { name: "Frame stride" }).selectOption("5");
  const frame = page.getByRole("slider", { name: "Selected frame" });
  const initial = Number(await frame.inputValue());
  await page.getByRole("button", { name: "Play frames" }).click();

  // Sample through both hold and commit periods. A pending frame must retain
  // the previous complete image instead of clearing or presenting a half-scene.
  const samples: Array<{ meanLuma: number; lumaDeviation: number }> = [];
  for (let sample = 0; sample < 18; sample++) {
    samples.push(await canvasSignal(page));
    await page.waitForTimeout(35);
  }
  await expect.poll(async () => Number(await frame.inputValue()), {
    timeout: 20_000,
    message: "all-layer playback never advanced",
  }).not.toBe(initial);
  await page.getByRole("button", { name: "Pause frame playback" }).click();
  await expect(scene).toHaveAttribute("data-scene-frame-commit-pending", "false", {
    timeout: 30_000,
  });
  samples.forEach((signal, index) => expectVisibleSceneSignal(signal, `all-layer sample ${index}`));
  expect(pageErrors).toEqual([]);
});

test("repeated close zoom never collapses the supercell camera to black", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openSupercell(page);
  const canvas = page.locator(".volume-render__canvas canvas");
  await canvas.hover();

  const samples: Array<{ meanLuma: number; lumaDeviation: number }> = [];
  for (let step = 0; step < 28; step++) {
    await page.mouse.wheel(0, -600);
    await page.waitForTimeout(25);
    samples.push(await canvasSignal(page));
  }
  samples.forEach((signal, index) => expectVisibleSceneSignal(signal, `close-zoom sample ${index}`));
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("volume sliders use interactive preview and settle back to full quality", async ({ page }) => {
  await openSupercell(page);
  await openControlGroup(page, "Cloud");
  const density = page.locator(".control-group", { hasText: "Cloud density" })
    .locator('input[type="range"]');
  await expect(density).toHaveAttribute("max", "2.5");

  await density.dispatchEvent("pointerdown");
  await expect(page.locator(".plot3d-control-notice")).toBeVisible();
  await density.evaluate((element) => {
    const input = element as HTMLInputElement;
    input.value = input.max;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await density.dispatchEvent("pointerup");
  await expect(page.locator(".plot3d-control-notice")).toBeHidden({ timeout: 2_000 });
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  await expect.poll(
    async () => (await canvasSignal(page)).lumaDeviation,
    { message: "maximum cloud density should retain spatial structure", timeout: 5_000 },
  ).toBeGreaterThan(6);
});
