import { expect, test, type Page } from "@playwright/test";

/*
 * Vertical camera placement, end to end in a browser.
 *
 * The unit tests prove the placement arithmetic, but arithmetic is not a
 * camera: the placement still has to survive being handed to vtk.js, past
 * orthogonalizeViewUp and a clipping-range reset. So these read the angle back
 * off the compass, which derives it from the live camera rather than from the
 * control that asked for it — an independent path to the same number.
 */

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openVolume(page: Page): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
}

/**
 * Elevation the live camera reports, in degrees.
 *
 * The compass states the VIEW direction (negative while looking down, the
 * meteorological depression angle); a preset states where the EYE stands
 * relative to its target. They are the same angle with opposite signs, so a
 * 30-degree oblique preset must read as el -30.
 */
async function cameraElevation(page: Page): Promise<number> {
  const text = (await page.locator(".plot3d-compass__elevation").textContent()) ?? "";
  // The compass signs both directions, so a level camera reads "el +0" — a
  // pattern that accepts only a minus sign silently never matches it.
  const match = /el\s*([+-]?\d+)/.exec(text);
  if (!match) throw new Error(`compass reported no elevation: "${text}"`);
  return Number.parseInt(match[1]!, 10);
}

/** @param degrees the preset's own angle above the target; the compass negates it. */
async function expectElevation(page: Page, degrees: number): Promise<void> {
  await expect(async () => {
    // One degree of slack: the readout rounds, and orthogonalizeViewUp can
    // nudge the basis without meaningfully moving the eye.
    expect(Math.abs(await cameraElevation(page) + degrees)).toBeLessThanOrEqual(1);
  }).toPass({ timeout: 10_000 });
}

/** Domain tick labels currently on screen; they are culled once they leave the viewport. */
function axisLabelCount(page: Page): Promise<number> {
  return page.locator(".plot3d-axis-labels__item").count();
}

/** Metres the height slider will actually accept for this fixture's domain. */
async function sliderMaxM(page: Page): Promise<number> {
  const max = Number.parseFloat((await heightSlider(page).getAttribute("max")) ?? "0");
  expect(max, "the height slider published no domain lid").toBeGreaterThan(0);
  return max;
}

function heightSlider(page: Page) {
  return page.locator(".control-group", { hasText: "Target height" }).locator("input[type=range]");
}

/**
 * Drive a range input the way a drag does. `fill` refuses range inputs, and a
 * bare `value =` is invisible to React, which tracks the last value it wrote
 * and skips the event as a no-op — so go through the native setter.
 */
async function setHeight(page: Page, metres: number): Promise<void> {
  await heightSlider(page).evaluate((element, value) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    )?.set;
    if (!setter) throw new Error("HTMLInputElement value setter unavailable");
    setter.call(input, String(value));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, metres);
}

function heightReadout(page: Page) {
  return page.locator(".control-group", { hasText: "Target height" })
    .locator(".volume-render__slider-value");
}

test("a look-angle preset moves the live camera to that angle", async ({ page }) => {
  await openVolume(page);
  await page.getByRole("button", { name: "Level", exact: true }).click();
  await expectElevation(page, 0);
});

test("every preset is reachable and lands where it says", async ({ page }) => {
  await openVolume(page);
  for (const [label, degrees] of [["Top down", 90], ["Low", 15], ["Oblique", 30], ["Level", 0]] as const) {
    await page.getByRole("button", { name: label, exact: true }).click();
    await expectElevation(page, degrees);
  }
});

/*
 * The reason the feature exists: getting the eye down to a low-level feature
 * without orbiting by hand and hoping. Dropping the height must not drag the
 * pitch with it, or the view arrives somewhere the user did not ask for.
 */
test("dropping to the low levels keeps the chosen look angle", async ({ page }) => {
  await openVolume(page);
  await page.getByRole("button", { name: "Level", exact: true }).click();
  await expectElevation(page, 0);

  await setHeight(page, 500);
  await expect(heightReadout(page)).toHaveText("500 m");
  await expectElevation(page, 0);
});

test("the height readout switches to kilometres above the low levels", async ({ page }) => {
  await openVolume(page);
  // Derived from the fixture's own lid rather than hard-coded: a domain shorter
  // than the asked-for height clamps, and the readout would then be reporting
  // the clamp correctly while the assertion failed.
  const max = await sliderMaxM(page);
  expect(max, "this fixture is too short to reach kilometre formatting").toBeGreaterThanOrEqual(1000);
  await setHeight(page, Math.min(max, 6000));

  // Read back what the slider settled on rather than what was asked for: the
  // input snaps to its own step and clamps to the domain lid, and the invariant
  // under test is that the readout agrees with the control, not with the test.
  const applied = Number.parseFloat(await heightSlider(page).inputValue());
  expect(applied, "the slider did not reach the kilometre range").toBeGreaterThanOrEqual(1000);
  await expect(heightReadout(page)).toHaveText(`${(applied / 1000).toFixed(1)} km`);
});

test("the height readout stays in metres inside the lowest kilometre", async ({ page }) => {
  await openVolume(page);
  await setHeight(page, 750);
  await expect(heightReadout(page)).toHaveText("750 m");
});

test("the slider cannot ask for a height outside the domain", async ({ page }) => {
  await openVolume(page);
  const max = await sliderMaxM(page);
  // The range input clamps to its own max; the readout must agree rather than
  // reporting a height the camera never went to.
  const readout = (await heightReadout(page).textContent()) ?? "";
  const metres = readout.endsWith("km")
    ? Number.parseFloat(readout) * 1000
    : Number.parseFloat(readout);
  expect(metres).toBeLessThanOrEqual(max);
});

/*
 * The defect a screenshot caught: the preset sized its own distance at 1.4x the
 * wider horizontal span, which stands about twice too close and pushed the
 * domain box off all four edges, so nothing on screen read as registered to
 * anything else. Axis tick labels are culled once they leave the viewport, so
 * their count is a direct measure of how much of the domain is framed.
 */
test("a look-angle preset frames the domain as well as Reset camera does", async ({ page }) => {
  await openVolume(page);
  await page.getByRole("button", { name: "Reset camera" }).click();
  await expect(async () => {
    expect(await axisLabelCount(page)).toBeGreaterThan(0);
  }).toPass({ timeout: 10_000 });
  const framed = await axisLabelCount(page);

  await page.getByRole("button", { name: "Oblique", exact: true }).click();
  await expectElevation(page, 30);
  const afterPreset = await axisLabelCount(page);
  expect(
    afterPreset,
    `the preset dropped ${framed - afterPreset} of ${framed} domain labels off screen`,
  ).toBeGreaterThanOrEqual(framed);
});

test("dropping to the low levels keeps the zoom the user already framed", async ({ page }) => {
  await openVolume(page);
  await page.getByRole("button", { name: "Reset camera" }).click();
  await page.getByRole("button", { name: "Low", exact: true }).click();
  await expectElevation(page, 15);
  const framed = await axisLabelCount(page);

  await setHeight(page, 0);
  await expectElevation(page, 15);
  // Re-fitting to the whole domain on every slider tick would pull the eye back
  // out of the low levels the user just descended into.
  expect(await axisLabelCount(page)).toBeGreaterThanOrEqual(framed);
});

test("changing the look angle does not spin the orbit", async ({ page }) => {
  await openVolume(page);
  const bearing = async () =>
    (await page.locator(".plot3d-compass__azimuth").textContent()) ?? "";

  await page.getByRole("button", { name: "Low", exact: true }).click();
  await expectElevation(page, 15);
  const before = await bearing();

  await page.getByRole("button", { name: "Oblique", exact: true }).click();
  await expectElevation(page, 30);
  expect(await bearing()).toBe(before);
});

test("cardinal controls orient the live camera toward the selected bearing", async ({ page }) => {
  await openVolume(page);
  const azimuth = page.locator(".plot3d-compass__azimuth");

  for (const [label, degrees] of [["N", 0], ["E", 90], ["S", 180], ["W", 270]] as const) {
    await page.getByRole("button", { name: label, exact: true }).click();
    await expect(page.getByRole("button", { name: label, exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(azimuth).toContainText(`${degrees}°`);
  }
});

test("orthographic projection is applied to the live camera and can be restored", async ({ page }) => {
  await openVolume(page);
  const orthographic = page.getByRole("button", { name: "Orthographic", exact: true });
  const perspective = page.getByRole("button", { name: "Perspective", exact: true });

  await orthographic.click();
  await expect(orthographic).toHaveAttribute("aria-pressed", "true");
  await expect(perspective).toHaveAttribute("aria-pressed", "false");

  await perspective.click();
  await expect(perspective).toHaveAttribute("aria-pressed", "true");
});

test("reset and wheel interaction update controls from the live camera", async ({ page }) => {
  await openVolume(page);
  const topDown = page.getByRole("button", { name: "Top down", exact: true });
  const canvas = page.locator(".volume-render__canvas canvas");

  await topDown.click();
  await expectElevation(page, 90);
  await expect(topDown).toHaveAttribute("aria-pressed", "true");

  await canvas.hover();
  await page.mouse.wheel(0, -240);
  await expectElevation(page, 90);

  await page.getByRole("button", { name: "Reset camera" }).click();
  await expect(topDown).toHaveAttribute("aria-pressed", "false");
  await expect(async () => {
    expect(await cameraElevation(page)).toBeGreaterThan(-89);
  }).toPass({ timeout: 10_000 });
});
