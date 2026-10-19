import { expect, test, type Page } from "@playwright/test";

/*
 * The NWS dBZ ramp on the 3D tab.
 *
 * That ramp's stops are anchored at physical reflectivity values over a fixed
 * -30..75 window, so pointed at vorticity the data lands below its opacity
 * floor and the volume renders completely empty. The 3D tab defaults to
 * vorticity. So "3D NWS colour doesn't work" was real, and it worked from the
 * Radar tab only because that tab's preset sets the FIELD as well as the ramp.
 *
 * The first fix greyed the button out, which made the report literally true —
 * the control was disabled in exactly the situation a user reaches for it, and
 * the tooltip explaining why is invisible on a touch screen and to anyone who
 * does not think to hover a dead button.
 *
 * Asking for the NWS ramp on a 3D scene is unambiguous: it means "show me
 * reflectivity". So it now selects the field the ramp describes, which is the
 * same pairing the Radar preset applies.
 */

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openVolume(page: Page): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Volume" }).click();
}

const fieldSelect = (page: Page) =>
  page.locator(".control-group", { hasText: "Field" }).locator("select").first();

const dbzPill = (page: Page) =>
  page.locator(".control-group", { hasText: "Colormap" })
    .getByRole("button", { name: "NWS dBZ", exact: true });

test("the dBZ colormap is clickable on a 3D volume", async ({ page }) => {
  await openVolume(page);
  await expect(dbzPill(page)).toBeEnabled();
});

test("picking dBZ selects the field that ramp describes", async ({ page }) => {
  await openVolume(page);

  const before = await fieldSelect(page).inputValue();
  expect(before, "the 3D tab does not default to reflectivity — that is the whole problem")
    .not.toContain("reflectiv");

  await dbzPill(page).click();

  await expect(dbzPill(page)).toHaveClass(/pill--active/);
  await expect(fieldSelect(page)).toHaveValue("reflectivity_dbz");
});

/*
 * `composite_reflectivity` also satisfies "is a reflectivity field" and sorts
 * earlier alphabetically, but it is a single plane. Selecting it would hand the
 * volume renderer a rank-2 array and fail the rank check — a first-match scan
 * of the field list does exactly that.
 */
test("the field chosen for dBZ is volumetric, not the 2-D composite", async ({ page }) => {
  await openVolume(page);
  await dbzPill(page).click();
  await expect(fieldSelect(page)).not.toHaveValue("composite_reflectivity");
});

test("choosing dBZ on a field that already is reflectivity leaves it alone", async ({ page }) => {
  await openVolume(page);
  await fieldSelect(page).selectOption("reflectivity_dbz");
  await dbzPill(page).click();
  await expect(fieldSelect(page)).toHaveValue("reflectivity_dbz");
});

/*
 * Switching away must not drag the field back. The pairing is a convenience on
 * the way in, not a lock: a user comparing ramps on reflectivity keeps their
 * field when they try Viridis.
 */
test("switching off dBZ keeps the field", async ({ page }) => {
  await openVolume(page);
  await dbzPill(page).click();
  await expect(fieldSelect(page)).toHaveValue("reflectivity_dbz");

  await page.locator(".control-group", { hasText: "Colormap" })
    .getByRole("button", { name: "Inferno", exact: true }).click();
  await expect(fieldSelect(page)).toHaveValue("reflectivity_dbz");
});
