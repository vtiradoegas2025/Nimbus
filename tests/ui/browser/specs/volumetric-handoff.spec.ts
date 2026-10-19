import { expect, test, type Page } from "@playwright/test";

/*
 * Taking the 2-D scalar on screen into a volume render.
 *
 * Switching to the 3-D tab used to drop the field the user was reading and open the volume
 * renderer on its own default, so the two views showed different quantities. These tests pin
 * the handoff end to end: the field actually arrives in the companion pane, the offer is
 * withheld from fields that have no vertical extent, and a second handoff overrides the first.
 */

/** The harness mounts PlotStudio bare; the app shell gives it the viewport. */
const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openStudio(page: Page): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await expect(page.locator(".plot-studio__view-menu summary")).toBeVisible();
}

/*
 * The control the user reaches for, in the figure's own control row beside the field it
 * acts on. It was briefly in the View popover, where it was two clicks deep and missed.
 */
const handoffButton = (page: Page) => page.locator(".plot-handoff-btn").first();

/*
 * The primary figure's field picker. Its accessible name is the stable contract;
 * nested-domain controls may appear before it and must not be mistaken for a field.
 */
const primaryFieldSelect = (page: Page) =>
  page.locator(".plot-studio__plot-area select[aria-label='Field']").first();

/** The companion's field picker, excluding the pane-type chooser in its header. */
const companionFieldSelect = (page: Page) =>
  page.locator(".plot-studio__pane--companion select[aria-label='Field']").first();

async function revealCompanionControls(page: Page): Promise<void> {
  const companion = page.locator(".plot-studio__pane--companion").first();
  const toggle = companion.locator(".plot-studio__pane-controls-toggle");
  if (await toggle.getAttribute("aria-expanded") === "false") await toggle.click();
  const volumeGroup = companion.locator(".plot3d-control-tab", { hasText: "Volume" }).first();
  if (await volumeGroup.getAttribute("aria-expanded") !== "true") await volumeGroup.click();
}

/*
 * Once split, each pane hides its controls behind its own Controls button — panes are small
 * and the figure gets the space. Reveal the primary's before reaching for its field picker.
 */
async function revealPrimaryControls(page: Page): Promise<void> {
  const primary = page.locator(".plot-studio__pane:not(.plot-studio__pane--companion)").first();
  // Single-pane: the primary renders flat with no pane header, and its controls are always shown.
  if ((await primary.count()) === 0) return;
  const toggle = primary.locator(".plot-studio__pane-controls-toggle");
  if ((await toggle.getAttribute("aria-expanded")) === "false") await toggle.click();
}

async function selectPrimaryField(page: Page, field: string): Promise<void> {
  await revealPrimaryControls(page);
  await primaryFieldSelect(page).selectOption(field);
  await expect(primaryFieldSelect(page)).toHaveValue(field);
}

test("the field on screen is what arrives in the 3D pane", async ({ page }) => {
  test.setTimeout(90_000);
  await openStudio(page);
  await selectPrimaryField(page, "theta_prime");

  // No menu to open: it sits in the control row, beside the field picker it acts on.
  await expect(handoffButton(page)).toBeVisible();
  await expect(handoffButton(page)).toBeEnabled();
  await handoffButton(page).click();

  // It lands beside the 2D view rather than replacing it, so the slice stays readable.
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(1);
  await expect(primaryFieldSelect(page)).toHaveValue("theta_prime");

  // The point of the whole feature: the volume renderer opens on the SAME field.
  await revealCompanionControls(page);
  await expect(companionFieldSelect(page)).toHaveValue("theta_prime", { timeout: 30_000 });
  const companion = page.locator(".plot-studio__pane--companion").first();
  await expect(companion.locator(".plot3d-badge__mode")).toContainText("Scientific");
  await expect(companion.locator(".plot3d-badge__sub"))
    .not.toContainText("reconstruction", { timeout: 30_000 });
});

test("a second handoff overrides the first", async ({ page }) => {
  /*
   * The volume renderer re-applies a preset only when its id changes. A fixed id would
   * leave this second handoff showing theta_prime while the slice showed w — the two panes
   * silently disagreeing, which reads as a working feature.
   */
  test.setTimeout(90_000);
  await openStudio(page);
  await selectPrimaryField(page, "theta_prime");
  await handoffButton(page).click();
  await revealCompanionControls(page);
  await expect(companionFieldSelect(page)).toHaveValue("theta_prime", { timeout: 30_000 });

  await selectPrimaryField(page, "w");
  await handoffButton(page).click();
  await expect(companionFieldSelect(page)).toHaveValue("w", { timeout: 30_000 });
});

test("the selected source plane becomes the 3D physical focus", async ({ page }) => {
  test.setTimeout(90_000);
  await openStudio(page);
  await selectPrimaryField(page, "theta_prime");
  const indexGroup = page.locator(".control-group").filter({
    has: page.locator(".control-group__label", { hasText: "Index" }),
  }).first();
  const indexSlider = indexGroup.locator('input[type="range"]');
  await indexSlider.evaluate((element) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    )?.set;
    if (!setter) throw new Error("HTMLInputElement value setter unavailable");
    setter.call(input, "11");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(indexSlider).toHaveValue("11");
  await handoffButton(page).click();
  await revealCompanionControls(page);

  const companion = page.locator(".plot-studio__pane--companion").first();
  const scene = companion.locator(".plot3d-overlay-host");
  await expect(scene).toHaveAttribute("data-source-plane-axis", "2");
  await expect(scene).toHaveAttribute("data-source-plane-index", "11");
  await expect(companion.locator(".plot3d-badge__sub"))
    .toContainText("focused from source axis 2 index 11");
});

test("a single-level field is refused, and the reason names it", async ({ page }) => {
  /*
   * composite_reflectivity is a column maximum: rank 2, already a reduction. Rendering it
   * as a volume would present a flat sheet as storm structure, so the offer is withheld —
   * visibly, with an explanation, rather than by silently vanishing.
   */
  await openStudio(page);
  await selectPrimaryField(page, "composite_reflectivity");

  await expect(handoffButton(page)).toBeVisible();
  await expect(handoffButton(page)).toBeDisabled();
  await expect(handoffButton(page)).toHaveAttribute("title", /composite_reflectivity/);
  await expect(handoffButton(page)).toHaveAttribute("title", /single level/);
});

/*
 * The 3-D field menus offer only what a 3-D renderer can draw.
 *
 * Both widgets listed every published field, so single-level products — column maxima,
 * surface diagnostics, sounding indices — sat in the menu and reported a shape error only
 * after the user picked one and the load came back. On a beast_xl run that was 16 of 37
 * fields. The fixture's composite_reflectivity is rank 2 for exactly this case.
 */
for (const renderer of ["Volume", "Isosurface"]) {
  test(`the ${renderer} field menu excludes single-level products`, async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
    await page.addStyleTag({ content: APP_SHELL });
    await page.getByRole("button", { name: "3D", exact: true }).click();
    await page.getByRole("button", { name: renderer, exact: true }).click();
    await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });
    await page.locator(".plot3d-control-tab", {
      hasText: renderer === "Volume" ? "Volume" : "Data",
    }).click();

    const options = await page.locator(".control-group", { hasText: "Field" })
      .locator("select").first().locator("option")
      .evaluateAll((nodes) => nodes.map((n) => (n as HTMLOptionElement).value).filter(Boolean));

    expect(options, "the volume field is offered").toContain("reflectivity_dbz");
    expect(options, "the 3D vertical velocity is offered").toContain("w");
    expect(options, "the column maximum is not a volume").not.toContain("composite_reflectivity");
  });
}

test("reducers do not offer the handoff at all", async ({ page }) => {
  /*
   * A vertical profile collapsed the horizontal dimensions on purpose, so it has no
   * volume form. A permanently disabled control there would be noise rather than
   * information, so the offer is absent entirely.
   */
  await openStudio(page);
  await page.getByRole("button", { name: "Soundings", exact: true }).click();
  await page.getByRole("button", { name: "Vertical", exact: true }).click();

  await expect(handoffButton(page)).toHaveCount(0);
});
