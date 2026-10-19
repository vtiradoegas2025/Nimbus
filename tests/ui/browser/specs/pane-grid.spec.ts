import { expect, test, type Page } from "@playwright/test";

/*
 * Multi-pane canvas.
 *
 * A supercell is read by comparing fields — reflectivity against vorticity
 * against the cloud body — and flipping between tabs turns that into a memory
 * test. Panes put them side by side at the same frame.
 *
 * The risks worth pinning are structural rather than cosmetic: every pane holds
 * its own WebGL context, so the layout must not exceed the budget; the frame
 * bar must survive the extra chrome (it is the control that fell off the bottom
 * before the toolbelt existed); and a companion that fails must not take the
 * primary figure with it.
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

/** Pane layout lives in the header's View popover; open it if it is not already open. */
async function openViewMenu(page: Page): Promise<void> {
  const menu = page.locator(".plot-studio__view-menu");
  if (!(await menu.evaluate((element) => (element as HTMLDetailsElement).open))) {
    await menu.locator("summary").click();
  }
  await expect(viewPopover(page)).toBeVisible();
}

/** Pick a pane layout the way a user does. The menu closes itself on choice. */
async function chooseLayout(page: Page, layout: string): Promise<void> {
  await openViewMenu(page);
  await panes(page).getByRole("button", { name: layout, exact: true }).click();
}

const viewPopover = (page: Page) => page.locator(".plot-studio__view-popover");
const panes = (page: Page) => viewPopover(page).locator(".plot-studio__panes-options");
const cameraLock = (page: Page) => viewPopover(page).getByRole("button", { name: /Cameras/ });

for (const viewport of [
  { width: 2560, height: 1320 },
  { width: 1512, height: 982 },
  { width: 800, height: 560 },
]) {
  test(`mixed soundings preserve neighboring map width and aligned headers at ${viewport.width}`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    await openStudio(page);
    await page.getByRole("button", { name: "Soundings", exact: true }).click();
    await page.getByRole("button", { name: "Hodograph", exact: true }).click();
    await chooseLayout(page, "Side by side");
    const primary = page.locator(".plot-studio__pane--primary");
    const companion = page.locator(".plot-studio__pane--companion");
    const picker = companion.getByRole("combobox", { name: "Companion pane plot type" });
    await picker.selectOption("slice");
    await expect(companion.locator(".plot-canvas .gtitle")).toBeVisible();
    const mapWidth = await companion.locator(".plot-canvas").evaluate((element) => element.clientWidth);
    const paneWidth = await companion.evaluate((element) => element.clientWidth);
    expect(mapWidth).toBeGreaterThan(paneWidth - 60);

    const headers = await page.locator(".plot-studio__pane-head").evaluateAll((elements) =>
      elements.map((element) => {
        const rect = element.getBoundingClientRect();
        return { height: rect.height, bottom: rect.bottom, width: element.clientWidth, contentWidth: element.scrollWidth };
      }));
    expect(Math.abs(headers[0].height - headers[1].height)).toBeLessThanOrEqual(1);
    expect(Math.abs(headers[0].bottom - headers[1].bottom)).toBeLessThanOrEqual(1);
    for (const header of headers) expect(header.contentWidth).toBeLessThanOrEqual(header.width + 1);

    await page.getByRole("button", { name: "Plan views", exact: true }).click();
    await page.getByRole("button", { name: "Slice", exact: true }).click();
    await expect.poll(() => companion.locator(".plot-canvas").evaluate((element) => element.clientWidth)).toBe(mapWidth);
    await picker.selectOption("hodograph");
    await expect(companion.locator(".plot-canvas .gtitle")).toBeVisible();
    const primaryWidth = await primary.locator(".plot-canvas").evaluate((element) => element.clientWidth);
    expect(primaryWidth).toBeGreaterThan(paneWidth - 60);
    const soundingWidth = await companion.locator(".plot-body").evaluate((element) => element.clientWidth);
    expect(soundingWidth).toBeLessThanOrEqual(680);
    await companion.getByRole("button", { name: "Controls", exact: true }).click();
    await expect(companion.locator(".plot-controls")).toBeVisible();
    await companion.getByRole("button", { name: "Hide controls", exact: true }).click();
    const bar = await page.locator(".plot-studio__frame-bar").boundingBox();
    expect(bar!.y + bar!.height).toBeLessThanOrEqual(viewport.height + 1);
    await page.screenshot({ path: info.outputPath("mixed-panes.png") });
  });
}

/*
 * Leaving a split.
 *
 * The only way back to one figure was View -> Single: two clicks into a popover for the
 * action a user wants the moment a comparison is over.
 */
test("a split can be left without opening a menu", async ({ page }) => {
  await openStudio(page);
  const exit = page.getByRole("button", { name: "Exit split", exact: true });
  await expect(exit, "nothing to exit from a single figure").toHaveCount(0);

  await chooseLayout(page, "Quad");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(3);

  await expect(exit).toBeVisible();
  await exit.click();
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(0);
  await expect(exit, "the control retires with the split").toHaveCount(0);
});

test("Escape leaves a split", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Side by side");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(1);

  await page.keyboard.press("Escape");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(0);
});

/*
 * Field linking.
 *
 * Linking is about WHICH quantity every pane shows; the camera lock is about where it is seen
 * from. They are separate switches, and toggling one must not move the other.
 */
test("linking fields is independent of the camera lock", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Side by side");
  await openViewMenu(page);

  const link = viewPopover(page).getByRole("button", { name: /Fields/ });
  const before = await cameraLock(page).getAttribute("aria-pressed");
  await expect(link).toHaveAttribute("aria-pressed", "false");

  await link.click();
  await expect(link).toHaveAttribute("aria-pressed", "true");
  await expect(cameraLock(page), "the camera lock did not move")
    .toHaveAttribute("aria-pressed", before ?? "true");
});

test("field linking is offered only when there is another pane to link to", async ({ page }) => {
  await openStudio(page);
  await openViewMenu(page);
  await expect(viewPopover(page).getByRole("button", { name: /Fields/ })).toBeDisabled();

  await chooseLayout(page, "Side by side");
  await openViewMenu(page);
  await expect(viewPopover(page).getByRole("button", { name: /Fields/ })).toBeEnabled();
});

test("pane layouts add exactly the figures they name", async ({ page }) => {
  await openStudio(page);

  // Single is the default and adds no companions.
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(0);

  await chooseLayout(page, "Side by side");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(1);

  await chooseLayout(page, "Quad");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(3);

  await chooseLayout(page, "Single");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(0);
});

/*
 * The primary must occupy exactly ONE grid cell. A 2D plot renders
 * `.plot-controls` and `.plot-body` as siblings, and unwrapped those became two
 * grid items — which overflowed a 2x2 into an implicit third row and collapsed
 * both explicit rows to zero height, so every pane rendered at 0 px tall.
 */
test("the primary figure occupies exactly one grid cell", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Quad");

  await expect(page.locator(".plot-studio__pane--primary")).toHaveCount(1);
  const rows = await page.locator(".plot-studio__plot-area").evaluate(
    (element) => getComputedStyle(element).gridTemplateRows,
  );
  // Two explicit rows, both with real height — an implicit third row is the
  // signature of the overflow this guards.
  const tracks = rows.trim().split(/\s+/);
  expect(tracks).toHaveLength(2);
  for (const track of tracks) {
    expect(Number.parseFloat(track)).toBeGreaterThan(80);
  }
});

/*
 * The frame bar is shared: panes show the same instant, and scrubbing must stay
 * reachable with the extra chrome present. This is the control that used to be
 * pushed off the bottom edge, so the pane row is exactly the kind of addition
 * that could reintroduce it.
 */
test("the shared frame bar survives every pane layout", async ({ page }) => {
  await openStudio(page);
  const viewport = page.viewportSize();
  if (!viewport) throw new Error("no viewport");

  for (const layout of ["Side by side", "Stacked", "Quad", "Single"]) {
    await chooseLayout(page, layout);
    const bar = page.locator(".plot-studio__frame-bar");
    await expect(bar).toBeVisible();
    const box = await bar.boundingBox();
    if (!box) throw new Error(`frame bar not laid out in ${layout}`);
    expect(
      Math.round(box.y + box.height - viewport.height),
      `${layout} pushes the frame bar past the bottom`,
    ).toBeLessThanOrEqual(0);
  }
});

test("each pane lays out with real area and stays inside the window", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Quad");

  const viewport = page.viewportSize();
  if (!viewport) throw new Error("no viewport");
  const cells = page.locator(".plot-studio__pane--companion");
  await expect(cells).toHaveCount(3);

  for (let index = 0; index < 3; index++) {
    const box = await cells.nth(index).boundingBox();
    if (!box) throw new Error(`pane ${index} is not laid out`);
    // A pane collapsed to a sliver is indistinguishable from a broken one.
    expect(box.width, `pane ${index} width`).toBeGreaterThan(100);
    expect(box.height, `pane ${index} height`).toBeGreaterThan(80);
    expect(Math.round(box.y + box.height - viewport.height)).toBeLessThanOrEqual(1);
  }
});

/*
 * The pane picker must survive being used.
 *
 * It used to sit inside `.plot-studio__toolbelt`, which scrolls internally so the frame bar never
 * gets pushed off the window. Choosing a multi-pane layout grew the grid, shrank the toolbelt, and
 * scrolled the picker into its own overflow — so the one control that turns split screen off was
 * the one control you could no longer see. It now lives in a header popover, which cannot be
 * clipped that way; this asserts the property directly rather than trusting the new placement.
 * `toBeVisible` does not catch it: the element is displayed and has a box, it is simply clipped by
 * an ancestor. Comparing its box against every scrolling ancestor's is what catches it.
 */
test("the pane picker stays on screen in every layout it can select", async ({ page }) => {
  await openStudio(page);
  const viewport = page.viewportSize();
  if (!viewport) throw new Error("no viewport");

  for (const layout of ["Side by side", "Stacked", "Quad", "Single"]) {
    await chooseLayout(page, layout);
    await openViewMenu(page);
    const box = await panes(page).boundingBox();
    if (!box) throw new Error(`pane picker not laid out in ${layout}`);

    expect(box.height, `${layout} collapses the pane picker`).toBeGreaterThan(10);
    expect(box.y, `${layout} pushes the pane picker above the window`)
      .toBeGreaterThanOrEqual(0);
    expect(
      Math.round(box.y + box.height - viewport.height),
      `${layout} pushes the pane picker past the bottom`,
    ).toBeLessThanOrEqual(0);

    // Not clipped by any scrolling ancestor either.
    const clipped = await panes(page).evaluate((element) => {
      const own = element.getBoundingClientRect();
      for (let node = element.parentElement; node; node = node.parentElement) {
        const style = getComputedStyle(node);
        if (style.overflowY === "visible" && style.overflowX === "visible") continue;
        const box = node.getBoundingClientRect();
        if (own.bottom > box.bottom + 1 || own.top < box.top - 1) return node.className;
      }
      return null;
    });
    expect(clipped, `${layout} clips the pane picker inside an ancestor`).toBeNull();
  }
});

/*
 * The whole point of split screen is seeing figures at once. Every plot renders
 * a control row sized for a full-window figure, and in a quarter pane that row
 * wraps until it is taller than the figure — a 2x2 of toolbars with slivers of
 * storm beneath them. The controls float over the figure instead, so this pins
 * the thing that matters: the canvas gets the pane.
 */
test("a pane spends its height on the figure, not on its toolbar", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Quad");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(3);

  const panels = page.locator(".plot-studio__pane");
  const count = await panels.count();
  expect(count).toBe(4);

  for (let index = 0; index < count; index++) {
    const pane = panels.nth(index);
    const paneBox = await pane.boundingBox();
    if (!paneBox) throw new Error(`pane ${index} is not laid out`);

    const figure = pane.locator(".plot-body, .volume-render__canvas, canvas").first();
    if (await figure.count() === 0) continue;
    const figureBox = await figure.boundingBox();
    if (!figureBox) continue;

    expect(
      figureBox.height / paneBox.height,
      `pane ${index} gives the figure only ${Math.round(100 * figureBox.height / paneBox.height)}% of its height`,
    ).toBeGreaterThan(0.5);
  }
});

/*
 * Controls are reachable, per pane, without costing the figure its space.
 * Hiding them is only defensible if getting them back is obvious, and opening
 * one pane's controls must not disturb the three it is being compared against.
 */
test("a pane's controls open over that pane alone", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Quad");

  const first = page.locator(".plot-studio__pane--companion").first();
  const others = page.locator(".plot-studio__pane--controls");

  await expect(first.locator(".volume-render__controls, .plot-controls")).toBeHidden();
  await expect(others).toHaveCount(0);

  await first.getByRole("button", { name: "Controls" }).click();
  await expect(first.locator(".volume-render__controls, .plot-controls").first()).toBeVisible();
  // Exactly one pane opened.
  await expect(others).toHaveCount(1);

  await first.getByRole("button", { name: "Hide controls" }).click();
  await expect(others).toHaveCount(0);
});

/*
 * Locked cameras mean every pane is viewed from the same place, so a compass in
 * each is one reading repeated, each copy competing with that pane's colorbar
 * for the same corner. One pane owns it; unlocking makes the views independent
 * and every 3D pane earns one back.
 *
 * Deliberately the MIXED case: this spec opens on Plan views, so the primary is
 * a 2D slice that draws no compass at all while the companions are 3D. Handing
 * the compass to "the primary" therefore left a locked grid with no orientation
 * reading anywhere, which is why the grid nominates the first 3D pane instead.
 */
test("a locked grid shows exactly one compass, in a pane that can draw one", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Quad");
  const compasses = page.locator(".plot3d-compass");

  await openViewMenu(page);
  await expect(cameraLock(page)).toHaveText("Cameras locked");
  await expect(compasses).toHaveCount(1);
  // The 2D primary cannot draw one, so the owner must be a companion.
  await expect(page.locator(".plot-studio__pane--companion .plot3d-compass")).toHaveCount(1);

  // Unlocked, each pane has its own viewpoint — every 3D pane reports it. The
  // 2D primary still draws none, which is why this is three and not four.
  await cameraLock(page).click();
  await expect(compasses).toHaveCount(3);
});

test("a companion pane selects its own plot type", async ({ page }) => {
  await openStudio(page);
  await chooseLayout(page, "Side by side");

  const companion = page.locator(".plot-studio__pane--companion").first();
  const picker = companion.getByRole("combobox", { name: "Companion pane plot type" });
  await expect(picker).toBeVisible();

  // The primary stays on its own selection: panes compare fields, so changing
  // one must not change the other.
  await picker.selectOption("slice");
  await expect(picker).toHaveValue("slice");
  await expect(page.locator(".plot-studio__picker--nav")).toBeVisible();
});

/*
 * Camera lock is what makes a comparison honest — two storms at different
 * angles differ for reasons that have nothing to do with the data. It is only
 * meaningful once there is more than one pane, and the control must say so
 * rather than sitting there inert.
 */
test("camera lock is offered only when there is something to lock together", async ({ page }) => {
  await openStudio(page);
  await openViewMenu(page);
  const lock = cameraLock(page);

  await expect(lock).toBeDisabled();
  await expect(lock).toHaveAttribute("title", /applies once more than one pane/);

  await chooseLayout(page, "Side by side");
  await openViewMenu(page);
  await expect(lock).toBeEnabled();
  await expect(lock).toHaveText("Cameras locked");

  await lock.click();
  await expect(lock).toHaveText("Cameras free");
});

test("a cardinal camera command reaches a locked 3D companion", async ({ page }) => {
  await openStudio(page);
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__canvas canvas").first())
    .toBeVisible({ timeout: 30_000 });

  await chooseLayout(page, "Side by side");
  const primary = page.locator(".plot-studio__pane--primary");
  const companion = page.locator(".plot-studio__pane--companion").first();
  await expect(companion.locator(".volume-render__canvas canvas"))
    .toBeVisible({ timeout: 30_000 });

  // Locked panes expose one authoritative control surface. The companion may
  // finish its first data fit later; camera sync must retain and replay this
  // command after that fit rather than letting the follower overwrite it.
  await primary.getByRole("button", { name: "Controls", exact: true }).click();
  await primary.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await primary.getByRole("button", { name: "E", exact: true }).click();
  await expect(primary.locator(".plot3d-compass__azimuth")).toContainText("90°");

  await companion.getByRole("button", { name: "Controls", exact: true }).click();
  await companion.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  const verticalScale = companion
    .locator(".control-group", { hasText: "Vertical scale" })
    .locator("input[type=range]");
  await verticalScale.evaluate((element) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    )?.set;
    if (!setter) throw new Error("HTMLInputElement value setter unavailable");
    setter.call(input, "3");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(companion.locator(".control-group", { hasText: "Vertical scale" })
    .locator(".volume-render__slider-value")).toHaveText("3.0×");

  await openViewMenu(page);
  await cameraLock(page).click();
  await expect(page.locator(".plot3d-compass")).toHaveCount(2);
  await expect(companion.locator(".plot3d-compass__azimuth"))
    .toContainText("90°", { timeout: 10_000 });
  const authorityElevation = await primary.locator(".plot3d-compass__elevation").textContent();
  await expect(companion.locator(".plot3d-compass__elevation"))
    .toHaveText(authorityElevation ?? "", { timeout: 10_000 });
});

/*
 * Toggling split screen must not rebuild the figure.
 *
 * The primary figure used to be wrapped conditionally — bare children in single-pane, a pane shell
 * in a grid. Swapping the element type at that position makes React unmount the subtree and mount a
 * fresh one, so every toggle tore down the plot and built it again: a blank canvas, a re-read of the
 * field, and for a 3D scene a full vtk rebuild costing seconds. `toBeVisible` cannot see this,
 * because the rebuilt figure looks identical once it arrives.
 *
 * Tagging the live DOM node is what catches it: React reuses the node when it updates in place, and
 * replaces it (losing the tag) when it remounts.
 */
test("toggling split screen updates the figure in place instead of rebuilding it", async ({ page }) => {
  await openStudio(page);
  const canvas = page.locator(".plot-canvas").first();
  await expect(canvas).toBeVisible();

  const before = await page.locator(".plot-body").first().boundingBox();
  if (!before) throw new Error("figure not laid out in single-pane");

  await canvas.evaluate((element) => { element.setAttribute("data-nimbus-identity", "original"); });

  await chooseLayout(page, "Side by side");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(1);
  await expect(
    page.locator(".plot-studio__pane--primary .plot-canvas").first(),
    "entering split screen rebuilt the primary figure",
  ).toHaveAttribute("data-nimbus-identity", "original");

  await chooseLayout(page, "Single");
  await expect(page.locator(".plot-studio__pane--companion")).toHaveCount(0);
  await expect(
    page.locator(".plot-canvas").first(),
    "leaving split screen rebuilt the primary figure",
  ).toHaveAttribute("data-nimbus-identity", "original");

  // ...and the shell it is wrapped in must hand its layout straight back, so single-pane
  // geometry is byte-for-byte what it was before the shell existed.
  const shellDisplay = await page.locator(".plot-studio__pane-flat")
    .evaluate((element) => getComputedStyle(element).display);
  expect(shellDisplay, "the single-pane shell must generate no box").toBe("contents");

  const after = await page.locator(".plot-body").first().boundingBox();
  if (!after) throw new Error("figure not laid out after returning to single-pane");
  expect(Math.round(after.width), "single-pane width changed").toBe(Math.round(before.width));
  expect(Math.round(after.height), "single-pane height changed").toBe(Math.round(before.height));
});
