import { expect, test, type Page } from "@playwright/test";

/*
 * Immersive 3D: the render gets the window.
 *
 * A rendered storm is read geometrically, and the navigation chrome above the
 * canvas costs roughly 430 px of a 1000 px window, so a 3D scene opened at about
 * half the height it could use. Immersive collapses that chrome.
 *
 * The risk it carries is not layout, it is DISCLOSURE. Every 3D scene is
 * labelled — fidelity mode, domain extent and vertical exaggeration, colour
 * scale, axis rulers — and those labels are what stop a render from being read
 * as more (or less) than it is. A presentation mode that quietly dropped them
 * would turn an honest figure into a pretty one. They are drawn inside the
 * canvas by the plot itself precisely so that hiding chrome cannot hide them,
 * and this spec pins that.
 */

/** The harness mounts PlotStudio bare; the app shell gives it the viewport. */
const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openRenderedSupercell(page: Page): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Rendered supercell", exact: true }).click();
  await expect(page.locator(".volume-render__canvas-wrap")).toBeVisible();
}

/** Fidelity, pane layout and Immersive live in the header's View popover. */
async function openViewMenu(page: Page): Promise<void> {
  const menu = page.locator(".plot-studio__view-menu");
  if (!(await menu.evaluate((element) => (element as HTMLDetailsElement).open))) {
    await menu.locator("summary").click();
  }
  await expect(page.locator(".plot-studio__view-popover")).toBeVisible();
}

async function canvasHeight(page: Page): Promise<number> {
  const box = await page.locator(".volume-render__canvas-wrap").boundingBox();
  if (!box) throw new Error("3D canvas is not laid out");
  return Math.round(box.height);
}

test("immersive gives the render the window and keeps every disclosure", async ({ page }) => {
  await openRenderedSupercell(page);
  const framed = await canvasHeight(page);

  await openViewMenu(page);
  await page.getByRole("button", { name: "Immersive" }).click();
  const immersive = await canvasHeight(page);

  // The whole point: materially more canvas, not a few reclaimed pixels.
  expect(immersive).toBeGreaterThan(framed * 1.4);

  // Navigation chrome is gone...
  await expect(page.locator(".plot-studio__category-bar")).toBeHidden();
  await expect(page.locator(".plot-studio__picker--nav")).toBeHidden();
  await expect(page.locator(".plot-studio__toolbelt")).toBeHidden();
  // Fidelity and pane layout live in the header's View menu, which goes with the header.
  await expect(page.locator(".plot-studio__view-menu")).toBeHidden();

  // ...but nothing that says what the pixels MEAN is.
  await expect(page.locator(".plot3d-badge")).toBeVisible();
  await expect(page.locator(".plot3d-badge")).toContainText("km");
  // The fidelity mode is disclosed on canvas, so hiding its control row above
  // the plot does not hide which contract the render was produced under.
  await expect(page.locator(".plot3d-badge__mode")).toBeVisible();
  // Axis rulers survive: a storm without a height scale is a picture, not a
  // measurement.
  expect(await page.locator(".plot3d-axis-labels__item").count()).toBeGreaterThan(0);
  // A storm is read across time, so scrubbing stays reachable.
  await expect(page.locator(".plot-studio__frame-bar")).toBeVisible();

  /*
   * The scene's own controls stay reachable too. Hiding them outright made
   * immersive cost the user every layer toggle, field picker and threshold —
   * the controls that decide what the render is OF. They float over the canvas
   * instead of occupying layout, which is why the canvas assertion above still
   * holds; this pins that they are present, operable, and take no height.
   */
  const controls = page.locator(".volume-render__controls");
  await expect(controls).toBeVisible();
  await controls.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await expect(controls.getByRole("button", { name: "Reset camera" })).toBeEnabled();
  const controlsBox = await controls.boundingBox();
  const canvasBox = await page.locator(".volume-render__canvas-wrap").boundingBox();
  if (!controlsBox || !canvasBox) throw new Error("immersive controls are not laid out");
  // Overlapping the canvas is the proof they cost no layout height.
  expect(controlsBox.y).toBeGreaterThan(canvasBox.y);
  expect(controlsBox.y).toBeLessThan(canvasBox.y + canvasBox.height);

  // And there is always a visible way out, plus the key everyone tries first.
  await expect(page.locator(".plot-studio__immersive-exit")).toBeVisible();
  // Restore the disclosure state used for the framed baseline before comparing
  // layout heights; an open Scene panel legitimately consumes framed space.
  await controls.locator(".plot3d-control-tab", { hasText: "Scene" }).click();
  await page.keyboard.press("Escape");
  expect(await canvasHeight(page)).toBe(framed);
});

test("immersive is offered for 3D scenes only", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });

  // A Plotly figure already scales to its container and its axes live outside
  // the canvas, so collapsing the chrome around one buys nothing and would cost
  // the axis labels. The control is absent rather than disabled.
  await page.getByRole("button", { name: "Plan views", exact: true }).click();
  await openViewMenu(page);
  await expect(page.getByRole("button", { name: "Immersive" })).toHaveCount(0);

  await page.getByRole("button", { name: "3D", exact: true }).click();
  await openViewMenu(page);
  await expect(page.getByRole("button", { name: "Immersive" })).toBeVisible();
});

/*
 * There is deliberately no test for "switching category while immersive
 * restores the chrome": the category bar is hidden in immersive, so a user
 * cannot reach it without exiting first. That is the safety property, not a
 * gap. The `immersiveAvailable` guard in PlotStudio still covers the paths that
 * change the plot type WITHOUT the category bar — an undo of a plot-view
 * selection, or a dataset whose available fields no longer support the current
 * 3D renderer — which are reachable programmatically but not by clicking.
 */
