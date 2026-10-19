import { expect, test, type Page } from "@playwright/test";

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function open3D(page: Page, plot: string): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: plot, exact: true }).click();
  await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });
  if (plot === "Rendered supercell") {
    await page.waitForFunction(() => {
      const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
      return data?.sceneFrameCommitPending === "false" && data.scenePresentationPending === "false";
    }, undefined, { timeout: 30_000 });
  }
}

async function openDirect3D(page: Page, plot: "isosurface" | "streamlines"): Promise<void> {
  await page.goto("/harness/index.html", { waitUntil: "load" });
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate((next) => window.__nimbusVtkHarness?.setPlot(next), plot);
  await expect(page.locator(".harness-shell")).toHaveAttribute("data-plot", plot);
  await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });
}

function group(page: Page, label: string) {
  return page.locator(".control-group").filter({
    has: page.locator(".control-group__label", { hasText: label }),
  }).first();
}

async function open3DGroup(page: Page, label: string): Promise<void> {
  const button = page.locator(".plot3d-control-tab", { hasText: label }).first();
  await button.click();
  await expect(button).toHaveAttribute("aria-expanded", "true");
}

test("a custom 3D background drives the canvas host and survives renderer changes", async ({ page }) => {
  await open3D(page, "Volume");
  await page.getByRole("button", { name: "All 3D", exact: true }).click();

  await open3DGroup(page, "Scene");
  await group(page, "3D background").locator("select").selectOption("custom");
  await group(page, "3D background").locator('input[type="color"]').fill("#31506f");
  const verticalScale = group(page, "Vertical scale").locator('input[type="range"]');
  await verticalScale.evaluate((element) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    )?.set;
    if (!setter) throw new Error("HTMLInputElement value setter unavailable");
    setter.call(input, "2");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.locator(".plot3d-overlay-host").first()).toHaveCSS(
    "--scene-bg",
    "#31506f",
  );

  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await open3DGroup(page, "Scene");
  await expect(group(page, "3D background").locator("select")).toHaveValue("custom");
  await expect(group(page, "Vertical scale").locator('input[type="range"]')).toHaveValue("2");
  await expect(page.locator(".plot3d-overlay-host").first()).toHaveCSS(
    "--scene-bg",
    "#31506f",
  );
});

test("independent scope preserves separate scene settings for each 3D renderer", async ({ page }) => {
  await open3D(page, "Volume");
  await expect(page.getByRole("button", { name: "This view", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await open3DGroup(page, "Scene");
  await group(page, "3D background").locator("select").selectOption("custom");
  await group(page, "3D background").locator('input[type="color"]').fill("#31506f");

  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await open3DGroup(page, "Scene");
  await expect(group(page, "3D background").locator("select")).not.toHaveValue("custom");

  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await open3DGroup(page, "Scene");
  await expect(group(page, "3D background").locator("select")).toHaveValue("custom");
  await expect(group(page, "3D background").locator('input[type="color"]')).toHaveValue("#31506f");
});

test("the scientific-resolution badge collapses and remembers the choice", async ({ page }) => {
  await open3D(page, "Volume");
  await page.getByRole("button", { name: "Hide scientific resolution details" }).click();
  await expect(page.locator(".plot3d-badge__sub")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Show scientific resolution details" })).toBeVisible();

  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await expect(page.getByRole("button", { name: "Show scientific resolution details" })).toBeVisible();
  await page.getByRole("button", { name: "Show scientific resolution details" }).click();
  await expect(page.locator(".plot3d-badge__mode")).toBeVisible();
});

test("streamline field and uniform-colour controls remain explicit", async ({ page }) => {
  await openDirect3D(page, "streamlines");
  await open3DGroup(page, "Field");
  await expect(group(page, "Vector field")).toBeVisible();
  await group(page, "Color").getByRole("button", { name: "Uniform", exact: true }).click();
  const color = group(page, "Uniform color").locator(
    'input[aria-label="Uniform streamline color"]',
  );
  await color.fill("#e26370");
  await expect(color).toHaveValue("#e26370");
  await expect(page.locator(".plot3d-colorbar", { hasText: "Speed" })).toHaveCount(0);
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
  await expect.poll(async () => {
    const current = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    return current?.evidence.layers.some(layer => layer.kind === "actor" && layer.visible
      && (layer.pointCount ?? 0) > 0 && !layer.scalarVisibility) ?? false;
  }, { timeout: 30_000 }).toBe(true);
  const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(capture).not.toBeNull();
  if (!capture) throw new Error("streamline appearance capture unavailable");
  expect(capture.evidence.descriptor.parameters?.colorMode).toBe("uniform");
  expect(capture.evidence.descriptor.parameters?.uniformColor).toBe("#e26370");
  expect(capture.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && layer.pointCount && !layer.scalarVisibility
  )).toBe(true);
});

test("light-scene isosurfaces retain signed tube colour through ambient fill", async ({ page }) => {
  await openDirect3D(page, "isosurface");
  await page.evaluate(() => window.__nimbusVtkHarness?.setStep(1));
  await open3DGroup(page, "Scene");
  await group(page, "3D background").locator("select").selectOption("light");
  await expect(page.locator(".plot3d-overlay-host")).toHaveCSS("--scene-bg", "#e6e8ec");
  await expect.poll(async () => {
    const capture = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    return capture?.evidence.layers.filter((layer) =>
      layer.kind === "actor" && layer.visible && (layer.cellCount ?? 0) > 0
        && layer.ambient === 0.72 && layer.diffuse === 0.46
    ).length ?? 0;
  }, { timeout: 30_000 }).toBeGreaterThan(0);
});

test("lighting presets and supercell material controls are exposed without raw light clutter", async ({ page }) => {
  await open3D(page, "Rendered supercell");

  await open3DGroup(page, "Scene");
  await group(page, "3D lighting").locator("select").selectOption("flat");
  await expect(group(page, "3D lighting").locator("select")).toHaveValue("flat");
  await open3DGroup(page, "Cloud");
  await expect(group(page, "Cloud tone").locator("select")).toHaveValue("auto");
  await expect(group(page, "Cloud brightness").locator('input[type="range"]')).toBeEnabled();
  await expect(group(page, "Cloud brightness").locator('input[type="range"]')).toHaveValue("0.85");
  await open3DGroup(page, "Vortex");
  await expect(group(page, "Vortex color")).toContainText("Neutral gray");
  await group(page, "Vortex core").getByRole("button").click();
  await group(page, "Vortex color").getByRole("button", { name: "Match cloud" }).click();
  await expect(group(page, "Vortex color")).toContainText("geometry follows the selected cloud tone");
  await open3DGroup(page, "Precip");
  await expect(group(page, "Precipitation").getByRole("button")).toBeEnabled();
  await expect(group(page, "Precipitation").getByRole("button"))
    .toHaveAttribute("title", "falling hydrometeors");
});

test("the 3D backend is shared on demand and always has a compatibility fallback", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await open3D(page, "Volume");
  await page.getByRole("button", { name: "All 3D", exact: true }).click();
  await open3DGroup(page, "Scene");

  const volumeBackend = group(page, "3D renderer");
  await expect(volumeBackend).toContainText("WebGL volume-safe");
  await volumeBackend.locator("select").selectOption("webgl");
  await expect(volumeBackend).toContainText("WebGL active");

  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await open3DGroup(page, "Scene");
  const isosurfaceBackend = group(page, "3D renderer");
  await expect(isosurfaceBackend.locator("select")).toHaveValue("webgl");
  await expect(isosurfaceBackend).toContainText("WebGL active");

  // Explicit WebGPU either qualifies and renders, or reports the deterministic
  // WebGL fallback. A failed adapter/device must never strand the plot.
  await isosurfaceBackend.locator("select").selectOption("webgpu");
  await expect(isosurfaceBackend).toContainText(/WebGPU active|WebGL fallback/, {
    timeout: 30_000,
  });
  await expect(page.locator(".volume-render__canvas canvas")).toBeVisible();
  await expect(page.locator(".volume-render__overlay--error")).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});
