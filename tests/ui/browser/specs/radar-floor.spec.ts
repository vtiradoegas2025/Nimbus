import { expect, test, type Page } from "@playwright/test";

/*
 * The ground-plane radar floor, end to end in a browser.
 *
 * Everything about this layer was unit-tested and none of it worked. The
 * planner sized its window against the bytes the response carries, the backend
 * charges separately for the bytes a cell-mean aggregation must read, and the
 * column-maximum reduction was refused outright on any grid deeper than about
 * 64 MB. The control reported "on · 512×512" throughout, because the read's
 * error was stored in state that nothing rendered.
 *
 * So these tests assert the two things the unit tests structurally could not:
 * that the request the planner builds is one the backend will actually serve,
 * and that when it is not, the user is told.
 */

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

async function openVolume(page: Page, terrain = false): Promise<void> {
  await page.goto(
    `/harness/index.html?mode=studio${terrain ? "&terrain=1" : ""}`,
    { waitUntil: "load" },
  );
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
}

async function openDirectVolume(page: Page): Promise<void> {
  await page.goto("/harness/index.html", { waitUntil: "load" });
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
}

/** The floor's own toggle, which doubles as its status readout. */
function floorToggle(page: Page) {
  return page.locator(".control-group", { hasText: "Radar floor" }).getByRole("button");
}

async function enableFloor(page: Page): Promise<void> {
  await floorToggle(page).click();
}

/**
 * Wait for a settled read, then fail loudly rather than on a timeout.
 *
 * `aria-busy` and not the label text: the previous sheet stays on the canvas
 * while the next one loads, which is what makes frame scrubbing smooth, so the
 * label keeps reporting the old lattice throughout. Reading it during a reload
 * returns the previous reduction's answer — which is exactly how this raced
 * when it polled on text alone.
 */
async function expectFloorLoaded(
  page: Page,
  settled: (label: string) => boolean = () => true,
): Promise<string> {
  const toggle = floorToggle(page);
  await expect(async () => {
    expect(await toggle.getAttribute("aria-busy"), "still reading").not.toBe("true");
    const current = (await toggle.textContent()) ?? "";
    expect(current, `floor read failed: ${await toggle.getAttribute("title")}`)
      .not.toContain("failed");
    expect(current, "floor never left its loading state").not.toContain("loading");
    // A caller-supplied end state, because a click that starts a new read has
    // not necessarily reached React by the time this first runs — checking
    // `aria-busy` alone can sample the settled state from BEFORE the click.
    expect(settled(current), `label "${current}" is not the expected end state`).toBe(true);
  }).toPass({ timeout: 20_000 });
  return (await toggle.textContent()) ?? "";
}

test("the surface floor loads at full native resolution", async ({ page }) => {
  await openVolume(page);
  await enableFloor(page);

  const label = await expectFloorLoaded(page);
  // The fixture grid is 48x36 and small enough to need no stride, so the floor
  // must arrive at full native horizontal resolution.
  expect(label).toContain("48×36");
});

test("radar floor and its legend require an explicit selection in the current scene", async ({ page }) => {
  await openVolume(page);
  await expect(floorToggle(page)).toContainText("off");
  await expect(page.locator(".plot3d-colorbar", { hasText: "floor" })).toHaveCount(0);

  await enableFloor(page);
  await page.getByRole("button", { name: "Column max", exact: true }).click();
  await expectFloorLoaded(page, (text) => !text.includes("no echo"));
  await expect(page.locator(".plot3d-colorbar", { hasText: "floor" })).toHaveCount(1);

  await page.reload({ waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
  await expect(floorToggle(page)).toContainText("off");
  await expect(page.locator(".plot3d-colorbar", { hasText: "floor" })).toHaveCount(0);
});

test("radar floor opacity updates the committed actor without changing its legend", async ({ page }) => {
  await openDirectVolume(page);
  await enableFloor(page);
  await page.getByRole("button", { name: "Column max", exact: true }).click();
  await expectFloorLoaded(page, (text) => !text.includes("no echo"));
  const opacity = page.getByRole("slider", { name: "Radar floor opacity" });
  await expect(opacity).toHaveValue("1");
  await opacity.fill("0.45");
  await expect(opacity).toHaveValue("0.45");
  await expect(page.locator(".plot3d-colorbar", { hasText: "floor" })).toHaveCount(1);
  await expect.poll(async () => {
    const current = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
    return current != null;
  }, { timeout: 30_000 }).toBe(true);
  const evidence = await page.evaluate(() => window.__nimbusVtkHarness?.captureAppearance() ?? null);
  expect(evidence).not.toBeNull();
  if (!evidence) throw new Error("radar-floor appearance capture unavailable");
  expect(evidence.evidence.layers.some((layer) =>
    layer.kind === "actor" && layer.visible && layer.opacity === 0.45
  )).toBe(true);
});

test("a terrain-following floor consumes exact geometry and drapes without flattening", async ({ page }) => {
  await openVolume(page, true);
  await enableFloor(page);

  // The terrain fixture can only settle if the schema-v4 centroid/volume body
  // survives IPC, bottom interfaces are reconstructed, and setTerrainGeometry
  // accepts the resulting 48x36 mesh. A flat fallback would not consume that
  // geometry and is deliberately absent from the terrain path.
  const label = await expectFloorLoaded(page);
  expect(label).toContain("48×36");
  await expect(floorToggle(page)).not.toContainText("failed");
});

/*
 * The fixture's storm is centred at k≈10 of 24 levels and its surface
 * reflectivity peaks at -8.7 dBZ — below the ramp's clear-air floor, so nothing
 * paints. That is a correct answer, and the control has to distinguish it from
 * a broken layer, because these are the same pixels otherwise: an empty canvas.
 *
 * This is also the exact trap the user hit. Surface said nothing was wrong
 * while showing nothing, and the advice it gave — switch to Column max — led
 * straight into a request the backend refused.
 */
test("an aloft-only storm reports no echo rather than silently drawing nothing", async ({ page }) => {
  await openVolume(page);
  await enableFloor(page);

  const label = await expectFloorLoaded(page);
  expect(label).toContain("no echo");
  await expect(floorToggle(page)).toHaveAttribute("title", /Column max/);
});

test("column maximum finds the echo the surface reduction cannot", async ({ page }) => {
  await openVolume(page);
  await enableFloor(page);
  await page.getByRole("button", { name: "Column max", exact: true }).click();

  // 58 dBZ aloft over 128 columns: the advice the surface state gives must
  // actually work, which before this change it did not — it errored.
  const label = await expectFloorLoaded(page, (text) => !text.includes("no echo"));
  expect(label).toContain("48×36");
});

/*
 * The substitution that makes column maximum affordable.
 *
 * `composite_reflectivity` IS the column maximum of `reflectivity_dbz`, so
 * reading it is exact rather than approximate — and on a real 512x512x160 grid
 * it is a 1 MB read instead of a 168 MB one the backend refuses. Because it is
 * a different array, the user is told which one was read.
 */
test("column maximum reads the engine's own composite, and says so", async ({ page }) => {
  await openVolume(page);
  await enableFloor(page);
  await page.getByRole("button", { name: "Column max", exact: true }).click();
  await expectFloorLoaded(page);

  await expect(floorToggle(page)).toHaveAttribute(
    "title",
    /composite_reflectivity/,
  );
});

test("the surface reduction does not substitute the composite", async ({ page }) => {
  await openVolume(page);
  await enableFloor(page);
  await expectFloorLoaded(page);

  // A column maximum is not a surface value; substituting it here would paint
  // echo aloft onto the ground plane.
  await expect(floorToggle(page)).not.toHaveAttribute(
    "title",
    /composite_reflectivity/,
  );
});

/*
 * Resolution independence: the whole reason the floor has its own data path.
 * Dropping 3D render fidelity shrinks the volume's voxel budget and must leave
 * the floor lattice untouched.
 */
test("floor resolution survives a drop in 3D fidelity", async ({ page }) => {
  await openVolume(page);
  await enableFloor(page);
  const before = await expectFloorLoaded(page);

  const fidelity = page.getByRole("button", { name: "Balanced", exact: true });
  if (await fidelity.count() > 0) {
    await fidelity.first().click();
    const after = await expectFloorLoaded(page);
    expect(after, "the floor is not derived from the volume").toBe(before);
  }
});

test("shared scope carries radar-floor selection across 3D renderers", async ({ page }) => {
  await openVolume(page);
  await page.getByRole("button", { name: "All 3D", exact: true }).click();
  await enableFloor(page);
  await page.getByRole("button", { name: "Column max", exact: true }).click();

  await page.getByRole("button", { name: "Isosurface", exact: true }).click();
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
  await expect(floorToggle(page)).toContainText("on");
  await expect(page.getByRole("button", { name: "Column max", exact: true }))
    .toHaveClass(/pill--active/);
});

/*
 * The disclosure that was missing entirely.
 *
 * A floor whose read fails must say so on the control. Previously the error was
 * captured in state and rendered nowhere, so a failed layer and a still-loading
 * layer were the same pixels forever — which is what "radar floor errors out"
 * looked like from the outside: nothing.
 */
test("a failed floor read is reported on the control", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });

  // Break exactly the floor's read, leaving every other field load intact, so
  // the failure path is reached the way a real backend refusal reaches it.
  await page.evaluate(() => {
    const target = window as unknown as {
      __TAURI_INTERNALS__?: { invoke?: (...args: unknown[]) => Promise<unknown> };
    };
    const original = target.__TAURI_INTERNALS__?.invoke;
    if (!original || !target.__TAURI_INTERNALS__) throw new Error("no Tauri mock to wrap");
    target.__TAURI_INTERNALS__.invoke = async (...args: unknown[]) => {
      const [command, payload] = args as [string, Record<string, unknown> | undefined];
      const request = payload?.request as { requestId?: string } | undefined;
      if (command === "read_field_window" && request?.requestId?.startsWith("floor:")) {
        throw new Error(
          "cell-mean field window must inspect 167772160 source bytes; request budget is 67108864",
        );
      }
      return original(...args);
    };
  });

  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
  await enableFloor(page);

  await expect(floorToggle(page)).toContainText("failed", { timeout: 20_000 });
  // Not only on the toggle: the reason is on screen without hovering, because a
  // tooltip is not a disclosure for anyone using a keyboard or a touch screen.
  await expect(page.locator(".volume-render__control-error")).toContainText(
    "source bytes",
  );
});
