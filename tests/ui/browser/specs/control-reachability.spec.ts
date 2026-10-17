import { expect, test, type Page } from "@playwright/test";

/*
 * Every control a 3D plot offers must be reachable by pointer.
 *
 * "I can't control several controls on the 3D plots" is not a rendering bug and
 * not something a unit test can see: the control exists in the DOM, has its
 * handler bound, and reports the right accessible name — it is simply covered
 * by another element, clipped out of its scroll container, or disabled for a
 * reason the user was never told. All three failure modes look identical from
 * JavaScript and identical in a screenshot.
 *
 * This walks the real control row and hit-tests each control at its own centre,
 * which is the one question that matters: if the user aims at this pixel, does
 * the event land on this control?
 */

const APP_SHELL = "#root{height:100vh;overflow:hidden}"
  + "#root>.plot-studio{height:100vh;overflow:hidden}";

interface ControlProbe {
  label: string;
  /** Name exposed through the same DOM relationships used by the accessibility tree. */
  accessibleName: string;
  tag: string;
  disabled: boolean;
  /** Why it is disabled, as the user would read it. Empty is the defect. */
  title: string;
  /** False when another element owns the centre pixel — a covered control. */
  ownsCentre: boolean;
  /** What actually sits at the centre pixel, when something else does. */
  covering: string | null;
  /** False when the control is scrolled or clipped outside its container. */
  insideViewport: boolean;
}

/** Open the studio on one 3D plot type, which is where the control row lives. */
async function open3D(page: Page, plot: string): Promise<void> {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: plot, exact: true }).click();
  await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });
}

/** Hit-test every interactive control inside the plot's own control row. */
async function probeControls(page: Page): Promise<ControlProbe[]> {
  return page.locator(".volume-render__controls").first().evaluate((row) => {
    const controls = Array.from(
      row.querySelectorAll("button, input, select"),
    ) as HTMLElement[];
    return controls.map((element) => {
      const box = element.getBoundingClientRect();
      const cx = box.left + box.width / 2;
      const cy = box.top + box.height / 2;
      const hit = document.elementFromPoint(cx, cy);
      const owns = hit != null && (hit === element || element.contains(hit) || hit.contains(element));
      const group = element.closest(".control-group");
      const label = group?.querySelector(".control-group__label")?.textContent?.trim() ?? "";
      const own = element.textContent?.trim() ?? "";
      const labelledBy = (element.getAttribute("aria-labelledby") ?? "")
        .split(/\s+/)
        .filter(Boolean)
        .map((id) => document.getElementById(id)?.textContent?.trim() ?? "")
        .filter(Boolean)
        .join(" ");
      const nativeLabels = "labels" in element
        ? Array.from((element as HTMLInputElement).labels ?? [])
          .map((labelElement) => labelElement.textContent?.trim() ?? "")
          .filter(Boolean)
          .join(" ")
        : "";
      const accessibleName = element.getAttribute("aria-label")?.trim()
        || labelledBy
        || nativeLabels
        || (element.tagName === "BUTTON" ? own : "");
      return {
        label: `${label}${own && own !== label ? ` / ${own}` : ""}`,
        accessibleName,
        tag: element.tagName.toLowerCase(),
        disabled: (element as HTMLButtonElement).disabled === true,
        title: element.getAttribute("title") ?? "",
        ownsCentre: owns,
        covering: owns
          ? null
          : `${hit?.tagName.toLowerCase() ?? "none"}.${hit?.className ?? ""}`.slice(0, 80),
        insideViewport: box.width > 0 && box.height > 0
          && box.top >= 0 && box.left >= 0
          && box.bottom <= window.innerHeight && box.right <= window.innerWidth,
      } satisfies ControlProbe;
    });
  });
}

/** The 3D plot types that render a `.volume-render__controls` row. */
const SCENES = ["Volume", "Isosurface", "Streamlines", "Rendered supercell"] as const;

for (const scene of SCENES) {
  test(`${scene}: every control is reachable at its own centre pixel`, async ({ page }) => {
    await open3D(page, scene);

    const probes = await probeControls(page);
    expect(probes.length, "no controls found — the plot did not mount").toBeGreaterThan(1);

    const covered = probes.filter((p) => !p.ownsCentre);
    expect(covered, `controls covered by another element: ${JSON.stringify(covered, null, 2)}`)
      .toEqual([]);

    const clipped = probes.filter((p) => !p.insideViewport);
    expect(clipped, `controls clipped outside the viewport: ${JSON.stringify(clipped, null, 2)}`)
      .toEqual([]);

    const unnamed = probes.filter((probe) => probe.accessibleName.length === 0);
    expect(unnamed, `controls without an accessible name: ${JSON.stringify(unnamed, null, 2)}`)
      .toEqual([]);
  });

  /*
   * A disabled control with no title is a dead end: the user aims, clicks, and
   * nothing happens, with nothing on screen explaining why or what to do
   * instead. Disabling is allowed; disabling silently is not.
   */
  test(`${scene}: a disabled control always says why`, async ({ page }) => {
    await open3D(page, scene);

    const probes = await probeControls(page);
    const mute = probes.filter((p) => p.disabled && p.title.length === 0);
    expect(mute, `disabled with no explanation: ${JSON.stringify(mute, null, 2)}`).toEqual([]);
  });
}

const GROUPS_BY_SCENE = {
  Volume: ["Volume", "Radar", "Scene"],
  Isosurface: ["Data", "Surface", "Radar", "Scene"],
  Streamlines: ["Field", "Tracing", "Radar", "Scene"],
  "Rendered supercell": ["Cloud", "Precip", "Radar", "Flow", "Vortex", "Scene"],
} as const;

for (const [scene, groups] of Object.entries(GROUPS_BY_SCENE)) {
  test(`${scene}: every disclosure group opens alone and remains reachable`, async ({ page }) => {
    await open3D(page, scene);

    for (const label of groups) {
      const button = page.locator(".plot3d-control-tab", { hasText: label }).first();
      await button.click();
      await expect(button).toHaveAttribute("aria-expanded", "true");
      await expect(page.locator(".plot3d-control-panel")).toHaveCount(1);

      const probes = await probeControls(page);
      const covered = probes.filter((probe) => !probe.ownsCentre);
      expect(covered, `${label} controls covered: ${JSON.stringify(covered, null, 2)}`)
        .toEqual([]);
      const clipped = probes.filter((probe) => !probe.insideViewport);
      expect(clipped, `${label} controls clipped: ${JSON.stringify(clipped, null, 2)}`)
        .toEqual([]);
      const unnamed = probes.filter((probe) => probe.accessibleName.length === 0);
      expect(unnamed, `${label} controls unnamed: ${JSON.stringify(unnamed, null, 2)}`)
        .toEqual([]);
    }
  });
}

test("3D paths: path and scene groups use the shared disclosure controls", async ({ page }) => {
  await page.goto("/harness/index.html", { waitUntil: "load" });
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness?.setPlot("trajectories"));
  await expect(page.locator(".volume-render__controls")).toBeVisible({ timeout: 30_000 });

  for (const label of ["Paths", "Scene"]) {
    const button = page.locator(".plot3d-control-tab", { hasText: label });
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "true");
    const probes = await probeControls(page);
    expect(probes.filter((probe) => !probe.ownsCentre), `${label} controls are covered`).toEqual([]);
    expect(probes.filter((probe) => !probe.insideViewport), `${label} controls are clipped`).toEqual([]);
  }
});

/*
 * The same question inside a pane.
 *
 * A pane hides its control row until asked, then floats it over that pane as an
 * absolutely-positioned panel capped at the pane's own height. That is a second
 * way for a control to become unreachable — not covered by a sibling, but
 * clipped by a scroll container smaller than the row it holds — and it does not
 * exist in the single-pane path these tests otherwise cover.
 */
test("a pane's controls are all reachable once opened", async ({ page }) => {
  await page.goto("/harness/index.html?mode=studio", { waitUntil: "load" });
  await page.addStyleTag({ content: APP_SHELL });
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  await expect(page.locator(".volume-render__controls").first()).toBeVisible({ timeout: 30_000 });

  // Pane layout lives in the header's View popover, which closes itself on choice.
  await page.locator(".plot-studio__view-menu summary").click();
  await page.locator(".plot-studio__view-popover .plot-studio__panes-options")
    .getByRole("button", { name: "Quad" }).click();
  const primary = page.locator(".plot-studio__pane--primary");
  await primary.getByRole("button", { name: "Controls" }).click();
  await expect(page.locator(".plot-studio__pane--controls")).toHaveCount(1);
  await primary.locator(".plot3d-control-tab", { hasText: "Volume" }).click();

  const probes = await page.locator(".plot-studio__pane--primary .volume-render__controls")
    .evaluate((row) => {
      const controls = Array.from(row.querySelectorAll("button, input, select")) as HTMLElement[];
      const panel = row.getBoundingClientRect();
      return controls.map((element) => {
        const box = element.getBoundingClientRect();
        const hit = document.elementFromPoint(
          box.left + box.width / 2,
          box.top + box.height / 2,
        );
        const label = element.closest(".control-group")
          ?.querySelector(".control-group__label")?.textContent?.trim() ?? "";
        return {
          label: `${label} / ${element.textContent?.trim() ?? ""}`.trim(),
          // Scrolled-out controls are fine — the panel scrolls. Controls
          // rendered outside the panel's own box are not: nothing can bring
          // them into view.
          escapesPanel: box.height > 0
            && (box.top < panel.top - row.scrollTop - 1
              || box.left < panel.left - 1
              || box.right > panel.right + 1),
          covered: hit != null && !element.contains(hit) && hit !== element
            && !hit.contains(element),
          visible: box.top >= panel.top - 1 && box.bottom <= panel.bottom + 1,
        };
      });
    });

  expect(probes.length, "the pane rendered no controls").toBeGreaterThan(3);
  const escaped = probes.filter((p) => p.escapesPanel);
  expect(escaped, `controls outside the panel that holds them: ${JSON.stringify(escaped)}`)
    .toEqual([]);
  // At least the first row must be usable without scrolling; a panel that opens
  // showing none of its own controls is not an affordance.
  expect(probes.filter((p) => p.visible && !p.covered).length,
    "no control is usable without scrolling the panel").toBeGreaterThan(0);
});

/*
 * The navigation bars are the third way to lose a control, and the quietest: a
 * renderer parked past the end of a horizontal scroller is not covered and not
 * clipped, it simply never enters the user's model of what the app offers.
 * Both bars are pinned above the scrolling toolbelt so they stay whole, so
 * neither may overflow its own width at any container size.
 */
const NAV_BARS = [
  ".plot-studio__category-bar",
  ".plot-studio__picker--nav .plot-studio__type-bar",
] as const;

async function expectBarsWhole(page: Page): Promise<void> {
  for (const selector of NAV_BARS) {
    const bar = page.locator(selector);
    await expect(bar).toBeVisible();
    const measured = await bar.evaluate((element) => ({
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
      overflowX: getComputedStyle(element).overflowX,
    }));
    expect(measured.overflowX, `${selector} is a horizontal scroller`)
      .not.toMatch(/auto|scroll/);
    // Sub-pixel layout rounding, not a hidden choice.
    expect(
      measured.scrollWidth,
      `${selector} overflows its own width (${measured.scrollWidth} > ${measured.clientWidth})`,
    ).toBeLessThanOrEqual(measured.clientWidth + 1);
  }

  // The wrapper is the other way to lose them, and the one that actually bit:
  // it holds both rows and can shrink below their height, turning the pair into
  // a vertical scroller while each bar measures fine on its own.
  const nav = await page.locator(".plot-studio__nav").evaluate((element) => ({
    scrollHeight: element.scrollHeight,
    clientHeight: element.clientHeight,
  }));
  expect(
    nav.scrollHeight,
    `the nav scrolls its own rows away (${nav.scrollHeight} > ${nav.clientHeight})`,
  ).toBeLessThanOrEqual(nav.clientHeight + 1);
}

test("the navigation bars never park a choice off the end of a scroller", async ({ page }) => {
  await open3D(page, "Volume");
  await expectBarsWhole(page);
});

/*
 * The nav is allowed to yield, just last. Pinning it outright once pushed the
 * scrubber 57 px past the bottom edge at this size, so the fix that stops it
 * scrolling has to keep the scrubber on screen here.
 */
test("a short window keeps the scrubber on screen", async ({ page }) => {
  await open3D(page, "Volume");
  await page.setViewportSize({ width: 800, height: 560 });
  const bar = page.locator(".plot-studio__frame-bar");
  await expect(bar).toBeVisible();
  const placement = await bar.evaluate((element) => ({
    bottom: Math.round(element.getBoundingClientRect().bottom),
    viewport: window.innerHeight,
  }));
  expect(
    placement.bottom,
    `the scrubber is ${placement.bottom - placement.viewport}px past the bottom edge`,
  ).toBeLessThanOrEqual(placement.viewport + 1);
});

test("the navigation bars stay whole in a narrow container", async ({ page }) => {
  await open3D(page, "Volume");
  // Below the 981px container breakpoint, where the bars used to be forced onto
  // one line and given their own scrollbars.
  await page.setViewportSize({ width: 880, height: 900 });
  await expect(page.locator(".plot-studio__category-bar")).toBeVisible();
  await expectBarsWhole(page);
});
