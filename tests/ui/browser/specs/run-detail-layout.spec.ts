import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

/*
 * Layout contract for the Run Archive detail column.
 *
 * These assert geometry the component tree cannot express in a unit test and
 * that the mock run records cannot reach (they carry short case names and no
 * live log). Rather than distorting the shared fixtures, the real stylesheets
 * are applied to synthetic markup that mirrors RunDetail's DOM, and the
 * resulting boxes are measured.
 *
 * Both cases here are regressions that shipped:
 *   - `.run-detail__log` was a ROW flex container, so the log panel's width
 *     collapsed to the content width of the log lines and LogStream's
 *     "collapse height to header-only" rule silently applied to the width.
 *   - `.run-detail__header h2` had no `min-width: 0`, so a long run name
 *     pushed the action buttons past the header and into the shell's clip.
 */

const SRC = fileURLToPath(new URL("../../../../ui/src/", import.meta.url));

const STYLES = [
  "styles/theme.css",
  "styles/base.css",
  "styles/App.css",
  "components/runs/RunDetail.css",
  "components/runs/LogStream.css",
].map((file) => readFileSync(join(SRC, file), "utf-8")).join("\n");

const LONG_CASE_NAME = "supercell_bubble_beast_xl_thompson_nx1024_2026_06_18_v3";

const markup = (collapsed: boolean) => `<!doctype html><html><head><style>${STYLES}</style></head><body>
<div class="app app--with-run-library">
  <div class="nav-rail"></div>
  <div class="queue-panel"></div>
  <main class="run-detail">
    <header class="run-detail__header">
      <div class="run-detail__title-row">
        <h2>${LONG_CASE_NAME}</h2>
        <span class="status-pill">running</span>
        <div class="run-detail__title-spacer"></div>
        <button class="run-detail__abort"><span>Abort</span></button>
        <button class="run-detail__open-dir"><span>Open dir</span></button>
        <button class="run-detail__view-plots"><span>View plots</span></button>
      </div>
    </header>
    <section class="run-detail__stats" style="height:120px">stats</section>
    <section class="run-detail__log">
      <div class="log-stream ${collapsed ? "log-stream--collapsed" : ""}">
        <header class="log-stream__header">
          <button class="log-stream__toggle"><span class="log-stream__title">Log</span></button>
          <span class="log-stream__meta">1284 lines</span>
        </header>
        ${collapsed ? "" : `<div class="log-stream__body">${
          Array.from({ length: 60 }, (_, i) => `<div class="log-stream__line">step ${i} sim_t=${i * 3}.00 max_w=12.4</div>`).join("")
        }</div>`}
      </div>
    </section>
  </main>
</div></body></html>`;

const measure = (page: import("@playwright/test").Page) => page.evaluate(() => {
  const box = (selector: string) => {
    const element = document.querySelector(selector);
    if (!element) throw new Error(`missing ${selector}`);
    const rect = element.getBoundingClientRect();
    return { width: Math.round(rect.width), height: Math.round(rect.height), right: Math.round(rect.right) };
  };
  return {
    section: box("section.run-detail__log"),
    panel: box(".log-stream"),
    titleRowRight: box(".run-detail__title-row").right,
    lastActionRight: box(".run-detail__view-plots").right,
    lastActionWidth: box(".run-detail__view-plots").width,
    // Guards the harness itself: if the stylesheets fail to load every box
    // below would still measure, but as block layout, and the assertions
    // would pass or fail for the wrong reason.
    sectionDirection: getComputedStyle(document.querySelector("section.run-detail__log")!).flexDirection,
  };
});

test("log panel fills the width of its section", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.setContent(markup(false));
  const m = await measure(page);

  expect(m.sectionDirection).toBe("column");
  // Section padding is --space-5 (20px) each side.
  expect(m.panel.width).toBe(m.section.width - 40);
});

test("collapsing the log returns its height to the sections above", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.setContent(markup(false));
  const expanded = await measure(page);

  await page.setContent(markup(true));
  const collapsed = await measure(page);

  // Collapsed shrinks on the BLOCK axis only — the panel stays full width.
  expect(collapsed.panel.width).toBe(expanded.panel.width);
  expect(collapsed.panel.height).toBeLessThan(60);
  expect(collapsed.section.height).toBeLessThan(expanded.section.height / 2);
});

test("a long run name ellipsizes instead of clipping the action buttons", async ({ page }) => {
  // The narrowest window tauri.conf.json permits, where the detail column is
  // tightest and the overflow originally appeared.
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.setContent(markup(false));
  const m = await measure(page);

  expect(m.lastActionRight).toBeLessThanOrEqual(m.titleRowRight);
  // The button keeps its intrinsic width rather than being squeezed.
  expect(m.lastActionWidth).toBeGreaterThan(80);
});
