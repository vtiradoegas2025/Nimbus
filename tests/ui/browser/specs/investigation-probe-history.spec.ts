import { expect, test } from "@playwright/test";

/*
 * "Build probe history" reads every frame in the run, one at a time. It used to
 * have no cancel, no busy state on its button, and -- the visible fault --
 * `historyProgress` was never reset, so the caption stayed frozen at the last
 * count for the rest of the session. A build that FAILED left something like
 * "37/152 frames" on screen, which reads as still-in-progress, while the error
 * appeared somewhere else entirely.
 *
 * `?readDelay=` slows each fixture field read so the in-flight state is
 * observable at all; without it the loop finishes within one animation frame.
 */

const HISTORY_FIGURE = ".investigation__figure:has(.investigation__history-empty)";
const CAPTION = `${HISTORY_FIGURE} figcaption span`;
const IDLE_CAPTION = "Load on request to keep browsing responsive";

test.beforeEach(({}, info) => info.setTimeout(60_000));

test("the probe history reports that it is building, and can be cancelled", async ({ page }) => {
  await page.goto("/harness/investigation.html?readDelay=150");

  const build = page.getByRole("button", { name: "Build probe history" });
  await expect(build).toBeEnabled();
  await expect(page.locator(CAPTION)).toHaveText(IDLE_CAPTION);

  await build.click();

  // While in flight: the action is unavailable and says so, a cancel appears,
  // and the caption counts rather than sitting on the idle sentence.
  const building = page.getByRole("button", { name: "Building…" });
  await expect(building).toBeDisabled();
  await expect(building).toHaveAttribute("title", /already being built/);
  const cancel = page.getByRole("button", { name: "Cancel" });
  await expect(cancel).toBeVisible();
  await expect(page.locator(CAPTION)).toHaveText(/\d+\/\d+ frames/);

  await cancel.click();

  // After cancelling, the counter must NOT be left behind: a stale count is
  // indistinguishable from an ongoing build.
  await expect(page.locator(CAPTION)).toHaveText(IDLE_CAPTION);
  await expect(page.getByRole("button", { name: "Build probe history" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Cancel" })).toHaveCount(0);
});

test("a completed build leaves no progress counter behind", async ({ page }) => {
  await page.goto("/harness/investigation.html");

  await page.getByRole("button", { name: "Build probe history" }).click();

  // On success the plot replaces the empty state entirely.
  await expect(page.locator(".investigation__history-canvas")).toBeVisible();
  await expect(page.locator(".investigation__history-empty")).toHaveCount(0);
  // No figure caption anywhere may still be counting frames. Checked across
  // every caption rather than with a scoped locator, because after success the
  // history figure no longer carries a distinguishing class.
  const captions = await page.locator(".investigation__figure figcaption").allInnerTexts();
  expect(captions.length).toBeGreaterThan(0);
  expect(captions.filter((text) => /\d+\/\d+ frames/.test(text))).toEqual([]);
});

test("a restored investigation continues saving later frame selections", async ({ page }) => {
  const savedFrame = () => page.evaluate(() => {
    const key = Object.keys(localStorage).find(value => value.startsWith("nimbus:investigation-workspace:v1:"));
    return key ? (JSON.parse(localStorage.getItem(key)!) as { stepIndex: number }).stepIndex : null;
  });
  await page.goto("/harness/investigation.html");
  await expect(page.locator(".investigation__profile-canvas")).toBeVisible();
  await page.getByLabel("Linked investigation time").fill("1");
  await expect.poll(savedFrame).toBe(1);
  await page.reload();
  await expect(page.getByLabel("Linked investigation time")).toHaveValue("1");
  await expect(page.locator(".investigation__profile-canvas")).toBeVisible();
  await expect.poll(savedFrame).toBe(1);
  await page.getByLabel("Linked investigation time").fill("2");
  await expect.poll(savedFrame).toBe(2);
  await page.reload();
  await expect(page.getByLabel("Linked investigation time")).toHaveValue("2");
  await expect(page.locator(".investigation__profile-canvas")).toBeVisible();
  await expect.poll(savedFrame).toBe(2);
});

test("changing a probe-history query cancels obsolete values and allows a fresh history", async ({ page }) => {
  await page.goto("/harness/investigation.html?readDelay=150");
  const build = page.getByRole("button", { name: "Build probe history" });
  await expect(build).toBeEnabled();
  await build.click();
  await expect(page.getByRole("button", { name: "Building…" })).toBeDisabled();
  await page.getByLabel("Investigation level").fill("2");
  await expect(build).toBeEnabled();
  await expect(page.getByRole("button", { name: "Cancel", exact: true })).toHaveCount(0);
  await expect(page.locator(".investigation__history-canvas")).toHaveCount(0);
  await page.waitForTimeout(800);
  await expect(page.locator(".investigation__history-canvas")).toHaveCount(0);
  await expect(page.locator(CAPTION)).toHaveText(IDLE_CAPTION);
  await build.click();
  await expect(page.locator(".investigation__history-canvas")).toBeVisible();
  await expect(page.locator(".investigation__error")).toHaveCount(0);
});

test("probe history uses declared source indices when any physical time is unavailable", async ({ page }) => {
  await page.goto("/harness/investigation.html?timeUnknown=1");
  await page.getByRole("button", { name: "Build probe history" }).click();
  const graph = page.locator(".investigation__history-canvas");
  await expect(graph).toBeVisible();
  const result = await graph.evaluate(element => {
    const chart = element as HTMLElement & { data: { x: number[] }[]; layout: { xaxis: { title: { text: string } } } };
    return { x: Array.from(chart.data[0].x), title: chart.layout.xaxis.title.text };
  });
  expect(result.title).toContain("source frame index");
  expect(result.title).toContain("physical time unavailable");
  expect(result.x).toEqual(Array.from({ length: result.x.length }, (_, index) => index));
  await page.getByLabel("Linked investigation time").fill("1");
  await expect(page.getByLabel("Linked investigation time").locator("..")).toContainText("physical time unavailable");
  await page.getByLabel("Annotation text").fill("This frame has no declared time");
  await expect(page.getByRole("button", { name: "Add at current probe", exact: true })).toBeDisabled();
});

test("publication refuses a source change during composed PNG encoding and then recovers", async ({ page }) => {
  const downloads: string[] = [];
  page.on("download", download => downloads.push(download.suggestedFilename()));
  await page.goto("/harness/investigation.html");
  await expect(page.locator(".investigation__profile-canvas")).toBeVisible();
  await page.getByRole("button", { name: "Publication", exact: true }).click();
  await page.evaluate(() => {
    const original = HTMLCanvasElement.prototype.toBlob;
    const state = { held: false, release: () => {} };
    Object.assign(window, { investigationBlobHold: state });
    HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
      if (this.width === 1520 && !state.held) {
        state.held = true;
        state.release = () => {
          HTMLCanvasElement.prototype.toBlob = original;
          original.call(this, callback, type, quality);
        };
        return;
      }
      original.call(this, callback, type, quality);
    };
  });
  const publish = page.getByRole("button", { name: "Export figure package", exact: true });
  await publish.click();
  await expect.poll(() => page.evaluate(() =>
    (window as unknown as { investigationBlobHold: { held: boolean } }).investigationBlobHold.held
  ), { timeout: 20_000 }).toBe(true);
  await page.getByLabel("Investigation level").fill("2");
  await page.evaluate(() =>
    (window as unknown as { investigationBlobHold: { release: () => void } }).investigationBlobHold.release()
  );
  await expect(page.locator(".investigation__publication output")).toContainText(/changed|Retry/);
  expect(downloads).toEqual([]);
  await expect(publish).toBeEnabled();
  const downloadPromise = page.waitForEvent("download");
  await publish.click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.nimbus-figure\.zip$/);
  await expect(page.locator(".investigation__publication output")).toContainText("verified provenance");
  expect(downloads).toHaveLength(1);
});
