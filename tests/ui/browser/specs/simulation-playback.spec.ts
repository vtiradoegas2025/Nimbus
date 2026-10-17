import { test, expect } from "@playwright/test";
import { writeFile } from "node:fs/promises";

interface PlaybackTimingSample { frame: number; at: number }
interface PlaybackTimingAudit {
  frames: PlaybackTimingSample[];
  longTasksMs: number[];
  observer: MutationObserver;
  performanceObserver: PerformanceObserver | null;
}

declare global {
  interface Window { __nimbusPlaybackTimingAudit?: PlaybackTimingAudit }
}

test("simulation timing follows irregular saved timestamps and preserves frame controls", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=studio&irregularPlayback=1");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  const timing = page.getByRole("combobox", { name: "Playback timing", exact: true });
  const slider = page.getByRole("slider", { name: "Selected frame", exact: true });
  await expect(timing).toBeVisible();
  await slider.fill("0");
  await timing.selectOption("simulation");
  await page.getByRole("combobox", { name: "Simulation playback speed", exact: true }).selectOption("10");
  await expect(page.getByRole("combobox", { name: "Playback speed", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await expect(slider).toHaveValue("1", { timeout: 10_000 });
  const start = Date.now();
  await expect(slider).toHaveValue("2", { timeout: 10_000 });
  const dwellMs = Date.now() - start;
  expect(dwellMs).toBeGreaterThan(650);
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  await timing.selectOption("frames");
  await expect(page.getByRole("combobox", { name: "Playback speed", exact: true })).toHaveValue("2");
  await expect(page.getByRole("combobox", { name: "Frame stride", exact: true })).toBeVisible();
  await info.attach("irregular-frame-dwell", { body: JSON.stringify({ dwellMs, simulationIntervalS: 8, rate: 10 }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("high requested 3D cadence waits for both panes and scrubbing pauses playback", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  const menu = page.locator(".plot-studio__view-menu");
  await menu.locator("summary").click();
  await menu.getByRole("button", { name: "Side by side", exact: true }).click();
  await page.getByRole("combobox", { name: "Companion pane plot type", exact: true }).selectOption("streamlines");
  await expect.poll(async () => {
    const captures = await page.evaluate(() => window.__nimbusStudioHarness!.capturePaneAppearances());
    return captures.length === 2 && captures.every(capture => capture?.evidence.descriptor.frameIndex === 0);
  }, { timeout: 30_000 }).toBe(true);
  const before = await page.evaluate(() => window.__nimbusStudioHarness!.capturePaneAppearances());
  const beforeTexture = await page.evaluate(() => window.__nimbusStudioHarness!.inspectPrimaryVolumeScalarImages());
  const primary = page.locator(".plot-studio__pane--primary");
  const legend = await primary.locator(".plot3d-colorbar__title").textContent();
  const axes = await primary.locator(".plot3d-axis-labels").allTextContents();
  expect(axes).toHaveLength(1);
  const speed = page.getByRole("combobox", { name: "Playback speed", exact: true });
  await speed.selectOption("60");
  await expect(speed).toHaveValue("60");
  const selected = page.getByRole("slider", { name: "Selected frame", exact: true });
  await page.evaluate(() => {
    window.__nimbusStudioHarness!.setFieldReadDelay(800);
    window.__nimbusStudioHarness!.clearFieldCache();
  });
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await expect(selected).toHaveValue("1", { timeout: 10_000 });
  await page.waitForTimeout(150);
  await expect(selected).toHaveValue("1");
  const pending = await page.evaluate(() => window.__nimbusStudioHarness!.capturePaneAppearances());
  expect(pending.map(capture => capture?.evidence.descriptor.frameIndex)).toEqual([0, 0]);
  expect(pending[0]?.evidence.descriptor.parameters?.scalarCompositionFingerprint).toBe(before[0]?.evidence.descriptor.parameters?.scalarCompositionFingerprint);
  expect(await page.evaluate(() => window.__nimbusStudioHarness!.inspectPrimaryVolumeScalarImages())).toEqual(beforeTexture);
  await expect(primary.locator(".plot3d-colorbar__title")).toHaveText(legend!);
  expect(await primary.locator(".plot3d-axis-labels").allTextContents()).toEqual(axes);
  await expect(selected).toHaveValue("2", { timeout: 30_000 });
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  const committed = await page.evaluate(() => window.__nimbusStudioHarness!.capturePaneAppearances());
  for (const capture of committed) {
    expect(capture?.evidence.descriptor.frameIndex).toBeGreaterThanOrEqual(1);
    expect(capture?.evidence.descriptor.frameIndex).toBeLessThanOrEqual(2);
  }
  await expect.poll(async () => {
    const captures = await page.evaluate(() => window.__nimbusStudioHarness!.capturePaneAppearances());
    return captures.every(capture => capture?.evidence.descriptor.frameIndex === 2);
  }, { timeout: 30_000 }).toBe(true);
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await selected.fill("1");
  await expect(page.getByRole("button", { name: "Play frames", exact: true })).toBeVisible();
  await page.waitForTimeout(250);
  await expect(selected).toHaveValue("1");
  await info.attach("delayed-two-pane-playback", { body: JSON.stringify({ requestedFps: 60, fieldReadDelayMs: 800, before: before.map(c => c?.evidence.descriptor), pending: pending.map(c => c?.evidence.descriptor), committed: committed.map(c => c?.evidence.descriptor) }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("completed volume playback records sequential presentation intervals and main-thread work", async ({ page }, info) => {
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.evaluate(() => window.__nimbusStudioHarness!.publishLiveThrough(20));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Volume", exact: true }).click();
  const tab = page.locator(".plot3d-control-tab", { hasText: "Volume" });
  if (await tab.getAttribute("aria-expanded") !== "true") await tab.click();
  await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("w");
  await expect(page.locator(".plot-studio__meta")).toContainText("showing t = 0.0 s");
  const frame = page.getByRole("slider", { name: "Selected frame", exact: true });
  await expect.poll(async () => Number(await frame.getAttribute("max"))).toBe(19);
  await page.getByRole("combobox", { name: "Playback speed", exact: true }).selectOption("30");
  await page.evaluate(() => {
    const frames: PlaybackTimingSample[] = [];
    const longTasksMs: number[] = [];
    const sample = () => {
      const value = document.querySelector(".plot-studio__meta")?.textContent?.match(/showing t = ([\d.]+) s/);
      if (!value) return;
      const frame = Number(value[1]) / 30;
      if (frames.at(-1)?.frame !== frame) frames.push({ frame, at: performance.now() });
    };
    const observer = new MutationObserver(sample);
    observer.observe(document.querySelector(".plot-studio__meta")!, { subtree: true, childList: true, characterData: true });
    const performanceObserver = PerformanceObserver.supportedEntryTypes.includes("longtask")
      ? new PerformanceObserver(list => longTasksMs.push(...list.getEntries().map(entry => entry.duration)))
      : null;
    performanceObserver?.observe({ entryTypes: ["longtask"] });
    window.__nimbusPlaybackTimingAudit = { frames, longTasksMs, observer, performanceObserver };
    sample();
  });
  await page.getByRole("button", { name: "Play frames", exact: true }).click();
  await page.waitForFunction(() => (window.__nimbusPlaybackTimingAudit?.frames.length ?? 0) >= 9, undefined, { timeout: 30_000 });
  await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
  const audit = await page.evaluate(() => {
    const { frames, longTasksMs, observer, performanceObserver } = window.__nimbusPlaybackTimingAudit!;
    observer.disconnect();
    performanceObserver?.disconnect();
    return { frames, longTasksMs, longTaskObserverSupported: performanceObserver != null };
  });
  expect(audit.frames.length).toBeGreaterThanOrEqual(9);
  audit.frames.forEach((sample, index) => expect(sample.frame).toBe(index));
  const intervals = audit.frames.slice(1).map((sample, index) => sample.at - audit.frames[index].at);
  const sorted = [...intervals].sort((a, b) => a - b);
  const evidence = {
    schema: "nimbus.studio.bounded-viewer-playback",
    project: info.project.name,
    requestedFps: 30,
    sourceShape: [48, 36, 24],
    viewport: page.viewportSize(),
    frameStride: 1,
    completedFrames: audit.frames,
    intervalMs: { p50: sorted[Math.floor(sorted.length * 0.5)], p95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))], values: intervals },
    completedFps: 1000 * intervals.length / intervals.reduce((sum, value) => sum + value, 0),
    longTasksMs: audit.longTasksMs,
    longTaskObserverSupported: audit.longTaskObserverSupported,
    graphics: await page.evaluate(() => {
      const canvas = document.querySelector<HTMLCanvasElement>(".volume-render__canvas canvas");
      const gl = canvas?.getContext("webgl2") ?? canvas?.getContext("webgl");
      const info = gl?.getExtension("WEBGL_debug_renderer_info");
      const box = canvas?.getBoundingClientRect();
      return { renderer: gl && info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) as string : null, vendor: gl && info ? gl.getParameter(info.UNMASKED_VENDOR_WEBGL) as string : null, devicePixelRatio: window.devicePixelRatio, cssSize: box ? [box.width, box.height] : null, backingSize: canvas ? [canvas.width, canvas.height] : null };
    }),
    resources: await page.evaluate(() => window.__nimbusStudioHarness!.resourceSnapshot()),
  };
  const path = info.outputPath("viewer-playback.json");
  await writeFile(path, JSON.stringify(evidence, null, 2));
  await info.attach("viewer-playback", { path, contentType: "application/json" });
});
