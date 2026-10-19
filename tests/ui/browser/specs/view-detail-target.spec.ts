import { test, expect } from "@playwright/test";

test("view detail settles once, ignores orbit jitter, and freezes during playback and recording", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
  const host = page.locator(".plot3d-overlay-host");
  const stormTarget = await host.getAttribute("data-detail-center-m");
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "View focus", exact: true }).click();
  const aim = (x: number, y: number) => page.evaluate(([x, y]) => window.__nimbusVtkHarness!.setCamera([x - 10000, y - 12000, 8000], [x, y, 600], false), [x, y]);
  await aim(1000, 1800);
  await expect(host).toHaveAttribute("data-detail-center-m", "1000,1800");
  for (const [x, y] of [[1500, 2000], [3500, 2500], [4600, 3000]]) {
    await aim(x, y);
    await expect(host).toHaveAttribute("data-detail-center-m", "1000,1800");
  }
  await expect(host).toHaveAttribute("data-detail-center-m", "4600,3000");
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  await page.evaluate(() => window.__nimbusVtkHarness!.setCamera([17000, -9000, 6000], [4600, 3000, 600], false));
  await page.waitForTimeout(550);
  await aim(4640, 3020); await page.waitForTimeout(550);
  await expect(host).toHaveAttribute("data-detail-center-m", "4600,3000");
  const after = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
  expect(after.ipc.startedJobs).toBe(before.ipc.startedJobs);
  const box = (await page.locator(".volume-render__canvas canvas").boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.mouse.down({ button: "middle" });
  await aim(1000, 4500);
  await page.waitForTimeout(550);
  await expect(host).toHaveAttribute("data-detail-center-m", "4600,3000");
  await page.mouse.up({ button: "middle" });
  await expect(host).toHaveAttribute("data-detail-center-m", "1000,4500");
  for (const recording of [false, true]) {
    const held = await host.getAttribute("data-detail-center-m");
    await page.evaluate(recording => window.__nimbusVtkHarness!.setPlayback(!recording, recording), recording);
    await aim(recording ? 4400 : 2300, recording ? 5100 : 1400);
    await page.waitForTimeout(550);
    await expect(host).toHaveAttribute("data-detail-center-m", held!);
    await expect(page.getByRole("button", { name: "Storm", exact: true })).toBeDisabled();
    await page.evaluate(() => window.__nimbusVtkHarness!.setPlayback(false, false));
    await expect(host).toHaveAttribute("data-detail-center-m", recording ? "4400,5100" : "2300,1400");
  }
  await page.getByRole("button", { name: "Storm", exact: true }).click();
  await expect(host).toHaveAttribute("data-detail-center-m", stormTarget!);
  await info.attach("view-target-read-stability", { body: JSON.stringify({ before, after }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("view targeting crosses refinement boundaries without resetting the Earth-frame camera", async ({ page }) => {
  await page.goto("/harness/index.html?derivedBricks=1&nestedVolumes=2");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  const host = page.locator(".plot3d-overlay-host");
  await expect(host).toHaveAttribute("data-output-hierarchy-valid", "true");
  await page.locator(".plot3d-control-tab", { hasText: "Flow" }).click();
  await page.getByTitle("w isosurface", { exact: true }).click();
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
    const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    return scene?.dataset.frameLoading === "false" && scene.dataset.renderedStep === "0"
      && data?.scenePresentationPending === "false" && data.committedSceneGeneration != null
      && data.committedSceneGeneration === data.requestedSceneGeneration
      && Number(data.flowPreparationCommits) > 0;
  });
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("button", { name: "View focus", exact: true }).click();
  for (const [x, y, level] of [[500, 500, "root"], [1200, 1000, "child_0000"], [1400, 1200, "child_0000/child_0000"], [500, 500, "root"]] as const) {
    const position: [number, number, number] = [x - 10000, y - 12000, 8000];
    const focal: [number, number, number] = [x, y, 600];
    await page.evaluate(({ position, focal }) => window.__nimbusVtkHarness!.setCamera(position, focal, false), { position, focal });
    await expect(host).toHaveAttribute("data-output-level", level);
    await page.waitForFunction(() => document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset.frameLoading === "false");
    await page.waitForTimeout(550);
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    expect(capture!.evidence.camera.focalPoint).toEqual(focal);
    expect(capture!.evidence.camera.position).toEqual(position);
    await expect(host).toHaveAttribute("data-output-level", level);
  }
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await expect.poll(() => page.evaluate(() => {
    const snapshot = window.__nimbusVtkHarness!.snapshot();
    return { handles: snapshot.preparedRegistry.handles,
      bytes: snapshot.visualization.committedBytes - snapshot.visualization.byStage["decoded-field"].committedBytes,
      canvases: snapshot.canvasCount };
  })).toEqual({ handles: 0, bytes: 0, canvases: 0 });
  const cacheRelease = await page.evaluate(() => {
    const harness = window.__nimbusVtkHarness!;
    const before = harness.snapshot().visualization.committedBytes;
    const released = harness.clearFieldCache();
    return { before, released, after: harness.snapshot().visualization.committedBytes };
  });
  expect(cacheRelease.released).toBe(cacheRelease.before);
  expect(cacheRelease.after).toBe(0);
});
