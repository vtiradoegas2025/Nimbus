import { expect, test } from "@playwright/test";
import { writeFile } from "node:fs/promises";
import { withSceneErrors } from "../support/sceneBudgets";

test("joint volume rays preserve native lattices, analytic mixtures, and actor-order independence", async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  const progress: unknown[] = [];
  page.on("console", message => {
    const text = message.text();
    if (text.startsWith("NIMBUS_JOINT_PROGRESS ") && progress.length < 270) progress.push(JSON.parse(text.slice("NIMBUS_JOINT_PROGRESS ".length)));
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  let rows;
  try { rows = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyJointVolumes()); }
  catch (error) {
    await testInfo.attach("joint-volume-failed-progress", { body: JSON.stringify(progress, null, 2), contentType: "application/json" });
    throw error;
  }
  await testInfo.attach("joint-volume-progress", { body: JSON.stringify(progress, null, 2), contentType: "application/json" });
  for (const row of rows) {
    for (const [index, pixel] of row.pixels.entries()) {
      await writeFile(testInfo.outputPath(`${row.mode}-${index}.png`), Buffer.from(row.captures[index].split(",")[1], "base64"));
      for (let channel = 0; channel < 3; channel++) {
        expect(Math.abs(pixel[channel] - row.expected[channel]), JSON.stringify({ mode: row.mode, pixel, expected: row.expected })).toBeLessThan(row.mode === "shadow" ? 0.007 : 0.015);
        expect(Math.abs(pixel[channel] - row.pixels[0][channel])).toBeLessThan(1.5 / 255);
      }
    }
  }
  await testInfo.attach("joint-optical-results", { body: JSON.stringify(rows.map(({ captures: _, ...row }) => row)), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("joint Supercell layer transitions retain active textures and bound GPU resources", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  const ready = async (layers: number) => {
    await withSceneErrors(page, () => page.waitForFunction(layers => {
      const error = document.querySelector(".volume-render__overlay--error");
      if (error) throw new Error(error.textContent ?? "Joint Supercell preparation failed");
      const scene = document.querySelector<HTMLElement>('[data-testid="scene"]');
      if (scene?.dataset.frameLoading !== "false") return false;
      const host = document.querySelector<HTMLElement>(".plot3d-overlay-host");
      if (host?.dataset.scenePresentationPending !== "false") return false;
      const evidence = window.__nimbusVtkHarness!.inspectAppearance();
      return evidence?.descriptor.parameters?.shadowSampleFactor === 4
        && evidence.descriptor.parameters?.jointVolumeLayers === layers;
    }, layers, { timeout: 120_000 }));
    const capture = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
    expect(capture?.evidence.descriptor.parameters?.shadowSampleFactor).toBe(4);
    expect(capture?.evidence.descriptor.parameters?.jointVolumeLayers).toBe(layers);
  };
  await ready(1);
  const tab = page.locator(".plot3d-control-tab", { hasText: "Precip" }).first();
  await tab.click();
  const toggle = page.locator(".control-group", { hasText: "Precipitation" }).getByRole("button").first();
  const snapshots = [];
  for (let cycle = 0; cycle < 4; cycle++) {
    await toggle.click(); await ready(2);
    const before = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    for (const side of [-1, 1]) await page.evaluate((side) => window.__nimbusVtkHarness!.setCamera(
      [3000 + side * 12000, 3000 - side * 10000, 7000], [3000, 3000, 1000], false,
    ), side);
    const after = await page.evaluate(() => window.__nimbusVtkHarness!.snapshot());
    expect(after.texture3DUploads).toBe(before.texture3DUploads);
    snapshots.push(after);
    await toggle.click(); await ready(1);
  }
  const textureBytes = snapshots.map((snapshot) => snapshot.gpuLogicalTextureBytes);
  const bufferBytes = snapshots.map((snapshot) => snapshot.gpuLogicalBufferBytes);
  await testInfo.attach("joint-resource-snapshots", { body: JSON.stringify(snapshots), contentType: "application/json" });
  console.log("JOINT_RESOURCE_SNAPSHOTS", JSON.stringify(snapshots.map(snapshot => ({
    textures: snapshot.textures, contexts: snapshot.webglContexts, textureBytes: snapshot.gpuLogicalTextureBytes,
    canvases: snapshot.canvases, bufferBytes: snapshot.gpuLogicalBufferBytes,
  }))));
  expect(Math.max(...textureBytes) - Math.min(...textureBytes)).toBeLessThan(64 * 1024);
  expect(Math.max(...bufferBytes) - Math.min(...bufferBytes)).toBeLessThan(4096);
  await testInfo.attach("joint-resource-plateau", { body: JSON.stringify({ textureBytes, bufferBytes }), contentType: "application/json" });
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(false));
  await page.waitForFunction(() => window.__nimbusVtkHarness!.snapshot().gpuLogicalBytes === 0);
  expect(errors).toEqual([]);
});
