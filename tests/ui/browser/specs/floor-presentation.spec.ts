import { test, expect } from "@playwright/test";

for (const plot of ["volume", "isosurface", "streamlines"] as const) test(`${plot} waits for a same-frame floor encoding before reporting complete presentation`, async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.goto("/harness/index.html");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(plot => { window.__nimbusVtkHarness!.setSize(640, 480);
    window.__nimbusVtkHarness!.setPlot(plot); window.__nimbusVtkHarness!.setStep(1); }, plot);
  const ready = () => page.waitForFunction(() => {
    const data = document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset;
    return data?.frameLoading === "false" && data.renderedStep === "1";
  }, undefined, { timeout: 90_000 });
  await ready();
  await page.locator(".plot3d-control-tab", { hasText: "Radar" }).click();
  await page.getByRole("group", { name: "Radar floor", exact: true }).getByRole("button").first().click();
  await page.getByRole("button", { name: "Column max", exact: true }).click();
  await ready();
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  await page.evaluate(() => {
    const state = { held: 0, released: false, release: () => {} };
    Object.assign(window, { floorEncodeHold: state });
    const pending: (() => void)[] = [];
    state.release = () => { state.released = true; for (const send of pending.splice(0)) send(); };
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, options) {
      if (!state.released && message?.kind === "encode") {
        state.held++;
        pending.push(() => post.call(this, message, options as StructuredSerializeOptions));
      } else post.call(this, message, options as StructuredSerializeOptions);
    };
  });
  await page.getByRole("button", { name: "Surface", exact: true }).click();
  await page.waitForFunction(() => (window as unknown as { floorEncodeHold: { held: number } }).floorEncodeHold.held > 0);
  await expect(page.getByTestId("scene")).toHaveAttribute("data-frame-loading", "true");
  const pending = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(pending!.evidence.descriptor.frameIndex).toBe(1);
  expect(pending!.evidence.layers).toEqual(before!.evidence.layers);
  await page.evaluate(() => (window as unknown as { floorEncodeHold: { release: () => void } }).floorEncodeHold.release());
  await ready();
  const after = await page.evaluate(() => window.__nimbusVtkHarness!.captureAppearance());
  expect(after!.evidence.descriptor.frameIndex).toBe(1);
  await info.attach("floor-stage-presentation", { body: JSON.stringify({ before: before!.evidence,
    pending: pending!.evidence, after: after!.evidence }), contentType: "application/json" });
});
