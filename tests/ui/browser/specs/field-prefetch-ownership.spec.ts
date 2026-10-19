import { expect, test } from "@playwright/test";

for (const plot of ["slice", "vertical"] as const) {
  for (const action of ["replaced", "deselected", "unmounted"] as const) {
    test(`${plot} cancels owned future-frame reads when its field is ${action}`, async ({ page }) => {
      await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&heldPrefetch=1`);
      await page.waitForFunction(() => (window.__nimbusRegularHarness?.snapshot().ownership.pending.length ?? 0) > 0);
      const pending = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().ownership.pending.map(read => read.requestId));
      if (action === "replaced") await page.getByRole("combobox", { name: "Field", exact: true }).selectOption("u");
      else if (action === "deselected") await page.evaluate(() => window.__nimbusRegularHarness!.setFrames([]));
      else await page.evaluate(() => window.__nimbusRegularHarness!.setPlot("series"));
      await expect.poll(async () => page.evaluate(() => {
        const state = window.__nimbusRegularHarness!.snapshot();
        return [state.ownership.pending.length, state.activeLoads, state.queuedLoads, state.prefetches];
      })).toEqual([0, 0, 0, 0]);
      const cancelled = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().ownership.cancelledReadIds);
      expect(cancelled).toEqual(expect.arrayContaining(pending));
      if (action === "deselected" && plot === "slice") {
        await expect(page.getByText("No data to plot.", { exact: true })).toBeVisible();
      }
      await expect(page.locator(".plot-overlay--error")).toHaveCount(0);
    });
  }
  test(`${plot} cancels the old prefetch when its bounded sampling window changes`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&heldPrefetch=1`);
    await page.waitForFunction(() => (window.__nimbusRegularHarness?.snapshot().ownership.pending.length ?? 0) === 1);
    const pending = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().ownership.pending[0].requestId);
    if (plot === "slice") await page.getByRole("button", { name: "x", exact: true }).click();
    else {
      await page.getByRole("slider").first().focus();
      await page.getByRole("slider").first().press("ArrowRight");
    }
    await page.waitForFunction(requestId => window.__nimbusRegularHarness!.snapshot().ownership.cancelledReadIds.includes(requestId), pending);
    await page.evaluate(() => window.__nimbusRegularHarness!.setFrames([]));
    await expect.poll(async () => page.evaluate(() => {
      const state = window.__nimbusRegularHarness!.snapshot();
      return [state.ownership.pending.length, state.activeLoads, state.queuedLoads, state.prefetches];
    })).toEqual([0, 0, 0, 0]);
  });
  test(`${plot} joins the pending next frame without cancelling or duplicating its read`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&heldPrefetch=1`);
    await page.waitForFunction(() => (window.__nimbusRegularHarness?.snapshot().ownership.pending.length ?? 0) === 1);
    const pending = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().ownership.pending[0]);
    expect(pending).toMatchObject({ field: "w", step: 1 });
    const before = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot().deduplicatedRequests);
    await page.evaluate(() => window.__nimbusRegularHarness!.setStep(1));
    await page.waitForFunction(before => window.__nimbusRegularHarness!.snapshot().deduplicatedRequests > before, before);
    await page.evaluate(() => window.__nimbusRegularHarness!.releaseReads());
    await page.waitForFunction(plot => {
      const host = document.querySelector(".js-plotly-plot") as (HTMLElement & { data?: { x?: number[]; z?: number[][] }[] }) | null;
      const values = plot === "vertical" ? host?.data?.[0]?.x : host?.data?.[0]?.z?.flat();
      return values && values.length > 0 && values.every(value => value >= 11);
    }, plot);
    const state = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot());
    expect(state.calls.filter(call => call.command === "read_field_window" && call.field === "w" && call.step === 1)).toHaveLength(1);
    expect(state.ownership.cancelledReadIds).not.toContain(pending.requestId);
    expect([state.ownership.pending.length, state.activeLoads, state.queuedLoads, state.prefetches]).toEqual([0, 0, 0, 0]);
    await expect(page.locator(".plot-overlay--error")).toHaveCount(0);
  });
  test(`${plot} does not prefetch after its primary field read fails`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=regular-dataflow&regularPlot=${plot}&failedRegularField=w`);
    await expect(page.locator(".plot-overlay--error")).toContainText("primary field read failed");
    const state = await page.evaluate(() => window.__nimbusRegularHarness!.snapshot());
    expect(state.calls.filter(call => call.command === "read_field_window" && call.field === "w").map(call => call.step)).toEqual([0]);
    expect([state.activeLoads, state.queuedLoads, state.prefetches]).toEqual([0, 0, 0]);
  });
}
