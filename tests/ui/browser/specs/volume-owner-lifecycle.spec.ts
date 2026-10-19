import { expect, test, type Page } from "@playwright/test";

async function prepared(page: Page) {
  await page.waitForFunction(() => {
    const state = window.__nimbusVolumeOwner?.snapshot();
    return state?.current && !state.loading && !state.error && state.bricks > 0 && state.preparedBytes > 0;
  });
}

for (const owner of ["single", "nested"]) {
  test(`${owner} brick read retries the same selection after an I/O limit increase`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=volume-owner&owner=${owner}&rejectReads=1`);
    await expect.poll(() => page.evaluate(() => window.__nimbusVolumeOwner?.snapshot().error)).toContain("I/O budget exceeded");
    await page.evaluate(() => window.__nimbusVolumeOwner!.increaseIoLimit());
    await prepared(page);
  });
  test(`${owner} brick preparation releases disabled field ownership and can restart`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=volume-owner&owner=${owner}`);
    await prepared(page);
    expect(await page.evaluate(() => window.__nimbusVolumeOwner!.snapshot().workers)).toBeGreaterThan(0);
    await page.evaluate(() => window.__nimbusVolumeOwner!.enable(false));
    await expect.poll(() => page.evaluate(() => {
      const state = window.__nimbusVolumeOwner!.snapshot();
      return [state.bricks, state.preparedBytes, state.reservedBytes, state.workers, state.loading];
    })).toEqual([0, 0, 0, 0, false]);
    await page.evaluate(() => window.__nimbusVolumeOwner!.enable(true));
    await prepared(page);
  });

  test(`${owner} rejected brick generation has a settled error and recovers`, async ({ page }) => {
    await page.goto(`/harness/index.html?mode=volume-owner&owner=${owner}`);
    await prepared(page);
    await page.evaluate(() => window.__nimbusVolumeOwner!.limit(1));
    await expect.poll(() => page.evaluate(() => {
      const state = window.__nimbusVolumeOwner!.snapshot();
      return [state.current, state.loading, Boolean(state.error), state.bricks];
    })).toEqual([true, false, true, 0]);
    await page.evaluate(() => window.__nimbusVolumeOwner!.limit(2_000_000));
    await prepared(page);
  });
}

for (const owner of ["single", "nested"]) test(`${owner} resident admission failure cannot publish unowned prepared bricks`, async ({ page }) => {
  await page.goto(`/harness/index.html?mode=volume-owner&owner=${owner}&rejectResident=1`);
  await page.waitForFunction(() => (window.__nimbusVolumeOwner?.snapshot().rejectedResidency ?? 0) > 0);
  await expect.poll(() => page.evaluate(() => {
    const state = window.__nimbusVolumeOwner!.snapshot();
    return { error: state.error, bricks: state.bricks, loading: state.loading, preparedBytes: state.preparedBytes };
  })).toEqual({ error: "Analytical fixture resident capacity rejection", bricks: 0, loading: false, preparedBytes: 0 });
  await page.evaluate(() => { window.__nimbusVolumeOwner!.permitResidency(); window.__nimbusVolumeOwner!.enable(false); });
  await page.waitForFunction(() => window.__nimbusVolumeOwner!.snapshot().workers === 0);
  await page.evaluate(() => window.__nimbusVolumeOwner!.enable(true));
  await prepared(page);
});

test("required volume background retries its failed read after an I/O limit increase", async ({ page }) => {
  await page.goto("/harness/index.html?mode=volume-owner&owner=background&rejectReads=1");
  await expect.poll(() => page.evaluate(() => window.__nimbusVolumeOwner?.snapshot().error)).toContain("I/O budget exceeded");
  await page.evaluate(() => window.__nimbusVolumeOwner!.increaseIoLimit());
  await page.waitForFunction(() => {
    const state = window.__nimbusVolumeOwner!.snapshot();
    return state.current && !state.error && !state.loading && state.bricks === 1;
  });
});

test("single brick supersession terminates a pending preparation and releases its construction receipt", async ({ page }) => {
  await page.goto("/harness/index.html?mode=volume-owner&owner=single&holdPreparation=1");
  await page.waitForFunction(() => (window.__nimbusVolumeOwner?.snapshot().heldPreparations ?? 0) > 0);
  expect(await page.evaluate(() => window.__nimbusVolumeOwner!.snapshot().reservedBytes)).toBeGreaterThan(0);
  await page.evaluate(() => window.__nimbusVolumeOwner!.limit(1));
  await expect.poll(() => page.evaluate(() => {
    const state = window.__nimbusVolumeOwner!.snapshot();
    return [state.loading, Boolean(state.error), state.bricks, state.reservedBytes, state.workers];
  })).toEqual([false, true, 0, 0, 0]);
  await page.evaluate(() => { window.__nimbusVolumeOwner!.permitPreparation(); window.__nimbusVolumeOwner!.limit(2_000_000); });
  await prepared(page);
});
