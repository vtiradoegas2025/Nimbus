import { expect, test, type Page } from "@playwright/test";

/*
 * Smoke test of the PRODUCTION bundle, not the test harness.
 *
 * Every other spec mounts a component into a purpose-built harness page. That
 * is the right way to test behaviour, but it cannot catch a fault in the real
 * app shell — routing, lazy chunk boundaries, the minified build. The defect
 * that started this work was exactly that shape: a hooks-order violation that
 * only surfaced in a minified production build, as React error #310, and took
 * the whole Plots view down behind an error boundary.
 *
 * So this drives `build/dist` as shipped. It runs without the Tauri IPC layer,
 * which is honest for a shell test: it proves the app boots, routes, and
 * renders its chrome. Anything that needs a dataset is covered by the harness
 * specs.
 */

const PROD_URL = "http://127.0.0.1:4310/";

test.beforeAll(async () => {
  const response = await fetch(PROD_URL, { signal: AbortSignal.timeout(2000) });
  expect(response.ok, `production preview at ${PROD_URL} returned ${response.status}`).toBe(true);
});

/** Anything React logs at error level, plus uncaught exceptions. */
function collectFailures(page: Page): string[] {
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(`console: ${message.text()}`);
  });
  return failures;
}

test("the production shell boots and routes without a React error", async ({ page }) => {
  const failures = collectFailures(page);
  await page.goto(PROD_URL, { waitUntil: "load" });

  // The app shell itself.
  await expect(page.locator(".app")).toBeVisible();
  await expect(page.locator(".nav-rail")).toBeVisible();

  /*
   * Visit every primary destination. Lazy chunk boundaries are per-route, so a
   * bundle-level fault can hide on a page the smoke test never opens — and
   * Plots is both the heaviest chunk and the one that was blank.
   */
  for (const destination of ["Projects", "Configure", "Runs", "Plots", "Settings"]) {
    await page.getByRole("button", { name: destination, exact: true }).click();
    await expect(page.locator(".app")).toBeVisible();
    // The error boundary's copy is the exact thing the user photographed.
    await expect(page.getByText("This view failed")).toHaveCount(0);
  }

  expect(failures, `console/page errors:\n${failures.join("\n")}`).toEqual([]);
});

/*
 * React #310 is a hooks-COUNT change between renders, so it needs a state
 * transition to trigger — mounting Plots once proves nothing. Leaving and
 * returning re-mounts it, and the datasetless -> chrome -> datasetless path is
 * the one that shipped broken.
 */
test("the Plots view survives being entered, left and re-entered", async ({ page }) => {
  const failures = collectFailures(page);
  await page.goto(PROD_URL, { waitUntil: "load" });

  for (let visit = 0; visit < 3; visit++) {
    await page.getByRole("button", { name: "Plots", exact: true }).click();
    await expect(page.getByText("This view failed")).toHaveCount(0);
    await page.getByRole("button", { name: "Configure", exact: true }).click();
  }

  await page.getByRole("button", { name: "Plots", exact: true }).click();
  await expect(page.getByText("This view failed")).toHaveCount(0);
  expect(failures, `console/page errors:\n${failures.join("\n")}`).toEqual([]);
});

/** The plot-side preference is new, and it is the one setting that restyles the shell. */
test("the plot-side preference is offered and moves the shell", async ({ page }) => {
  const failures = collectFailures(page);
  await page.goto(PROD_URL, { waitUntil: "load" });
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();

  const sideGroup = page.getByRole("radiogroup", { name: "Plot side" });
  await expect(sideGroup).toBeVisible();
  await sideGroup.getByRole("radio", { name: "Left" }).click();

  // Saved settings publish a root data attribute that CSS selects on; the
  // in-session apply is what the user sees immediately.
  await page.getByRole("button", { name: /Save settings/ }).click();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.dataset.plotSide))
    .toBe("left");

  const railBox = await page.locator(".nav-rail").boundingBox();
  const appBox = await page.locator(".app").boundingBox();
  if (!railBox || !appBox) throw new Error("shell is not laid out");
  // The rail moves to the right half: the visual side flipped.
  expect(railBox.x).toBeGreaterThan(appBox.x + appBox.width / 2);

  expect(failures, `console/page errors:\n${failures.join("\n")}`).toEqual([]);
});

test("comfortable text size persists through a production reload", async ({ page }) => {
  await page.goto(PROD_URL, { waitUntil: "load" });
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings categories" }).getByRole("button", { name: "Appearance", exact: true }).click();
  await page.getByRole("combobox", { name: "Text size", exact: true }).selectOption("comfortable");
  await page.getByRole("button", { name: /Save settings/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-text-size", "comfortable");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-text-size", "comfortable");
});
