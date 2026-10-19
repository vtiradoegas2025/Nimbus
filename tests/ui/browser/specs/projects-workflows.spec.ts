import { expect, test, type Page, type Locator } from "@playwright/test";

const select = (page: Page, name: string) => page.getByRole("button", { name: `Select ${name}`, exact: true }).click();
const details = (page: Page) => page.getByRole("region", { name: "Project details", exact: true });
async function reachable(control: Locator) {
  await control.scrollIntoViewIfNeeded();
  expect(await control.evaluate(el => {
    const box = el.getBoundingClientRect(), hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
    return box.width > 0 && box.height > 0 && (hit === el || el.contains(hit));
  })).toBe(true);
}
test.beforeEach(async ({ page }) => {
  page.on("pageerror", error => { throw error; });
  await page.goto("/harness/config.html?mode=projects");
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1100, height: 720 }, { width: 620, height: 700 }]) {
  test(`project library and details fit ${viewport.width}x${viewport.height}`, async ({ page }, info) => {
    await page.setViewportSize(viewport);
    await select(page, "CM1 sounding review");
    await expect(details(page)).toContainText("long-source-directory/");
    await expect(details(page).getByRole("button", { name: "Relink comparison…", exact: true })).toBeVisible();
    const panel = page.getByRole("region", { name: "Projects", exact: true });
    expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    expect(await details(page).evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    for (const button of await details(page).getByRole("button").all()) await reachable(button);
    await details(page).evaluate(el => { el.scrollTop = 0; });
    await page.screenshot({ path: info.outputPath("projects-library.png") });
    await page.getByRole("button", { name: /Examples and workflows/ }).click();
    await reachable(page.getByRole("button", { name: "Open included example", exact: true }));
    await page.getByRole("button", { name: "Open included example", exact: true }).click();
    expect(await page.evaluate(() => window.__nimbusProjectsFixture.events().some(name => name.startsWith("example:")))).toBe(true);
  });
}

test("search, source filters, ordering and selection use metadata without loading sources", async ({ page }) => {
  const rows = page.locator(".project-navigator__row");
  await expect(rows).toHaveCount(3);
  await expect(rows.first()).toContainText("CM1 sounding review");
  await page.getByRole("combobox", { name: "Sort projects", exact: true }).selectOption("name");
  await expect(rows.first()).toContainText("Classic supercell");
  await page.getByRole("searchbox", { name: "Find projects", exact: true }).fill("theta_prime");
  await expect(rows).toHaveCount(1);
  await expect(details(page)).toContainText("Research workspace");
  await expect(details(page).locator("dd").last()).toHaveText("24");
  await page.getByRole("searchbox", { name: "Find projects", exact: true }).fill("missing/cm1");
  await expect(rows).toHaveCount(1);
  await expect(details(page).getByRole("heading", { name: "CM1 sounding review" })).toBeVisible();
  await page.getByRole("combobox", { name: "Data source", exact: true }).selectOption("nimbus_run");
  await expect(page.getByRole("heading", { name: "No matching projects" })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(rows).toHaveCount(3);
  expect(await page.evaluate(() => window.__nimbusProjectsFixture.events())).toEqual([]);
});

test("rename, duplicate and confirmed deletion persist while failures keep the project", async ({ page }) => {
  await select(page, "Classic supercell");
  await details(page).getByRole("button", { name: "Rename", exact: true }).click();
  await page.getByLabel("Project name", { exact: true }).fill("Renamed research");
  await page.getByRole("button", { name: "Save name", exact: true }).click();
  await expect(details(page).getByRole("heading", { name: "Renamed research" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "Select Renamed research", exact: true })).toBeVisible();
  await details(page).getByRole("button", { name: "Duplicate", exact: true }).click();
  await expect(page.locator(".project-navigator__row")).toHaveCount(4);
  await expect(details(page).getByRole("heading", { name: "Renamed research copy" })).toBeVisible();
  await details(page).getByRole("button", { name: "Delete project", exact: true }).click();
  await page.getByRole("button", { name: "Keep project", exact: true }).click();
  await expect(page.locator(".project-navigator__row")).toHaveCount(4);
  await details(page).getByRole("button", { name: "Delete project", exact: true }).click();
  await page.evaluate(() => window.__nimbusProjectsFixture.failNext());
  await page.getByRole("button", { name: "Confirm delete", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Fixture storage rejected");
  await expect(page.locator(".project-navigator__row")).toHaveCount(4);
  await page.getByRole("button", { name: "Confirm delete", exact: true }).click();
  await expect(page.locator(".project-navigator__row")).toHaveCount(3);
  await page.reload();
  await expect(page.getByRole("button", { name: "Select Renamed research copy", exact: true })).toHaveCount(0);
});

test("source verification stays visible and prevents overlapping project actions", async ({ page }) => {
  await select(page, "CM1 sounding review");
  await page.evaluate(() => window.__nimbusProjectsFixture.holdVerification());
  await details(page).getByRole("button", { name: "Verify sources", exact: true }).click();
  await expect(page.getByRole("button", { name: "Checking sources…", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Select Classic supercell", exact: true })).toBeDisabled();
  await expect(details(page).getByRole("button", { name: "Duplicate", exact: true })).toBeDisabled();
  await page.evaluate(() => window.__nimbusProjectsFixture.releaseVerification());
  await expect(page.getByRole("status")).toContainText("CM1 sounding review: 1 of 2 project sources");
  await expect(page.getByRole("status")).toContainText("missing");
  expect(await page.evaluate(() => window.__nimbusProjectsFixture.events().filter(name => name.startsWith("verify:")))).toEqual(["verify:project_cm1-fixture"]);
  await details(page).getByRole("button", { name: "Relink primary…", exact: true }).click();
  await expect(details(page)).toContainText("/relinked/source");
  await details(page).getByRole("button", { name: "Verify sources", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("All 2 project sources are available");
});

test("recipes round-trip and oversized imports are rejected before dispatch", async ({ page }) => {
  await details(page).getByRole("button", { name: "Publish recipe", exact: true }).click();
  const publication = await page.evaluate(() => window.__nimbusProjectsFixture.publication());
  expect(JSON.parse(publication).fieldData).toBe("references_only");
  await page.locator('input[type="file"]').setInputFiles({ name: "project.json", mimeType: "application/json", buffer: Buffer.from(publication) });
  await expect(page.locator(".project-navigator__row")).toHaveCount(4);
  const before = await page.evaluate(() => window.__nimbusProjectsFixture.events().length);
  await page.locator('input[type="file"]').setInputFiles({ name: "oversized.json", mimeType: "application/json", buffer: Buffer.alloc(750001, "x") });
  await expect(page.getByRole("alert")).toContainText("750 kB");
  expect(await page.evaluate(() => window.__nimbusProjectsFixture.events().length)).toBe(before);
});

test("the ordinary Projects page reports source checks and failed saves beside the project", async ({ page }) => {
  await page.goto("/harness/run-library.html");
  await page.getByRole("button", { name: "Projects", exact: true }).click();
  await page.getByRole("button", { name: "Open included example", exact: true }).click();
  await expect(page.getByRole("button", { name: "Plots", exact: true })).toHaveAttribute("aria-current", "page");
  await page.getByRole("button", { name: "Projects", exact: true }).click();
  const title = await details(page).locator("h3").first().textContent();
  await details(page).getByRole("button", { name: "Verify sources", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("verification requires Nimbus Studio desktop");
  await details(page).getByRole("button", { name: "Rename", exact: true }).click();
  await page.getByLabel("Project name", { exact: true }).fill("Must not claim saved");
  await page.evaluate(() => {
    const write = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key: string, value: string) {
      if (key.startsWith("nimbus:studio:projects:v1")) throw new Error("fixture disk full");
      return write.call(this, key, value);
    };
  });
  await page.getByRole("button", { name: "Save name", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Could not save project changes");
  await expect(details(page).locator("h3").first()).toHaveText(title!);
  await expect(page.getByLabel("Project name", { exact: true })).toHaveValue("Must not claim saved");
});
