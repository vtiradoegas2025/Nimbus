import { expect, test } from "@playwright/test";

type BudgetGraph = HTMLElement & {
  data: { name: string; x: number[]; y: (number | null)[]; connectgaps: boolean }[];
  layout: { xaxis: { title: { text: string } }; yaxis: { title: { text: string } } };
};

for (const gap of [false, true]) test(`parcel budget preserves source tendencies and coordinates${gap ? " across missing samples" : ""}`, async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`/harness/index.html?mode=studio${gap ? "&trajectoryBudgetGap=1" : ""}`);
  await page.getByRole("button", { name: "Trajectories", exact: true }).click();
  await page.getByRole("button", { name: "Vorticity budget", exact: true }).click();
  await page.getByRole("combobox", { name: "Parcel", exact: true }).selectOption("2");
  await expect(page.locator(".gtitle")).toHaveText("Vorticity budget — parcel 2");
  const snapshot = () => page.evaluate(() => {
    const graph = document.querySelector(".plot-canvas") as BudgetGraph;
    return { data: graph.data, xTitle: graph.layout.xaxis.title.text, yTitle: graph.layout.yaxis.title.text };
  });
  const height = await snapshot();
  const terms = [Math.fround(.001), Math.fround(.002), Math.fround(-.0005)];
  const expected = [...terms, terms.reduce((sum, value) => sum + value, 0)];
  expect(height.data.map(trace => trace.name)).toEqual(["Tilting", "Stretching", "Baroclinic", "Sum"]);
  expect(height.xTitle).toBe("Height (km)");
  expect(height.yTitle).toBe("vorticity tendency (s⁻²)");
  for (const [index, trace] of height.data.entries()) {
    expect(trace.connectgaps).toBe(false);
    expect(trace.x).toEqual(Array.from({ length: 30 }, (_, sample) => (116 + sample * 55) / 1000));
    expect(trace.y).toEqual(Array.from({ length: 30 }, (_, sample) => gap && sample === 15 ? null : expected[index]));
  }
  await page.getByRole("button", { name: "Time", exact: true }).click();
  await expect.poll(async () => (await snapshot()).xTitle).toBe("t (s)");
  const time = await snapshot();
  for (const [index, trace] of time.data.entries()) {
    expect(trace.x).toEqual(Array.from({ length: 30 }, (_, sample) => sample * 10));
    expect(trace.y).toEqual(height.data[index].y);
  }
  await expect(page.locator(".plot-overlay--error, .error-boundary")).toHaveCount(0);
  expect(errors).toEqual([]);
});
