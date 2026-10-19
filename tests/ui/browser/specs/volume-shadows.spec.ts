import { expect, test } from "@playwright/test";

test("volume shadow integration preserves physical attenuation across ray strides and display scales", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  const rows = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeShadows());
  const upstream = await page.evaluate(() => window.__nimbusVtkHarness!.qualifyVolumeShadows(true));
  expect(rows).toHaveLength(324);
  await testInfo.attach("physical-shadow-integration", { body: JSON.stringify(rows), contentType: "application/json" });
  await testInfo.attach("installed-vtk-shadow-reference", { body: JSON.stringify(upstream), contentType: "application/json" });
  console.log("SHADOW_ERROR", JSON.stringify({
    installedVtk: Math.max(...upstream.map((row) => Math.abs(row.actualShadow - row.expectedShadow))),
    repaired: Math.max(...rows.map((row) => Math.abs(row.actualShadow - row.expectedShadow))),
  }));
  for (const row of rows) {
    expect(Math.abs(row.actualShadow - row.expectedShadow), JSON.stringify(row)).toBeLessThan(1.5 / 255);
    expect(Math.abs(row.actualPrimary - row.expectedPrimary), JSON.stringify(row)).toBeLessThan(1.5 / 255);
  }
  expect(errors).toEqual([]);
});
