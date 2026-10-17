import { expect, test } from "@playwright/test";

test("remote visualization profiles validate, persist and expose connection ownership", async ({ page }) => {
  await page.goto("/harness/config.html?mode=workspace-settings");
  const panel = page.getByRole("region", { name: "Remote visualization workspace" });
  await panel.getByLabel("Workspace name", { exact: true }).fill("Teaching node");
  await panel.getByLabel("Workspace SSH host", { exact: true }).fill("research-node");
  await panel.getByLabel("Workspace URL path", { exact: true }).fill("//other/service");
  await panel.getByRole("button", { name: "Save workspace", exact: true }).click();
  await expect(panel.getByRole("alert")).toContainText("absolute URL path");
  await panel.getByLabel("Workspace URL path", { exact: true }).fill("/vnc.html");
  await panel.getByRole("button", { name: "Save workspace", exact: true }).click();
  await expect(panel.getByRole("button", { name: "Connect workspace", exact: true })).toBeDisabled();
  await page.reload();
  await expect(panel.getByLabel("Workspace SSH host", { exact: true })).toHaveValue("research-node");
  await expect(panel).toContainText("This does not move the current local plot or its data");
  await panel.getByRole("button", { name: "Remove workspace", exact: true }).click();
  await expect(panel.getByLabel("Saved workspace", { exact: true }).locator("option")).toHaveCount(1);
});

test("remote workspace cancellation, opening and dropped connections reflect native state", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => localStorage.setItem("nimbus-studio:remote-workspaces:v1", JSON.stringify({ version: 1, profiles: [
    { id: "research", name: "Research node", host: "node", port: 6080, path: "/vnc.html" },
  ] })));
  await page.goto("/harness/config.html?mode=remote-workspace");
  const connect = page.getByRole("button", { name: "Connect workspace", exact: true });
  const open = page.getByRole("button", { name: "Open remote desktop", exact: true });
  await connect.click();
  await expect(open).toBeDisabled();
  await page.getByRole("button", { name: "Cancel connection", exact: true }).click();
  await expect(connect).toBeEnabled();
  await connect.click();
  await expect(page.getByRole("status")).toContainText("Connecting to node");
  await expect(page.getByRole("button", { name: "Cancel connection", exact: true })).toBeEnabled();
  await page.evaluate(() => window.__nimbusRemoteWorkspaceFixture.release());
  await expect(open).toBeEnabled();
  await expect(connect).toBeDisabled();
  await open.click();
  await expect.poll(() => page.evaluate(() => window.__nimbusRemoteWorkspaceFixture.opened())).toBe(1);
  await page.evaluate(() => window.__nimbusRemoteWorkspaceFixture.disconnect());
  await expect(page.getByRole("alert")).toContainText("SSH disconnected");
  await expect(open).toBeDisabled();
  await expect(connect).toBeEnabled();
  expect(errors).toEqual([]);
});
