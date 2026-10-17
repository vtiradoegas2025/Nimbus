import { test, expect } from "@playwright/test";

test("unchanged saved Supercell fields reuse preparation textures and scene across display edits", async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.addInitScript(() => {
    const counters = { preparations: 0, uploads: 0, compiles: 0 };
    Object.assign(window, { supercellWorkCounters: counters });
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, options) {
      if (message?.kind === "prepare" && message.request?.kind === "cloud") counters.preparations++;
      post.call(this, message, options as StructuredSerializeOptions);
    };
    for (const method of ["texImage3D", "texSubImage3D", "compileShader"] as const) {
      const prototype = WebGL2RenderingContext.prototype as unknown as Record<string, (...args: unknown[]) => unknown>;
      const original = prototype[method];
      prototype[method] = function(...args) {
        if (method === "compileShader") counters.compiles++; else counters.uploads++;
        return Reflect.apply(original, this, args);
      };
    }
  });
  await page.goto("/harness/index.html?mode=studio");
  await page.addStyleTag({ content: "#root{height:100vh;overflow:hidden}#root>.plot-studio{height:100vh;overflow:hidden}" });
  await page.waitForFunction(() => Boolean(window.__nimbusStudioHarness));
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.getByRole("button", { name: "Rendered supercell", exact: true }).click();
  const settled = () => page.waitForFunction(() => {
    const data = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    return data?.scenePresentationPending === "false" && data.committedSceneGeneration === data.requestedSceneGeneration
      && document.querySelector<HTMLElement>(".plot3d-control-notice.is-visible") == null
      && Array.from(document.querySelectorAll<HTMLCanvasElement>(".volume-render__canvas canvas")).every(canvas => {
        const rect = canvas.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0
          && Math.abs(canvas.width * rect.height - canvas.height * rect.width) <= rect.width + rect.height;
      });
  }, undefined, { timeout: 90_000 });
  await settled();
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const snapshot = () => page.evaluate(async () => ({ counters: { ...(window as unknown as { supercellWorkCounters: object }).supercellWorkCounters },
    resources: window.__nimbusStudioHarness!.resourceSnapshot(), captures: await window.__nimbusStudioHarness!.capturePaneAppearances(),
    scalars: window.__nimbusStudioHarness!.inspectPrimaryVolumeScalarImages() }));
  const before = await snapshot();
  for (const mode of ["Presentation", "Scientific", "Presentation", "Scientific"]) {
    const menu = page.locator(".plot-studio__view-menu");
    if (!(await menu.evaluate(element => (element as HTMLDetailsElement).open))) await menu.locator("summary").click();
    await menu.getByRole("group", { name: "3D rendering fidelity", exact: true }).getByRole("button", { name: mode, exact: true }).click();
    await menu.locator("summary").click();
    await settled();
  }
  await page.locator(".plot3d-control-tab", { hasText: "Cloud" }).click();
  await page.getByRole("slider", { name: "Cloud density", exact: true }).fill("1.4");
  await settled();
  const after = await snapshot();
  expect(after.counters).toEqual(before.counters);
  expect(after.scalars).toEqual(before.scalars);
  expect(after.resources.webglContexts).toBe(before.resources.webglContexts);
  expect(after.captures[0]!.evidence.descriptor.frameIndex).toBe(before.captures[0]!.evidence.descriptor.frameIndex);
  expect(after.captures[0]!.evidence.camera).toEqual(before.captures[0]!.evidence.camera);
  await info.attach("supercell-work-reuse", { body: JSON.stringify({ before, after }), contentType: "application/json" });
});
