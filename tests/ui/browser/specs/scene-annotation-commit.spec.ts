import { expect, test } from "@playwright/test";

test("Supercell includes initial axis geometry in its coherent volume draw", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.text().includes("flushSync")) errors.push(message.text()); });
  await page.addInitScript(() => {
    const shaders = new WeakSet<object>();
    const programs = new WeakMap<object, number>();
    const locations = new WeakMap<object, { program: number; name: string }>();
    const objects = new WeakMap<object, number>();
    let objectId = 0, programId = 0;
    const counts: string[][] = [];
    (window as unknown as { volumeDrawIdentities: typeof counts }).volumeDrawIdentities = counts;
    interface State { program: number; unit: number; bindings: Record<string, number>; versions: Record<string, number>;
      uniforms: Record<string, Record<string, unknown>>; viewport: unknown[]; draws: string[] }
    const states = new WeakMap<object, State>();
    const state = (gl: object): State => {
      let value = states.get(gl);
      if (!value) { value = { program: 0, unit: 33984, bindings: {}, versions: {}, uniforms: {}, viewport: [], draws: [] }; states.set(gl, value); counts.push(value.draws); }
      return value;
    };
    const id = (value: object | null) => { if (!value) return 0; let result = objects.get(value); if (!result) { result = ++objectId; objects.set(value, result); } return result; };
    for (const type of [WebGLRenderingContext, WebGL2RenderingContext]) {
      const prototype = type.prototype as unknown as Record<string, (...args: unknown[]) => unknown>;
      const wrap = (name: string, before: (gl: object, args: unknown[], result?: unknown) => void, after = false) => {
        const original = prototype[name];
        if (!original) return;
        prototype[name] = function(...args) { if (!after) before(this, args); const result = original.apply(this, args); if (after) before(this, args, result); return result; };
      };
      wrap("shaderSource", (_, [shader, source]) => { if (String(source).includes("nimbusCoefficients")) shaders.add(shader as object); });
      wrap("attachShader", (_, [program, shader]) => { if (shaders.has(shader as object) && !programs.has(program as object)) programs.set(program as object, ++programId); });
      wrap("useProgram", (gl, [program]) => { state(gl).program = programs.get(program as object) ?? 0; });
      wrap("getUniformLocation", (_, [program, name], result) => { if (result && programs.has(program as object)) locations.set(result as object, { program: programs.get(program as object)!, name: String(name) }); }, true);
      for (const name of ["uniform1f", "uniform1i", "uniform2f", "uniform3f", "uniform4f", "uniform1fv", "uniform2fv", "uniform3fv", "uniform4fv", "uniformMatrix3fv", "uniformMatrix4fv"]) wrap(name, (gl, [location, ...args]) => {
        const binding = locations.get(location as object); if (!binding) return;
        const values = state(gl).uniforms[binding.program] ??= {};
        values[binding.name] = args.map(value => ArrayBuffer.isView(value) ? Array.from(value as unknown as ArrayLike<number>) : value);
      });
      wrap("activeTexture", (gl, [unit]) => { state(gl).unit = Number(unit); });
      wrap("bindTexture", (gl, [target, texture]) => { const value = state(gl); value.bindings[`${value.unit}:${target}`] = id(texture as object | null); });
      for (const name of ["texImage2D", "texSubImage2D", "texStorage3D", "texImage3D", "texSubImage3D"]) wrap(name, (gl, [target]) => {
        const value = state(gl), texture = value.bindings[`${value.unit}:${target}`]; value.versions[texture] = (value.versions[texture] ?? 0) + 1;
      });
      wrap("viewport", (gl, args) => { state(gl).viewport = args; });
      wrap("drawArrays", gl => { const value = state(gl); if (value.program) value.draws.push(JSON.stringify({ program: value.program, uniforms: value.uniforms[value.program], viewport: value.viewport, bindings: value.bindings, versions: value.versions })); });
    }
  });
  await page.goto("/harness/index.html?mounted=0");
  await page.waitForFunction(() => Boolean(window.__nimbusVtkHarness));
  await page.evaluate(() => window.__nimbusVtkHarness!.setPlot("supercell"));
  await expect(page.locator(".harness-shell")).toHaveAttribute("data-plot", "supercell");
  await page.evaluate(() => window.__nimbusVtkHarness!.setMounted(true));
  await page.waitForFunction(() => {
    const scene = document.querySelector<HTMLElement>('[data-testid="scene"]')?.dataset;
    const host = document.querySelector<HTMLElement>(".plot3d-overlay-host")?.dataset;
    return scene?.renderedStep === "0" && scene.frameLoading === "false"
      && host?.scenePresentationPending === "false" && host.sceneFrameCommitPending === "false";
  });
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const draws = await page.evaluate(() => (window as unknown as { volumeDrawIdentities: string[][] }).volumeDrawIdentities.filter(value => value.length));
  expect(draws).toHaveLength(1);
  expect(draws[0].length).toBeGreaterThan(0);
  expect(new Set(draws[0]).size).toBe(draws[0].length);
  const labels = await page.locator(".plot3d-axis-labels__item").evaluateAll(elements => elements.map(element => {
    const label = element as HTMLElement;
    return { text: label.textContent, left: parseFloat(label.style.left), top: parseFloat(label.style.top) };
  }));
  expect(labels.length).toBeGreaterThan(3);
  expect(labels.every(label => label.text && Number.isFinite(label.left) && Number.isFinite(label.top))).toBe(true);
  expect((await page.evaluate(() => window.__nimbusVtkHarness!.inspectAppearance()))?.descriptor.frameIndex).toBe(0);
  const before = await page.evaluate(() => window.__nimbusVtkHarness!.captureSavedFrame());
  expect(before).toMatch(/^data:image\/png;base64,/);
  await page.evaluate(() => window.__nimbusVtkHarness!.setSize(512, 384));
  await expect.poll(() => page.locator(".plot3d-axis-labels__item").evaluateAll(elements => elements.map(element => {
    const label = element as HTMLElement;
    return { text: label.textContent, left: parseFloat(label.style.left), top: parseFloat(label.style.top) };
  }))).not.toEqual(labels);
  const resized = await page.locator(".plot3d-axis-labels__item").evaluateAll(elements => elements.map(element => {
    const label = element as HTMLElement;
    return { text: label.textContent, left: parseFloat(label.style.left), top: parseFloat(label.style.top) };
  }));
  expect(resized.length).toBeGreaterThan(3);
  expect(resized.every(label => label.text && Number.isFinite(label.left) && Number.isFinite(label.top))).toBe(true);
  const after = await page.evaluate(() => window.__nimbusVtkHarness!.captureSavedFrame());
  expect(after).toMatch(/^data:image\/png;base64,/);
  expect(after).not.toBe(before);
  expect((await page.evaluate(() => window.__nimbusVtkHarness!.inspectAppearance()))?.descriptor.frameIndex).toBe(0);
  expect(errors).toEqual([]);
});
