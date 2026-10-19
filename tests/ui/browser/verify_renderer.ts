import { chromium, type FullConfig } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export default async function verifyRenderer(config: FullConfig) {
  const project = config.projects.find(candidate => candidate.name === "chromium-lavapipe");
  if (!project) return;
  if (process.platform !== "linux" || !process.env.VK_ICD_FILENAMES) {
    throw new Error("The lavapipe profile requires Linux and an explicit VK_ICD_FILENAMES selection");
  }
  const browser = await chromium.launch(project.use.launchOptions);
  try {
    const page = await browser.newPage();
    const identity = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 16;
      const gl = canvas.getContext("webgl2");
      if (!gl) throw new Error("The selected browser cannot create a WebGL2 context");
      const debug = gl.getExtension("WEBGL_debug_renderer_info");
      if (!debug) throw new Error("The selected browser does not expose its renderer identity");
      gl.clearColor(0.25, 0.5, 0.75, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.finish();
      const pixel = new Uint8Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      return {
        renderer: String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)),
        vendor: String(gl.getParameter(debug.UNMASKED_VENDOR_WEBGL)),
        userAgent: navigator.userAgent,
        contextLost: gl.isContextLost(),
        error: gl.getError(),
        pixel: Array.from(pixel),
      };
    });
    const directory = path.join(project.outputDir, "renderer");
    await mkdir(directory, { recursive: true });
    await writeFile(path.join(directory, "lavapipe.json"), JSON.stringify({
      ...identity,
      icd: process.env.VK_ICD_FILENAMES,
      threads: process.env.LP_NUM_THREADS ?? null,
      launchOptions: project.use.launchOptions,
    }, null, 2) + "\n");
    if (!/Mesa.*Vulkan.*llvmpipe/i.test(identity.renderer)
      || /SwiftShader/i.test(identity.renderer)
      || identity.contextLost || identity.error !== 0
      || identity.pixel.some((value, index) => Math.abs(value - [64, 128, 191, 255][index]) > 1)) {
      throw new Error(`The lavapipe profile did not produce a valid Mesa software Vulkan image: ${JSON.stringify(identity)}`);
    }
  } finally {
    await browser.close();
  }
}
