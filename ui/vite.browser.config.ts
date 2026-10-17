import "../tests/ui/prepare.mjs";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { beastReferenceBridgePlugin } from "../tests/ui/browser/support/realRunBridge";

const integrationRoot = fileURLToPath(new URL("../tests/ui/browser", import.meta.url));
const uiRoot = fileURLToPath(new URL(".", import.meta.url));

/** Dedicated test server. This entry point is never included in the Tauri build. */
export default defineConfig({
  root: integrationRoot,
  publicDir: fileURLToPath(new URL("./build/test-public", import.meta.url)),
  plugins: [
    react(),
    beastReferenceBridgePlugin(process.env.NIMBUS_BEAST_XL_REFERENCE_RUN),
  ].filter((plugin) => plugin != null),
  clearScreen: false,
  /*
   * Point the Python tab at the interpreter vendored under
   * build/test-public/pyodide by scripts/vendor_pyodide.mjs. The app build
   * leaves this unset and keeps the pinned CDN release; only the test bundle is
   * redirected, so specs boot a real CPython without depending on network
   * weather. Substituted textually, which is why the runtime reads the whole
   * `import.meta.env.VITE_PYODIDE_BASE_URL` expression rather than
   * destructuring it.
   */
  define: {
    "import.meta.env.VITE_PYODIDE_BASE_URL": JSON.stringify("/pyodide/"),
  },
  resolve: {
    alias: {
      "@nimbus": fileURLToPath(new URL("./src", import.meta.url)),
      // src/** imports itself through @/; the harness bundles those modules.
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 4178,
    strictPort: true,
    fs: { allow: [uiRoot, integrationRoot] },
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4178,
    strictPort: true,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
    },
  },
  build: {
    target: "esnext",
    outDir: fileURLToPath(new URL("./build/browser-test-dist", import.meta.url)),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        vtk: fileURLToPath(new URL("../tests/ui/browser/harness/index.html", import.meta.url)),
        config: fileURLToPath(new URL("../tests/ui/browser/harness/config.html", import.meta.url)),
        runLibrary: fileURLToPath(new URL("../tests/ui/browser/harness/run-library.html", import.meta.url)),
        queuePanel: fileURLToPath(new URL("../tests/ui/browser/harness/queue-panel.html", import.meta.url)),
        resourceBudget: fileURLToPath(new URL("../tests/ui/browser/harness/resource-budget.html", import.meta.url)),
        investigation: fileURLToPath(new URL("../tests/ui/browser/harness/investigation.html", import.meta.url)),
        cm1Report: fileURLToPath(new URL("../tests/ui/browser/harness/cm1-report.html", import.meta.url)),
        python: fileURLToPath(new URL("../tests/ui/browser/harness/python.html", import.meta.url)),
      },
    },
  },
});
