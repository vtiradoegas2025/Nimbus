import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 1420,
    strictPort: true,
    host: "127.0.0.1",
    // Cross-origin isolation enables SharedArrayBuffer, which is what carries
    // the Stop button's interrupt byte into the Pyodide worker. Both headers
    // must be present together; the production responses get the same pair via
    // tauri.conf.json's app.security.headers.
    //
    // require-corp rather than credentialless, which is what this used to send.
    // credentialless is a Chromium-only value: WebKit does not recognise it, so
    // it does not isolate at all and `crossOriginIsolated` is false — and on
    // macOS WebKit IS the shipped app, since Tauri renders in WKWebView. The
    // old value was chosen when require-corp rejected WebKit's same-origin
    // module workers; that is no longer true, and the browser-test build proves
    // it by booting the interpreter worker and cancelling a cell under WebKit
    // with require-corp set. Every response here already carries
    // Cross-Origin-Resource-Policy, which is what require-corp asks of them.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
    },
  },
  build: {
    // Every generated artifact lives under build/ so the package root stays
    // readable. Tauri reads this path from src-tauri/tauri.conf.json
    // (build.frontendDist) — keep the two in step.
    outDir: "build/dist",
    emptyOutDir: true,
    // Plotly is ~4 MB minified and ships in its own lazy chunk
    // (PlotStudio-*.js). Raise the warning threshold so local-app builds
    // don't pretend the chunk is a problem.
    chunkSizeWarningLimit: 6000,
  },
});
