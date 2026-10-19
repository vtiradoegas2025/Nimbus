import "../tests/ui/prepare.mjs";
import { defineConfig, devices } from "@playwright/test";
import { arch, availableParallelism, cpus, platform, totalmem } from "node:os";

const executablePath = process.env.NIMBUS_BROWSER_EXECUTABLE;
const lavapipeEnabled = process.env.NIMBUS_BROWSER_LAVAPIPE === "1";
const hardwareWebGpuEnabled = process.env.NIMBUS_WEBGPU_HARDWARE === "1";
const windowsHardwarePerformanceEnabled = process.env.NIMBUS_WINDOWS_UI_PERFORMANCE === "1";
const hardwareWebGpuExecutable = process.env.NIMBUS_WEBGPU_EXECUTABLE
  ?? executablePath
;

export default defineConfig({
  testDir: "../tests/ui/browser",
  testMatch: /^(?!.*webgpu-hardware\.spec\.ts$).*\.spec\.ts$/,
  // Shard individual tests while retaining one browser worker per job.
  fullyParallel: true,
  workers: 1,
  globalSetup: "../tests/ui/browser/verify_renderer.ts",
  metadata: {
    hostOs: platform(),
    hostArchitecture: arch(),
    availableCpuCount: availableParallelism(),
    cpuModel: cpus()[0]?.model ?? "unavailable",
    totalRamMiB: Math.round(totalmem() / (1024 * 1024)),
  },
  /*
   * Ceiling for ordinary specs. The soak test raises its own via
   * test.setTimeout, because its runtime is set by NIMBUS_SOAK_MINUTES and a
   * fixed ceiling here cannot bound it: at NIMBUS_SOAK_MINUTES=30 the soak
   * loop alone consumed exactly this budget, leaving nothing for the
   * assertions, the teardown, or the metrics write, so the 30-minute soak
   * timed out every time no matter what it measured.
   */
  timeout: 10 * 60_000,
  expect: { timeout: 20_000 },
  // Generated artifacts are consolidated under build/ (see vite.config.ts).
  // CI uploads both of these paths — keep .github/workflows/ci.yml in step.
  outputDir: "build/test-results/playwright",
  reporter: process.env.CI
    ? [["line"], ["html", { open: "never", outputFolder: "build/playwright-report" }]]
    : "line",
  use: {
    baseURL: "http://127.0.0.1:4178",
    actionTimeout: 30_000,
    navigationTimeout: 45_000,
    viewport: { width: 1280, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [{
    command: "npm run preview:browser-test",
    // The harness entry points live under /harness/, so the server root is
    // empty and cannot be used as the readiness probe.
    url: "http://127.0.0.1:4178/harness/index.html",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  }, {
    command: "npm run preview:production-test",
    url: "http://127.0.0.1:4310/",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  }],
  projects: [
    {
      name: lavapipeEnabled ? "chromium-lavapipe" : "chromium-swiftshader",
      use: {
        ...devices["Desktop Chrome"],
        actionTimeout: 60_000,
        launchOptions: {
          executablePath,
          args: [
            ...(lavapipeEnabled ? [
              "--use-angle=vulkan",
              "--use-vulkan=native",
              "--enable-features=Vulkan",
              "--disable-vulkan-surface",
              "--ignore-gpu-blocklist",
            ] : [
              "--use-gl=swiftshader",
              "--enable-unsafe-swiftshader",
              "--disable-gpu-sandbox",
            ]),
            // Lets the soak force a collection before reading the heap.
            // usedJSHeapSize climbs between collections, so without this a
            // "did memory plateau" check is really asking where in the GC
            // cycle the last sample happened to land. WebKit exposes neither
            // gc() nor performance.memory, so the soak's heap check is
            // chromium-only by construction and says so in its metrics.
            "--js-flags=--expose-gc",
            /*
             * Makes usedJSHeapSize a live number instead of a cached bucket.
             *
             * Chrome quantizes performance.memory and refreshes it at most
             * every ~20 minutes, for fingerprinting reasons. Without this flag
             * the soak's baseline (taken a third of the way in) and its final
             * read are routinely the SAME cached value, so the retention
             * assertion compares a number to itself and the monotonic-growth
             * check sees a flat series it can never fail on. Measured
             * directly: mounting five different plot types twenty times each
             * returned a byte-identical 22.0 MiB every time without the flag,
             * and a differentiated per-plot slope with it.
             */
            "--enable-precise-memory-info",
          ],
        },
      },
    },
    {
      name: "webkit-standard",
      use: {
        ...devices["Desktop Safari"],
        deviceScaleFactor: 1,
      },
    },
    {
      name: "webkit-retina",
      use: {
        ...devices["Desktop Safari"],
        deviceScaleFactor: 2,
      },
    },
    ...(hardwareWebGpuEnabled ? [{
      name: "chromium-webgpu-hardware",
      testMatch: /webgpu-hardware\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          executablePath: hardwareWebGpuExecutable,
          headless: false,
          args: [
            "--enable-unsafe-webgpu",
            "--ignore-gpu-blocklist",
          ],
        },
      },
    }] : []),
    ...(windowsHardwarePerformanceEnabled ? [{
      // Real Windows/Edge interaction timings are a separate release record
      // from the deterministic SwiftShader regression lane. The self-hosted
      // runner label promises an interactive desktop and physical adapter.
      name: "chromium-windows-hardware-performance",
      testMatch: /supercell-responsiveness\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          executablePath: hardwareWebGpuExecutable,
          headless: false,
          args: ["--ignore-gpu-blocklist"],
        },
      },
    }] : []),
  ],
});
