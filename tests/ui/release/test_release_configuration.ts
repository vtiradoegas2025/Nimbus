import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

let passed = 0;
const failures: string[] = [];

function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else failures.push(`FAIL ${label}`);
}

const here = dirname(fileURLToPath(import.meta.url));
const uiRoot = resolve(here, "../../../ui");
const tauri = JSON.parse(
  readFileSync(resolve(uiRoot, "src-tauri/tauri.conf.json"), "utf8"),
) as {
  app?: {
    windows?: Array<{ backgroundThrottling?: string }>;
    security?: {
      csp?: null | Record<string, string>;
      headers?: Record<string, string>;
    };
  };
};
const packageJson = JSON.parse(
  readFileSync(resolve(uiRoot, "package.json"), "utf8"),
) as { scripts?: Record<string, string> };
const nativeBuild = readFileSync(resolve(uiRoot, "scripts/build_native_app.sh"), "utf8");
const ci = readFileSync(resolve(uiRoot, "../.github/workflows/ci.yml"), "utf8");

for (const platform of ["macos", "linux", "windows"]) {
  const config = JSON.parse(readFileSync(resolve(uiRoot, `src-tauri/tauri.${platform}.conf.json`), "utf8"));
  check(config.bundle.resources["../../vulkan/shaders"] === "vulkan/shaders", `${platform} bundles simulator shaders at its executable-relative lookup path`);
}

const csp = tauri.app?.security?.csp;
check(csp != null && typeof csp === "object", "production Tauri CSP is enabled");
if (csp) {
  check(csp["default-src"]?.includes("'self'"), "default sources fail closed to the app");
  check(csp["object-src"] === "'none'", "object embedding is disabled");
  check(csp["frame-ancestors"] === "'none'", "the privileged app cannot be framed");
  check(!csp["script-src"]?.includes("'unsafe-inline'"), "inline scripts are not globally enabled");
  check(csp["worker-src"]?.includes("'self'"), "bundled computation workers remain available");
  check(csp["connect-src"]?.includes("ipc:"), "Tauri IPC remains explicitly available");
}

const headers = tauri.app?.security?.headers ?? {};
check(headers["Cross-Origin-Opener-Policy"] === "same-origin", "COOP isolates the application");
check(headers["Cross-Origin-Embedder-Policy"] === "require-corp", "COEP supports interrupt buffers");
check(
  tauri.app?.windows?.every((window) => window.backgroundThrottling === "disabled") === true,
  "packaged windows keep scientific workers and lifecycle watchdogs active when occluded",
);
check(headers["Cross-Origin-Resource-Policy"] === "same-origin", "bundled resources are same-origin");
check(packageJson.scripts?.["test:unit"] === "node ../tests/ui/run_unit_tests.mjs", "aggregate unit gate is wired");
check(
  nativeBuild.includes('codesign --verify --deep --strict "$SOURCE_APP"'),
  "bundle-only macOS builds are strictly signature-verified",
);
check(
  ci.includes('codesign --verify --deep --strict "$app"'),
  "CI rejects an unsealed macOS bundle",
);

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else {
  console.log(`release configuration tests: ${passed} passed, 0 failed.`);
}
