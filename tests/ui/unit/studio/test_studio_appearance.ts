import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  STUDIO_SETTINGS_DEFAULTS,
  isStudioAppearance,
  normalizeStudioSettings,
} from "../../../../ui/src/lib/studio/studioSettings.ts";
import { resolveStudioAppearance } from "../../../../ui/src/lib/studio/studioAppearance.ts";

let checks = 0;
const check = (condition: boolean, message: string) => {
  if (!condition) throw new Error(message);
  checks += 1;
};

check(resolveStudioAppearance("system", true) === "dark", "system resolves a dark OS preference");
check(resolveStudioAppearance("system", false) === "light", "system resolves a light OS preference");
check(resolveStudioAppearance("light", true) === "light", "explicit light overrides the OS");
check(resolveStudioAppearance("dark", false) === "dark", "explicit dark overrides the OS");
check(["system", "light", "dark"].every(isStudioAppearance), "all supported appearances validate");
check(!isStudioAppearance("blue") && !isStudioAppearance(null), "unknown appearances fail validation");

const migrated = normalizeStudioSettings({ runsDirectory: "/research/runs" });
check(migrated.appearance === "system", "legacy records migrate to system appearance");
check(migrated.runsDirectory === "/research/runs", "legacy string settings remain intact");
const invalid = normalizeStudioSettings({ appearance: "neon", runsDirectory: 42, defaultTemplate: null });
check(invalid.appearance === STUDIO_SETTINGS_DEFAULTS.appearance, "invalid appearance falls back safely");
check(invalid.runsDirectory === STUDIO_SETTINGS_DEFAULTS.runsDirectory, "invalid string settings fall back safely");
check(normalizeStudioSettings(null).ffmpegPathOverride === "", "non-object storage falls back safely");

/* ---------- Plotly chrome must follow the theme ----------
 * Plotly renders its own SVG and cannot read CSS custom properties, so plot
 * components resolve the --plot-* tokens through usePlotChrome. A component
 * that inlines a dark literal instead keeps dark chrome on the light theme:
 * gridlines vanish into the paper and legend/annotation boxes go dark-on-dark.
 * These checks keep both halves of that contract honest. */

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "..", "..", "..", "..", "ui", "src");
const themeCss = readFileSync(join(SRC, "styles", "theme.css"), "utf-8");

const PLOT_CHROME_TOKENS = [
  "--plot-gridline",
  "--plot-zeroline",
  "--plot-overlay-bg",
  "--plot-ink",
  "--plot-text",
];

/** Read a token's value out of a single CSS block. */
const tokenIn = (block: string, token: string): string | null => {
  const match = block.match(new RegExp(`${token}\\s*:\\s*([^;]+);`));
  return match ? match[1].trim() : null;
};

const lightBlockStart = themeCss.indexOf(':root[data-theme="light"]');
check(lightBlockStart > 0, "theme.css declares a light theme block");
const darkBlock = themeCss.slice(0, lightBlockStart);
const lightBlock = themeCss.slice(lightBlockStart);

for (const token of PLOT_CHROME_TOKENS) {
  const dark = tokenIn(darkBlock, token);
  const light = tokenIn(lightBlock, token);
  check(dark != null, `${token} is defined for the dark theme`);
  check(light != null, `${token} is defined for the light theme`);
  check(dark !== light, `${token} actually changes between themes`);
  // Plotly parses colors with tinycolor, which does not understand oklch() and
  // silently falls back to black. Keep these tokens in rgb/rgba/hex.
  check(
    !/oklch|color-mix|lab\(|lch\(/.test(`${dark ?? ""}${light ?? ""}`),
    `${token} uses a Plotly-parseable color space`,
  );
}

// No plot component may inline a chrome color; they must come from the tokens.
// The fallback table inside common.tsx is the one intentional exception — it
// only applies when there is no document to resolve against.
const plotsDir = join(SRC, "components", "plots");
const CHROME_LAYOUT_KEYS = /\b(gridcolor|zerolinecolor|bgcolor)\s*:\s*("|')/;
for (const file of readdirSync(plotsDir).filter((name) => name.endsWith(".tsx"))) {
  const source = readFileSync(join(plotsDir, file), "utf-8");
  for (const line of source.split("\n")) {
    const match = line.match(CHROME_LAYOUT_KEYS);
    if (!match) continue;
    // `transparent` inherits the themed page background, so it is always safe.
    check(
      /:\s*("|')(?:transparent|rgba\(0,\s*0,\s*0,\s*0\))\1/.test(line),
      `${file} resolves Plotly chrome from the theme instead of a literal: ${line.trim()}`,
    );
  }
}

/* ---------- Trace colors must be legible on the default (dark) theme ----------
 * The chrome checks above cover gridlines and backgrounds; the DATA is a
 * separate exposure. PlotCanvas renders onto a transparent paper, so a trace
 * whose color is hardcoded is drawn straight onto the page background and is
 * legible in at most one theme. VectorField shipped `line: { color: "#1f2933" }`
 * — measured at 1.34:1 against the resolved --color-bg, i.e. an apparently
 * empty plot on the theme most users are looking at. Every other trace literal
 * in the directory measures 6:1 or better. */

/** WCAG relative luminance for an sRGB hex triple. */
const luminance = (hex: string): number => {
  const full = hex.length === 4
    ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
    : hex;
  const channels = [1, 3, 5].map((offset) => parseInt(full.slice(offset, offset + 2), 16) / 255);
  const linear = channels.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
};

// --color-bg for the dark theme, resolved in Chromium from its oklch()
// declaration. The assertion below fails if the token drifts away from the
// value this number was measured at.
const DARK_BG_TOKEN = "oklch(14.8% 0.006 255)";
const DARK_BG_LUMINANCE = luminance("#090b0d");
check(
  tokenIn(darkBlock, "--color-bg") === DARK_BG_TOKEN,
  `--color-bg changed; re-measure DARK_BG_LUMINANCE (found ${tokenIn(darkBlock, "--color-bg")})`,
);

/** Minimum contrast for a data trace against the page background. Below ~3:1 a
 *  1-2px line is not reliably distinguishable from the paper. */
const MIN_TRACE_CONTRAST = 3;
// Only inline trace styling. Palette lists offered to the user (CompositePlot's
// "Black"/"White" contour choices) are deliberate selections, not defaults, and
// are matched against the plot's own fill rather than the page.
const TRACE_COLOR = /\b(line|marker)\s*:\s*\{[^}]*\bcolor\s*:\s*"(#[0-9a-fA-F]{3,6})"/g;

let traceColorsChecked = 0;
for (const file of readdirSync(plotsDir).filter((name) => name.endsWith(".tsx"))) {
  const source = readFileSync(join(plotsDir, file), "utf-8");
  for (const match of source.matchAll(TRACE_COLOR)) {
    const hex = match[2];
    const ratio =
      (Math.max(luminance(hex), DARK_BG_LUMINANCE) + 0.05) /
      (Math.min(luminance(hex), DARK_BG_LUMINANCE) + 0.05);
    traceColorsChecked += 1;
    check(
      ratio >= MIN_TRACE_CONTRAST,
      `${file} draws a trace in ${hex}, only ${ratio.toFixed(2)}:1 on the dark theme — ` +
        `resolve it from usePlotChrome().ink instead`,
    );
  }
}
check(traceColorsChecked > 0, "the trace-color scan found something to check");

/* ---------- Scene overlays must use scene ink, not page ink ----------
 * The five vtk.js scenes clear to a fixed near-black in BOTH themes (the
 * `background: [0.055, 0.063, 0.075]` triples), so the --scene-* tokens are
 * declared once at :root and deliberately never flip. Any element painted onto
 * that canvas has to take its text color from that family. VolumeRender3D's
 * status overlay used --color-text-secondary/--color-text-muted/--color-warn/
 * --color-error, which resolve to near-black under the light theme: the
 * "waiting for frame" and error text disappeared into the scene. */

const SCENE_TOKENS = [
  "--scene-bg",
  "--scene-scrim",
  "--scene-scrim-strong",
  "--scene-ink",
  "--scene-ink-muted",
  "--scene-accent",
  "--scene-warn",
  "--scene-error",
  "--scene-hairline",
  "--scene-chip",
];

for (const token of SCENE_TOKENS) {
  const dark = tokenIn(darkBlock, token);
  const light = tokenIn(lightBlock, token);
  check(dark != null, `${token} is declared for the dark scene`);
  // The scene's clear colour now follows the user's appearance
  // (lib/plot/sceneBackground.ts), so its ink must follow too: these were pinned
  // at :root while the scene was hardcoded near-black, and leaving them pinned
  // would paint pale ink onto pale paper.
  check(light != null, `${token} is declared for the light scene`);
  // A token DERIVED from --scene-bg is textually identical in both blocks and
  // still resolves differently, so only the base tokens can be compared by text.
  const derived = (dark ?? "").includes("var(--scene-bg)");
  check(
    derived || dark !== light,
    `${token} actually differs between the two scenes`,
  );
}

/** Declaration bodies of every rule in a flat stylesheet, `@media` wrappers
 *  included by taking the text after the innermost brace. */
const ruleBodies = (css: string): string[] =>
  css
    .split("}")
    .map((fragment) => fragment.slice(fragment.lastIndexOf("{") + 1))
    .filter((body) => body.includes(":"));

let sceneRulesChecked = 0;
for (const file of readdirSync(plotsDir).filter((name) => name.endsWith(".css"))) {
  for (const body of ruleBodies(readFileSync(join(plotsDir, file), "utf-8"))) {
    if (!/background\s*:[^;]*--scene-scrim/.test(body)) continue;
    sceneRulesChecked += 1;
    const color = body.match(/(?:^|[;\s])color\s*:\s*([^;]+)/);
    check(
      color == null || color[1].includes("--scene-"),
      `${file} paints ${color?.[1].trim()} on a scene scrim — use a --scene-* ink token`,
    );
  }
}
check(sceneRulesChecked > 0, "the scene-overlay scan found something to check");

console.log(`studio appearance: ${checks} checks passed`);
