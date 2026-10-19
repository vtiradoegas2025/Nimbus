import assert from "node:assert/strict";
import {
  contrastAdjustedColor,
  DEFAULT_SCENE_APPEARANCE,
  normalizeHexColor,
  resolveSceneAppearance,
  sceneContrastRatio,
} from "../../../../ui/src/lib/plot/sceneAppearance.ts";
import {
  cloudSceneColors,
  precipitationSceneColor,
  vortexSceneColors,
} from "../../../../ui/src/lib/plot/supercellSceneAppearance.ts";

assert.equal(normalizeHexColor("#AbC"), "#aabbcc");
assert.equal(normalizeHexColor("bad", "#123456"), "#123456");

const themeDark = resolveSceneAppearance(DEFAULT_SCENE_APPEARANCE, "dark");
const themeLight = resolveSceneAppearance(DEFAULT_SCENE_APPEARANCE, "light");
assert.equal(themeDark.isLight, false);
assert.equal(themeLight.isLight, true);
assert.notDeepEqual(themeDark.ink, themeLight.ink);

const custom = resolveSceneAppearance({
  ...DEFAULT_SCENE_APPEARANCE,
  backgroundMode: "custom",
  customBackground: "#fafafa",
}, "dark");
assert.equal(custom.backgroundHex, "#fafafa");
assert.equal(custom.isLight, true);

const lowContrast: readonly [number, number, number] = [0.95, 0.95, 0.95];
const adjusted = contrastAdjustedColor(lowContrast, custom.background);
assert.ok(sceneContrastRatio(adjusted, custom.background) >= 2.2);

const lightCloud = cloudSceneColors("auto", "#ffffff", 1, "subtle", themeLight);
const darkCloud = cloudSceneColors("auto", "#ffffff", 1, "subtle", themeDark);
assert.ok(sceneContrastRatio(lightCloud.body, themeLight.background) >= 1.75);
assert.ok(sceneContrastRatio(darkCloud.body, themeDark.background) >= 1.75);
assert.notDeepEqual(lightCloud.body, darkCloud.body);
assert.ok(darkCloud.body.every((channel) => channel < 0.95));
const neutralVortex = vortexSceneColors("neutral", darkCloud, themeDark);
const matchedVortex = vortexSceneColors("cloud", darkCloud, themeDark);
const diagnosticVortex = vortexSceneColors("diagnostic", darkCloud, themeDark);
assert.ok(neutralVortex.core.every((channel) => Math.abs(channel - neutralVortex.core[0]) < 0.09));
assert.deepEqual(matchedVortex.core, darkCloud.body);
assert.notDeepEqual(diagnosticVortex.core, diagnosticVortex.opposite);
assert.ok(sceneContrastRatio(
  precipitationSceneColor("qh", null, themeLight),
  themeLight.background,
) >= 2);

console.log("scene appearance tests passed");
