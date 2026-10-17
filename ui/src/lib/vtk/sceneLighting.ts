import vtkLight from "@kitware/vtk.js/Rendering/Core/Light.js";
import { hexToSceneRgb, lightingSettings, shadowMaterial, type SceneLightingInput, type SceneLightingPreset } from "@/lib/plot/sceneAppearance";

type SceneLight = ReturnType<typeof vtkLight.newInstance>;

interface SceneRenderer {
  getLights: () => SceneLight[];
  addLight: (light: SceneLight) => void;
  removeLight: (light: SceneLight) => void;
  setAutomaticLightCreation: (enabled: boolean) => void;
  setLightFollowCamera: (enabled: boolean) => void;
  setTwoSidedLighting: (enabled: boolean) => void;
}

export interface VtkSceneLighting {
  key: SceneLight;
  fill: SceneLight;
  rim: SceneLight;
  apply: (preset: SceneLightingInput, lightBackground: boolean) => void;
  dispose: () => void;
}

const INTENSITIES: Record<SceneLightingPreset, readonly [number, number, number]> = {
  neutral: [1.0, 0.22, 0.35],
  soft: [0.72, 0.46, 0.18],
  contrast: [1.18, 0.08, 0.52],
  // Flat materials disable lighting. Zeroing the lights as well prevents them
  // affecting any actor a third-party VTK source adds with its default material.
  flat: [0, 0, 0],
};

/** Install one deterministic world-space rig. vtk.js's automatic camera light
 * otherwise makes identical geometry change character as the camera orbits. */
export function createVtkSceneLighting(renderer: SceneRenderer): VtkSceneLighting {
  renderer.setAutomaticLightCreation(false);
  // WebGL can create its headlight during initial sizing, before the scene rig
  // is installed. Disabling automatic creation does not remove that light.
  for (const light of renderer.getLights()) {
    if (light.lightTypeIsHeadLight()) renderer.removeLight(light);
  }
  renderer.setLightFollowCamera(false);
  renderer.setTwoSidedLighting(true);

  const key = vtkLight.newInstance();
  key.setLightTypeToSceneLight();
  key.setDirectionAngle(35, 50);
  key.setColor(1.0, 0.97, 0.92);
  renderer.addLight(key);

  const fill = vtkLight.newInstance();
  fill.setLightTypeToSceneLight();
  fill.setDirectionAngle(18, -130);
  fill.setColor(0.72, 0.82, 1.0);
  renderer.addLight(fill);

  const rim = vtkLight.newInstance();
  rim.setLightTypeToSceneLight();
  rim.setDirectionAngle(30, 205);
  rim.setColor(1, 1, 1);
  renderer.addLight(rim);

  let disposed = false;
  return {
    key,
    fill,
    rim,
    apply(input, lightBackground) {
      if (disposed) return;
      const settings = lightingSettings(input);
      const preset = settings.preset;
      fill.setColor(...(settings.fillTint ? hexToSceneRgb(settings.fillTint) : [0.72, 0.82, 1] as [number, number, number]));
      const [keyIntensity, fillIntensity, rimIntensity] = INTENSITIES[preset];
      // Pale scenes need slightly less fill or translucent bodies lose their
      // internal gradients. Hue and data transfer functions remain untouched.
      const fillScale = lightBackground ? 0.78 : 1;
      key.setIntensity(keyIntensity);
      fill.setIntensity(fillIntensity * fillScale);
      rim.setIntensity(rimIntensity);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      renderer.removeLight(key);
      renderer.removeLight(fill);
      renderer.removeLight(rim);
      key.delete();
      fill.delete();
      rim.delete();
    },
  };
}

interface SurfaceProperty {
  setLighting: (enabled: boolean) => void;
  setAmbient: (value: number) => void;
  setDiffuse: (value: number) => void;
  setSpecular: (value: number) => void;
  setSpecularPower?: (value: number) => void;
}

export function applySurfaceLighting(
  property: SurfaceProperty,
  input: SceneLightingInput,
  materialKind: "opaque" | "translucent" = "opaque",
): void {
  const preset = lightingSettings(input).preset;
  const flat = preset === "flat";
  property.setLighting(!flat);
  if (flat) return;
  const material = preset === "soft"
    ? { ambient: 0.42, diffuse: 0.62 }
    : preset === "contrast"
      ? { ambient: 0.14, diffuse: 0.96 }
      : { ambient: 0.28, diffuse: 0.78 };
  // Alpha blending attenuates a translucent surface against the scene once
  // more after lighting. A modest ambient lift keeps its data colour readable
  // without flattening the normals that carry the physical shape.
  const ambient = materialKind === "translucent"
    ? Math.min(0.7, material.ambient + 0.18)
    : material.ambient;
  const diffuse = materialKind === "translucent"
    ? material.diffuse * 0.88
    : material.diffuse;
  const adjusted = shadowMaterial(ambient, diffuse, input);
  property.setAmbient(adjusted.ambient);
  property.setDiffuse(adjusted.diffuse);
  property.setSpecular(0);
  property.setSpecularPower?.(1);
}

interface VolumeProperty {
  setShade: (enabled: boolean) => void;
  setAmbient: (value: number) => void;
  setDiffuse: (value: number) => void;
  setSpecular: (value: number) => void;
}

export function applyVolumeLighting(
  property: VolumeProperty,
  input: SceneLightingInput,
  kind: "scalar" | "structure" | "cloud" = "scalar",
): void {
  const preset = lightingSettings(input).preset;
  const flat = preset === "flat";
  property.setShade(!flat);
  property.setSpecular(0);
  if (flat) {
    property.setAmbient(1);
    property.setDiffuse(0);
    return;
  }
  if (kind === "cloud") {
    const material = preset === "soft"
      ? { ambient: 0.6, diffuse: 0.44 }
      : preset === "contrast"
        ? { ambient: 0.3, diffuse: 0.72 }
        : { ambient: 0.48, diffuse: 0.58 };
    const adjusted = shadowMaterial(material.ambient, material.diffuse, input);
    property.setAmbient(adjusted.ambient);
    property.setDiffuse(adjusted.diffuse);
    return;
  }
  if (kind === "structure") {
    const material = preset === "soft"
      ? { ambient: 0.7, diffuse: 0.3 }
      : preset === "contrast"
        ? { ambient: 0.4, diffuse: 0.62 }
        : { ambient: 0.6, diffuse: 0.4 };
    const adjusted = shadowMaterial(material.ambient, material.diffuse, input);
    property.setAmbient(adjusted.ambient);
    property.setDiffuse(adjusted.diffuse);
    return;
  }
  const material = preset === "soft"
    ? { ambient: 0.55, diffuse: 0.5 }
    : preset === "contrast"
      ? { ambient: 0.2, diffuse: 0.9 }
      : { ambient: 0.4, diffuse: 0.6 };
  const adjusted = shadowMaterial(material.ambient, material.diffuse, input);
  property.setAmbient(adjusted.ambient);
  property.setDiffuse(adjusted.diffuse);
}
