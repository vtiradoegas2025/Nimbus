import vtkMacro from "@kitware/vtk.js/macros";
import vtkWebGpuPolyDataMapper from "@kitware/vtk.js/Rendering/WebGPU/PolyDataMapper";
import vtkWebGpuCellArrayMapper from "@kitware/vtk.js/Rendering/WebGPU/CellArrayMapper";
import { BufferUsage } from "@kitware/vtk.js/Rendering/WebGPU/BufferManager/Constants";
import { registerOverride } from "@kitware/vtk.js/Rendering/WebGPU/ViewNodeFactory";

interface ShaderDescription {
  getCode(): string;
  setCode(code: string): void;
}

interface ShaderPipeline {
  getShaderDescription(stage: "vertex" | "fragment"): ShaderDescription;
}

type ShaderReplacement = (hash: string, pipeline: ShaderPipeline, vertexInput: unknown) => void;

interface CellMapper {
  getShaderReplacements(): Map<string, ShaderReplacement>;
  getTopologyFromUsage(usage: number): string;
  updateUBO(): void;
}

interface PolyDataMapper {
  createCellArrayMapper(): CellMapper;
}

const SURFACE_LIGHTING = `
  var diffuseLight = vec3<f32>(0.0);
  for (var i = 0; i < rendererUBO.LightCount; i++) {
    let light = rendererLightSSBO.values[i];
    let direction = normalize((rendererUBO.WCVCNormals * vec4<f32>(light.LightDir.xyz, 0.0)).xyz);
    let radiance = light.LightColor.rgb * light.LightColor.w * 0.2;
    diffuseLight += max(dot(normal, direction), 0.0) * radiance;
  }
  computedColor = vec4<f32>(
    ambientIntensity * ambientColor.rgb + diffuseIntensity * diffuseColor.rgb * _diffuseMap.rgb * diffuseLight,
    opacity);
`;

function extendCell(api: object, model: object, initialValues: object = {}): void {
  vtkWebGpuCellArrayMapper.extend(api, model, initialValues);
  const cell = api as CellMapper;
  const topology = cell.getTopologyFromUsage;
  // VTK expands strips to triangle indices, but its WebGPU mapper selects
  // line-list for Strips. The primitive type must match the expanded indices.
  cell.getTopologyFromUsage = (usage) => usage === BufferUsage.Strips ? "triangle-list" : topology(usage);
  const state = model as {
    WebGPUActor: { getRenderable(): { getProperty(): { getLighting(): boolean } } };
    WebGPURenderWindow: { getDevice(): unknown };
    UBO: { setValue(name: string, value: number): void; sendIfNeeded(device: unknown): void };
  };
  const update = cell.updateUBO;
  cell.updateUBO = () => {
    update();
    if (!state.WebGPUActor.getRenderable().getProperty().getLighting()) {
      for (const suffix of ["", "BF"]) {
        state.UBO.setValue(`AmbientIntensity${suffix}`, 1);
        state.UBO.setValue(`DiffuseIntensity${suffix}`, 0);
      }
      state.UBO.sendIfNeeded(state.WebGPURenderWindow.getDevice());
    }
  };
}

const newCell = vtkMacro.newInstance(extendCell, "vtkNimbusWebGPUCellArrayMapper");

function extend(api: object, model: object, initialValues: object = {}): void {
  const publicApi = api as PolyDataMapper;
  vtkWebGpuPolyDataMapper.extend(publicApi, model, initialValues);
  publicApi.createCellArrayMapper = () => {
    const cell = newCell() as unknown as CellMapper;
    const replacements = cell.getShaderReplacements();
    const original = replacements.get("replaceShaderLight")!;
    replacements.set("replaceShaderLight", (hash, pipeline, vertexInput) => {
      const fragment = pipeline.getShaderDescription("fragment");
      const code = fragment.getCode();
      if (hash.includes("sel") || !code.includes("var normal:")) {
        original(hash, pipeline, vertexInput);
        return;
      }
      // vtk.js 36's WebGPU PBR material drops Phong ambient and scales light
      // intensity by five. Nimbus uses the same directional Lambert rig and
      // ambient term on both APIs so switching backend preserves scalar color.
      fragment.setCode(code.replace("//VTK::Light::Impl", SURFACE_LIGHTING));
    });
    const texture = replacements.get("replaceShaderTCoord")!;
    replacements.set("replaceShaderTCoord", (hash, pipeline, vertexInput) => {
      texture(hash, pipeline, vertexInput);
      const fragment = pipeline.getShaderDescription("fragment");
      const code = fragment.getCode();
      if (!code.includes("var texColor = textureSample(DiffuseTexture")) return;
      // Scalar lookup textures already supply diffuse/ambient color and alpha
      // in Color::Impl. Multiplying by that texture again squares the colors.
      fragment.setCode(code
        .replace("diffuseColor.rgb * _diffuseMap.rgb * diffuseLight", "diffuseColor.rgb * diffuseLight")
        .replace("diffuse * _diffuseMap.rgb, opacity", "diffuse, opacity")
        .replace("computedColor.a = opacity * _diffuseMap.a;", "computedColor.a = opacity;"));
    });
    return cell;
  };
}

let installed = false;

export function installVtkWebGpuSurfaceLighting(): void {
  if (installed) return;
  registerOverride("vtkMapper", vtkMacro.newInstance(extend, "vtkNimbusWebGPUPolyDataMapper"));
  installed = true;
}
