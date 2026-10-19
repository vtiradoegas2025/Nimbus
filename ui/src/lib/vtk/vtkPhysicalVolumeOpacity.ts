import { volumePlanningSampleDistance, flatVolumePreview } from "./volumeShaderBudget";
import { acquireVolumeValidity, getVolumeValidity } from "./volumeValidity";
import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkOpenGLVolumeMapper from "@kitware/vtk.js/Rendering/OpenGL/VolumeMapper.js";
import { Filter } from "@kitware/vtk.js/Rendering/OpenGL/Texture/Constants.js";
import { createFloat32VolumeTexture, isFloat32VolumeTexture, volumeScalarTextureLayout, type VolumeScalarTexture } from "./volumeScalarSampling";
import { planVolumeTextureLayout } from "./volumeTexturePacking";
import { VolumeSamplingShaderMemo } from "./volumeSamplingShader";
import { volumeTextureCapabilities } from "./volumeTextureCapabilities";
import { jointVolumeTopologyKey, specializeJointVolumeTopology } from "./jointVolumeTopology";
import { registerOverride } from "@kitware/vtk.js/Rendering/OpenGL/ViewNodeFactory.js";
import { acquireVolumeFramebuffer } from "./sharedVolumeFramebuffer";
import { hasJointVolumeBinding, resolveJointVolume, setJointVolumeCleanup } from "./jointVolumeRegistry";
import { jointVolumeViewGeometry, type jointVolumeGeometry } from "./jointVolumeGeometry";
import { acquireVolumeBackground, volumeBackgroundBinding, VOLUME_BACKGROUND_GLSL } from "./volumeBackgroundTexture";

interface JointUniforms {
  geometry: ReturnType<typeof jointVolumeGeometry>;
  references: number[]; roles: number[]; reach: number[]; steps: number[]; sampleSteps: number[];
  surface: number[][]; lighting: number[][];
  sampleBudget: number;
  shaderSampleBudget?: number;
  temporalMasked: number[];
  temporalPeers: number[];
  temporalFraction: number;
}

interface Volume {
  getProperty(index?: number): { getScalarOpacityUnitDistance(component: number): number; getInterpolationType(): number };
}

interface ShaderCell {
  getProgram(): { isUniformUsed?: (name: string) => boolean; setUniformf(name: string, value: number): void; setUniform3fv(name: string, value: number[]): void;
    setUniform4fv(name: string, value: number[]): void; setUniformMatrix3x3(name: string, value: number[]): void } | null;
}

interface VolumeMapper {
  setCameraShaderParameters(cell: ShaderCell, renderer: unknown, volume: Volume): void;
  getNeedToRebuildShaders(cell: ShaderCell, renderer: unknown, volume: Volume): boolean;
  getParent(): object;
  setParent(parent: object): void;
  volumePass(prepass: boolean, renderPass: unknown): void;
  buildPass(): void;
  zBufferPass(prepass: boolean, pass: { getZBufferTexture(): unknown }): void;
  renderPiece(renderer: unknown, volume: Volume): void;
  replaceShaderValues(shaders: { Fragment: string }, renderer: unknown, volume: Volume): void;
  renderPieceStart(renderer: unknown, volume: Volume): void;
  renderPieceDraw(renderer: unknown, volume: Volume): void;
  delete(): void;
  buildBufferObjects(renderer: unknown, volume: Volume): void;
  getCurrentSampleDistance(renderer: unknown): number;
  setMapperShaderParameters(cell: ShaderCell, renderer: unknown, volume: Volume): void;
}

const jointNodes = new WeakMap<object, { node: VolumeMapper; mapper: object; layers: number }>();

function extend(api: object, state: object, initialValues: object = {}): void {
  vtkOpenGLVolumeMapper.extend(api, state, initialValues);
  const mapper = api as VolumeMapper;
  const model = state as {
    zBufferTexture: unknown;
    openGLCamera: { getKeyMatrices(renderer: unknown): { wcvc: ArrayLike<number> } };
    _lastScale: number;
    tris: { releaseGraphicsResources(window: object): void; delete(): void };
    jitterTexture: { releaseGraphicsResources(window: object): void; delete(): void };
    copyVAO?: { releaseGraphicsResources(): void; delete(): void } | null;
    copyShader: unknown;
    _openGLRenderWindow: {
      getShaderCache(): { readyShaderProgramArray(vertex: string, fragment: string, geometry: string): unknown };
      getContext(): WebGL2RenderingContext;
      getGraphicsResourceForObject(object: object): { oglObject: VolumeScalarTexture | null; hash: string | null } | undefined;
      setGraphicsResourceForObject(object: object, texture: VolumeScalarTexture, hash: string): void;
    };
    framebuffer: ReturnType<typeof acquireVolumeFramebuffer>["framebuffer"];
    previousState: { maximumNumberOfSamples: number };
    numberOfComponents: number;
    currentValidInputs: Array<{ inputIndex: number; imageData: { getDimensions(): number[]; getBounds(): number[];
      getPointData(): { getScalars(): Parameters<typeof createFloat32VolumeTexture>[2] } } }>;
    scalarTextures: Array<ReturnType<typeof createFloat32VolumeTexture>>;
    renderable: { getImageSampleDistance(): number; getViewSpecificProperties(): { OpenGL?: { nimbusPhysicalOpacity?: boolean; nimbusOpenUpperFaces?: number[];
      nimbusPhysicalReference?: number; nimbusInverseSceneScale?: number[] } } };
  };
  const enabled = () => model.renderable.getViewSpecificProperties().OpenGL?.nimbusPhysicalOpacity === true;
  const samplingMemo = new VolumeSamplingShaderMemo();
  const samplingShader = (volume: Volume) => {
    const capabilities = volumeTextureCapabilities(model._openGLRenderWindow.getContext());
    return samplingMemo.get(model.currentValidInputs.map(({ imageData }) =>
      planVolumeTextureLayout(imageData.getDimensions(), capabilities.maximumDimension)), model.currentValidInputs.map(({ inputIndex }) =>
      !capabilities.floatLinear && volume.getProperty(inputIndex).getInterpolationType() !== 0), {
      validity: model.currentValidInputs.map(({ imageData }) => getVolumeValidity(imageData) != null),
      nearest: model.currentValidInputs.map(({ inputIndex }) => volume.getProperty(inputIndex).getInterpolationType() === 0),
    });
  };
  const render = mapper.renderPiece;
  const delegation = initialValues as { nimbusDelegate?: boolean; nimbusActor?: Volume };
  mapper.renderPiece = (renderer, volume) => {
    if (delegation.nimbusDelegate) { render(renderer, delegation.nimbusActor!); return; }
    const original = model.renderable;
    const joint = resolveJointVolume(original);
    if (!joint) { if (!hasJointVolumeBinding(original)) render(renderer, volume); return; }
    if (joint.lead !== original) return;
    let entry = jointNodes.get(joint.owner);
    let previous: VolumeMapper | null = null;
    // vtk.js retains scalar resource slots when input count shrinks. Replace
    // that view node, registering the new users before retiring the old ones.
    if (!entry || entry.mapper !== joint.mapper || entry.layers !== joint.layers) {
      previous = entry?.node ?? null;
      entry = { node: createPhysicalMapper({ renderable: joint.mapper, nimbusDelegate: true, nimbusActor: joint.volume }) as unknown as VolumeMapper,
        mapper: joint.mapper, layers: joint.layers };
      jointNodes.set(joint.owner, entry);
      setJointVolumeCleanup(joint.owner, () => { jointNodes.get(joint.owner)?.node.delete(); jointNodes.delete(joint.owner); });
    }
    entry.node.setParent(mapper.getParent());
    entry.node.buildPass();
    entry.node.zBufferPass(true, { getZBufferTexture: () => model.zBufferTexture });
    try { entry.node.volumePass(true, {}); }
    finally { previous?.delete(); }
  };
  const build = mapper.buildBufferObjects;
  const distance = mapper.getCurrentSampleDistance;
  const parameters = mapper.setMapperShaderParameters;
  const cameraParameters = mapper.setCameraShaderParameters;
  mapper.setCameraShaderParameters = (cell, renderer, volume) => {
    const joint = (model.renderable.getViewSpecificProperties().OpenGL as { nimbusJoint?: JointUniforms } | undefined)?.nimbusJoint;
    const program = cell.getProgram();
    if (!joint || !program) { cameraParameters(cell, renderer, volume); return; }
    const uniforms = { ...program, setUniformf: (name: string, value: number) => {
      const bounded = name === "dcxmin" || name === "dcxmax" || name === "dcymin" || name === "dcymax";
      program.setUniformf(name, bounded ? Math.max(-1, Math.min(1, value)) : value);
    } };
    cameraParameters({ ...cell, getProgram: () => uniforms }, renderer, volume);
  };
  let scalarShaderSamples: number | null = null;
  const replaceShaders = mapper.replaceShaderValues;
  mapper.replaceShaderValues = (shaders, renderer, volume) => {
    if (enabled()) {
      shaders.Fragment = shaders.Fragment.replace("vec4 getColorAtPos(vec3 posVC) {", `${VOLUME_BACKGROUND_GLSL}\nvec4 getColorAtPos(vec3 posVC) {`);
      shaders.Fragment = shaders.Fragment.replace("return getColorForValue(texValue, posVC, posIS);", `float weight = nimbusBackgroundWeight(texValue.r, posIS);
        if (weight <= 0.0) return vec4(0.0);
        vec4 color = getColorForValue(texValue, posVC, posIS);
        if (weight < 1.0) color.a = 1.0 - pow(max(0.0, 1.0 - color.a), weight);
        return color;`);
      // A conservative shared loop ceiling avoids compiling a distinct program
      // for every brick dimension. Ray termination still uses its exact bounds.
      shaders.Fragment = shaders.Fragment.replace("//VTK::MaximumNumberOfSamples",
        String(volumeShaderSampleCeiling(scalarShaderSamples ?? model.previousState.maximumNumberOfSamples)));
    }
    replaceShaders(shaders, renderer, volume);
    if (enabled()) {
      const joint = (model.renderable.getViewSpecificProperties().OpenGL as { nimbusJoint?: JointUniforms } | undefined)?.nimbusJoint;
      if (joint) shaders.Fragment = specializeJointVolumeTopology(shaders.Fragment, joint);
      const sampling = samplingShader(volume);
      shaders.Fragment = shaders.Fragment.replace("vec4 rawSampleTexture(vec3 pos) {", `${sampling.scalar}\nvec4 rawSampleTexture(vec3 pos) {`);
      for (let component = 0; component < 4; component++) shaders.Fragment = shaders.Fragment.replaceAll(
        `texture(volumeTexture[${component}], pos)`, `nimbusScalarTexture(volumeTexture[${component}], pos, ${component})`);
      shaders.Fragment = shaders.Fragment.replace("vec4 rawSampleTexture(vec3 pos) {", `${sampling.validity}\nvec4 rawSampleTexture(vec3 pos) {`);
      // Joint sampling replaces rawSampleTexture before this hook and keeps its
      // per-layer helpers ahead of it. Forward declarations serve both paths.
      shaders.Fragment = shaders.Fragment.replace("//VTK::System::Dec", "//VTK::System::Dec\nvec4 nimbusScalarTexture(highp sampler3D image, vec3 coordinate, int component);\nbool nimbusValid0(vec3 pos);\nbool nimbusValid1(vec3 pos);\nbool nimbusValid2(vec3 pos);\nbool nimbusValid3(vec3 pos);");
      shaders.Fragment = shaders.Fragment.replace("mat4 computeMat4Normal(vec3 posIS, vec4 tValue) {", "mat4 computeMat4Normal(vec3 posIS, vec4 tValue) {\n if (!nimbusValidNormal(posIS)) return mat4(0.0);");
      shaders.Fragment = shaders.Fragment.replace(/(vec4 computeNormalForDensity\(vec3 posIS, out vec3 scalarInterp\[2\],[\s\S]*?\{)/, "$1\n if (!nimbusValidNormal(posIS)) { scalarInterp[0] = vec3(0.0); scalarInterp[1] = vec3(0.0); return vec4(0.0); }");
      shaders.Fragment = shaders.Fragment.replace("//VTK::System::Dec", "//VTK::System::Dec\nuniform float nimbusFlatPreview;");
      shaders.Fragment = shaders.Fragment.replace("vec4 getColorForValue(vec4 tValue, vec3 posVC, vec3 posIS) {", `vec4 getColorForValue(vec4 tValue, vec3 posVC, vec3 posIS) {
        if (!nimbusValid0(posIS)) return vec4(0.0);
        #if vtkNumberOfComponents == 1 && !defined(vtkComponent0Proportional)
        if (nimbusFlatPreview > 0.5) {
          float height = volume.transferFunctionsSampleHeight[0];
          float alpha = getOpacityFromTexture(tValue.r, 0, height);
          #ifdef EnabledGradientOpacity
          alpha *= computeGradientOpacityFactor(computeMat4Normal(posIS, tValue)[0].a, 0);
          #endif
          return vec4(getColorFromTexture(tValue.r, 0, height), clamp(alpha, 0.0, 1.0));
        }
        #endif
      `);
    }
  };
  const start = mapper.renderPieceStart;
  const draw = mapper.renderPieceDraw;
  type Validity = NonNullable<ReturnType<typeof acquireVolumeValidity>>;
  let validity: Array<{ image: object; lease: Validity | null }> = [];
  let validityWindow: object | null = null;
  let background: ReturnType<typeof acquireVolumeBackground> | null = null;
  mapper.renderPieceDraw = (renderer, volume) => {
    for (const item of validity) item.lease?.texture.activate();
    background?.texture.activate();
    try { draw(renderer, volume); }
    finally { for (const item of validity) item.lease?.texture.deactivate(); background?.texture.deactivate(); }
  };
  const dispose = mapper.delete;
  let target: ReturnType<typeof acquireVolumeFramebuffer> | null = null;
  let targetWindow: object | null = null;
  let deleted = false;
  mapper.renderPieceStart = (renderer, volume) => {
    if (delegation.nimbusDelegate) model._lastScale = Math.max(1, model.renderable.getImageSampleDistance() ** 2);
    if (targetWindow !== model._openGLRenderWindow) {
      if (target) target.release();
      else model.framebuffer.delete();
      targetWindow = model._openGLRenderWindow;
      target = acquireVolumeFramebuffer(targetWindow);
      model.framebuffer = target.framebuffer;
    }
    // VTK draws and copies each volume synchronously, so all actors in this
    // render window can reuse one preview target without retaining its pixels.
    start(renderer, volume);
    if (enabled()) {
      {
        const manual = !volumeTextureCapabilities(model._openGLRenderWindow.getContext()).floatLinear;
        for (const texture of model.scalarTextures) if (manual || volumeScalarTextureLayout(texture)?.tiled) {
          texture.setMinificationFilter(Filter.NEAREST);
          texture.setMagnificationFilter(Filter.NEAREST);
          texture.activate();
        }
      }
      const data = volumeBackgroundBinding(model.renderable)?.data;
      if (!background || validityWindow !== model._openGLRenderWindow
          || (data ? data !== background.data : background.data.length !== 1)) {
        background?.release(); background = acquireVolumeBackground(model._openGLRenderWindow, model.renderable);
      }
      if (!model.copyShader) warmVolumePreviewCopyShader(model._openGLRenderWindow);
      const next = model.currentValidInputs.map(({ imageData }, index) => {
        const previous = validity[index];
        if (validityWindow === model._openGLRenderWindow && previous?.image === imageData
            && previous.lease?.mask === getVolumeValidity(imageData)
            && (!previous.lease || previous.lease.dimensions.every((value, axis) => value === imageData.getDimensions()[axis]))) return previous;
        const lease = acquireVolumeValidity(model._openGLRenderWindow, imageData, imageData.getDimensions());
        previous?.lease?.release();
        return { image: imageData, lease };
      });
      for (let index = next.length; index < validity.length; index++) validity[index].lease?.release();
      validity = next;
      validityWindow = model._openGLRenderWindow;
    }
  };
  mapper.delete = () => {
    if (deleted) return;
    deleted = true;
    background?.release(); background = null;
    for (const item of validity) item.lease?.release();
    validity = [];
    target?.release();
    target = null;
    if (model._openGLRenderWindow) {
      model.tris.releaseGraphicsResources(model._openGLRenderWindow);
      model.jitterTexture.releaseGraphicsResources(model._openGLRenderWindow);
      model.copyVAO?.releaseGraphicsResources();
    }
    model.tris.delete(); model.jitterTexture.delete(); model.copyVAO?.delete();
    dispose();
  };
  let tableDistance: number | null = null;
  let planningDistance: number | null = null;
  const needsShaders = mapper.getNeedToRebuildShaders;
  let compiledSamplingKey: string | null = null;
  mapper.getNeedToRebuildShaders = (cell, renderer, volume) => {
    const joint = (model.renderable.getViewSpecificProperties().OpenGL as { nimbusJoint?: JointUniforms } | undefined)?.nimbusJoint;
    scalarShaderSamples = null;
    if (!joint && enabled()) {
      const bounds = (model.currentValidInputs[0] as { imageData?: { getBounds: () => number[] } })?.imageData?.getBounds();
      if (bounds) scalarShaderSamples = Math.max(1, Math.ceil(Math.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        / volumePlanningSampleDistance(model.renderable, distance(renderer))));
    }
    if (joint) {
      const b = joint.geometry.bounds;
      planningDistance = Math.hypot(b[1] - b[0], b[3] - b[2], b[5] - b[4]) / (Math.max(joint.shaderSampleBudget ?? 0, joint.sampleBudget) + 0.25);
    }
    try {
      const currentKey = enabled() ? JSON.stringify([samplingShader(volume).key, joint ? jointVolumeTopologyKey(joint) : null]) : null;
      const changed = currentKey !== compiledSamplingKey;
      compiledSamplingKey = currentKey;
      return needsShaders(cell, renderer, volume) || changed;
    }
    finally { planningDistance = null; }
  };
  mapper.getCurrentSampleDistance = (renderer) => tableDistance ?? planningDistance ?? distance(renderer);
  mapper.buildBufferObjects = (renderer, volume) => {
    if (enabled()) {
      const joint = (model.renderable.getViewSpecificProperties().OpenGL as { nimbusJoint?: JointUniforms } | undefined)?.nimbusJoint;
      if (!joint && (model.numberOfComponents !== 1 || model.currentValidInputs.length !== 1)) {
        throw new Error("Physical volume opacity requires one scalar input per mapper");
      }
      tableDistance = volume.getProperty().getScalarOpacityUnitDistance(0);
      if (!(tableDistance > 0) || !Number.isFinite(tableDistance)) {
        tableDistance = null;
        throw new Error("Volume opacity reference distance must be finite and positive");
      }
      for (const { imageData } of model.currentValidInputs) {
        const scalars = imageData.getPointData().getScalars();
        const resource = model._openGLRenderWindow.getGraphicsResourceForObject(scalars);
        const hash = `${scalars.getMTime()}`;
        const layout = volumeScalarTextureLayout(resource?.oglObject);
        if (!resource?.oglObject?.getHandle() || resource.hash !== hash || !isFloat32VolumeTexture(resource.oglObject)
            || !layout?.sourceShape.every((count, axis) => count === imageData.getDimensions()[axis])) {
          const texture = createFloat32VolumeTexture(model._openGLRenderWindow, imageData.getDimensions(), scalars);
          model._openGLRenderWindow.setGraphicsResourceForObject(scalars, texture, hash);
        }
      }
    }
    try {
      build(renderer, volume);
    } finally {
      tableDistance = null;
    }
  };
  mapper.setMapperShaderParameters = (cell, renderer, volume) => {
    parameters(cell, renderer, volume);
    if (!enabled()) return;
    const program = cell.getProgram();
    if (program?.isUniformUsed?.("nimbusNearestScalarFilter")) program.setUniform4fv("nimbusNearestScalarFilter", [0, 1, 2, 3].map(component =>
      component < model.currentValidInputs.length
        && volume.getProperty(model.currentValidInputs[component].inputIndex).getInterpolationType() === 0 ? 1 : 0));
    if (program?.isUniformUsed?.("nimbusBackgroundSettings")) {
      const focus = volumeBackgroundBinding(model.renderable);
      program.setUniform4fv("nimbusBackgroundSettings", focus ? [focus.filterValues ? 1 : 2, focus.profile.threshold, focus.profile.threshold * 2, Number(focus.profile.positiveOnly)] : [0, 1, 2, 0]);
      program.setUniform4fv("nimbusBackgroundInterior", focus?.interior ?? [-1e20, 1e20, -1e20, 1e20]);
      program.setUniform4fv("nimbusBackgroundFeather", [...(focus?.feather ?? [1, 1]), 0, 0]);
      const info = model.scalarTextures[0].getVolumeInfo();
      program.setUniform4fv("nimbusBackgroundScalar", [info.scale[0], info.offset[0], 0, 0]);
      (program as typeof program & { setUniformi(name: string, value: number): void }).setUniformi("nimbusBackgroundTexture", background!.texture.getTextureUnit());
    }
    if (program?.isUniformUsed?.("nimbusValidityEnabled")) {
      program.setUniform4fv("nimbusValidityEnabled", [0, 1, 2, 3].map((index) => validity[index]?.lease ? 1 : 0));
      const uniforms = program as typeof program & { setUniformi(name: string, value: number): void };
      for (let index = 0; index < 4; index++) {
        const name = `nimbusValidityTexture[${index}]`;
        if (program.isUniformUsed?.(name)) uniforms.setUniformi(name,
          validity[index]?.lease?.texture.getTextureUnit() ?? model.scalarTextures[0].getTextureUnit());
      }
    }
    if (program?.isUniformUsed?.("nimbusFlatPreview")) program.setUniformf("nimbusFlatPreview", flatVolumePreview(model.renderable) ? 1 : 0);
    const reference = volume.getProperty().getScalarOpacityUnitDistance(0);
    const sample = distance(renderer);
    if (!(reference > 0) || !Number.isFinite(reference) || !(sample > 0) || !Number.isFinite(sample)) {
      throw new Error("Volume optical distances must be finite and positive");
    }
    cell.getProgram()?.setUniformf("nimbusOpacityRatio", sample / reference);
    const properties = model.renderable.getViewSpecificProperties().OpenGL;
    cell.getProgram()?.setUniformf("nimbusPhysicalReference", properties?.nimbusPhysicalReference ?? 0);
    cell.getProgram()?.setUniform3fv("nimbusInverseSceneScale", properties?.nimbusInverseSceneScale ?? [1, 1, 1]);
    const faces = properties?.nimbusOpenUpperFaces;
    if (faces) cell.getProgram()?.setUniform3fv("nimbusOpenUpperFaces", faces);
    const joint = (properties as { nimbusJoint?: JointUniforms } | undefined)?.nimbusJoint;
    if (joint) {
      const program = cell.getProgram();
      const geometry = jointVolumeViewGeometry(joint.geometry, model.openGLCamera.getKeyMatrices(renderer).wcvc);
      program?.setUniform3fv("nimbusLayerOriginVC", geometry.flatMap((layer) => layer.origin));
      program?.setUniform3fv("nimbusFixedOriginVC", geometry.flatMap(layer => layer.fixedOrigin));
      program?.setUniform3fv("nimbusLayerSpacing", joint.geometry.layers.flatMap((layer) => layer.spacing));
      geometry.forEach((layer, index) => {
        program?.setUniformMatrix3x3(`nimbusLayerToIS[${index}]`, layer.toIndex);
        program?.setUniformMatrix3x3(`nimbusLayerStepVC[${index}]`, layer.step);
        program?.setUniformMatrix3x3(`nimbusLayerNormalVC[${index}]`, layer.normal);
        program?.setUniformMatrix3x3(`nimbusFixedToIS[${index}]`, layer.fixedToIndex);
      });
      program?.setUniform4fv("nimbusReferences", joint.references);
      program?.setUniformf("nimbusTemporalFraction", joint.temporalFraction);
      program?.setUniform4fv("nimbusScalarScales", [0, 1, 2, 3].map(index => model.scalarTextures[index]?.getVolumeInfo().scale[0] ?? 1));
      program?.setUniform4fv("nimbusScalarOffsets", [0, 1, 2, 3].map(index => model.scalarTextures[index]?.getVolumeInfo().offset[0] ?? 0));
      program?.setUniform4fv("nimbusReach", joint.reach);
      program?.setUniform4fv("nimbusShadowSteps", joint.steps);
      program?.setUniform4fv("nimbusSampleSteps", joint.sampleSteps);
      program?.setUniform4fv("nimbusSurface", joint.surface.slice(0, model.numberOfComponents).flat());
      program?.setUniform4fv("nimbusLighting", joint.lighting.slice(0, model.numberOfComponents).flat());
    }
  };
}

const previewCopyPrograms = new WeakMap<object, unknown>();
function warmVolumePreviewCopyShader(window: { getShaderCache(): { readyShaderProgramArray(vertex: string, fragment: string, geometry: string): unknown } }): unknown {
  let program = previewCopyPrograms.get(window);
  if (!program) {
    // Match vtk.js's preview blit program and compile during the first data
    // render. Otherwise a newly split primary pane pays for it on first drag.
    program = window.getShaderCache().readyShaderProgramArray([
      "//VTK::System::Dec",
      "attribute vec4 vertexDC;",
      "uniform vec2 tfactor;",
      "varying vec2 tcoord;",
      "void main() { tcoord = vec2(vertexDC.x*0.5 + 0.5, vertexDC.y*0.5 + 0.5) * tfactor; gl_Position = vertexDC; }",
    ].join("\n"), [
      "//VTK::System::Dec",
      "//VTK::Output::Dec",
      "uniform sampler2D texture1;",
      "varying vec2 tcoord;",
      "void main() { gl_FragData[0] = texture2D(texture1,tcoord); }",
    ].join("\n"), "");
    previewCopyPrograms.set(window, program);
  }
  return program;
}

let installed = false;
const createPhysicalMapper = vtkMacro.newInstance(extend, "vtkNimbusPhysicalVolumeMapper");

export function volumeShaderSampleCeiling(samples: number): number {
  if (!Number.isSafeInteger(samples) || samples < 1) throw new Error("Volume shader sample count must be positive");
  return 2 ** Math.ceil(Math.log2(Math.max(128, samples)));
}

export function installPhysicalVolumeOpacity(): void {
  if (installed) return;
  // vtk.js caches opacity textures by transfer-function identity, not by ray
  // step or unit distance. Cache the reference curve; correct distance in GLSL.
  registerOverride("vtkVolumeMapper", createPhysicalMapper);
  installed = true;
}
