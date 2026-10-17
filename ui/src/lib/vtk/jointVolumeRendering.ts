import { affineWindTransform, identityAffine, type AffineWind } from "@/lib/plot/affineWind";
import { settledVolumeDistance } from "./volumeShaderBudget";
import { getVolumeValidity, setVolumeValidity } from "./volumeValidity";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume.js";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper.js";
import { configureCoherentVolumeSampling } from "./coherentVolumeSampling";
import { configureJointVolumeShader } from "./jointVolumeShader";
import { jointVolumeGeometry } from "./jointVolumeGeometry";
import { hasJointVolumeBinding, registerJointVolume, releaseJointVolume, type JointVolumeFrame } from "./jointVolumeRegistry";

type Actor = ReturnType<typeof vtkVolume.newInstance>;
type Mapper = ReturnType<typeof vtkVolumeMapper.newInstance> & {
  getViewSpecificProperties(): { OpenGL: Record<string, any> };
};
export interface JointVolumeSource { actor: Actor; role: "context" | "cloud" | "condensation" | "independent" | "history"; shadowReachM?: number }
const roles = { context: 0, cloud: 1, condensation: 2, independent: 3, history: 4 };

export function bindJointVolumes(sources: readonly JointVolumeSource[]) {
  if (sources.length < 2 || sources.length > 6) throw new Error("Joint volume binding requires two to six registered layers");
  if (sources.some(source => source.shadowReachM != null
      && (!(source.shadowReachM >= 0) || !Number.isFinite(source.shadowReachM)))) {
    throw new Error("Joint volume shadow reach must be finite and non-negative");
  }
  const sourceKeys = sources.map(({ actor }) => actor.getMapper());
  if (sourceKeys.some((source) => !source || hasJointVolumeBinding(source)) || new Set(sourceKeys).size !== sourceKeys.length) {
    throw new Error("Joint volume sources require distinct, unbound mappers");
  }
  const mapper = vtkVolumeMapper.newInstance() as Mapper;
  mapper.setAutoAdjustSampleDistances(false);
  configureCoherentVolumeSampling(mapper);
  configureJointVolumeShader(mapper);
  const volume = vtkVolume.newInstance();
  volume.setMapper(mapper);
  const property = volume.getProperty();
  property.setIndependentComponents(true);
  property.setInterpolationTypeToLinear();
  property.setAmbient(0); property.setDiffuse(1); property.setSpecular(0);
  for (let i = 0; i < 4; i++) { volume.setProperty(i, property); property.setScalarOpacityUnitDistance(i, 1); }
  const proxyImages = new WeakMap<object, object>();
  const owner = {};
  let bounds: number[] = [0, 1, 0, 1, 0, 1];
  let imageTime = 0;
  let disposed = false;
  let temporal: { pairs: readonly { current: Actor; previous: Actor }[]; fraction: number; displacement: readonly [number, number, number]; wind?: { model: AffineWind; intervalS: number; scale: { xy: number; z: number } } } | null = null;
  const resolve = (requested: object): JointVolumeFrame | null => {
    if (disposed) return null;
    const active = sources.filter(({ actor }) => actor.getVisibility()
      && actor.getMapper()?.getInputData()?.getNumberOfPoints() > 0);
    if (active.length === 0) { releaseJointVolume(owner); return null; }
    if (active.length > 4) throw new Error("Joint rendering supports at most four active textures");
    const lead = active[0].actor.getMapper()!;
    if (requested !== lead) return { owner, layers: active.length, lead, mapper, volume };
    const sourceMappers = active.map(({ actor }) => actor.getMapper()! as Mapper);
    const images = sourceMappers.map((source) => source.getInputData());
    for (const [index, image] of images.entries()) {
      if (image.getPointData().getScalars().getNumberOfComponents() !== 1
          || active[index].actor.getMatrix().some((value: number, i: number) => value !== Number(i % 5 === 0))) {
        throw new Error("Joint volumes require scalar images with identity actor transforms");
      }
      if (active[index].actor.getProperty().getSpecular() !== 0) throw new Error("Joint scientific volumes require zero specular material");
      if (sourceMappers[index].getClippingPlanes().length > 0) throw new Error("Joint volume source clipping planes require explicit ownership integration");
    }
    const peers = [-1, -1, -1, -1];
    const transforms = active.map(() => identityAffine());
    if (temporal) for (const pair of temporal.pairs) {
      const current = active.findIndex(source => source.actor === pair.current);
      const previous = active.findIndex(source => source.actor === pair.previous);
      if (current < 0 || previous < 0) continue;
      peers[current] = previous;
      if (temporal.wind) {
        transforms[current] = affineWindTransform(temporal.wind.model, -(1 - temporal.fraction) * temporal.wind.intervalS, temporal.wind.scale);
        transforms[previous] = affineWindTransform(temporal.wind.model, temporal.fraction * temporal.wind.intervalS, temporal.wind.scale);
      } else for (let axis = 0; axis < 3; axis++) {
        transforms[current][12 + axis] = -(1 - temporal.fraction) * temporal.displacement[axis];
        transforms[previous][12 + axis] = temporal.fraction * temporal.displacement[axis];
      }
    }
    // exp(|G|t) bounds both motion directions and keeps the shader loop budget fixed across a transition.
    const budgetMatrix = temporal?.wind ? affineWindTransform({ centerM: [0,0,0], velocityMs: [0,0,0],
      gradientPerS: temporal.wind.model.gradientPerS.map(Math.abs) }, temporal.wind.intervalS, temporal.wind.scale) : identityAffine();
    const budgetInverseNorm = Math.sqrt(Math.max(...[0,1,2].map(row => [0,1,2].reduce((sum,col) => sum + budgetMatrix[col*4+row],0)))
      * Math.max(...[0,1,2].map(col => [0,1,2].reduce((sum,row) => sum + budgetMatrix[col*4+row],0))));
    const geometry = jointVolumeGeometry(images.map((image, index) => ({
      getDimensions: () => image.getDimensions(), getSpacing: () => image.getSpacing(),
      getDirection: () => image.getDirection(),
      getOrigin: () => image.getOrigin(), getMotionTransform: () => transforms[index],
    })));
    bounds = geometry.bounds;
    imageTime = Math.max(...images.map((image) => image.getMTime()));
    let first = proxyImages.get(images[0]);
    if (!first) {
      const image = images[0];
      first = { ...image, getBounds: () => [...bounds], getMTime: () => Math.max(image.getMTime(), imageTime) };
      proxyImages.set(images[0], first!);
    }
    setVolumeValidity(first!, getVolumeValidity(images[0]));
    const inverseScale = (sourceMappers[0].getViewSpecificProperties().OpenGL?.nimbusInverseSceneScale ?? [1, 1, 1]) as number[];
    const references = [1, 1, 1, 1], roleValues = [-1, -1, -1, -1], reach = [0, 0, 0, 0], steps = [1, 1, 1, 1];
    const sampleSteps = [1, 1, 1, 1];
    const surface: number[][] = [], lighting: number[][] = [];
    let primaryWork = 0, shadowWork = 0, shaderPrimaryWork = 0, shaderShadowWork = 0;
    for (let i = 0; i < active.length; i++) {
      const source = sourceMappers[i], material = active[i].actor.getProperty();
      const optical = source.getViewSpecificProperties().OpenGL;
      const reference = Number(optical?.nimbusPhysicalReference);
      if (!(reference > 0) || !Number.isFinite(reference)
          || inverseScale.some((value, axis) => value !== optical?.nimbusInverseSceneScale?.[axis])) {
        throw new Error("Joint volumes require a common display scale and physical opacity references");
      }
      references[i] = reference; roleValues[i] = roles[active[i].role];
      sampleSteps[i] = source.getSampleDistance() * geometry.layers[i].sampleScale;
      const physicalDiagonal = Math.hypot(...geometry.layers[i].worldLength.map((value, axis) => value * inverseScale[axis]));
      reach[i] = active[i].shadowReachM ?? physicalDiagonal * material.getGlobalIlluminationReach();
      steps[i] = sampleSteps[i] * reference / material.getScalarOpacityUnitDistance(0) * source.getVolumeShadowSamplingDistFactor();
      if (!(sampleSteps[i] > 0) || !(steps[i] > 0) || !Number.isFinite(sampleSteps[i]) || !Number.isFinite(steps[i])) {
        throw new Error("Joint volume sample distances must be finite and positive");
      }
      primaryWork += Math.ceil(Math.hypot(...geometry.layers[i].worldLength) / sampleSteps[i]);
      shadowWork += Math.ceil(physicalDiagonal / steps[i]);
      const settled = (settledVolumeDistance(source) ?? source.getSampleDistance()) / budgetInverseNorm;
      const budgetLength = [0,1,2].map(row => geometry.layers[i].length.reduce((sum,value,col) => sum + budgetMatrix[col*4+row] * value,0));
      shaderPrimaryWork += Math.ceil(Math.hypot(...budgetLength) / settled);
      const budgetDiagonal = Math.hypot(...budgetLength.map((value, axis) => value * inverseScale[axis]));
      shaderShadowWork += Math.ceil(budgetDiagonal / (steps[i] * settled / sampleSteps[i]));
      surface.push([material.getAmbient(), material.getDiffuse(), 0, 0]);
      lighting.push([Number(material.getShade()), Number((material as typeof material & { getComputeNormalFromOpacity(): boolean }).getComputeNormalFromOpacity()), material.getVolumetricScatteringBlending(), material.getAnisotropy()]);
      property.setRGBTransferFunction(i, material.getRGBTransferFunction(0));
      property.setScalarOpacity(i, material.getScalarOpacity(0));
      if (i >= mapper.getNumberOfInputPorts()) mapper.addInputData(images[i]);
      else mapper.setInputData(i === 0 ? first : images[i], i);
    }
    for (let i = active.length; i < 4; i++) {
      if (i < mapper.getNumberOfInputPorts()) mapper.setInputData(null, i);
      surface.push([0, 0, 0, 0]); lighting.push([0, 0, 0, 0]);
    }
    property.setShade(active.some(({ actor }) => actor.getProperty().getShade()));
    property.setVolumetricScatteringBlending(Math.max(...lighting.map((value) => value[2])));
    mapper.setSampleDistance(Math.min(...sampleSteps.slice(0, active.length)));
    mapper.setImageSampleDistance(Math.min(...sourceMappers.map((source) => source.getImageSampleDistance())));
    const sampleBudget = Math.max(primaryWork, shadowWork) + 18;
    if (!Number.isSafeInteger(sampleBudget)) throw new Error("Joint volume ray budget exceeds numeric capacity");
    const shaderSampleBudget = Math.max(shaderPrimaryWork, shaderShadowWork) + 18;
    mapper.setMaximumSamplesPerRay(Math.max(sampleBudget, shaderSampleBudget + 2, ...sourceMappers.map((source) => source.getMaximumSamplesPerRay())));
    const props = mapper.getViewSpecificProperties();
    // Only the small per-layer uniforms change here; immutable scalar arrays
    // and their transfer functions retain their identities across camera moves.
    const gl = props.OpenGL;
    gl.nimbusPhysicalReference = 1;
    gl.nimbusInverseSceneScale = inverseScale;
    gl.nimbusJoint = { geometry, references, roles: roleValues, reach, steps, sampleSteps, sampleBudget, shaderSampleBudget, surface, lighting, temporalMasked: [0, 1, 2, 3].map(index => images[index] && getVolumeValidity(images[index]) ? 1 : 0), temporalPeers: peers, temporalFraction: temporal?.fraction ?? 1 };
    return { owner, layers: active.length, lead: sourceMappers[0], mapper, volume };
  };
  const detach = sources.map(({ actor }) => {
    const source = actor.getMapper();
    if (!source) throw new Error("Joint volume source has no mapper");
    return registerJointVolume(source, () => resolve(source));
  });
  const visibility = sources.map(({ actor }) => actor.onModified(() => {
    if (!sources.some((source) => source.actor.getVisibility())) releaseJointVolume(owner);
  }));
  return {
    setTemporal(pairs: readonly { current: Actor; previous: Actor }[], fraction: number, displacement: readonly [number, number, number], wind?: { model: AffineWind; intervalS: number; scale: { xy: number; z: number } }) {
      if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1 || !displacement.every(Number.isFinite)) {
        throw new Error("Cloud interpolation requires a bounded fraction and finite displacement");
      }
      temporal = pairs.length ? { pairs, fraction, displacement, wind } : null;
    },
    describe: () => {
      const count = sources.filter(({ actor }) => actor.getVisibility() && actor.getMapper()?.getInputData()?.getNumberOfPoints() > 0).length;
      return { mode: count > 1 ? "joint-optical-depth" : "single-volume", layers: count };
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      visibility.forEach((subscription) => subscription.unsubscribe());
      releaseJointVolume(owner);
      detach.forEach((unsubscribe) => unsubscribe());
      mapper.delete(); volume.delete(); property.delete();
    },
  };
}
