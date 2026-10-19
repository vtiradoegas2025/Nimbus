import assert from "node:assert/strict";
import vtkLight from "@kitware/vtk.js/Rendering/Core/Light.js";
import { DEFAULT_SCENE_APPEARANCE, normalizeShadowDepth, normalizeFillTint, resolveSceneAppearance, shadowMaterial } from "../../../../ui/src/lib/plot/sceneAppearance.ts";
import { applySurfaceLighting, applyVolumeLighting, createVtkSceneLighting } from "../../../../ui/src/lib/vtk/sceneLighting.ts";

assert.equal(normalizeShadowDepth(NaN),1);
assert.equal(normalizeShadowDepth(Infinity),1);
assert.equal(normalizeShadowDepth(-2),0);
assert.equal(normalizeShadowDepth(8),2);
assert.equal(normalizeFillTint("#A0B1C2"),"#a0b1c2");
assert.equal(normalizeFillTint("invalid"),null);
assert.deepEqual(resolveSceneAppearance(DEFAULT_SCENE_APPEARANCE,"dark").lightingOptions,{preset:"neutral",shadowDepth:1,fillTint:null});
assert.deepEqual(shadowMaterial(0.42,0.62,"soft"),{ambient:0.42,diffuse:0.62});
const settings = { preset: "neutral" as const, shadowDepth: 0, fillTint: "#ff0000" };
assert.deepEqual(shadowMaterial(0.48,0.58,settings),{ambient:1,diffuse:0});
assert.deepEqual(shadowMaterial(0.48,0.58,{...settings,shadowDepth:2}),{ambient:0,diffuse:1});
let ambient=0,diffuse=0,shade=false,lit=false;
const property = {setAmbient:(v:number)=>{ambient=v;},setDiffuse:(v:number)=>{diffuse=v;},setShade:(v:boolean)=>{shade=v;},setLighting:(v:boolean)=>{lit=v;},setSpecular:()=>{}};
applyVolumeLighting(property,settings,"cloud");
assert.deepEqual([ambient,diffuse,shade],[1,0,true]);
applyVolumeLighting(property,{...settings,preset:"flat",shadowDepth:2},"cloud");
assert.deepEqual([ambient,diffuse,shade],[1,0,false]);
applySurfaceLighting(property,{...settings,shadowDepth:2});
assert.equal(ambient,0);assert.equal(lit,true);
applySurfaceLighting(property,{...settings,preset:"flat"});assert.equal(lit,false);
const headlight = vtkLight.newInstance();
headlight.setLightTypeToHeadLight();
let lights = [headlight];
const rig = createVtkSceneLighting({
  getLights: () => [...lights],
  addLight: light => { lights.push(light); },
  removeLight: light => { lights = lights.filter(candidate => candidate !== light); },
  setAutomaticLightCreation: () => {},
  setLightFollowCamera: () => {},
  setTwoSidedLighting: () => {},
});
assert.equal(lights.length, 3);
assert.ok(lights.every(light => light.lightTypeIsSceneLight()));
rig.apply(settings,false);
assert.deepEqual(rig.fill.getColor(),[1,0,0]);
rig.apply("neutral",false);
assert.deepEqual(rig.fill.getColor(),[0.72,0.82,1]);
rig.apply({...settings,preset:"flat"},false);
assert.equal(rig.fill.getIntensity(),0);
rig.dispose();
assert.equal(lights.length, 0);
headlight.delete();
console.log("Shadow controls: defaults, bounds, tint reset, and exact-color bypass passed");
