import { selectCameraVolumeLevel } from "../../../../ui/src/lib/plot/volumeLodPolicy.ts";

const levels = [
  { level: 0, stride: [1, 1, 1] as [number, number, number] },
  { level: 1, stride: [2, 2, 2] as [number, number, number] },
  { level: 2, stride: [4, 4, 4] as [number, number, number] },
];
let passed = 0;
const check = (condition: boolean, message: string) => {
  if (!condition) throw new Error(message);
  passed += 1;
};

check(selectCameraVolumeLevel(levels, {
  nativeSpacingM: 100, cameraRangeM: 1_000_000, viewportHeightPx: 600,
}) === 2, "far cameras select the coarsest subpixel level");
check(selectCameraVolumeLevel(levels, {
  nativeSpacingM: 100, cameraRangeM: 1_000, viewportHeightPx: 600,
}) === 0, "close cameras retain native detail");
check(selectCameraVolumeLevel(levels, {
  nativeSpacingM: 100, cameraRangeM: 1_000, viewportHeightPx: 600,
  currentLevel: 0, interacting: true,
}) === 0, "interaction retains the committed scientific generation");
check(selectCameraVolumeLevel(levels, {
  nativeSpacingM: 100, cameraRangeM: 1_000_000, viewportHeightPx: 600,
  currentLevel: 1, interacting: true,
}) === 1, "a zoom gesture cannot replace the committed generation mid-motion");
check(selectCameraVolumeLevel(levels, {
  nativeSpacingM: 100, cameraRangeM: 172_000, viewportHeightPx: 600,
  currentLevel: 1,
}) === 1, "refinement hysteresis prevents a threshold orbit from thrashing");
check(selectCameraVolumeLevel(levels, {
  nativeSpacingM: 100, cameraRangeM: 100_000, viewportHeightPx: 600,
  currentLevel: 1,
}) === 0, "a materially closer camera crosses hysteresis and refines");

console.log(`\nvolume LOD policy tests: ${passed} passed, 0 failed.`);
