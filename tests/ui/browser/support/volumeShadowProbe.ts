import { VOLUME_OPTICAL_RATIO_GLSL, VOLUME_SHADOW_INTEGRATION_GLSL } from "@nimbus/lib/vtk/volumeShadowSampling";
import vtkVolumeShader from "@kitware/vtk.js/Rendering/OpenGL/glsl/vtkVolumeFS.glsl.js";

export function qualifyVolumeShadowSampling(upstream = false) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const gl = canvas.getContext("webgl2", { antialias: false });
  if (!gl) throw new Error("Shadow qualification requires WebGL2");
  const program = gl.createProgram()!;
  const shaders: WebGLShader[] = [];
  const texture = gl.createTexture()!;
  const buffer = gl.createBuffer()!;
  try {
    const shadowSource = upstream ? vtkVolumeShader.match(/float computeVolumeShadowWithoutCache\(vec3 posVC, vec3 lightDirNormVC\) \{[\s\S]*?\n  \}/)?.[0]
      : VOLUME_SHADOW_INTEGRATION_GLSL;
    if (!shadowSource) throw new Error("Installed VTK shadow template is unavailable");
    const compile = (kind: number, source: string) => {
      const shader = gl.createShader(kind)!;
      shaders.push(shader);
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "Shadow shader failed");
      gl.attachShader(program, shader);
    };
    compile(gl.VERTEX_SHADER, `#version 300 es
      in vec2 position;
      void main() { gl_Position = vec4(position, 0.0, 1.0); }`);
    compile(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      precision highp int;
      #define texture2D texture
      #define vtkMaximumNumberOfSamples 4096
      #define vtkClippingPlanesOn
      struct Volume {
        mat3 vecVCToISMatrix;
        ivec3 dimensions;
        vec3 spacing;
        vec4 opacityTextureScale;
        vec4 opacityTextureShift;
        float diagonalLength;
        float globalIlluminationReach;
      };
      uniform Volume volume;
      uniform float sampleDistance, volumeShadowSampleDistance, nimbusOpacityRatio, nimbusPhysicalReference;
      uniform vec3 nimbusInverseSceneScale, startPoint, lightDirection, primaryDirection;
      uniform highp sampler2D opacityTexture;
      uniform int clip_numPlanes;
      uniform vec3 vClipPlaneOrigins[1], vClipPlaneNormals[1];
      out vec4 result;
      const float fragmentSeed = 0.5;
      const float EPSILON = 0.001;
      const float infinity = 3.402823466e38;
      vec3 posVCtoIS(vec3 point) { return volume.vecVCToISMatrix * point; }
      vec3 vecVCToIS(vec3 direction) { return volume.vecVCToISMatrix * direction; }
      vec4 getTextureValue(vec3 point) { return vec4(0.5); }
      bool nimbusValid0(vec3 point) { return true; }
      vec2 rayIntersectVolumeDistances(vec3 point, vec3 direction) {
        vec3 p = posVCtoIS(point), d = vecVCToIS(direction);
        vec2 interval = vec2(-1e30, 1e30);
        for (int axis = 0; axis < 3; ++axis) {
          if (d[axis] == 0.0) { if (p[axis] < 0.0 || p[axis] > 1.0) return vec2(1.0, -1.0); }
          else {
            float a = -p[axis] / d[axis], b = (1.0 - p[axis]) / d[axis];
            interval.x = max(interval.x, min(a,b)); interval.y = min(interval.y, max(a,b));
          }
        }
        return interval;
      }
      ${VOLUME_OPTICAL_RATIO_GLSL}
      float getOpacityFromTexture(float scalar, int component, float height) {
        return 1.0 - pow(1.0 - nimbusReferenceOpacity(scalar, component, height), nimbusRayOpticalRatio(primaryDirection, sampleDistance));
      }
      ${shadowSource}
      void main() {
        float shadow = computeVolumeShadowWithoutCache(startPoint, normalize(lightDirection));
        float raw = nimbusReferenceOpacity(0.5, 0, 0.5);
        float primary = 1.0 - pow(1.0 - raw, nimbusRayOpticalRatio(primaryDirection, sampleDistance));
        result = vec4(shadow, primary, 0.0, 1.0);
      }`);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "Shadow program failed");
    gl.useProgram(program);
    const location = (name: string) => gl.getUniformLocation(program, name);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const attribute = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(attribute); gl.vertexAttribPointer(attribute, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.uniform1i(location("opacityTexture"), 0);
    gl.uniform3iv(location("volume.dimensions"), [10, 10, 10]);
    gl.uniform4fv(location("volume.opacityTextureScale"), [1, 1, 1, 1]);
    gl.uniform4fv(location("volume.opacityTextureShift"), [0, 0, 0, 0]);
    gl.uniform1f(location("nimbusPhysicalReference"), 500);
    const rows = [];
    for (const scale of [[1, 1, 1], [1, 1, 5]]) {
      for (const direction of [[1, 0, 0], [0, 0, 1], [1, 0, 1]]) {
        for (const primaryStep of [10, 37, 91]) for (const shadowFactor of [1, 4, 8]) {
          for (const clipped of [false, true]) for (const reach of [0, 0.1, 1]) {
            const raw = 0.5;
            const displayDirection = direction.map((v, axis) => v * scale[axis]);
            const norm = Math.hypot(...displayDirection);
            const unit = displayDirection.map((v) => v / norm);
            const distance = Math.min(...unit.map((v, axis) => v > 0 ? 500 * scale[axis] / v : Infinity));
            const clipDistance = clipped ? distance * 0.43 : distance;
            const physicalPerDisplay = Math.hypot(...unit.map((v, axis) => v / scale[axis]));
            gl.uniform3fv(location("volume.spacing"), scale.map((v) => v * 100));
            gl.uniform3fv(location("nimbusInverseSceneScale"), scale.map((v) => 1 / v));
            gl.uniformMatrix3fv(location("volume.vecVCToISMatrix"), false, [1 / (1000 * scale[0]), 0, 0, 0, 1 / (1000 * scale[1]), 0, 0, 0, 1 / (1000 * scale[2])]);
            gl.uniform1f(location("volume.diagonalLength"), Math.hypot(...scale.map((v) => v * 1000)));
            gl.uniform1f(location("volume.globalIlluminationReach"), reach);
            gl.uniform1f(location("sampleDistance"), primaryStep);
            gl.uniform1f(location("volumeShadowSampleDistance"), primaryStep * shadowFactor);
            gl.uniform3fv(location("startPoint"), scale.map((v) => v * 500));
            gl.uniform3fv(location("lightDirection"), unit);
            gl.uniform3fv(location("primaryDirection"), unit);
            gl.uniform1i(location("clip_numPlanes"), Number(clipped));
            gl.uniform3fv(location("vClipPlaneOrigins"), scale.map((v, axis) => v * 500 + unit[axis] * clipDistance));
            gl.uniform3fv(location("vClipPlaneNormals"), unit.map((v) => -v));
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array([raw]));
            gl.drawArrays(gl.TRIANGLES, 0, 3);
            const pixel = new Uint8Array(4);
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
            if (gl.getError() !== gl.NO_ERROR) throw new Error("Shadow probe generated a WebGL error");
            rows.push({ scale, direction, primaryStep, shadowFactor, clipped, reach,
              expectedShadow: (1 - raw) ** (Math.min(clipDistance * physicalPerDisplay, Math.sqrt(3) * 1000 * reach) / 500), actualShadow: pixel[0] / 255,
              expectedPrimary: 1 - (1 - raw) ** (primaryStep * physicalPerDisplay / 500), actualPrimary: pixel[1] / 255,
            });
          }
        }
      }
    }
    return rows;
  } finally {
    gl.deleteBuffer(buffer); gl.deleteTexture(texture); gl.deleteProgram(program);
    shaders.forEach((shader) => gl.deleteShader(shader));
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
