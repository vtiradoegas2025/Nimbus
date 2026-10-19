import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkTexture from "@kitware/vtk.js/Rendering/OpenGL/Texture";
import { Filter } from "@kitware/vtk.js/Rendering/OpenGL/Texture/Constants";
import { createFloat32VolumeTexture, VOLUME_SCALAR_SAMPLING_GLSL } from "@nimbus/lib/vtk/volumeScalarSampling";

export async function qualifyVolumeScalarPrecision() {
  const container = document.createElement("div");
  Object.assign(container.style, { position: "fixed", width: "128px", height: "32px", left: "0", top: "0" });
  document.body.append(container);
  const window = vtkGenericRenderWindow.newInstance();
  window.setContainer(container); window.resize();
  const view = window.getApiSpecificRenderWindow();
  const gl = view.getContext() as WebGL2RenderingContext;
  const nativeLinear = gl.getExtension("OES_texture_float_linear") != null;
  const owned: Array<{ delete(): void }> = [];
  const textures: Array<ReturnType<typeof vtkTexture.newInstance>> = [];
  const rows = [];
  try {
    const compile = (kind: number, source: string) => {
      const shader = gl.createShader(kind)!;
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "Precision shader failed");
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, `#version 300 es
      in vec2 position; void main() { gl_Position = vec4(position, 0.0, 1.0); }`);
    const fragment = compile(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float; precision highp int; precision highp sampler3D;
      uniform highp sampler3D source;
      uniform vec3 coordinate;
      ${VOLUME_SCALAR_SAMPLING_GLSL}
      out vec4 result;
      void main() {
        uint value = floatBitsToUint(nimbusScalarTexture(source, coordinate, 0).r);
        result = vec4(float(value & 255u), float((value >> 8u) & 255u), float((value >> 16u) & 255u), float(value >> 24u)) / 255.0;
      }`);
    const program = gl.createProgram()!;
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "Precision program failed");
    gl.useProgram(program);
    const buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const attribute = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(attribute); gl.vertexAttribPointer(attribute, 2, gl.FLOAT, false, 0, 0);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.DITHER); gl.viewport(0, 0, 1, 1);
    for (const dimensions of [[16, 8, 4], [32, 16, 8]]) for (const offset of [0, 280, 1000]) {
      const values = Float32Array.from({ length: dimensions[0] * dimensions[1] * dimensions[2] }, (_, index) =>
        offset + 0.01 * (index % dimensions[0]) + 0.007 * (Math.floor(index / dimensions[0]) % dimensions[1]) - 0.003 * Math.floor(index / (dimensions[0] * dimensions[1])));
      values[0] = -79; values[values.length - 1] = 1236;
      const data = vtkDataArray.newInstance({ values, numberOfComponents: 1 });
      owned.push(data);
      const proxyContext = new Proxy(gl, { get(target, key) {
        if (key === "getExtension") return (name: string) => name === "OES_texture_float_linear" ? null : target.getExtension(name);
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      } });
      const proxyView = { ...view, getContext: () => proxyContext };
      for (const mode of ["upstream-without-float-linear", "float32-manual", ...(nativeLinear ? ["float32-native"] : [])]) {
        const targetView = mode === "float32-native" ? view : proxyView;
        const texture = mode.startsWith("float32") ? createFloat32VolumeTexture(targetView, dimensions, data)
          : vtkTexture.newInstance({ minificationFilter: Filter.LINEAR, magnificationFilter: Filter.LINEAR });
        textures.push(texture);
        if (!mode.startsWith("float32")) {
          texture.setOpenGLRenderWindow(targetView);
          texture.create3DFilterableFromDataArray({ width: dimensions[0], height: dimensions[1], depth: dimensions[2], dataArray: data, preferSizeOverAccuracy: false });
        }
        texture.activate();
        gl.uniform1i(gl.getUniformLocation(program, "source"), texture.getTextureUnit());
        gl.uniform4fv(gl.getUniformLocation(program, "nimbusManualScalarFilter"), [mode === "float32-manual" ? 1 : 0, 0, 0, 0]);
        let maximumError = 0;
        const samples = [];
        for (const index of [[3, 2, 1], [dimensions[0] * 0.6, dimensions[1] * 0.5 + 0.25, dimensions[2] * 0.5]]) {
          const coordinate = index.map((value, axis) => (value + 0.5) / dimensions[axis]);
          gl.uniform3fv(gl.getUniformLocation(program, "coordinate"), coordinate);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
          const pixel = new Uint8Array(4);
          gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
          const actual = new DataView(pixel.buffer).getFloat32(0, true);
          const expected = Math.fround(offset + 0.01 * index[0] + 0.007 * index[1] - 0.003 * index[2]);
          maximumError = Math.max(maximumError, Math.abs(actual - expected));
          const texelCenter = index.every(Number.isInteger);
          const stored = texelCenter ? values[index[0] + dimensions[0] * (index[1] + dimensions[1] * index[2])] : null;
          samples.push({ index, actual, expected, texelCenter, stored });
        }
        rows.push({ mode, dimensions, offset, maximumError, samples, nativeFloatLinear: nativeLinear,
          internalFormat: texture.getInternalFormat(vtkDataArray.VtkDataTypes.FLOAT, 1),
          allocatedBytes: (texture as unknown as { getAllocatedGPUMemoryInBytes(): number }).getAllocatedGPUMemoryInBytes(), expectedScalarBytes: values.byteLength,
          glError: gl.getError() });
        texture.deactivate();
      }
    }
    gl.deleteBuffer(buffer); gl.deleteProgram(program); gl.deleteShader(vertex); gl.deleteShader(fragment);
  } finally {
    textures.forEach(texture => { texture.releaseGraphicsResources(view); texture.delete(); });
    owned.forEach(value => value.delete());
    window.getRenderWindow().getInteractor()?.unbindEvents();
    window.setContainer(null as unknown as HTMLElement); window.delete(); container.remove();
  }
  return rows;
}
