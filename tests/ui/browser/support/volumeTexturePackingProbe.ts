import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import { createFloat32VolumeTexture, volumeScalarTextureLayout, VOLUME_SCALAR_SAMPLING_GLSL } from "@nimbus/lib/vtk/volumeScalarSampling";
import { acquireVolumeValidity, setVolumeValidity, VOLUME_VALIDITY_GLSL } from "@nimbus/lib/vtk/volumeValidity";
import { volumeTextureAddress } from "@nimbus/lib/vtk/volumeTexturePacking";
import { disposeVtkScene } from "@nimbus/lib/vtk/vtkLifecycle";
import { volumeSamplingShader } from "@nimbus/lib/vtk/volumeSamplingShader";
import { Filter } from "@kitware/vtk.js/Rendering/OpenGL/Texture/Constants";

export async function qualifyVolumeTexturePacking() {
  const container = document.createElement("div");
  Object.assign(container.style, { position: "fixed", width: "512px", height: "1024px", left: "0", top: "0" });
  document.body.append(container);
  const window = vtkGenericRenderWindow.newInstance(); window.setContainer(container); window.resize();
  const view = window.getApiSpecificRenderWindow();
  const gl = view.getContext() as WebGL2RenderingContext;
  const actualMaximum = Number(gl.getParameter(gl.MAX_3D_TEXTURE_SIZE));
  const cleanup: (() => void)[] = [];
  try {
  const compile = (kind: number, source: string) => {
    const shader = gl.createShader(kind)!; cleanup.push(() => gl.deleteShader(shader)); gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "Packing shader failed");
    return shader;
  };
  const vertex = compile(gl.VERTEX_SHADER, `#version 300 es
    in vec2 position; void main() { gl_Position = vec4(position, 0.0, 1.0); }`);
  const programs = new Map<string, WebGLProgram>();
  const programFor = (key: string, scalar: string, validity: string, fetch: string, maskSample: string) => {
    const cached = programs.get(key);
    if (cached) return cached;
    const fragment = compile(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float; precision highp int; precision highp sampler3D;
    uniform highp sampler3D source;
    uniform vec3 coordinate;
    uniform int readGrid, outputWidth, cells, maskValue;
    uniform vec3 probeDimensions;
    struct DummyVolume { vec3 inverseDimensions; }; uniform DummyVolume volume;
    ${scalar}
    ${validity}
    out vec4 result;
    void main() {
      float sampled;
      if (readGrid == 1) {
        int ordinal = int(gl_FragCoord.x) + outputWidth * int(gl_FragCoord.y);
        if (ordinal >= cells) { result = vec4(0.0); return; }
        ivec3 dimensions = ivec3(probeDimensions);
        ivec3 index = ivec3(ordinal % dimensions.x, (ordinal / dimensions.x) % dimensions.y,
          ordinal / (dimensions.x * dimensions.y));
        sampled = ${fetch}.r;
      } else sampled = maskValue == 2 ? (nimbusValid0(coordinate) ? 1.0 : 0.0)
        : maskValue == 1 ? ${maskSample} : nimbusScalarTexture(source, coordinate, 0).r;
      uint bits = floatBitsToUint(sampled);
      result = vec4(float(bits & 255u), float((bits >> 8u) & 255u), float((bits >> 16u) & 255u), float(bits >> 24u)) / 255.0;
    }`);
    const program = gl.createProgram()!; cleanup.push(() => gl.deleteProgram(program)); gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "Packing program failed");
    programs.set(key,program);
    return program;
  };
  const buffer = gl.createBuffer()!; cleanup.push(() => gl.deleteBuffer(buffer)); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  const rows = [];
    for (const [shape, maximum] of [[[2,3,41],8], [[23,2,3],8], [[2,31,3],8], [[2,3,actualMaximum + 3],actualMaximum], [[8,6,4],actualMaximum]] as const) for (const pattern of ["positive-affine", "signed-mantissas"] as const) {
      const dimensions = [...shape];
      const count = dimensions.reduce((total, value) => total * value, 1);
      const values = Float32Array.from({ length: count }, (_, ordinal) => 280
        + 0.01 * (ordinal % dimensions[0])
        + 0.007 * (Math.floor(ordinal / dimensions[0]) % dimensions[1])
        - 0.003 * Math.floor(ordinal / (dimensions[0] * dimensions[1])));
      const sourceBits = new Uint32Array(values.buffer);
      if (pattern === "signed-mantissas") {
        const bits = [0x80000000, 0, 0x3f800001, 0xbf000001, 0x438c0001, 0xc38c0001, 0x447a0001, 0xc47a0001];
        for (let index = 0; index < count; index += 11) sourceBits[index] = bits[Math.floor(index / 11) % bits.length];
      }
      const before = sourceBits.slice();
      const mask = Uint8Array.from({ length: count }, (_, ordinal) => ordinal % 13 === 0 ? 0 : 255);
      const data = vtkDataArray.newInstance({ values, numberOfComponents: 1 });
      const context = new Proxy(gl, { get(target, key) {
        if (key === "getParameter") return (parameter: number) => parameter === target.MAX_3D_TEXTURE_SIZE ? maximum : target.getParameter(parameter);
        const value = Reflect.get(target,key,target); return typeof value === "function" ? value.bind(target) : value;
      } });
      const targetView = { ...view, getContext: () => context };
      const texture = createFloat32VolumeTexture(targetView, dimensions, data);
      const image = {}; setVolumeValidity(image, mask);
      const validity = acquireVolumeValidity(targetView, image, dimensions)!;
      const layout = volumeScalarTextureLayout(texture)!;
      for (const shaderMode of layout.tiled ? ["generic", "production"] as const
        : gl.getExtension("OES_texture_float_linear") ? ["production-manual", "production-hardware"] as const : ["production-manual"] as const) {
      const specialization = volumeSamplingShader([layout], [shaderMode !== "production-hardware"], { validity: [true], nearest: [false] });
      const generic = shaderMode === "generic";
      const program = programFor(generic ? "generic" : `${shaderMode}:${specialization.key}`,
        generic ? VOLUME_SCALAR_SAMPLING_GLSL : specialization.scalar,
        generic ? VOLUME_VALIDITY_GLSL : specialization.validity,
        generic ? "nimbusLogicalTexel(source, index, dimensions, 0)" : layout.tiled ? "nimbusPackedTexel0(source, index)" : "texelFetch(source, index, 0)",
        generic ? "nimbusValidityValue(source, coordinate, 0)" : layout.tiled ? "nimbusPackedSample0(source, coordinate).r" : "texture(source, coordinate).r");
      const nativeNearest = volumeSamplingShader([layout], [false], { validity: [true], nearest: [true] });
      const nearestProgram = generic ? program : programFor(`nearest:${nativeNearest.key}`,
        nativeNearest.scalar, nativeNearest.validity,
        layout.tiled ? "nimbusPackedTexel0(source, index)" : "texelFetch(source, index, 0)",
        layout.tiled ? "nimbusPackedSample0(source, coordinate).r" : "texture(source, coordinate).r");
      const attribute = gl.getAttribLocation(program, "position");
      gl.enableVertexAttribArray(attribute); gl.vertexAttribPointer(attribute, 2, gl.FLOAT, false, 0, 0);
      const read = (input: typeof texture | typeof validity.texture, grid: boolean, coordinate: number[], maskMode = 0, nearest = false) => {
        const activeProgram = nearest ? nearestProgram : program;
        if (!layout.tiled && input === texture) {
          const filter = nearest || !gl.getExtension("OES_texture_float_linear") ? Filter.NEAREST : Filter.LINEAR;
          texture.setMinificationFilter(filter);
          texture.setMagnificationFilter(filter);
        }
        gl.useProgram(activeProgram); input.activate();
        const attribute = gl.getAttribLocation(activeProgram, "position");
        gl.enableVertexAttribArray(attribute); gl.vertexAttribPointer(attribute, 2, gl.FLOAT, false, 0, 0);
        gl.uniform1i(gl.getUniformLocation(activeProgram,"source"), input.getTextureUnit());
        gl.uniform1i(gl.getUniformLocation(activeProgram,"nimbusValidityTexture[0]"), input.getTextureUnit());
        gl.uniform4fv(gl.getUniformLocation(activeProgram,"nimbusValidityEnabled"),[1,0,0,0]);
        gl.uniform4fv(gl.getUniformLocation(activeProgram,"nimbusTiledTexture"),[Number(layout.tiled),0,0,0]);
        gl.uniform3fv(gl.getUniformLocation(activeProgram,"nimbusSourceDimensions[0]"),dimensions);
        gl.uniform3fv(gl.getUniformLocation(activeProgram,"probeDimensions"),dimensions);
        gl.uniform4fv(gl.getUniformLocation(activeProgram,"nimbusManualScalarFilter"),[1,0,0,0]);
        gl.uniform4fv(gl.getUniformLocation(activeProgram,"nimbusNearestScalarFilter"),[nearest?1:0,0,0,0]);
        gl.uniform1i(gl.getUniformLocation(activeProgram,"readGrid"),grid?1:0);
        gl.uniform1i(gl.getUniformLocation(activeProgram,"maskValue"),maskMode);
        gl.uniform1i(gl.getUniformLocation(activeProgram,"cells"),count);
        gl.uniform3fv(gl.getUniformLocation(activeProgram,"coordinate"),coordinate);
        const width = grid ? Math.min(512,count) : 1, height = grid ? Math.ceil(count/width) : 1;
        gl.uniform1i(gl.getUniformLocation(activeProgram,"outputWidth"),width);
        gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.DITHER); gl.viewport(0,0,width,height);
        gl.drawArrays(gl.TRIANGLES,0,3);
        const pixels = new Uint8Array(width*height*4); gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
        input.deactivate(); return new Float32Array(pixels.buffer);
      };
      const fetched = read(texture,true,[0,0,0]);
      const fetchedBits = new Uint32Array(fetched.buffer);
      const fetchedMask = read(validity.texture,true,[0,0,0]);
      let scalarMismatches=0, maskMismatches=0;
      for (let ordinal=0;ordinal<count;ordinal++) {
        if (fetchedBits[ordinal] !== sourceBits[ordinal]) scalarMismatches++;
        if (fetchedMask[ordinal] !== mask[ordinal]/255) maskMismatches++;
      }
      const reference = (source: Float32Array | Uint8Array, index: number[]) => {
        const lower=index.map((value,axis)=>Math.floor(Math.max(0,Math.min(dimensions[axis]-1,value))));
        const upper=lower.map((value,axis)=>Math.min(dimensions[axis]-1,value+1));
        const fraction=index.map((value,axis)=>Math.max(0,Math.min(dimensions[axis]-1,value))-lower[axis]);
        let value=0;
        for(let corner=0;corner<8;corner++) {
          const xyz=lower.map((item,axis)=>(corner>>axis)&1?upper[axis]:item);
          const weight=fraction.reduce((total,item,axis)=>total*((corner>>axis)&1?item:1-item),1);
          value+=weight*source[xyz[0]+dimensions[0]*(xyz[1]+dimensions[1]*xyz[2])] / (source instanceof Uint8Array?255:1);
        }
        return value;
      };
      const probes = [[0.25,0.75,0.5], dimensions.map(value=>value-1.25),
        [Math.min(1.25,dimensions[0]-1),Math.min(1.5,dimensions[1]-1),Math.min(7.75,dimensions[2]-1)],
        [0,0,0], dimensions.map(value=>value-1), [-0.75,0,0], dimensions.map(value=>value-0.25)];
      const seams: { axis: number; index: number[]; addresses: number[][] }[] = [];
      for (const axis of [0,1,2]) {
        let added = 0;
        for (let ordinal = 0; ordinal < count && added < 3; ordinal++) {
          const index = [ordinal%dimensions[0], Math.floor(ordinal/dimensions[0])%dimensions[1], Math.floor(ordinal/(dimensions[0]*dimensions[1]))];
          if (index[axis] + 1 >= dimensions[axis]) continue;
          const next = [...index]; next[axis]++;
          const address = volumeTextureAddress(index,layout), nextAddress = volumeTextureAddress(next,layout);
          if (address[1] === nextAddress[1] && address[2] === nextAddress[2]) continue;
          const fractional = [...index]; fractional[axis] += 0.375;
          seams.push({axis,index:fractional,addresses:[address,nextAddress]}); probes.push(fractional); added++;
        }
      }
      const interpolation=[];
      for(const requestedIndex of probes) {
        const coordinate=requestedIndex.map((value,axis)=>Math.fround((value+0.5)/dimensions[axis]));
        const index=coordinate.map((value,axis)=>Math.max(0,Math.min(dimensions[axis]-1,Math.fround(Math.fround(value*dimensions[axis])-0.5))));
        const expectedMask = reference(mask,index);
        const nearestIndex = coordinate.map((value,axis)=>Math.max(0,Math.min(dimensions[axis]-1,Math.floor(Math.fround(value*dimensions[axis])))));
        const nearestOrdinal = nearestIndex[0]+dimensions[0]*(nearestIndex[1]+dimensions[1]*nearestIndex[2]);
        const nearest = read(texture,false,coordinate,0,true);
        interpolation.push({ requestedIndex, index, coordinate, scalar: read(texture,false,coordinate)[0], expectedScalar: reference(values,index),
          mask: read(validity.texture,false,coordinate,1)[0], expectedMask,
          valid: read(validity.texture,false,coordinate,2)[0], expectedValid: expectedMask >= 0.999999 ? 1 : 0,
          nearestBits: new Uint32Array(nearest.buffer)[0], expectedNearestBits: sourceBits[nearestOrdinal] });
      }
      rows.push({ sourceShape:dimensions, pattern, shaderMode, actualMaximum, admittedMaximum:maximum, layout, seams,
        scalarMismatches, maskMismatches, interpolation,
        checkedScalarCells:count, unchangedSource: sourceBits.every((value,index)=>value===before[index]),
        unchangedMask: mask.every((value,index)=>value===(index%13===0?0:255)),
        minimum: data.getRange(0)[0], textureMinimum: (texture.getVolumeInfo() as unknown as {dataComputedOffset:number[]}).dataComputedOffset[0],
        gpuBytes:(texture as unknown as {getAllocatedGPUMemoryInBytes():number}).getAllocatedGPUMemoryInBytes(),
        maskGpuBytes:(validity.texture as unknown as {getAllocatedGPUMemoryInBytes():number}).getAllocatedGPUMemoryInBytes(),
        interpolationTolerance: Math.max(...data.getRange(0).map(Math.abs))*2e-6,
        hardwareInterpolation: shaderMode === "production-hardware",
        maskInterpolationTolerance: layout.tiled ? 2e-6 : 1/256,
        nativeShaderHasPacking: !layout.tiled && /nimbusPacked|nimbusTextureAddress|ordinal/.test(specialization.scalar),
        nativeHardwareShaderHasTexelFetch: shaderMode === "production-hardware" && /texelFetch/.test(specialization.scalar),
        glError:gl.getError() });
      }
      validity.release(); texture.releaseGraphicsResources(view); texture.delete(); data.delete();
    }
    return rows;
  } finally {
    gl.useProgram(null);
    disposeVtkScene({ clearScene: () => {}, cleanupSteps: [...cleanup.reverse(), () => container.remove()], renderWindow: window });
  }
}
