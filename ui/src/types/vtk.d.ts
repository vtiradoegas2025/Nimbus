/*
 * Module shims for vtk.js sub-paths that ship without .d.ts files.
 *
 * @kitware/vtk.js ships TS declarations for most public modules but
 * has gaps in `Filters/General` and a few rendering primitives. The
 * runtime exists; we just declare it as `any` so tsc stops complaining.
 *
 * Add entries here as needed when a new vtk.js sub-path is imported.
 */

declare module "@kitware/vtk.js/Rendering/OpenGL/glsl/vtkVolumeFS.glsl.js" {
  const shader: string;
  export default shader;
}

declare module "@kitware/vtk.js/Rendering/OpenGL/ShaderProgram.js" {
  const vtkShaderProgram: {
    substitute(source: string, search: string, replacement: string, all?: boolean): { result: string };
  };
  export default vtkShaderProgram;
}

declare module "@kitware/vtk.js/Rendering/OpenGL/VolumeMapper.js" {
  const vtkOpenGLVolumeMapper: {
    extend(publicApi: object, model: object, initialValues?: object): void;
  };
  export default vtkOpenGLVolumeMapper;
}

declare module "@kitware/vtk.js/Rendering/OpenGL/Framebuffer.js" {
  const vtkFramebuffer: {
    extend(publicApi: object, model: object, initialValues?: object): void;
  };
  export default vtkFramebuffer;
}

declare module "@kitware/vtk.js/Rendering/OpenGL/ViewNodeFactory.js" {
  export function registerOverride(name: string, factory: (initialValues?: object) => object): void;
  const factory: { newInstance(): { createNode(renderable: object): object } };
  export default factory;
}

declare module "@kitware/vtk.js/Rendering/OpenGL/Renderer.js" {
  const renderer: { extend(publicApi: object, model: object, initialValues?: object): void };
  export default renderer;
}

declare module "@kitware/vtk.js/Rendering/OpenGL/ForwardPass.js" {
  const pass: { newInstance(initialValues?: object): { delete(): void } };
  export default pass;
}

declare module "@kitware/vtk.js/Rendering/WebGPU/PolyDataMapper" {
  const vtkWebGpuPolyDataMapper: {
    extend(publicApi: object, model: object, initialValues?: object): void;
  };
  export default vtkWebGpuPolyDataMapper;
}

declare module "@kitware/vtk.js/Rendering/WebGPU/ViewNodeFactory" {
  export function registerOverride(name: string, factory: (initialValues?: object) => object): void;
}

declare module "@kitware/vtk.js/Rendering/WebGPU/CellArrayMapper" {
  const vtkWebGpuCellArrayMapper: {
    extend(publicApi: object, model: object, initialValues?: object): void;
  };
  export default vtkWebGpuCellArrayMapper;
}

declare module "@kitware/vtk.js/Rendering/WebGPU/BufferManager/Constants" {
  export const BufferUsage: { Strips: number };
}

declare module "@kitware/vtk.js/Filters/General/ImageMarchingCubes" {
  const vtkImageMarchingCubes: {
    extend(publicAPI: object, model: object, initialValues?: object): void;
    newInstance(initialValues?: {
      contourValue?: number;
      computeNormals?: boolean;
      mergePoints?: boolean;
    }): any;
  };
  export default vtkImageMarchingCubes;
}

declare module "@kitware/vtk.js/Filters/General/ImageMarchingCubes.js" {
  export { default } from "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
}

declare module "@kitware/vtk.js/Filters/General/ImageStreamline" {
  const vtkImageStreamline: {
    newInstance(initialValues?: {
      integrationStep?: number;
      maximumNumberOfSteps?: number;
    }): any;
  };
  export default vtkImageStreamline;
}

declare module "@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter" {
  const vtkWindowedSincPolyDataFilter: {
    newInstance(initialValues?: {
      numberOfIterations?: number;
      passBand?: number;
      boundarySmoothing?: boolean;
      featureEdgeSmoothing?: boolean;
      nonManifoldSmoothing?: boolean;
      normalizeCoordinates?: boolean;
    }): any;
  };
  export default vtkWindowedSincPolyDataFilter;
}

declare module "@kitware/vtk.js/Filters/Core/PolyDataNormals" {
  const vtkPolyDataNormals: {
    newInstance(initialValues?: {
      computePointNormals?: boolean;
      computeCellNormals?: boolean;
    }): any;
  };
  export default vtkPolyDataNormals;
}
