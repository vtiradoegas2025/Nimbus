import type { ExternalProviderExecutionRequest } from "./externalDerivedProviders";
import type { InjectionContext } from "@/lib/platform/pyodideProtocol";
import { gridForWindowedField } from "@/lib/science/windowedGrid";

export function pythonProviderContext(request: ExternalProviderExecutionRequest): InjectionContext {
  const fields = request.provider.inputs.map(({ alias }) => ({ name: alias, field: request.inputs[alias].field }));
  const reference = fields[0]?.field;
  const grid = reference ? gridForWindowedField(reference, request.grid) : request.grid;
  const geometry = reference?.windowGeometry;
  const modelLevelIndices = grid?.terrainFollowing && reference?.shape.length === 3
    ? Array.from({ length: reference.shape[2] }, (_, index) =>
      (geometry?.origin?.[2] ?? 0) + index * (geometry?.stride?.[2] ?? 1))
    : undefined;
  return {
    fields,
    grid,
    modelLevelIndices,
    simulationTimeS: request.frame.simulationTimeS,
    stepIndex: request.frame.index,
    frameIndex: null,
    frameCount: 0,
    sources: [{ role: "primary", displayName: request.datasetIdentity, frameCount: 0 }],
    fieldUnits: Object.fromEntries(request.provider.inputs.flatMap(({ alias }) => request.inputs[alias].unit
      ? [[alias, request.inputs[alias].unit!]] : [])),
  };
}
