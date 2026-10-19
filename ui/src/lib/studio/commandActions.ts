import { openLocalDatasetSource } from "@/lib/dataset/datasetRuntime";
import type { DatasetSourceKind, LocalDatasetSource } from "@/lib/dataset/datasetSources";
import type { StudioCommandId } from "./commandRegistry";

export type DatasetSourceOpener = (
  kind: DatasetSourceKind,
  mode?: "files" | "folder",
) => Promise<LocalDatasetSource | null>;

export type DatasetOpenResult =
  | { status: "opened"; source: LocalDatasetSource }
  | { status: "cancelled" }
  | { status: "not-handled" };

export async function openDatasetForCommand(
  commandId: StudioCommandId,
  opener: DatasetSourceOpener = openLocalDatasetSource,
): Promise<DatasetOpenResult> {
  const request = datasetRequestForCommand(commandId);
  if (!request) return { status: "not-handled" };
  const source = await opener(request.kind, request.mode);
  return source ? { status: "opened", source } : { status: "cancelled" };
}

export function datasetRequestForCommand(
  commandId: StudioCommandId,
): { kind: DatasetSourceKind; mode: "files" | "folder" } | null {
  switch (commandId) {
    case "dataset.open-cm1-history":
      return { kind: "cm1_netcdf", mode: "files" };
    case "dataset.open-nimbus-history":
      return { kind: "nimbus_run", mode: "folder" };
    default:
      return null;
  }
}
