import type { ScalarPreparationRequest, PreparedVtkImage } from "./vtkPreparation";

export type VolumeBrickWorkerRequest =
  | { kind: "prepare"; requestId: number; request: ScalarPreparationRequest }
  | { kind: "prepare-anomaly-batch"; requestId: number; requests: ScalarPreparationRequest[] }
  | { kind: "close" };

export type VolumeBrickWorkerReply =
  | { kind: "ready" }
  | { kind: "closed" }
  | { kind: "prepared"; requestId: number; ok: true; prepared: PreparedVtkImage; bytes: number }
  | { kind: "prepared-batch"; requestId: number; ok: true; prepared: PreparedVtkImage[]; bytes: number }
  | { kind: "prepared" | "prepared-batch"; requestId: number; ok: false; error: string };
