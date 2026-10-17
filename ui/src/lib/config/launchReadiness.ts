/** Launch/validation readiness: reason is the primitive; blocked-ness is derived from it. */

export type CatalogKind = "ready" | (string & {});
export type PreflightKind = "pending" | "blocked" | (string & {});
export type SpawnKind = "starting" | (string & {});

/** The subset of ConfigEditor's CheckpointState this policy depends on. */
export type CheckpointStatus =
  | { kind: "empty" }
  | { kind: "checking"; path: string }
  | { kind: "accepted"; requestedPath: string }
  | { kind: "error"; path: string; message: string };

export interface LaunchReadinessInputs {
  /** False in a browser/mock context, where no real binary can be launched. */
  desktop: boolean;
  catalogKind: CatalogKind;
  preflightKind: PreflightKind;
  spawnKind: SpawnKind;
  /** Empty string means a fresh run, in which case checkpoint state is moot. */
  resumeCheckpointPath: string;
  checkpoint: CheckpointStatus;
}

const NOT_DESKTOP =
  "A real simulator can only be preflighted and launched from Nimbus Studio desktop";
const CATALOG_PENDING = "Waiting for the configured simulator's scheme catalog";
const PREFLIGHT_PENDING = "Checking the exact binary before launch";
const SPAWN_STARTING = "Starting the run";

/**
 * Conditions shared by Run and Validate, in the order the user should hear
 * them: the environment first, then the catalog, then work already in flight.
 */
function commonBlock(inputs: LaunchReadinessInputs): string | null {
  if (!inputs.desktop) return NOT_DESKTOP;
  if (inputs.catalogKind !== "ready") return CATALOG_PENDING;
  if (inputs.preflightKind === "pending") return PREFLIGHT_PENDING;
  if (inputs.spawnKind === "starting") return SPAWN_STARTING;
  return null;
}

/**
 * Why the resume checkpoint blocks a launch, or null. Only meaningful when a
 * resume path is set — an unaccepted checkpoint for a path the user has since
 * cleared must not block anything.
 */
export function describeCheckpointBlock(
  resumeCheckpointPath: string,
  checkpoint: CheckpointStatus,
): string | null {
  if (resumeCheckpointPath.length === 0) return null;
  if (checkpoint.kind === "checking") return "Inspecting the resume checkpoint";
  if (checkpoint.kind === "error") {
    return `The resume checkpoint could not be read: ${checkpoint.message}`;
  }
  if (checkpoint.kind === "accepted" && checkpoint.requestedPath === resumeCheckpointPath) {
    return null;
  }
  // Accepted-but-for-a-different-path, or never inspected at all.
  return "The resume checkpoint has not been accepted — open Checkpoints and continuation to review it";
}

/** Why launching is blocked, or null when it is ready to launch. */
export function describeLaunchBlock(inputs: LaunchReadinessInputs): string | null {
  const common = commonBlock(inputs);
  // Preflight findings and the checkpoint are launch-only concerns: validation
  // is precisely how the user clears the former, so it must stay reachable.
  if (common != null && common !== SPAWN_STARTING) return common;
  if (inputs.preflightKind === "blocked") {
    return "Edit the configuration to invalidate these blocking findings";
  }
  const checkpoint = describeCheckpointBlock(inputs.resumeCheckpointPath, inputs.checkpoint);
  if (checkpoint != null) return checkpoint;
  return common;
}

/** Why validating is blocked, or null when it is ready to validate. */
export function describeValidationBlock(inputs: LaunchReadinessInputs): string | null {
  return commonBlock(inputs);
}
