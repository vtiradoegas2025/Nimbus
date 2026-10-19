/*
 * Stop-button escalation policy: interrupt first (KeyboardInterrupt), then force-terminate.
 * Interrupt only reaches Python bytecode; numpy kernels / awaits / non-COOP windows need the hard stop.
 * Pure module — testable without a worker or DOM.
 */

/** Whether Stop has already asked politely and is waiting for an answer. */
export type StopStage = "idle" | "interrupting";

/** Grace period before the worker is force-terminated. Not a latency budget — it is the window for an interruptible cell to respond. */
export const STOP_GRACE_MS = 2000;

/** What the caller should do to the interpreter right now. */
export type StopAction = "interrupt" | "terminate";

export interface StopDecision {
  readonly action: StopAction;
  /**
   * Milliseconds after which to force, or null when this decision is already
   * the forceful one and nothing further will follow.
   */
  readonly escalateAfterMs: number | null;
  /** The sentence to put in the panel's status line. */
  readonly status: string;
  /** The stage the caller moves to once the action has been carried out. */
  readonly nextStage: StopStage;
}

const GRACE_SECONDS = STOP_GRACE_MS / 1000;

/**
 * Spelled out wherever the interpreter is taken, because it is the part of Stop
 * a user can be surprised by. The cell stops either way; only this route also
 * empties the namespace.
 */
const COST = "Variables, imports and installed packages are gone; your scripts are not affected.";

export const STOP_STATUS_INTERRUPTING =
  `Interrupting… if the cell does not yield within ${GRACE_SECONDS} s it will be force-stopped.`;

/** Reported when the run stops within the grace period. Does not distinguish raised-interrupt from natural finish; both leave the interpreter intact. */
export const STOP_STATUS_INTERRUPTED =
  "Stopped. The interpreter was left intact, so variables and imports survive.";

export const STOP_STATUS_FORCED =
  `Force-stopped: the cell ignored the interrupt, so the interpreter was replaced. ${COST}`;

export const STOP_STATUS_NO_INTERRUPT =
  "Stopped by replacing the interpreter: this window is not cross-origin isolated, so a "
  + `running cell cannot be interrupted. ${COST}`;

/** Decide the next Stop action. canInterrupt=false collapses to immediate terminate (no polite route available). */
export function decideStop(
  input: { readonly stage: StopStage; readonly canInterrupt: boolean },
): StopDecision {
  if (input.stage === "idle" && input.canInterrupt) {
    return {
      action: "interrupt",
      escalateAfterMs: STOP_GRACE_MS,
      status: STOP_STATUS_INTERRUPTING,
      nextStage: "interrupting",
    };
  }
  return {
    action: "terminate",
    escalateAfterMs: null,
    status: input.canInterrupt ? STOP_STATUS_FORCED : STOP_STATUS_NO_INTERRUPT,
    nextStage: "idle", // interpreter gone; next press starts fresh
  };
}

/** The button's label, which names the action the NEXT press will take. */
export function stopButtonLabel(stage: StopStage): string {
  return stage === "interrupting" ? "Force stop" : "Stop";
}

/** The button's tooltip, for the same press. */
export function stopButtonTitle(stage: StopStage, canInterrupt: boolean): string {
  if (stage === "interrupting") {
    return `Replace the interpreter now instead of waiting out the ${GRACE_SECONDS} s grace `
      + `period. ${COST}`;
  }
  return canInterrupt
    ? "Raise KeyboardInterrupt in the running cell. A cell that ignores it — a numpy kernel, "
      + `a download, a pending await — is force-stopped after ${GRACE_SECONDS} s instead.`
    : `Replace the interpreter, which is the only way to stop a cell here. ${COST}`;
}
