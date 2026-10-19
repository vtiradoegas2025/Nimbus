/*
 * Following another pane's field. Pure — no React.
 * Unit-tested in scripts/plot/test_field_link.ts.
 */

/** What a pane should do about the field another pane just moved to. */
export interface FieldLinkDecision {
  /** Field to switch to, or null to stay where it is. */
  adopt: string | null;
  /** The field it could not follow, for an on-screen note. Null when nothing was refused. */
  declined: string | null;
}

const STAY: FieldLinkDecision = Object.freeze({ adopt: null, declined: null });

/**
 * Whether this pane can follow `linkedField`.
 *
 * A pane never follows silently into something it cannot draw: a volume renderer asked for a
 * single-level product keeps what it is showing and reports the refusal, so the two panes are
 * never both claiming to be linked while showing different quantities.
 *
 * Only user-initiated changes should reach this; adopting must not re-broadcast, or two linked
 * panes would drive each other.
 */
export function resolveFieldLink(
  linkedField: string | null | undefined,
  currentField: string | null,
  canShow: (field: string) => boolean,
): FieldLinkDecision {
  if (!linkedField) return STAY;
  if (linkedField === currentField) return STAY;
  return canShow(linkedField)
    ? { adopt: linkedField, declined: null }
    : { adopt: null, declined: linkedField };
}

/** On-canvas explanation for a field this renderer refused to follow. */
export function describeDeclinedFieldLink(field: string, reason: string): string {
  return `Linked panes moved to ${field}; ${reason}`;
}

/** Reason text for a 3-D renderer that cannot draw a single-level field. */
export const SINGLE_LEVEL_LINK_REASON =
  "it has no vertical dimension, so this view stayed on its own field.";
