/* Decides what a plot shows while its next figure is still resolving. Pure - no React, no DOM. */

/** A figure kept from an earlier render, tagged with the view it was built for. */
export interface RetainedFigure<T> {
  scope: string;
  figure: T;
}

export interface RetainedSelection<T> {
  /** What to render now. Null only when there is nothing valid to show. */
  shown: T | null;
  /** True while `shown` is the previous figure and the current one has not resolved. */
  stale: boolean;
  /** What to carry into the next render. */
  retain: RetainedFigure<T> | null;
}

/**
 * Choose between the figure just built and the one held from before.
 *
 * Plots that combine independently-loaded fields go null for the window between the first field
 * arriving and the last, because a figure mixing two frames would be a lie. Rendering the canvas on
 * that null tears the whole graph down and rebuilds it; showing the previous figure instead lets it
 * update in place.
 *
 * `scope` names everything that would make the held figure wrong to show - run, field names, plane,
 * reducer. A scope change discards it rather than presenting another view's picture as the current
 * one, which is the difference between a smooth transition and a lie.
 */
export function selectRetainedFigure<T>(
  current: T | null | undefined,
  retained: RetainedFigure<T> | null,
  scope: string,
): RetainedSelection<T> {
  if (current != null) {
    return { shown: current, stale: false, retain: { scope, figure: current } };
  }
  if (retained != null && retained.scope === scope) {
    return { shown: retained.figure, stale: true, retain: retained };
  }
  // Out of scope: drop it so a later return to this scope cannot resurrect a figure
  // built for a view the user has since left.
  return { shown: null, stale: false, retain: null };
}
