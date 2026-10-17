import type { GridInfo, NestedHistoryLevel, StepInfo } from "./runOutputs";

const TIME_RELATIVE_TOLERANCE = 1e-9;
const COORDINATE_TOLERANCE_M = 1e-6;

export interface ResolvedOutputLevel {
  levelId: "root" | string;
  parentLevelId: "root" | string | null;
  runDir: string;
  step: StepInfo;
  /** Zero is the root; increasing values are successively finer levels. */
  depth: number;
  horizontalCellAreaM2: number;
}

export type OutputHierarchyVerdict =
  | {
      ok: true;
      levels: ResolvedOutputLevel[];
      /** Deepest currently published nodes; each identifies one selectable branch. */
      leaves: ResolvedOutputLevel[];
      /** Backward-compatible automatic choice. Ambiguous siblings resolve to root. */
      finest: ResolvedOutputLevel;
      disclosure: string;
    }
  | {
      ok: false;
      levels: [ResolvedOutputLevel];
      leaves: [ResolvedOutputLevel];
      finest: ResolvedOutputLevel;
      reason: string;
    };

export interface OutputHierarchySelection {
  level: ResolvedOutputLevel;
  /** Root-to-selected-level chain, including both endpoints. */
  chain: ResolvedOutputLevel[];
  requestedLevelId: "auto" | string;
  disclosure: string;
  fallbackReason: string | null;
}

export interface NestedVolumeOwnershipLayer {
  level: ResolvedOutputLevel;
  excludeChildBoundsM: GridInfo["physicalBoundsM"] | null;
}

function finitePositive(value: number | undefined): value is number {
  return value != null && Number.isFinite(value) && value > 0;
}

function horizontalArea(grid: GridInfo | null): number | null {
  if (!grid || grid.coordinateSystem !== "cartesian") return null;
  if (!finitePositive(grid.drM)) return null;
  const dy = grid.dyM ?? grid.drM;
  return finitePositive(dy) ? grid.drM * dy : null;
}

function finiteBounds(grid: GridInfo | null): GridInfo["physicalBoundsM"] | null {
  const bounds = grid?.physicalBoundsM;
  if (!bounds
      || !bounds.lower.every(Number.isFinite)
      || !bounds.upper.every(Number.isFinite)
      || bounds.lower.some((value, axis) => value >= bounds.upper[axis])) return null;
  return bounds;
}

function sameFrameTime(left: number | null, right: number | null): boolean {
  if (left == null || right == null || !Number.isFinite(left) || !Number.isFinite(right)) return false;
  const tolerance = Math.max(
    1e-6,
    Math.abs(left) * TIME_RELATIVE_TOLERANCE,
    Math.abs(right) * TIME_RELATIVE_TOLERANCE,
  );
  return Math.abs(left - right) <= tolerance;
}

function containsBounds(parent: GridInfo["physicalBoundsM"], child: GridInfo["physicalBoundsM"]): boolean {
  if (!parent || !child) return false;
  return parent.lower.every((lower, axis) =>
    child.lower[axis] >= lower - COORDINATE_TOLERANCE_M
      && child.upper[axis] <= parent.upper[axis] + COORDINATE_TOLERANCE_M
  );
}

function containsPoint(level: ResolvedOutputLevel, point: readonly [number, number] | null): boolean {
  if (!point) return false;
  const bounds = finiteBounds(level.step.grid);
  return bounds != null
    && point[0] >= bounds.lower[0] - COORDINATE_TOLERANCE_M
    && point[0] <= bounds.upper[0] + COORDINATE_TOLERANCE_M
    && point[1] >= bounds.lower[1] - COORDINATE_TOLERANCE_M
    && point[1] <= bounds.upper[1] + COORDINATE_TOLERANCE_M;
}

function frameForRootStep(root: StepInfo, level: NestedHistoryLevel): StepInfo | null {
  const exactIndex = level.steps.find((candidate) => candidate.index === root.index);
  if (!exactIndex) return null;
  return exactIndex.publicationState === "complete"
      && exactIndex.publicationGeneration != null
      && exactIndex.publicationGeneration === root.publicationGeneration
      && sameFrameTime(exactIndex.simulationTimeS, root.simulationTimeS)
    ? exactIndex
    : null;
}

function hasCanonicalGraphIdentity(level: NestedHistoryLevel): boolean {
  const parts = level.levelId.split("/");
  if (parts.length !== level.depth
      || parts.some((part) => !/^child_[0-9]{4}$/.test(part))) return false;
  const expectedParent = parts.length === 1 ? "root" : parts.slice(0, -1).join("/");
  return level.parentLevelId === expectedParent;
}

function hierarchyFailure(root: ResolvedOutputLevel, reason: string): OutputHierarchyVerdict {
  return { ok: false, levels: [root], leaves: [root], finest: root, reason };
}

function levelDisclosure(level: ResolvedOutputLevel): string {
  if (level.depth === 0) return "Root history";
  const spacing = level.step.grid?.drM;
  const label = level.levelId.split("/").map((part) => part.replace("_", " ")).join(" › ");
  return `${label} · ${spacing != null ? `${spacing.toLocaleString()} m` : "fine"} native`;
}

/**
 * Resolve the complete co-temporal refinement graph for one displayed root
 * frame. Missing child cadence removes that branch for this frame. Any
 * structurally false parent relationship fails the entire graph closed to the
 * root; Studio never repairs hierarchy from geometry alone.
 */
export function resolveOutputHierarchyFrame(
  rootRunDir: string,
  root: StepInfo,
  nested: readonly NestedHistoryLevel[],
): OutputHierarchyVerdict {
  const rootArea = horizontalArea(root.grid);
  const rootBounds = finiteBounds(root.grid);
  const rootLevel: ResolvedOutputLevel = {
    levelId: "root",
    parentLevelId: null,
    runDir: rootRunDir,
    step: root,
    depth: 0,
    horizontalCellAreaM2: rootArea ?? Number.POSITIVE_INFINITY,
  };
  if (nested.length === 0) {
    return { ok: true, levels: [rootLevel], leaves: [rootLevel], finest: rootLevel, disclosure: "Root history" };
  }
  if (root.publicationState !== "complete" || root.publicationGeneration == null) {
    return hierarchyFailure(rootLevel, "Nested history requires a complete generation-bound root frame.");
  }
  if (rootArea == null || rootBounds == null) {
    return hierarchyFailure(rootLevel, "Nested history requires finite Cartesian root spacing and Earth bounds.");
  }

  const definitions = new Map<string, NestedHistoryLevel>();
  for (const level of nested) {
    if (!hasCanonicalGraphIdentity(level)) {
      return hierarchyFailure(rootLevel, `${level.levelId} has a non-canonical hierarchy identity.`);
    }
    if (level.levelId === "root" || definitions.has(level.levelId)) {
      return hierarchyFailure(rootLevel, `Duplicate or reserved nested level identity '${level.levelId}'.`);
    }
    definitions.set(level.levelId, level);
  }
  const ordered = [...nested].sort((left, right) => left.depth - right.depth || left.levelId.localeCompare(right.levelId));
  const resolved: ResolvedOutputLevel[] = [rootLevel];
  const byId = new Map<string, ResolvedOutputLevel>([["root", rootLevel]]);
  for (const level of ordered) {
    const parentDefinition = level.parentLevelId === "root" ? null : definitions.get(level.parentLevelId);
    const expectedDepth = (parentDefinition?.depth ?? 0) + 1;
    if ((level.parentLevelId !== "root" && !parentDefinition) || level.depth !== expectedDepth) {
      return hierarchyFailure(rootLevel, `${level.levelId} has an invalid explicit parent/depth relationship.`);
    }
    const step = frameForRootStep(root, level);
    if (!step) continue;
    const parent = byId.get(level.parentLevelId);
    // A grandchild cannot float into the graph on a cadence where its parent
    // did not publish a matching frame.
    if (!parent) continue;
    const area = horizontalArea(step.grid);
    const bounds = finiteBounds(step.grid);
    const parentBounds = finiteBounds(parent.step.grid);
    if (area == null || bounds == null || parentBounds == null) {
      return hierarchyFailure(rootLevel, `${level.levelId} has incomplete Cartesian Earth geometry.`);
    }
    if (!(area < parent.horizontalCellAreaM2)) {
      return hierarchyFailure(rootLevel, `${level.levelId} is not finer than its declared parent.`);
    }
    if (!containsBounds(parentBounds, bounds)) {
      return hierarchyFailure(rootLevel, `${level.levelId} extends outside its declared parent Earth bounds.`);
    }
    const resolvedLevel: ResolvedOutputLevel = {
      levelId: level.levelId,
      parentLevelId: level.parentLevelId,
      runDir: level.runDir,
      step,
      depth: level.depth,
      horizontalCellAreaM2: area,
    };
    resolved.push(resolvedLevel);
    byId.set(level.levelId, resolvedLevel);
  }

  const parentIds = new Set(resolved.slice(1).map((level) => level.parentLevelId));
  const leaves = resolved.filter((level) => !parentIds.has(level.levelId));
  const nonRootLeaves = leaves.filter((level) => level.depth > 0);
  const finest = nonRootLeaves.length === 1 ? nonRootLeaves[0] : rootLevel;
  return {
    ok: true,
    levels: resolved,
    leaves: leaves.length > 0 ? leaves : [rootLevel],
    finest,
    disclosure: nonRootLeaves.length > 1
      ? `${nonRootLeaves.length} refinement branches · select or follow focus`
      : levelDisclosure(finest),
  };
}

function chainToLevel(verdict: OutputHierarchyVerdict, level: ResolvedOutputLevel): ResolvedOutputLevel[] {
  const byId = new Map(verdict.levels.map((candidate) => [candidate.levelId, candidate]));
  const chain: ResolvedOutputLevel[] = [];
  let current: ResolvedOutputLevel | undefined = level;
  while (current) {
    chain.push(current);
    current = current.parentLevelId ? byId.get(current.parentLevelId) : undefined;
  }
  return chain.reverse();
}

/** Select one explicit branch/level and fall back only through its declared
 * ancestors when a requested field is absent. Auto mode follows the finest
 * node containing the Earth-coordinate focus; without a focus it selects only
 * an unambiguous single branch. */
export function selectOutputHierarchyLevel(
  verdict: OutputHierarchyVerdict,
  options: {
    requestedLevelId?: "auto" | string;
    focusPointM?: readonly [number, number] | null;
    requiredFields?: readonly string[];
  } = {},
): OutputHierarchySelection {
  const root = verdict.levels[0];
  const requestedLevelId = options.requestedLevelId ?? "auto";
  const requiredFields = options.requiredFields ?? [];
  let target = root;
  let fallbackReason: string | null = null;
  if (verdict.ok) {
    if (requestedLevelId !== "auto") {
      const explicit = verdict.levels.find((level) => level.levelId === requestedLevelId);
      if (explicit) target = explicit;
      else fallbackReason = `Selected level '${requestedLevelId}' is unavailable for this frame.`;
    } else {
      const focused = options.focusPointM
        ? verdict.levels.filter((level) => level.depth > 0 && containsPoint(level, options.focusPointM ?? null))
        : [];
      if (focused.length > 0) {
        const compatible = focused.filter(level => requiredFields.every(field => level.step.fields.includes(field)));
        target = (compatible.length ? compatible : focused).sort((left, right) =>
          left.horizontalCellAreaM2 - right.horizontalCellAreaM2
            || right.depth - left.depth
            || left.levelId.localeCompare(right.levelId)
        )[0];
      } else if (options.focusPointM) {
        if (verdict.levels.length > 1) fallbackReason = "No refinement level covers the focus point; using root data.";
      } else if (verdict.leaves.filter((level) => level.depth > 0).length === 1) {
        target = verdict.leaves.find((level) => level.depth > 0) ?? root;
      } else if (verdict.levels.length > 1) {
        fallbackReason = "Auto level needs a focus point when multiple refinement branches are present.";
      }
    }
  } else {
    fallbackReason = verdict.reason;
  }

  const requestedTarget = target;
  const targetChain = chainToLevel(verdict, target);
  let fieldMatch: ResolvedOutputLevel | null = null;
  for (let index = targetChain.length - 1; index >= 0; index -= 1) {
    const candidate = targetChain[index];
    if (requiredFields.every((field) => candidate.step.fields.includes(field))) {
      fieldMatch = candidate;
      break;
    }
  }
  target = fieldMatch ?? root;
  if (target.levelId !== requestedTarget.levelId && requiredFields.length > 0) {
    fallbackReason = fieldMatch
      ? `${requestedTarget.levelId} does not publish ${requiredFields.join(", ")}; using ${target.levelId}.`
      : `No level on ${requestedTarget.levelId}'s branch publishes ${requiredFields.join(", ")}.`;
  }
  return {
    level: target,
    chain: chainToLevel(verdict, target),
    requestedLevelId,
    disclosure: levelDisclosure(target),
    fallbackReason,
  };
}

/** Finest level on the default unambiguous branch that publishes all fields. */
export function finestLevelForFields(
  verdict: OutputHierarchyVerdict,
  fields: readonly string[],
): ResolvedOutputLevel {
  return selectOutputHierarchyLevel(verdict, { requiredFields: fields }).level;
}

export function planNestedVolumeOwnership(
  selection: OutputHierarchySelection,
  field: string,
): NestedVolumeOwnershipLayer[] {
  if (!field) throw new Error("Nested volume ownership requires a field");
  if (selection.chain.some((level) => !level.step.fields.includes(field))) {
    throw new Error(`Every nested ownership level must publish '${field}'`);
  }
  return selection.chain.map((level, index) => {
    const child = selection.chain[index + 1] ?? null;
    const childBounds = child ? finiteBounds(child.step.grid) : null;
    if (child && !childBounds) {
      throw new Error(`${child.levelId} has no finite exclusion bounds`);
    }
    return {
      level,
      excludeChildBoundsM: childBounds ? {
        lower: [...childBounds.lower],
        upper: [...childBounds.upper],
      } : null,
    };
  });
}
