import {
  FIELD_BRICK_HIERARCHY_SCHEMA,
  FIELD_BRICK_HIERARCHY_VERSION,
  adjacentBrickIds,
  assertCompatibleBrickHierarchies,
  expandVectorBrickNeighborhood,
  planAllContourBricks,
  planContourBricks,
  planVolumeBricks,
  planVectorBrickNeighborhood,
  scalarIntervalHasOpacity,
  selectContourHierarchyLevel,
  selectVectorBrickFeatureFocus,
  selectVolumeHierarchyPlan,
  unionContourBrickPlans,
  validateFieldBrickHierarchy,
  type FieldBrickHierarchy,
} from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";

let passed = 0;
const failures: string[] = [];

function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

function throws(fn: () => unknown, text: string, label: string): void {
  try {
    fn();
    failures.push(label);
  } catch (error) {
    check(error instanceof Error && error.message.includes(text), label);
  }
}

function hierarchy(): FieldBrickHierarchy {
  const shape: [number, number, number] = [130, 70, 35];
  const brickShape: [number, number, number] = [64, 64, 32];
  const bricks = [];
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 2; j += 1) {
      for (let k = 0; k < 2; k += 1) {
        const origin: [number, number, number] = [i * 64, j * 64, k * 32];
        const extent = origin.map((value, axis) =>
          Math.min(brickShape[axis], shape[axis] - value)) as [number, number, number];
        const cells = extent.reduce((product, count) => product * count, 1);
        const strong = i === 1 && j === 0;
        bricks.push({
          id: `l0:${i}:${j}:${k}`,
          index: [i, j, k] as [number, number, number],
          origin,
          extent,
          scalarRange: {
            minimum: strong ? -2 : 0,
            maximum: strong ? 3 : 0.1,
            finiteCount: cells,
            nonFiniteCount: 0,
          },
          physicalBoundsM: {
            lower: origin.map((value) => value * 100) as [number, number, number],
            upper: origin.map((value, axis) => (value + extent[axis]) * 100) as [number, number, number],
          },
          contentSha256: String(i * 4 + j * 2 + k).padStart(64, "0"),
        });
      }
    }
  }
  return {
    schema: FIELD_BRICK_HIERARCHY_SCHEMA,
    schemaVersion: FIELD_BRICK_HIERARCHY_VERSION,
    field: "vorticity_magnitude",
    publicationGeneration: "generation-23",
    sourceShape: shape,
    sourceIndexOffset: [10, 20, 30],
    sourceDtype: "float32",
    sampleLocation: "cell_center",
    extremaSemantics: "stored_values",
    extremaAbsolutePadding: 0,
    cullingSafe: true,
    levels: [{
      level: 0,
      stride: [1, 1, 1],
      shape,
      brickShape,
      source: { kind: "primary_field" },
      bricks,
    }],
  };
}

const source = hierarchy();
const valid = validateFieldBrickHierarchy(source, {
  field: source.field,
  publicationGeneration: source.publicationGeneration,
  sourceShape: source.sourceShape,
  sourceIndexOffset: source.sourceIndexOffset,
});
check(valid.ok, "complete native hierarchy validates");
if (!valid.ok) throw new Error(valid.reason);
check(valid.hierarchy.levels[0].bricks.length === 12, "edge bricks complete the topology");
check(valid.hierarchy.levels[0].bricks.at(-1)?.extent.join("x") === "2x6x3", "edge extent is exact");

const multilevelCandidate = structuredClone(source);
multilevelCandidate.schemaVersion = 2;
const lodShape: [number, number, number] = [65, 35, 18];
const lodBrickShape: [number, number, number] = [64, 64, 32];
multilevelCandidate.levels.push({
  level: 1,
  stride: [2, 2, 2],
  shape: lodShape,
  brickShape: lodBrickShape,
  source: {
    kind: "npy_lod",
    path: "vorticity_magnitude.lod1.npy",
    contentSha256: "f".repeat(64),
    aggregation: "recursive_cell_mean_2x2x2",
  },
  bricks: [0, 1].map((i) => {
    const origin: [number, number, number] = [i * 64, 0, 0];
    const extent: [number, number, number] = [Math.min(64, 65 - origin[0]), 35, 18];
    const cells = extent.reduce((product, count) => product * count, 1);
    return {
      id: `l1:${i}:0:0`,
      index: [i, 0, 0] as [number, number, number],
      origin,
      extent,
      scalarRange: { minimum: -2, maximum: 3, finiteCount: cells, nonFiniteCount: 0 },
      physicalBoundsM: {
        lower: [origin[0] * 200, 0, 0] as [number, number, number],
        upper: [Math.min(130, (origin[0] + extent[0]) * 2) * 100, 7000, 3500] as [number, number, number],
      },
      contentSha256: String(32 + i).padStart(64, "0"),
    };
  }),
});
const multilevel = validateFieldBrickHierarchy(multilevelCandidate);
check(multilevel.ok, "durable multilevel hierarchy validates");
if (!multilevel.ok) throw new Error(multilevel.reason);
const selectedLevel = selectContourHierarchyLevel(multilevel.hierarchy, {
  maximumPreparedCells: 50_000,
  threshold: 1,
});
check(selectedLevel === 1, "contour selection falls back to the finest admitted durable LOD");
const lodPlan = planContourBricks(multilevel.hierarchy, 1, { level: selectedLevel });
check(lodPlan.items[0].readWindow.origin.join("x") === "10x20x30", "LOD reads preserve storage offset");
check(lodPlan.items[0].readWindow.extent.join("x") === "130x70x35", "LOD reads map level cells onto exact source support");
check(lodPlan.items[0].readWindow.stride.join("x") === "2x2x2", "LOD reads disclose source stride");
check(lodPlan.items[0].readWindow.maxSourceBytes === lodPlan.items[0].readWindow.maxOutputBytes,
  "durable LOD source accounting reads only the reduced payload");
throws(() => selectContourHierarchyLevel(multilevel.hierarchy, {
  maximumPreparedCells: 50_000,
  threshold: 1,
  terrainFollowing: true,
}), "coarsest durable", "terrain selection never uses a level without durable physical geometry");
const terrainMultilevelCandidate = structuredClone(multilevel.hierarchy);
terrainMultilevelCandidate.schemaVersion = 3;
const terrainLodSource = terrainMultilevelCandidate.levels[1].source;
if (terrainLodSource.kind !== "npy_lod") throw new Error("terrain test LOD source is invalid");
terrainLodSource.geometry = {
  kind: "physical_volume_centroids",
  xM: { path: "physical_x_center_m.lod1.npy", contentSha256: "a".repeat(64) },
  yM: { path: "physical_y_center_m.lod1.npy", contentSha256: "b".repeat(64) },
  zM: { path: "physical_z_center_m.lod1.npy", contentSha256: "c".repeat(64) },
  volumeM3: { path: "physical_volume_m3.lod1.npy", contentSha256: "d".repeat(64) },
};
const terrainMultilevel = validateFieldBrickHierarchy(terrainMultilevelCandidate);
check(terrainMultilevel.ok, "durable terrain LOD geometry validates");
if (!terrainMultilevel.ok) throw new Error(terrainMultilevel.reason);
check(selectContourHierarchyLevel(terrainMultilevel.hierarchy, {
  maximumPreparedCells: 50_000,
  threshold: 1,
  terrainFollowing: true,
}) === 1, "terrain selection uses the finest admitted level with durable physical geometry");
check(selectVolumeHierarchyPlan(
  multilevel.hierarchy,
  [[-2, 1], [3, 1]],
  { maximumResidentBytes: 200_000 },
).level === 1, "volume selection chooses the finest complete level within aggregate residency");
const terrainVolumePlan = selectVolumeHierarchyPlan(
  terrainMultilevel.hierarchy,
  [[-2, 1], [3, 1]],
  { maximumResidentBytes: 200_000, terrainFollowing: true },
);
check(terrainVolumePlan.level === 1,
  "terrain volume selection uses durable physical geometry at the admitted LOD");
check(terrainVolumePlan.items.every((item) =>
  (item.window.maxOutputBytes ?? 0) > item.preparedScalarBytes),
"terrain volume admission separates physical input geometry from GPU scalar residency");
const vectorV = structuredClone(multilevel.hierarchy);
vectorV.field = "v";
for (const level of vectorV.levels) {
  if (level.source.kind === "npy_lod") level.source.path = `v.lod${level.level}.npy`;
}
const vectorW = structuredClone(multilevel.hierarchy);
vectorW.field = "w";
for (const level of vectorW.levels) {
  if (level.source.kind === "npy_lod") level.source.path = `w.lod${level.level}.npy`;
}
const vectorPlan = planVectorBrickNeighborhood(
  [multilevel.hierarchy, vectorV, vectorW],
  { maximumPreparedCells: 50_000, brickRadius: 1 },
);
check(vectorPlan.level === 1 && vectorPlan.items.length === 2,
  "vector adjacency planning selects a complete durable neighborhood under budget");
check(vectorPlan.preparedCells === vectorPlan.items.reduce((sum, item) =>
  sum + (item.window.maxOutputBytes ?? 0) / 4, 0),
"vector adjacency planning accounts every prepared cell once");
const terrainVectorV = structuredClone(terrainMultilevel.hierarchy);
terrainVectorV.field = "v";
const terrainVectorW = structuredClone(terrainMultilevel.hierarchy);
terrainVectorW.field = "w";
const terrainVectorPlan = planVectorBrickNeighborhood(
  [terrainMultilevel.hierarchy, terrainVectorV, terrainVectorW],
  { maximumPreparedCells: 50_000, brickRadius: 1, terrainFollowing: true },
);
check(terrainVectorPlan.level === 1,
  "terrain vector planning selects a durable geometry LOD under cell admission");
check(terrainVectorPlan.items.every((item) =>
  item.window.maxOutputBytes === (() => {
    const cells = item.window.extent.map((extent, axis) =>
      Math.floor((extent - 1) / item.window.stride[axis]) + 1
    ).reduce((product, count) => product * count, 1);
    return cells * 36 + (cells % 2 === 1 ? 4 : 0);
  })()), "terrain vector windows admit scalar, centroid, volume, and alignment payloads");
const featureFocus = selectVectorBrickFeatureFocus(
  [valid.hierarchy, { ...valid.hierarchy, field: "v" }, { ...valid.hierarchy, field: "w" }],
);
check(featureFocus[0] >= 64 && featureFocus[0] < 128 && featureFocus[1] < 64,
  "component extrema locate the strongest vector brick without field reads");
const elevated = structuredClone(valid.hierarchy);
for (const brick of elevated.levels[0].bricks) {
  if (brick.index[2] === 1) {
    brick.scalarRange.minimum = -100;
    brick.scalarRange.maximum = 100;
  }
}
const nearSurfaceFocus = selectVectorBrickFeatureFocus(
  [elevated, { ...elevated, field: "v" }, { ...elevated, field: "w" }],
  { maximumSourceZFraction: 0.25 },
);
check(nearSurfaceFocus[2] < 32,
  "near-surface vortex focus excludes stronger elevated bricks");
throws(() => selectVectorBrickFeatureFocus(
  [valid.hierarchy, { ...valid.hierarchy, field: "v" }, { ...valid.hierarchy, field: "w" }],
  { maximumSourceZFraction: 0 },
), "height fraction", "invalid focus height constraint fails closed");
const nativeSeedPlan = planVectorBrickNeighborhood(
  [valid.hierarchy, { ...valid.hierarchy, field: "v" }, { ...valid.hierarchy, field: "w" }],
  {
    maximumPreparedCells: 64 * 64 * 32,
    expansion: { lower: [0, 0, 0], upper: [0, 0, 0] },
  },
);
check(nativeSeedPlan.level === 0 && nativeSeedPlan.items.length === 1,
  "trajectory planning begins with only the focus brick");
const expanded = expandVectorBrickNeighborhood(nativeSeedPlan, {
  lower: [true, false, false],
  upper: [true, true, false],
});
check(expanded?.lower.join("x") === "1x0x0" && expanded.upper.join("x") === "1x1x0",
  "trajectory exits expand only requested in-domain faces");
const edgePlan = planVectorBrickNeighborhood(
  [valid.hierarchy, { ...valid.hierarchy, field: "v" }, { ...valid.hierarchy, field: "w" }],
  {
    focusSourceIndex: [0, 0, 0],
    maximumPreparedCells: 64 * 64 * 32,
    expansion: { lower: [0, 0, 0], upper: [0, 0, 0] },
  },
);
check(expandVectorBrickNeighborhood(edgePlan, {
  lower: [true, true, true], upper: [false, false, false],
}) == null, "dataset-boundary exits do not request nonexistent bricks");
throws(() => planVectorBrickNeighborhood(
  [valid.hierarchy, { ...valid.hierarchy, field: "v" }, { ...valid.hierarchy, field: "w" }],
  { maximumPreparedCells: 1, expansion: { lower: [0, 0, 0], upper: [0, 0, 0] } },
), "coarsest", "trajectory expansion fails before reads when no durable level fits");

for (const [mutate, reason, label] of [
  [(value: any) => { value.schemaVersion = 4; }, "schema/version", "future schema fails closed"],
  [(value: any) => { value.levels[0].bricks.pop(); }, "incomplete", "missing brick fails closed"],
  [(value: any) => { value.levels[0].bricks[0].id = "wrong"; }, "noncanonical", "brick id is canonical"],
  [(value: any) => { value.levels[0].bricks[0].origin[0] = 1; }, "topology-aligned", "origin alignment is required"],
  [(value: any) => { value.levels[0].bricks[0].extent[0] = 63; }, "gap or overlap", "extent coverage is required"],
  [(value: any) => { value.levels[0].bricks[0].scalarRange.finiteCount -= 1; }, "cover", "validity counts cover samples"],
  [(value: any) => { value.levels[0].bricks[0].contentSha256 = "abc"; }, "checksum", "checksums are mandatory"],
  [(value: any) => { value.levels[0].bricks[0].physicalBoundsM.upper[0] = 0; }, "increasing", "physical bounds are increasing"],
] as const) {
  const candidate = structuredClone(source);
  mutate(candidate);
  const result = validateFieldBrickHierarchy(candidate);
  check(!result.ok && result.reason.includes(reason), label);
}
check(!validateFieldBrickHierarchy(source, { field: "w" }).ok, "field identity cannot drift");
check(!validateFieldBrickHierarchy(source, { publicationGeneration: "new" }).ok, "generation cannot drift");
check(!validateFieldBrickHierarchy(source, { sourceShape: [1, 2, 3] }).ok, "shape cannot drift");
check(!validateFieldBrickHierarchy(source, { sourceIndexOffset: [0, 0, 0] }).ok, "storage offset cannot drift");

const contour = planContourBricks(valid.hierarchy, 1, { maximumSourceBytes: 16 * 1024 * 1024 });
check(contour.items.length === 2, "extrema cull every non-crossing contour brick");
check(contour.culledBrickCount === 10, "contour cull count is reported");
check(contour.items.every((item) => item.brickIndex[0] === 1 && item.brickIndex[1] === 0), "selected contour bricks match extrema");
const interior = contour.items.find((item) => item.brickIndex[2] === 0)!;
check(interior.readWindow.origin.join("x") === "73x20x30", "contour read includes lower normal halo and storage offset");
check(interior.readWindow.extent.join("x") === "66x65x33", "contour read includes clamped upper halo");
check(interior.ownedCellOrigin.join("x") === "64x0x0", "cell ownership begins at core origin");
check(interior.ownedCellExtent.join("x") === "64x64x32", "interior cells have one deterministic owner");
check(contour.sourceBytes === contour.items.reduce((sum, item) => sum + (item.readWindow.maxSourceBytes ?? 0), 0), "contour source accounting is exact");
const terrainContour = planContourBricks(valid.hierarchy, 1, { terrainFollowing: true });
check(terrainContour.items.every((item) =>
  item.readWindow.maxOutputBytes === (() => {
    const cells = item.readWindow.extent.reduce((count, extent) => count * extent, 1);
    return cells * 36 + (cells % 2 === 1 ? 4 : 0);
  })()), "terrain brick output admits scalar, centroid, volume, and alignment payloads");
check(terrainContour.sourceBytes > contour.sourceBytes,
  "terrain brick source accounting includes geometry and aggregation working memory");
throws(() => planContourBricks(valid.hierarchy, 1, { maximumSourceBytes: 1 }), "requires", "contour admission precedes reads");
throws(() => planContourBricks(valid.hierarchy, Number.NaN), "finite", "invalid contour threshold fails");
const oppositeContour = planContourBricks(valid.hierarchy, -1);
const signedUnion = unionContourBrickPlans(contour, oppositeContour);
check(signedUnion.items.length === contour.items.length, "signed union loads overlapping bricks once");
check(signedUnion.sourceBytes === contour.sourceBytes, "signed union accounts shared source once");
check(signedUnion.items.map((item) => item.brickId).join(",")
  === [...signedUnion.items].map((item) => item.brickId).sort().join(","), "signed union order is deterministic");
const companion = { ...valid.hierarchy, field: "v" };
assertCompatibleBrickHierarchies([valid.hierarchy, companion]);
check(planAllContourBricks(valid.hierarchy).items.length === 12, "derived planning retains every diagnostic input brick");
const incompatible = structuredClone(companion);
incompatible.levels[0].bricks[0].origin[0] = 1;
throws(() => assertCompatibleBrickHierarchies([valid.hierarchy, incompatible]), "ownership geometry", "multi-input brick ownership fails closed");

check(!scalarIntervalHasOpacity(0, 0.1, [[0, 0], [0.2, 0], [1, 1]]), "transparent interval is culled");
check(scalarIntervalHasOpacity(0, 0.5, [[0, 0], [0.2, 0], [1, 1]]), "interpolated opacity interval is retained");
check(scalarIntervalHasOpacity(2, 3, [[0, 0], [1, 1]]), "upper clamped opacity remains visible");
check(!scalarIntervalHasOpacity(-3, -2, [[0, 0], [1, 1]]), "lower clamped transparency is culled");

const volume = planVolumeBricks(valid.hierarchy, [[-2, 0], [0.5, 0], [1, 0.8], [3, 1]], {
  focusIndex: [70, 20, 10],
  maximumResidentBytes: 64 * 64 * 32 * 4,
});
check(volume.transparentBrickCount === 10, "volume skips transfer-transparent bricks");
check(volume.items.length === 1, "volume residency budget admits a deterministic subset");
check(volume.deferredBrickCount === 1, "volume reports visible deferred bricks");
check(volume.items[0].brickId === "l0:1:0:0", "focus-nearest volume brick is first");
check(volume.residentBytes === volume.items.reduce((sum, item) => sum + (item.window.maxOutputBytes ?? 0), 0), "volume residency accounting is exact");
const nestedVolume = planVolumeBricks(valid.hierarchy, [[-2, 1], [3, 1]], {
  maximumResidentBytes: 4 * 1024 * 1024,
  excludePhysicalBoundsM: { lower: [6400, 0, 0], upper: [12800, 6400, 3200] },
});
check(nestedVolume.excludedBrickCount === 1
  && !nestedVolume.items.some((item) => item.brickId === "l0:1:0:0"),
"nested child ownership skips fully covered parent bricks before native reads");
throws(() => planVolumeBricks(valid.hierarchy, [], { maximumResidentBytes: 0 }), "budget", "invalid volume budget fails");

const middleNeighbors = adjacentBrickIds(valid.hierarchy.levels[0], "l0:1:0:0");
check(middleNeighbors.join(",") === "l0:0:0:0,l0:1:0:1,l0:1:1:0,l0:2:0:0", "six-face adjacency is deterministic and bounded");
const cornerNeighbors = adjacentBrickIds(valid.hierarchy.levels[0], "l0:0:0:0");
check(cornerNeighbors.join(",") === "l0:0:0:1,l0:0:1:0,l0:1:0:0", "corner has three neighbors");
throws(() => adjacentBrickIds(valid.hierarchy.levels[0], "missing"), "unavailable", "unknown adjacency source fails");

const unsafe = { ...valid.hierarchy, extremaSemantics: "pre_storage_source" as const, cullingSafe: false };
throws(() => planContourBricks(unsafe, 1), "safe decoded-value bounds", "lossy source extrema cannot cull contours");
throws(() => planVolumeBricks(unsafe, [], { maximumResidentBytes: 1 }), "safe decoded-value bounds", "lossy source extrema cannot skip volume bricks");
const decodedBounds = {
  ...valid.hierarchy,
  extremaSemantics: "decoded_value_bounds" as const,
  extremaAbsolutePadding: 0.01,
};
check(planContourBricks(decodedBounds, 1).items.length === 2, "decoded-value bounds safely cull compressed bricks");

console.log(`\nfield brick hierarchy tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
