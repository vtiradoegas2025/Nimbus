import { streamlineBoxSeedDimensions } from "../../../../ui/src/lib/plot/streamlineSeedDomain.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

const full = streamlineBoxSeedDimensions({
  nr: 101,
  nth: 81,
  nz: 41,
  drM: 100,
  dyM: 200,
  dzM: 250,
  scalarOriginM: [-5000, -8000, 125],
}, [17, 17, 17], [100, 200, 250], [-1000, -1000, 125]);
check(full.domainSpan.join(",") === "10000,16000,10000",
  "seed dimensions use the full logical grid instead of a resident ROI");
check(full.halfSize.join(",") === "900,1440,4000",
  "storm-box proportions remain stable across brick expansion");
check(full.centerZ === 4625, "seed height is anchored to the full scalar origin");

const stretched = streamlineBoxSeedDimensions({
  scalarCoordinatesM: [[0, 100, 350], [-40, 0, 60], [50, 80, 140]],
}, [2, 2, 2], [1, 1, 1], [0, 0, 0]);
check(stretched.domainSpan.join(",") === "350,100,90",
  "declared scalar coordinates own stretched-grid seed spans");
check(Math.abs(stretched.centerZ - 90.5) < 1e-12,
  "stretched-grid seed height uses the declared first scalar center");

console.log(`\nstreamline seed-domain tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
