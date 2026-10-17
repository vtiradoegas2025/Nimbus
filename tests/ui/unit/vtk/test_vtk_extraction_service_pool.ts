import { VtkExtractionServicePool } from "../../../../ui/src/lib/vtk/vtkExtractionServicePool.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else { failed += 1; failures.push(`FAIL ${label}`); }
}

const pool = new VtkExtractionServicePool(3);
const owners = [{}, {}, {}, {}];
const retired: number[] = [];
owners.forEach((owner, index) => pool.park(owner, () => retired.push(index)));
check(pool.snapshot().warmServices === 3, "pool retains only the semantic warm set");
check(pool.snapshot().maximumWarmServices === 3, "pool reports its explicit warm cardinality");
check(retired.join(",") === "0", "fourth idle service retires the oldest one");
check(pool.snapshot().evictions === 1, "idle eviction is observable");
pool.activate(owners[1]!);
check(pool.snapshot().warmServices === 2, "active work leaves the idle pool without retirement");
check(retired.join(",") === "0", "activation never terminates visible work");
pool.park(owners[1]!, () => retired.push(1));
pool.remove(owners[2]!);
check(pool.snapshot().warmServices === 2, "explicit worker disposal removes its idle entry");
pool.reset();
check(pool.snapshot().warmServices === 0, "reset retires every remaining idle service");
check(new Set(retired).has(1) && new Set(retired).has(3), "reset invokes registered retirement callbacks");
check(
  (() => { try { new VtkExtractionServicePool(-1); return false; } catch { return true; } })(),
  "invalid cardinality fails closed",
);

console.log(`\nvtkExtractionServicePool tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
