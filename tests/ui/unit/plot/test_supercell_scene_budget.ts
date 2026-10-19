import assert from "node:assert/strict";
import { planSupercellSceneBudget, type SceneFieldRole } from "../../../../ui/src/lib/plot/supercellSceneBudget.ts";

const roles: SceneFieldRole[] = ["detail", "context", "auxiliary"];
const base = {
  baseCells: 16_000_000, paneScale: 1, systemRamBytes: 48 * 1024 ** 3,
  fields: [{ identity: "cloud", role: "detail" as const, bytesPerCell: 4 }],
  preparedChannels: { detail: 1 },
};
const cloud = planSupercellSceneBudget(base);
assert.equal(cloud.detailCells, 16_000_000);
assert.equal(cloud.estimatedPeakBytes, 512_000_000);
assert.deepEqual(planSupercellSceneBudget({ ...base, fields: [...base.fields, ...base.fields] }), cloud);
assert.equal(planSupercellSceneBudget({ ...base, fields: [...base.fields,
  { identity: "cloud", role: "context", bytesPerCell: 4 }], preparedChannels: { detail: 1, context: 1 },
}).uniqueFields, 2);
let cases = 0;
for (const ram of [null, 2, 8, 16, 48, 128].map(v => v == null ? null : v * 1024 ** 3)) {
  for (const paneScale of [1, 0.5, 0.25]) {
    for (const terrain of [false, true]) {
      for (let mask = 1; mask < 8; mask++) {
        const enabled = roles.filter((_, i) => mask & (1 << i));
        const fields = enabled.flatMap(role => ["cloud", "qr", "qg", "qh", "u", "v", "w", "rho"].map(identity => ({
          identity, role, bytesPerCell: terrain ? 36 : 4,
        })));
        const plan = planSupercellSceneBudget({ ...base, systemRamBytes: ram, paneScale,
          baseCells: Math.floor(base.baseCells * paneScale), fields,
          preparedChannels: Object.fromEntries(enabled.map(role => [role, 3])),
          scratchChannels: Object.fromEntries(enabled.map(role => [role, 2])),
        });
        assert.ok(plan.estimatedPeakBytes <= plan.planningEnvelopeBytes);
        for (const cap of [plan.detailCells, plan.contextCells, plan.auxiliaryCells]) {
          assert.ok(Number.isSafeInteger(cap) && cap >= 8 && cap <= base.baseCells * paneScale);
        }
        assert.equal(plan.uniqueFields, fields.length);
        cases++;
      }
    }
  }
}
const heavy = { ...base, fields: ["cloud", "qr", "qg", "qh", "u", "v", "w"].map(identity => ({
  identity, role: "detail" as const, bytesPerCell: 4,
})), preparedChannels: { detail: 2 }, scratchChannels: { detail: 2 }, systemRamBytes: 8 * 1024 ** 3 };
assert.ok(planSupercellSceneBudget(heavy).detailCells < planSupercellSceneBudget({ ...base, systemRamBytes: heavy.systemRamBytes }).detailCells);
assert.ok(planSupercellSceneBudget({ ...heavy, fields: heavy.fields.map(f => ({ ...f, bytesPerCell: 36 })) }).detailCells
  < planSupercellSceneBudget(heavy).detailCells);
const qualified = planSupercellSceneBudget({ ...heavy, qualifiedCapacityBytes: 200_000_000 });
assert.ok(qualified.estimatedPeakBytes <= 200_000_000);
assert.equal(qualified.planningEnvelopeBytes, 200_000_000);
for (const invalid of [0, -1, NaN, Infinity, 1.1]) {
  assert.throws(() => planSupercellSceneBudget({ ...base, paneScale: invalid }));
}
assert.throws(() => planSupercellSceneBudget({ ...base, fields: [...base.fields,
  { ...base.fields[0], bytesPerCell: 36 }],
}));
assert.throws(() => planSupercellSceneBudget({ ...base, qualifiedCapacityBytes: 10 }));
assert.throws(() => planSupercellSceneBudget({ ...base, preparedChannels: { detail: -1 } }));
console.log(`Supercell shared planning envelope: ${cases} hardware/pane/terrain/layer cases passed`);
