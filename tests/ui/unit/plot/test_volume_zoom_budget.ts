import assert from "node:assert/strict";
import { volumeZoomCellBudget } from "../../../../ui/src/lib/plot/volumeZoomBudget.ts";
const input = { baseCells: 16_000_000, sourceCells: 512*512*160, systemRamBytes: 48*1024**3,
  nativeSpacingM: 100, rangeM: 110_000, viewportHeightPx: 972, paneScale: 1, active: false };
assert.equal(volumeZoomCellBudget(input), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000 }), input.sourceCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000, systemRamBytes: 8*1024**3 }), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000, systemRamBytes: null }), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000, sourceCells: 100_000_000 }), 64_000_000);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 150_000, sourceCells: 100_000_000 }), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 1_000, nativeSpacingM: 20, sourceCells: 1024*1024*800 }), 64_000_000);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 1_000, nativeSpacingM: 20, sourceCells: 1024*1024*800, paneScale: 0.25, baseCells: 4_000_000 }), 16_000_000);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000, paneScale: 0.25 }), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 85_000, active: true }), input.sourceCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 150_000, active: true }), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000, systemRamBytes: Infinity }), input.baseCells);
assert.equal(volumeZoomCellBudget({ ...input, rangeM: 44_000, paneScale: NaN }), input.baseCells);
console.log("Volume zoom budget: native promotion, memory/quad bounds and hysteresis passed");
