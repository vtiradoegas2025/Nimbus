import assert from "node:assert/strict";
import { PanePresentationBarrier, panePresentationToken } from "../../../../ui/src/lib/plot/panePresentationBarrier.ts";

function permutations(values: string[]): string[][] {
  return values.length ? values.flatMap((value, index) => permutations(values.filter((_, position) => position !== index))
    .map(rest => [value, ...rest])) : [[]];
}
for (const count of [2, 4]) {
  const panes = Array.from({ length: count }, (_, index) => `pane-${index}`);
  for (const order of permutations(panes)) {
    const barrier = new PanePresentationBarrier("frame-0", panes);
    const owners = new Map(panes.map(pane => [pane, {}]));
    const token = panePresentationToken("frame-0", panes);
    for (const pane of panes) barrier.register(token, pane, owners.get(pane)!);
    assert.equal(barrier.getSnapshot().allowed, false);
    order.forEach((pane, index) => {
      barrier.report(token, pane, owners.get(pane)!, true, null);
      assert.equal(barrier.getSnapshot().allowed, index === count - 1);
    });
    barrier.report(token, panes[0], owners.get(panes[0])!, false, null);
    assert.equal(barrier.getSnapshot().allowed, true, "same-frame controls do not reset the completed saved-time cohort");
    barrier.configure("frame-1", panes);
    const next = panePresentationToken("frame-1", panes);
    for (const pane of panes) barrier.register(next, pane, owners.get(pane)!);
    assert.equal(barrier.getSnapshot().allowed, false);
    for (const pane of panes) barrier.report(token, pane, owners.get(pane)!, true, null);
    assert.equal(barrier.getSnapshot().allowed, false, "stale ready reports cannot publish a new frame");
    for (const pane of panes.slice(1)) barrier.report(next, pane, owners.get(pane)!, true, null);
    barrier.report(next, panes[0], owners.get(panes[0])!, false, "field failed");
    assert.equal(barrier.getSnapshot().allowed, false);
    assert.equal(barrier.getSnapshot().errors[panes[0]], "field failed");
    const replacement = {};
    barrier.register(next, panes[0], replacement);
    barrier.report(next, panes[0], replacement, false, null);
    barrier.retire(next, panes[0], owners.get(panes[0])!);
    barrier.report(next, panes[0], owners.get(panes[0])!, true, null);
    assert.equal(barrier.getSnapshot().allowed, false, "retired scene owners cannot remove replacement readiness");
    barrier.report(next, panes[0], replacement, true, null);
    assert.equal(barrier.getSnapshot().allowed, true);
    assert.deepEqual(barrier.getSnapshot().errors, {});
    barrier.configure("frame-2", panes);
    const third = panePresentationToken("frame-2", panes);
    for (const pane of panes) barrier.register(third, pane, owners.get(pane)!);
    for (const pane of panes.slice(1)) barrier.report(third, pane, owners.get(pane)!, true, null);
    barrier.configure("frame-2", panes.slice(1));
    assert.equal(barrier.getSnapshot().allowed, true, "removing a delayed pane cannot deadlock the remaining cohort");
    barrier.configure("scrub-latest", panes);
    barrier.report(third, panes[0], replacement, true, null);
    assert.equal(barrier.getSnapshot().allowed, false);
    const latest = panePresentationToken("scrub-latest", panes);
    for (const pane of panes) barrier.register(latest, pane, owners.get(pane)!);
    for (const pane of panes) barrier.report(latest, pane, owners.get(pane)!, true, null);
    assert.equal(barrier.getSnapshot().allowed, true);
  }
}
assert.equal(new PanePresentationBarrier("frame", []).getSnapshot().allowed, true);
assert.equal(new PanePresentationBarrier("frame", ["only"]).getSnapshot().allowed, true);
assert.equal(panePresentationToken("frame", ["b", "a", "a"]), panePresentationToken("frame", ["a", "b"]));
console.log("All2/4-pane arrival orders preserve atomic saved-time release, stale scrub rejection, error recovery, replacement owners, membership cleanup, and same-frame controls");
