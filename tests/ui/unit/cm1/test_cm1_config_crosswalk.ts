import assert from "node:assert/strict";

import {
  CM1_FIELD_LANDMARKS,
  CM1_SECTION_LANDMARKS,
  cm1FieldLandmark,
  cm1RelationshipLabel,
  cm1SectionLandmark,
} from "../../../../ui/src/lib/cm1/cm1ConfigCrosswalk.ts";
import { qualifyKey, SCHEMA_SECTIONS } from "../../../../ui/src/lib/config/configSchema.ts";

let checks = 0;
function check(condition: unknown, message: string): asserts condition {
  checks += 1;
  assert.ok(condition, message);
}

const sectionIds = new Set(SCHEMA_SECTIONS.map((section) => section.id));
const fieldPaths = new Set(
  SCHEMA_SECTIONS.flatMap((section) => section.fields.map((field) => qualifyKey(section, field))),
);

check(CM1_SECTION_LANDMARKS.length === 14, "expected bounded set of CM1-oriented sections");
check(CM1_FIELD_LANDMARKS.length === 36, "expected maintained CM1 field landmark count");
check(new Set(CM1_SECTION_LANDMARKS.map((item) => item.sectionId)).size === CM1_SECTION_LANDMARKS.length, "section landmarks are unique");
check(new Set(CM1_FIELD_LANDMARKS.map((item) => item.nimbusPath)).size === CM1_FIELD_LANDMARKS.length, "field landmarks are unique");

for (const item of CM1_SECTION_LANDMARKS) {
  check(sectionIds.has(item.sectionId), `section ${item.sectionId} exists in the live schema`);
  check(item.cm1Keys.length > 0, `section ${item.sectionId} exposes familiar CM1 names`);
  check(item.note.length >= 24, `section ${item.sectionId} carries a difference boundary`);
  check(cm1SectionLandmark(item.sectionId) === item, `section lookup is stable for ${item.sectionId}`);
}

for (const item of CM1_FIELD_LANDMARKS) {
  check(fieldPaths.has(item.nimbusPath), `field ${item.nimbusPath} exists in the live schema`);
  check(item.cm1Keys.length > 0, `field ${item.nimbusPath} exposes a CM1 control`);
  check(item.note.length >= 24, `field ${item.nimbusPath} discloses the mapping boundary`);
  check(cm1FieldLandmark(item.nimbusPath) === item, `field lookup is stable for ${item.nimbusPath}`);
  check(cm1RelationshipLabel(item.relationship).length > 0, `field ${item.nimbusPath} has an accessible relationship label`);
}

const direct = CM1_FIELD_LANDMARKS.filter((item) => item.relationship === "same_scalar");
check(direct.length === 10, "same-scalar label remains narrowly scoped");
check(direct.every((item) => /^(grid|time)\./.test(item.nimbusPath)), "only grid/time scalars claim direct transfer");
check(cm1FieldLandmark("microphysics.scheme")?.relationship === "different", "ptype never claims scheme equivalence");
check(cm1FieldLandmark("extensions.checkpoint.every_s")?.note.includes("never a CM1 restart"), "checkpoint boundary is explicit");
check(cm1FieldLandmark("radiation.scheme") === null, "unmaintained CM1 concepts are not invented");

console.log(`cm1 config crosswalk: ${checks} checks passed`);
