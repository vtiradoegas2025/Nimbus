import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

/*
 * Layout contract for the Configure form's numeric fields.
 *
 * `.schema-form__unit` is absolutely positioned inside the field, and the
 * native spin buttons render at the input's content-box right edge — so the
 * label's position and `padding-right` are coupled. At `padding-right: 36px`
 * the widest unit in the schema ("kg/kg") was painted straight through the
 * spin arrows and over the last digits of the value.
 *
 * The stylesheet is applied to synthetic markup rather than driving the real
 * Configure page, because these are sub-pixel geometry assertions about a
 * single control and the units in play are fixed by the schema.
 */

const SRC = fileURLToPath(new URL("../../../../ui/src/", import.meta.url));

const STYLES = [
  "styles/theme.css",
  "styles/base.css",
  "components/config/SchemaForm.css",
].map((file) => readFileSync(join(SRC, file), "utf-8")).join("\n");

// The widest unit strings the schema actually produces.
const WIDEST_UNITS = ["kg/kg", "m²/s", "J/kg"];

const markup = `<!doctype html><html><head><style>${STYLES}</style></head><body>
<div class="schema-form" style="width:320px">
${WIDEST_UNITS.map((unit, index) => `
  <div class="schema-form__control">
    <div class="schema-form__number" data-unit="${unit}">
      <input id="field-${index}" type="number" value="0.0125" step="any">
      <span class="schema-form__unit">${unit}</span>
    </div>
  </div>`).join("")}
</div></body></html>`;

test("unit labels stay clear of the value and the spin buttons", async ({ page }) => {
  await page.setContent(markup);

  const geometry = await page.evaluate(() => {
    return [...document.querySelectorAll<HTMLElement>(".schema-form__number")].map((wrap) => {
      const input = wrap.querySelector("input")!;
      const unit = wrap.querySelector<HTMLElement>(".schema-form__unit")!;
      const inputBox = input.getBoundingClientRect();
      const unitBox = unit.getBoundingClientRect();
      const reserve = parseFloat(getComputedStyle(input).paddingRight);
      return {
        unit: wrap.dataset.unit,
        reserve,
        // Insets measured leftward from the input's right border.
        labelNearInset: inputBox.right - unitBox.right,
        labelFarInset: inputBox.right - unitBox.left,
      };
    });
  });

  // Guards the harness: if the stylesheet failed to load the label would not
  // be positioned and every inset below would collapse to a passing 0.
  expect(geometry[0].reserve).toBeGreaterThan(0);

  for (const field of geometry) {
    // The label must sit entirely within the reserved gutter, never over the
    // value's text area (which ends at `padding-right`).
    expect(field.labelFarInset, `${field.unit} label intrudes into the value`)
      .toBeLessThan(field.reserve);
  }
});

test("the spin buttons do not sit under the unit label", async ({ page }) => {
  await page.setContent(markup);

  const input = page.locator("#field-0"); // kg/kg — the widest label
  const box = (await input.boundingBox())!;

  // Walk inward from the right border and record where spin-up responds.
  const responsiveInsets: number[] = [];
  for (let inset = 3; inset <= 90; inset += 3) {
    await input.fill("10");
    await page.mouse.click(box.x + box.width - inset, box.y + box.height * 0.3);
    if ((await input.inputValue()) !== "10") responsiveInsets.push(inset);
  }
  expect(responsiveInsets.length, "no spin button found — probe is not measuring anything")
    .toBeGreaterThan(0);

  const labelFarInset = await page.evaluate(() => {
    const wrap = document.querySelector(".schema-form__number")!;
    const inputBox = wrap.querySelector("input")!.getBoundingClientRect();
    const unitBox = wrap.querySelector(".schema-form__unit")!.getBoundingClientRect();
    return inputBox.right - unitBox.left;
  });

  // Every responsive point must be further left than the label's left edge.
  expect(Math.min(...responsiveInsets)).toBeGreaterThan(labelFarInset);
});
