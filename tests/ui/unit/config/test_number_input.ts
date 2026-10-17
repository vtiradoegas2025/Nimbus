import { clampNumberInput, validateNumberInput } from "../../../../ui/src/lib/config/numberInput.ts";

let passed = 0;
const check = (condition: boolean, label: string) => {
  if (!condition) throw new Error(label);
  passed += 1;
};

check(validateNumberInput("4", "int", 1, 10).valid, "in-range integer is valid");
check(!validateNumberInput("11", "int", 1, 10).valid, "integer above maximum is rejected");
check(!validateNumberInput("1.5", "int", 1, 10).valid, "fractional integer is rejected");
check(!validateNumberInput("-1", "float", 0, 2).valid, "float below minimum is rejected");
check(clampNumberInput(11, 1, 10) === 10, "blur clamp honors maximum");
check(clampNumberInput(-1, 0, 2) === 0, "blur clamp honors minimum");

console.log(`number input: ${passed} checks passed`);
