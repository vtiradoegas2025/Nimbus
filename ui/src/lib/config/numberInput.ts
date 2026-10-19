export interface NumberInputValidation {
  parsed: number | null;
  valid: boolean;
  message: string | null;
}

export function validateNumberInput(
  raw: string,
  kind: "int" | "float",
  minimum?: number,
  maximum?: number,
): NumberInputValidation {
  if (raw.trim() === "") return { parsed: null, valid: false, message: "Enter a number." };
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || (kind === "int" && !Number.isInteger(parsed))) {
    return { parsed: null, valid: false, message: kind === "int" ? "Enter a whole number." : "Enter a finite number." };
  }
  if (minimum != null && parsed < minimum) return { parsed, valid: false, message: `Minimum ${minimum}.` };
  if (maximum != null && parsed > maximum) return { parsed, valid: false, message: `Maximum ${maximum}.` };
  return { parsed, valid: true, message: null };
}

export function clampNumberInput(value: number, minimum?: number, maximum?: number): number {
  return Math.min(maximum ?? Number.POSITIVE_INFINITY, Math.max(minimum ?? Number.NEGATIVE_INFINITY, value));
}
