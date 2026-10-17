export function parseContourLevels(input: string): { levels: number[]; error: string | null } {
  const text = input.trim();
  if (!text) return { levels: [], error: null };
  const range = text.includes(":");
  const tokens = text.split(range ? ":" : ",").map(value => value.trim());
  const values = tokens.map(Number);
  if (tokens.some(value => !value) || values.some(value => !Number.isFinite(value))
      || (range && tokens.length !== 3)) {
    return { levels: [], error: "Enter finite values separated by commas, or start:end:step." };
  }
  if (!range) {
    const levels = [...new Set(values)];
    return levels.length <= 12 ? { levels, error: null }
      : { levels: [], error: "Use at most 12 contour levels." };
  }
  const [start, end, step] = values;
  const distance = (end - start) / step;
  if (step === 0 || !Number.isFinite(distance) || distance < 0) {
    return { levels: [], error: "The step must be nonzero and move from start toward end." };
  }
  const count = Math.floor(distance + Number.EPSILON * Math.max(1, distance) * 4) + 1;
  if (count > 12) return { levels: [], error: "Use at most 12 contour levels." };
  const levels = Array.from({ length: count }, (_, index) => start + index * step);
  if (levels.some(value => !Number.isFinite(value)) || new Set(levels).size !== levels.length) {
    return { levels: [], error: "The step is too small to distinguish these contour levels." };
  }
  return { levels, error: null };
}
