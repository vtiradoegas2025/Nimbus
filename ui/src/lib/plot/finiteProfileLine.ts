export function finiteProfileLine(values: ArrayLike<number>): (number | null)[] {
  return Array.from(values, value => Number.isFinite(value) ? value : null);
}

export function finiteHodographLine(u: ArrayLike<number>, v: ArrayLike<number>) {
  if (u.length !== v.length) throw new Error("Hodograph components must share one profile");
  const x: (number | null)[] = new Array(u.length);
  const y: (number | null)[] = new Array(v.length);
  let magnitude = 1;
  for (let index = 0; index < u.length; index++) {
    const supported = Number.isFinite(u[index]) && Number.isFinite(v[index]);
    x[index] = supported ? u[index] : null;
    y[index] = supported ? v[index] : null;
    if (supported) magnitude = Math.max(magnitude, Math.abs(u[index]), Math.abs(v[index]));
  }
  return { x, y, magnitude };
}
