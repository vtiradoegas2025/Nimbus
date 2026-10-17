import type { BrickBoundaryContacts, Triple } from "@/lib/plot/fieldBrickHierarchy";

export function emptyBrickBoundaryContacts(): BrickBoundaryContacts {
  return { lower: [false, false, false], upper: [false, false, false] };
}

export function mergeBrickBoundaryContacts(
  left: BrickBoundaryContacts,
  right: BrickBoundaryContacts,
): BrickBoundaryContacts {
  return {
    lower: left.lower.map((value, axis) => value || right.lower[axis]) as [boolean, boolean, boolean],
    upper: left.upper.map((value, axis) => value || right.upper[axis]) as [boolean, boolean, boolean],
  };
}

export function detectStreamlineBoundaryContacts(
  points: ArrayLike<number>,
  lines: ArrayLike<number>,
  shape: Triple,
  spacing: Triple,
  origin: Triple,
): BrickBoundaryContacts {
  const contacts = emptyBrickBoundaryContacts();
  const lower = origin;
  const upper = origin.map((value, axis) =>
    value + Math.max(0, shape[axis] - 1) * spacing[axis]) as Triple;
  const tolerance = spacing.map((value) => Math.max(Math.abs(value) * 1.5, 1e-6)) as Triple;
  let cursor = 0;
  while (cursor < lines.length) {
    const count = Number(lines[cursor]);
    if (!Number.isSafeInteger(count) || count <= 0 || cursor + count >= lines.length) break;
    const endpoint = Number(lines[cursor + count]);
    if (Number.isSafeInteger(endpoint) && endpoint >= 0 && endpoint * 3 + 2 < points.length) {
      for (let axis = 0; axis < 3; axis += 1) {
        const coordinate = Number(points[endpoint * 3 + axis]);
        if (!Number.isFinite(coordinate)) continue;
        if (coordinate <= lower[axis] + tolerance[axis]) contacts.lower[axis] = true;
        if (coordinate >= upper[axis] - tolerance[axis]) contacts.upper[axis] = true;
      }
    }
    cursor += count + 1;
  }
  return contacts;
}

export function detectSeedBoundaryContacts(
  seeds: ArrayLike<number>,
  shape: Triple,
  spacing: Triple,
  origin: Triple,
): BrickBoundaryContacts {
  const contacts = emptyBrickBoundaryContacts();
  const upper = origin.map((value, axis) =>
    value + Math.max(0, shape[axis] - 1) * spacing[axis]) as Triple;
  for (let offset = 0; offset + 2 < seeds.length; offset += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      const coordinate = Number(seeds[offset + axis]);
      if (!Number.isFinite(coordinate)) continue;
      if (coordinate < origin[axis]) contacts.lower[axis] = true;
      if (coordinate > upper[axis]) contacts.upper[axis] = true;
    }
  }
  return contacts;
}
