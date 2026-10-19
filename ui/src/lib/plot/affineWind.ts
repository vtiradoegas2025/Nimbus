export interface AffineWind {
  centerM: readonly [number, number, number];
  velocityMs: readonly [number, number, number];
  gradientPerS: readonly number[];
}
export type AffineMatrix = number[];
export function identityAffine(): AffineMatrix { return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; }
export function multiplyAffine(a: readonly number[], b: readonly number[]): AffineMatrix {
  return Array.from({ length: 16 }, (_, index) => {
    const row = index % 4, col = Math.floor(index / 4);
    let value = 0;
    for (let k = 0; k < 4; k++) value += a[k * 4 + row] * b[col * 4 + k];
    return value;
  });
}
export function transformPoint(matrix: readonly number[], point: readonly number[]): [number, number, number] {
  return [0, 1, 2].map(row => matrix[12 + row] + matrix[row] * point[0] + matrix[4 + row] * point[1] + matrix[8 + row] * point[2]) as [number, number, number];
}
export function inverse3(matrix: readonly number[]): number[] | null {
  const [a,b,c,d,e,f,g,h,i] = matrix;
  const determinant = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-14) return null;
  return [e*i-f*h,c*h-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*h-e*g,b*g-a*h,a*e-b*d].map(value => value / determinant);
}
export function inverseAffine(matrix: readonly number[]): AffineMatrix {
  if (matrix.length !== 16 || !matrix.every(Number.isFinite) || matrix[3] !== 0 || matrix[7] !== 0 || matrix[11] !== 0 || matrix[15] !== 1) throw new Error("Invalid affine transform");
  const inverse = inverse3([matrix[0],matrix[4],matrix[8],matrix[1],matrix[5],matrix[9],matrix[2],matrix[6],matrix[10]]);
  if (!inverse) throw new Error("Singular affine transform");
  const result = identityAffine();
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) result[col * 4 + row] = inverse[row * 3 + col];
    result[12 + row] = -inverse[row * 3] * matrix[12] - inverse[row * 3 + 1] * matrix[13] - inverse[row * 3 + 2] * matrix[14];
  }
  return result;
}
export function affineWindTransform(wind: AffineWind, timeS: number, scale = { xy: 1, z: 1 }): AffineMatrix {
  if (!Number.isFinite(timeS) || wind.gradientPerS.length !== 9 || wind.centerM.length !== 3 || wind.velocityMs.length !== 3
      || ![...wind.gradientPerS, ...wind.centerM, ...wind.velocityMs].every(Number.isFinite)
      || ![scale.xy, scale.z].every(value => Number.isFinite(value) && value > 0)) throw new Error("Invalid affine wind motion");
  const rate = Math.max(...[0, 1, 2].map(row => wind.gradientPerS.slice(row * 3, row * 3 + 3).reduce((sum, value) => sum + Math.abs(value), 0))) * Math.abs(timeS);
  if (rate > 1 || !Number.isFinite(rate)) throw new Error("Wind deformation exceeds the preview range");
  const halves = Math.max(0, Math.ceil(Math.log2(Math.max(1, rate * 4))));
  const dt = timeS / 2 ** halves;
  const generator = new Array<number>(16).fill(0);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) generator[col * 4 + row] = wind.gradientPerS[row * 3 + col] * dt;
    generator[12 + row] = wind.velocityMs[row] * dt;
  }
  let result = identityAffine(), term = identityAffine();
  for (let order = 1; order <= 14; order++) {
    term = multiplyAffine(term, generator).map(value => value / order);
    result = result.map((value, index) => value + term[index]);
  }
  for (let power = 0; power < halves; power++) result = multiplyAffine(result, result);
  for (let row = 0; row < 3; row++) result[12 + row] += wind.centerM[row]
    - result[row] * wind.centerM[0] - result[4 + row] * wind.centerM[1] - result[8 + row] * wind.centerM[2];
  const factors = [scale.xy, scale.xy, scale.z];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) result[col * 4 + row] *= factors[row] / factors[col];
    result[12 + row] *= factors[row];
  }
  if (!result.every(Number.isFinite)) throw new Error("Affine wind transform overflow");
  return result;
}
export function averageAffineWind(left: AffineWind, right: AffineWind): AffineWind {
  const centerM = left.centerM.map((value, axis) => (value + right.centerM[axis]) * 0.5) as [number, number, number];
  const velocityAt = (wind: AffineWind, row: number) => wind.velocityMs[row]
    + [0, 1, 2].reduce((sum, col) => sum + wind.gradientPerS[row * 3 + col] * (centerM[col] - wind.centerM[col]), 0);
  return { centerM, velocityMs: [0, 1, 2].map(row => (velocityAt(left, row) + velocityAt(right, row)) * 0.5) as [number, number, number],
    gradientPerS: left.gradientPerS.map((value, index) => (value + right.gradientPerS[index]) * 0.5) };
}
