export const ZOOM_VOLUME_CELL_LIMIT = 64_000_000;

export function volumeZoomCellBudget(input: {
  baseCells: number;
  sourceCells: number | null;
  systemRamBytes: number | null;
  nativeSpacingM: number;
  rangeM: number | null;
  viewportHeightPx: number;
  paneScale: number;
  active: boolean;
}): number {
  if (!Number.isFinite(input.systemRamBytes) || !Number.isFinite(input.paneScale) || input.paneScale <= 0
      || !Number.isSafeInteger(input.sourceCells) || !Number.isFinite(input.viewportHeightPx)
      || !input.systemRamBytes || input.systemRamBytes < 16 * 1024 ** 3
      || !input.sourceCells || input.sourceCells <= input.baseCells
      || !Number.isFinite(input.rangeM) || !(input.rangeM! > 0)
      || !(input.nativeSpacingM > 0) || !Number.isFinite(input.nativeSpacingM)
      || !(input.viewportHeightPx > 0)) return input.baseCells;
  const ceiling = Math.floor(ZOOM_VOLUME_CELL_LIMIT * Math.min(1, Math.max(0, input.paneScale)));
  const nativeCells = Math.min(input.sourceCells, ceiling);
  const projectedCellPixels = input.nativeSpacingM * input.viewportHeightPx
    / (2 * Math.tan(Math.PI / 12) * input.rangeM!);
  return projectedCellPixels > (input.active ? 1.5 : 2.5)
    ? Math.max(input.baseCells, nativeCells) : input.baseCells;
}
