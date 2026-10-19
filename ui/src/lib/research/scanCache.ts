import type { ScanResult } from "./statistics";
const LIMIT = 16 * 1024 * 1024;
const entries = new Map<string, {
    value: ScanResult;
    bytes: number;
}>();
let bytes = 0;
function cost(value: ScanResult, key: string): number {
    return 512 + key.length * 2 + 8 * (value.valueCounts.length + value.heightCounts.length + value.heightMeans.length +
        [...value.cfad, ...value.joint, ...value.footprint].reduce((sum, row) => sum + row.length, 0));
}
export function cachedResearchScan(key: string): ScanResult | null {
    const entry = entries.get(key);
    if (!entry)
        return null;
    entries.delete(key);
    entries.set(key, entry);
    return structuredClone(entry.value);
}
export function cacheResearchScan(key: string, value: ScanResult): void {
    const size = cost(value, key);
    if (size > LIMIT)
        return;
    const previous = entries.get(key);
    if (previous) {
        bytes -= previous.bytes;
        entries.delete(key);
    }
    while (entries.size && (bytes + size > LIMIT || entries.size >= 64)) {
        const first = entries.keys().next().value!;
        bytes -= entries.get(first)!.bytes;
        entries.delete(first);
    }
    entries.set(key, { value: structuredClone(value), bytes: size });
    bytes += size;
}
export function researchScanCacheBytes(): number { return bytes; }
export function clearResearchScanCache(): void { entries.clear(); bytes = 0; }
