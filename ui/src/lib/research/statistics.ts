export interface HistogramSpec {
    min: number;
    max: number;
    bins: number;
}
export interface PhysicalBox {
    lower: [
        number,
        number,
        number
    ];
    upper: [
        number,
        number,
        number
    ];
}
export interface ScanSpec {
    value: HistogramSpec;
    height: HistogramSpec;
    joint?: HistogramSpec;
    threshold: number;
    below: boolean;
    maskThreshold?: number;
    reflectivity: boolean;
    region?: PhysicalBox;
    footprint?: {
        x: HistogramSpec;
        y: HistogramSpec;
    };
    massFactor?: number;
    axesM: [
        number[],
        number[],
        number[]
    ];
    cellAreaM2: number | null;
    cellVolumeM3: number | null;
}
export interface ScanChunk {
    origin: number[];
    shape: number[];
    values: Float32Array;
    other?: Float32Array;
    mask?: Float32Array;
    density?: Float32Array;
    centroids?: {
        x: Float64Array;
        y: Float64Array;
        z: Float64Array;
    } | null;
    volumes?: Float64Array | null;
}
export interface ScanResult {
    finite: number;
    missing: number;
    masked: number;
    min: number | null;
    max: number | null;
    mean: number | null;
    selected: number;
    selectedMean: number | null;
    areaKm2: number | null;
    volumeKm3: number | null;
    massKg: number | null;
    valueCounts: number[];
    heightCounts: number[];
    heightMeans: (number | null)[];
    cfad: number[][];
    joint: number[][];
    belowRange: number;
    aboveRange: number;
    paired: number;
    footprint: number[][];
}
export function validateBins(spec: HistogramSpec, minimum = 2): void {
    if (!Number.isFinite(spec.min) || !Number.isFinite(spec.max) || spec.max <= spec.min
        || !Number.isSafeInteger(spec.bins) || spec.bins < minimum || spec.bins > 128)
        throw new Error("Bins require finite increasing bounds and 2–128 intervals");
}
export function binIndex(value: number, spec: HistogramSpec): number {
    if (!Number.isFinite(value) || value < spec.min || value > spec.max)
        return -1;
    return Math.min(spec.bins - 1, Math.floor((value - spec.min) / (spec.max - spec.min) * spec.bins));
}
export function binCenters(spec: HistogramSpec): number[] { return Array.from({ length: spec.bins }, (_, i) => spec.min + (i + .5) * (spec.max - spec.min) / spec.bins); }
export class ResearchAccumulator {
    private spec: ScanSpec;
    private counts: Float64Array;
    private heights: Float64Array;
    private sums: Float64Array;
    private cfad: Float64Array;
    private joint: Float64Array;
    private finite = 0;
    private missing = 0;
    private masked = 0;
    private min = Infinity;
    private max = -Infinity;
    private sum = 0;
    private selected = 0;
    private selectedSum = 0;
    private volume = 0;
    private volumeKnown = true;
    private mass = 0;
    private massKnown = true;
    private footprintCounts: Float64Array;
    private footprintSelected: Float64Array;
    private belowRange = 0;
    private aboveRange = 0;
    private paired = 0;
    constructor(spec: ScanSpec) {
        validateBins(spec.value);
        validateBins(spec.height);
        if (spec.joint)
            validateBins(spec.joint);
        if (!Number.isFinite(spec.threshold) || (spec.maskThreshold != null && !Number.isFinite(spec.maskThreshold)))
            throw new Error("Thresholds must be finite");
        if (spec.axesM.some(axis => !axis.length || axis.some((v, i) => !Number.isFinite(v) || i > 0 && v <= axis[i - 1])))
            throw new Error("Physical axes must be finite and increasing");
        if (spec.region && spec.region.lower.some((v, i) => !Number.isFinite(v) || !Number.isFinite(spec.region!.upper[i]) || v > spec.region!.upper[i]))
            throw new Error("Physical region bounds are invalid");
        for (const value of [spec.cellAreaM2, spec.cellVolumeM3])
            if (value != null && (!Number.isFinite(value) || value <= 0))
                throw new Error("Physical cell measures must be positive");
        if (spec.footprint) {
            validateBins(spec.footprint.x, 1);
            validateBins(spec.footprint.y, 1);
        }
        this.footprintCounts = new Float64Array(spec.footprint ? spec.footprint.x.bins * spec.footprint.y.bins : 0);
        this.footprintSelected = new Float64Array(this.footprintCounts.length);
        this.spec = spec;
        this.counts = new Float64Array(spec.value.bins);
        this.heights = new Float64Array(spec.height.bins);
        this.sums = new Float64Array(spec.height.bins);
        this.cfad = new Float64Array(spec.value.bins * spec.height.bins);
        this.joint = new Float64Array(spec.value.bins * (spec.joint?.bins ?? 0));
    }
    append(chunk: ScanChunk): void {
        const { spec } = this;
        if (chunk.shape.length !== 3 || chunk.origin.length !== 3 || chunk.shape.some((n, i) => !Number.isSafeInteger(n) || n < 1 || !Number.isSafeInteger(chunk.origin[i]) || chunk.origin[i] < 0 || chunk.origin[i] + n > spec.axesM[i].length))
            throw new Error("Research tile geometry is invalid");
        const length = chunk.shape.reduce((a, b) => a * b, 1);
        for (const array of [chunk.values, chunk.other, chunk.mask, chunk.density, chunk.centroids?.x, chunk.centroids?.y, chunk.centroids?.z, chunk.volumes])
            if (array && array.length !== length)
                throw new Error("Research tile fields are not co-located");
        if (spec.maskThreshold != null && !chunk.mask)
            throw new Error("Required mask field is missing");
        if (spec.joint && !chunk.other)
            throw new Error("Required joint-distribution field is missing");
        const [, ny, nz] = chunk.shape;
        for (let n = 0; n < length; n++) {
            const i = chunk.origin[0] + Math.floor(n / (ny * nz)), j = chunk.origin[1] + Math.floor(n / nz) % ny, k = chunk.origin[2] + n % nz;
            const x = chunk.centroids?.x[n] ?? spec.axesM[0][i], y = chunk.centroids?.y[n] ?? spec.axesM[1][j], z = chunk.centroids?.z[n] ?? spec.axesM[2][k];
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z))
                throw new Error("Physical sample coordinates are missing");
            if (spec.region && (x < spec.region.lower[0] || x > spec.region.upper[0] || y < spec.region.lower[1] || y > spec.region.upper[1] || z < spec.region.lower[2] || z > spec.region.upper[2]))
                continue;
            if (chunk.mask && (!Number.isFinite(chunk.mask[n]) || chunk.mask[n] < spec.maskThreshold!)) {
                this.masked++;
                continue;
            }
            const value = chunk.values[n];
            if (!Number.isFinite(value)) {
                this.missing++;
                continue;
            }
            this.finite++;
            this.min = Math.min(this.min, value);
            this.max = Math.max(this.max, value);
            const linear = spec.reflectivity ? 10 ** (value / 10) : value;
            if (!Number.isFinite(linear))
                throw new Error("Field values exceed the numerical range of this analysis");
            this.sum += linear;
            const b = binIndex(value, spec.value), h = binIndex(z / 1000, spec.height);
            if (b >= 0)
                this.counts[b]++;
            else if (value < spec.value.min)
                this.belowRange++;
            else
                this.aboveRange++;
            if (h >= 0) {
                this.heights[h]++;
                this.sums[h] += linear;
                if (b >= 0)
                    this.cfad[h * spec.value.bins + b]++;
            }
            if (chunk.other && spec.joint && Number.isFinite(chunk.other[n])) {
                this.paired++;
                const c = binIndex(chunk.other[n], spec.joint);
                if (b >= 0 && c >= 0)
                    this.joint[c * spec.value.bins + b]++;
            }
            const selected = spec.below ? value <= spec.threshold : value >= spec.threshold;
            if (spec.footprint) {
                const fx = binIndex(x / 1000, spec.footprint.x), fy = binIndex(y / 1000, spec.footprint.y);
                if (fx >= 0 && fy >= 0) {
                    const pixel = fy * spec.footprint.x.bins + fx;
                    this.footprintCounts[pixel]++;
                    if (selected)
                        this.footprintSelected[pixel]++;
                }
            }
            if (selected) {
                this.selected++;
                this.selectedSum += linear;
                const volume = chunk.volumes?.[n] ?? spec.cellVolumeM3;
                if (volume == null || !Number.isFinite(volume) || volume <= 0)
                    this.volumeKnown = false;
                else
                    this.volume += volume;
            }
            const volume = chunk.volumes?.[n] ?? spec.cellVolumeM3;
            if (!chunk.density || !Number.isFinite(chunk.density[n]) || chunk.density[n] <= 0 || volume == null || !(volume > 0) || value < 0)
                this.massKnown = false;
            else
                this.mass += value * (spec.massFactor ?? 1) * chunk.density[n] * volume;
        }
    }
    finish(): ScanResult {
        const mean = (sum: number, n: number) => n ? this.spec.reflectivity ? 10 * Math.log10(sum / n) : sum / n : null;
        return { finite: this.finite, missing: this.missing, masked: this.masked, min: this.finite ? this.min : null, max: this.finite ? this.max : null,
            mean: mean(this.sum, this.finite), selected: this.selected, selectedMean: mean(this.selectedSum, this.selected),
            areaKm2: !this.finite || this.spec.cellAreaM2 == null ? null : this.selected * this.spec.cellAreaM2 / 1e6,
            volumeKm3: this.finite > 0 && this.volumeKnown && (this.spec.cellVolumeM3 != null || this.selected > 0) ? this.volume / 1e9 : null,
            massKg: this.massKnown && this.finite ? this.mass : null,
            valueCounts: Array.from(this.counts), heightCounts: Array.from(this.heights), heightMeans: Array.from(this.heights, (n, i) => mean(this.sums[i], n)),
            cfad: Array.from(this.heights, (_, h) => Array.from(this.counts, (_, b) => this.heights[h] ? this.cfad[h * this.spec.value.bins + b] / this.heights[h] * 100 : NaN)),
            joint: Array.from({ length: this.spec.joint?.bins ?? 0 }, (_, c) => Array.from(this.counts, (_, b) => this.joint[c * this.spec.value.bins + b])),
            footprint: Array.from({ length: this.spec.footprint?.y.bins ?? 0 }, (_, j) => Array.from({ length: this.spec.footprint?.x.bins ?? 0 }, (_, i) => { const n = j * this.spec.footprint!.x.bins + i; return this.footprintCounts[n] ? 100 * this.footprintSelected[n] / this.footprintCounts[n] : NaN; })),
            belowRange: this.belowRange, aboveRange: this.aboveRange, paired: this.paired };
    }
}
