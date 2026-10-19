import workerUrl from "./parcels.worker.ts?worker&url";
import type { ParcelAnalysis, ParcelFilter } from "./parcels";
let sequence = 0;
export class ParcelClient {
    private worker = new Worker(`${workerUrl}${workerUrl.includes("?") ? "&" : "?"}instance=${++sequence}`, { type: "module", name: "nimbus-parcel-analysis" });
    private pending: ((error: Error) => void) | null = null;
    private closed = false;
    load(values: Float32Array, shape: number[], signal: AbortSignal) { return this.request({ kind: "load", values, shape }, [values.buffer], signal); }
    async analyze(filter: ParcelFilter, signal: AbortSignal): Promise<ParcelAnalysis> { const result = await this.request({ kind: "analyze", filter }, [], signal); if (!result)
        throw new Error("Parcel analysis returned no result"); return result; }
    private request(message: unknown, transfer: Transferable[], signal: AbortSignal): Promise<ParcelAnalysis | undefined> {
        if (this.closed || signal.aborted)
            return Promise.reject(new DOMException("Analysis cancelled", "AbortError"));
        if (this.pending)
            return Promise.reject(new Error("Wait for the current parcel analysis"));
        return new Promise((resolve, reject) => {
            const cleanup = () => { clearTimeout(timer); signal.removeEventListener("abort", abort); this.pending = null; this.worker.onmessage = this.worker.onerror = this.worker.onmessageerror = null; };
            const fail = (error: Error) => { cleanup(); this.closed = true; this.worker.terminate(); reject(error); };
            const abort = () => fail(new DOMException("Analysis cancelled", "AbortError"));
            const timer = setTimeout(() => fail(new Error("Parcel analysis exceeded its time limit")), 60000);
            this.pending = fail;
            signal.addEventListener("abort", abort, { once: true });
            this.worker.onmessage = ({ data }) => { if (!data.ok) {
                fail(new Error(data.error));
                return;
            } cleanup(); resolve(data.result); };
            this.worker.onerror = e => { e.preventDefault(); fail(new Error("Parcel worker failed")); };
            this.worker.onmessageerror = () => fail(new Error("Parcel reply could not be decoded"));
            try {
                this.worker.postMessage(message, transfer);
            }
            catch (error) {
                fail(error instanceof Error ? error : new Error(String(error)));
            }
        });
    }
    dispose() { this.pending?.(new DOMException("Analysis cancelled", "AbortError")); this.closed = true; this.worker.terminate(); }
}
