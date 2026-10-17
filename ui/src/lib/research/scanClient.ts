import workerUrl from "./statistics.worker.ts?worker&url";
import type { ScanReply, ScanRequest } from "./statistics.worker";
let workerSequence = 0;
export class ResearchScanClient {
    private worker = new Worker(`${workerUrl}${workerUrl.includes("?") ? "&" : "?"}instance=${++workerSequence}`, { type: "module", name: "nimbus-research-statistics" });
    private pending: ((error: Error) => void) | null = null;
    private disposed = false;
    async request(message: ScanRequest, transfer: Transferable[] = [], signal?: AbortSignal): Promise<Extract<ScanReply, {
        ok: true;
    }>> {
        if (this.disposed || signal?.aborted)
            throw new DOMException("Analysis cancelled", "AbortError");
        if (this.pending)
            throw new Error("Research worker requires ordered tiles");
        return new Promise((resolve, reject) => {
            const cleanup = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); this.pending = null; this.worker.onmessage = this.worker.onerror = this.worker.onmessageerror = null; };
            const fail = (error: Error) => { cleanup(); this.disposed = true; this.worker.terminate(); reject(error); };
            const abort = () => fail(new DOMException("Analysis cancelled", "AbortError"));
            const timer = setTimeout(() => fail(new Error("Research tile exceeded its processing time limit")), 60000);
            this.pending = fail;
            signal?.addEventListener("abort", abort, { once: true });
            this.worker.onmessage = ({ data }: MessageEvent<ScanReply>) => { if (!data.ok) {
                fail(new Error(data.error));
                return;
            } cleanup(); resolve(data); };
            this.worker.onerror = e => { e.preventDefault(); fail(new Error("Research worker failed")); };
            this.worker.onmessageerror = () => fail(new Error("Research worker reply could not be decoded"));
            try {
                this.worker.postMessage(message, transfer);
            }
            catch (error) {
                fail(error instanceof Error ? error : new Error(String(error)));
            }
        });
    }
    dispose(): void { this.pending?.(new DOMException("Analysis cancelled", "AbortError")); this.disposed = true; this.worker.terminate(); }
}
