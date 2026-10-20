import type { Page } from "@playwright/test";
import { CLIENT_OPS } from "@nimbus/lib/platform/pyodideProtocol";

interface PythonStreamGate {
  arm(): void;
  release(): void;
  state(): { armed: boolean; held: boolean; requests: number };
}

declare global {
  interface Window { __nimbusPythonStreamGate: PythonStreamGate }
}

export async function installPythonStreamGate(page: Page): Promise<void> {
  await page.addInitScript(({ loadField }) => {
    let armed = false;
    let requests = 0;
    let releaseResponse: (() => void) | null = null;
    const OriginalWorker = window.Worker;
    window.__nimbusPythonStreamGate = {
      arm() {
        if (armed || releaseResponse) throw new Error("Python streaming gate is already active");
        armed = true;
        requests = 0;
      },
      release() {
        armed = false;
        const release = releaseResponse;
        releaseResponse = null;
        release?.();
      },
      state: () => ({ armed, held: releaseResponse !== null, requests }),
    };
    window.Worker = new Proxy(OriginalWorker, {
      construct(target, args) {
        const worker = Reflect.construct(target, args) as Worker;
        if (!String(args[0]).includes("pyodideWorker")) return worker;
        const heldRequests = new Set<number>();
        worker.addEventListener("message", ({ data }: MessageEvent<{ kind?: string; op?: string; id?: number; args?: unknown[] }>) => {
          if (armed && data.kind === "request" && data.op === loadField && data.args?.[0] === "w" && data.id != null) {
            requests++;
            heldRequests.add(data.id);
          }
        });
        const post = worker.postMessage.bind(worker);
        worker.postMessage = ((message: { kind?: string; id?: number }, options?: unknown) => {
          const deliver = () => Reflect.apply(post, worker, options === undefined ? [message] : [message, options]);
          if (armed && message.kind === "response" && message.id != null && heldRequests.delete(message.id)) {
            if (releaseResponse) throw new Error("Python streaming gate received multiple field responses");
            // Hold only the real field response; worker stdout/status events stay live.
            releaseResponse = deliver;
          } else deliver();
        }) as typeof worker.postMessage;
        return worker;
      },
    });
  }, { loadField: CLIENT_OPS.loadFrameField });
}
