import {
  evaluateDerivedWorkerRequest,
  type DerivedFieldWorkerReply,
  type DerivedFieldWorkerRequest,
} from "./derivedFieldWorkerProtocol";

const workerScope = self as unknown as {
  onmessage: ((event: { data: DerivedFieldWorkerRequest | { kind: "close" } }) => void) | null;
  postMessage: (message: DerivedFieldWorkerReply | { closed: true }, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

// One worker serves every slab of a field in turn, so it stays alive between messages.
workerScope.onmessage = (event) => {
  if ("kind" in event.data) {
    workerScope.postMessage({ closed: true }, []);
    workerScope.close();
    return;
  }
  try {
    const result = evaluateDerivedWorkerRequest(event.data);
    // Transferred, not cloned: the slab result is freshly allocated here and never read again.
    workerScope.postMessage({ ok: true, result }, [result.data.buffer as ArrayBuffer]);
  } catch (error) {
    workerScope.postMessage({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

export {};
