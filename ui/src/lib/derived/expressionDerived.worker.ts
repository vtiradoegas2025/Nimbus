import {
  evaluateExpressionWorkerRequest,
  type ExpressionDerivedWorkerReply,
  type ExpressionDerivedWorkerRequest,
} from "./expressionDerivedWorkerProtocol";

const workerScope = self as unknown as {
  onmessage: ((event: { data: ExpressionDerivedWorkerRequest }) => void) | null;
  postMessage: (message: ExpressionDerivedWorkerReply, transfer: ArrayBuffer[]) => void;
};

workerScope.onmessage = (event) => {
  try {
    const result = evaluateExpressionWorkerRequest(event.data);
    workerScope.postMessage({ ok: true, result }, [result.data.buffer as ArrayBuffer]);
  } catch (error) {
    workerScope.postMessage({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

export {};
