import { useEffect, useRef, useState } from "react";
import { subscribeNativeIoBudgetLimit } from "./nativeWindowBudget";

export function useIoBudgetRetry(error: string | null): number {
  const currentError = useRef(error);
  currentError.current = error;
  const [revision, setRevision] = useState(0);
  useEffect(() => subscribeNativeIoBudgetLimit(() => {
    if (currentError.current && /budget|I\/O.*limit/i.test(currentError.current)) {
      setRevision(value => value + 1);
    }
  }), []);
  return revision;
}
