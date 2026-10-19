import { createContext, useContext, useLayoutEffect, useMemo, useRef, useSyncExternalStore, type ReactNode } from "react";
import { PanePresentationBarrier, panePresentationToken, type PanePresentationSnapshot } from "@/lib/plot/panePresentationBarrier";
import { usePaneContext } from "./paneContext";

interface PresentationContextValue { barrier: PanePresentationBarrier; token: string; paneIds: readonly string[] }
const PresentationContext = createContext<PresentationContextValue | null>(null);
const SINGLE: PanePresentationSnapshot = { token: "", allowed: true, errors: {}, readyPaneIds: [] };
const singleSnapshot = () => SINGLE;
const singleSubscribe = () => () => {};

export function PanePresentationProvider({ identity, paneIds, children }: {
  identity: string; paneIds: readonly string[]; children: ReactNode;
}) {
  const current = useRef<PanePresentationBarrier>();
  if (!current.current) current.current = new PanePresentationBarrier(identity, paneIds);
  const barrier = current.current;
  const token = panePresentationToken(identity, paneIds);
  const value = useMemo(() => ({ barrier, token, paneIds }), [barrier, token, paneIds]);
  useLayoutEffect(() => { barrier.configure(identity, paneIds); }, [barrier, identity, paneIds, token]);
  return <PresentationContext.Provider value={value}>{children}</PresentationContext.Provider>;
}

export function usePanePresentationBarrier({ ready, error }: { ready: boolean; error?: string | null }) {
  const context = useContext(PresentationContext);
  const { paneId } = usePaneContext();
  const owner = useMemo(() => ({}), []);
  const snapshot = useSyncExternalStore(context?.barrier.subscribe ?? singleSubscribe,
    context?.barrier.getSnapshot ?? singleSnapshot, singleSnapshot);
  const barrier = context?.barrier;
  const token = context?.token ?? "";
  const member = context?.paneIds.includes(paneId) ?? false;
  useLayoutEffect(() => {
    if (!barrier || !member) return;
    barrier.register(token, paneId, owner);
    return () => barrier.retire(token, paneId, owner);
  }, [barrier, token, paneId, owner, member, snapshot.token]);
  useLayoutEffect(() => {
    if (!barrier || !member) return;
    barrier.report(token, paneId, owner, ready, error ?? null);
  }, [barrier, token, paneId, owner, member, ready, error, snapshot.token]);
  const allowed = !member || snapshot.token === token && snapshot.allowed;
  return { allowed, pending: !allowed };
}

export function usePanePresentationSnapshot(): PanePresentationSnapshot {
  const context = useContext(PresentationContext);
  return useSyncExternalStore(context?.barrier.subscribe ?? singleSubscribe,
    context?.barrier.getSnapshot ?? singleSnapshot, singleSnapshot);
}

export function usePanePresentationSynchronized(): boolean {
  const context = useContext(PresentationContext);
  const { paneId } = usePaneContext();
  return Boolean(context && context.paneIds.length > 1 && context.paneIds.includes(paneId));
}
