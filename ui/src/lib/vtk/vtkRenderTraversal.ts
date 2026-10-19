interface TraversalObserver { before(): void; after(): void }
const observers = new WeakMap<object, Set<TraversalObserver>>();

export function subscribeVtkRenderTraversal(view: object, observer: TraversalObserver): () => void {
  let entries = observers.get(view);
  if (!entries) { entries = new Set(); observers.set(view, entries); }
  entries.add(observer);
  return () => { entries!.delete(observer); if (entries!.size === 0) observers.delete(view); };
}

export function beginVtkRenderTraversal(view: object): () => void {
  const active = [...(observers.get(view) ?? [])];
  active.forEach(observer => observer.before());
  return () => active.forEach(observer => observer.after());
}
