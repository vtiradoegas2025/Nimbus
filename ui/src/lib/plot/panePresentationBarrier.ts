export interface PanePresentationSnapshot {
  token: string;
  allowed: boolean;
  errors: Readonly<Record<string, string>>;
  readyPaneIds: readonly string[];
}

export function panePresentationToken(identity: string, paneIds: readonly string[]): string {
  return JSON.stringify([identity, [...new Set(paneIds)].sort()]);
}

export class PanePresentationBarrier {
  private token = "";
  private members = new Set<string>();
  private ready = new Map<string, { owner: object; ready: boolean; error: string | null }>();
  private listeners = new Set<() => void>();
  private snapshot: PanePresentationSnapshot = { token: "", allowed: true, errors: {}, readyPaneIds: [] };

  constructor(identity: string, paneIds: readonly string[]) { this.configure(identity, paneIds); }
  getSnapshot = (): PanePresentationSnapshot => this.snapshot;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
  configure(identity: string, paneIds: readonly string[]): void {
    const token = panePresentationToken(identity, paneIds);
    if (token === this.token) return;
    const previousIdentity = this.token ? (JSON.parse(this.token) as [string, string[]])[0] : null;
    if (identity !== previousIdentity) this.ready.clear();
    this.token = token;
    this.members = new Set(paneIds);
    for (const paneId of this.ready.keys()) if (!this.members.has(paneId)) this.ready.delete(paneId);
    this.publish(false);
  }
  report(token: string, paneId: string, owner: object, ready: boolean, error: string | null): void {
    if (token !== this.token || !this.members.has(paneId)) return;
    const previous = this.ready.get(paneId);
    if (previous?.owner !== owner) return;
    if (previous?.owner === owner && previous.ready === ready && previous.error === error) return;
    this.ready.set(paneId, { owner, ready, error });
    this.publish(this.snapshot.allowed);
  }
  register(token: string, paneId: string, owner: object): void {
    if (token !== this.token || !this.members.has(paneId) || this.ready.get(paneId)?.owner === owner) return;
    this.ready.set(paneId, { owner, ready: false, error: null });
    this.publish(this.snapshot.allowed);
  }
  retire(token: string, paneId: string, owner: object): void {
    if (token !== this.token || this.ready.get(paneId)?.owner !== owner) return;
    this.ready.delete(paneId);
    this.publish(this.snapshot.allowed);
  }
  private publish(alreadyAllowed: boolean): void {
    const errors = Object.fromEntries([...this.ready].flatMap(([paneId, state]) => state.error ? [[paneId, state.error]] : []));
    const allowed = alreadyAllowed || this.members.size <= 1
      || [...this.members].every(paneId => this.ready.get(paneId)?.ready && !this.ready.get(paneId)?.error);
    this.snapshot = { token: this.token, allowed, errors,
      readyPaneIds: [...this.members].filter(paneId => this.ready.get(paneId)?.ready && !this.ready.get(paneId)?.error) };
    for (const listener of this.listeners) listener();
  }
}
