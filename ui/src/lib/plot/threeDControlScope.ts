import { useCallback, useEffect, useState } from "react";

export type ThreeDControlScope = "independent" | "shared";

const SCOPE_STORAGE_KEY = "nimbus-studio:3d-control-scope:v1";
const SCOPE_CHANGE_EVENT = "nimbus-studio:3d-control-scope-change";
const SHARED_SETTING_EVENT = "nimbus-studio:3d-shared-setting-change";

function readScope(): ThreeDControlScope {
  if (typeof window === "undefined") return "independent";
  try {
    return window.localStorage.getItem(SCOPE_STORAGE_KEY) === "shared"
      ? "shared"
      : "independent";
  } catch {
    return "independent";
  }
}

export interface ThreeDControlScopeState {
  scope: ThreeDControlScope;
  setScope: (scope: ThreeDControlScope) => void;
}

export function useThreeDControlScope(): ThreeDControlScopeState {
  const [scope, setScopeState] = useState<ThreeDControlScope>(readScope);
  useEffect(() => {
    const onChange = (event: Event) => {
      if (!(event instanceof CustomEvent)) return;
      setScopeState(event.detail === "shared" ? "shared" : "independent");
    };
    window.addEventListener(SCOPE_CHANGE_EVENT, onChange);
    return () => window.removeEventListener(SCOPE_CHANGE_EVENT, onChange);
  }, []);
  const setScope = useCallback((next: ThreeDControlScope) => {
    setScopeState(next);
    try {
      window.localStorage.setItem(SCOPE_STORAGE_KEY, next);
    } catch {
      // Scope still changes for this session when storage is unavailable.
    }
    window.dispatchEvent(new CustomEvent(SCOPE_CHANGE_EVENT, { detail: next }));
  }, []);
  return { scope, setScope };
}

function settingStorageKey(key: string): string {
  return `nimbus-studio:3d-shared:${key}:v1`;
}

function localSettingStorageKey(viewKey: string, key: string): string {
  return `nimbus-studio:3d-local:${viewKey}:${key}:v1`;
}

export function readShared3DSetting<T>(key: string, validate: (value: unknown) => value is T): T | null {
  if (typeof window === "undefined") return null;
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(settingStorageKey(key)) ?? "null");
    return validate(value) ? value : null;
  } catch {
    return null;
  }
}

export function publishShared3DSetting<T>(key: string, value: T): void {
  try {
    window.localStorage.setItem(settingStorageKey(key), JSON.stringify(value));
  } catch {
    // Live synchronization still works when persistence is unavailable.
  }
  window.dispatchEvent(new CustomEvent(SHARED_SETTING_EVENT, {
    detail: { key, value },
  }));
}

export function readLocal3DSetting<T>(
  viewKey: string,
  key: string,
  validate: (value: unknown) => value is T,
): T | null {
  if (typeof window === "undefined") return null;
  try {
    const value: unknown = JSON.parse(
      window.localStorage.getItem(localSettingStorageKey(viewKey, key)) ?? "null",
    );
    return validate(value) ? value : null;
  } catch {
    return null;
  }
}

export function storeLocal3DSetting<T>(viewKey: string, key: string, value: T): void {
  try {
    window.localStorage.setItem(localSettingStorageKey(viewKey, key), JSON.stringify(value));
  } catch {
    // Per-view state remains live even when it cannot be persisted.
  }
}

export function subscribeShared3DSetting<T>(
  key: string,
  validate: (value: unknown) => value is T,
  onValue: (value: T) => void,
): () => void {
  const listener = (event: Event) => {
    if (!(event instanceof CustomEvent) || event.detail?.key !== key) return;
    if (validate(event.detail.value)) onValue(event.detail.value);
  };
  window.addEventListener(SHARED_SETTING_EVENT, listener);
  return () => window.removeEventListener(SHARED_SETTING_EVENT, listener);
}
