import { useCallback, useEffect, useState } from "react";
import { installVtkWebGpuSurfaceLighting } from "./vtkWebGpuSurfaceLighting";
import { installOrderedRenderer } from "./orderedRenderer";
import { installOwnedRenderWindow } from "./ownedRenderWindow";
import {
  readLocal3DSetting,
  storeLocal3DSetting,
  type ThreeDControlScope,
} from "@/lib/plot/threeDControlScope";

export type VtkBackendPreference = "auto" | "webgpu" | "webgl";
export type VtkRenderingBackend = "webgpu" | "webgl";
export type VtkRenderingWorkload = "geometry" | "volume";

/** vtk.js 36's WebGPU volume pass does not yet match its WebGL volume path for
 * scene-light materials, opacity-derived normals, volumetric scattering, or
 * close camera entry. WebGL remains fully GPU accelerated and is the qualified
 * correctness path for volume workloads. */
export const VTK_WEBGPU_VOLUME_LIMITATION =
  "Shaded volume rendering uses the feature-complete WebGL GPU path; WebGPU remains active for qualified geometry scenes.";

const STORAGE_KEY = "nimbus-studio:vtk-backend:v1";
const CHANGE_EVENT = "nimbus-studio:vtk-backend-change";
export const VTK_BACKEND_FAILURE_EVENT = "nimbus-studio:vtk-backend-failure";

let qualificationPromise: Promise<boolean> | null = null;
let sessionWebGpuFailure: string | null = null;

interface NavigatorWithGpu extends Navigator {
  gpu?: {
    requestAdapter: (options?: { powerPreference?: "high-performance" }) => Promise<object | null>;
  };
}

function normalizePreference(value: unknown): VtkBackendPreference {
  return value === "webgpu" || value === "webgl" ? value : "auto";
}

function loadSharedPreference(): VtkBackendPreference {
  if (typeof window === "undefined") return "auto";
  try {
    return normalizePreference(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return "auto";
  }
}

function loadLocalPreference(viewKey: string): VtkBackendPreference | null {
  return readLocal3DSetting(
    viewKey,
    "vtk-backend",
    (value): value is VtkBackendPreference =>
      value === "auto" || value === "webgpu" || value === "webgl",
  );
}

/** Adapter discovery selects the candidate backend. The mounted
 * vtkWebGPURenderWindow must still initialize its own device/context before
 * the UI reports it active. */
export function qualifyWebGpu(): Promise<boolean> {
  if (qualificationPromise) return qualificationPromise;
  qualificationPromise = (async () => {
    if (sessionWebGpuFailure || typeof navigator === "undefined") return false;
    const gpu = (navigator as NavigatorWithGpu).gpu;
    if (!gpu?.requestAdapter) return false;
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      return adapter != null;
    } catch {
      return false;
    }
  })();
  return qualificationPromise;
}

export function reportVtkBackendFailure(reason: string): void {
  sessionWebGpuFailure = reason || "WebGPU device unavailable";
  qualificationPromise = Promise.resolve(false);
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(VTK_BACKEND_FAILURE_EVENT, {
      detail: sessionWebGpuFailure,
    }));
  }
}

export interface VtkBackendState {
  preference: VtkBackendPreference;
  setPreference: (preference: VtkBackendPreference) => void;
  backend: VtkRenderingBackend;
  qualifying: boolean;
  fallbackReason: string | null;
  /** Confirm that vtk.js created and initialized the requested API-specific view. */
  confirmActiveApi: (className: string) => void;
  /** A capability fallback, rather than an adapter/device failure. */
  volumeSafeFallback: boolean;
}

export function useVtkBackend(
  scope: ThreeDControlScope = "shared",
  viewKey = "default",
  workload: VtkRenderingWorkload = "geometry",
): VtkBackendState {
  const [preference, setPreferenceState] = useState<VtkBackendPreference>(() =>
    scope === "independent"
      ? loadLocalPreference(viewKey) ?? loadSharedPreference()
      : loadSharedPreference()
  );
  const [backend, setBackend] = useState<VtkRenderingBackend>("webgl");
  const [qualifying, setQualifying] = useState(false);
  const [fallbackReason, setFallbackReason] = useState<string | null>(sessionWebGpuFailure);

  useEffect(() => {
    const onSharedChange = (event: Event) => {
      if (scope !== "shared" || !(event instanceof CustomEvent)) return;
      setPreferenceState(normalizePreference(event.detail));
    };
    const onFailure = (event: Event) => {
      const reason = event instanceof CustomEvent && typeof event.detail === "string"
        ? event.detail
        : "WebGPU device unavailable";
      setFallbackReason(reason);
      setBackend("webgl");
      setQualifying(false);
    };
    window.addEventListener(CHANGE_EVENT, onSharedChange);
    window.addEventListener(VTK_BACKEND_FAILURE_EVENT, onFailure);
    return () => {
      window.removeEventListener(CHANGE_EVENT, onSharedChange);
      window.removeEventListener(VTK_BACKEND_FAILURE_EVENT, onFailure);
    };
  }, [scope]);

  useEffect(() => {
    setPreferenceState(scope === "shared"
      ? loadSharedPreference()
      : loadLocalPreference(viewKey) ?? loadSharedPreference());
  }, [scope, viewKey]);

  useEffect(() => {
    let active = true;
    if (workload === "volume") {
      setBackend("webgl");
      setQualifying(false);
      setFallbackReason(
        preference === "webgl" ? null : VTK_WEBGPU_VOLUME_LIMITATION,
      );
      return () => { active = false; };
    }
    if (preference === "webgl" || sessionWebGpuFailure) {
      setBackend("webgl");
      setQualifying(false);
      setFallbackReason(sessionWebGpuFailure);
      return () => { active = false; };
    }
    setQualifying(true);
    void qualifyWebGpu().then((qualified) => {
      if (!active) return;
      if (qualified && !sessionWebGpuFailure) {
        // Adapter/device qualification is necessary but not sufficient. Keep
        // the control in "qualifying" until the component reports that the
        // vtkWebGPURenderWindow itself has initialized its device/context.
        setBackend("webgpu");
        setFallbackReason(null);
      } else {
        setQualifying(false);
        setBackend("webgl");
        setFallbackReason(
          sessionWebGpuFailure
            ?? "WebGPU is unavailable in this browser or native webview; using WebGL",
        );
      }
    });
    return () => { active = false; };
  }, [preference, workload]);

  const setPreference = useCallback((next: VtkBackendPreference) => {
    setPreferenceState(next);
    if (scope === "independent") {
      storeLocal3DSetting(viewKey, "vtk-backend", next);
      return;
    }
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // The live selection still applies; persistence is best-effort.
    }
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: next }));
  }, [scope, viewKey]);

  const confirmActiveApi = useCallback((className: string) => {
    const expected = backend === "webgpu"
      ? "vtkWebGPURenderWindow"
      : "vtkOpenGLRenderWindow";
    if (className !== expected) {
      reportVtkBackendFailure(
        `vtk.js requested ${expected} but created ${className || "an unknown renderer"}`,
      );
      return;
    }
    setQualifying(false);
    if (backend === "webgl") setFallbackReason(sessionWebGpuFailure);
  }, [backend]);

  return {
    preference,
    setPreference,
    backend,
    qualifying,
    fallbackReason,
    confirmActiveApi,
    volumeSafeFallback:
      workload === "volume" && backend === "webgl" && preference !== "webgl",
  };
}

export function vtkRenderWindowOptions(
  background: [number, number, number],
  backend: VtkRenderingBackend,
): {
  background: [number, number, number];
  listenWindowResize: false;
  defaultViewAPI: "WebGPU" | "WebGL";
} {
  if (backend === "webgpu") installVtkWebGpuSurfaceLighting();
  else { installOrderedRenderer(); installOwnedRenderWindow(); }
  return {
    background,
    listenWindowResize: false,
    defaultViewAPI: backend === "webgpu" ? "WebGPU" : "WebGL",
  };
}

export function exposeVtkBackendEvidence(
  container: HTMLElement,
  renderWindow: {
    getApiSpecificRenderWindow: () => {
      getClassName?: () => string;
      getInitialized?: () => boolean;
      onInitialized?: (callback: () => void) => { unsubscribe: () => void };
      onDeviceLost?: (callback: (info?: { reason?: string; message?: string }) => void) => {
        unsubscribe: () => void;
      };
    };
  },
  requestedBackend: VtkRenderingBackend,
  onActive?: (className: string) => void,
): () => void {
  const api = renderWindow.getApiSpecificRenderWindow();
  const activeApi = api.getClassName?.() ?? "unknown";
  const expectedApi = requestedBackend === "webgpu"
    ? "vtkWebGPURenderWindow"
    : "vtkOpenGLRenderWindow";
  container.dataset.vtkRequestedBackend = requestedBackend;
  delete container.dataset.vtkActiveApi;
  let disposed = false;
  let initializedSubscription: { unsubscribe: () => void } | undefined;
  let deviceLostSubscription: { unsubscribe: () => void } | undefined;
  let initializationTimer: ReturnType<typeof setTimeout> | undefined;
  const confirm = () => {
    if (disposed) return;
    container.dataset.vtkActiveApi = activeApi;
    onActive?.(activeApi);
    if (initializationTimer != null) clearTimeout(initializationTimer);
    initializationTimer = undefined;
  };
  if (activeApi !== expectedApi) {
    reportVtkBackendFailure(
      `vtk.js requested ${expectedApi} but created ${activeApi}`,
    );
  } else if (requestedBackend === "webgpu" && !api.getInitialized?.()) {
    initializedSubscription = api.onInitialized?.(confirm);
    deviceLostSubscription = api.onDeviceLost?.((info) => {
      if (disposed) return;
      delete container.dataset.vtkActiveApi;
      reportVtkBackendFailure(
        `WebGPU device lost (${info?.reason ?? "unknown"}): ${info?.message ?? "no detail"}`,
      );
    });
    initializationTimer = setTimeout(() => {
      if (!disposed && !api.getInitialized?.()) {
        reportVtkBackendFailure("vtk.js WebGPU renderer did not initialize within 15 seconds");
      }
    }, 15_000);
  } else {
    confirm();
  }
  return () => {
    disposed = true;
    initializedSubscription?.unsubscribe();
    deviceLostSubscription?.unsubscribe();
    if (initializationTimer != null) clearTimeout(initializationTimer);
    if (container.dataset.vtkActiveApi === activeApi) delete container.dataset.vtkActiveApi;
    if (container.dataset.vtkRequestedBackend === requestedBackend) {
      delete container.dataset.vtkRequestedBackend;
    }
  };
}
