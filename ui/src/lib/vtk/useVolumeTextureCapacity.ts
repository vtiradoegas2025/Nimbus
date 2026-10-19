import { useCallback, useState } from "react";
import { volumeTextureCapabilities } from "./volumeTextureCapabilities";

export const MINIMUM_WEBGL2_VOLUME_CELLS = 256 ** 3;
interface TextureWindow {
  getApiSpecificRenderWindow(): object;
}

export function useVolumeTextureCapacity(generation: number) {
  const [state, setState] = useState<{ generation: number; maximumDimension: number; maximumCells: number; error: string | null } | null>(null);
  const probe = useCallback((window: TextureWindow) => {
    try {
      const view = window.getApiSpecificRenderWindow() as { initialize?: () => void; getContext?: () => WebGL2RenderingContext | null };
      view.initialize?.();
      const gl = view.getContext?.();
      if (!gl || typeof gl.texSubImage3D !== "function" || gl.isContextLost()) {
        throw new Error("Scientific volume rendering requires an available WebGL2 texture context");
      }
      const { maximumDimension, maximumCells } = volumeTextureCapabilities(gl);
      setState({ generation, maximumDimension, maximumCells, error: null });
    } catch (reason) {
      setState({ generation, maximumDimension: 0, maximumCells: 0,
        error: reason instanceof Error ? reason.message : String(reason) });
    }
  }, [generation]);
  const current = state?.generation === generation ? state : null;
  return { probe, known: current != null && current.error == null,
    maximumDimension: current?.maximumDimension ?? 0,
    maximumCells: current?.maximumCells || MINIMUM_WEBGL2_VOLUME_CELLS,
    error: current?.error ?? null };
}
