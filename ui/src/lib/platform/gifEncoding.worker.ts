import { GifEncoding } from "./gifEncoding";

export type GifRequest =
  | { kind: "frame"; width: number; height: number; fps: number; rgba: Uint8ClampedArray }
  | { kind: "finish" };
export type GifReply = { ok: true; bytes?: Uint8Array } | { ok: false; error: string };

const scope = self as unknown as {
  onmessage: (event: MessageEvent<GifRequest>) => void;
  postMessage: (reply: GifReply, transfer?: Transferable[]) => void;
  close: () => void;
};
let encoder: GifEncoding | null = null;
scope.onmessage = ({ data }) => {
  try {
    if (data.kind === "frame") {
      encoder ??= new GifEncoding(data.width, data.height, data.fps);
      encoder.append(data.rgba);
      scope.postMessage({ ok: true });
    } else {
      if (!encoder) throw new Error("No GIF frames captured");
      const bytes = encoder.finish();
      scope.postMessage({ ok: true, bytes }, [bytes.buffer]);
      scope.close();
    }
  } catch (error) {
    scope.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) });
    scope.close();
  }
};
