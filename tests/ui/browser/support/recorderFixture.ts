const sessions = new Map<string, Array<{ index: number; width: number; height: number }>>();
let pendingPrepare: (() => void) | null = null;
let pendingProbe: (() => void) | null = null;
let probeCount = 0;
let prepared = 0;
let cleaned = 0;
const appended: Array<{ index: number; width: number; height: number }> = [];
const encoded: Array<{ fps: number; frames: number; outputPath: string }> = [];

export const recorderFixture = {
  releaseProbe: () => { pendingProbe?.(); pendingProbe = null; },
  releasePreparation: () => { pendingPrepare?.(); pendingPrepare = null; },
  snapshot: () => ({ probeCount, probePending: pendingProbe !== null, prepared, cleaned, preparationPending: pendingPrepare !== null,
    activeSessions: sessions.size, appended: [...appended], encoded: [...encoded] }),
};

export async function recorderInvoke(command: string, payload: unknown): Promise<{ value: unknown } | null> {
  const args = (payload ?? {}) as Record<string, unknown>;
  if (command === "ffmpeg_detect") {
    probeCount++;
    if (new URLSearchParams(window.location.search).get("recorderProbe") === "held") await new Promise<void>(resolve => { pendingProbe = resolve; });
    return { value: { path: "/fixture/ffmpeg", version: "bounded native bridge fixture" } };
  }
  if (command === "prepare_encode_session") {
    if (new URLSearchParams(window.location.search).get("recorderPreparation") === "held") {
      await new Promise<void>(resolve => { pendingPrepare = resolve; });
    }
    const id = `fixture-encode-${++prepared}`;
    sessions.set(id, []);
    return { value: id };
  }
  if (command === "append_encode_frame") {
    const frames = sessions.get(String(args.sessionId));
    if (!frames || args.index !== frames.length) throw new Error("Native staging requires one exact ordered frame per index");
    const bytes = Uint8Array.from(atob(String(args.frameBase64)), character => character.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    if (view.getUint32(0) !== 0x89504e47) throw new Error("Native staging expected a PNG frame");
    const frame = { index: Number(args.index), width: view.getUint32(16), height: view.getUint32(20) };
    frames.push(frame); appended.push(frame);
    return { value: null };
  }
  if (command === "encode_movie_dir") {
    const id = String(args.sessionId), frames = sessions.get(id);
    if (!frames?.length) throw new Error("Native staging has no complete frames");
    encoded.push({ fps: Number(args.fps), frames: frames.length, outputPath: String(args.outputPath) });
    sessions.delete(id);
    return { value: null };
  }
  if (command === "cleanup_encode_session") {
    if (sessions.delete(String(args.sessionId))) cleaned++;
    return { value: null };
  }
  return null;
}
