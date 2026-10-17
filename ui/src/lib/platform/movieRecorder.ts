/*
 * Plot panel → MP4 recorder.
 * Streams frames as base64 PNG to a Rust-side temp dir one at a time (keeps
 * IPC payload bounded), then asks ffmpeg to mux on finish().
 * Encoder: H.264 / libx264 / yuv420p / +faststart — see src-tauri/src/commands/movie.rs.
 */
import {
  appendEncodeFrame,
  cleanupEncodeSession,
  encodeMovieDir,
  prepareEncodeSession,
} from "./tauri";
import { captureFrameDataUri, dataUriToBase64Png, type CaptureSize } from "@/lib/run/frameCapture";
import { assertCaptureCurrent, type CaptureGuard } from "@/lib/run/captureGuard";

export interface MovieRecorderOptions {
  /** Frames per second baked into the MP4. Clamped on the Rust side
   *  to [1, 60]. */
  fps: number;
  /** Override path to an ffmpeg binary; falls back to PATH when empty.
   *  Sourced from Settings on the caller side. */
  ffmpegPathOverride?: string;
}

export class MovieRecorder {
  private readonly fps: number;
  private readonly ffmpegPathOverride: string | undefined;
  private readonly size: CaptureSize;
  // Session id (lazily created on first frame); count doubles as next frame index.
  private sessionId: string | null = null;
  private count = 0;

  constructor(opts: MovieRecorderOptions & CaptureSize) {
    this.fps = Math.max(1, Math.min(60, opts.fps));
    this.ffmpegPathOverride = opts.ffmpegPathOverride?.trim() || undefined;
    this.size = { width: opts.width, height: opts.height };
  }

  /** Capture the panel at `el` and stream it as one PNG to the session's temp dir. */
  async captureFrame(el: HTMLElement, guard: CaptureGuard = {}): Promise<void> {
    assertCaptureCurrent(guard);
    if (this.sessionId == null) {
      this.sessionId = await prepareEncodeSession();
    }
    assertCaptureCurrent(guard);
    const dataUri = await captureFrameDataUri(el, this.size, guard);
    assertCaptureCurrent(guard);
    await appendEncodeFrame(this.sessionId, this.count, dataUriToBase64Png(dataUri));
    assertCaptureCurrent(guard);
    this.count += 1;
  }

  frameCount(): number {
    return this.count;
  }

  /** Mux frames into an MP4 at `outputPath`. Rust deletes temp frames on return. Session is consumed. */
  async finish(outputPath: string): Promise<void> {
    if (this.count === 0 || this.sessionId == null) {
      throw new Error("No frames captured");
    }
    const session = this.sessionId;
    // Consume the session first: encode_movie_dir removes the temp dir, so dispose() must not re-clean.
    this.sessionId = null;
    this.count = 0;
    await encodeMovieDir(session, outputPath, this.fps, this.ffmpegPathOverride);
  }

  /** Delete the session's temp frames (cancel/error path). No-op after finish(); never throws. */
  async dispose(): Promise<void> {
    const session = this.sessionId;
    if (session == null) return;
    this.sessionId = null;
    this.count = 0;
    try {
      await cleanupEncodeSession(session);
    } catch {
      // Best-effort cleanup; worst case temp dir lingers under ~/.nimbus/tmp.
    }
  }
}
