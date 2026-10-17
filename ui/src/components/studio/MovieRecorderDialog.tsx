import { useEffect, useMemo, useRef, useState } from "react";
import { GIF_MAX_BYTES, GIF_MAX_FRAMES } from "@/lib/platform/gifLimits";
import { captureViewpoints } from "@/lib/vtk/captureViewpoints";
import { lookupVtkRenderWindow } from "@/lib/vtk/vtkCaptureRegistry";
import { PORTABLE_VIEWPOINT_COUNTS } from "@/lib/platform/portableViewpoints";
import { GifRecorder } from "@/lib/platform/gifRecorder";
import { MovieRecorder } from "@/lib/platform/movieRecorder";
import {
  captureFrameIndices,
  setRecording,
  waitForFrameReady,
} from "@/lib/run/frameReady";
import {
  captureFrameIsReady,
  type CaptureFrameIdentity,
  type DisplayedFrameIdentity,
} from "@/lib/run/frameIdentity";
import { loadStudioSettings } from "@/lib/studio/studioSettings";
import {
  ffmpegDetect,
  pickGifSavePath,
  pickPngSavePath,
  pickViewpointsSavePath,
  pickMovieSavePath,
  writeBinaryFile,
  type FfmpegInfo,
} from "@/lib/platform/tauri";
import { Icon } from "@/components/shell/Icon";
import { captureFrameDataUri, dataUriToBase64Png } from "@/lib/run/frameCapture";
import { assertCaptureCurrent, type CaptureGuard } from "@/lib/run/captureGuard";
import "./MovieRecorderDialog.css";

export type RecorderFormat = "gif" | "mp4" | "png" | "html";

interface MovieRecorderDialogProps {
  /** Total number of steps in the loaded run. */
  totalSteps: number;
  /** Currently displayed step index — used as the default start. */
  currentStepIdx: number;
  /** Drives the plot to advance to the next step. */
  setStepIdx: (idx: number) => void;
  /** CSS selector that locates the active plot canvas. */
  plotCanvasSelector: string;
  /** Max ms to wait per frame before treating it as a stale capture; scaled to the grid size by the caller. */
  maxFrameWaitMs: number;
  /** Immutable absolute frame identities captured when the dialog opens. */
  frameIdentities: ReadonlyArray<CaptureFrameIdentity>;
  /** Reads the plot's latest post-render identity without a stale React closure. */
  getDisplayedFrameIdentity: () => DisplayedFrameIdentity | null;
  /** Base filename for the save dialog (extension appended from format). */
  defaultFilename: string;
  /** Close the dialog (called on cancel, error, or completion). */
  onClose: () => void;
  /** Explicit invoker for focus restoration. Required for pointer-opened
   *  dialogs in Safari, which intentionally does not focus clicked buttons. */
  returnFocusRef?: React.RefObject<HTMLElement>;
  /** Keeps renderers in exact multi-layer presentation mode while capturing. */
  onRecordingChange?: (recording: boolean) => void;
}

const FPS_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 2, label: "2 fps" },
  { value: 5, label: "5 fps" },
  { value: 10, label: "10 fps" },
  { value: 15, label: "15 fps" },
  { value: 24, label: "24 fps" },
  { value: 30, label: "30 fps" },
];

const FRAME_STRIDE_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 1, label: "Every frame" },
  { value: 2, label: "Every 2nd" },
  { value: 5, label: "Every 5th" },
  { value: 10, label: "Every 10th" },
];

// Floor for plot types that don't report load progress; upper bound is `maxFrameWaitMs`.
const MIN_FRAME_MS = 200;

type RecordPhase = "config" | "recording" | "encoding" | "saving" | "done" | "error";

const FOCUSABLE_SELECTOR = [
  "button:not(:disabled)",
  "input:not(:disabled)",
  "select:not(:disabled)",
  "textarea:not(:disabled)",
  "a[href]",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

function moveRovingRadioFocus(
  event: React.KeyboardEvent<HTMLButtonElement>,
  onSelect: (value: RecorderFormat) => void,
) {
  const group = event.currentTarget.closest<HTMLElement>('[role="radiogroup"]');
  if (!group) return;
  const radios = Array.from(
    group.querySelectorAll<HTMLButtonElement>('[role="radio"]'),
  );
  const currentIndex = radios.indexOf(event.currentTarget);
  if (currentIndex < 0) return;

  let nextIndex: number;
  switch (event.key) {
    case "ArrowRight":
    case "ArrowDown":
      nextIndex = (currentIndex + 1) % radios.length;
      break;
    case "ArrowLeft":
    case "ArrowUp":
      nextIndex = (currentIndex - 1 + radios.length) % radios.length;
      break;
    case "Home":
      nextIndex = 0;
      break;
    case "End":
      nextIndex = radios.length - 1;
      break;
    default:
      return;
  }

  event.preventDefault();
  const next = radios[nextIndex];
  const value = next?.dataset.radioValue as RecorderFormat | undefined;
  if (!next || !value) return;
  onSelect(value);
  next.focus();
}

export function MovieRecorderDialog({
  totalSteps,
  currentStepIdx,
  setStepIdx,
  plotCanvasSelector,
  maxFrameWaitMs,
  frameIdentities,
  getDisplayedFrameIdentity,
  defaultFilename,
  onClose,
  returnFocusRef,
  onRecordingChange,
}: MovieRecorderDialogProps) {
  const [format, setFormat] = useState<RecorderFormat>("gif");
  const [viewpointCount, setViewpointCount] = useState(24);
  const [startStep, setStartStep] = useState(0);
  const [endStep, setEndStep] = useState(Math.max(0, totalSteps - 1));
  const [fps, setFps] = useState(10);
  const [frameStride, setFrameStride] = useState(1);
  const [phase, setPhase] = useState<RecordPhase>("config");
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [ffmpegInfo, setFfmpegInfo] = useState<FfmpegInfo | null>(null);
  const selectFormat = (next: RecorderFormat) => {
    if (next === "mp4" && format !== "mp4") setFfmpegInfo(null);
    setErrorMsg(null);
    setFormat(next);
  };
  const cancelRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const activeRef = useRef(false);
  const releaseRecordingRef = useRef<(() => void) | null>(null);
  const recordingChangeRef = useRef(onRecordingChange);
  recordingChangeRef.current = onRecordingChange;
  const frameIdentityKey = useMemo(() => JSON.stringify(frameIdentities), [frameIdentities]);
  const sourceRef = useRef({ frameIdentityKey, plotCanvasSelector, getDisplayedFrameIdentity });
  sourceRef.current = { frameIdentityKey, plotCanvasSelector, getDisplayedFrameIdentity };
  const dialogRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  const phaseRef = useRef<RecordPhase>(phase);
  onCloseRef.current = onClose;
  phaseRef.current = phase;
  // Tracks whether the component is still mounted, so awaited callbacks in the
  // capture/encode/save flow don't call setState after unmount.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      cancelRef.current = true;
      abortRef.current?.abort();
      mountedRef.current = false;
      releaseRecordingRef.current?.();
    };
  }, []);

  // A modal owns keyboard focus for its complete mounted lifetime. Keep the
  // caller's trigger so closing returns the user to the exact place they left.
  useEffect(() => {
    const previouslyFocused = returnFocusRef?.current ?? (
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
    );
    const dialog = dialogRef.current;
    if (!dialog) return;

    const focusFirst = () => {
      const selectedRadio = dialog.querySelector<HTMLElement>(
        '[role="radio"][aria-checked="true"]',
      );
      const firstControl = dialog.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
      (selectedRadio ?? firstControl ?? dialog).focus({ preventScroll: true });
    };
    const focusFrame = window.requestAnimationFrame(focusFirst);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (phaseRef.current === "encoding") return;
        event.preventDefault();
        event.stopPropagation();
        cancelRef.current = true;
        abortRef.current?.abort();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab") return;

      const controls = Array.from(
        dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
      ).filter((control) => control.getClientRects().length > 0);
      if (controls.length === 0) {
        event.preventDefault();
        dialog.focus({ preventScroll: true });
        return;
      }
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      } else if (!dialog.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      }
    };

    const onFocusIn = (event: FocusEvent) => {
      if (!dialog.contains(event.target as Node)) focusFirst();
    };

    document.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("focusin", onFocusIn, true);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.removeEventListener("keydown", onKeyDown, true);
      document.removeEventListener("focusin", onFocusIn, true);
      if (previouslyFocused?.isConnected) {
        previouslyFocused.focus({ preventScroll: true });
      }
    };
  }, [returnFocusRef]);

  // Reset start/end when the run changes underneath us.
  useEffect(() => {
    setStartStep(0);
    setEndStep(Math.max(0, totalSteps - 1));
  }, [totalSteps]);

  // Probe ffmpeg on every MP4 format selection; re-probe on flip in case Settings changed.
  useEffect(() => {
    if (format !== "mp4") return;
    let cancelled = false;
    setFfmpegInfo(null);
    const override = loadStudioSettings().ffmpegPathOverride.trim() || undefined;
    (async () => {
      try {
        const info = await ffmpegDetect(override);
        if (!cancelled) setFfmpegInfo(info);
      } catch (e) {
        if (!cancelled) {
          setFfmpegInfo({ path: null, version: null });
          setErrorMsg(
            `Couldn't probe ffmpeg: ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [format]);

  const startRecording = async () => {
    if (activeRef.current || (format === "mp4" && !ffmpegInfo?.path)) return;
    if (format !== "png" && format !== "html" && startStep > endStep) {
      setErrorMsg("Start step must be ≤ end step.");
      return;
    }
    const canvas = document.querySelector<HTMLElement>(plotCanvasSelector);
    if (!canvas) {
      setErrorMsg("Couldn't find the plot canvas. Switch to a plot view and try again.");
      return;
    }
    const size = captureOutputSize(canvas, format);
    const captureSteps = (format === "png" || format === "html") ? [currentStepIdx] : captureFrameIndices(totalSteps, startStep, endStep, frameStride);
    const sourceKey = frameIdentityKey;
    const controlScope = canvas.closest(".volume-render") ?? canvas.closest(".plot-body")?.parentElement ?? canvas.parentElement;
    const controls = () => JSON.stringify(Array.from(controlScope?.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button[aria-pressed],button[role=radio]") ?? [])
      .map(control => [control.tagName, control.getAttribute("aria-label"), "value" in control ? control.value : null,
        control instanceof HTMLInputElement ? control.checked : null, control.getAttribute("aria-pressed"), control.getAttribute("aria-checked")]));
    const initialControls = controls();
    const abort = new AbortController();
    abortRef.current = abort;
    cancelRef.current = false;
    activeRef.current = true;
    setErrorMsg(null);
    setPhase("recording");
    setProgress({ done: 0, total: format === "html" ? viewpointCount : captureSteps.length });
    setRecording(true);
    recordingChangeRef.current?.(true);
    let released = false;
    const releaseRecording = () => {
      if (released) return;
      released = true;
      setRecording(false);
      recordingChangeRef.current?.(false);
    };
    releaseRecordingRef.current = releaseRecording;
    const sessionGuard: CaptureGuard = {
      signal: abort.signal,
      assertCurrent: () => {
        if (!mountedRef.current || cancelRef.current) throw new DOMException("Capture cancelled", "AbortError");
        if (sourceRef.current.plotCanvasSelector !== plotCanvasSelector || sourceRef.current.frameIdentityKey !== sourceKey ||
          document.querySelector(plotCanvasSelector) !== canvas || controls() !== initialControls) {
          throw new Error("The plot source or controls changed during export. Reopen the dialog and try again.");
        }
      },
    };
    let gif: GifRecorder | null = null;
    let movie: MovieRecorder | null = null;
    let png: string | null = null;
    let portable: Uint8Array | null = null;
    try {
      if (format === "gif") {
        if (captureSteps.length > GIF_MAX_FRAMES) throw new Error("Choose fewer source frames or MP4 for this animation");
        gif = new GifRecorder({ fps, ...size });
      }
      if (format === "mp4") movie = new MovieRecorder({ fps, ...size, ffmpegPathOverride: ffmpegInfo!.path! });
      for (let position = 0; position < captureSteps.length; position++) {
        assertCaptureCurrent(sessionGuard);
        const index = captureSteps[position];
        const expected = frameIdentities[index];
        if (!expected) throw new Error(`Frame ${index} is no longer available.`);
        setStepIdx(index);
        const exact = () => captureFrameIsReady(expected, sourceRef.current.getDisplayedFrameIdentity());
        const ready = await waitForFrameReady(MIN_FRAME_MS, maxFrameWaitMs, exact, abort.signal);
        assertCaptureCurrent(sessionGuard);
        if (!ready) throw new Error(`Step ${expected.stepIndex} didn't finish rendering within ${Math.round(maxFrameWaitMs / 1000)} s — recording stopped rather than capture a stale or partial frame. Wait for every enabled layer to finish, or choose a shorter saved-frame range.`);
        const frameGuard: CaptureGuard = {
          signal: abort.signal,
          assertCurrent: () => {
            assertCaptureCurrent(sessionGuard);
            if (!exact()) throw new Error("The displayed frame changed during capture. No stale or partial frame was exported.");
          },
        };
        if (format === "html") {
          portable = await captureViewpoints(canvas, size, viewpointCount, defaultFilename, expected, frameGuard, done => {
            if (mountedRef.current) setProgress({ done, total: viewpointCount });
          });
        } else if (gif) await gif.captureFrame(canvas, frameGuard);
        else if (movie) await movie.captureFrame(canvas, frameGuard);
        else png = await captureFrameDataUri(canvas, size, frameGuard);
        assertCaptureCurrent(frameGuard);
        if (format !== "html") setProgress({ done: position + 1, total: captureSteps.length });
      }
      setPhase("saving");
      const path = await (format === "html" ? pickViewpointsSavePath(`${defaultFilename}.html`) : format === "png" ? pickPngSavePath(`${defaultFilename}.png`) : gif ? pickGifSavePath(`${defaultFilename}.gif`) : pickMovieSavePath(`${defaultFilename}.mp4`));
      assertCaptureCurrent(sessionGuard);
      if (path === null) { setPhase("config"); return; }
      if (movie) {
        setPhase("encoding");
        await movie.finish(path);
      } else {
        const bytes = portable ?? (gif ? await gif.finish(abort.signal) : Uint8Array.from(atob(dataUriToBase64Png(png!)), character => character.charCodeAt(0)));
        assertCaptureCurrent(sessionGuard);
        await writeBinaryFile(path, bytes);
      }
      assertCaptureCurrent(sessionGuard);
      setSavedPath(path);
      setPhase("done");
    } catch (error) {
      if (!mountedRef.current || abort.signal.aborted) return;
      setErrorMsg(error instanceof Error ? error.message : String(error));
      setPhase("error");
    } finally {
      gif?.dispose();
      await movie?.dispose();
      activeRef.current = false;
      if (abortRef.current === abort) abortRef.current = null;
      releaseRecording();
      if (releaseRecordingRef.current === releaseRecording) releaseRecordingRef.current = null;
    }
  };

  const onCancel = () => {
    if (phase !== "encoding") {
      cancelRef.current = true;
      abortRef.current?.abort();
      onClose();
    }
  };

  const recording = phase === "recording";
  const encoding = phase === "encoding";
  const busy = recording || encoding || phase === "saving";
  const mp4Missing = format === "mp4" && ffmpegInfo !== null && !ffmpegInfo.path;
  const mp4Checking = format === "mp4" && ffmpegInfo === null;
  const exportFrames = captureFrameIndices(totalSteps, startStep, endStep, frameStride).length;
  const gifTooLong = format === "gif" && exportFrames > GIF_MAX_FRAMES;
  const captureCanvas = document.querySelector<HTMLElement>(plotCanvasSelector);
  const has3d = !!lookupVtkRenderWindow(captureCanvas);
  const canStart = totalSteps > 0 && !mp4Missing && !mp4Checking && !gifTooLong && (format !== "html" || has3d);
  const outputSize = captureCanvas ? captureOutputSize(captureCanvas, format) : null;
  const progressRatio = encoding
    ? 1
    : progress.total > 0
      ? Math.min(1, progress.done / progress.total)
      : 0;
  const progressText = encoding
    ? "Encoding in progress"
    : `${progress.done} of ${progress.total} frames`;

  return (
    <div
      className="movie-recorder__backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="movie-recorder-title"
      ref={dialogRef}
      tabIndex={-1}
    >
      <div className="movie-recorder__dialog">
        <header className="movie-recorder__header">
          <Icon name="play" size={14} />
          <h3 id="movie-recorder-title">Record animation</h3>
        </header>

        {phase === "config" && (
          <div className="movie-recorder__body">
            <div className="movie-recorder__field">
              <span id="movie-recorder-format-label">Format</span>
              <div
                className="movie-recorder__format"
                role="radiogroup"
                aria-labelledby="movie-recorder-format-label"
              >
                <button
                  type="button"
                  className={
                    format === "gif" ? "movie-recorder__format--active" : ""
                  }
                  onClick={() => selectFormat("gif")}
                  onKeyDown={(event) => moveRovingRadioFocus(event, selectFormat)}
                  role="radio"
                  aria-checked={format === "gif"}
                  tabIndex={format === "gif" ? 0 : -1}
                  data-radio-value="gif"
                >
                  GIF
                </button>
                <button
                  type="button"
                  className={
                    format === "mp4" ? "movie-recorder__format--active" : ""
                  }
                  onClick={() => selectFormat("mp4")}
                  onKeyDown={(event) => moveRovingRadioFocus(event, selectFormat)}
                  role="radio"
                  aria-checked={format === "mp4"}
                  tabIndex={format === "mp4" ? 0 : -1}
                  data-radio-value="mp4"
                >
                  MP4
                </button>
                <button type="button" className={format === "png" ? "movie-recorder__format--active" : ""}
                  onClick={() => selectFormat("png")} onKeyDown={event => moveRovingRadioFocus(event, selectFormat)}
                  role="radio" aria-checked={format === "png"} tabIndex={format === "png" ? 0 : -1} data-radio-value="png">PNG</button>
                <button type="button" className={format === "html" ? "movie-recorder__format--active" : ""}
                  onClick={() => selectFormat("html")} onKeyDown={event => moveRovingRadioFocus(event, selectFormat)}
                  role="radio" aria-checked={format === "html"} tabIndex={format === "html" ? 0 : -1} data-radio-value="html">Interactive viewpoints</button>
              </div>
            </div>

            {mp4Checking && <p role="status">Checking the MP4 encoder…</p>}
            {gifTooLong && <p role="alert">GIF supports up to {GIF_MAX_FRAMES} frames. Choose fewer source frames or MP4.</p>}
            {mp4Missing && (
              <div className="movie-recorder__missing">
                <Icon name="warning" size={12} />
                <div>
                  <strong>ffmpeg not found.</strong> Install with{" "}
                  <code>brew install ffmpeg</code> (macOS) or{" "}
                  <code>apt install ffmpeg</code> (Linux), or point to an
                  existing binary in Settings → Encoding.
                </div>
              </div>
            )}

            {format === "html" && <>
              <label className="movie-recorder__field"><span>Viewpoints</span><select value={viewpointCount} onChange={event => setViewpointCount(Number(event.target.value))}>
                {PORTABLE_VIEWPOINT_COUNTS.map(count => <option key={count} value={count}>{count} views · {360 / count}° spacing</option>)}
              </select></label>
              <p className="movie-recorder__hint">One portable HTML file with lossless PNGs at the current saved time and elevation. Rotate through captured angles and magnify offline. Keeps legends and source/scene metadata. This is a set of images, not a free-rotation 3D model; magnification adds no detail. Maximum file size: 64 MiB. Your camera is restored after capture.</p>
              {!has3d && <p role="alert">Choose a 3D plot to export interactive viewpoints.</p>}
            </>}
            {format !== "png" && format !== "html" && <>
            <label className="movie-recorder__field">
              <span>Start step</span>
              <StepInput
                value={startStep}
                max={Math.max(0, totalSteps - 1)}
                onCommit={(n) => setStartStep(clamp(n, 0, totalSteps - 1))}
              />
            </label>
            <label className="movie-recorder__field">
              <span>End step</span>
              <StepInput
                value={endStep}
                max={Math.max(0, totalSteps - 1)}
                onCommit={(n) => setEndStep(clamp(n, 0, totalSteps - 1))}
              />
            </label>
            <label className="movie-recorder__field">
              <span>FPS</span>
              <select
                value={fps}
                onChange={(e) => setFps(Number(e.target.value))}
              >
                {FPS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="movie-recorder__field">
              <span>Source frames</span>
              <select
                value={frameStride}
                onChange={(event) => setFrameStride(Number(event.target.value))}
              >
                {FRAME_STRIDE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            </>}
            <div className="movie-recorder__hint">
              Primary plot only · {outputSize ? `${outputSize.width} × ${outputSize.height} output pixels` : "plot unavailable"}.
              {format === "html" ? ` ${viewpointCount} viewpoints at saved step ${currentStepIdx}.` : format === "png" ? ` Current saved step: ${currentStepIdx}; full-quality image with scientific legends.` : <>
                {" "}{exportFrames} exported frames · every enabled layer settles before capture.
                FPS controls output timing; source frames controls which solved steps are included. Current step: {currentStepIdx}.
                {format === "gif" && ` GIF encoding runs in a worker with a ${GIF_MAX_BYTES / 1024 / 1024} MiB encoded-data limit. Long animations may need fewer source frames or MP4. GIF timing uses centiseconds.`}
              </>}
              {" "}Image pixels interpolate the saved source; export does not add model resolution or intermediate times.
              {format === "mp4" && ffmpegInfo?.version && (
                <>
                  <br />
                  <span style={{ opacity: 0.8 }}>
                    Encoder: {ffmpegInfo.version}
                  </span>
                </>
              )}
            </div>
            {errorMsg && (
              <div className="movie-recorder__error">
                <Icon name="warning" size={12} />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        )}

        {busy && (
          <div className="movie-recorder__body">
            <div className="movie-recorder__progress-label" id="movie-recorder-progress-label">
              {recording
                ? "Capturing frames…"
                : encoding
                  ? "Encoding MP4 with ffmpeg…"
                  : `Saving ${format.toUpperCase()}…`}
            </div>
            <div
              className="movie-recorder__progress-bar"
              role="progressbar"
              aria-labelledby="movie-recorder-progress-label"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={encoding ? undefined : Math.round(progressRatio * 100)}
              aria-valuetext={progressText}
            >
              <div
                className="movie-recorder__progress-fill"
                style={{ transform: `scaleX(${progressRatio})` }}
              />
            </div>
            <div className="movie-recorder__progress-meta" aria-live="polite">
              {encoding
                ? "Encoding time depends on frame count and this computer. Keep Studio open until it finishes."
                : `${progress.done} / ${progress.total} frames`}
            </div>
          </div>
        )}

        {phase === "done" && savedPath && (
          <div className="movie-recorder__body">
            <div className="movie-recorder__done">
              <Icon name="dot" size={12} />
              <span>Saved to {savedPath}</span>
            </div>
          </div>
        )}

        {phase === "error" && (
          <div className="movie-recorder__body">
            <div className="movie-recorder__error">
              <Icon name="warning" size={12} />
              <span>{errorMsg ?? "Recording failed"}</span>
            </div>
          </div>
        )}

        <footer className="movie-recorder__footer">
          <button
            type="button"
            className="btn"
            onClick={onCancel}
            disabled={phase === "encoding"}
            title={
              phase === "encoding"
                ? "Keep Studio open until the encoder finishes"
                : undefined
            }
          >
            {phase === "encoding"
              ? "Working…"
              : phase === "done" || phase === "error"
                ? "Close"
                : "Cancel"}
          </button>
          {phase === "config" && (
            <button
              type="button"
              className="btn btn--primary"
              onClick={startRecording}
              disabled={!canStart}
              title={
                mp4Missing
                  ? "Install ffmpeg or set the path in Settings to enable MP4"
                  : undefined
              }
            >
              <Icon name="play" size={12} />
              <span>{format === "html" ? "Export viewpoints" : format === "png" ? "Save current frame" : "Start recording"}</span>
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}

function captureOutputSize(element: HTMLElement, format: RecorderFormat) {
  const canvas = element instanceof HTMLCanvasElement ? element : element.querySelector("canvas");
  if ((format === "png" || format === "html") && canvas && canvas.width > 0 && canvas.height > 0) return { width: canvas.width, height: canvas.height };
  const rect = element.getBoundingClientRect();
  if (format === "png") {
    const ratio = Number.isFinite(window.devicePixelRatio) && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
    return { width: Math.max(1, Math.round(rect.width * ratio)), height: Math.max(1, Math.round(rect.height * ratio)) };
  }
  return { width: toEven(Math.max(200, Math.round(rect.width))), height: toEven(Math.max(200, Math.round(rect.height))) };
}

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, Math.floor(v)));
}

interface StepInputProps {
  value: number;
  max: number;
  onCommit: (next: number) => void;
}

/** Number field that can hold an empty draft while being retyped. A plain
 *  controlled input coerces "" to 0 and snaps back mid-edit, so clearing the
 *  field and typing a new step rendered a malformed value like "02". Blur
 *  drops the draft so the field resyncs to the committed step. */
function StepInput({ value, max, onCommit }: StepInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="number"
      min={0}
      max={max}
      value={draft ?? String(value)}
      onChange={(e) => {
        const raw = e.target.value;
        setDraft(raw);
        if (raw === "") return;
        const parsed = Number(raw);
        if (Number.isFinite(parsed)) onCommit(parsed);
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

/** Round down to the nearest even integer. MP4 (yuv420p / libx264) requires
 *  even frame dimensions; an odd width or height fails the encode. */
function toEven(v: number): number {
  return v - (v % 2);
}
