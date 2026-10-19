use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use serde::Serialize;

use crate::error::{AppError, AppResult};

/*
 * Movie export — encodes a sequence of PNG frames into an MP4 via the
 * qualified `ffmpeg` binary.
 *
 * Frames stream to disk one at a time via a session flow:
 * prepare_encode_session mints a temp dir, append_encode_frame writes each
 * base64 PNG into it, and encode_movie_dir muxes the directory with ffmpeg.
 * This bounds the IPC payload to a single frame at a time — a large recording
 * (e.g. ~90 frames of a 42M-cell "beast" run, hundreds of MB of base64) would
 * otherwise ship the whole buffer in one command and risk an OOM / IPC failure.
 * cleanup_encode_session removes a half-captured session on cancel.
 *
 * ffmpeg discovery order:
 *   1. Caller-supplied override path (from Settings).
 *   2. The application-local Windows runtime bundled by setup.
 *   3. `which ffmpeg` / `where ffmpeg` on $PATH and maintained native paths.
 *
 * Missing ffmpeg produces a structured AppError::NotFound the UI renders with
 * platform-appropriate setup hints. Windows installers bundle the matching
 * UCRT64 runtime so installation is self-contained; macOS/Linux setup installs
 * and verifies the native package before building Studio.
 */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FfmpegInfo {
    /// Resolved absolute path to the ffmpeg binary. None when not found.
    pub path: Option<String>,
    /// First line of `ffmpeg -version` output. None when not found.
    pub version: Option<String>,
}

#[tauri::command]
pub fn ffmpeg_detect(override_path: Option<String>) -> AppResult<FfmpegInfo> {
    let resolved = resolve_ffmpeg(override_path)?;
    let info = match resolved {
        Some(path) => {
            let version = probe_ffmpeg_version(&path).ok();
            FfmpegInfo {
                path: Some(path.display().to_string()),
                version,
            }
        }
        None => FfmpegInfo {
            path: None,
            version: None,
        },
    };
    Ok(info)
}

/// Mint a fresh temp directory for one recording's frames and return its opaque
/// session id (a single path component under the encoding root). Frames are
/// appended one at a time so the whole recording never crosses IPC at once.
#[tauri::command]
pub async fn prepare_encode_session() -> AppResult<String> {
    tauri::async_runtime::spawn_blocking(|| -> AppResult<String> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let id = format!("encode-{ts}-{}-{seq}", std::process::id());
        let dir = encoding_temp_root().join(&id);
        std::fs::create_dir_all(&dir).map_err(|e| {
            AppError::Io(std::io::Error::new(
                e.kind(),
                format!("create encode session dir {}: {e}", dir.display()),
            ))
        })?;
        Ok(id)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

/// Decode one base64 PNG and write it as frame_{index:05}.png into the session
/// directory. Frames arrive pre-base64'd from the webview (PNG data URIs out of
/// Plotly / VTK capture); a base64 string per frame keeps each IPC payload to
/// roughly one PNG, versus a Vec of every frame in a single command.
#[tauri::command]
pub async fn append_encode_frame(
    session_id: String,
    index: u32,
    frame_base64: String,
) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || -> AppResult<()> {
        let dir = session_dir(&session_id)?;
        if !dir.is_dir() {
            return Err(AppError::NotFound(format!(
                "encode session '{session_id}' not found (prepare it first)"
            )));
        }
        write_one_frame(&dir, index, &frame_base64)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

/// Mux a session's frame_*.png into an H.264 MP4 at `output_path`, then delete
/// the session directory regardless of how ffmpeg exits (RAII guard).
#[tauri::command]
pub async fn encode_movie_dir(
    session_id: String,
    output_path: String,
    fps: u32,
    ffmpeg_path_override: Option<String>,
) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || -> AppResult<()> {
        let dir = session_dir(&session_id)?;
        if !dir.is_dir() {
            return Err(AppError::NotFound(format!(
                "encode session '{session_id}' not found"
            )));
        }
        // Remove the session dir on every exit path below (ffmpeg ok or not).
        let _guard = TempEncodeDir { path: dir.clone() };

        // An empty session (no frames appended) would make ffmpeg fail obscurely
        // on a missing input pattern — surface a clear error instead. Counting
        // only real frames matters: a stray .DS_Store would otherwise pass this
        // gate and hand ffmpeg the very error it exists to prevent.
        let frame_count = count_session_frames(&dir);
        if frame_count == 0 {
            return Err(AppError::Invalid("no frames supplied".into()));
        }

        let fps_clamped = fps.clamp(1, 60);
        let ffmpeg = resolve_ffmpeg(ffmpeg_path_override)?.ok_or_else(|| {
            AppError::NotFound(
                "ffmpeg not found on PATH; set the path in Settings → Encoding".into(),
            )
        })?;

        let outpath = PathBuf::from(expand_tilde(&output_path));
        if let Some(parent) = outpath.parent() {
            // Belt-and-braces: the save dialog already creates the directory by
            // virtue of the user picking a location, but mkdir -p so a
            // programmatic invocation doesn't blow up on a stale parent.
            std::fs::create_dir_all(parent).map_err(|e| {
                AppError::Io(std::io::Error::new(
                    e.kind(),
                    format!("create movie output parent {}: {e}", parent.display()),
                ))
            })?;
        }

        // ffmpeg truncates its output the instant it starts, so encoding
        // straight to the user's path would destroy a previous good export on
        // any failure. Build beside it and rename only on success.
        let partial = partial_output_path(&outpath, &session_id);
        let mut partial_guard = PartialOutput {
            path: partial.clone(),
            keep: false,
        };

        let pattern = dir.join("frame_%05d.png");
        let mut child = Command::new(&ffmpeg)
            .arg("-y")
            .arg("-loglevel")
            .arg("error")
            .arg("-framerate")
            .arg(fps_clamped.to_string())
            .arg("-i")
            .arg(&pattern)
            // Guarantee even dimensions for yuv420p (4:2:0 chroma): pad up to the
            // next even width/height. libx264 errors out (-22 "Invalid argument")
            // on an odd-sized frame, and a capture canvas is frequently odd. This
            // is a no-op when already even (the recorder rounds capture size down
            // to even), so it never changes a valid frame.
            .arg("-vf")
            .arg("pad=ceil(iw/2)*2:ceil(ih/2)*2")
            .arg("-c:v")
            .arg("libx264")
            .arg("-preset")
            .arg("medium")
            .arg("-crf")
            .arg("23")
            // yuv420p is broadly compatible with QuickTime / browsers; libx264
            // would default to a high-444 mode that many players reject.
            .arg("-pix_fmt")
            .arg("yuv420p")
            // Move the moov atom to the front so the file starts playing before
            // the whole stream downloads — matters for users sharing via web /
            // Slack / chat.
            .arg("-movflags")
            .arg("+faststart")
            .arg(&partial)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| AppError::Subprocess(format!("spawn ffmpeg {}: {e}", ffmpeg.display())))?;

        // Drain stderr on its own thread: a wedged encode that fills the pipe
        // would otherwise deadlock before the timeout could fire.
        let stderr_drain = child.stderr.take().map(|mut stream| {
            std::thread::spawn(move || {
                let mut captured = String::new();
                let _ = stream.read_to_string(&mut captured);
                captured
            })
        });

        let exit = wait_with_timeout(&mut child, encode_timeout(frame_count));
        let detail = stderr_drain
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default();
        let detail = detail.trim();

        match exit? {
            Some(status) if status.success() => {}
            Some(status) => {
                return Err(AppError::Subprocess(if detail.is_empty() {
                    format!("ffmpeg exited with status {status}")
                } else {
                    format!("ffmpeg exited with status {status}: {detail}")
                }));
            }
            None => {
                return Err(AppError::Subprocess(format!(
                    "ffmpeg did not finish within {}s and was terminated",
                    encode_timeout(frame_count).as_secs()
                )));
            }
        }

        crate::platform::atomic_replace_file(&partial, &outpath).map_err(|e| {
            AppError::Io(std::io::Error::new(
                e.kind(),
                format!("publish movie to {}: {e}", outpath.display()),
            ))
        })?;
        partial_guard.keep();
        Ok(())
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

/// Delete a session's frames without muxing — the cancel / error path so a
/// half-captured recording doesn't leak PNGs. Best-effort.
#[tauri::command]
pub async fn cleanup_encode_session(session_id: String) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || -> AppResult<()> {
        let dir = session_dir(&session_id)?;
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

/* ---------- internals ---------- */

fn resolve_ffmpeg(override_path: Option<String>) -> AppResult<Option<PathBuf>> {
    if let Some(p) = override_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let pb = PathBuf::from(expand_tilde(p));
        if !pb.exists() {
            return Err(AppError::NotFound(pb.display().to_string()));
        }
        return Ok(Some(pb));
    }
    Ok(which_ffmpeg())
}

fn which_ffmpeg() -> Option<PathBuf> {
    #[cfg(windows)]
    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            let bundled = directory.join("bin").join("ffmpeg.exe");
            if bundled.is_file() {
                return Some(bundled);
            }
        }
    }

    // PATH lookup with the OS-appropriate executable name. Avoids pulling
    // in the `which` crate — single shell-out keeps the dependency
    // surface tiny.
    #[cfg(unix)]
    let (which, name) = ("which", "ffmpeg");
    #[cfg(windows)]
    let (which, name) = ("where", "ffmpeg.exe");

    if let Ok(out) = Command::new(which).arg(name).output() {
        if out.status.success() {
            if let Ok(stdout) = String::from_utf8(out.stdout) {
                if let Some(first) = stdout
                    .lines()
                    .next()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    return Some(PathBuf::from(first));
                }
            }
        }
    }

    // PATH lookup misses when the app is launched from the macOS Finder / Dock,
    // which does NOT inherit the interactive shell PATH — so a Homebrew ffmpeg in
    // /opt/homebrew/bin is invisible to `which` even though a terminal would find
    // it. Probe the common install locations before giving up, so MP4 export
    // works out of the box without the user setting an override in Settings.
    #[cfg(unix)]
    const CANDIDATES: &[&str] = &[
        "/opt/homebrew/bin/ffmpeg", // Apple-silicon Homebrew
        "/usr/local/bin/ffmpeg",    // Intel Homebrew / manual install
        "/opt/local/bin/ffmpeg",    // MacPorts
        "/usr/bin/ffmpeg",          // Linux distro packages
    ];
    #[cfg(windows)]
    const CANDIDATES: &[&str] = &[
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ];
    CANDIDATES.iter().map(PathBuf::from).find(|pb| pb.is_file())
}

fn probe_ffmpeg_version(ffmpeg: &Path) -> AppResult<String> {
    let out = Command::new(ffmpeg)
        .arg("-version")
        .output()
        .map_err(|e| AppError::Subprocess(format!("ffmpeg -version: {e}")))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(stdout.lines().next().unwrap_or("").trim().to_string())
}

/// Counter that disambiguates two sessions minted in the same nanosecond.
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Resolve a session id to its frames directory, refusing anything that isn't a
/// single `encode-*` component under the encoding root. Defense in depth: the id
/// is minted by prepare_encode_session, but it crosses the IPC boundary, so the
/// frame-write / cleanup commands must not be able to touch an arbitrary path.
fn session_dir(session_id: &str) -> AppResult<PathBuf> {
    if !session_id.starts_with("encode-")
        || session_id.contains('/')
        || session_id.contains('\\')
        || session_id.contains("..")
    {
        return Err(AppError::Invalid(format!(
            "invalid encode session id: {session_id}"
        )));
    }
    Ok(encoding_temp_root().join(session_id))
}

/// Decode a single base64 PNG (bare payload or full data URI) and write it as
/// frame_{index:05}.png into `dir`.
fn write_one_frame(dir: &Path, index: u32, frame_base64: &str) -> AppResult<()> {
    // Tolerate either a bare base64 payload or a full data URI, depending on
    // whether the caller already stripped the "data:image/png;base64," prefix.
    let payload = frame_base64
        .strip_prefix("data:image/png;base64,")
        .unwrap_or(frame_base64);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|e| AppError::Invalid(format!("frame {index} base64 decode: {e}")))?;
    let path = dir.join(format!("frame_{:05}.png", index));
    std::fs::write(&path, &bytes).map_err(|e| {
        AppError::Io(std::io::Error::new(
            e.kind(),
            format!("write {}: {e}", path.display()),
        ))
    })
}

/// RAII guard: adopts an existing session directory and deletes it on drop, so a
/// finished or failed encode doesn't leak hundreds of PNGs into ~/.nimbus/.
/// Deletes the in-progress encode target unless `keep()` is called.
struct PartialOutput {
    path: PathBuf,
    keep: bool,
}

impl PartialOutput {
    fn keep(&mut self) {
        self.keep = true;
    }
}

impl Drop for PartialOutput {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Sibling of the destination — same directory, so the publish is a rename
/// within one filesystem rather than a copy that could half-succeed. The
/// session id keeps concurrent encodes to the same destination separate.
fn partial_output_path(outpath: &Path, session_id: &str) -> PathBuf {
    let name = outpath
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("movie.mp4");
    outpath.with_file_name(format!(".{name}.{session_id}.partial"))
}

/// Frames ffmpeg will actually consume. Anything not matching the exact
/// `frame_NNNNN.png` pattern the writer emits is ignored.
fn count_session_frames(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_str().is_some_and(is_frame_file_name))
        .count()
}

fn is_frame_file_name(name: &str) -> bool {
    name.strip_prefix("frame_")
        .and_then(|rest| rest.strip_suffix(".png"))
        .is_some_and(|digits| digits.len() == 5 && digits.bytes().all(|b| b.is_ascii_digit()))
}

/// Encoding is legitimately slow at 4K, so the cap scales with the frame
/// count; the floor and ceiling keep a tiny recording from timing out early
/// and a wedged ffmpeg from pinning the worker forever.
const ENCODE_TIMEOUT_PER_FRAME: Duration = Duration::from_secs(2);
const ENCODE_TIMEOUT_FLOOR: Duration = Duration::from_secs(60);
const ENCODE_TIMEOUT_CEILING: Duration = Duration::from_secs(1800);
const ENCODE_POLL: Duration = Duration::from_millis(100);

fn encode_timeout(frame_count: usize) -> Duration {
    let scaled = ENCODE_TIMEOUT_FLOOR.saturating_add(
        ENCODE_TIMEOUT_PER_FRAME.saturating_mul(frame_count.min(u32::MAX as usize) as u32),
    );
    scaled.clamp(ENCODE_TIMEOUT_FLOOR, ENCODE_TIMEOUT_CEILING)
}

/// `Ok(Some(status))` when it exited on its own, `Ok(None)` when the deadline
/// passed and it was killed. Never returns while the child is still alive.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> AppResult<Option<std::process::ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => {}
            Err(e) => return Err(AppError::Subprocess(format!("wait for ffmpeg: {e}"))),
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(ENCODE_POLL);
    }
}

struct TempEncodeDir {
    path: PathBuf,
}

impl Drop for TempEncodeDir {
    fn drop(&mut self) {
        // Best-effort cleanup. If a frame is locked by a slow scanner /
        // antivirus we leave the dir behind rather than panicking.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Sessions leaked by an app quit or crash mid-recording. Cleanup on the JS
/// side is best-effort and cannot survive a teardown, so a full recording's
/// worth of PNGs (hundreds of MB at 4K) would otherwise accumulate forever
/// with nothing to reap it.
pub fn sweep_stale_encode_sessions(max_age: Duration) {
    sweep_stale_encode_sessions_in(&encoding_temp_root(), max_age);
}

fn sweep_stale_encode_sessions_in(root: &Path, max_age: Duration) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("encode-") {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .and_then(|modified| {
                SystemTime::now()
                    .duration_since(modified)
                    .map_err(|_| std::io::Error::other("modified in the future"))
            })
            .map(|age| age >= max_age)
            // A directory whose mtime we cannot read is left alone; a live
            // session is worth more than the disk it occupies.
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

fn encoding_temp_root() -> PathBuf {
    // Co-locate under the user's nimbus dir so cleanup is obvious if
    // an encode crashes the host process and the Drop never fires.
    crate::platform::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nimbus")
        .join("tmp")
}

fn expand_tilde(p: &str) -> String {
    crate::platform::expand_tilde(p).display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "nimbus-movie-test-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn session_dir_accepts_a_well_formed_id() {
        let dir = session_dir("encode-123-456-7").expect("valid id");
        assert_eq!(dir.parent(), Some(encoding_temp_root().as_path()));
        assert_eq!(
            dir.file_name().and_then(|s| s.to_str()),
            Some("encode-123-456-7"),
        );
    }

    #[test]
    fn session_dir_rejects_traversal_and_separators() {
        // A session id crosses the IPC boundary, so the frame-write / cleanup
        // commands must never resolve it to a path outside the encoding root.
        for bad in [
            "../etc",
            "encode-../escape",
            "encode-a/b",
            "encode-a\\b",
            "foo", // missing the encode- prefix
            "",
        ] {
            assert!(session_dir(bad).is_err(), "expected '{bad}' to be rejected");
        }
    }

    #[test]
    fn write_one_frame_round_trips_base64_and_data_uri() {
        let dir = unique_test_dir("roundtrip");
        let payload: &[u8] = &[0u8, 1, 2, 3, 250, 251, 252, 253];
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);

        write_one_frame(&dir, 5, &b64).expect("write bare base64 frame");
        assert_eq!(
            std::fs::read(dir.join("frame_00005.png")).unwrap(),
            payload,
            "bare base64 frame round-trips to the decoded bytes",
        );

        // A full data URI (prefix not stripped on the JS side) must also work.
        let data_uri = format!("data:image/png;base64,{b64}");
        write_one_frame(&dir, 6, &data_uri).expect("write data-uri frame");
        assert_eq!(std::fs::read(dir.join("frame_00006.png")).unwrap(), payload);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression: the guard counted any directory entry, so a stray
    /// .DS_Store passed the "no frames" gate and handed ffmpeg the exact
    /// obscure failure the gate exists to prevent.
    #[test]
    fn only_canonical_frame_files_count_as_frames() {
        for good in ["frame_00000.png", "frame_00042.png", "frame_99999.png"] {
            assert!(is_frame_file_name(good), "{good} should count");
        }
        for bad in [
            ".DS_Store",
            "frame_1.png",
            "frame_000001.png",
            "frame_0000a.png",
            "frame_00001.jpg",
            "frame_00001.png.partial",
            "frame_.png",
        ] {
            assert!(!is_frame_file_name(bad), "{bad} should not count");
        }
    }

    #[test]
    fn frame_count_ignores_non_frame_entries() {
        let dir = unique_test_dir("framecount");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".DS_Store"), b"x").unwrap();
        assert_eq!(count_session_frames(&dir), 0, "only junk is not a frame");
        std::fs::write(dir.join("frame_00000.png"), b"x").unwrap();
        std::fs::write(dir.join("frame_00001.png"), b"x").unwrap();
        assert_eq!(count_session_frames(&dir), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression: ffmpeg wrote straight to the user's path with `-y`, so a
    /// failed encode truncated whatever was already there.
    #[test]
    fn partial_output_is_a_hidden_sibling_and_is_removed_unless_kept() {
        let dir = unique_test_dir("partial");
        std::fs::create_dir_all(&dir).unwrap();
        let outpath = dir.join("supercell.mp4");
        let partial = partial_output_path(&outpath, "encode-1-2-3");
        assert_eq!(
            partial.parent(),
            outpath.parent(),
            "rename must stay on one filesystem"
        );
        assert_ne!(partial, outpath);

        std::fs::write(&partial, b"half an mp4").unwrap();
        drop(PartialOutput {
            path: partial.clone(),
            keep: false,
        });
        assert!(!partial.exists(), "an abandoned partial is deleted");

        std::fs::write(&partial, b"half an mp4").unwrap();
        let mut kept = PartialOutput {
            path: partial.clone(),
            keep: false,
        };
        kept.keep();
        drop(kept);
        assert!(partial.exists(), "a kept partial survives for the rename");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encode_timeout_scales_with_frames_and_stays_bounded() {
        assert_eq!(
            encode_timeout(0),
            ENCODE_TIMEOUT_FLOOR,
            "a tiny encode still gets the floor"
        );
        assert!(
            encode_timeout(100) > encode_timeout(10),
            "longer recordings get more budget"
        );
        assert_eq!(
            encode_timeout(usize::MAX),
            ENCODE_TIMEOUT_CEILING,
            "never unbounded"
        );
    }

    #[test]
    fn wait_with_timeout_kills_and_reports_a_wedged_child() {
        let mut child = Command::new("sleep")
            .arg("30")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let exit = wait_with_timeout(&mut child, Duration::from_millis(120)).expect("wait");
        assert!(exit.is_none(), "an overrunning child reports as timed out");
        assert!(
            child.try_wait().expect("try_wait").is_some(),
            "the timed-out child must be killed and reaped, not left running"
        );
    }

    #[test]
    fn wait_with_timeout_returns_a_prompt_exit() {
        let mut child = Command::new("true")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn true");
        let exit = wait_with_timeout(&mut child, Duration::from_secs(10)).expect("wait");
        assert!(exit.expect("exited on its own").success());
    }

    #[test]
    fn sweep_removes_only_aged_encode_sessions() {
        let root = unique_test_dir("sweep");
        let fresh = root.join(format!("encode-sweep-fresh-{}", std::process::id()));
        let other = root.join(format!("not-a-session-{}", std::process::id()));
        std::fs::create_dir_all(&fresh).unwrap();
        std::fs::create_dir_all(&other).unwrap();

        // Nothing just-created is old enough to reap.
        sweep_stale_encode_sessions_in(&root, Duration::from_secs(3600));
        assert!(fresh.exists(), "a live session must survive");
        assert!(other.exists(), "unrelated directories are never touched");

        // Zero age makes every session stale.
        sweep_stale_encode_sessions_in(&root, Duration::ZERO);
        assert!(!fresh.exists(), "an aged session is reaped");
        assert!(other.exists(), "the sweep stays scoped to encode- prefixes");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_one_frame_rejects_bad_base64() {
        let dir = unique_test_dir("badb64");
        // '!' is not a valid base64 character — must surface as an error, not a
        // silently-truncated frame.
        assert!(write_one_frame(&dir, 0, "not!base64!").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
