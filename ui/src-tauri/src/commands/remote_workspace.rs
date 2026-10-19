use crate::error::{AppError, AppResult};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{atomic::{AtomicBool, Ordering}, mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri_plugin_shell::ShellExt;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Endpoint {
    id: String,
    name: String,
    host: String,
    port: u16,
    path: String,
}
impl Endpoint {
    fn validate(&self) -> AppResult<()> {
        if self.id.trim().is_empty() || self.id.len() > 128 || self.name.trim().is_empty() || self.name.len() > 256
            || self.id.chars().chain(self.name.chars()).any(char::is_control)
            || self.host.is_empty() || self.host.len() > 255 || !self.host.as_bytes()[0].is_ascii_alphanumeric()
            || !self.host.bytes().all(|b| b.is_ascii_alphanumeric() || b"._@:[]-".contains(&b)) || self.port == 0
            || !self.path.starts_with('/') || self.path.starts_with("//") || self.path.len() > 1024
            || !self.path.bytes().all(|b| b.is_ascii_alphanumeric() || b"/_.-".contains(&b))
            || self.path.split('/').any(|p| p == "." || p == "..") {
            return Err(AppError::Invalid("Invalid remote workspace host, port or URL path".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceStatus {
    state: String,
    endpoint: Option<Endpoint>,
    url: Option<String>,
    error: Option<String>,
}
struct Tunnel { child: Child, url: String }
impl Drop for Tunnel {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}
struct Workspace {
    state: &'static str,
    endpoint: Option<Endpoint>,
    tunnel: Option<Tunnel>,
    cancel: Arc<AtomicBool>,
    error: Option<String>,
}
static WORKSPACE: Lazy<Mutex<Workspace>> = Lazy::new(|| Mutex::new(Workspace {
    state: "disconnected", endpoint: None, tunnel: None, cancel: Arc::new(AtomicBool::new(false)), error: None,
}));
impl Workspace {
    fn snapshot(&mut self) -> WorkspaceStatus {
        if let Some(tunnel) = self.tunnel.as_mut() {
            if !matches!(tunnel.child.try_wait(), Ok(None)) {
                self.tunnel.take(); self.state = "failed";
                self.error = Some("SSH disconnected. Reconnect to resume viewing the remote desktop; its jobs are managed on the node.".into());
            }
        }
        WorkspaceStatus { state: self.state.into(), endpoint: self.endpoint.clone(),
            url: self.tunnel.as_ref().map(|t| t.url.clone()), error: self.error.clone() }
    }
}
fn locked() -> AppResult<std::sync::MutexGuard<'static, Workspace>> {
    WORKSPACE.lock().map_err(|_| AppError::Task("Remote workspace state unavailable".into()))
}

fn start_tunnel(endpoint: &Endpoint, ssh: &Path, cancel: &AtomicBool) -> AppResult<Tunnel> {
    endpoint.validate()?;
    let reservation = TcpListener::bind(("127.0.0.1", 0))?;
    let local_port = reservation.local_addr()?.port();
    let script = format!("import http.client,sys; c=http.client.HTTPConnection('127.0.0.1',{},timeout=8); c.request('GET','{}'); r=c.getresponse(); ok=200<=r.status<400 or r.status in (401,403); c.close(); ok or sys.exit('Remote workspace URL did not respond successfully'); print('NIMBUS_WORKSPACE_READY',flush=True); sys.stdin.buffer.read()", endpoint.port, endpoint.path);
    let remote = format!("exec python3 -u -c '{}'", script.replace('\'', "'\\''"));
    let mut command = Command::new(ssh);
    command.args(["-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        "-o", "ExitOnForwardFailure=yes", "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ControlPersist=no", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
        "-o", "PermitLocalCommand=no", "-o", "RemoteCommand=none", "-L"])
        .arg(format!("127.0.0.1:{local_port}:127.0.0.1:{}", endpoint.port))
        .arg("--").arg(&endpoint.host).arg(remote)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    drop(reservation);
    let child = command.spawn().map_err(|e| AppError::Subprocess(format!("Start SSH: {e}")))?;
    let mut tunnel = Tunnel { child, url: format!("http://127.0.0.1:{local_port}{}", endpoint.path) };
    let output = tunnel.child.stdout.take().unwrap();
    let error = tunnel.child.stderr.take().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(output.take(4096)).read_line(&mut line);
        let _ = ready_tx.send(result.is_ok() && line.trim() == "NIMBUS_WORKSPACE_READY");
    });
    let (error_tx, error_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = error; let mut saved = Vec::new(); let mut chunk = [0; 1024];
        while let Ok(n) = reader.read(&mut chunk) {
            if n == 0 { break; }
            let keep = n.min(4096_usize.saturating_sub(saved.len())); saved.extend_from_slice(&chunk[..keep]);
        }
        let _ = error_tx.send(String::from_utf8_lossy(&saved).to_string());
    });
    let deadline = Instant::now() + Duration::from_secs(25);
    loop {
        if cancel.load(Ordering::Acquire) { return Err(AppError::Task("Remote workspace connection cancelled".into())); }
        if tunnel.child.try_wait()?.is_some() {
            let detail = error_rx.recv_timeout(Duration::from_millis(100)).unwrap_or_default();
            return Err(AppError::Subprocess(format!("SSH workspace connection failed. Check the SSH alias, host key, credentials and remote desktop service. {detail}")));
        }
        if let Ok(true) = ready_rx.try_recv() { return Ok(tunnel); }
        if Instant::now() >= deadline { return Err(AppError::Subprocess("SSH workspace connection timed out".into())); }
        thread::sleep(Duration::from_millis(20));
    }
}

#[tauri::command]
pub fn connect_remote_workspace(endpoint: Endpoint) -> AppResult<WorkspaceStatus> {
    endpoint.validate()?;
    let cancel = {
        let mut state = locked()?;
        state.snapshot();
        if state.state == "connecting" || state.state == "connected" {
            return Err(AppError::Invalid("Disconnect the current workspace before connecting another".into()));
        }
        state.cancel = Arc::new(AtomicBool::new(false)); state.state = "connecting";
        state.endpoint = Some(endpoint.clone()); state.error = None;
        state.cancel.clone()
    };
    let initial = locked()?.snapshot();
    tauri::async_runtime::spawn_blocking(move || -> AppResult<WorkspaceStatus> {
        let result = start_tunnel(&endpoint, Path::new("ssh"), &cancel);
        let mut state = locked()?;
        if !Arc::ptr_eq(&state.cancel, &cancel) || cancel.load(Ordering::Acquire) { return Ok(state.snapshot()); }
        match result {
            Ok(tunnel) => { state.tunnel = Some(tunnel); state.state = "connected"; }
            Err(error) => { state.state = "failed"; state.error = Some(error.to_string()); }
        }
        Ok(state.snapshot())
    });
    Ok(initial)
}

#[tauri::command]
pub fn remote_workspace_status() -> AppResult<WorkspaceStatus> { Ok(locked()?.snapshot()) }
#[tauri::command]
pub async fn disconnect_remote_workspace() -> AppResult<WorkspaceStatus> {
    tauri::async_runtime::spawn_blocking(|| {
        let tunnel = { let mut state = locked()?;
            state.cancel.store(true, Ordering::Release); state.state = "disconnected"; state.error = None;
            state.endpoint = None; state.tunnel.take() };
        drop(tunnel);
        Ok(locked()?.snapshot())
    }).await.map_err(|e| AppError::Task(e.to_string()))?
}
#[tauri::command]
pub fn open_remote_workspace(app: tauri::AppHandle) -> AppResult<()> {
    let mut state = locked()?;
    let status = state.snapshot();
    let url = status.url.ok_or_else(|| AppError::Invalid("Connect a remote workspace first".into()))?;
    #[allow(deprecated)]
    app.shell().open(url, None).map_err(|e| AppError::Task(e.to_string()))
}
pub fn shutdown() {
    if let Ok(mut state) = WORKSPACE.lock() { state.cancel.store(true, Ordering::Release); state.tunnel.take(); }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpStream;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicU64;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn endpoint(port: u16) -> Endpoint {
        Endpoint { id: "node".into(), name: "Research desktop".into(), host: "user@node".into(), port, path: "/vnc.html".into() }
    }
    fn fixture() -> (std::path::PathBuf, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("nimbus-workspace-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir(&root).unwrap();
        let ssh = root.join("ssh-fixture");
        std::fs::write(&ssh, r#"#!/usr/bin/env python3
import sys, shlex, os
assert sys.argv[-2] == 'user@node'
for option in ['BatchMode=yes','StrictHostKeyChecking=yes','ExitOnForwardFailure=yes','ControlPath=none','ForwardAgent=no']:
    assert option in sys.argv
forward=sys.argv[sys.argv.index('-L')+1].split(':')
assert forward[0]=='127.0.0.1' and forward[2]=='127.0.0.1'
command=shlex.split(sys.argv[-1])
assert command[:4]==['exec','python3','-u','-c']
prefix='''import socket,threading
listener=socket.socket(); listener.bind(('127.0.0.1',PORT)); listener.listen()
def serve():
    while True:
        client,_=listener.accept()
        client.recv(4096); client.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: 5\\r\\nConnection: close\\r\\n\\r\\nscene'); client.close()
threading.Thread(target=serve,daemon=True).start()
'''.replace('PORT',forward[1])
os.execvp('python3',['python3','-u','-c',prefix+command[4]])
"#).unwrap();
        std::fs::set_permissions(&ssh, std::fs::Permissions::from_mode(0o755)).unwrap();
        (root, ssh)
    }
    fn service(status: &str) -> (u16, thread::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let response = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            let mut request = [0; 1024]; let n = stream.read(&mut request).unwrap();
            assert!(String::from_utf8_lossy(&request[..n]).starts_with("GET /vnc.html HTTP/1.1"));
            let _ = stream.write_all(response.as_bytes());
        });
        (port, worker)
    }
    #[test]
    fn workspace_rejects_invalid_endpoints_before_starting_ssh() {
        assert!(endpoint(6080).validate().is_ok());
        for path in ["https://node/", "//node/", "/../x", "/./x", "/vnc.html?password=secret", "/a\r\nHost:x", "/a\\b", "/%2fhost"] {
            let mut e = endpoint(6080); e.path = path.into(); assert!(e.validate().is_err(), "{path}");
        }
        for host in ["-node", "node with spaces", "node\nother", ""] {
            let mut e = endpoint(6080); e.host = host.into(); assert!(e.validate().is_err());
        }
        assert!(endpoint(0).validate().is_err());
    }
    #[test]
    fn workspace_checks_service_and_closes_owned_forward_on_drop() {
        let (root, ssh) = fixture(); let (port, server) = service("200 OK");
        let tunnel = start_tunnel(&endpoint(port), &ssh, &AtomicBool::new(false)).unwrap();
        let address = tunnel.url.trim_start_matches("http://").split('/').next().unwrap().to_string();
        let mut stream = TcpStream::connect(&address).unwrap();
        stream.write_all(b"GET /vnc.html HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut response = String::new(); stream.read_to_string(&mut response).unwrap(); assert!(response.ends_with("scene"));
        drop(tunnel); assert!(TcpStream::connect(&address).is_err());
        server.join().unwrap(); std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn workspace_rejects_missing_service_path_and_accepts_authentication_page() {
        let (root, ssh) = fixture();
        for (status, ok) in [("404 Not Found", false), ("401 Unauthorized", true)] {
            let (port, server) = service(status);
            let result = start_tunnel(&endpoint(port), &ssh, &AtomicBool::new(false));
            assert_eq!(result.is_ok(), ok); drop(result); server.join().unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn workspace_cancel_during_connection_reaps_transport() {
        let (root, ssh) = fixture();
        std::fs::write(&ssh, "#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n").unwrap();
        let started = Instant::now();
        let cancelled = Arc::new(AtomicBool::new(false)); let trigger = cancelled.clone();
        let timer = thread::spawn(move || { thread::sleep(Duration::from_millis(150)); trigger.store(true, Ordering::Release); });
        assert!(start_tunnel(&endpoint(6080), &ssh, &cancelled).is_err());
        assert!(started.elapsed() < Duration::from_secs(3)); timer.join().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn workspace_exit_is_reported_without_claiming_remote_jobs_stopped() {
        let (root, ssh) = fixture(); let (port, server) = service("200 OK");
        let mut tunnel = start_tunnel(&endpoint(port), &ssh, &AtomicBool::new(false)).unwrap();
        tunnel.child.kill().unwrap(); tunnel.child.wait().unwrap();
        let mut state = Workspace { state: "connected", endpoint: Some(endpoint(port)), tunnel: Some(tunnel), cancel: Arc::new(AtomicBool::new(false)), error: None };
        let status = state.snapshot(); assert_eq!(status.state, "failed"); assert!(status.url.is_none());
        assert!(status.error.unwrap().contains("jobs are managed on the node"));
        server.join().unwrap(); std::fs::remove_dir_all(root).unwrap();
    }
}
