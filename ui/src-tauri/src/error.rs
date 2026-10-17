use serde::Serialize;
use thiserror::Error;

/*
 * Single error type for all Tauri commands.
 *
 * Tauri requires return-error types to be `Serialize` so they cross the IPC
 * boundary as JSON. `thiserror` gives us `Display` for logging; the manual
 * `Serialize` impl emits `{ "code": "...", "message": "..." }`.
 *
 * The code is load-bearing: the frontend has to tell "not published yet"
 * (retry) from "malformed" (stop), and that is not recoverable from prose.
 * Callers must branch on `code`, never on message text.
 */

#[derive(Error, Debug)]
pub enum AppError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("path not found: {0}")]
    NotFound(String),

    #[error("invalid argument: {0}")]
    Invalid(String),

    #[error("subprocess failed: {0}")]
    Subprocess(String),

    #[error("run not found: {0}")]
    UnknownRun(String),

    #[error("background task failed: {0}")]
    Task(String),
}

impl AppError {
    /// Stable wire discriminant. Renaming one of these is a breaking IPC
    /// change — `ui/src/lib/platform/appInvoke.ts` matches on these exact
    /// strings.
    pub fn code(&self) -> &'static str {
        match self {
            AppError::Io(_) => "io",
            AppError::NotFound(_) => "not_found",
            AppError::Invalid(_) => "invalid",
            AppError::Subprocess(_) => "subprocess",
            AppError::UnknownRun(_) => "unknown_run",
            AppError::Task(_) => "task",
        }
    }
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("AppError", 2)?;
        s.serialize_field("code", self.code())?;
        s.serialize_field("message", &self.to_string())?;
        s.end()
    }
}

pub type AppResult<T> = Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_serializes_code_and_message() {
        let cases: Vec<(AppError, &str)> = vec![
            (AppError::Io(std::io::Error::other("disk")), "io"),
            (AppError::NotFound("/x".into()), "not_found"),
            (AppError::Invalid("bad".into()), "invalid"),
            (AppError::Subprocess("boom".into()), "subprocess"),
            (AppError::UnknownRun("run-1".into()), "unknown_run"),
            (AppError::Task("joined".into()), "task"),
        ];
        for (error, expected_code) in cases {
            let display = error.to_string();
            let json = serde_json::to_value(&error).expect("serialize");
            assert_eq!(json["code"], expected_code);
            assert_eq!(json["message"], display);
        }
    }

    /// The frontend retries `io`/`not_found` and stops on the rest, so a
    /// mis-coded variant silently turns a permanent failure into a hot loop.
    #[test]
    fn transient_and_permanent_variants_do_not_share_codes() {
        assert_eq!(AppError::NotFound(String::new()).code(), "not_found");
        assert_eq!(AppError::Invalid(String::new()).code(), "invalid");
        assert_ne!(
            AppError::Invalid(String::new()).code(),
            AppError::NotFound(String::new()).code()
        );
    }
}
