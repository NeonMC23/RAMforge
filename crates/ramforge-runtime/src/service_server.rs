//! Minimal local JSON-lines transport over the inference service.
//!
//! This is intentionally transport-independent in its core: the same
//! `InferenceService` can be driven by a TUI, GUI, or a future HTTP adapter.
//! The stdio adapter exists so the service contract is executable today without
//! adding a network dependency or an unvalidated public server.

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use serde::Deserialize;

use crate::inference_service::{GenerationOptions, InferenceService, ServiceError, ServiceState};
use crate::machine_profile::{collect_machine_profile, MachineProfileConfig};
use crate::memory_hierarchy::{
    build_memory_hierarchy, build_placement_model_for_plan, evaluate_hierarchy_feasibility,
};
use crate::model_requirements::{model_requirements, ModelRequirementConfig};

#[derive(Debug, Deserialize)]
struct LineRequest {
    #[serde(default)]
    id: Option<u64>,
    command: String,
    #[serde(default)]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
}

fn default_max_tokens() -> u32 {
    32
}

#[derive(Debug)]
pub enum ServiceServerError {
    Io(String),
    Service(ServiceError),
}

impl std::fmt::Display for ServiceServerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "service transport I/O: {message}"),
            Self::Service(error) => write!(formatter, "service: {error}"),
        }
    }
}

impl std::error::Error for ServiceServerError {}

/// Serve JSON-lines over stdin/stdout. The caller owns the loaded service.
///
/// A client sends one JSON object per line. Generation streams `token` events
/// followed by `done` or `error`. A line containing exactly
/// `{"command":"cancel"}` sets the shared cooperative cancellation flag, which
/// is checked between token decodes.
pub fn serve_stdio(mut service: InferenceService) -> Result<(), ServiceServerError> {
    let cancel = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel::<String>();
    let reader_cancel = Arc::clone(&cancel);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if line.trim() == r#"{"command":"cancel"}"# || line.trim() == "cancel" {
                reader_cancel.store(true, Ordering::Relaxed);
            }
            if sender.send(line).is_err() {
                break;
            }
        }
    });

    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    for line in receiver {
        if line.trim().is_empty() {
            continue;
        }
        let request: LineRequest = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                write_event(
                    &mut output,
                    serde_json::json!({
                        "id": serde_json::Value::Null,
                        "event": "error",
                        "kind": "invalid_request",
                        "error": error.to_string(),
                    }),
                )?;
                continue;
            }
        };
        let id = request.id.unwrap_or(0);
        match request.command.as_str() {
            "capabilities" => write_event(
                &mut output,
                serde_json::json!({
                    "id": id,
                    "event": "result",
                    "capabilities": service.capabilities(),
                }),
            )?,
            "status" => write_event(
                &mut output,
                serde_json::json!({
                    "id": id,
                    "event": "result",
                    "status": service.status(),
                }),
            )?,
            "memory" => match service.memory_observability() {
                Ok(observability) => write_event(
                    &mut output,
                    serde_json::json!({
                        "id": id,
                        "event": "result",
                        "gpu_observability": observability,
                    }),
                )?,
                Err(error) => write_event(
                    &mut output,
                    serde_json::json!({
                        "id": id,
                        "event": "error",
                        "kind": error_kind(&error),
                        "error": error.to_string(),
                    }),
                )?,
            },
            "architecture" => write_event(
                &mut output,
                serde_json::json!({
                    "id": id,
                    "event": "result",
                    "architecture": service.plan().architecture,
                    "provenance": "GGUF metadata captured in the resolved runtime plan",
                }),
            )?,
            "hierarchy" => {
                let machine = collect_machine_profile(&MachineProfileConfig::default());
                let hierarchy = build_memory_hierarchy(&machine);
                let plan = service.plan();
                let placement = build_placement_model_for_plan(plan, None);
                let (feasibility, requirements_note) =
                    match ramforge_core::discover_model_set(Path::new(&plan.model_path)) {
                        Ok(set) => {
                            let requirements = model_requirements(
                                &set.merged_model(),
                                ModelRequirementConfig {
                                    context_tokens: plan.effective.context_tokens,
                                    batch_tokens: plan.effective.batch_tokens,
                                    kv_bytes_per_element: 2,
                                },
                            );
                            (
                                Some(evaluate_hierarchy_feasibility(
                                    &machine,
                                    &requirements,
                                    plan,
                                )),
                                None,
                            )
                        }
                        Err(error) => (
                            None,
                            Some(format!("model requirements unavailable: {error}")),
                        ),
                    };
                write_event(
                    &mut output,
                    serde_json::json!({
                        "id": id,
                        "event": "result",
                        "memory_hierarchy": hierarchy,
                        "placement": placement,
                        "feasibility": feasibility,
                        "requirements_note": requirements_note,
                        "note": "fresh OS observation; not a snapshot captured at model load and does not include the original RAM budget; KV uses an f16 assumption",
                    }),
                )?;
            }
            "load" => {
                let result = service.load().and_then(|_| service.create_session());
                write_service_result(&mut output, id, result, &service)?;
            }
            "session" => {
                let result = service.create_session();
                write_service_result(&mut output, id, result, &service)?;
            }
            "unload" => {
                service.unload();
                write_event(
                    &mut output,
                    serde_json::json!({
                        "id": id,
                        "event": "result",
                        "status": service.status(),
                    }),
                )?;
            }
            "generate" => {
                cancel.store(false, Ordering::Relaxed);
                let options = GenerationOptions {
                    prompt: request.prompt,
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                };
                let mut sink_error = None;
                let result = service.generate(&options, &cancel, |piece| {
                    if let Err(error) = write_event(
                        &mut output,
                        serde_json::json!({
                            "id": id,
                            "event": "token",
                            "text": piece,
                        }),
                    ) {
                        sink_error = Some(error);
                        return Err("transport closed".to_string());
                    }
                    Ok(())
                });
                match result {
                    Ok(report) => write_event(
                        &mut output,
                        serde_json::json!({
                            "id": id,
                            "event": "done",
                            "text": report.text,
                            "token_ids": report.token_ids,
                            "stats": report.stats,
                            "effective_context_tokens": report.effective_context_tokens,
                            "effective_batch_tokens": report.effective_batch_tokens,
                            "effective_ubatch_tokens": report.effective_ubatch_tokens,
                            "effective_threads": report.effective_threads,
                            "effective_sequence_max": report.effective_sequence_max,
                        }),
                    )?,
                    Err(ServiceError::Cancelled) => write_event(
                        &mut output,
                        serde_json::json!({
                            "id": id,
                            "event": "cancelled",
                        }),
                    )?,
                    Err(ServiceError::Sink(_)) => {
                        if let Some(error) = sink_error {
                            return Err(ServiceServerError::Io(error.to_string()));
                        }
                        write_event(
                            &mut output,
                            serde_json::json!({
                                "id": id,
                                "event": "error",
                                "kind": "transport",
                            }),
                        )?;
                    }
                    Err(error) => write_event(
                        &mut output,
                        serde_json::json!({
                            "id": id,
                            "event": "error",
                            "kind": error_kind(&error),
                            "error": error.to_string(),
                        }),
                    )?,
                }
            }
            "cancel" => {
                cancel.store(true, Ordering::Relaxed);
                write_event(
                    &mut output,
                    serde_json::json!({
                        "id": id,
                        "event": "result",
                        "cancelled": true,
                    }),
                )?;
            }
            _ => write_event(
                &mut output,
                serde_json::json!({
                    "id": id,
                    "event": "error",
                    "kind": "unknown_command",
                    "error": format!("unknown command '{}'", request.command),
                }),
            )?,
        }
    }
    Ok(())
}

fn write_service_result<W: Write>(
    output: &mut W,
    id: u64,
    result: Result<(), ServiceError>,
    service: &InferenceService,
) -> Result<(), ServiceServerError> {
    match result {
        Ok(()) => write_event(
            output,
            serde_json::json!({
                "id": id,
                "event": "result",
                "status": service.status(),
            }),
        ),
        Err(error) => write_event(
            output,
            serde_json::json!({
                "id": id,
                "event": "error",
                "kind": error_kind(&error),
                "error": error.to_string(),
                "state": service.state(),
            }),
        ),
    }
}

fn write_event<W: Write>(
    output: &mut W,
    value: serde_json::Value,
) -> Result<(), ServiceServerError> {
    serde_json::to_writer(&mut *output, &value)
        .map_err(|error| ServiceServerError::Io(error.to_string()))?;
    output
        .write_all(b"\n")
        .map_err(|error| ServiceServerError::Io(error.to_string()))?;
    output
        .flush()
        .map_err(|error| ServiceServerError::Io(error.to_string()))
}

fn error_kind(error: &ServiceError) -> &'static str {
    match error {
        ServiceError::InvalidState { .. } => "invalid_state",
        ServiceError::Configuration(_) => "configuration",
        ServiceError::Model(_) => "model",
        ServiceError::Tokenizer(_) => "tokenizer",
        ServiceError::Context(_) => "context",
        ServiceError::Execution(_) => "execution",
        ServiceError::UnsupportedExecution(_) => "unsupported_execution",
        ServiceError::Replacement { .. } => "replacement",
        ServiceError::Cancelled => "cancelled",
        ServiceError::Sink(_) => "transport",
    }
}

/// True when the service has a reusable context in a testable state.
pub fn is_ready(service: &InferenceService) -> bool {
    service.state() == ServiceState::Ready
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults_are_bounded() {
        let request: LineRequest =
            serde_json::from_str(r#"{"command":"generate","prompt":"x"}"#).unwrap();
        assert_eq!(request.max_tokens, 32);
        assert_eq!(request.temperature, 0.0);
    }

    #[test]
    fn error_kinds_are_explicit() {
        assert_eq!(error_kind(&ServiceError::Cancelled), "cancelled");
        assert_eq!(
            error_kind(&ServiceError::Execution("x".to_string())),
            "execution"
        );
    }
}
