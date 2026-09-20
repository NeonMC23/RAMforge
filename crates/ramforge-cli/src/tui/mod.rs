pub mod app;
pub mod diagnostics;
mod render;
mod terminal;

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ramforge_core::GgmlType;
use ramforge_runtime::calibration::{
    CalibrationLimits, CalibrationPlan, CalibrationRunner, CalibrationTaskProgressState,
};
use ramforge_runtime::discovery::StorageDiscovery;
use ramforge_runtime::orchestration::RuntimeOrchestrator;
use ramforge_runtime::plan_persistence::{
    PersistedExecutionPlan, PersistedPlanCompatibilityContext, PlanPersistenceError,
};
use ramforge_runtime::sampling::Sampler;

use app::{AppAction, CalibrationTaskView, Screen, TuiApp, TuiError};
use diagnostics::{build_diagnostic_result, export_run_json, DiagnosticInputs};
use terminal::TerminalSession;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = TerminalSession::new()?;
    let mut app = TuiApp::default();

    while !app.should_quit {
        draw_app(&mut terminal, &app, None)?;
        let command = terminal.read_command(app.accepts_text())?;
        if let Some(action) = app.handle(command) {
            draw_app(&mut terminal, &app, None)?;
            execute_action(&mut app, &mut terminal, action)?;
        }
    }
    Ok(())
}

fn execute_action(
    app: &mut TuiApp,
    terminal: &mut TerminalSession,
    action: AppAction,
) -> io::Result<()> {
    match action {
        AppAction::LoadPlan(path) => {
            app.load_finished(load_plan(&path));
        }
        AppAction::ValidateModelPath(path) => {
            let result = StorageDiscovery
                .discover(&path)
                .map(|_| ())
                .map_err(TuiError::Storage);
            if let Some(next_action) = app.model_path_validation_finished(result) {
                draw_app(terminal, app, None)?;
                execute_action(app, terminal, next_action)?;
            }
        }
        AppAction::Analyze => {
            let result = analyze(app);
            app.analysis_finished(result);
        }
        AppAction::AnalyzeLoadedPlan => {
            let result = analyze_loaded_plan(app);
            app.loaded_analysis_finished(result);
        }
        AppAction::RunCalibration => run_calibration(app, terminal)?,
        AppAction::CreatePlan => {
            let result = create_plan(app);
            app.planning_finished(result);
        }
        AppAction::ValidatePlan => {
            let result = validate_plan(app);
            app.plan_validation_finished(result);
        }
        AppAction::ActivateRuntime => {
            let result = activate_runtime(app);
            app.runtime_activation_finished(result);
        }
        AppAction::GeneratePrompt(prompt, max_tokens) => {
            let result = generate_prompt(app, terminal, &prompt, max_tokens);
            app.generation_finished_diagnostic(result);
        }
        AppAction::SavePlan(path) => {
            let result = save_plan(app, &path);
            app.save_finished(result);
        }
        AppAction::ExportRun(run_id, path) => {
            let result = export_run(app, run_id, &path);
            app.export_finished(result);
        }
    }
    Ok(())
}

fn load_plan(path: &Path) -> Result<PersistedExecutionPlan, TuiError> {
    PersistedExecutionPlan::load(path).map_err(TuiError::Persistence)
}

fn analyze(app: &TuiApp) -> Result<ramforge_runtime::PlanningSession, TuiError> {
    let user = app.user_profile.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution preferences",
        message: "preferences are unavailable".to_string(),
    })?;
    RuntimeOrchestrator
        .analyze(Path::new(&app.model_input), user.ram_budget_bytes)
        .map_err(TuiError::Analysis)
}

fn analyze_loaded_plan(
    app: &TuiApp,
) -> Result<
    (
        ramforge_runtime::PlanningSession,
        ramforge_runtime::planner::ExecutionPlan,
        ramforge_runtime::PlanCompatibilityReport,
    ),
    TuiError,
> {
    let persisted = app.persisted_plan.as_ref().ok_or_else(|| TuiError::Input {
        field: "persisted plan",
        message: "loaded plan is unavailable".to_string(),
    })?;
    let session = RuntimeOrchestrator
        .analyze(Path::new(&app.model_input), persisted.ram_budget_bytes)
        .map_err(TuiError::Analysis)?;
    let compilation = session.compilation_context();
    let report = persisted.compatibility_report(PersistedPlanCompatibilityContext {
        compilation,
        calibration: None,
    });
    let execution_plan = persisted
        .materialize_for_validation(compilation)
        .map_err(TuiError::Persistence)?;
    Ok((session, execution_plan, report))
}

fn run_calibration(app: &mut TuiApp, terminal: &mut TerminalSession) -> io::Result<()> {
    let plan_result = build_calibration_plan(app);
    let plan = match plan_result {
        Ok(plan) => plan,
        Err(error) => {
            app.calibration_finished(Err(error), Vec::new());
            return Ok(());
        }
    };
    app.calibration_started(&plan.tasks);
    let mut task_views = app.calibration_tasks.clone();
    draw_app(terminal, app, Some(&task_views))?;

    let mut draw_error = None;
    let result = match (app.user_profile.as_ref(), app.planning_session.as_ref()) {
        (Some(user), Some(session)) => CalibrationRunner
            .run_with_progress(
                &plan,
                user,
                &session.machine,
                &session.model,
                &session.storage,
                &session.capabilities,
                |progress| {
                    if let Some(view) = task_views.get_mut(progress.task_index) {
                        match progress.state {
                            CalibrationTaskProgressState::Started => view.mark_running(),
                            CalibrationTaskProgressState::Finished => {
                                if let Some(observation) = progress.observation {
                                    view.finish(
                                        observation.status,
                                        observation.value,
                                        render::observation_unit(observation.unit),
                                        observation.status_reason,
                                    );
                                }
                            }
                        }
                    }
                    if draw_error.is_none() {
                        draw_error = draw_app(terminal, app, Some(&task_views)).err();
                    }
                },
            )
            .map_err(TuiError::Calibration),
        _ => Err(TuiError::Input {
            field: "calibration",
            message: "planning analysis or preferences are unavailable".to_string(),
        }),
    };
    if let Some(error) = draw_error {
        return Err(error);
    }
    app.calibration_finished(result, task_views);
    Ok(())
}

fn build_calibration_plan(app: &TuiApp) -> Result<CalibrationPlan, TuiError> {
    let user = app.user_profile.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution preferences",
        message: "preferences are unavailable".to_string(),
    })?;
    let session = app
        .planning_session
        .as_ref()
        .ok_or_else(|| TuiError::Input {
            field: "analysis",
            message: "planning analysis is unavailable".to_string(),
        })?;
    CalibrationPlan::build(
        user,
        &session.machine,
        &session.model,
        &session.storage,
        &session.capabilities,
        CalibrationLimits::conservative(user),
    )
    .map_err(TuiError::Calibration)
}

fn create_plan(app: &TuiApp) -> Result<ramforge_runtime::planner::ExecutionPlan, TuiError> {
    let user = app.user_profile.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution preferences",
        message: "preferences are unavailable".to_string(),
    })?;
    let session = app
        .planning_session
        .as_ref()
        .ok_or_else(|| TuiError::Input {
            field: "analysis",
            message: "planning analysis is unavailable".to_string(),
        })?;
    session
        .create_plan(user, app.calibration_result.as_ref())
        .map_err(TuiError::Planning)
}

fn validate_plan(app: &TuiApp) -> Result<(), TuiError> {
    let session = app
        .planning_session
        .as_ref()
        .ok_or_else(|| TuiError::Input {
            field: "analysis",
            message: "planning analysis is unavailable".to_string(),
        })?;
    let plan = app.execution_plan.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution plan",
        message: "plan is unavailable".to_string(),
    })?;
    session
        .compile_plan(plan)
        .map(|_| ())
        .map_err(TuiError::Compilation)
}

fn activate_runtime(
    app: &mut TuiApp,
) -> Result<ramforge_runtime::inference::InferenceEngine, TuiError> {
    let plan = app.execution_plan.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution plan",
        message: "plan is unavailable".to_string(),
    })?;
    let session = app
        .planning_session
        .as_mut()
        .ok_or_else(|| TuiError::Input {
            field: "analysis",
            message: "planning analysis is unavailable".to_string(),
        })?;
    session.activate_plan(plan).map_err(|error| match error {
        error @ ramforge_runtime::orchestration::OrchestrationError::PlanCompilation(_) => {
            TuiError::Compilation(error)
        }
        error => TuiError::RuntimeConstruction(error),
    })
}

fn generate_prompt(
    app: &mut TuiApp,
    _terminal: &mut TerminalSession,
    prompt: &str,
    max_tokens: usize,
) -> Result<(usize, String, diagnostics::DiagnosticResult), TuiError> {
    // Pull out the data we need from app before borrowing runtime.
    let prompt_token_count;
    let calibration_label;
    let calibration_identifier;
    let calibration_applied_ids;
    let calibration_consumed;
    let tensor_type_counts: BTreeMap<String, (u64, u64)>;
    {
        let runtime = app.active_runtime.as_mut().ok_or_else(|| TuiError::Input {
            field: "runtime",
            message: "validated runtime is not active".to_string(),
        })?;
        runtime.set_profiling(true);
        prompt_token_count = runtime.tokenizer.encode(prompt, true).len();
        tensor_type_counts =
            runtime
                .data_source
                .model()
                .tensors
                .iter()
                .fold(BTreeMap::new(), |mut acc, t| {
                    let label = format!("{:?}", t.ggml_type);
                    let e = acc.entry(label).or_insert((0, 0));
                    e.0 += 1;
                    if let Some(b) = t.byte_length {
                        e.1 += b;
                    }
                    acc
                });
    }

    let prompt_atom = app.lab.live_prompt_tokens.clone();
    let gen_atom = app.lab.live_generated_tokens.clone();
    let phase_atom = app.lab.live_phase.clone();
    prompt_atom.store(
        prompt_token_count as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    gen_atom.store(0, std::sync::atomic::Ordering::Relaxed);
    phase_atom.store(0, std::sync::atomic::Ordering::Relaxed);

    calibration_label = app::calibration_level_label(app.calibration_level()).to_string();
    let cal = app.calibration_result.as_ref().map(|c| {
        let applied = app
            .execution_plan
            .as_ref()
            .map(|p| p.calibration.observation_identifiers.clone())
            .unwrap_or_default();
        (
            Some(c.identifier.clone()),
            applied,
            app.execution_plan
                .as_ref()
                .map(|p| p.calibration.calibration_identifier.is_some())
                .unwrap_or(false),
        )
    });
    (
        calibration_identifier,
        calibration_applied_ids,
        calibration_consumed,
    ) = cal.unwrap_or((None, Vec::new(), false));

    let sampler = Sampler::greedy();
    let started = Instant::now();
    let result;
    let profile;
    let memory_report;
    let runtime_config;
    let generated_token_count_pre;
    {
        let runtime = app.active_runtime.as_mut().ok_or_else(|| TuiError::Input {
            field: "runtime",
            message: "validated runtime is not active".to_string(),
        })?;
        emit_generation_start_diagnostic(runtime, prompt_token_count, max_tokens);
        // Simple local counter for live callback (no runtime access in closure).
        let mut emitted_tokens: u64 = 0;
        let pa = phase_atom.clone();
        let ga = gen_atom.clone();
        result = runtime.generate_with_callback(prompt, max_tokens, &sampler, |text| {
            // Callback receives text; we do not need to retain generated_text here since
            // generate_with_callback returns the full text.
            let _ = text;
            pa.store(1, std::sync::atomic::Ordering::Relaxed);
            emitted_tokens = emitted_tokens.saturating_add(1);
            ga.store(emitted_tokens, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        });
        let elapsed = started.elapsed();
        profile = runtime.generation_profile();
        memory_report = runtime.memory_report();
        runtime_config = runtime.runtime_config().clone();
        generated_token_count_pre = result
            .as_ref()
            .map(|(tokens, _)| tokens.len())
            .unwrap_or(profile.runtime.tokens as usize);
        emit_generation_summary(
            runtime,
            &profile,
            prompt_token_count,
            generated_token_count_pre,
            elapsed,
        );
    }
    let elapsed = started.elapsed();
    let generated_token_count = generated_token_count_pre;

    let diag = build_diagnostic_result(DiagnosticInputs {
        runtime_config: &runtime_config,
        profile: &profile,
        memory_report: &memory_report,
        tensor_type_counts,
        wall_elapsed: elapsed,
        prompt_tokens: prompt_token_count,
        generated_tokens: generated_token_count,
        max_tokens,
        calibration_level_label: calibration_label,
        calibration_identifier,
        calibration_applied_ids,
        calibration_consumed,
    });

    phase_atom.store(2, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok((_tokens, generated_text)) => Ok((prompt_token_count, generated_text, diag)),
        Err(e) => Err(TuiError::Generation(e)),
    }
}

fn export_run(app: &TuiApp, run_id: u64, path: &Path) -> Result<PathBuf, TuiError> {
    let record = app
        .lab
        .run_history
        .iter()
        .find(|r| r.run_id == run_id)
        .ok_or_else(|| TuiError::Input {
            field: "run",
            message: format!("run id {run_id} not found in session history"),
        })?;
    export_run_json(record, path).map_err(|error| TuiError::Generation(error.to_string()))?;
    Ok(path.to_path_buf())
}

fn emit_generation_start_diagnostic(
    runtime: &ramforge_runtime::inference::InferenceEngine,
    prompt_token_count: usize,
    max_tokens: usize,
) {
    let config = runtime.runtime_config();
    let q4_0_tensors = runtime
        .data_source
        .model()
        .tensors
        .iter()
        .filter(|tensor| tensor.ggml_type == GgmlType::Q4_0)
        .count();
    let mut stderr = io::stderr().lock();
    let _ = write!(
        stderr,
        "\r\nRAMforge generation diagnostic (start)\r\n\
         prompt_tokens={} max_tokens={} cpu_threads={} q4_0_tensors={} \
         cache_enabled={} cache_capacity_bytes={} read_coalescing={} grouped_buffer_reuse={}\r\n",
        prompt_token_count,
        max_tokens,
        runtime.backend.num_threads,
        q4_0_tensors,
        config.layer_cache_enabled,
        runtime.model.layer_cache_capacity_bytes(),
        config.read_coalescing_enabled,
        config.grouped_read_buffer_reuse_enabled,
    );
    let _ = stderr.flush();
}

fn emit_generation_summary(
    runtime: &ramforge_runtime::inference::InferenceEngine,
    profile: &ramforge_runtime::inference::GenerationProfile,
    prompt_token_count: usize,
    generated_token_count: usize,
    elapsed: Duration,
) {
    let runtime_profile = &profile.runtime;
    let elapsed_seconds = elapsed.as_secs_f64();
    let tokens_per_second = if elapsed_seconds > 0.0 {
        generated_token_count as f64 / elapsed_seconds
    } else {
        0.0
    };
    let mut stderr = io::stderr().lock();
    let _ = write!(
        stderr,
        "\r\nRAMforge generation diagnostic (summary)\r\n\
         prompt_tokens={} generated_tokens={} elapsed_seconds={:.3} tokens_per_second={:.6} cpu_threads={}\r\n\
         cache_capacity_bytes={} peak_cached_layers={} model_layers={} peak_cache_bytes={} cache_hits={} cache_misses={} cache_evictions={}\r\n\
         logical_tensor_reads={} physical_reads={} physical_bytes={} coalesced_ranges={} read_seconds={:.3}\r\n\
         layer_loads={} layer_releases={} prompt_forwards={} decode_forwards={} terminal_forwards_skipped={}\r\n\
         profile_total_seconds={:.3} prompt_seconds={:.3} layer_load_seconds={:.3} layer_compute_seconds={:.3} logits_seconds={:.3}\r\n\
         quantized_matvec_seconds={:.3} float_matvec_seconds={:.3} dequantization_seconds={:.3} grouped_quantized_copies={} grouped_quantized_copy_seconds={:.3}\r\n\
         peak_managed_bytes={} budget_bytes={}\r\n",
        prompt_token_count,
        generated_token_count,
        elapsed_seconds,
        tokens_per_second,
        runtime.backend.num_threads,
        profile.layer_cache_capacity_bytes,
        runtime_profile.peak_cached_layer_count,
        runtime.model.config.block_count,
        runtime_profile.peak_cache_bytes,
        runtime_profile.cache_hits,
        runtime_profile.cache_misses,
        runtime_profile.cache_evictions,
        profile.io.logical_tensor_reads,
        profile.io.read_operations,
        profile.io.bytes_read,
        profile.io.coalesced_ranges,
        profile.io.elapsed.as_secs_f64(),
        runtime_profile.layer_loads,
        runtime_profile.layer_releases,
        runtime_profile.prompt_forwards,
        runtime_profile.decode_forwards,
        runtime_profile.terminal_forwards_skipped,
        runtime_profile.total.as_secs_f64(),
        runtime_profile.prompt.as_secs_f64(),
        runtime_profile.layer_load.as_secs_f64(),
        runtime_profile.layer_compute.as_secs_f64(),
        runtime_profile.logits.as_secs_f64(),
        runtime_profile.quantized_matvec.as_secs_f64(),
        runtime_profile.float_matvec.as_secs_f64(),
        runtime_profile.dequantization.as_secs_f64(),
        runtime_profile.grouped_quantized_copy_count,
        runtime_profile.grouped_quantized_copy_time.as_secs_f64(),
        profile.ramforge_peak_bytes,
        profile.ramforge_budget_bytes,
    );
    let _ = stderr.flush();
}

fn save_plan(app: &TuiApp, path: &Path) -> Result<PathBuf, TuiError> {
    let plan = app.execution_plan.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution plan",
        message: "plan is unavailable".to_string(),
    })?;
    let persisted =
        PersistedExecutionPlan::from_execution_plan(plan).map_err(TuiError::Persistence)?;
    match persisted.save_new(path) {
        Ok(()) => Ok(path.to_path_buf()),
        Err(PlanPersistenceError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists => {
            Err(TuiError::DestinationExists(path.to_path_buf()))
        }
        Err(error) => Err(TuiError::Persistence(error)),
    }
}

fn draw_app(
    terminal: &mut TerminalSession,
    app: &TuiApp,
    progress_override: Option<&[CalibrationTaskView]>,
) -> io::Result<()> {
    let runtime_config = if app.screen == Screen::PlanValid {
        app.planning_session
            .as_ref()
            .zip(app.execution_plan.as_ref())
            .and_then(|(session, plan)| session.compile_plan(plan).ok())
    } else {
        None
    };
    terminal.draw(&render::render(
        app,
        runtime_config.as_ref(),
        progress_override,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_plan_file_preserves_persistence_error() {
        let path =
            std::env::temp_dir().join(format!("ramforge-missing-plan-{}.rfp", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            load_plan(&path),
            Err(TuiError::Persistence(PlanPersistenceError::Io(ref error)))
                if error.kind() == io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn save_plan_requires_an_execution_plan() {
        let app = TuiApp::default();
        let error = save_plan(&app, Path::new("/tmp/unused.rfp")).unwrap_err();
        assert!(matches!(error, TuiError::Input { .. }));
    }
}
