pub mod app;
mod render;
mod terminal;

use std::io;
use std::path::{Path, PathBuf};

use ramforge_runtime::calibration::{
    CalibrationLimits, CalibrationPlan, CalibrationRunner, CalibrationTaskProgressState,
};
use ramforge_runtime::discovery::StorageDiscovery;
use ramforge_runtime::orchestration::RuntimeOrchestrator;
use ramforge_runtime::plan_persistence::{
    PersistedExecutionPlan, PersistedPlanCompatibilityContext, PlanPersistenceError,
};

use app::{AppAction, CalibrationTaskView, Screen, TuiApp, TuiError};
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
        AppAction::SavePlan(path) => {
            let result = save_plan(app, &path);
            app.save_finished(result);
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

fn save_plan(app: &TuiApp, path: &Path) -> Result<PathBuf, TuiError> {
    let plan = app.execution_plan.as_ref().ok_or_else(|| TuiError::Input {
        field: "execution plan",
        message: "plan is unavailable".to_string(),
    })?;
    let persisted = PersistedExecutionPlan::from_execution_plan(plan)
        .map_err(TuiError::Persistence)?;
    match persisted.save_new(path) {
        Ok(()) => Ok(path.to_path_buf()),
        Err(PlanPersistenceError::Io(error))
            if error.kind() == io::ErrorKind::AlreadyExists =>
        {
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
        let path = std::env::temp_dir().join(format!(
            "ramforge-missing-plan-{}.rfp",
            std::process::id()
        ));
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
