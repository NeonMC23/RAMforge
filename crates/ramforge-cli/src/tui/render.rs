use std::fmt::Write as _;

use ramforge_runtime::planner::{
    Availability, ModelExecutionCompatibility, ObservationUnit, PlanCompatibility,
    PlanReasonCode,
};
use ramforge_runtime::runtime_config::RuntimeConfig;

use super::app::{
    calibration_level_for_index, calibration_level_label, mode_label,
    CalibrationTaskDisplayState, CalibrationTaskView, ModelFlow, PlanOrigin, Screen, TuiApp,
};

pub fn render(
    app: &TuiApp,
    runtime_config: Option<&RuntimeConfig>,
    progress_override: Option<&[CalibrationTaskView]>,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "RAMforge");
    let _ = writeln!(
        output,
        "Local GGUF inference with resource-aware execution planning."
    );
    let _ = writeln!(output, "{}", "-".repeat(72));

    match app.screen {
        Screen::Welcome => render_welcome(app, &mut output),
        Screen::LoadPlan => render_load_plan(app, &mut output),
        Screen::ModelInput => render_model_input(app, &mut output),
        Screen::Preferences => render_preferences(app, &mut output),
        Screen::Analyzing => render_analyzing(&mut output),
        Screen::ModelInfo => render_model_info(app, &mut output),
        Screen::CalibrationSelect => render_calibration_select(app, &mut output),
        Screen::Calibrating => render_calibration(
            app,
            progress_override.unwrap_or(&app.calibration_tasks),
            &mut output,
        ),
        Screen::PlanReview => render_plan_review(app, &mut output),
        Screen::PlanValidation => render_plan_validation(&mut output),
        Screen::PlanValid => render_plan_valid(app, runtime_config, &mut output),
        Screen::PlanCompatibility => render_plan_compatibility(app, &mut output),
        Screen::SavePlan => render_save_plan(app, &mut output),
        Screen::Error => render_error(app, &mut output),
    }

    if let Some(error) = app.error.as_ref() {
        if app.screen != Screen::Error {
            let _ = writeln!(output);
            let _ = writeln!(output, "{}", error.title());
            let _ = writeln!(output, "{error}");
        }
    }
    if let Some(message) = app.status_message.as_deref() {
        let _ = writeln!(output);
        let _ = writeln!(output, "{message}");
    }
    let _ = writeln!(output);
    let _ = write!(output, "{}", footer(app));
    output
}

fn render_welcome(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output);
    menu_line(output, app.menu_index == 0, "Select model");
    menu_line(output, app.menu_index == 1, "Load plan");
    menu_line(output, app.menu_index == 2, "Exit");
}

fn render_load_plan(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Load persisted plan");
    let _ = writeln!(output, "Enter a PersistedExecutionPlan path.");
    let _ = writeln!(output);
    let _ = writeln!(output, "> {}_", app.load_path_input);
    let _ = writeln!(output);
    let _ = writeln!(output, "The plan will not be executed or modified.");
}

fn render_model_input(app: &TuiApp, output: &mut String) {
    let _ = writeln!(
        output,
        "{}",
        if app.model_flow == ModelFlow::LoadedPlan {
            "Current model context"
        } else {
            "Model selection"
        }
    );
    let _ = writeln!(
        output,
        "{}",
        if app.model_flow == ModelFlow::LoadedPlan {
            "Enter the GGUF path to validate the loaded plan against."
        } else {
            "Enter an absolute or relative GGUF path."
        }
    );
    let _ = writeln!(output);
    let _ = writeln!(output, "> {}_", app.model_input);
    let _ = writeln!(output);
    let _ = writeln!(output, "No filesystem scan is performed.");
}

fn render_preferences(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Execution preferences");
    let _ = writeln!(output, "Values are passed to UserProfile without clamping.");
    let _ = writeln!(output);
    preference_line(output, app.menu_index == 0, "RAM budget", &app.ram_input);
    preference_line(output, app.menu_index == 1, "Mode", mode_label(app.mode));
    if app.mode == ramforge_runtime::planner::OperatingMode::Advanced {
        preference_line(
            output,
            app.menu_index == 2,
            "CPU threads",
            optional_text(&app.advanced_thread_input),
        );
        preference_line(
            output,
            app.menu_index == 3,
            "Layer cache",
            app.advanced_cache.label(),
        );
        preference_line(
            output,
            app.menu_index == 4,
            "Cache capacity",
            optional_text(&app.advanced_cache_capacity_input),
        );
        preference_line(
            output,
            app.menu_index == 5,
            "Read coalescing",
            app.advanced_read_coalescing.label(),
        );
        preference_line(
            output,
            app.menu_index == 6,
            "Grouped buffer reuse",
            app.advanced_grouped_buffer.label(),
        );
        menu_line(output, app.menu_index == 7, "Analyze model");
    } else {
        menu_line(output, app.menu_index == 2, "Analyze model");
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Modes: Background / Constrained | Balanced / Normal | Maximum Performance | Advanced"
    );
    let _ = writeln!(
        output,
        "Enter cycles mode and existing Advanced override choices."
    );
}

fn render_analyzing(output: &mut String) {
    let _ = writeln!(output);
    let _ = writeln!(output, "Analyzing...");
    let _ = writeln!(
        output,
        "Using RAMforge discovery, GGUF inspection, capability, and planning APIs."
    );
}

fn render_model_info(app: &TuiApp, output: &mut String) {
    let Some(session) = app.planning_session.as_ref() else {
        let _ = writeln!(output, "Model analysis is unavailable.");
        return;
    };
    let model = &session.model;
    let info = &session.model_info;
    let machine = &session.machine;
    let storage = &session.storage;
    let capabilities = &session.capabilities;

    let _ = writeln!(output, "Model information");
    field(output, "Selected path", &app.model_input);
    field(
        output,
        "Resolved path",
        &storage.model_path.display().to_string(),
    );
    field(output, "Name", model.identity.display_name.as_deref().unwrap_or("Unavailable"));
    field(output, "Architecture", model.architecture.as_deref().unwrap_or("Unknown"));
    field(output, "File size", &format_bytes(model.file_size_bytes));
    field(output, "Tensor count", &model.tensor_count.to_string());
    field(output, "Context length", &optional_u64(model.context_size));
    field(output, "Embedding length", &optional_u64(info.embedding_length));
    field(output, "Layer count", &optional_u64(model.layer_count));
    field(
        output,
        "Vocabulary size",
        &info
            .vocab_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "Unavailable".to_string()),
    );
    field(
        output,
        "Execution",
        match model.execution_compatibility {
            ModelExecutionCompatibility::Executable => "Supported",
            ModelExecutionCompatibility::InspectPlanOnly => "Not supported (inspect/plan only)",
            ModelExecutionCompatibility::UnknownArchitecture => "Not supported (unknown architecture)",
        },
    );

    let _ = writeln!(output);
    let _ = writeln!(output, "Tensor distribution");
    for format in &model.tensor_formats {
        let _ = writeln!(
            output,
            "  {:<10} tensors={:<6} bytes={:<12} runtime_supported={}",
            format.format,
            format.tensor_count,
            format
                .byte_count
                .map(format_bytes)
                .unwrap_or_else(|| "Unavailable".to_string()),
            yes_no(format.runtime_supported)
        );
    }

    let _ = writeln!(output);
    let _ = writeln!(output, "Machine and storage");
    field(
        output,
        "CPU",
        machine.cpu_model.as_deref().unwrap_or("Unavailable"),
    );
    field(
        output,
        "Logical CPUs",
        &machine.logical_cpu_cores.to_string(),
    );
    field(
        output,
        "Available RAM",
        &machine
            .available_ram_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "Unavailable".to_string()),
    );
    field(output, "Storage", &format!("{:?}", storage.kind));
    field(
        output,
        "Filesystem",
        storage.filesystem_type.as_deref().unwrap_or("Unknown"),
    );
    availability_line(output, "CPU backend", machine.cpu_backend);
    if machine.gpus.is_empty() {
        let _ = writeln!(output, "  GPU inventory: none reported");
    } else {
        for gpu in &machine.gpus {
            availability_line(output, &format!("GPU {}", gpu.identifier), gpu.backend);
        }
    }

    let _ = writeln!(output);
    let _ = writeln!(output, "Capabilities");
    field(output, "Storage usable", yes_no(capabilities.storage_usable));
    field(output, "CPU backend usable", yes_no(capabilities.cpu_backend_usable));
    field(output, "GPU backend usable", yes_no(capabilities.gpu_backend_usable));
    field(
        output,
        "Model executable",
        yes_no(capabilities.model_execution_compatible),
    );
    field(output, "Layer cache possible", yes_no(capabilities.layer_cache_possible));
    field(
        output,
        "Read coalescing possible",
        yes_no(capabilities.read_coalescing_possible),
    );
    field(
        output,
        "Grouped buffer possible",
        yes_no(capabilities.grouped_read_buffer_reuse_possible),
    );

    let _ = writeln!(output);
    menu_line(output, app.menu_index == 0, "Continue to calibration");
    menu_line(output, app.menu_index == 1, "Change preferences");
}

fn render_calibration_select(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Calibration");
    let _ = writeln!(
        output,
        "Calibration measures this machine/storage/model environment."
    );
    let _ = writeln!(output, "It does not modify Planner rules.");
    let _ = writeln!(output);
    for index in 0..4 {
        menu_line(
            output,
            app.menu_index == index,
            calibration_level_label(calibration_level_for_index(index)),
        );
    }
}

fn render_calibration(
    app: &TuiApp,
    tasks: &[CalibrationTaskView],
    output: &mut String,
) {
    let _ = writeln!(
        output,
        "Calibration: {}",
        calibration_level_label(app.calibration_level())
    );
    let _ = writeln!(output, "Real bounded CalibrationPlan tasks:");
    let _ = writeln!(output);
    for (index, task) in tasks.iter().enumerate() {
        let _ = writeln!(
            output,
            "  {}/{}  {}  [{}]",
            index + 1,
            tasks.len(),
            task.kind,
            task_state_label(task.state)
        );
        let _ = writeln!(output, "       task: {}", task.identifier);
        if let Some(value) = task.value {
            let _ = writeln!(
                output,
                "       result: {} {}",
                value,
                task.unit.as_deref().unwrap_or("")
            );
        }
        if let Some(reason) = task.reason.as_deref() {
            let _ = writeln!(output, "       reason: {reason}");
        }
    }
    if app.calibration_complete {
        let _ = writeln!(output);
        let _ = writeln!(output, "Press Enter to create the plan.");
    }
}

fn render_plan_review(app: &TuiApp, output: &mut String) {
    let Some(plan) = app.execution_plan.as_ref() else {
        let _ = writeln!(output, "Execution plan is unavailable.");
        return;
    };
    render_plan(plan, output);
    let _ = writeln!(output);
    if app.plan_origin == PlanOrigin::LoadedPlan {
        let compatible = app
            .compatibility_report
            .as_ref()
            .is_some_and(|report| report.state != PlanCompatibility::Incompatible);
        menu_line(
            output,
            app.menu_index == 0,
            if compatible {
                "Validate/compile loaded plan"
            } else {
                "Back to compatibility"
            },
        );
        menu_line(output, app.menu_index == 1, "Plan with current model");
    } else {
        menu_line(output, app.menu_index == 0, "Validate plan");
        menu_line(output, app.menu_index == 1, "Change preferences");
    }
}

fn render_plan_validation(output: &mut String) {
    let _ = writeln!(output, "Validating plan...");
    let _ = writeln!(
        output,
        "ExecutionPlan -> PlanCompiler -> RuntimeConfig"
    );
}

fn render_plan_valid(
    app: &TuiApp,
    runtime_config: Option<&RuntimeConfig>,
    output: &mut String,
) {
    let _ = writeln!(output, "Plan valid");
    if let Some(config) = runtime_config {
        let _ = writeln!(output);
        let _ = writeln!(output, "RuntimeConfig");
        field(output, "CPU threads", &config.cpu_thread_count.to_string());
        field(output, "RAM budget", &format_bytes(config.ram_budget_bytes));
        field(
            output,
            "Layer cache",
            if config.layer_cache_enabled {
                "Enabled"
            } else {
                "Disabled"
            },
        );
        field(
            output,
            "Layer cache capacity",
            &format_bytes(config.layer_cache_capacity_bytes),
        );
        field(
            output,
            "Read coalescing",
            enabled_disabled(config.read_coalescing_enabled),
        );
        field(
            output,
            "Grouped buffer reuse",
            enabled_disabled(config.grouped_read_buffer_reuse_enabled),
        );
        field(output, "Execution device", &format!("{:?}", config.execution_device));
    }
    let _ = writeln!(output);
    if app.plan_origin == PlanOrigin::LoadedPlan {
        menu_line(output, true, "Back to compatibility");
    } else {
        menu_line(output, app.menu_index == 0, "Save plan");
        menu_line(output, app.menu_index == 1, "Back to plan");
    }
}

fn render_plan_compatibility(app: &TuiApp, output: &mut String) {
    let Some(report) = app.compatibility_report.as_ref() else {
        let _ = writeln!(output, "Compatibility result is unavailable.");
        return;
    };
    let state = match report.state {
        PlanCompatibility::Valid => "VALID",
        PlanCompatibility::CompatibleRecalibrationRecommended => {
            "COMPATIBLE — RECALIBRATION RECOMMENDED"
        }
        PlanCompatibility::Incompatible => "INCOMPATIBLE",
    };
    let _ = writeln!(output, "Plan compatibility");
    let _ = writeln!(output, "{state}");
    let _ = writeln!(output);

    if let (Some(persisted), Some(session)) = (
        app.persisted_plan.as_ref(),
        app.planning_session.as_ref(),
    ) {
        let _ = writeln!(output, "Source context");
        field(
            output,
            "Model fingerprint",
            &persisted.model_descriptor_fingerprint.to_string(),
        );
        field(
            output,
            "Machine fingerprint",
            &persisted.source_machine_fingerprint.to_string(),
        );
        field(
            output,
            "Storage fingerprint",
            &persisted.source_storage_fingerprint.to_string(),
        );
        let _ = writeln!(output, "Current context");
        field(
            output,
            "Machine fingerprint",
            &session.machine.fingerprint().to_string(),
        );
        field(
            output,
            "Storage fingerprint",
            &session.storage.fingerprint().to_string(),
        );
        field(
            output,
            "CPU",
            session.machine.cpu_model.as_deref().unwrap_or("Unavailable"),
        );
        field(
            output,
            "Logical CPUs",
            &session.machine.logical_cpu_cores.to_string(),
        );
        field(output, "Storage kind", &format!("{:?}", session.storage.kind));
        field(
            output,
            "Model fingerprint",
            &session.model.identity.descriptor_fingerprint.to_string(),
        );
    }
    if !report.reasons.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Compatibility reasons");
        for reason in &report.reasons {
            let _ = writeln!(output, "  {reason}");
        }
    }
    let _ = writeln!(output);
    menu_line(output, app.menu_index == 0, "Review persisted plan");
    menu_line(output, app.menu_index == 1, "Plan with current model");
    menu_line(output, app.menu_index == 2, "Select another model");
}

fn render_save_plan(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Save plan");
    let _ = writeln!(
        output,
        "Enter a new destination. Existing files are not overwritten."
    );
    let _ = writeln!(output);
    let _ = writeln!(output, "> {}_", app.save_path_input);
}

fn render_error(app: &TuiApp, output: &mut String) {
    if let Some(error) = app.error.as_ref() {
        let _ = writeln!(output, "{}", error.title());
        let _ = writeln!(output);
        let _ = writeln!(output, "{error}");
    } else {
        let _ = writeln!(
            output,
            "Internal UI state error: structured error details are unavailable."
        );
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Press Esc or Enter to return.");
}

fn render_plan(plan: &ramforge_runtime::planner::ExecutionPlan, output: &mut String) {
    let _ = writeln!(output, "Execution plan");
    let _ = writeln!(output, "Decisions and existing Planner reasons");
    decision_reason(
        output,
        plan,
        "RAM budget",
        &format_bytes(plan.ram_budget_bytes),
        &[PlanReasonCode::UserRamBudgetPreserved],
    );
    decision_reason(
        output,
        plan,
        "CPU threads",
        &plan.cpu_thread_count.to_string(),
        &[
            PlanReasonCode::ThreadCountSelectedByPreset,
            PlanReasonCode::ThreadCountSelectedByAdvancedOverride,
        ],
    );
    decision_reason(
        output,
        plan,
        "Execution strategy",
        &format!("{:?}", plan.strategy),
        &[
            PlanReasonCode::CpuStrategySelected,
            PlanReasonCode::GpuStrategySelected,
        ],
    );
    decision_reason(
        output,
        plan,
        "Layer cache",
        enabled_disabled(plan.layer_cache.enabled),
        &[
            PlanReasonCode::LayerCacheEnabledWithinStaticCapacity,
            PlanReasonCode::LayerCacheDisabledByPreset,
            PlanReasonCode::LayerCacheDisabledBecauseNoCompleteLayerFits,
        ],
    );
    decision_reason(
        output,
        plan,
        "Read coalescing",
        enabled_disabled(plan.io.read_coalescing_enabled),
        &[
            PlanReasonCode::ReadCoalescingEnabled,
            PlanReasonCode::ReadCoalescingDisabled,
        ],
    );
    decision_reason(
        output,
        plan,
        "Grouped buffer reuse",
        enabled_disabled(plan.io.grouped_read_buffer_reuse_enabled),
        &[
            PlanReasonCode::GroupedReadBufferReuseEnabled,
            PlanReasonCode::GroupedReadBufferReuseDisabled,
        ],
    );
    decision_reason(
        output,
        plan,
        "GPU execution",
        enabled_disabled(plan.gpu.enabled),
        &[
            PlanReasonCode::GpuStrategySelected,
            PlanReasonCode::GpuDisabledByUser,
            PlanReasonCode::GpuUnavailable,
        ],
    );
    decision_reason(
        output,
        plan,
        "Calibration",
        plan.calibration
            .calibration_identifier
            .as_deref()
            .unwrap_or("Not applied"),
        &[
            PlanReasonCode::CalibrationObservationsApplied,
            PlanReasonCode::CalibrationNotRequested,
            PlanReasonCode::CalibrationNotApplicable,
        ],
    );

    let _ = writeln!(output, "Reservations");
    field(
        output,
        "Managed lower bound",
        &format_bytes(plan.reservations.managed_lower_bound_bytes),
    );
    field(
        output,
        "Persistent resident",
        &format_bytes(plan.reservations.persistent_resident_bytes),
    );
    field(
        output,
        "Largest layer peak",
        &format_bytes(plan.reservations.largest_layer_load_peak_bytes),
    );
    field(
        output,
        "Remaining after lower bound",
        &format_bytes(plan.reservations.remaining_budget_after_lower_bound_bytes),
    );
}

fn decision_reason(
    output: &mut String,
    plan: &ramforge_runtime::planner::ExecutionPlan,
    decision: &str,
    value: &str,
    reason_codes: &[PlanReasonCode],
) {
    if let Some(reason) = plan
        .reasons
        .iter()
        .find(|reason| reason_codes.contains(&reason.code))
    {
        match reason.value {
            Some(reason_value) => {
                let _ = writeln!(
                    output,
                    "  {decision}: {value} | Reason: {:?} ({reason_value})",
                    reason.code
                );
            }
            None => {
                let _ = writeln!(
                    output,
                    "  {decision}: {value} | Reason: {:?}",
                    reason.code
                );
            }
        }
    } else {
        let _ = writeln!(output, "  {decision}: {value}");
    }
}

fn menu_line(output: &mut String, selected: bool, label: &str) {
    let _ = writeln!(output, "{} {label}", if selected { ">" } else { " " });
}

fn preference_line(output: &mut String, selected: bool, label: &str, value: &str) {
    let _ = writeln!(
        output,
        "{} {:<24} {}",
        if selected { ">" } else { " " },
        label,
        if value.is_empty() { "<required>" } else { value }
    );
}

fn field(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(output, "  {:<28} {value}", label);
}

fn availability_line(output: &mut String, label: &str, availability: Availability) {
    let _ = writeln!(
        output,
        "  {:<20} Detected={} Supported={} Usable={}",
        label,
        yes_no(availability.detected),
        yes_no(availability.supported),
        yes_no(availability.usable)
    );
}

fn optional_text(value: &str) -> &str {
    if value.is_empty() {
        "Planner default"
    } else {
        value
    }
}

fn optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "Unavailable".to_string())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn enabled_disabled(value: bool) -> &'static str {
    if value {
        "Enabled"
    } else {
        "Disabled"
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "Yes"
    } else {
        "No"
    }
}

fn task_state_label(state: CalibrationTaskDisplayState) -> &'static str {
    match state {
        CalibrationTaskDisplayState::Pending => "Pending",
        CalibrationTaskDisplayState::Running => "Running",
        CalibrationTaskDisplayState::Measured => "Measured",
        CalibrationTaskDisplayState::Unavailable => "Unavailable",
        CalibrationTaskDisplayState::Skipped => "Skipped",
        CalibrationTaskDisplayState::Failed => "Failed",
    }
}

fn footer(app: &TuiApp) -> &'static str {
    match app.screen {
        Screen::LoadPlan | Screen::ModelInput | Screen::SavePlan => {
            "Type path  Enter Submit  Backspace Delete  Esc Back  Ctrl-C Quit"
        }
        Screen::Preferences if app.accepts_text() => {
            "Type value  Up/Down Navigate  Enter Select  Esc Back  Ctrl-C Quit"
        }
        Screen::Analyzing | Screen::PlanValidation => "Please wait  Ctrl-C Quit",
        Screen::Calibrating if !app.calibration_complete => "Calibration running",
        _ => "Up/Down Navigate  Enter Select  Esc Back  q Quit",
    }
}

fn observation_unit_label(unit: ObservationUnit) -> &'static str {
    match unit {
        ObservationUnit::BytesPerSecond => "bytes/s",
        ObservationUnit::ElementsPerSecond => "elements/s",
        ObservationUnit::Nanoseconds => "ns",
    }
}

pub fn observation_unit(unit: ObservationUnit) -> String {
    observation_unit_label(unit).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welcome_render_contains_real_actions_and_navigation() {
        let rendered = render(&TuiApp::default(), None, None);
        assert!(rendered.contains("Select model"));
        assert!(rendered.contains("Load plan"));
        assert!(rendered.contains("Exit"));
        assert!(rendered.contains("Up/Down"));
        assert!(!rendered.contains("Chat"));
    }

    #[test]
    fn compatibility_screen_uses_only_existing_categories() {
        for (state, label) in [
            (PlanCompatibility::Valid, "VALID"),
            (
                PlanCompatibility::CompatibleRecalibrationRecommended,
                "COMPATIBLE — RECALIBRATION RECOMMENDED",
            ),
            (PlanCompatibility::Incompatible, "INCOMPATIBLE"),
        ] {
            let mut app = TuiApp::default();
            app.screen = Screen::PlanCompatibility;
            app.compatibility_report = Some(ramforge_runtime::PlanCompatibilityReport {
                state,
                reasons: Vec::new(),
            });
            let rendered = render(&app, None, None);
            assert!(rendered.contains(label));
        }
    }

    #[test]
    fn calibration_screen_never_fabricates_results() {
        let mut app = TuiApp::default();
        app.screen = Screen::Calibrating;
        app.calibration_tasks.push(CalibrationTaskView {
            identifier: "task".to_string(),
            kind: "CPU scalar F32 dot".to_string(),
            state: CalibrationTaskDisplayState::Pending,
            value: None,
            unit: None,
            reason: None,
        });
        let rendered = render(&app, None, None);
        assert!(rendered.contains("Pending"));
        assert!(!rendered.contains("result:"));
    }
}
