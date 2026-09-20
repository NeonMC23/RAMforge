use std::fmt::Write as _;

use ramforge_runtime::planner::{
    Availability, ModelExecutionCompatibility, ObservationUnit, PlanCompatibility, PlanReasonCode,
};
use ramforge_runtime::runtime_config::RuntimeConfig;

use super::app::{
    calibration_level_for_index, calibration_level_label, mode_label, CalibrationTaskDisplayState,
    CalibrationTaskView, MaxTokensPreset, ModelFlow, PlanOrigin, RamPreset, Screen, TestPreset,
    TuiApp,
};
use super::diagnostics::{format_bytes as diag_format_bytes, format_seconds};

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
        Screen::TestPreset => render_test_preset(app, &mut output),
        Screen::ReviewConfig => render_review_config(app, runtime_config, &mut output),
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
        Screen::RuntimeActivation => render_runtime_activation(&mut output),
        Screen::GenerationInput => render_generation_input(app, &mut output),
        Screen::GenerationRunning => render_generation_running(app, &mut output),
        Screen::GenerationResult => render_generation_result(app, &mut output),
        Screen::RunHistory => render_run_history(app, &mut output),
        Screen::RunDetail => render_run_detail(app, &mut output),
        Screen::RunCompareSelect => render_run_compare_select(app, &mut output),
        Screen::RunCompare => render_run_compare(app, &mut output),
        Screen::ExportPath => render_export_path(app, &mut output),
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
    let ram_preset = RamPreset::ALL[app.lab.ram_preset_index.min(RamPreset::ALL.len() - 1)];
    let ram_value = if app.lab.ram_preset_index == RamPreset::ALL.len() - 1 {
        if app.ram_input.is_empty() {
            "<custom>"
        } else {
            app.ram_input.as_str()
        }
    } else {
        app.ram_input.as_str()
    };
    preference_line(
        output,
        app.menu_index == 0,
        "RAM budget",
        &format!("{} [{}]", ram_value, ram_preset.label()),
    );
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
        menu_line(output, app.menu_index == 7, "Test preset...");
        menu_line(output, app.menu_index == 8, "Analyze model");
    } else {
        menu_line(output, app.menu_index == 2, "Test preset...");
        menu_line(output, app.menu_index == 3, "Analyze model");
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "RAM presets: 500 MiB | 1 GiB | 2 GiB | 4 GiB | 6 GiB | 8 GiB | 12 GiB | Custom"
    );
    let _ = writeln!(
        output,
        "Modes: Background / Constrained | Balanced / Normal | Maximum Performance | Advanced"
    );
    let _ = writeln!(
        output,
        "{}",
        if app.preference_editing {
            "Editing selected value. Enter or Esc ends editing."
        } else {
            "Enter edits text or opens submenu. Left/Right cycles presets/options."
        }
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
    field(
        output,
        "Name",
        model
            .identity
            .display_name
            .as_deref()
            .unwrap_or("Unavailable"),
    );
    field(
        output,
        "Architecture",
        model.architecture.as_deref().unwrap_or("Unknown"),
    );
    field(output, "File size", &format_bytes(model.file_size_bytes));
    field(output, "Tensor count", &model.tensor_count.to_string());
    field(output, "Context length", &optional_u64(model.context_size));
    field(
        output,
        "Embedding length",
        &optional_u64(info.embedding_length),
    );
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
            ModelExecutionCompatibility::UnknownArchitecture => {
                "Not supported (unknown architecture)"
            }
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
    field(
        output,
        "Storage usable",
        yes_no(capabilities.storage_usable),
    );
    field(
        output,
        "CPU backend usable",
        yes_no(capabilities.cpu_backend_usable),
    );
    field(
        output,
        "GPU backend usable",
        yes_no(capabilities.gpu_backend_usable),
    );
    field(
        output,
        "Model executable",
        yes_no(capabilities.model_execution_compatible),
    );
    field(
        output,
        "Layer cache possible",
        yes_no(capabilities.layer_cache_possible),
    );
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

fn render_calibration(app: &TuiApp, tasks: &[CalibrationTaskView], output: &mut String) {
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
    let _ = writeln!(output, "ExecutionPlan -> PlanCompiler -> RuntimeConfig");
}

fn render_plan_valid(app: &TuiApp, runtime_config: Option<&RuntimeConfig>, output: &mut String) {
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
        field(
            output,
            "Execution device",
            &format!("{:?}", config.execution_device),
        );
    }
    let _ = writeln!(output);
    menu_line(output, app.menu_index == 0, "Activate runtime");
    if app.plan_origin == PlanOrigin::LoadedPlan {
        menu_line(output, app.menu_index == 1, "Back to compatibility");
    } else {
        menu_line(output, app.menu_index == 1, "Save plan");
        menu_line(output, app.menu_index == 2, "Back to plan");
    }
}

fn render_runtime_activation(output: &mut String) {
    let _ = writeln!(output, "Activating validated runtime...");
    let _ = writeln!(
        output,
        "ExecutionPlan -> PlanCompiler -> RuntimeConfig -> InferenceEngine"
    );
}

fn render_generation_input(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Single-prompt generation");
    let _ = writeln!(output, "Executable runtime: active");
    let _ = writeln!(output, "Sampling: greedy");
    let _ = writeln!(output, "Maximum generated tokens: {}", app.lab.max_tokens);
    let _ = writeln!(output, "Enter one prompt. Generation starts only on Enter.");
    let _ = writeln!(output);
    let _ = writeln!(output, "> {}_", app.prompt_input);
}

fn render_generation_running(app: &TuiApp, output: &mut String) {
    let live = app.snapshot_live_status();
    let _ = writeln!(output, "Generation running...");
    let _ = writeln!(output, "Phase: {}", live.phase.label());
    let _ = writeln!(
        output,
        "Elapsed: {}",
        format_seconds(live.elapsed().as_secs_f64())
    );
    let _ = writeln!(output, "Prompt tokens: {}", live.prompt_tokens);
    let _ = writeln!(output, "Generated tokens: {}", live.generated_tokens);
    let _ = writeln!(output, "Maximum tokens: {}", live.max_tokens);
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Streaming text output will appear in the result view."
    );
    let _ = writeln!(output, "No percentage or ETA is fabricated.");
}

fn render_generation_result(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Generation result");
    if let Some(diag) = app.lab.last_result.as_ref() {
        let g = &diag.generation;
        let c = &diag.cpu_compute;
        let m = &diag.memory;
        let ca = &diag.cache;
        let io = &diag.io;
        let ly = &diag.layers;
        let _ = writeln!(output);
        let _ = writeln!(output, "Generation");
        field(
            output,
            "Total elapsed",
            &format_seconds(g.total_elapsed_seconds),
        );
        field(
            output,
            "Prompt/prefill",
            &format_seconds(g.prompt_elapsed_seconds),
        );
        field(
            output,
            "Decode",
            g.decode_elapsed_seconds
                .map(format_seconds)
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(output, "Prompt tokens", &g.prompt_tokens.to_string());
        field(output, "Generated tokens", &g.generated_tokens.to_string());
        field(output, "Max tokens", &g.max_tokens.to_string());
        field(
            output,
            "Tokens/sec (overall)",
            g.tokens_per_second_overall
                .map(|v| format!("{v:.3}"))
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(
            output,
            "Tokens/sec (gen)",
            g.tokens_per_second_generated
                .map(|v| format!("{v:.3}"))
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(output, "Prompt forwards", &g.prompt_forwards.to_string());
        field(output, "Decode forwards", &g.decode_forwards.to_string());

        let _ = writeln!(output);
        let _ = writeln!(output, "CPU / Compute");
        field(output, "CPU threads", &c.cpu_threads.to_string());
        field(output, "F32 matvec", &format_seconds(c.f32_matvec_seconds));
        field(
            output,
            "Quantized matvec",
            &format_seconds(c.quantized_matvec_seconds),
        );
        if c.q4_0_matvec_calls > 0 {
            field(
                output,
                "Q4_0 matvec calls",
                &c.q4_0_matvec_calls.to_string(),
            );
            field(output, "Q4_0 matvec rows", &c.q4_0_matvec_rows.to_string());
            field(
                output,
                "Q4_0 input elements",
                &c.q4_0_matvec_input_elements.to_string(),
            );
            field(output, "Q4_0 blocks", &c.q4_0_matvec_blocks.to_string());
            field(
                output,
                "Q4_0 weight bytes",
                &c.q4_0_matvec_weight_bytes.to_string(),
            );
        }
        field(
            output,
            "Dequantization",
            &format_seconds(c.dequantization_seconds),
        );
        field(
            output,
            "Layer compute",
            &format_seconds(c.layer_compute_seconds),
        );
        field(output, "Logits", &format_seconds(c.logits_seconds));
        field(output, "Sampling", &format_seconds(g.sampling_seconds));

        let _ = writeln!(output);
        let _ = writeln!(output, "Memory");
        field(
            output,
            "Configured budget",
            &diag_format_bytes(m.configured_budget_bytes),
        );
        field(
            output,
            "Managed peak",
            &diag_format_bytes(m.managed_peak_bytes),
        );
        field(
            output,
            "Managed current",
            &diag_format_bytes(m.managed_current_bytes),
        );
        field(
            output,
            "Process RSS",
            m.process_rss_bytes
                .map(diag_format_bytes)
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(
            output,
            "System memory",
            match (m.system_total_bytes, m.system_available_bytes) {
                (Some(t), Some(a)) => format!(
                    "{} total, {} available",
                    diag_format_bytes(t),
                    diag_format_bytes(a)
                ),
                _ => "Unavailable".to_string(),
            }
            .as_str(),
        );
        field(
            output,
            "Cache capacity",
            &diag_format_bytes(ca.capacity_bytes),
        );

        let _ = writeln!(output);
        let _ = writeln!(output, "Cache");
        field(output, "Hits", &ca.hits.to_string());
        field(output, "Misses", &ca.misses.to_string());
        field(output, "Evictions", &ca.evictions.to_string());
        field(
            output,
            "Peak cached layers",
            &ca.peak_cached_layers.to_string(),
        );
        field(
            output,
            "Peak cache bytes",
            &diag_format_bytes(ca.peak_cache_bytes),
        );

        let _ = writeln!(output);
        let _ = writeln!(output, "I/O");
        field(
            output,
            "Logical reads",
            &io.logical_tensor_reads.to_string(),
        );
        field(output, "Physical reads", &io.physical_reads.to_string());
        field(
            output,
            "Logical bytes",
            &diag_format_bytes(io.logical_tensor_bytes),
        );
        field(
            output,
            "Physical bytes",
            &diag_format_bytes(io.physical_bytes),
        );
        field(
            output,
            "Read time",
            &format_seconds(io.read_elapsed_seconds),
        );
        field(output, "Seeks avoided", &io.seeks_avoided.to_string());
        field(output, "Actual seeks", &io.seek_operations.to_string());
        field(output, "Coalesced ranges", &io.coalesced_ranges.to_string());
        field(output, "Buffer reuses", &io.read_buffer_reuses.to_string());
        field(
            output,
            "Buffer growths",
            &io.read_buffer_growths.to_string(),
        );

        let _ = writeln!(output);
        let _ = writeln!(output, "Layers");
        field(output, "Loads", &ly.loads.to_string());
        field(output, "Releases", &ly.releases.to_string());
        field(output, "Load time", &format_seconds(ly.load_seconds));
        field(output, "Compute time", &format_seconds(ly.compute_seconds));
        field(output, "Release time", &format_seconds(ly.release_seconds));

        let _ = writeln!(output);
        let _ = writeln!(output, "Calibration");
        field(output, "Level", &diag.calibration.level);
        field(
            output,
            "Identifier",
            diag.calibration
                .calibration_identifier
                .as_deref()
                .unwrap_or("Not Applicable"),
        );
        field(
            output,
            "Consumed by planning",
            if diag.calibration.consumed_by_planning {
                "Yes"
            } else {
                "No"
            },
        );

        let _ = writeln!(output);
        let _ = writeln!(output, "Observations");
        if diag.observations.is_empty() {
            let _ = writeln!(output, "  None");
        } else {
            for obs in &diag.observations {
                let _ = writeln!(output, "  - {obs}");
            }
        }

        let _ = writeln!(output);
        let _ = writeln!(output, "Generated text");
        if app.lab.last_generated_text.is_empty() {
            let _ = writeln!(output, "<empty generated text>");
        } else {
            for line in app.lab.last_generated_text.lines() {
                let _ = writeln!(output, "{line}");
            }
        }
    } else {
        let _ = writeln!(output, "No diagnostic result available.");
        let _ = writeln!(output);
        let _ = writeln!(output, "Generated text");
        if app.lab.last_generated_text.is_empty() {
            // Fall back to legacy generation_result field
            if let Some(gr) = app.generation_result.as_ref() {
                if gr.generated_text.is_empty() {
                    let _ = writeln!(output, "<empty generated text>");
                } else {
                    for line in gr.generated_text.lines() {
                        let _ = writeln!(output, "{line}");
                    }
                }
            } else {
                let _ = writeln!(output, "<empty generated text>");
            }
        } else {
            for line in app.lab.last_generated_text.lines() {
                let _ = writeln!(output, "{line}");
            }
        }
    }
    let _ = writeln!(output);
    menu_line(output, app.menu_index == 0, "Generate another prompt");
    menu_line(output, app.menu_index == 1, "Run history");
    menu_line(output, app.menu_index == 2, "Export last run JSON");
    menu_line(output, app.menu_index == 3, "Return to plan");
}

fn render_test_preset(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Test presets");
    let _ = writeln!(
        output,
        "Bundles of existing UserProfile / generation values for repeatable runs."
    );
    let _ = writeln!(output);
    for (index, preset) in TestPreset::ALL.iter().enumerate() {
        menu_line(output, app.menu_index == index, preset.label());
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Presets only set existing Planner/UserProfile values. They do not add new optimization rules.");
    let _ = writeln!(
        output,
        "Current: {}",
        TestPreset::ALL[app.lab.test_preset_index.min(TestPreset::ALL.len() - 1)].label()
    );
}

fn render_review_config(app: &TuiApp, runtime_config: Option<&RuntimeConfig>, output: &mut String) {
    let _ = writeln!(output, "Test configuration");
    if let Some(session) = app.planning_session.as_ref() {
        let model = &session.model;
        field(output, "Model path", &app.model_input);
        field(
            output,
            "Name",
            model
                .identity
                .display_name
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(
            output,
            "Architecture",
            model.architecture.as_deref().unwrap_or("Unknown"),
        );
        field(output, "Tensor count", &model.tensor_count.to_string());
        let _ = writeln!(output);
        let _ = writeln!(output, "Tensor / quantization summary");
        for tf in &model.tensor_formats {
            let _ = writeln!(
                output,
                "  {:<10} count={:<6} bytes={:<12} supported={}",
                tf.format,
                tf.tensor_count,
                tf.byte_count
                    .map(diag_format_bytes)
                    .unwrap_or_else(|| "Unavailable".to_string()),
                yes_no(tf.runtime_supported)
            );
        }
    }
    if let Some(config) = runtime_config {
        let _ = writeln!(output);
        let _ = writeln!(output, "Resolved execution configuration");
        field(
            output,
            "RAM budget",
            &diag_format_bytes(config.ram_budget_bytes),
        );
        field(output, "Mode", mode_label(app.mode));
        field(output, "CPU threads", &config.cpu_thread_count.to_string());
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
            "Cache capacity",
            &diag_format_bytes(config.layer_cache_capacity_bytes),
        );
        field(
            output,
            "Read coalescing",
            if config.read_coalescing_enabled {
                "Enabled"
            } else {
                "Disabled"
            },
        );
        field(
            output,
            "Grouped buffer reuse",
            if config.grouped_read_buffer_reuse_enabled {
                "Enabled"
            } else {
                "Disabled"
            },
        );
        field(
            output,
            "Execution device",
            match config.execution_device {
                ramforge_runtime::runtime_config::RuntimeExecutionDevice::Cpu => "CPU only",
            },
        );
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Calibration");
    field(
        output,
        "Level",
        calibration_level_label(app.calibration_level()),
    );
    let _ = writeln!(output);
    preference_line(
        output,
        app.menu_index == 0,
        "Max tokens",
        if app.lab.max_tokens_preset_index == MaxTokensPreset::ALL.len() - 1 {
            if app.lab.custom_max_tokens_input.is_empty() {
                "<custom>"
            } else {
                &app.lab.custom_max_tokens_input
            }
        } else {
            MaxTokensPreset::ALL[app.lab.max_tokens_preset_index].label()
        },
    );
    menu_line(output, app.menu_index == 1, "Enter prompt and run");
    menu_line(output, app.menu_index == 2, "Back to plan");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Use Left/Right to change the max-tokens preset. Enter cycles presets."
    );
}

fn render_run_history(app: &TuiApp, output: &mut String) {
    let _ = writeln!(
        output,
        "Run history (session-only, bounded to {} entries)",
        super::diagnostics::MAX_RUN_HISTORY
    );
    if app.lab.run_history.is_empty() {
        let _ = writeln!(output, "  No runs recorded yet.");
    } else {
        for (index, run) in app.lab.run_history.iter().enumerate() {
            let model = run
                .model_name
                .as_deref()
                .or(run.model_architecture.as_deref())
                .unwrap_or("Unknown model");
            let selected = app.menu_index == index;
            let _ = writeln!(
                output,
                "{} #{}  {}  {}  tok/s={}  elapsed={}",
                if selected { ">" } else { " " },
                run.run_id,
                run.timestamp_unix_seconds,
                model,
                run.diagnostics
                    .generation
                    .tokens_per_second_generated
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "-".to_string()),
                format_seconds(run.diagnostics.generation.total_elapsed_seconds),
            );
        }
    }
    let _ = writeln!(output);
    let compare_idx = app.lab.run_history.len();
    menu_line(output, app.menu_index == compare_idx, "Compare two runs");
    menu_line(output, app.menu_index == compare_idx + 1, "Back to result");
}

fn render_run_detail(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Run detail");
    if let Some(run) = app.selected_history_run() {
        let g = &run.diagnostics.generation;
        field(output, "Run ID", &run.run_id.to_string());
        field(output, "Timestamp", &run.timestamp_unix_seconds.to_string());
        field(
            output,
            "Model",
            run.model_name.as_deref().unwrap_or("Unavailable"),
        );
        field(output, "Model path", &run.model_path);
        field(output, "Mode", &run.user_config.mode_label);
        field(
            output,
            "RAM budget",
            &diag_format_bytes(run.user_config.ram_budget_bytes),
        );
        field(
            output,
            "CPU threads",
            &run.runtime_config.cpu_thread_count.to_string(),
        );
        field(output, "Max tokens", &g.max_tokens.to_string());
        field(output, "Generated tokens", &g.generated_tokens.to_string());
        field(output, "Elapsed", &format_seconds(g.total_elapsed_seconds));
        field(
            output,
            "Tokens/sec",
            g.tokens_per_second_generated
                .map(|v| format!("{v:.3}"))
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(output, "Success", if run.success { "Yes" } else { "No" });
        if let Some(err) = run.error.as_deref() {
            field(output, "Error", err);
        }
    } else {
        let _ = writeln!(output, "No run selected.");
    }
    let _ = writeln!(output);
    menu_line(output, app.menu_index == 0, "Back to history");
    menu_line(output, app.menu_index == 1, "Export JSON");
}

fn render_run_compare_select(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Compare runs");
    if app.lab.compare_first_id.is_none() {
        let _ = writeln!(output, "Select the FIRST run.");
    } else {
        let _ = writeln!(output, "Select the SECOND run to compare against.");
    }
    let _ = writeln!(output);
    for (index, run) in app.lab.run_history.iter().enumerate() {
        let mut marker = if app.menu_index == index { ">" } else { " " }.to_string();
        if Some(run.run_id) == app.lab.compare_first_id {
            marker.push_str(" [1]");
        }
        let _ = writeln!(
            output,
            "{marker} #{} elapsed={} tok/s={}",
            run.run_id,
            format_seconds(run.diagnostics.generation.total_elapsed_seconds),
            run.diagnostics
                .generation
                .tokens_per_second_generated
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".to_string()),
        );
    }
    let _ = writeln!(output);
    menu_line(output, app.menu_index == app.lab.run_history.len(), "Back");
}

fn render_run_compare(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Run comparison");
    if let Some(cmp) = app.lab.comparison.as_ref() {
        let _ = writeln!(
            output,
            "Comparing run #{} vs run #{}",
            cmp.left_id, cmp.right_id
        );
        let _ = writeln!(output);
        let _ = writeln!(output, "  {:<24} {:>20} {:>20}", "Metric", "Left", "Right");
        for f in &cmp.fields {
            let _ = writeln!(output, "  {:<24} {:>20} {:>20}", f.label, f.left, f.right);
        }
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "No scores, rankings, or value judgments are emitted."
        );
    } else {
        let _ = writeln!(output, "Comparison unavailable.");
    }
    let _ = writeln!(output);
    menu_line(output, true, "Back to history");
}

fn render_export_path(app: &TuiApp, output: &mut String) {
    let _ = writeln!(output, "Export JSON");
    let _ = writeln!(output, "Existing files are not overwritten.");
    let _ = writeln!(output);
    let _ = writeln!(output, "> {}_", app.lab.export_path_input);
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

    if let (Some(persisted), Some(session)) =
        (app.persisted_plan.as_ref(), app.planning_session.as_ref())
    {
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
            session
                .machine
                .cpu_model
                .as_deref()
                .unwrap_or("Unavailable"),
        );
        field(
            output,
            "Logical CPUs",
            &session.machine.logical_cpu_cores.to_string(),
        );
        field(
            output,
            "Storage kind",
            &format!("{:?}", session.storage.kind),
        );
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
                let _ = writeln!(output, "  {decision}: {value} | Reason: {:?}", reason.code);
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
        if value.is_empty() {
            "<required>"
        } else {
            value
        }
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
        Screen::GenerationInput => {
            "Type prompt  Enter Generate  Backspace Delete  Esc Back  Ctrl-C Quit"
        }
        Screen::ExportPath => "Type path  Enter Export  Backspace Delete  Esc Back  Ctrl-C Quit",
        Screen::Preferences if app.accepts_text() => {
            "Type value  Enter/Esc Finish editing  Up/Down Move  Ctrl-C Quit"
        }
        Screen::ReviewConfig
            if app.menu_index == 0
                && app.lab.max_tokens_preset_index == MaxTokensPreset::ALL.len() - 1 =>
        {
            "Type number  Enter/Esc Finish editing  Up/Down Move  Ctrl-C Quit"
        }
        Screen::Analyzing | Screen::PlanValidation | Screen::RuntimeActivation => {
            "Please wait  Ctrl-C Quit"
        }
        Screen::GenerationRunning => "Generation running",
        Screen::Calibrating if !app.calibration_complete => "Calibration running",
        _ => "Arrow keys Navigate/Change  Enter Select  Esc Back  q Quit",
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
        assert!(rendered.contains("Arrow keys"));
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
    fn generation_result_renders_only_the_actual_latest_result() {
        let mut app = TuiApp::default();
        app.screen = Screen::GenerationResult;
        app.lab.last_generated_text = "actual output".to_string();
        app.generation_result = Some(super::super::app::SinglePromptResult {
            generated_text: "actual output".to_string(),
            generated_token_count: 2,
        });
        let rendered = render(&app, None, None);
        assert!(rendered.contains("actual output"));
        assert!(rendered.contains("Generation"));
        assert!(!rendered.contains("conversation"));
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
