use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ramforge_core::parse_memory_size;
use ramforge_runtime::calibration::{CalibrationError, CalibrationTask, CalibrationTestKind};
use ramforge_runtime::discovery::StorageDiscoveryError;
use ramforge_runtime::inference::InferenceEngine;
use ramforge_runtime::orchestration::{OrchestrationError, PlanningSession};
use ramforge_runtime::plan_persistence::{
    PersistedExecutionPlan, PlanCompatibilityReport, PlanPersistenceError,
};
use ramforge_runtime::planner::{
    AdvancedOverrides, CalibrationLevel, CalibrationResult, ExecutionPlan, ObservationStatus,
    ObservationStatusReason, OperatingMode, PlanCompatibility, UserProfile,
};

use super::diagnostics::{
    DiagnosticResult, GenerationParamsSnapshot, GenerationPhase, LiveStatus, RunComparison,
    RunRecord, RuntimeConfigSnapshot, UserConfigSnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Welcome,
    LoadPlan,
    ModelInput,
    Preferences,
    TestPreset,
    ReviewConfig,
    Analyzing,
    ModelInfo,
    CalibrationSelect,
    Calibrating,
    PlanReview,
    PlanValidation,
    PlanValid,
    RuntimeActivation,
    GenerationInput,
    GenerationRunning,
    GenerationResult,
    RunHistory,
    RunDetail,
    RunCompareSelect,
    RunCompare,
    ExportPath,
    PlanCompatibility,
    SavePlan,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFlow {
    NewPlan,
    LoadedPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanOrigin {
    NewPlan,
    LoadedPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputContext {
    None,
    LoadPlanPath,
    ModelPath,
    PreferenceValue,
    GenerationPrompt,
    SavePlanPath,
    ExportPath,
    CustomMaxTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiCommand {
    Up,
    Down,
    Left,
    Right,
    Enter,
    Back,
    Backspace,
    Quit,
    Character(char),
    None,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AppAction {
    LoadPlan(PathBuf),
    ValidateModelPath(PathBuf),
    Analyze,
    AnalyzeLoadedPlan,
    RunCalibration,
    CreatePlan,
    ValidatePlan,
    ActivateRuntime,
    GeneratePrompt(String, usize),
    SavePlan(PathBuf),
    ExportRun(u64, PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideChoice {
    PlannerDefault,
    Enabled,
    Disabled,
}

impl OverrideChoice {
    pub fn next(self) -> Self {
        match self {
            Self::PlannerDefault => Self::Enabled,
            Self::Enabled => Self::Disabled,
            Self::Disabled => Self::PlannerDefault,
        }
    }

    pub fn previous(self) -> Self {
        match self {
            Self::PlannerDefault => Self::Disabled,
            Self::Enabled => Self::PlannerDefault,
            Self::Disabled => Self::Enabled,
        }
    }

    pub fn as_option(self) -> Option<bool> {
        match self {
            Self::PlannerDefault => None,
            Self::Enabled => Some(true),
            Self::Disabled => Some(false),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PlannerDefault => "Planner default",
            Self::Enabled => "Enabled",
            Self::Disabled => "Disabled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationTaskDisplayState {
    Pending,
    Running,
    Measured,
    Unavailable,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationTaskView {
    pub identifier: String,
    pub kind: String,
    pub state: CalibrationTaskDisplayState,
    pub value: Option<u64>,
    pub unit: Option<String>,
    pub reason: Option<String>,
}

impl CalibrationTaskView {
    pub fn pending(task: &CalibrationTask) -> Self {
        Self {
            identifier: task.identifier.clone(),
            kind: calibration_kind_label(task.kind),
            state: CalibrationTaskDisplayState::Pending,
            value: None,
            unit: None,
            reason: None,
        }
    }

    pub fn mark_running(&mut self) {
        self.state = CalibrationTaskDisplayState::Running;
        self.value = None;
        self.unit = None;
        self.reason = None;
    }

    pub fn finish(
        &mut self,
        status: ObservationStatus,
        value: Option<u64>,
        unit: String,
        reason: Option<ObservationStatusReason>,
    ) {
        self.state = match status {
            ObservationStatus::Measured => CalibrationTaskDisplayState::Measured,
            ObservationStatus::Unavailable => CalibrationTaskDisplayState::Unavailable,
            ObservationStatus::Skipped => CalibrationTaskDisplayState::Skipped,
            ObservationStatus::Failed => CalibrationTaskDisplayState::Failed,
        };
        self.value = value;
        self.unit = Some(unit);
        self.reason = reason.map(|reason| format!("{reason:?}"));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinglePromptResult {
    pub generated_text: String,
    pub generated_token_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamPreset {
    M500,
    G1,
    G2,
    G4,
    G6,
    G8,
    G12,
    Custom,
}

impl RamPreset {
    pub const ALL: &'static [Self] = &[
        Self::M500,
        Self::G1,
        Self::G2,
        Self::G4,
        Self::G6,
        Self::G8,
        Self::G12,
        Self::Custom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::M500 => "500 MiB",
            Self::G1 => "1 GiB",
            Self::G2 => "2 GiB",
            Self::G4 => "4 GiB",
            Self::G6 => "6 GiB",
            Self::G8 => "8 GiB",
            Self::G12 => "12 GiB",
            Self::Custom => "Custom...",
        }
    }

    pub fn bytes(self) -> Option<u64> {
        const MIB: u64 = 1024 * 1024;
        const GIB: u64 = 1024 * MIB;
        match self {
            Self::M500 => Some(500 * MIB),
            Self::G1 => Some(GIB),
            Self::G2 => Some(2 * GIB),
            Self::G4 => Some(4 * GIB),
            Self::G6 => Some(6 * GIB),
            Self::G8 => Some(8 * GIB),
            Self::G12 => Some(12 * GIB),
            Self::Custom => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokensPreset {
    T1,
    T8,
    T16,
    T32,
    T64,
    T128,
    Custom,
}

impl MaxTokensPreset {
    pub const ALL: &'static [Self] = &[
        Self::T1,
        Self::T8,
        Self::T16,
        Self::T32,
        Self::T64,
        Self::T128,
        Self::Custom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::T1 => "1",
            Self::T8 => "8",
            Self::T16 => "16",
            Self::T32 => "32",
            Self::T64 => "64",
            Self::T128 => "128",
            Self::Custom => "Custom...",
        }
    }

    pub fn value(self) -> Option<usize> {
        match self {
            Self::T1 => Some(1),
            Self::T8 => Some(8),
            Self::T16 => Some(16),
            Self::T32 => Some(32),
            Self::T64 => Some(64),
            Self::T128 => Some(128),
            Self::Custom => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestPreset {
    Quick,
    Balanced,
    LowRam,
    Cache,
    StorageIo,
    MaximumCpu,
    Custom,
}

impl TestPreset {
    pub const ALL: &'static [Self] = &[
        Self::Quick,
        Self::Balanced,
        Self::LowRam,
        Self::Cache,
        Self::StorageIo,
        Self::MaximumCpu,
        Self::Custom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Quick => "Quick Test",
            Self::Balanced => "Balanced Test",
            Self::LowRam => "Low RAM Test",
            Self::Cache => "Cache Test",
            Self::StorageIo => "Storage / I/O Test",
            Self::MaximumCpu => "Maximum CPU Test",
            Self::Custom => "Custom Test",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticLabState {
    // RAM preset selection (mirrors/affects ram_input)
    pub ram_preset_index: usize,
    pub custom_ram_input: String,
    // Max tokens selection
    pub max_tokens_preset_index: usize,
    pub max_tokens: usize,
    pub custom_max_tokens_input: String,
    // Test preset
    pub test_preset_index: usize,
    // Live generation status
    pub live: LiveStatus,
    pub live_started_at: Option<Instant>,
    // Last completed diagnostic result (the current screen data)
    pub last_result: Option<DiagnosticResult>,
    pub last_run_success: bool,
    pub last_error: Option<String>,
    pub last_generated_text: String,
    pub last_prompt_tokens: usize,
    pub last_wall_elapsed_ms: u64,
    // Bounded session history
    pub run_history: Vec<RunRecord>,
    pub selected_history_id: Option<u64>,
    pub compare_first_id: Option<u64>,
    pub comparison: Option<RunComparison>,
    // Export path input
    pub export_path_input: String,
    // Thread/cache live display state shared by callbacks
    pub live_prompt_tokens: Arc<AtomicU64>,
    pub live_generated_tokens: Arc<AtomicU64>,
    pub live_phase: Arc<AtomicU64>,
    pub live_stop: Arc<AtomicBool>,
    pub prompt_input_before_run: String,
}

#[derive(Debug)]
pub enum TuiError {
    Input {
        field: &'static str,
        message: String,
    },
    Storage(StorageDiscoveryError),
    Analysis(OrchestrationError),
    Calibration(CalibrationError),
    Planning(OrchestrationError),
    Compilation(OrchestrationError),
    RuntimeConstruction(OrchestrationError),
    Generation(String),
    Persistence(PlanPersistenceError),
    DestinationExists(PathBuf),
}

impl TuiError {
    pub fn title(&self) -> &'static str {
        match self {
            Self::Input { .. } => "Input validation failed",
            Self::Storage(_) => "Storage discovery failed",
            Self::Analysis(error) => orchestration_error_title(error),
            Self::Calibration(_) => "Calibration failed",
            Self::Planning(_) => "Planning failed",
            Self::Compilation(_) => "Plan compilation failed",
            Self::RuntimeConstruction(_) => "Runtime construction failed",
            Self::Generation(_) => "Generation failed",
            Self::Persistence(_) | Self::DestinationExists(_) => "Plan persistence failed",
        }
    }
}

impl fmt::Display for TuiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { field, message } => write!(formatter, "{field}: {message}"),
            Self::Storage(error) => write!(formatter, "{error}"),
            Self::Analysis(error)
            | Self::Planning(error)
            | Self::Compilation(error)
            | Self::RuntimeConstruction(error) => write!(formatter, "{error}"),
            Self::Generation(error) => write!(formatter, "{error}"),
            Self::Calibration(error) => write!(formatter, "{error}"),
            Self::Persistence(error) => write!(formatter, "{error}"),
            Self::DestinationExists(path) => {
                write!(formatter, "destination already exists: {}", path.display())
            }
        }
    }
}

fn orchestration_error_title(error: &OrchestrationError) -> &'static str {
    match error {
        OrchestrationError::StorageDiscovery(_) => "Storage discovery failed",
        OrchestrationError::ModelInspection(_) => "GGUF inspection failed",
        OrchestrationError::ModelProfileConstruction(_) => "Model inspection failed",
        OrchestrationError::StaticPlanning(_) => "Static planning failed",
        OrchestrationError::MachineDiscovery(_) => "Machine discovery failed",
        OrchestrationError::CapabilityConstruction { .. } => "Capability construction failed",
        OrchestrationError::Planning(_) => "Planning failed",
        OrchestrationError::PlanCompilation(_) => "Plan compilation failed",
        OrchestrationError::RuntimeSourceUnavailable => "Runtime source unavailable",
        OrchestrationError::RuntimeConstruction(_) => "Runtime construction failed",
    }
}

#[derive(Debug)]
pub struct TuiApp {
    pub screen: Screen,
    pub should_quit: bool,
    pub menu_index: usize,
    pub load_path_input: String,
    pub model_input: String,
    pub model_flow: ModelFlow,
    pub plan_origin: PlanOrigin,
    pub ram_input: String,
    pub mode: OperatingMode,
    pub advanced_thread_input: String,
    pub advanced_cache: OverrideChoice,
    pub advanced_cache_capacity_input: String,
    pub advanced_read_coalescing: OverrideChoice,
    pub advanced_grouped_buffer: OverrideChoice,
    pub preference_editing: bool,
    pub calibration_index: usize,
    pub calibration_complete: bool,
    pub calibration_tasks: Vec<CalibrationTaskView>,
    pub save_path_input: String,
    pub status_message: Option<String>,
    pub error: Option<TuiError>,
    error_return: Screen,
    pub user_profile: Option<UserProfile>,
    pub planning_session: Option<PlanningSession>,
    pub calibration_result: Option<CalibrationResult>,
    pub execution_plan: Option<ExecutionPlan>,
    pub persisted_plan: Option<PersistedExecutionPlan>,
    pub compatibility_report: Option<PlanCompatibilityReport>,
    pub prompt_input: String,
    pub generation_result: Option<SinglePromptResult>,
    pub active_runtime: Option<InferenceEngine>,
    pub lab: DiagnosticLabState,
}

impl Default for TuiApp {
    fn default() -> Self {
        Self {
            screen: Screen::Welcome,
            should_quit: false,
            menu_index: 0,
            load_path_input: String::new(),
            model_input: String::new(),
            model_flow: ModelFlow::NewPlan,
            plan_origin: PlanOrigin::NewPlan,
            ram_input: "1GiB".to_string(),
            mode: OperatingMode::BalancedNormal,
            advanced_thread_input: String::new(),
            advanced_cache: OverrideChoice::PlannerDefault,
            advanced_cache_capacity_input: String::new(),
            advanced_read_coalescing: OverrideChoice::PlannerDefault,
            advanced_grouped_buffer: OverrideChoice::PlannerDefault,
            preference_editing: false,
            calibration_index: 0,
            calibration_complete: false,
            calibration_tasks: Vec::new(),
            save_path_input: String::new(),
            status_message: None,
            error: None,
            error_return: Screen::Welcome,
            user_profile: None,
            planning_session: None,
            calibration_result: None,
            execution_plan: None,
            persisted_plan: None,
            compatibility_report: None,
            prompt_input: String::new(),
            generation_result: None,
            active_runtime: None,
            lab: DiagnosticLabState {
                ram_preset_index: 1, // 1 GiB default
                custom_ram_input: String::new(),
                max_tokens_preset_index: 3, // 32 default
                max_tokens: 32,
                custom_max_tokens_input: String::new(),
                test_preset_index: 1, // Balanced
                live: LiveStatus::default(),
                live_started_at: None,
                last_result: None,
                last_run_success: false,
                last_error: None,
                last_generated_text: String::new(),
                last_prompt_tokens: 0,
                last_wall_elapsed_ms: 0,
                run_history: Vec::new(),
                selected_history_id: None,
                compare_first_id: None,
                comparison: None,
                export_path_input: String::new(),
                live_prompt_tokens: Arc::new(AtomicU64::new(0)),
                live_generated_tokens: Arc::new(AtomicU64::new(0)),
                live_phase: Arc::new(AtomicU64::new(0)),
                live_stop: Arc::new(AtomicBool::new(false)),
                prompt_input_before_run: String::new(),
            },
        }
    }
}

impl TuiApp {
    pub fn handle(&mut self, command: UiCommand) -> Option<AppAction> {
        if command == UiCommand::Quit {
            self.should_quit = true;
            return None;
        }
        self.status_message = None;
        if self.screen != Screen::Error {
            self.error = None;
        }
        match self.screen {
            Screen::Welcome => self.handle_welcome(command),
            Screen::LoadPlan => self.handle_load_plan(command),
            Screen::ModelInput => self.handle_model_input(command),
            Screen::Preferences => self.handle_preferences(command),
            Screen::TestPreset => self.handle_test_preset(command),
            Screen::ReviewConfig => self.handle_review_config(command),
            Screen::ModelInfo => self.handle_model_info(command),
            Screen::CalibrationSelect => self.handle_calibration_select(command),
            Screen::Calibrating => self.handle_calibrating(command),
            Screen::PlanReview => self.handle_plan_review(command),
            Screen::PlanValid => self.handle_plan_valid(command),
            Screen::GenerationInput => self.handle_generation_input(command),
            Screen::GenerationResult => self.handle_generation_result(command),
            Screen::RunHistory => self.handle_run_history(command),
            Screen::RunDetail => self.handle_run_detail(command),
            Screen::RunCompareSelect => self.handle_run_compare_select(command),
            Screen::RunCompare => self.handle_run_compare(command),
            Screen::ExportPath => self.handle_export_path(command),
            Screen::PlanCompatibility => self.handle_plan_compatibility(command),
            Screen::SavePlan => self.handle_save_plan(command),
            Screen::Error => {
                if command == UiCommand::Back || command == UiCommand::Enter {
                    self.screen = self.error_return;
                    self.error = None;
                    self.menu_index = 0;
                }
                None
            }
            Screen::Analyzing
            | Screen::PlanValidation
            | Screen::RuntimeActivation
            | Screen::GenerationRunning => None,
        }
    }

    pub fn active_input_context(&self) -> InputContext {
        match self.screen {
            Screen::LoadPlan => InputContext::LoadPlanPath,
            Screen::ModelInput => InputContext::ModelPath,
            Screen::Preferences if self.preference_editing => InputContext::PreferenceValue,
            Screen::ReviewConfig
                if self.preference_editing
                    || (self.menu_index == 0 && self.custom_max_tokens_focused()) =>
            {
                InputContext::CustomMaxTokens
            }
            Screen::GenerationInput => InputContext::GenerationPrompt,
            Screen::SavePlan => InputContext::SavePlanPath,
            Screen::ExportPath => InputContext::ExportPath,
            _ => InputContext::None,
        }
    }

    fn custom_max_tokens_focused(&self) -> bool {
        self.lab.max_tokens_preset_index == MaxTokensPreset::ALL.len() - 1
    }

    pub fn accepts_text(&self) -> bool {
        self.active_input_context() != InputContext::None
    }

    pub fn load_finished(&mut self, result: Result<PersistedExecutionPlan, TuiError>) {
        match result {
            Ok(plan) => {
                self.persisted_plan = Some(plan);
                self.compatibility_report = None;
                self.execution_plan = None;
                self.planning_session = None;
                self.model_input.clear();
                self.model_flow = ModelFlow::LoadedPlan;
                self.plan_origin = PlanOrigin::LoadedPlan;
                self.screen = Screen::ModelInput;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => self.show_error(error, Screen::LoadPlan),
        }
    }

    pub fn model_path_validation_finished(
        &mut self,
        result: Result<(), TuiError>,
    ) -> Option<AppAction> {
        match result {
            Ok(()) if self.model_flow == ModelFlow::LoadedPlan => {
                self.screen = Screen::Analyzing;
                self.menu_index = 0;
                self.error = None;
                Some(AppAction::AnalyzeLoadedPlan)
            }
            Ok(()) => {
                self.screen = Screen::Preferences;
                self.menu_index = 0;
                self.preference_editing = false;
                self.error = None;
                None
            }
            Err(error) => {
                self.show_error(error, Screen::ModelInput);
                None
            }
        }
    }

    pub fn loaded_analysis_finished(
        &mut self,
        result: Result<(PlanningSession, ExecutionPlan, PlanCompatibilityReport), TuiError>,
    ) {
        match result {
            Ok((session, plan, report)) => {
                self.planning_session = Some(session);
                self.execution_plan = Some(plan);
                self.compatibility_report = Some(report);
                self.plan_origin = PlanOrigin::LoadedPlan;
                self.screen = Screen::PlanCompatibility;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => self.show_error(error, Screen::ModelInput),
        }
    }

    pub fn analysis_finished(&mut self, result: Result<PlanningSession, TuiError>) {
        match result {
            Ok(session) => {
                self.planning_session = Some(session);
                self.screen = Screen::ModelInfo;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => self.show_error(error, Screen::Preferences),
        }
    }

    pub fn calibration_started(&mut self, tasks: &[CalibrationTask]) {
        self.calibration_tasks = tasks.iter().map(CalibrationTaskView::pending).collect();
        self.calibration_complete = false;
        self.screen = Screen::Calibrating;
    }

    pub fn calibration_finished(
        &mut self,
        result: Result<CalibrationResult, TuiError>,
        task_views: Vec<CalibrationTaskView>,
    ) {
        self.calibration_tasks = task_views;
        match result {
            Ok(calibration) => {
                self.calibration_result = Some(calibration);
                self.calibration_complete = true;
                self.screen = Screen::Calibrating;
                self.status_message =
                    Some("Calibration complete. Press Enter to plan.".to_string());
            }
            Err(error) => self.show_error(error, Screen::CalibrationSelect),
        }
    }

    pub fn planning_finished(&mut self, result: Result<ExecutionPlan, TuiError>) {
        match result {
            Ok(plan) => {
                self.execution_plan = Some(plan);
                self.plan_origin = PlanOrigin::NewPlan;
                self.screen = Screen::PlanReview;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => self.show_error(error, Screen::CalibrationSelect),
        }
    }

    pub fn plan_validation_finished(&mut self, result: Result<(), TuiError>) {
        match result {
            Ok(()) => {
                self.screen = Screen::PlanValid;
                self.menu_index = 0;
                self.error = None;
                self.status_message = Some("Plan valid".to_string());
            }
            Err(error) => {
                self.screen = Screen::PlanReview;
                self.error = Some(error);
            }
        }
    }

    pub fn runtime_activation_finished(&mut self, result: Result<InferenceEngine, TuiError>) {
        match result {
            Ok(runtime) => {
                self.active_runtime = Some(runtime);
                self.prompt_input.clear();
                self.generation_result = None;
                self.screen = Screen::ReviewConfig;
                self.menu_index = 1;
                self.error = None;
            }
            Err(error) => self.show_error(error, Screen::PlanReview),
        }
    }

    pub fn generation_finished(&mut self, result: Result<SinglePromptResult, TuiError>) {
        match result {
            Ok(result) => {
                self.generation_result = Some(result);
                self.screen = Screen::GenerationResult;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => self.show_error(error, Screen::GenerationInput),
        }
    }

    pub fn save_finished(&mut self, result: Result<PathBuf, TuiError>) {
        match result {
            Ok(path) => {
                self.screen = Screen::PlanValid;
                self.menu_index = 0;
                self.error = None;
                self.status_message = Some(format!("Plan saved: {}", path.display()));
            }
            Err(error) => {
                self.screen = Screen::SavePlan;
                self.error = Some(error);
            }
        }
    }

    pub fn calibration_level(&self) -> CalibrationLevel {
        calibration_level_for_index(self.calibration_index)
    }

    pub fn preference_count(&self) -> usize {
        if self.mode == OperatingMode::Advanced {
            9
        } else {
            4
        }
    }

    fn apply_ram_preset(&mut self, index: usize) {
        self.lab.ram_preset_index = index.min(RamPreset::ALL.len() - 1);
        let preset = RamPreset::ALL[self.lab.ram_preset_index];
        if let Some(bytes) = preset.bytes() {
            use super::diagnostics::format_bytes;
            self.ram_input = format_bytes(bytes).replace(' ', "");
            // normalize to the parser's expected unit
            self.ram_input = normalize_memory_input(bytes);
        } else {
            self.ram_input = self.lab.custom_ram_input.clone();
        }
    }

    fn apply_test_preset(&mut self, preset: TestPreset) {
        match preset {
            TestPreset::Quick => {
                self.apply_ram_preset(1); // 1 GiB
                self.mode = OperatingMode::BalancedNormal;
                self.lab.max_tokens = 8;
                self.lab.max_tokens_preset_index = 1;
                self.advanced_cache = OverrideChoice::PlannerDefault;
                self.advanced_read_coalescing = OverrideChoice::PlannerDefault;
                self.advanced_grouped_buffer = OverrideChoice::PlannerDefault;
                self.advanced_thread_input.clear();
                self.advanced_cache_capacity_input.clear();
            }
            TestPreset::Balanced => {
                self.apply_ram_preset(3); // 4 GiB
                self.mode = OperatingMode::BalancedNormal;
                self.lab.max_tokens = 32;
                self.lab.max_tokens_preset_index = 3;
                self.advanced_cache = OverrideChoice::PlannerDefault;
                self.advanced_read_coalescing = OverrideChoice::PlannerDefault;
                self.advanced_grouped_buffer = OverrideChoice::PlannerDefault;
                self.advanced_thread_input.clear();
                self.advanced_cache_capacity_input.clear();
            }
            TestPreset::LowRam => {
                self.apply_ram_preset(0); // 500 MiB
                self.mode = OperatingMode::BackgroundConstrained;
                self.lab.max_tokens = 16;
                self.lab.max_tokens_preset_index = 2;
                self.advanced_cache = OverrideChoice::Disabled;
                self.advanced_read_coalescing = OverrideChoice::Enabled;
                self.advanced_grouped_buffer = OverrideChoice::Disabled;
                self.advanced_thread_input.clear();
                self.advanced_cache_capacity_input.clear();
            }
            TestPreset::Cache => {
                self.apply_ram_preset(3); // 4 GiB
                self.mode = OperatingMode::BalancedNormal;
                self.lab.max_tokens = 64;
                self.lab.max_tokens_preset_index = 4;
                self.advanced_cache = OverrideChoice::Enabled;
                self.advanced_read_coalescing = OverrideChoice::Enabled;
                self.advanced_grouped_buffer = OverrideChoice::Enabled;
                self.advanced_thread_input.clear();
                self.advanced_cache_capacity_input.clear();
            }
            TestPreset::StorageIo => {
                self.apply_ram_preset(2); // 2 GiB
                self.mode = OperatingMode::BackgroundConstrained;
                self.lab.max_tokens = 16;
                self.lab.max_tokens_preset_index = 2;
                self.advanced_cache = OverrideChoice::Disabled;
                self.advanced_read_coalescing = OverrideChoice::Enabled;
                self.advanced_grouped_buffer = OverrideChoice::Enabled;
                self.advanced_thread_input.clear();
                self.advanced_cache_capacity_input.clear();
            }
            TestPreset::MaximumCpu => {
                self.apply_ram_preset(5); // 8 GiB
                self.mode = OperatingMode::MaximumPerformance;
                self.lab.max_tokens = 128;
                self.lab.max_tokens_preset_index = 5;
                self.advanced_cache = OverrideChoice::PlannerDefault;
                self.advanced_read_coalescing = OverrideChoice::PlannerDefault;
                self.advanced_grouped_buffer = OverrideChoice::PlannerDefault;
                self.advanced_thread_input.clear();
                self.advanced_cache_capacity_input.clear();
            }
            TestPreset::Custom => {
                // Leave everything as-is; user edits individual fields.
            }
        }
        self.lab.test_preset_index = TestPreset::ALL
            .iter()
            .position(|p| *p == preset)
            .unwrap_or(6);
    }

    pub fn max_tokens(&self) -> usize {
        self.lab.max_tokens
    }

    pub fn selected_run(&self, id: u64) -> Option<&RunRecord> {
        self.lab.run_history.iter().find(|r| r.run_id == id)
    }

    pub fn selected_history_run(&self) -> Option<&RunRecord> {
        self.lab
            .selected_history_id
            .and_then(|id| self.selected_run(id))
    }

    fn start_generation_live(&mut self) {
        self.lab.live = LiveStatus {
            started_at: Some(Instant::now()),
            prompt_tokens: 0,
            generated_tokens: 0,
            max_tokens: self.lab.max_tokens,
            phase: GenerationPhase::Prompt,
        };
        self.lab.live_started_at = Some(Instant::now());
        self.lab.live_prompt_tokens.store(0, Ordering::Relaxed);
        self.lab.live_generated_tokens.store(0, Ordering::Relaxed);
        self.lab
            .live_phase
            .store(phase_to_u64(GenerationPhase::Prompt), Ordering::Relaxed);
        self.lab.live_stop.store(false, Ordering::Relaxed);
    }

    pub fn snapshot_live_status(&self) -> LiveStatus {
        let phase = match self.lab.live_phase.load(Ordering::Relaxed) {
            1 => GenerationPhase::Decode,
            2 => GenerationPhase::Finished,
            3 => GenerationPhase::Failed,
            _ => GenerationPhase::Prompt,
        };
        LiveStatus {
            started_at: self.lab.live_started_at,
            prompt_tokens: self.lab.live_prompt_tokens.load(Ordering::Relaxed) as usize,
            generated_tokens: self.lab.live_generated_tokens.load(Ordering::Relaxed) as usize,
            max_tokens: self.lab.max_tokens,
            phase,
        }
    }

    pub fn generation_finished_diagnostic(
        &mut self,
        result: Result<(usize, String, DiagnosticResult), TuiError>,
    ) {
        match result {
            Ok((prompt_tokens, generated_text, diag)) => {
                self.lab.last_result = Some(diag.clone());
                self.lab.last_run_success = true;
                self.lab.last_error = None;
                self.lab.last_generated_text = generated_text.clone();
                self.lab.last_prompt_tokens = prompt_tokens;
                self.lab.live.phase = GenerationPhase::Finished;
                self.generation_result = Some(SinglePromptResult {
                    generated_text,
                    generated_token_count: diag.generation.generated_tokens,
                });
                // Append to bounded history
                let id = super::diagnostics::next_run_id(&self.lab.run_history);
                let runtime_cfg = self
                    .planning_session
                    .as_ref()
                    .zip(self.execution_plan.as_ref())
                    .and_then(|(session, plan)| session.compile_plan(plan).ok());
                let rc_snapshot: RuntimeConfigSnapshot = runtime_cfg
                    .as_ref()
                    .map(RuntimeConfigSnapshot::from)
                    .unwrap_or(RuntimeConfigSnapshot {
                        cpu_thread_count: 0,
                        ram_budget_bytes: 0,
                        layer_cache_enabled: false,
                        layer_cache_capacity_bytes: 0,
                        read_coalescing_enabled: false,
                        grouped_read_buffer_reuse_enabled: false,
                        execution_device: "Unavailable".to_string(),
                    });
                let (model_name, arch, mf, machf, storagf) = self
                    .planning_session
                    .as_ref()
                    .map(|s| {
                        (
                            s.model.identity.display_name.clone(),
                            s.model.architecture.clone(),
                            Some(s.model.identity.descriptor_fingerprint),
                            Some(s.machine.fingerprint()),
                            Some(s.storage.fingerprint()),
                        )
                    })
                    .unwrap_or((None, None, None, None, None));
                let _calibration_id = self
                    .calibration_result
                    .as_ref()
                    .map(|c| c.identifier.clone());
                let record = RunRecord {
                    run_id: id,
                    timestamp_unix_seconds: super::diagnostics::unix_timestamp(),
                    model_name,
                    model_architecture: arch,
                    model_descriptor_fingerprint: mf,
                    machine_fingerprint: machf,
                    storage_fingerprint: storagf,
                    model_path: self.model_input.clone(),
                    user_config: UserConfigSnapshot {
                        ram_budget_bytes: diag.memory.configured_budget_bytes,
                        ram_input: self.ram_input.clone(),
                        mode_label: mode_label(self.mode).to_string(),
                        preset_label: Some(
                            TestPreset::ALL
                                [self.lab.test_preset_index.min(TestPreset::ALL.len() - 1)]
                            .label()
                            .to_string(),
                        ),
                        calibration_level: calibration_level_label(self.calibration_level())
                            .to_string(),
                    },
                    runtime_config: rc_snapshot,
                    generation_params: GenerationParamsSnapshot {
                        prompt_token_count: prompt_tokens,
                        max_tokens: self.lab.max_tokens,
                        sampler: "greedy".to_string(),
                    },
                    diagnostics: diag,
                    success: true,
                    error: None,
                };
                self.lab.run_history.push(record);
                while self.lab.run_history.len() > super::diagnostics::MAX_RUN_HISTORY {
                    self.lab.run_history.remove(0);
                }
                self.lab.selected_history_id = Some(id);
                self.screen = Screen::GenerationResult;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => {
                self.lab.last_run_success = false;
                self.lab.last_error = Some(error.to_string());
                self.lab.live.phase = GenerationPhase::Failed;
                self.show_error(error, Screen::GenerationInput);
            }
        }
    }

    pub fn export_finished(&mut self, result: Result<PathBuf, TuiError>) {
        match result {
            Ok(path) => {
                self.status_message = Some(format!("Exported: {}", path.display()));
                let target = self
                    .lab
                    .selected_history_id
                    .map(|_| Screen::RunDetail)
                    .unwrap_or(Screen::GenerationResult);
                self.screen = target;
                self.menu_index = 0;
                self.error = None;
            }
            Err(error) => {
                self.screen = Screen::ExportPath;
                self.error = Some(error);
            }
        }
    }

    fn return_from_runtime_activation(&mut self) {
        self.screen = if self.plan_origin == PlanOrigin::LoadedPlan {
            Screen::PlanCompatibility
        } else {
            Screen::PlanReview
        };
        self.menu_index = 0;
    }

    fn leave_runtime_session(&mut self) {
        self.active_runtime = None;
        self.prompt_input.clear();
        self.generation_result = None;
        self.screen = Screen::PlanReview;
        self.menu_index = 0;
    }

    fn start_new_model_flow(&mut self) {
        self.active_runtime = None;
        self.prompt_input.clear();
        self.generation_result = None;
        self.model_flow = ModelFlow::NewPlan;
        self.plan_origin = PlanOrigin::NewPlan;
        self.persisted_plan = None;
        self.compatibility_report = None;
        self.planning_session = None;
        self.execution_plan = None;
        self.calibration_result = None;
        self.model_input.clear();
        self.screen = Screen::ModelInput;
        self.menu_index = 0;
    }

    fn return_loaded_model_to_planning(&mut self) {
        self.active_runtime = None;
        self.prompt_input.clear();
        self.generation_result = None;
        if let Some(persisted) = self.persisted_plan.as_ref() {
            self.ram_input = format!("{}B", persisted.ram_budget_bytes);
            self.mode = persisted.mode;
        }
        self.model_flow = ModelFlow::NewPlan;
        self.plan_origin = PlanOrigin::NewPlan;
        self.persisted_plan = None;
        self.compatibility_report = None;
        self.execution_plan = None;
        self.calibration_result = None;
        self.screen = Screen::Preferences;
        self.menu_index = 0;
        self.preference_editing = false;
    }

    fn show_error(&mut self, error: TuiError, return_to: Screen) {
        self.screen = Screen::Error;
        self.error = Some(error);
        self.error_return = return_to;
        self.menu_index = 0;
    }

    fn handle_welcome(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Up => self.move_menu_up(3),
            UiCommand::Down => self.move_menu_down(3),
            UiCommand::Enter if self.menu_index == 0 => {
                self.start_new_model_flow();
            }
            UiCommand::Enter if self.menu_index == 1 => {
                self.screen = Screen::LoadPlan;
                self.load_path_input.clear();
                self.menu_index = 0;
            }
            UiCommand::Enter => self.should_quit = true,
            _ => {}
        }
        None
    }

    fn handle_load_plan(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Character(character) if !character.is_control() => {
                self.load_path_input.push(character)
            }
            UiCommand::Backspace => {
                self.load_path_input.pop();
            }
            UiCommand::Back => {
                self.screen = Screen::Welcome;
                self.menu_index = 1;
            }
            UiCommand::Enter if self.load_path_input.is_empty() => {
                self.error = Some(TuiError::Input {
                    field: "persisted plan path",
                    message: "a path is required".to_string(),
                });
            }
            UiCommand::Enter => {
                self.screen = Screen::Analyzing;
                return Some(AppAction::LoadPlan(PathBuf::from(
                    self.load_path_input.as_str(),
                )));
            }
            _ => {}
        }
        None
    }

    fn handle_model_input(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Character(character) if !character.is_control() => {
                self.model_input.push(character)
            }
            UiCommand::Backspace => {
                self.model_input.pop();
            }
            UiCommand::Back => {
                self.screen = if self.model_flow == ModelFlow::LoadedPlan {
                    Screen::LoadPlan
                } else {
                    Screen::Welcome
                };
                self.menu_index = 0;
            }
            UiCommand::Enter if self.model_input.is_empty() => {
                self.error = Some(TuiError::Input {
                    field: "model path",
                    message: "a path is required".to_string(),
                });
            }
            UiCommand::Enter => {
                self.screen = Screen::Analyzing;
                return Some(AppAction::ValidateModelPath(PathBuf::from(
                    self.model_input.as_str(),
                )));
            }
            _ => {}
        }
        None
    }

    fn handle_preferences(&mut self, command: UiCommand) -> Option<AppAction> {
        if self.preference_editing {
            match command {
                UiCommand::Character(character) if !character.is_control() => {
                    self.edit_preference_character(character)
                }
                UiCommand::Backspace => self.edit_preference_backspace(),
                UiCommand::Enter | UiCommand::Back => {
                    self.preference_editing = false;
                    self.sync_ram_from_input();
                }
                UiCommand::Up => {
                    self.preference_editing = false;
                    self.sync_ram_from_input();
                    self.move_menu_up(self.preference_count());
                }
                UiCommand::Down => {
                    self.preference_editing = false;
                    self.sync_ram_from_input();
                    self.move_menu_down(self.preference_count());
                }
                _ => {}
            }
            return None;
        }

        match command {
            UiCommand::Up => self.move_menu_up(self.preference_count()),
            UiCommand::Down => self.move_menu_down(self.preference_count()),
            UiCommand::Left => self.change_preference_value(false),
            UiCommand::Right => self.change_preference_value(true),
            UiCommand::Back => {
                self.preference_editing = false;
                self.screen = Screen::ModelInput;
                self.menu_index = 0;
            }
            UiCommand::Enter => match self.preference_field() {
                PreferenceField::Ram => {
                    // Only enter edit mode when Custom preset is selected.
                    if self.lab.ram_preset_index == RamPreset::ALL.len() - 1 {
                        self.preference_editing = true;
                    } else {
                        self.cycle_ram_preset(true);
                    }
                }
                PreferenceField::AdvancedThreads | PreferenceField::AdvancedCacheCapacity => {
                    self.preference_editing = true;
                }
                PreferenceField::Mode
                | PreferenceField::AdvancedCache
                | PreferenceField::AdvancedReadCoalescing
                | PreferenceField::AdvancedGroupedBuffer => {
                    self.change_preference_value(true);
                }
                PreferenceField::TestPresetEntry => {
                    self.screen = Screen::TestPreset;
                    self.menu_index = self.lab.test_preset_index;
                }
                PreferenceField::Continue => match self.build_user_profile() {
                    Ok(profile) => {
                        self.user_profile = Some(profile);
                        self.preference_editing = false;
                        self.active_runtime = None;
                        self.prompt_input.clear();
                        self.generation_result = None;
                        self.planning_session = None;
                        self.calibration_result = None;
                        self.execution_plan = None;
                        self.persisted_plan = None;
                        self.compatibility_report = None;
                        self.plan_origin = PlanOrigin::NewPlan;
                        self.model_flow = ModelFlow::NewPlan;
                        self.screen = Screen::Analyzing;
                        return Some(AppAction::Analyze);
                    }
                    Err(error) => self.error = Some(error),
                },
            },
            _ => {}
        }
        None
    }

    fn cycle_ram_preset(&mut self, forward: bool) {
        let len = RamPreset::ALL.len();
        if forward {
            self.lab.ram_preset_index = (self.lab.ram_preset_index + 1) % len;
        } else {
            self.lab.ram_preset_index = (self.lab.ram_preset_index + len - 1) % len;
        }
        let preset = RamPreset::ALL[self.lab.ram_preset_index];
        if let Some(bytes) = preset.bytes() {
            self.ram_input = normalize_memory_input(bytes);
        } else {
            self.ram_input = self.lab.custom_ram_input.clone();
        }
    }

    fn sync_ram_from_input(&mut self) {
        if self.lab.ram_preset_index == RamPreset::ALL.len() - 1 {
            self.lab.custom_ram_input = self.ram_input.clone();
        }
    }

    fn handle_test_preset(&mut self, command: UiCommand) -> Option<AppAction> {
        let count = TestPreset::ALL.len() + 1; // items + back option shown as footer Escape
        match command {
            UiCommand::Up => self.move_menu_up(TestPreset::ALL.len()),
            UiCommand::Down => self.move_menu_down(TestPreset::ALL.len()),
            UiCommand::Back => {
                self.screen = Screen::Preferences;
                self.menu_index = if self.mode == OperatingMode::Advanced {
                    7
                } else {
                    2
                };
            }
            UiCommand::Enter => {
                let preset = TestPreset::ALL[self.menu_index.min(TestPreset::ALL.len() - 1)];
                self.apply_test_preset(preset);
                self.screen = Screen::Preferences;
                self.menu_index = self.preference_count() - 1;
            }
            _ => {
                let _ = count;
            }
        }
        None
    }

    fn handle_review_config(&mut self, command: UiCommand) -> Option<AppAction> {
        // Items: max_tokens row, start button, back button  (3 lines; max_tokens row itself cycles preset)
        let count = 4;
        match command {
            UiCommand::Up => self.move_menu_up(count),
            UiCommand::Down => self.move_menu_down(count),
            UiCommand::Left | UiCommand::Right if self.menu_index == 0 => {
                let dir = command == UiCommand::Right;
                let len = MaxTokensPreset::ALL.len();
                if dir {
                    self.lab.max_tokens_preset_index = (self.lab.max_tokens_preset_index + 1) % len;
                } else {
                    self.lab.max_tokens_preset_index =
                        (self.lab.max_tokens_preset_index + len - 1) % len;
                }
                let p = MaxTokensPreset::ALL[self.lab.max_tokens_preset_index];
                if let Some(v) = p.value() {
                    self.lab.max_tokens = v;
                } else {
                    // Custom: start editing
                    self.preference_editing = true;
                }
            }
            UiCommand::Back => {
                self.active_runtime = None;
                self.screen = Screen::PlanValid;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 0 => {
                if self.lab.max_tokens_preset_index == MaxTokensPreset::ALL.len() - 1 {
                    self.preference_editing = true;
                } else {
                    // cycle
                    self.cycle_max_tokens();
                }
            }
            UiCommand::Enter if self.menu_index == 1 => {
                // Go to prompt input
                self.prompt_input.clear();
                self.lab.prompt_input_before_run.clear();
                self.screen = Screen::GenerationInput;
                self.menu_index = 0;
            }
            UiCommand::Enter => {
                self.active_runtime = None;
                self.screen = Screen::PlanValid;
                self.menu_index = 0;
            }
            _ => {}
        }
        None
    }

    fn cycle_max_tokens(&mut self) {
        let len = MaxTokensPreset::ALL.len();
        self.lab.max_tokens_preset_index = (self.lab.max_tokens_preset_index + 1) % len;
        if let Some(v) = MaxTokensPreset::ALL[self.lab.max_tokens_preset_index].value() {
            self.lab.max_tokens = v;
        }
    }

    fn handle_run_history(&mut self, command: UiCommand) -> Option<AppAction> {
        let count = self.lab.run_history.len() + 2; // entries + compare + back
        match command {
            UiCommand::Up => self.move_menu_up(count.max(1)),
            UiCommand::Down => self.move_menu_down(count.max(1)),
            UiCommand::Back => {
                self.screen = Screen::GenerationResult;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index < self.lab.run_history.len() => {
                let run = &self.lab.run_history[self.menu_index];
                self.lab.selected_history_id = Some(run.run_id);
                self.screen = Screen::RunDetail;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == self.lab.run_history.len() => {
                // Compare
                if self.lab.run_history.len() >= 2 {
                    self.lab.compare_first_id = None;
                    self.lab.comparison = None;
                    self.screen = Screen::RunCompareSelect;
                    self.menu_index = 0;
                }
            }
            UiCommand::Enter => {
                self.screen = Screen::GenerationResult;
                self.menu_index = 0;
            }
            _ => {}
        }
        None
    }

    fn handle_run_detail(&mut self, command: UiCommand) -> Option<AppAction> {
        // Items: 0 = back to history, 1 = Export JSON
        match command {
            UiCommand::Up | UiCommand::Down => self.menu_index = 1 - self.menu_index.min(1),
            UiCommand::Back => {
                self.screen = Screen::RunHistory;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::RunHistory;
                self.menu_index = 0;
            }
            UiCommand::Enter => {
                self.lab.export_path_input.clear();
                if let Some(id) = self.lab.selected_history_id {
                    if let Some(r) = self.selected_run(id) {
                        let default_path = format!("ramforge-run-{}.json", r.run_id);
                        self.lab.export_path_input = default_path;
                    }
                }
                self.screen = Screen::ExportPath;
                self.menu_index = 0;
            }
            _ => {}
        }
        None
    }

    fn handle_run_compare_select(&mut self, command: UiCommand) -> Option<AppAction> {
        let count = self.lab.run_history.len() + 1;
        match command {
            UiCommand::Up => self.move_menu_up(count.max(1)),
            UiCommand::Down => self.move_menu_down(count.max(1)),
            UiCommand::Back => {
                if self.lab.compare_first_id.is_some() {
                    self.lab.compare_first_id = None;
                } else {
                    self.screen = Screen::RunHistory;
                    self.menu_index = 0;
                }
            }
            UiCommand::Enter if self.menu_index < self.lab.run_history.len() => {
                let run = &self.lab.run_history[self.menu_index];
                let id = run.run_id;
                match self.lab.compare_first_id {
                    None => {
                        self.lab.compare_first_id = Some(id);
                    }
                    Some(first) if first == id => {
                        self.lab.compare_first_id = None;
                    }
                    Some(first) => {
                        if let (Some(a), Some(b)) =
                            (self.selected_run(first), self.selected_run(id))
                        {
                            self.lab.comparison = Some(super::diagnostics::compare_runs(a, b));
                            self.screen = Screen::RunCompare;
                            self.menu_index = 0;
                        }
                    }
                }
            }
            UiCommand::Enter => {
                self.screen = Screen::RunHistory;
                self.menu_index = 0;
            }
            _ => {}
        }
        None
    }

    fn handle_run_compare(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Back | UiCommand::Enter => {
                self.lab.comparison = None;
                self.lab.compare_first_id = None;
                self.screen = Screen::RunHistory;
                self.menu_index = 0;
            }
            UiCommand::Up | UiCommand::Down => {}
            _ => {}
        }
        None
    }

    fn handle_export_path(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Character(character) if !character.is_control() => {
                self.lab.export_path_input.push(character)
            }
            UiCommand::Backspace => {
                self.lab.export_path_input.pop();
            }
            UiCommand::Back => {
                let target = self
                    .lab
                    .selected_history_id
                    .map(|_| Screen::RunDetail)
                    .unwrap_or(Screen::GenerationResult);
                self.screen = target;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.lab.export_path_input.is_empty() => {
                self.error = Some(TuiError::Input {
                    field: "export path",
                    message: "a destination path is required".to_string(),
                });
            }
            UiCommand::Enter => {
                if let Some(id) = self.lab.selected_history_id {
                    return Some(AppAction::ExportRun(
                        id,
                        PathBuf::from(self.lab.export_path_input.as_str()),
                    ));
                } else if let Some(id) = self.lab.run_history.last().map(|r| r.run_id) {
                    return Some(AppAction::ExportRun(
                        id,
                        PathBuf::from(self.lab.export_path_input.as_str()),
                    ));
                }
            }
            _ => {}
        }
        None
    }

    fn handle_model_info(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Up | UiCommand::Down => self.menu_index = 1 - self.menu_index.min(1),
            UiCommand::Back => {
                self.screen = Screen::Preferences;
                self.menu_index = 0;
                self.preference_editing = false;
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::CalibrationSelect;
                self.menu_index = self.calibration_index;
            }
            UiCommand::Enter => {
                self.screen = Screen::Preferences;
                self.menu_index = 0;
                self.preference_editing = false;
            }
            _ => {}
        }
        None
    }

    fn handle_calibration_select(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Up | UiCommand::Left => self.move_menu_up(4),
            UiCommand::Down | UiCommand::Right => self.move_menu_down(4),
            UiCommand::Back => {
                self.screen = Screen::ModelInfo;
                self.menu_index = 0;
            }
            UiCommand::Enter => {
                self.calibration_index = self.menu_index;
                let level = self.calibration_level();
                if let Some(user) = self.user_profile.as_mut() {
                    user.calibration_level = level;
                }
                self.calibration_result = None;
                if level == CalibrationLevel::None {
                    self.screen = Screen::Analyzing;
                    return Some(AppAction::CreatePlan);
                }
                self.screen = Screen::Calibrating;
                self.calibration_complete = false;
                return Some(AppAction::RunCalibration);
            }
            _ => {}
        }
        None
    }

    fn handle_calibrating(&mut self, command: UiCommand) -> Option<AppAction> {
        if command == UiCommand::Back {
            self.screen = Screen::CalibrationSelect;
            self.menu_index = self.calibration_index;
            return None;
        }
        if command == UiCommand::Enter && self.calibration_complete {
            self.screen = Screen::Analyzing;
            return Some(AppAction::CreatePlan);
        }
        None
    }

    fn handle_plan_review(&mut self, command: UiCommand) -> Option<AppAction> {
        if self.plan_origin == PlanOrigin::LoadedPlan {
            return self.handle_loaded_plan_review(command);
        }
        match command {
            UiCommand::Up | UiCommand::Down => self.menu_index = 1 - self.menu_index.min(1),
            UiCommand::Back => {
                self.screen = Screen::CalibrationSelect;
                self.menu_index = self.calibration_index;
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::PlanValidation;
                self.error = None;
                return Some(AppAction::ValidatePlan);
            }
            UiCommand::Enter => {
                self.screen = Screen::Preferences;
                self.menu_index = 0;
                self.preference_editing = false;
            }
            _ => {}
        }
        None
    }

    fn handle_loaded_plan_review(&mut self, command: UiCommand) -> Option<AppAction> {
        let compatible = self
            .compatibility_report
            .as_ref()
            .is_some_and(|report| report.state != PlanCompatibility::Incompatible);
        match command {
            UiCommand::Up | UiCommand::Down => self.menu_index = 1 - self.menu_index.min(1),
            UiCommand::Back => {
                self.screen = Screen::PlanCompatibility;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 0 && compatible => {
                self.screen = Screen::PlanValidation;
                self.error = None;
                return Some(AppAction::ValidatePlan);
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::PlanCompatibility;
                self.menu_index = 0;
            }
            UiCommand::Enter => self.return_loaded_model_to_planning(),
            _ => {}
        }
        None
    }

    fn handle_plan_valid(&mut self, command: UiCommand) -> Option<AppAction> {
        let item_count = if self.plan_origin == PlanOrigin::LoadedPlan {
            2
        } else {
            3
        };
        match command {
            UiCommand::Up => self.move_menu_up(item_count),
            UiCommand::Down => self.move_menu_down(item_count),
            UiCommand::Back => self.return_from_runtime_activation(),
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::RuntimeActivation;
                self.error = None;
                return Some(AppAction::ActivateRuntime);
            }
            UiCommand::Enter if self.plan_origin == PlanOrigin::NewPlan && self.menu_index == 1 => {
                self.screen = Screen::SavePlan;
                self.save_path_input.clear();
                self.error = None;
            }
            UiCommand::Enter => self.return_from_runtime_activation(),
            _ => {}
        }
        None
    }

    fn handle_generation_input(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Character(character) if !character.is_control() => {
                self.prompt_input.push(character)
            }
            UiCommand::Backspace => {
                self.prompt_input.pop();
            }
            UiCommand::Back => {
                self.screen = Screen::ReviewConfig;
                self.menu_index = 1;
            }
            UiCommand::Enter if self.prompt_input.is_empty() => {
                self.error = Some(TuiError::Input {
                    field: "prompt",
                    message: "prompt text is required".to_string(),
                });
            }
            UiCommand::Enter => {
                if self.lab.max_tokens == 0 {
                    self.error = Some(TuiError::Input {
                        field: "max tokens",
                        message: "max tokens must be greater than zero".to_string(),
                    });
                    return None;
                }
                self.screen = Screen::GenerationRunning;
                self.generation_result = None;
                self.lab.prompt_input_before_run = self.prompt_input.clone();
                self.start_generation_live();
                return Some(AppAction::GeneratePrompt(
                    self.prompt_input.clone(),
                    self.lab.max_tokens,
                ));
            }
            _ => {}
        }
        None
    }

    fn handle_generation_result(&mut self, command: UiCommand) -> Option<AppAction> {
        // Menu items:
        // 0: Generate another
        // 1: Run history
        // 2: Export last run JSON
        // 3: Return to plan
        let count = 4;
        match command {
            UiCommand::Up => self.move_menu_up(count),
            UiCommand::Down => self.move_menu_down(count),
            UiCommand::Back => self.leave_runtime_session(),
            UiCommand::Enter if self.menu_index == 0 => {
                self.prompt_input.clear();
                self.generation_result = None;
                self.screen = Screen::GenerationInput;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 1 => {
                self.screen = Screen::RunHistory;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 2 => {
                self.lab.export_path_input.clear();
                if let Some(r) = self.lab.run_history.last() {
                    self.lab.selected_history_id = Some(r.run_id);
                    self.lab.export_path_input = format!("ramforge-run-{}.json", r.run_id);
                }
                self.screen = Screen::ExportPath;
                self.menu_index = 0;
            }
            UiCommand::Enter => self.leave_runtime_session(),
            _ => {}
        }
        None
    }

    fn handle_plan_compatibility(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Up => self.move_menu_up(3),
            UiCommand::Down => self.move_menu_down(3),
            UiCommand::Back => {
                self.screen = Screen::LoadPlan;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::PlanReview;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 1 => self.return_loaded_model_to_planning(),
            UiCommand::Enter => self.start_new_model_flow(),
            _ => {}
        }
        None
    }

    fn handle_save_plan(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Character(character) if !character.is_control() => {
                self.save_path_input.push(character)
            }
            UiCommand::Backspace => {
                self.save_path_input.pop();
            }
            UiCommand::Back => {
                self.screen = Screen::PlanValid;
                self.menu_index = 0;
                self.error = None;
            }
            UiCommand::Enter if self.save_path_input.is_empty() => {
                self.error = Some(TuiError::Input {
                    field: "save path",
                    message: "a destination path is required".to_string(),
                });
            }
            UiCommand::Enter => {
                return Some(AppAction::SavePlan(PathBuf::from(
                    self.save_path_input.as_str(),
                )))
            }
            _ => {}
        }
        None
    }

    fn move_menu_up(&mut self, item_count: usize) {
        self.menu_index = if self.menu_index == 0 {
            item_count - 1
        } else {
            self.menu_index - 1
        };
    }

    fn move_menu_down(&mut self, item_count: usize) {
        self.menu_index = (self.menu_index + 1) % item_count;
    }

    fn build_user_profile(&self) -> Result<UserProfile, TuiError> {
        let ram_budget_bytes =
            parse_memory_size(&self.ram_input).map_err(|error| TuiError::Input {
                field: "RAM budget",
                message: error.to_string(),
            })?;
        let mut profile = UserProfile::new(ram_budget_bytes, self.mode);
        if self.mode == OperatingMode::Advanced {
            profile.advanced = AdvancedOverrides {
                thread_count: parse_optional_usize(
                    &self.advanced_thread_input,
                    "advanced CPU threads",
                )?,
                layer_cache_enabled: self.advanced_cache.as_option(),
                layer_cache_capacity_bytes: parse_optional_memory(
                    &self.advanced_cache_capacity_input,
                    "advanced cache capacity",
                )?,
                read_coalescing_enabled: self.advanced_read_coalescing.as_option(),
                grouped_read_buffer_reuse_enabled: self.advanced_grouped_buffer.as_option(),
                gpu_device_id: None,
            };
        }
        profile.validate().map_err(|error| TuiError::Input {
            field: "execution preferences",
            message: error.to_string(),
        })?;
        Ok(profile)
    }

    fn preference_field(&self) -> PreferenceField {
        if self.mode != OperatingMode::Advanced {
            return match self.menu_index.min(3) {
                0 => PreferenceField::Ram,
                1 => PreferenceField::Mode,
                2 => PreferenceField::TestPresetEntry,
                _ => PreferenceField::Continue,
            };
        }
        match self.menu_index.min(8) {
            0 => PreferenceField::Ram,
            1 => PreferenceField::Mode,
            2 => PreferenceField::AdvancedThreads,
            3 => PreferenceField::AdvancedCache,
            4 => PreferenceField::AdvancedCacheCapacity,
            5 => PreferenceField::AdvancedReadCoalescing,
            6 => PreferenceField::AdvancedGroupedBuffer,
            7 => PreferenceField::TestPresetEntry,
            _ => PreferenceField::Continue,
        }
    }

    fn change_preference_value(&mut self, forward: bool) {
        match self.preference_field() {
            PreferenceField::Mode => {
                self.mode = if forward {
                    next_mode(self.mode)
                } else {
                    previous_mode(self.mode)
                };
                if self.menu_index >= self.preference_count() {
                    self.menu_index = self.preference_count() - 1;
                }
            }
            PreferenceField::Ram => {
                self.cycle_ram_preset(forward);
            }
            PreferenceField::AdvancedCache => {
                self.advanced_cache = if forward {
                    self.advanced_cache.next()
                } else {
                    self.advanced_cache.previous()
                }
            }
            PreferenceField::AdvancedReadCoalescing => {
                self.advanced_read_coalescing = if forward {
                    self.advanced_read_coalescing.next()
                } else {
                    self.advanced_read_coalescing.previous()
                }
            }
            PreferenceField::AdvancedGroupedBuffer => {
                self.advanced_grouped_buffer = if forward {
                    self.advanced_grouped_buffer.next()
                } else {
                    self.advanced_grouped_buffer.previous()
                }
            }
            PreferenceField::AdvancedThreads
            | PreferenceField::AdvancedCacheCapacity
            | PreferenceField::TestPresetEntry
            | PreferenceField::Continue => {}
        }
    }

    fn edit_preference_character(&mut self, character: char) {
        match self.preference_field() {
            PreferenceField::Ram => {
                self.ram_input.push(character);
                self.lab.custom_ram_input = self.ram_input.clone();
            }
            PreferenceField::AdvancedThreads => self.advanced_thread_input.push(character),
            PreferenceField::AdvancedCacheCapacity => {
                self.advanced_cache_capacity_input.push(character)
            }
            _ => {}
        }
        // Handle custom max tokens editing when on review config
        if self.screen == Screen::ReviewConfig
            && self.menu_index == 0
            && self.lab.max_tokens_preset_index == MaxTokensPreset::ALL.len() - 1
        {
            if character.is_ascii_digit() {
                self.lab.custom_max_tokens_input.push(character);
                if let Ok(v) = self.lab.custom_max_tokens_input.parse::<usize>() {
                    self.lab.max_tokens = v;
                }
            }
        }
    }

    fn edit_preference_backspace(&mut self) {
        match self.preference_field() {
            PreferenceField::Ram => {
                self.ram_input.pop();
                self.lab.custom_ram_input = self.ram_input.clone();
            }
            PreferenceField::AdvancedThreads => {
                self.advanced_thread_input.pop();
            }
            PreferenceField::AdvancedCacheCapacity => {
                self.advanced_cache_capacity_input.pop();
            }
            _ => {}
        }
        if self.screen == Screen::ReviewConfig
            && self.menu_index == 0
            && self.lab.max_tokens_preset_index == MaxTokensPreset::ALL.len() - 1
        {
            self.lab.custom_max_tokens_input.pop();
            if let Ok(v) = self.lab.custom_max_tokens_input.parse::<usize>() {
                self.lab.max_tokens = v;
            } else if self.lab.custom_max_tokens_input.is_empty() {
                self.lab.max_tokens = 0;
            }
        }
    }
}

fn phase_to_u64(phase: GenerationPhase) -> u64 {
    match phase {
        GenerationPhase::Prompt => 0,
        GenerationPhase::Decode => 1,
        GenerationPhase::Finished => 2,
        GenerationPhase::Failed => 3,
    }
}

fn normalize_memory_input(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    if bytes % GIB == 0 {
        format!("{}GiB", bytes / GIB)
    } else if bytes % MIB == 0 {
        format!("{}MiB", bytes / MIB)
    } else {
        format!("{bytes}B")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreferenceField {
    Ram,
    Mode,
    AdvancedThreads,
    AdvancedCache,
    AdvancedCacheCapacity,
    AdvancedReadCoalescing,
    AdvancedGroupedBuffer,
    TestPresetEntry,
    Continue,
}

fn parse_optional_usize(value: &str, field: &'static str) -> Result<Option<usize>, TuiError> {
    if value.is_empty() {
        return Ok(None);
    }
    value
        .parse::<usize>()
        .map(Some)
        .map_err(|error| TuiError::Input {
            field,
            message: error.to_string(),
        })
}

fn parse_optional_memory(value: &str, field: &'static str) -> Result<Option<u64>, TuiError> {
    if value.is_empty() {
        return Ok(None);
    }
    parse_memory_size(value)
        .map(Some)
        .map_err(|error| TuiError::Input {
            field,
            message: error.to_string(),
        })
}

pub fn mode_label(mode: OperatingMode) -> &'static str {
    match mode {
        OperatingMode::BackgroundConstrained => "Background / Constrained",
        OperatingMode::BalancedNormal => "Balanced / Normal",
        OperatingMode::MaximumPerformance => "Maximum Performance",
        OperatingMode::Advanced => "Advanced",
    }
}

fn next_mode(mode: OperatingMode) -> OperatingMode {
    match mode {
        OperatingMode::BackgroundConstrained => OperatingMode::BalancedNormal,
        OperatingMode::BalancedNormal => OperatingMode::MaximumPerformance,
        OperatingMode::MaximumPerformance => OperatingMode::Advanced,
        OperatingMode::Advanced => OperatingMode::BackgroundConstrained,
    }
}

fn previous_mode(mode: OperatingMode) -> OperatingMode {
    match mode {
        OperatingMode::BackgroundConstrained => OperatingMode::Advanced,
        OperatingMode::BalancedNormal => OperatingMode::BackgroundConstrained,
        OperatingMode::MaximumPerformance => OperatingMode::BalancedNormal,
        OperatingMode::Advanced => OperatingMode::MaximumPerformance,
    }
}

pub fn calibration_level_for_index(index: usize) -> CalibrationLevel {
    match index % 4 {
        0 => CalibrationLevel::None,
        1 => CalibrationLevel::Quick,
        2 => CalibrationLevel::Standard,
        _ => CalibrationLevel::Thorough,
    }
}

pub fn calibration_level_label(level: CalibrationLevel) -> &'static str {
    match level {
        CalibrationLevel::None => "None",
        CalibrationLevel::Quick => "Quick",
        CalibrationLevel::Standard => "Standard",
        CalibrationLevel::Thorough => "Thorough",
    }
}

fn calibration_kind_label(kind: CalibrationTestKind) -> String {
    match kind {
        CalibrationTestKind::CpuFloatDot => "CPU scalar F32 dot".to_string(),
        CalibrationTestKind::MemoryCopyBandwidth => "Memory copy bandwidth".to_string(),
        CalibrationTestKind::SequentialStorageRead => "Sequential storage read".to_string(),
        CalibrationTestKind::StorageReadLatency => "Storage read latency".to_string(),
        CalibrationTestKind::QuantizedRowDecode(format) => {
            format!("{} row decode", format.format_name())
        }
        CalibrationTestKind::StrategyLatency(strategy) => {
            format!("{strategy:?} latency")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_welcome_and_does_not_request_generation() {
        let app = TuiApp::default();
        assert_eq!(app.screen, Screen::Welcome);
        assert!(!app.should_quit);
        assert!(app.execution_plan.is_none());
    }

    #[test]
    fn welcome_uses_arrow_navigation_only() {
        let mut app = TuiApp::default();
        app.handle(UiCommand::Down);
        assert_eq!(app.menu_index, 1);
        app.handle(UiCommand::Up);
        assert_eq!(app.menu_index, 0);
        app.handle(UiCommand::Up);
        assert_eq!(app.menu_index, 2);

        app.handle(UiCommand::Character('j'));
        app.handle(UiCommand::Character('k'));
        assert_eq!(app.menu_index, 2);
    }

    #[test]
    fn welcome_exposes_load_plan_and_back_navigation() {
        let mut app = TuiApp::default();
        app.handle(UiCommand::Down);
        app.handle(UiCommand::Enter);
        assert_eq!(app.screen, Screen::LoadPlan);
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::Welcome);
        assert_eq!(app.menu_index, 1);
    }

    #[test]
    fn load_plan_requests_exact_path_and_recovers_from_errors() {
        let mut app = TuiApp::default();
        app.screen = Screen::LoadPlan;
        app.load_path_input = "relative/plan.rfp".to_string();
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::LoadPlan(PathBuf::from("relative/plan.rfp")))
        );
        app.load_finished(Err(TuiError::Persistence(
            PlanPersistenceError::InvalidMagic,
        )));
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(
            app.error.as_ref().unwrap().title(),
            "Plan persistence failed"
        );
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::LoadPlan);
    }

    #[test]
    fn model_input_accepts_exact_path_and_rejects_empty_input() {
        let mut app = TuiApp::default();
        app.handle(UiCommand::Enter);
        assert_eq!(app.screen, Screen::ModelInput);
        assert_eq!(app.active_input_context(), InputContext::ModelPath);
        app.handle(UiCommand::Character('j'));
        app.handle(UiCommand::Character('k'));
        assert_eq!(app.model_input, "jk");
        app.model_input.clear();
        assert!(app.handle(UiCommand::Enter).is_none());
        assert!(matches!(app.error.as_ref(), Some(TuiError::Input { .. })));

        app.model_input = "relative/model.gguf".to_string();
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::ValidateModelPath(PathBuf::from(
                "relative/model.gguf"
            )))
        );
        assert!(app.model_path_validation_finished(Ok(())).is_none());
        assert_eq!(app.screen, Screen::Preferences);
        assert_eq!(app.active_input_context(), InputContext::None);
        let stored_path = app.model_input.clone();
        app.handle(UiCommand::Character('x'));
        assert_eq!(app.model_input, stored_path);
        app.handle(UiCommand::Down);
        assert_eq!(app.menu_index, 1);

        app.screen = Screen::Analyzing;
        assert!(app
            .model_path_validation_finished(Err(TuiError::Storage(
                StorageDiscoveryError::InvalidPath,
            )))
            .is_none());
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(
            app.error.as_ref().unwrap().title(),
            "Storage discovery failed"
        );
    }

    #[test]
    fn loaded_plan_model_context_requests_compatibility_analysis() {
        let mut app = TuiApp::default();
        app.model_flow = ModelFlow::LoadedPlan;
        app.screen = Screen::Analyzing;
        assert_eq!(
            app.model_path_validation_finished(Ok(())),
            Some(AppAction::AnalyzeLoadedPlan)
        );
        assert_eq!(app.screen, Screen::Analyzing);
    }

    #[test]
    fn preference_left_and_right_change_horizontal_options() {
        let mut app = TuiApp::default();
        app.screen = Screen::Preferences;
        app.menu_index = 1;
        assert_eq!(app.mode, OperatingMode::BalancedNormal);
        app.handle(UiCommand::Right);
        assert_eq!(app.mode, OperatingMode::MaximumPerformance);
        app.handle(UiCommand::Left);
        assert_eq!(app.mode, OperatingMode::BalancedNormal);
        assert!(!app.preference_editing);
    }

    #[test]
    fn preference_text_editing_requires_explicit_enter() {
        let mut app = TuiApp::default();
        app.screen = Screen::Preferences;
        app.menu_index = 0;
        assert_eq!(app.ram_input, "1GiB");
        // Navigate to Custom RAM preset first. Starting at index 1 (1 GiB), need 6 Rights.
        for _ in 0..6 {
            app.handle(UiCommand::Right);
        }
        assert_eq!(app.lab.ram_preset_index, RamPreset::ALL.len() - 1);
        app.ram_input.clear();
        app.lab.custom_ram_input.clear();
        app.handle(UiCommand::Character('8'));
        assert!(app.ram_input.is_empty()); // must press Enter first to edit
        app.handle(UiCommand::Enter);
        assert_eq!(app.active_input_context(), InputContext::PreferenceValue);
        app.handle(UiCommand::Character('8'));
        assert!(app.ram_input.ends_with('8'));
        app.handle(UiCommand::Enter);
        assert_eq!(app.active_input_context(), InputContext::None);
    }

    #[test]
    fn preference_changes_use_existing_profile_contracts_without_clamping() {
        let mut app = TuiApp::default();
        app.screen = Screen::Preferences;
        app.ram_input = "2GiB".to_string();
        app.menu_index = 1;
        app.handle(UiCommand::Enter);
        assert_eq!(app.mode, OperatingMode::MaximumPerformance);
        app.menu_index = 3; // Analyze (3rd index in non-advanced mode with Test preset)
        assert_eq!(app.handle(UiCommand::Enter), Some(AppAction::Analyze));
        assert_eq!(
            app.user_profile.as_ref().unwrap().ram_budget_bytes,
            2 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn advanced_preferences_are_passed_to_user_profile_validation() {
        let mut app = TuiApp::default();
        app.screen = Screen::Preferences;
        app.ram_input = "1GiB".to_string();
        app.mode = OperatingMode::Advanced;
        app.advanced_thread_input = "0".to_string();
        app.menu_index = 8; // Analyze in advanced mode (after Test preset entry)
        assert!(app.handle(UiCommand::Enter).is_none());
        assert!(matches!(app.error.as_ref(), Some(TuiError::Input { .. })));
        assert_eq!(app.screen, Screen::Preferences);
    }

    #[test]
    fn calibration_selection_is_explicit() {
        let mut app = TuiApp::default();
        app.screen = Screen::CalibrationSelect;
        app.user_profile = Some(UserProfile::new(1024, OperatingMode::BalancedNormal));
        app.menu_index = 0;
        assert_eq!(app.handle(UiCommand::Enter), Some(AppAction::CreatePlan));
        assert_eq!(
            app.user_profile.as_ref().unwrap().calibration_level,
            CalibrationLevel::None
        );

        app.screen = Screen::CalibrationSelect;
        app.menu_index = 2;
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::RunCalibration)
        );
        assert_eq!(
            app.user_profile.as_ref().unwrap().calibration_level,
            CalibrationLevel::Standard
        );
    }

    #[test]
    fn calibration_failure_returns_to_a_structured_error_state() {
        let mut app = TuiApp::default();
        app.screen = Screen::Calibrating;
        app.calibration_finished(
            Err(TuiError::Calibration(CalibrationError::InvalidPlan(
                "test failure",
            ))),
            Vec::new(),
        );
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(app.error.as_ref().unwrap().title(), "Calibration failed");
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::CalibrationSelect);
    }

    #[test]
    fn planner_failure_returns_to_a_structured_error_state() {
        let mut app = TuiApp::default();
        app.screen = Screen::Analyzing;
        app.planning_finished(Err(TuiError::Planning(OrchestrationError::Planning(
            ramforge_runtime::planner::PlannerError::Infeasible(
                ramforge_runtime::planner::FeasibilityRejection::ModelNotExecutable,
            ),
        ))));
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(app.error.as_ref().unwrap().title(), "Planning failed");
    }

    #[test]
    fn incompatible_loaded_plan_cannot_request_compilation() {
        let mut app = TuiApp::default();
        app.plan_origin = PlanOrigin::LoadedPlan;
        app.compatibility_report = Some(PlanCompatibilityReport {
            state: PlanCompatibility::Incompatible,
            reasons: Vec::new(),
        });
        app.screen = Screen::PlanReview;
        assert!(app.handle(UiCommand::Enter).is_none());
        assert_eq!(app.screen, Screen::PlanCompatibility);
    }

    #[test]
    fn compatible_loaded_plan_requires_explicit_compilation() {
        let mut app = TuiApp::default();
        app.plan_origin = PlanOrigin::LoadedPlan;
        app.compatibility_report = Some(PlanCompatibilityReport {
            state: PlanCompatibility::Valid,
            reasons: Vec::new(),
        });
        app.screen = Screen::PlanReview;
        assert_eq!(app.handle(UiCommand::Enter), Some(AppAction::ValidatePlan));
        assert_eq!(app.screen, Screen::PlanValidation);
        assert!(app.active_runtime.is_none());
        app.plan_validation_finished(Ok(()));
        assert_eq!(app.screen, Screen::PlanValid);
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::ActivateRuntime)
        );
    }

    #[test]
    fn recalibration_recommendation_never_starts_calibration_automatically() {
        let mut app = TuiApp::default();
        app.plan_origin = PlanOrigin::LoadedPlan;
        app.compatibility_report = Some(PlanCompatibilityReport {
            state: PlanCompatibility::CompatibleRecalibrationRecommended,
            reasons: Vec::new(),
        });
        app.screen = Screen::PlanCompatibility;
        app.menu_index = 1;
        assert!(app.handle(UiCommand::Enter).is_none());
        assert_eq!(app.screen, Screen::Preferences);
        assert!(app.calibration_result.is_none());
    }

    #[test]
    fn compiler_failure_stays_recoverable_on_plan_review() {
        let mut app = TuiApp::default();
        app.screen = Screen::PlanValidation;
        app.plan_validation_finished(Err(TuiError::Input {
            field: "compiler",
            message: "rejected".to_string(),
        }));
        assert_eq!(app.screen, Screen::PlanReview);
        assert!(app.error.is_some());
    }

    #[test]
    fn successful_validation_does_not_generate_and_preserves_explicit_save() {
        let mut app = TuiApp::default();
        app.screen = Screen::PlanValidation;
        app.plan_validation_finished(Ok(()));
        assert_eq!(app.screen, Screen::PlanValid);
        assert!(app.active_runtime.is_none());
        assert!(app.generation_result.is_none());
        app.handle(UiCommand::Down);
        app.handle(UiCommand::Enter);
        assert_eq!(app.screen, Screen::SavePlan);
        app.save_path_input = "/tmp/plan.rfp".to_string();
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::SavePlan(PathBuf::from("/tmp/plan.rfp")))
        );
    }

    #[test]
    fn runtime_activation_is_an_explicit_post_validation_action() {
        let mut app = TuiApp::default();
        app.screen = Screen::PlanValid;
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::ActivateRuntime)
        );
        assert_eq!(app.screen, Screen::RuntimeActivation);
        assert!(app.active_runtime.is_none());
    }

    #[test]
    fn runtime_construction_errors_are_distinct_and_recoverable() {
        let mut app = TuiApp::default();
        app.screen = Screen::RuntimeActivation;
        app.runtime_activation_finished(Err(TuiError::RuntimeConstruction(
            OrchestrationError::RuntimeSourceUnavailable,
        )));
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(
            app.error.as_ref().unwrap().title(),
            "Runtime construction failed"
        );
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::PlanReview);
    }

    #[test]
    fn prompt_submission_is_the_only_generation_action() {
        let mut app = TuiApp::default();
        app.screen = Screen::GenerationInput;
        assert_eq!(app.active_input_context(), InputContext::GenerationPrompt);
        app.handle(UiCommand::Character('H'));
        app.handle(UiCommand::Character('i'));
        assert_eq!(app.prompt_input, "Hi");
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::GeneratePrompt(
                "Hi".to_string(),
                app.lab.max_tokens
            ))
        );
        assert_eq!(app.screen, Screen::GenerationRunning);
        assert_eq!(app.active_input_context(), InputContext::None);
        app.handle(UiCommand::Character('x'));
        assert_eq!(app.prompt_input, "Hi");
    }

    #[test]
    fn generation_result_and_error_paths_are_recoverable() {
        let mut app = TuiApp::default();
        app.screen = Screen::GenerationRunning;
        app.prompt_input = "submitted".to_string();
        app.generation_finished(Ok(SinglePromptResult {
            generated_text: "result".to_string(),
            generated_token_count: 1,
        }));
        assert_eq!(app.screen, Screen::GenerationResult);
        assert_eq!(app.active_input_context(), InputContext::None);
        app.handle(UiCommand::Character('x'));
        assert_eq!(app.prompt_input, "submitted");
        app.handle(UiCommand::Enter);
        assert_eq!(app.screen, Screen::GenerationInput);
        assert!(app.generation_result.is_none());
        assert_eq!(app.active_input_context(), InputContext::GenerationPrompt);
        app.handle(UiCommand::Character('n'));
        assert_eq!(app.prompt_input, "n");

        app.screen = Screen::GenerationRunning;
        app.generation_finished(Err(TuiError::Generation("budget failure".to_string())));
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(app.error.as_ref().unwrap().title(), "Generation failed");
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::GenerationInput);
    }

    #[test]
    fn persistence_failure_stays_on_save_screen() {
        let mut app = TuiApp::default();
        app.screen = Screen::SavePlan;
        app.save_finished(Err(TuiError::DestinationExists(PathBuf::from(
            "/tmp/existing.rfp",
        ))));
        assert_eq!(app.screen, Screen::SavePlan);
        assert!(app.error.is_some());
    }

    #[test]
    fn back_navigation_and_quit_are_consistent() {
        let mut app = TuiApp::default();
        app.handle(UiCommand::Enter);
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::Welcome);
        app.handle(UiCommand::Quit);
        assert!(app.should_quit);
    }
}
