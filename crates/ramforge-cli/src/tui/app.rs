use std::fmt;
use std::path::PathBuf;

use ramforge_core::parse_memory_size;
use ramforge_runtime::calibration::{
    CalibrationError, CalibrationTask, CalibrationTestKind,
};
use ramforge_runtime::discovery::StorageDiscoveryError;
use ramforge_runtime::orchestration::{
    OrchestrationError, PlanningSession,
};
use ramforge_runtime::plan_persistence::{
    PersistedExecutionPlan, PlanCompatibilityReport, PlanPersistenceError,
};
use ramforge_runtime::planner::{
    AdvancedOverrides, CalibrationLevel, CalibrationResult, ExecutionPlan,
    ObservationStatus, ObservationStatusReason, OperatingMode, PlanCompatibility, UserProfile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Welcome,
    LoadPlan,
    ModelInput,
    Preferences,
    Analyzing,
    ModelInfo,
    CalibrationSelect,
    Calibrating,
    PlanReview,
    PlanValidation,
    PlanValid,
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
pub enum UiCommand {
    Up,
    Down,
    Enter,
    Back,
    Backspace,
    Quit,
    Character(char),
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppAction {
    LoadPlan(PathBuf),
    ValidateModelPath(PathBuf),
    Analyze,
    AnalyzeLoadedPlan,
    RunCalibration,
    CreatePlan,
    ValidatePlan,
    SavePlan(PathBuf),
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
            Self::Persistence(_) | Self::DestinationExists(_) => "Plan persistence failed",
        }
    }
}

impl fmt::Display for TuiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { field, message } => write!(formatter, "{field}: {message}"),
            Self::Storage(error) => write!(formatter, "{error}"),
            Self::Analysis(error) | Self::Planning(error) | Self::Compilation(error) => {
                write!(formatter, "{error}")
            }
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
            ram_input: String::new(),
            mode: OperatingMode::BalancedNormal,
            advanced_thread_input: String::new(),
            advanced_cache: OverrideChoice::PlannerDefault,
            advanced_cache_capacity_input: String::new(),
            advanced_read_coalescing: OverrideChoice::PlannerDefault,
            advanced_grouped_buffer: OverrideChoice::PlannerDefault,
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
            Screen::ModelInfo => self.handle_model_info(command),
            Screen::CalibrationSelect => self.handle_calibration_select(command),
            Screen::Calibrating => self.handle_calibrating(command),
            Screen::PlanReview => self.handle_plan_review(command),
            Screen::PlanValid => self.handle_plan_valid(command),
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
            Screen::Analyzing | Screen::PlanValidation => None,
        }
    }

    pub fn accepts_text(&self) -> bool {
        match self.screen {
            Screen::LoadPlan | Screen::ModelInput | Screen::SavePlan => true,
            Screen::Preferences => matches!(
                self.preference_field(),
                PreferenceField::Ram
                    | PreferenceField::AdvancedThreads
                    | PreferenceField::AdvancedCacheCapacity
            ),
            _ => false,
        }
    }

    pub fn load_finished(
        &mut self,
        result: Result<PersistedExecutionPlan, TuiError>,
    ) {
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
        result: Result<
            (PlanningSession, ExecutionPlan, PlanCompatibilityReport),
            TuiError,
        >,
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

    pub fn analysis_finished(
        &mut self,
        result: Result<PlanningSession, TuiError>,
    ) {
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
        self.calibration_tasks = tasks
            .iter()
            .map(CalibrationTaskView::pending)
            .collect();
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
                self.status_message = Some("Calibration complete. Press Enter to plan.".to_string());
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
            8
        } else {
            3
        }
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
        match command {
            UiCommand::Up => self.move_menu_up(self.preference_count()),
            UiCommand::Down => self.move_menu_down(self.preference_count()),
            UiCommand::Back => {
                self.screen = Screen::ModelInput;
                self.menu_index = 0;
            }
            UiCommand::Backspace => self.edit_preference_backspace(),
            UiCommand::Character(character) if !character.is_control() => {
                self.edit_preference_character(character)
            }
            UiCommand::Enter => match self.preference_field() {
                PreferenceField::Mode => {
                    self.mode = next_mode(self.mode);
                    if self.menu_index >= self.preference_count() {
                        self.menu_index = self.preference_count() - 1;
                    }
                }
                PreferenceField::AdvancedCache => {
                    self.advanced_cache = self.advanced_cache.next()
                }
                PreferenceField::AdvancedReadCoalescing => {
                    self.advanced_read_coalescing = self.advanced_read_coalescing.next()
                }
                PreferenceField::AdvancedGroupedBuffer => {
                    self.advanced_grouped_buffer = self.advanced_grouped_buffer.next()
                }
                PreferenceField::Continue => match self.build_user_profile() {
                    Ok(profile) => {
                        self.user_profile = Some(profile);
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
                PreferenceField::Ram
                | PreferenceField::AdvancedThreads
                | PreferenceField::AdvancedCacheCapacity => {}
            },
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
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::CalibrationSelect;
                self.menu_index = self.calibration_index;
            }
            UiCommand::Enter => {
                self.screen = Screen::Preferences;
                self.menu_index = 0;
            }
            _ => {}
        }
        None
    }

    fn handle_calibration_select(&mut self, command: UiCommand) -> Option<AppAction> {
        match command {
            UiCommand::Up => self.move_menu_up(4),
            UiCommand::Down => self.move_menu_down(4),
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
        if self.plan_origin == PlanOrigin::LoadedPlan {
            if command == UiCommand::Back || command == UiCommand::Enter {
                self.screen = Screen::PlanCompatibility;
                self.menu_index = 0;
            }
            return None;
        }
        match command {
            UiCommand::Up | UiCommand::Down => self.menu_index = 1 - self.menu_index.min(1),
            UiCommand::Back => {
                self.screen = Screen::PlanReview;
                self.menu_index = 0;
            }
            UiCommand::Enter if self.menu_index == 0 => {
                self.screen = Screen::SavePlan;
                self.save_path_input.clear();
                self.error = None;
            }
            UiCommand::Enter => {
                self.screen = Screen::PlanReview;
                self.menu_index = 0;
            }
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

    fn start_new_model_flow(&mut self) {
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
    }

    fn show_error(&mut self, error: TuiError, return_to: Screen) {
        self.screen = Screen::Error;
        self.error = Some(error);
        self.error_return = return_to;
        self.menu_index = 0;
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
        let ram_budget_bytes = parse_memory_size(&self.ram_input).map_err(|error| {
            TuiError::Input {
                field: "RAM budget",
                message: error.to_string(),
            }
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
            return match self.menu_index.min(2) {
                0 => PreferenceField::Ram,
                1 => PreferenceField::Mode,
                _ => PreferenceField::Continue,
            };
        }
        match self.menu_index.min(7) {
            0 => PreferenceField::Ram,
            1 => PreferenceField::Mode,
            2 => PreferenceField::AdvancedThreads,
            3 => PreferenceField::AdvancedCache,
            4 => PreferenceField::AdvancedCacheCapacity,
            5 => PreferenceField::AdvancedReadCoalescing,
            6 => PreferenceField::AdvancedGroupedBuffer,
            _ => PreferenceField::Continue,
        }
    }

    fn edit_preference_character(&mut self, character: char) {
        match self.preference_field() {
            PreferenceField::Ram => self.ram_input.push(character),
            PreferenceField::AdvancedThreads => self.advanced_thread_input.push(character),
            PreferenceField::AdvancedCacheCapacity => {
                self.advanced_cache_capacity_input.push(character)
            }
            _ => {}
        }
    }

    fn edit_preference_backspace(&mut self) {
        match self.preference_field() {
            PreferenceField::Ram => {
                self.ram_input.pop();
            }
            PreferenceField::AdvancedThreads => {
                self.advanced_thread_input.pop();
            }
            PreferenceField::AdvancedCacheCapacity => {
                self.advanced_cache_capacity_input.pop();
            }
            _ => {}
        }
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
        assert_eq!(app.error.as_ref().unwrap().title(), "Plan persistence failed");
        app.handle(UiCommand::Back);
        assert_eq!(app.screen, Screen::LoadPlan);
    }

    #[test]
    fn model_input_accepts_exact_path_and_rejects_empty_input() {
        let mut app = TuiApp::default();
        app.handle(UiCommand::Enter);
        assert_eq!(app.screen, Screen::ModelInput);
        assert!(app.handle(UiCommand::Enter).is_none());
        assert!(matches!(
            app.error.as_ref(),
            Some(TuiError::Input { .. })
        ));

        app.model_input = "relative/model.gguf".to_string();
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::ValidateModelPath(PathBuf::from(
                "relative/model.gguf"
            )))
        );
        assert!(app.model_path_validation_finished(Ok(())).is_none());
        assert_eq!(app.screen, Screen::Preferences);

        app.screen = Screen::Analyzing;
        assert!(app
            .model_path_validation_finished(Err(TuiError::Storage(
                StorageDiscoveryError::InvalidPath,
            )))
            .is_none());
        assert_eq!(app.screen, Screen::Error);
        assert_eq!(app.error.as_ref().unwrap().title(), "Storage discovery failed");
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
    fn preference_changes_use_existing_profile_contracts_without_clamping() {
        let mut app = TuiApp::default();
        app.screen = Screen::Preferences;
        app.ram_input = "2GiB".to_string();
        app.menu_index = 1;
        app.handle(UiCommand::Enter);
        assert_eq!(app.mode, OperatingMode::MaximumPerformance);
        app.menu_index = 2;
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
        app.menu_index = 7;
        assert!(app.handle(UiCommand::Enter).is_none());
        assert!(matches!(
            app.error.as_ref(),
            Some(TuiError::Input { .. })
        ));
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
        assert_eq!(app.handle(UiCommand::Enter), Some(AppAction::RunCalibration));
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
        app.planning_finished(Err(TuiError::Planning(
            OrchestrationError::Planning(
                ramforge_runtime::planner::PlannerError::Infeasible(
                    ramforge_runtime::planner::FeasibilityRejection::ModelNotExecutable,
                ),
            ),
        )));
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
    fn successful_validation_reaches_plan_valid_and_save_request() {
        let mut app = TuiApp::default();
        app.screen = Screen::PlanValidation;
        app.plan_validation_finished(Ok(()));
        assert_eq!(app.screen, Screen::PlanValid);
        app.handle(UiCommand::Enter);
        assert_eq!(app.screen, Screen::SavePlan);
        app.save_path_input = "/tmp/plan.rfp".to_string();
        assert_eq!(
            app.handle(UiCommand::Enter),
            Some(AppAction::SavePlan(PathBuf::from("/tmp/plan.rfp")))
        );
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
