//! Deterministic predictive-prefetch simulation.
//!
//! Simulation is not hardware evidence. The goal is to make lifetime, staging,
//! residency, eviction, and end-to-end cost accounting explicit and testable.

use std::collections::{BTreeMap, BTreeSet};

use forge_core::TensorMappingInfo;

use crate::execution_profile::{AccessPattern, ExecutionProfile};
use crate::prediction::{predict_from_profile, PredictionConfig};
use crate::prediction_planner::{plan_predictions, PredictiveAction};
use crate::residency_engine::{ModelGeneration, ModelId, ResourceInventory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimInitialResidency {
    Resident,
    NonResident,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimPrefetchEffect {
    MakesResident,
    NoEffect,
    Fail,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimPrefetchTiming {
    OnTime,
    Late,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimResource {
    pub name: String,
    pub bytes: u64,
    pub file_backed: bool,
    pub known_size: bool,
    pub initial: SimInitialResidency,
    /// Resources with the same group share mapping pages in the synthetic
    /// inventory, allowing overlap/deduplication scenarios.
    pub overlap_group: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimCosts {
    pub setup_ns: u64,
    pub prediction_ns: u64,
    pub planning_ns: u64,
    pub observation_ns_per_page: u64,
    pub staging_ns: u64,
    pub reread_ns_per_byte: u64,
    pub execution_ns_per_access: u64,
    pub synchronization_ns: u64,
}

impl Default for SimCosts {
    fn default() -> Self {
        Self {
            setup_ns: 1_000,
            prediction_ns: 100,
            planning_ns: 50,
            observation_ns_per_page: 10,
            staging_ns: 500,
            reread_ns_per_byte: 2,
            execution_ns_per_access: 1_000,
            synchronization_ns: 25,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimScenario {
    pub name: String,
    pub budget_bytes: u64,
    pub staging_window_bytes: u64,
    pub speculative_budget_bytes: u64,
    pub min_confidence_bps: u16,
    pub prediction_horizon: usize,
    pub resources: Vec<SimResource>,
    pub training_sequence: Vec<String>,
    pub test_sequence: Vec<String>,
    pub prefetch_effect: SimPrefetchEffect,
    pub prefetch_timing: SimPrefetchTiming,
    pub observe: bool,
    pub costs: SimCosts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimStrategy {
    Baseline,
    Reactive,
    Predictive,
    Conservative,
    Aggressive,
}

impl SimStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Reactive => "reactive",
            Self::Predictive => "predictive",
            Self::Conservative => "conservative",
            Self::Aggressive => "aggressive",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimResult {
    pub scenario: String,
    pub strategy: &'static str,
    pub initial_resident_bytes: u64,
    pub initial_nonresident_bytes: u64,
    pub initial_unknown_bytes: u64,
    pub predicted_resources: usize,
    pub correct_predictions: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision_bps: Option<u16>,
    pub recall_bps: Option<u16>,
    pub staged_bytes: u64,
    pub newly_resident_bytes: u64,
    pub redundant_staged_bytes: u64,
    pub ineffective_staged_bytes: u64,
    pub wasted_staged_bytes: u64,
    pub deferred_bytes: u64,
    pub rejected_bytes: u64,
    pub evicted_before_access_bytes: u64,
    pub reread_bytes: u64,
    pub peak_resident_bytes: u64,
    pub budget_violations: u64,
    pub prediction_cost_ns: u64,
    pub planning_cost_ns: u64,
    pub observation_cost_ns: u64,
    pub staging_cost_ns: u64,
    pub reread_cost_ns: u64,
    pub execution_cost_ns: u64,
    pub synchronization_cost_ns: u64,
    pub total_end_to_end_ns: u64,
    pub net_benefit_ns: i128,
    pub improved: bool,
    pub benefit_known: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimComparison {
    pub scenario: String,
    pub baseline: SimResult,
    pub reactive: SimResult,
    pub predictive: SimResult,
    pub conservative: SimResult,
    pub aggressive: SimResult,
}

impl SimComparison {
    pub fn result(&self, strategy: SimStrategy) -> &SimResult {
        match strategy {
            SimStrategy::Baseline => &self.baseline,
            SimStrategy::Reactive => &self.reactive,
            SimStrategy::Predictive => &self.predictive,
            SimStrategy::Conservative => &self.conservative,
            SimStrategy::Aggressive => &self.aggressive,
        }
    }

    pub fn net_benefit(&self, strategy: SimStrategy) -> i128 {
        self.result(strategy).net_benefit_ns
    }
}

pub fn simulate(scenario: &SimScenario) -> Result<SimComparison, String> {
    if scenario.budget_bytes == 0 {
        return Err("simulation budget must be greater than zero".to_string());
    }
    if scenario.staging_window_bytes > scenario.budget_bytes {
        return Err("staging window exceeds simulation budget".to_string());
    }
    let (inventory, profile) = scenario_inputs(scenario)?;
    let baseline = run_strategy(scenario, &inventory, &profile, SimStrategy::Baseline, None)?;
    let reactive = run_strategy(scenario, &inventory, &profile, SimStrategy::Reactive, None)?;
    let predictive = run_strategy(
        scenario,
        &inventory,
        &profile,
        SimStrategy::Predictive,
        None,
    )?;
    let conservative = run_strategy(
        scenario,
        &inventory,
        &profile,
        SimStrategy::Conservative,
        None,
    )?;
    let aggressive = run_strategy(
        scenario,
        &inventory,
        &profile,
        SimStrategy::Aggressive,
        None,
    )?;
    let baseline_total = baseline.total_end_to_end_ns;
    let mut comparison = SimComparison {
        scenario: scenario.name.clone(),
        baseline,
        reactive,
        predictive,
        conservative,
        aggressive,
    };
    for strategy in [
        SimStrategy::Reactive,
        SimStrategy::Predictive,
        SimStrategy::Conservative,
        SimStrategy::Aggressive,
    ] {
        let result = match strategy {
            SimStrategy::Baseline => &mut comparison.baseline,
            SimStrategy::Reactive => &mut comparison.reactive,
            SimStrategy::Predictive => &mut comparison.predictive,
            SimStrategy::Conservative => &mut comparison.conservative,
            SimStrategy::Aggressive => &mut comparison.aggressive,
        };
        result.net_benefit_ns = baseline_total as i128 - result.total_end_to_end_ns as i128;
        result.improved = result.benefit_known && result.net_benefit_ns > 0;
    }
    Ok(comparison)
}

fn run_strategy(
    scenario: &SimScenario,
    inventory: &ResourceInventory,
    profile: &ExecutionProfile,
    strategy: SimStrategy,
    _unused: Option<()>,
) -> Result<SimResult, String> {
    let (config, observe) = strategy_config(scenario, strategy);
    let predictions = if config.enabled {
        predict_from_profile(&config, Some(profile), inventory, None)
    } else {
        crate::prediction::PredictionSet::empty(
            inventory.model_id.clone(),
            inventory.generation,
            "prediction disabled",
        )
    };
    let resident_before = scenario
        .resources
        .iter()
        .filter(|resource| resource.initial == SimInitialResidency::Resident)
        .map(|resource| resource.bytes)
        .sum();
    let initial_nonresident = scenario
        .resources
        .iter()
        .filter(|resource| resource.initial == SimInitialResidency::NonResident)
        .map(|resource| resource.bytes)
        .sum();
    let initial_unknown = scenario
        .resources
        .iter()
        .filter(|resource| resource.initial == SimInitialResidency::Unknown)
        .map(|resource| resource.bytes)
        .sum();
    let plan = if config.enabled {
        plan_predictions(
            &config,
            &predictions,
            inventory,
            Some(resident_before),
            Some(scenario.budget_bytes),
            0,
            false,
        )
    } else {
        Default::default()
    };
    let selected_names: BTreeSet<String> = plan
        .decisions
        .iter()
        .filter(|decision| {
            matches!(
                decision.action,
                PredictiveAction::StageNow | PredictiveAction::Prefetch
            )
        })
        .map(|decision| decision.name.clone())
        .collect();
    let predicted_names: BTreeSet<String> = predictions
        .predictions
        .iter()
        .map(|prediction| prediction.name.clone())
        .collect();
    let test_names: BTreeSet<String> = scenario.test_sequence.iter().cloned().collect();
    let correct = predicted_names.intersection(&test_names).count();
    let false_positives = predicted_names.difference(&test_names).count();
    let false_negatives = test_names.difference(&predicted_names).count();
    let precision_bps =
        (!predicted_names.is_empty()).then(|| ((correct * 10_000) / predicted_names.len()) as u16);
    let recall_bps =
        (!test_names.is_empty()).then(|| ((correct * 10_000) / test_names.len()) as u16);

    let mut resident: BTreeMap<String, (u64, usize, bool)> = BTreeMap::new();
    let mut resident_bytes = 0u64;
    let mut peak = 0u64;
    let mut budget_violations = 0u64;
    let mut staged_bytes = 0u64;
    let mut redundant_staged_bytes = 0u64;
    let mut newly_resident_bytes = 0u64;
    let mut ineffective_staged_bytes = 0u64;
    let mut deferred_bytes = 0u64;
    let mut rejected_bytes = 0u64;
    let mut staged_pending: BTreeSet<String> = BTreeSet::new();
    let mut evicted_before_access_bytes = 0u64;
    let mut reread_bytes = 0u64;
    let mut benefit_known =
        initial_unknown == 0 && scenario.prefetch_effect != SimPrefetchEffect::Unknown;
    let mut notes = Vec::new();

    for resource in &scenario.resources {
        if resource.initial == SimInitialResidency::Resident {
            if resident_bytes.saturating_add(resource.bytes) <= scenario.budget_bytes {
                resident.insert(resource.name.clone(), (resource.bytes, 0, false));
                resident_bytes = resident_bytes.saturating_add(resource.bytes);
            } else {
                budget_violations = budget_violations.saturating_add(1);
                notes.push(format!(
                    "initial resident resource {} exceeds budget",
                    resource.name
                ));
            }
        }
    }

    if scenario.prefetch_timing == SimPrefetchTiming::OnTime {
        for name in &selected_names {
            let Some(resource) = scenario
                .resources
                .iter()
                .find(|resource| resource.name == *name)
            else {
                continue;
            };
            if resident.contains_key(name) {
                redundant_staged_bytes = redundant_staged_bytes.saturating_add(resource.bytes);
                staged_bytes = staged_bytes.saturating_add(resource.bytes);
                continue;
            }
            match scenario.prefetch_effect {
                SimPrefetchEffect::MakesResident => {
                    if resident_bytes.saturating_add(resource.bytes) > scenario.budget_bytes {
                        deferred_bytes = deferred_bytes.saturating_add(resource.bytes);
                    } else {
                        resident.insert(name.clone(), (resource.bytes, 0, true));
                        resident_bytes = resident_bytes.saturating_add(resource.bytes);
                        peak = peak.max(resident_bytes);
                        staged_bytes = staged_bytes.saturating_add(resource.bytes);
                        newly_resident_bytes = newly_resident_bytes.saturating_add(resource.bytes);
                        staged_pending.insert(name.clone());
                    }
                }
                SimPrefetchEffect::NoEffect => {
                    staged_bytes = staged_bytes.saturating_add(resource.bytes);
                    ineffective_staged_bytes =
                        ineffective_staged_bytes.saturating_add(resource.bytes);
                }
                SimPrefetchEffect::Fail => {
                    rejected_bytes = rejected_bytes.saturating_add(resource.bytes);
                    benefit_known = false;
                }
                SimPrefetchEffect::Unknown => {
                    staged_bytes = staged_bytes.saturating_add(resource.bytes);
                    ineffective_staged_bytes =
                        ineffective_staged_bytes.saturating_add(resource.bytes);
                    benefit_known = false;
                }
            }
        }
    }

    let mut last_order = 0usize;
    for name in &scenario.test_sequence {
        last_order += 1;
        let Some(resource) = scenario
            .resources
            .iter()
            .find(|resource| resource.name == *name)
        else {
            return Err(format!(
                "simulation sequence references unknown resource {name}"
            ));
        };
        if resident.contains_key(name) {
            staged_pending.remove(name);
            if let Some(entry) = resident.get_mut(name) {
                entry.1 = last_order;
            }
            continue;
        }
        // Reactive load of the currently accessed resource. Evict staged/other
        // pages only when the budget requires it.
        while resident_bytes.saturating_add(resource.bytes) > scenario.budget_bytes {
            let victim = resident
                .iter()
                .min_by_key(|(_, (_, order, _))| *order)
                .map(|(name, _)| name.clone());
            let Some(victim) = victim else {
                break;
            };
            if let Some((bytes, _, _)) = resident.remove(&victim) {
                resident_bytes = resident_bytes.saturating_sub(bytes);
                if staged_pending.remove(&victim) {
                    evicted_before_access_bytes = evicted_before_access_bytes.saturating_add(bytes);
                }
            }
        }
        reread_bytes = reread_bytes.saturating_add(resource.bytes);
        if resident_bytes.saturating_add(resource.bytes) > scenario.budget_bytes {
            budget_violations = budget_violations.saturating_add(1);
            continue;
        }
        resident.insert(name.clone(), (resource.bytes, last_order, false));
        resident_bytes = resident_bytes.saturating_add(resource.bytes);
        peak = peak.max(resident_bytes);
    }
    let wasted_staged_bytes = selected_names
        .iter()
        .filter(|name| !test_names.contains(*name))
        .filter_map(|name| {
            scenario
                .resources
                .iter()
                .find(|resource| resource.name == *name)
                .map(|resource| resource.bytes)
        })
        .sum();
    if scenario.prefetch_timing == SimPrefetchTiming::Late && !selected_names.is_empty() {
        staged_bytes = selected_names
            .iter()
            .filter_map(|name| {
                scenario
                    .resources
                    .iter()
                    .find(|resource| resource.name == *name)
            })
            .map(|resource| resource.bytes)
            .sum();
        notes.push("prefetch arrived after the evaluation accesses".to_string());
    }
    if initial_unknown > 0 {
        notes.push(
            "initial unknown residency treated as nonresident for cost, not as proof".to_string(),
        );
    }
    if scenario.prefetch_effect == SimPrefetchEffect::Unknown {
        notes.push("prefetch residency outcome is unknown".to_string());
    }
    let total_pages = scenario
        .resources
        .iter()
        .map(|resource| resource.bytes.div_ceil(4096))
        .sum::<u64>();
    let prediction_cost_ns = if config.enabled {
        scenario.costs.prediction_ns
    } else {
        0
    };
    let planning_cost_ns = if config.enabled {
        scenario.costs.planning_ns
    } else {
        0
    };
    let observation_cost_ns = if observe {
        scenario
            .costs
            .observation_ns_per_page
            .saturating_mul(total_pages)
    } else {
        0
    };
    let staging_cost_ns = if staged_bytes > 0 || deferred_bytes > 0 || rejected_bytes > 0 {
        scenario.costs.staging_ns
    } else {
        0
    };
    let reread_cost_ns = scenario
        .costs
        .reread_ns_per_byte
        .saturating_mul(reread_bytes);
    let execution_cost_ns = scenario
        .costs
        .execution_ns_per_access
        .saturating_mul(scenario.test_sequence.len() as u64);
    let total_end_to_end_ns = scenario
        .costs
        .setup_ns
        .saturating_add(prediction_cost_ns)
        .saturating_add(planning_cost_ns)
        .saturating_add(observation_cost_ns)
        .saturating_add(staging_cost_ns)
        .saturating_add(reread_cost_ns)
        .saturating_add(execution_cost_ns)
        .saturating_add(scenario.costs.synchronization_ns);
    Ok(SimResult {
        scenario: scenario.name.clone(),
        strategy: strategy.label(),
        initial_resident_bytes: resident_before,
        initial_nonresident_bytes: initial_nonresident,
        initial_unknown_bytes: initial_unknown,
        predicted_resources: predicted_names.len(),
        correct_predictions: correct,
        false_positives,
        false_negatives,
        precision_bps,
        recall_bps,
        staged_bytes,
        newly_resident_bytes,
        redundant_staged_bytes,
        ineffective_staged_bytes,
        wasted_staged_bytes,
        deferred_bytes,
        rejected_bytes,
        evicted_before_access_bytes,
        reread_bytes,
        peak_resident_bytes: peak,
        budget_violations,
        prediction_cost_ns,
        planning_cost_ns,
        observation_cost_ns,
        staging_cost_ns,
        reread_cost_ns,
        execution_cost_ns,
        synchronization_cost_ns: scenario.costs.synchronization_ns,
        total_end_to_end_ns,
        net_benefit_ns: 0,
        improved: false,
        benefit_known,
        notes,
    })
}

fn strategy_config(scenario: &SimScenario, strategy: SimStrategy) -> (PredictionConfig, bool) {
    match strategy {
        SimStrategy::Baseline | SimStrategy::Reactive => (PredictionConfig::default(), false),
        SimStrategy::Predictive => (
            PredictionConfig {
                enabled: true,
                min_confidence_bps: scenario.min_confidence_bps,
                staging_window_bytes: scenario.staging_window_bytes,
                speculative_budget_bytes: scenario.speculative_budget_bytes,
                max_in_flight: 1,
                lookahead: scenario.prediction_horizon,
            },
            scenario.observe,
        ),
        SimStrategy::Conservative => (
            PredictionConfig {
                enabled: true,
                min_confidence_bps: 7_500,
                staging_window_bytes: scenario.staging_window_bytes / 2,
                speculative_budget_bytes: 0,
                max_in_flight: 1,
                lookahead: scenario.prediction_horizon,
            },
            scenario.observe,
        ),
        SimStrategy::Aggressive => (
            PredictionConfig {
                enabled: true,
                min_confidence_bps: 2_500,
                staging_window_bytes: scenario.staging_window_bytes,
                speculative_budget_bytes: scenario.speculative_budget_bytes,
                max_in_flight: 1,
                lookahead: scenario.prediction_horizon,
            },
            scenario.observe,
        ),
    }
}

fn scenario_inputs(
    scenario: &SimScenario,
) -> Result<(ResourceInventory, ExecutionProfile), String> {
    let mut mappings = Vec::new();
    let mut offsets: BTreeMap<u32, u64> = BTreeMap::new();
    for (index, resource) in scenario.resources.iter().enumerate() {
        let group = resource.overlap_group.unwrap_or(index as u32);
        let offset = if resource.overlap_group.is_some() {
            0
        } else {
            *offsets.get(&group).unwrap_or(&0)
        };
        let size = if resource.known_size {
            resource.bytes
        } else {
            0
        };
        let page_size = 4096u64;
        let page_start = offset & !(page_size - 1);
        let page_end = if size == 0 {
            page_start
        } else {
            (offset + size + page_size - 1) & !(page_size - 1)
        };
        mappings.push(TensorMappingInfo {
            name: resource.name.clone(),
            file_backed: resource.file_backed,
            mapping_index: resource.file_backed.then_some(group),
            mapping_size: offset.saturating_add(size).max(1),
            offset,
            size,
            page_size,
            page_start,
            page_end,
        });
        if size > 0 && resource.overlap_group.is_none() {
            offsets.insert(group, offset.saturating_add(size.max(1)));
        }
    }
    let inventory = ResourceInventory::from_tensor_mappings(
        ModelId(scenario.name.clone()),
        ModelGeneration(1),
        &mappings,
    )
    .map_err(|error| error.to_string())?;
    let mut profile = ExecutionProfile::new(ModelId(scenario.name.clone()), ModelGeneration(1));
    profile.complete_observations = 1;
    for (order, name) in scenario.training_sequence.iter().enumerate() {
        let access = profile.access.entry(name.clone()).or_insert(AccessPattern {
            first_order: order,
            last_order: order,
            observed_consumers: 0,
        });
        access.first_order = access.first_order.min(order);
        access.last_order = access.last_order.max(order);
        access.observed_consumers = access.observed_consumers.saturating_add(1);
    }
    Ok((inventory, profile))
}

/// Required deterministic scenario family. `budget` and `window` are explicit
/// so callers can compare the same initial conditions across strategies.
pub fn standard_scenarios(budget_bytes: u64, staging_window_bytes: u64) -> Vec<SimScenario> {
    let base =
        |name: &str, resources: Vec<SimResource>, train: &[&str], test: &[&str]| SimScenario {
            name: name.to_string(),
            budget_bytes,
            staging_window_bytes,
            speculative_budget_bytes: staging_window_bytes,
            min_confidence_bps: 5_000,
            prediction_horizon: 8,
            resources,
            training_sequence: train.iter().map(|name| (*name).to_string()).collect(),
            test_sequence: test.iter().map(|name| (*name).to_string()).collect(),
            prefetch_effect: SimPrefetchEffect::MakesResident,
            prefetch_timing: SimPrefetchTiming::OnTime,
            observe: false,
            costs: SimCosts::default(),
        };
    let r = |name: &str, initial| SimResource {
        name: name.to_string(),
        bytes: 4096,
        file_backed: true,
        known_size: true,
        initial,
        overlap_group: None,
    };
    let mut result = vec![
        base(
            "useful",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
        base(
            "already-resident",
            vec![r("a", SimInitialResidency::Resident)],
            &["a"],
            &["a"],
        ),
        base(
            "no-residency-effect",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
        base(
            "predicted-never-accessed",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &[],
        ),
        base(
            "incorrect-prediction",
            vec![
                r("a", SimInitialResidency::NonResident),
                r("b", SimInitialResidency::NonResident),
            ],
            &["a"],
            &["b"],
        ),
        base(
            "incorrect-already-resident",
            vec![
                r("a", SimInitialResidency::Resident),
                r("b", SimInitialResidency::NonResident),
            ],
            &["a"],
            &["b"],
        ),
        base(
            "late-prefetch",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
        base(
            "staging-rejected",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
        base(
            "deferred-then-accessed",
            vec![
                r("a", SimInitialResidency::NonResident),
                r("b", SimInitialResidency::NonResident),
            ],
            &["a", "b"],
            &["b", "a"],
        ),
        base(
            "evicted-before-access",
            vec![
                r("a", SimInitialResidency::NonResident),
                r("b", SimInitialResidency::NonResident),
            ],
            &["a"],
            &["b", "a"],
        ),
        base(
            "unknown-initial",
            vec![r("a", SimInitialResidency::Unknown)],
            &["a"],
            &["a"],
        ),
        base(
            "unknown-effect",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
        base(
            "overlapping-duplicate",
            vec![
                SimResource {
                    overlap_group: Some(0),
                    ..r("a", SimInitialResidency::NonResident)
                },
                SimResource {
                    overlap_group: Some(0),
                    ..r("b", SimInitialResidency::NonResident)
                },
            ],
            &["a", "b"],
            &["a", "b"],
        ),
        base(
            "constrained-competition",
            vec![
                r("a", SimInitialResidency::NonResident),
                r("b", SimInitialResidency::NonResident),
                r("c", SimInitialResidency::NonResident),
            ],
            &["a", "b", "c"],
            &["a", "b", "c"],
        ),
        base(
            "observation-overhead",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
        base(
            "prediction-cost-worse",
            vec![r("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ),
    ];
    // Scenario-specific conditions.
    result[2].prefetch_effect = SimPrefetchEffect::NoEffect;
    result[6].prefetch_timing = SimPrefetchTiming::Late;
    result[7].staging_window_bytes = 0;
    result[7].speculative_budget_bytes = 0;
    result[8].staging_window_bytes = 4096;
    result[8].speculative_budget_bytes = 0;
    result[9].budget_bytes = 4096;
    result[9].staging_window_bytes = 4096;
    result[9].speculative_budget_bytes = 0;
    result[11].prefetch_effect = SimPrefetchEffect::Unknown;
    result[13].budget_bytes = 4096;
    result[13].staging_window_bytes = 4096;
    result[13].speculative_budget_bytes = 0;
    result[14].observe = true;
    result[14].costs.observation_ns_per_page = 1_000_000;
    result[15].costs.prediction_ns = 1_000_000;
    result[15].costs.staging_ns = 1_000_000;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(name: &str, initial: SimInitialResidency) -> SimResource {
        SimResource {
            name: name.to_string(),
            bytes: 4096,
            file_backed: true,
            known_size: true,
            initial,
            overlap_group: None,
        }
    }

    fn scenario(
        name: &str,
        resources: Vec<SimResource>,
        initial_sequence: &[&str],
        test_sequence: &[&str],
    ) -> SimScenario {
        SimScenario {
            name: name.to_string(),
            budget_bytes: 8192,
            staging_window_bytes: 8192,
            speculative_budget_bytes: 8192,
            min_confidence_bps: 5_000,
            prediction_horizon: 4,
            resources,
            training_sequence: initial_sequence
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            test_sequence: test_sequence
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            prefetch_effect: SimPrefetchEffect::MakesResident,
            prefetch_timing: SimPrefetchTiming::OnTime,
            observe: false,
            costs: SimCosts::default(),
        }
    }

    #[test]
    fn useful_prefetch_beats_reactive_but_prediction_precision_is_not_benefit() {
        let result = simulate(&scenario(
            "useful",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        ))
        .expect("simulation");
        assert!(result.predictive.newly_resident_bytes > 0);
        assert!(result.predictive.redundant_staged_bytes == 0);
        assert_eq!(result.predictive.correct_predictions, 1);
        assert!(result.predictive.net_benefit_ns > 0);
        assert!(result.predictive.improved);
        assert_eq!(result.baseline.budget_violations, 0);
    }

    #[test]
    fn already_resident_prefetch_is_redundant() {
        let result = simulate(&scenario(
            "redundant",
            vec![resource("a", SimInitialResidency::Resident)],
            &["a"],
            &["a"],
        ))
        .expect("simulation");
        assert_eq!(result.predictive.redundant_staged_bytes, 4096);
        assert_eq!(result.predictive.newly_resident_bytes, 0);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn no_effect_staging_is_ineffective() {
        let mut scenario = scenario(
            "no-effect",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        );
        scenario.prefetch_effect = SimPrefetchEffect::NoEffect;
        let result = simulate(&scenario).expect("simulation");
        assert_eq!(result.predictive.ineffective_staged_bytes, 4096);
        assert_eq!(result.predictive.newly_resident_bytes, 0);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn unknown_residency_is_not_counted_as_a_benefit() {
        let mut scenario = scenario(
            "unknown",
            vec![resource("a", SimInitialResidency::Unknown)],
            &["a"],
            &["a"],
        );
        scenario.prefetch_effect = SimPrefetchEffect::Unknown;
        let result = simulate(&scenario).expect("simulation");
        assert_eq!(result.predictive.initial_unknown_bytes, 4096);
        assert!(!result.predictive.benefit_known);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn wrong_prediction_wastes_staging() {
        let result = simulate(&scenario(
            "wrong",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &[],
        ))
        .expect("simulation");
        assert_eq!(result.predictive.false_positives, 1);
        assert!(result.predictive.wasted_staged_bytes > 0);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn late_prefetch_does_not_help() {
        let mut scenario = scenario(
            "late",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        );
        scenario.prefetch_timing = SimPrefetchTiming::Late;
        let result = simulate(&scenario).expect("simulation");
        assert_eq!(result.predictive.newly_resident_bytes, 0);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn staging_can_be_rejected_by_budget() {
        let mut scenario = scenario(
            "rejected",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        );
        scenario.staging_window_bytes = 0;
        scenario.speculative_budget_bytes = 0;
        let result = simulate(&scenario).expect("simulation");
        assert_eq!(result.predictive.newly_resident_bytes, 0);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn observation_overhead_can_exceed_benefit() {
        let mut scenario = scenario(
            "observation",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        );
        scenario.observe = true;
        scenario.costs.observation_ns_per_page = 1_000_000;
        let result = simulate(&scenario).expect("simulation");
        assert!(result.predictive.observation_cost_ns > 0);
        assert!(!result.predictive.improved);
    }

    #[test]
    fn standard_scenarios_are_deterministic_and_budget_safe() {
        let scenarios = standard_scenarios(12 * 1024, 8 * 1024);
        assert_eq!(scenarios.len(), 16);
        for scenario in scenarios {
            let first = simulate(&scenario).expect("simulation");
            let second = simulate(&scenario).expect("repeat");
            assert_eq!(first, second, "nondeterministic {}", scenario.name);
            for strategy in [
                SimStrategy::Baseline,
                SimStrategy::Reactive,
                SimStrategy::Predictive,
                SimStrategy::Conservative,
                SimStrategy::Aggressive,
            ] {
                let result = first.result(strategy);
                assert_eq!(
                    result.budget_violations, 0,
                    "{} {}",
                    scenario.name, result.strategy
                );
                assert!(result.peak_resident_bytes <= scenario.budget_bytes);
                assert!(!result.improved || result.benefit_known);
            }
        }
    }

    #[test]
    fn overlapping_duplicate_is_staged_once_and_budget_is_respected() {
        let scenarios = standard_scenarios(12 * 1024, 8 * 1024);
        let overlap = scenarios
            .iter()
            .find(|scenario| scenario.name == "overlapping-duplicate")
            .expect("overlap scenario");
        let result = simulate(overlap).expect("simulation");
        assert_eq!(result.predictive.predicted_resources, 2);
        assert!(result.predictive.staged_bytes <= 4096);
        assert_eq!(result.predictive.budget_violations, 0);
    }

    #[test]
    fn all_strategies_are_deterministic() {
        let scenario = scenario(
            "determinism",
            vec![resource("a", SimInitialResidency::NonResident)],
            &["a"],
            &["a"],
        );
        let first = simulate(&scenario).expect("first");
        let second = simulate(&scenario).expect("second");
        assert_eq!(first, second);
    }
}
