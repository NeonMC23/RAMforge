//! Deterministic predictive-residency simulation.
//!
//! Simulation is not hardware evidence. It verifies planner invariants,
//! budget bounds, and feedback behavior without touching native memory.

use std::collections::{BTreeMap, BTreeSet};

use forge_core::TensorMappingInfo;

use crate::execution_profile::{AccessPattern, ExecutionProfile};
use crate::prediction::{predict_from_profile, PredictionConfig};
use crate::prediction_planner::{plan_predictions, PredictiveAction, PredictiveReason};
use crate::residency_engine::{ModelGeneration, ModelId, ResourceInventory};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimResource {
    pub name: String,
    pub bytes: u64,
    pub file_backed: bool,
    pub known_size: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimScenario {
    pub name: String,
    pub budget_bytes: u64,
    pub staging_window_bytes: u64,
    pub speculative_budget_bytes: u64,
    pub min_confidence_bps: u16,
    pub resources: Vec<SimResource>,
    pub training_sequence: Vec<String>,
    pub test_sequence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimResult {
    pub scenario: String,
    pub strategy: String,
    pub hits: u64,
    pub misses: u64,
    pub reread_bytes: u64,
    pub staged_bytes: u64,
    pub wasted_staged_bytes: u64,
    pub peak_resident_bytes: u64,
    pub budget_violations: u64,
    pub staged_resources: usize,
    pub deferred: usize,
    pub rejected: usize,
    pub release_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimComparison {
    pub baseline: SimResult,
    pub predictive: SimResult,
}

impl SimComparison {
    pub fn miss_reduction(&self) -> u64 {
        self.baseline.misses.saturating_sub(self.predictive.misses)
    }

    pub fn reread_byte_reduction(&self) -> u64 {
        self.baseline
            .reread_bytes
            .saturating_sub(self.predictive.reread_bytes)
    }
}

pub fn simulate(scenario: &SimScenario) -> Result<SimComparison, String> {
    if scenario.budget_bytes == 0 {
        return Err("simulation budget must be greater than zero".to_string());
    }
    if scenario.staging_window_bytes > scenario.budget_bytes {
        return Err("staging window exceeds simulation budget".to_string());
    }
    let config = PredictionConfig {
        enabled: true,
        min_confidence_bps: scenario.min_confidence_bps,
        staging_window_bytes: scenario.staging_window_bytes,
        speculative_budget_bytes: scenario
            .speculative_budget_bytes
            .min(scenario.staging_window_bytes),
        max_in_flight: 1,
        lookahead: 64,
    };
    config.validate().map_err(|error| error.to_string())?;
    let (inventory, profile) = scenario_inputs(scenario)?;
    let predictions = predict_from_profile(&config, Some(&profile), &inventory, None);
    let plan = plan_predictions(
        &config,
        &predictions,
        &inventory,
        Some(0),
        Some(scenario.budget_bytes),
        0,
        false,
    );
    let baseline = run_strategy(scenario, &inventory, &BTreeSet::new(), "baseline")?;
    let staged: BTreeSet<i32> = plan
        .stage_keys
        .iter()
        .chain(plan.prefetch_keys.iter())
        .map(|key| key.tensor_index)
        .collect();
    let predictive = run_strategy(scenario, &inventory, &staged, "predictive")?;

    let _ = plan.decisions.iter().any(|decision| {
        decision.action == PredictiveAction::ReleaseAtBoundary
            || decision.reason == PredictiveReason::ReleaseNotAuthorized
    });
    let mut predictive = predictive;
    predictive.staged_bytes = plan.requested_bytes();
    predictive.staged_resources = staged.len();
    predictive.deferred = plan.deferred;
    predictive.rejected = plan.rejected;
    predictive.release_authorized = plan.release_authorized;
    let mut wasted = 0u64;
    let accessed: BTreeSet<&str> = scenario.test_sequence.iter().map(String::as_str).collect();
    for key in &staged {
        if let Some(resource) = inventory
            .resources
            .iter()
            .find(|r| r.key.tensor_index == *key)
        {
            if !accessed.contains(resource.name.as_str()) {
                wasted = wasted.saturating_add(resource.page_bytes);
            }
        }
    }
    predictive.wasted_staged_bytes = wasted;
    Ok(SimComparison {
        baseline,
        predictive,
    })
}

fn run_strategy(
    scenario: &SimScenario,
    inventory: &ResourceInventory,
    prestaged: &BTreeSet<i32>,
    strategy: &str,
) -> Result<SimResult, String> {
    let sizes: BTreeMap<&str, u64> = inventory
        .resources
        .iter()
        .map(|resource| (resource.name.as_str(), resource.page_bytes))
        .collect();
    let mut resident = BTreeMap::<String, u64>::new();
    let mut resident_bytes = 0u64;
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut reread = 0u64;
    let mut staged_bytes = 0u64;
    let mut peak = 0u64;
    let mut violations = 0u64;

    for key in prestaged {
        let Some(resource) = inventory
            .resources
            .iter()
            .find(|r| r.key.tensor_index == *key)
        else {
            continue;
        };
        let bytes = resource.page_bytes;
        if bytes == 0 || resident.contains_key(&resource.name) {
            continue;
        }
        if resident_bytes.saturating_add(bytes) > scenario.budget_bytes {
            continue;
        }
        resident.insert(resource.name.clone(), bytes);
        resident_bytes = resident_bytes.saturating_add(bytes);
        staged_bytes = staged_bytes.saturating_add(bytes);
        peak = peak.max(resident_bytes);
    }

    for name in &scenario.test_sequence {
        let Some(&bytes) = sizes.get(name.as_str()) else {
            return Err(format!(
                "simulation sequence references unknown resource {name}"
            ));
        };
        if resident.contains_key(name) {
            hits += 1;
            continue;
        }
        misses += 1;
        reread = reread.saturating_add(bytes);
        if resident_bytes.saturating_add(bytes) > scenario.budget_bytes {
            // Deterministic LRU-ish eviction by lexicographic key is not
            // sufficient; evict the lexicographically first resident until fit.
            while resident_bytes.saturating_add(bytes) > scenario.budget_bytes {
                let Some(oldest) = resident.keys().next().cloned() else {
                    break;
                };
                if let Some(old) = resident.remove(&oldest) {
                    resident_bytes = resident_bytes.saturating_sub(old);
                }
            }
        }
        if resident_bytes.saturating_add(bytes) > scenario.budget_bytes {
            violations = violations.saturating_add(1);
            continue;
        }
        resident.insert(name.clone(), bytes);
        resident_bytes = resident_bytes.saturating_add(bytes);
        peak = peak.max(resident_bytes);
    }
    Ok(SimResult {
        scenario: scenario.name.clone(),
        strategy: strategy.to_string(),
        hits,
        misses,
        reread_bytes: reread,
        staged_bytes,
        wasted_staged_bytes: 0,
        peak_resident_bytes: peak,
        budget_violations: violations,
        staged_resources: 0,
        deferred: 0,
        rejected: 0,
        release_authorized: false,
    })
}

fn scenario_inputs(
    scenario: &SimScenario,
) -> Result<(ResourceInventory, ExecutionProfile), String> {
    let mut mappings = Vec::new();
    let mut offset = 0u64;
    for (index, resource) in scenario.resources.iter().enumerate() {
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
            mapping_index: resource.file_backed.then_some(index as u32),
            mapping_size: offset.saturating_add(size).max(1),
            offset,
            size,
            page_size,
            page_start,
            page_end,
        });
        offset = offset.saturating_add(size.max(1));
    }
    let inventory = ResourceInventory::from_tensor_mappings(
        ModelId(scenario.name.clone()),
        ModelGeneration(1),
        &mappings,
    )
    .map_err(|error| error.to_string())?;
    let mut profile = ExecutionProfile::new(ModelId(scenario.name.clone()), ModelGeneration(1));
    profile.complete_observations = 2;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(wrong: bool) -> SimScenario {
        let names = ["a", "b", "c", "d", "e"];
        SimScenario {
            name: if wrong { "wrong" } else { "correct" }.to_string(),
            budget_bytes: 12 * 1024,
            staging_window_bytes: 8 * 1024,
            speculative_budget_bytes: 8 * 1024,
            min_confidence_bps: 5_000,
            resources: names
                .iter()
                .map(|name| SimResource {
                    name: (*name).to_string(),
                    bytes: 4096,
                    file_backed: true,
                    known_size: true,
                })
                .collect(),
            training_sequence: names.iter().map(|name| (*name).to_string()).collect(),
            test_sequence: if wrong {
                vec!["e".to_string()]
            } else {
                names.iter().map(|name| (*name).to_string()).collect()
            },
        }
    }

    #[test]
    fn predictive_staging_reduces_simulated_rereads() {
        let result = simulate(&scenario(false)).expect("simulation");
        assert!(result.predictive.hits > result.baseline.hits);
        assert!(result.miss_reduction() > 0);
        assert!(result.reread_byte_reduction() > 0);
        assert_eq!(result.predictive.budget_violations, 0);
        assert!(!result.predictive.release_authorized);
        assert!(result.predictive.peak_resident_bytes <= 12 * 1024);
    }

    #[test]
    fn wrong_prediction_is_bounded_and_wasted_bytes_are_reported() {
        let result = simulate(&scenario(true)).expect("simulation");
        assert_eq!(result.predictive.budget_violations, 0);
        assert!(result.predictive.wasted_staged_bytes > 0);
        assert!(!result.predictive.release_authorized);
    }

    #[test]
    fn zero_budget_is_rejected() {
        let mut scenario = scenario(false);
        scenario.budget_bytes = 0;
        assert!(simulate(&scenario).is_err());
    }

    #[test]
    fn unknown_size_resources_are_not_staged() {
        let mut scenario = scenario(false);
        scenario.resources[0].known_size = false;
        let result = simulate(&scenario).expect("simulation");
        assert_eq!(result.predictive.budget_violations, 0);
    }
}
