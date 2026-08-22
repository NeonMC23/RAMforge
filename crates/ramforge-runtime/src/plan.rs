//! Planning logic for `ramforge plan` command
//!
//! Simple deterministic allocation strategy for Milestone 2.
//! Future milestones will replace this with an advanced execution planner.

use ramforge_core::{memory::MemoryBudget, GgufModel};

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub file_size: u64,
    pub architecture: Option<String>,
    pub tensor_count: usize,
    pub ram_requested: u64,
    pub cache_capacity: u64,
    pub overhead_reserved: u64,
    pub available: u64,
    pub fits_in_ram: bool,
    pub file_backed_needed: u64,
    pub total_tensor_bytes: Option<u64>,
    pub budget: MemoryBudget,
}

pub fn plan_model(model: &GgufModel, ram_budget_bytes: u64) -> Result<PlanResult, String> {
    let mut budget = MemoryBudget::new(ram_budget_bytes).map_err(|e| e.to_string())?;

    // Simple deterministic strategy: 80% cache, 10% overhead
    let cache_capacity = (ram_budget_bytes as f64 * 0.8) as u64;
    let cache_capacity = if ram_budget_bytes < 2 * 1024 * 1024 {
        // For very small budgets, use 50% to leave room
        ram_budget_bytes / 2
    } else {
        cache_capacity.max(1024 * 1024).min(ram_budget_bytes.saturating_sub(1024 * 1024))
    };

    let overhead = (ram_budget_bytes as f64 * 0.1) as u64;
    let overhead = overhead.min(budget.available_bytes().saturating_sub(cache_capacity));

    budget
        .allocate("tensor_cache", cache_capacity)
        .map_err(|e| e.to_string())?;
    let mut overhead_reserved = 0;
    if overhead > 0 && budget.can_allocate(overhead) {
        budget.allocate("runtime_overhead", overhead).map_err(|e| e.to_string())?;
        overhead_reserved = overhead;
    }

    let file_size = model.file_size;
    let fits_in_ram = file_size <= ram_budget_bytes;
    let file_backed_needed = if fits_in_ram {
        0
    } else {
        file_size - ram_budget_bytes
    };

    let architecture = model.info().architecture.clone();
    let tensor_count = model.tensors.len();
    let total_tensor_bytes = model.total_tensor_bytes();
    let available = budget.available_bytes();

    Ok(PlanResult {
        file_size,
        architecture,
        tensor_count,
        ram_requested: ram_budget_bytes,
        cache_capacity,
        overhead_reserved,
        available,
        fits_in_ram,
        file_backed_needed,
        total_tensor_bytes,
        budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::GgufModel;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn dummy_model(file_size: u64) -> GgufModel {
        GgufModel {
            path: PathBuf::from("/tmp/dummy.gguf"),
            file_size,
            version: 3,
            metadata: BTreeMap::new(),
            tensors: vec![],
            alignment: 32,
            data_start_offset: 0,
        }
    }

    #[test]
    fn test_plan_fits() {
        let model = dummy_model(1_000_000);
        let plan = plan_model(&model, 8 * 1024 * 1024 * 1024).unwrap();
        assert!(plan.fits_in_ram);
        assert_eq!(plan.file_backed_needed, 0);
        assert!(plan.cache_capacity > 0);
        assert!(plan.cache_capacity <= plan.ram_requested);
    }

    #[test]
    fn test_plan_exceeds() {
        let model = dummy_model(10 * 1024 * 1024 * 1024);
        let plan = plan_model(&model, 8 * 1024 * 1024 * 1024).unwrap();
        assert!(!plan.fits_in_ram);
        assert_eq!(plan.file_backed_needed, 2 * 1024 * 1024 * 1024);
    }
}
