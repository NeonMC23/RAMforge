//! Planning logic for `ramforge plan` command
//!
//! Milestone 6 reporting: the plan distinguishes *file size vs budget*
//! (out-of-core viability) from what the runtime actually charges. The
//! runtime no longer pre-reserves a fake "tensor_cache capacity +
//! runtime overhead" inside the budget; it charges real allocations
//! (resident weights, one layer at a time, KV cache, scoped temps).
//! The cache capacity reported here is an informational bound only.

use ramforge_core::{memory::MemoryBudget, GgufModel};

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub file_size: u64,
    pub architecture: Option<String>,
    pub tensor_count: usize,
    pub ram_requested: u64,
    /// Informational cache bound (charged per entry at runtime, not pre-reserved)
    pub cache_capacity: u64,
    /// Kept for CLI compatibility; the runtime reserves scoped temps on
    /// demand instead of a static overhead allocation, so this is 0.
    pub overhead_reserved: u64,
    pub available: u64,
    pub fits_in_ram: bool,
    pub file_backed_needed: u64,
    pub total_tensor_bytes: Option<u64>,
    pub budget: MemoryBudget,
}

pub fn plan_model(model: &GgufModel, ram_budget_bytes: u64) -> Result<PlanResult, String> {
    let budget = MemoryBudget::new(ram_budget_bytes).map_err(|e| e.to_string())?;

    // Informational cache bound: 80% of budget (50% for tiny budgets).
    // Contents are charged per entry at runtime via insert_budgeted.
    let cache_capacity = (ram_budget_bytes as f64 * 0.8) as u64;
    let cache_capacity = if ram_budget_bytes < 2 * 1024 * 1024 {
        ram_budget_bytes / 2
    } else {
        cache_capacity.max(1024 * 1024).min(ram_budget_bytes.saturating_sub(1024 * 1024))
    };

    // No static overhead pre-reservation: scoped `tmp:*` guards charge
    // exact temps at runtime for exactly their lifetimes.
    let overhead_reserved = 0;

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
