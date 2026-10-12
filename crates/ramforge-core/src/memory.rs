//! Memory budget abstraction for RAMforge
//!
//! `MemoryBudget` is RAMforge-managed logical allocation accounting. It is not
//! a hard process-RSS cap, an operating-system memory limit, or a page-cache
//! limit. It only covers allocations that RAMforge explicitly reserves and
//! keeps charged for their owned lifetime, such as tensor caches, KV caches,
//! and bounded read or decode buffers.
//!
//! The managed-allocation invariant is:
//! - calculate sizes with checked arithmetic;
//! - reserve before allocating;
//! - keep the charge while the owned allocation is live;
//! - release only after the allocation has been dropped; and
//! - roll back a reservation when allocation or a later construction step
//!   fails.
//!
//! Allocator metadata, fragmentation, thread stacks, dependency/native
//! allocations, process RSS, and filesystem page cache remain outside this
//! logical ledger unless a caller explicitly measures them separately.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::error::{MemoryError, ParseSizeError};

/// A reusable memory budget abstraction
///
/// Internally all values are exact byte counts (u64).
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    total: u64,
    allocations: BTreeMap<String, u64>,
    used: u64,
    peak_used: u64,
}

impl MemoryBudget {
    /// Create a new budget with total capacity in bytes
    pub fn new(total_bytes: u64) -> Result<Self, MemoryError> {
        if total_bytes == 0 {
            return Err(MemoryError::InvalidSize(total_bytes));
        }
        Ok(Self {
            total: total_bytes,
            allocations: BTreeMap::new(),
            used: 0,
            peak_used: 0,
        })
    }

    /// Total budget in bytes
    pub fn total_bytes(&self) -> u64 {
        self.total
    }

    /// Currently used bytes (sum of all allocations)
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Highest simultaneously tracked usage since this budget was created.
    pub fn peak_used_bytes(&self) -> u64 {
        self.peak_used
    }

    /// Reserved bytes – currently alias for used, but kept for API clarity
    /// In future, reserved vs actively used may diverge.
    pub fn reserved_bytes(&self) -> u64 {
        self.used
    }

    /// Available bytes = total - used.
    ///
    /// `MemoryBudget` maintains the invariant `used <= total`; a violation is
    /// a programming/accounting error and must not be hidden by saturation.
    pub fn available_bytes(&self) -> u64 {
        self.total
            .checked_sub(self.used)
            .expect("MemoryBudget invariant violated: used bytes exceed total")
    }

    /// Check if a given size can be allocated.
    pub fn can_allocate(&self, bytes: u64) -> bool {
        bytes <= self.available_bytes()
    }

    /// Reserve / allocate `bytes` with a meaningful name
    ///
    /// Fails if:
    /// - bytes == 0
    /// - name already exists
    /// - not enough available budget
    pub fn allocate(&mut self, name: impl Into<String>, bytes: u64) -> Result<(), MemoryError> {
        let name = name.into();
        if bytes == 0 {
            return Err(MemoryError::InvalidSize(bytes));
        }
        if self.allocations.contains_key(&name) {
            return Err(MemoryError::AlreadyExists { name });
        }
        if !self.can_allocate(bytes) {
            return Err(MemoryError::Insufficient {
                name: name.clone(),
                requested: bytes,
                available: self.available_bytes(),
                total: self.total,
                used: self.used,
            });
        }
        let new_used = self
            .used
            .checked_add(bytes)
            .ok_or_else(|| MemoryError::AccountingOverflow {
                name: name.clone(),
                requested: bytes,
                used: self.used,
            })?;
        self.allocations.insert(name, bytes);
        self.used = new_used;
        self.peak_used = self.peak_used.max(self.used);
        Ok(())
    }

    /// Alias for allocate, for semantic clarity when reserving for future use
    pub fn reserve(&mut self, name: impl Into<String>, bytes: u64) -> Result<(), MemoryError> {
        self.allocate(name, bytes)
    }

    /// Release an allocation by name, returning its size if it existed.
    ///
    /// The allocation is removed only after the accounting subtraction is
    /// checked. An underflow is reported instead of being silently saturated.
    pub fn release(&mut self, name: &str) -> Result<u64, MemoryError> {
        let bytes = self
            .allocations
            .get(name)
            .copied()
            .ok_or_else(|| MemoryError::NotFound {
                name: name.to_string(),
            })?;
        let new_used = self
            .used
            .checked_sub(bytes)
            .ok_or_else(|| MemoryError::AccountingUnderflow {
                name: name.to_string(),
                released: bytes,
                used: self.used,
            })?;
        self.allocations.remove(name);
        self.used = new_used;
        Ok(bytes)
    }

    /// Atomically resize an existing named allocation and return its old size.
    ///
    /// This is used to settle a conservative load-transient charge to the
    /// exact resident size without briefly removing the charge while the
    /// resident object is already alive. If a growth cannot fit, both the
    /// allocation and total usage remain unchanged.
    pub fn resize(&mut self, name: &str, new_bytes: u64) -> Result<u64, MemoryError> {
        if new_bytes == 0 {
            return Err(MemoryError::InvalidSize(new_bytes));
        }

        let old_bytes =
            self.allocations
                .get(name)
                .copied()
                .ok_or_else(|| MemoryError::NotFound {
                    name: name.to_string(),
                })?;

        let new_used = if new_bytes > old_bytes {
            let additional = new_bytes - old_bytes;
            if !self.can_allocate(additional) {
                return Err(MemoryError::Insufficient {
                    name: name.to_string(),
                    requested: additional,
                    available: self.available_bytes(),
                    total: self.total,
                    used: self.used,
                });
            }
            self.used
                .checked_add(additional)
                .ok_or_else(|| MemoryError::AccountingOverflow {
                    name: name.to_string(),
                    requested: additional,
                    used: self.used,
                })?
        } else {
            self.used
                .checked_sub(old_bytes - new_bytes)
                .ok_or_else(|| MemoryError::AccountingUnderflow {
                    name: name.to_string(),
                    released: old_bytes - new_bytes,
                    used: self.used,
                })?
        };

        *self
            .allocations
            .get_mut(name)
            .expect("allocation existence checked above") = new_bytes;
        self.used = new_used;
        self.peak_used = self.peak_used.max(self.used);
        Ok(old_bytes)
    }

    /// Commit a physically completed resize that was protected by a separate
    /// reservation.
    ///
    /// `target` is the existing charge for the old allocation and
    /// `reservation` is a charge for the new allocation. While the physical
    /// resize is in progress both charges must exist, because old and new
    /// storage may overlap. On success they are atomically replaced by one
    /// `target` charge of `final_bytes`.
    pub fn commit_resize(
        &mut self,
        target: &str,
        reservation: &str,
        final_bytes: u64,
    ) -> Result<(), MemoryError> {
        if final_bytes == 0 {
            return Err(MemoryError::InvalidSize(final_bytes));
        }
        if target == reservation {
            return Err(MemoryError::AlreadyExists {
                name: reservation.to_string(),
            });
        }
        let old_bytes = self
            .allocations
            .get(target)
            .copied()
            .ok_or_else(|| MemoryError::NotFound {
                name: target.to_string(),
            })?;
        let reserved_bytes = self
            .allocations
            .get(reservation)
            .copied()
            .ok_or_else(|| MemoryError::NotFound {
                name: reservation.to_string(),
            })?;
        let base_used = self
            .used
            .checked_sub(old_bytes)
            .and_then(|used| used.checked_sub(reserved_bytes))
            .ok_or_else(|| MemoryError::AccountingUnderflow {
                name: target.to_string(),
                released: old_bytes,
                used: self.used,
            })?;
        let new_used = base_used
            .checked_add(final_bytes)
            .ok_or_else(|| MemoryError::AccountingOverflow {
                name: target.to_string(),
                requested: final_bytes,
                used: base_used,
            })?;
        if new_used > self.total {
            return Err(MemoryError::Insufficient {
                name: target.to_string(),
                requested: final_bytes,
                available: self
                    .total
                    .checked_sub(base_used)
                    .expect("MemoryBudget invariant violated during resize commit"),
                total: self.total,
                used: base_used,
            });
        }

        self.allocations.remove(target);
        self.allocations.remove(reservation);
        self.allocations.insert(target.to_string(), final_bytes);
        self.used = new_used;
        self.peak_used = self.peak_used.max(new_used);
        Ok(())
    }

    /// Atomically rename every allocation whose name starts with `old_prefix`.
    /// Byte totals and peak usage are unchanged. All destination names are
    /// validated before any mutation occurs.
    pub fn rename_prefix(
        &mut self,
        old_prefix: &str,
        new_prefix: &str,
    ) -> Result<Vec<String>, MemoryError> {
        let renames: Vec<(String, String)> = self
            .allocations
            .keys()
            .filter(|name| name.starts_with(old_prefix))
            .map(|old_name| {
                (
                    old_name.clone(),
                    format!("{}{}", new_prefix, &old_name[old_prefix.len()..]),
                )
            })
            .collect();
        if renames.is_empty() {
            return Err(MemoryError::NotFound {
                name: old_prefix.to_string(),
            });
        }
        for (_, new_name) in &renames {
            if self.allocations.contains_key(new_name) {
                return Err(MemoryError::AlreadyExists {
                    name: new_name.clone(),
                });
            }
        }

        let mut new_names = Vec::with_capacity(renames.len());
        for (old_name, new_name) in renames {
            let bytes = self
                .allocations
                .remove(&old_name)
                .expect("rename source validated above");
            self.allocations.insert(new_name.clone(), bytes);
            new_names.push(new_name);
        }
        Ok(new_names)
    }

    /// Get allocation size by name
    pub fn get(&self, name: &str) -> Option<u64> {
        self.allocations.get(name).copied()
    }

    /// List all allocations
    pub fn allocations(&self) -> &BTreeMap<String, u64> {
        &self.allocations
    }

    /// Human readable summary
    pub fn summary(&self) -> String {
        format!(
            "total={} used={} available={}",
            self.total,
            self.used,
            self.available_bytes()
        )
    }

    /// Execute `f` while a temporary allocation of `bytes` is charged under
    /// `name`, releasing it afterwards no matter how `f` returns.
    ///
    /// This is the scoped helper for short-lived working buffers (activations,
    /// logits, sampler temporaries, attention scores, streamed chunks, ...):
    /// the reservation is released exactly once on both ordinary success and
    /// error returns. The closure receives `&mut MemoryBudget` so
    /// nested accounting (e.g. per-layer allocations) keeps working.
    ///
    /// A `bytes == 0` reservation is a no-op (runs `f` without accounting).
    pub fn with_temp<T, E>(
        &mut self,
        name: &str,
        bytes: u64,
        f: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<MemoryError>,
    {
        if bytes == 0 {
            return f(self);
        }
        self.allocate(name, bytes).map_err(E::from)?;
        // Catch the closure's unwind long enough to release the charge, then
        // resume the original panic. Without this guard, a panic in a caller
        // or test hook would leave the logical reservation live until the
        // entire budget is dropped.
        let result = catch_unwind(AssertUnwindSafe(|| f(self)));
        let release_result = self.release(name);
        match (result, release_result) {
            (Ok(result), Ok(_)) => result,
            (Ok(_), Err(error)) => Err(E::from(error)),
            (Err(payload), Ok(_)) => resume_unwind(payload),
            (Err(_), Err(error)) => {
                panic!("temporary allocation '{}' failed to release during unwind: {}", name, error)
            }
        }
    }

    /// Adapt the existing single host-memory budget to the Phase-3 budget
    /// representation without changing allocation or enforcement behavior.
    pub fn resource_budgets(&self) -> crate::resource::ResourceBudgets {
        crate::resource::ResourceBudgets::from_memory_budget(self)
    }
}

/// Parse a human-friendly memory size string into exact bytes
///
/// Accepted syntax (case-insensitive, optional whitespace):
/// - `1024` or `1024B` → bytes
/// - `8K`, `8KB`, `8KiB`, `8Ki` → KiB (1024)
/// - `8M`, `8MB`, `8MiB`, `8Mi` → MiB (1024^2)
/// - `8G`, `8GB`, `8GiB`, `8Gi` → GiB (1024^3)
/// - `8T`, `8TB`, `8TiB`, `8Ti` → TiB (1024^4)
///
/// Also supports decimal variants where `KB` = 1000, `MB` = 1000^2, etc.
/// when the unit is explicitly `KB`, `MB`, `GB`, `TB` without `i`.
/// For simplicity, `K`, `M`, `G`, `T` alone are treated as binary (KiB, MiB, GiB, TiB)
/// which matches common RAM budgeting.
///
/// Float values are allowed: `1.5G` → 1.5 GiB
///
/// Rejects malformed or ambiguous values with clear errors.
pub fn parse_memory_size(s: &str) -> Result<u64, ParseSizeError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(ParseSizeError::Empty);
    }

    // Split numeric prefix and unit suffix
    // Find first char that is not digit, dot, or minus/plus (we will reject negative later)
    let mut num_end = 0;
    let chars: Vec<char> = trimmed.chars().collect();
    let mut seen_dot = false;
    for (i, c) in chars.iter().enumerate() {
        if c.is_ascii_digit() {
            num_end = i + 1;
        } else if *c == '.' && !seen_dot {
            seen_dot = true;
            num_end = i + 1;
        } else if i == 0 && (*c == '+' || *c == '-') {
            // allow sign for error handling, but will reject negative
            num_end = i + 1;
        } else {
            break;
        }
    }

    if num_end == 0 {
        return Err(ParseSizeError::InvalidFormat(format!(
            "no numeric part in '{}'",
            s
        )));
    }

    let num_str = trimmed[..num_end].trim();
    let unit_str = trimmed[num_end..].trim().to_ascii_lowercase();

    // Parse number as f64 to allow float
    let num: f64 = num_str.parse().map_err(|_| {
        ParseSizeError::InvalidFormat(format!("invalid number '{}' in '{}'", num_str, s))
    })?;

    if num <= 0.0 {
        return Err(ParseSizeError::NonPositive(s.to_string()));
    }

    if !num.is_finite() {
        return Err(ParseSizeError::InvalidFormat(format!(
            "non-finite number in '{}'",
            s
        )));
    }

    let multiplier: u64 = match unit_str.as_str() {
        "" | "b" => 1,
        // KiB variants
        "k" | "ki" | "kib" | "kb" => {
            // For KB we treat as 1000 if we want to distinguish, but per doc we treat K/KB as 1024 for RAM budgeting
            // To support both, we check: if unit is exactly "kb" we use 1000, otherwise 1024.
            // However spec examples: 8G, 8GiB, 8192M, 512MiB – all binary works.
            // We'll implement: KB = 1000, KiB/K = 1024
            if unit_str == "kb" {
                1_000
            } else {
                1_024
            }
        }
        "m" | "mi" | "mib" => 1_024u64.pow(2),
        "mb" => 1_000_000,
        "g" | "gi" | "gib" => 1_024u64.pow(3),
        "gb" => 1_000_000_000,
        "t" | "ti" | "tib" => 1_024u64.pow(4),
        "tb" => 1_000_000_000_000,
        _ => {
            return Err(ParseSizeError::UnknownUnit(format!(
                "unit '{}' in '{}' is not recognized (expected B, K, M, G, T, KiB, MiB, GiB, TiB, KB, MB, GB, TB)",
                unit_str, s
            )))
        }
    };

    // Compute bytes: num * multiplier, check overflow
    let bytes_f = num * (multiplier as f64);
    // `u64::MAX` is rounded to 2^64 in f64, so reject that boundary as
    // well as non-finite products before the float-to-integer conversion.
    if !bytes_f.is_finite() || bytes_f >= (u64::MAX as f64) {
        return Err(ParseSizeError::Overflow(s.to_string()));
    }
    let bytes = bytes_f as u64;

    if bytes == 0 {
        return Err(ParseSizeError::NonPositive(s.to_string()));
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_sizes() {
        assert_eq!(parse_memory_size("1024").unwrap(), 1024);
        assert_eq!(parse_memory_size("1024B").unwrap(), 1024);
        assert_eq!(parse_memory_size("8K").unwrap(), 8 * 1024);
        assert_eq!(parse_memory_size("8KB").unwrap(), 8 * 1000);
        assert_eq!(parse_memory_size("8KiB").unwrap(), 8 * 1024);
        assert_eq!(parse_memory_size("8M").unwrap(), 8 * 1024 * 1024);
        assert_eq!(parse_memory_size("512MiB").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_memory_size("8G").unwrap(), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("8GiB").unwrap(), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("8192M").unwrap(), 8192 * 1024 * 1024);
        assert_eq!(
            parse_memory_size("1.5G").unwrap(),
            (1.5 * 1024.0 * 1024.0 * 1024.0) as u64
        );
        assert_eq!(parse_memory_size(" 8G ").unwrap(), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("8g").unwrap(), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("8Gi").unwrap(), 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_memory_size("").is_err());
        assert!(parse_memory_size("   ").is_err());
        assert!(parse_memory_size("abc").is_err());
        assert!(parse_memory_size("8X").is_err());
        assert!(parse_memory_size("-8G").is_err());
        assert!(parse_memory_size("0G").is_err());
        assert!(parse_memory_size("0").is_err());
        assert!(parse_memory_size("18446744073709551615B").is_err());
    }

    #[test]
    fn test_budget_enforcement() {
        let mut budget = MemoryBudget::new(8 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(budget.total_bytes(), 8 * 1024 * 1024 * 1024);
        assert_eq!(budget.available_bytes(), 8 * 1024 * 1024 * 1024);
        budget.allocate("cache", 4 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(budget.used_bytes(), 4 * 1024 * 1024 * 1024);
        assert_eq!(budget.available_bytes(), 4 * 1024 * 1024 * 1024);
        // Exceeding should fail
        let err = budget
            .allocate("too_big", 5 * 1024 * 1024 * 1024)
            .unwrap_err();
        match err {
            MemoryError::Insufficient { .. } => {}
            _ => panic!("expected Insufficient"),
        }
        // Release and allocate again
        budget.release("cache").unwrap();
        assert_eq!(budget.used_bytes(), 0);
        budget.allocate("new", 8 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(budget.available_bytes(), 0);
    }

    #[test]
    fn test_budget_accounting() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("a", 400).unwrap();
        budget.allocate("b", 300).unwrap();
        assert_eq!(budget.used_bytes(), 700);
        assert_eq!(budget.available_bytes(), 300);
        assert_eq!(budget.get("a"), Some(400));
        budget.release("a").unwrap();
        assert_eq!(budget.used_bytes(), 300);
        assert_eq!(budget.available_bytes(), 700);
        assert_eq!(budget.peak_used_bytes(), 700);
    }

    #[test]
    fn test_duplicate_allocation() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("a", 100).unwrap();
        let err = budget.allocate("a", 100).unwrap_err();
        match err {
            MemoryError::AlreadyExists { .. } => {}
            _ => panic!("expected AlreadyExists"),
        }
    }

    #[test]
    fn test_resize_is_atomic_for_settle_and_failure() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("load", 600).unwrap();
        budget.allocate("other", 300).unwrap();

        // Settling a conservative transient charge never removes its name.
        assert_eq!(budget.resize("load", 200).unwrap(), 600);
        assert_eq!(budget.get("load"), Some(200));
        assert_eq!(budget.used_bytes(), 500);

        // A failed growth leaves both the charge and total exactly unchanged.
        let before = budget.used_bytes();
        let err = budget.resize("load", 800).unwrap_err();
        assert!(matches!(err, MemoryError::Insufficient { .. }));
        assert_eq!(budget.get("load"), Some(200));
        assert_eq!(budget.used_bytes(), before);
    }

    #[test]
    fn test_commit_resize_transfers_overlap_reservation_atomically() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("kv_cache", 200).unwrap();
        budget.allocate("kv_cache:growth", 400).unwrap();
        budget.commit_resize("kv_cache", "kv_cache:growth", 400).unwrap();
        assert_eq!(budget.get("kv_cache"), Some(400));
        assert!(budget.get("kv_cache:growth").is_none());
        assert_eq!(budget.used_bytes(), 400);
    }

    #[test]
    fn test_commit_resize_failure_preserves_both_charges() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("kv_cache", 200).unwrap();
        budget.allocate("kv_cache:growth", 400).unwrap();
        budget.allocate("other", 400).unwrap();
        let before = budget.used_bytes();
        let error = budget
            .commit_resize("kv_cache", "kv_cache:growth", 700)
            .unwrap_err();
        assert!(matches!(error, MemoryError::Insufficient { .. }));
        assert_eq!(budget.used_bytes(), before);
        assert_eq!(budget.get("kv_cache"), Some(200));
        assert_eq!(budget.get("kv_cache:growth"), Some(400));
        assert_eq!(budget.get("other"), Some(400));
    }

    #[test]
    fn test_release_reports_missing_or_underflow_without_saturation() {
        let mut budget = MemoryBudget::new(100).unwrap();
        assert!(matches!(
            budget.release("missing"),
            Err(MemoryError::NotFound { .. })
        ));
        budget.allocate("value", 40).unwrap();
        assert_eq!(budget.release("value").unwrap(), 40);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn test_rename_prefix_preserves_exact_accounting() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("layer:2:a", 100).unwrap();
        budget.allocate("layer:2:b", 200).unwrap();
        budget.allocate("other", 50).unwrap();
        let before = budget.used_bytes();

        let mut names = budget.rename_prefix("layer:2:", "cache:layer:2:").unwrap();
        names.sort();
        assert_eq!(
            names,
            vec!["cache:layer:2:a".to_string(), "cache:layer:2:b".to_string(),]
        );
        assert_eq!(budget.used_bytes(), before);
        assert_eq!(budget.get("cache:layer:2:a"), Some(100));
        assert_eq!(budget.get("cache:layer:2:b"), Some(200));
        assert!(budget.get("layer:2:a").is_none());
        assert_eq!(budget.get("other"), Some(50));
    }

    #[test]
    fn test_with_temp_releases_on_success() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let v = budget
            .with_temp("scratch", 400, |b| {
                assert_eq!(b.used_bytes(), 400);
                Ok::<u32, MemoryError>(42u32)
            })
            .unwrap();
        assert_eq!(v, 42);
        assert_eq!(
            budget.used_bytes(),
            0,
            "temp must be released after success"
        );
        assert!(budget.get("scratch").is_none());
    }

    #[test]
    fn test_with_temp_releases_on_error() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let r: Result<(), String> = budget.with_temp("scratch", 400, |b| {
            assert_eq!(b.used_bytes(), 400);
            // nested allocation still works while guard is active
            b.allocate("nested", 100)?;
            b.release("nested")?;
            Err("inner failure".to_string())
        });
        assert!(r.is_err());
        assert_eq!(budget.used_bytes(), 0, "temp must be released after error");
        assert!(budget.get("scratch").is_none());
        assert!(budget.get("nested").is_none());
    }

    #[test]
    fn test_with_temp_zero_bytes_is_noop() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let v = budget
            .with_temp("noop", 0, |b| {
                assert_eq!(b.used_bytes(), 0);
                Ok::<u32, MemoryError>(7u32)
            })
            .unwrap();
        assert_eq!(v, 7);
        assert!(budget.get("noop").is_none());
    }

    #[test]
    fn test_with_temp_releases_on_panic_before_resuming_unwind() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result: Result<(), MemoryError> = budget.with_temp("scratch", 400, |b| {
                assert_eq!(b.get("scratch"), Some(400));
                panic!("synthetic temporary-scope panic");
            });
            let _ = result;
        }));
        assert!(panic.is_err());
        assert_eq!(budget.used_bytes(), 0);
        assert!(budget.get("scratch").is_none());
    }

    #[test]
    fn test_with_temp_insufficient_budget() {
        let mut budget = MemoryBudget::new(100).unwrap();
        let r: Result<(), String> = budget.with_temp("huge", 200, |_| Ok(()));
        assert!(r.is_err());
        assert_eq!(budget.used_bytes(), 0);
    }
}
