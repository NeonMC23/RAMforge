//! Memory budget abstraction for RAMforge
//!
//! RAMforge-managed memory is defined as memory that is explicitly tracked
//! via `MemoryBudget`. It does NOT represent total process RSS or OS page
//! cache. It only tracks allocations that RAMforge itself accounts for, such
//! as tensor cache, KV cache, prefetch buffers, etc.
//!
//! This design makes it difficult for future code to bypass the budget
//! accidentally: all RAMforge-managed allocations should go through
//! `MemoryBudget::allocate` / `reserve`.

use std::collections::BTreeMap;

use crate::error::{MemoryError, ParseSizeError};

/// A reusable memory budget abstraction
///
/// Internally all values are exact byte counts (u64).
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    total: u64,
    allocations: BTreeMap<String, u64>,
    used: u64,
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

    /// Reserved bytes – currently alias for used, but kept for API clarity
    /// In future, reserved vs actively used may diverge.
    pub fn reserved_bytes(&self) -> u64 {
        self.used
    }

    /// Available bytes = total - used
    pub fn available_bytes(&self) -> u64 {
        self.total.saturating_sub(self.used)
    }

    /// Check if a given size can be allocated
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
        self.allocations.insert(name, bytes);
        self.used = self.used.checked_add(bytes).expect("used overflow checked by can_allocate");
        Ok(())
    }

    /// Alias for allocate, for semantic clarity when reserving for future use
    pub fn reserve(&mut self, name: impl Into<String>, bytes: u64) -> Result<(), MemoryError> {
        self.allocate(name, bytes)
    }

    /// Release an allocation by name, returning its size if it existed
    pub fn release(&mut self, name: &str) -> Result<u64, MemoryError> {
        if let Some(bytes) = self.allocations.remove(name) {
            self.used = self.used.saturating_sub(bytes);
            Ok(bytes)
        } else {
            Err(MemoryError::NotFound {
                name: name.to_string(),
            })
        }
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
            self.total, self.used, self.available_bytes()
        )
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
    let num: f64 = num_str
        .parse()
        .map_err(|_| ParseSizeError::InvalidFormat(format!("invalid number '{}' in '{}'", num_str, s)))?;

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
    if bytes_f > (u64::MAX as f64) {
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
        assert_eq!(parse_memory_size("1.5G").unwrap(), (1.5 * 1024.0 * 1024.0 * 1024.0) as u64);
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
        let err = budget.allocate("too_big", 5 * 1024 * 1024 * 1024).unwrap_err();
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
}
