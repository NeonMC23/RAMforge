//! KV cache for autoregressive generation.
//!
//! Each layer stores K and V in the explicit layout
//! `[position][kv_head][head_dim]`. `seq_len` is the number of committed
//! historical positions, so a forward at position `p` must read history
//! `[0, p)` and append its current K/V at slot `p` only after all layers have
//! consumed that history. Prompt and decode use the same append/read contract;
//! only the caller's token-position schedule differs.
//!
//! Explicitly represented, grows as tokens are generated, avoids recomputing
//! previous tokens, and accounts for memory usage in RAMforge's budget.

use ramforge_core::error::MemoryError;
use ramforge_core::memory::MemoryBudget;

/// KV cache for one layer or all layers?
/// We implement per-model cache that holds K and V for all layers.

#[derive(Debug, Clone)]
pub struct KvCache {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    // For each layer, we store contiguous K and V: [max_seq_len * n_kv_heads * head_dim]
    // We track current seq_len
    pub seq_len: usize,
    k_caches: Vec<Vec<f32>>, // per layer
    v_caches: Vec<Vec<f32>>,
    // Memory accounting covers the requested f32 payload capacities. The
    // Vec headers, allocator metadata, and fragmentation remain outside this
    // logical payload ledger.
    pub bytes_per_layer: usize,
}

#[derive(Debug, Clone, Copy)]
struct KvLayout {
    elems_per_layer: usize,
    bytes_per_layer: usize,
    total_bytes: usize,
}

impl KvCache {
    /// Growth granularity: capacity grows in chunks of this many tokens to
    /// avoid paying a realloc per generated token, without allocating the
    /// full possible context upfront.
    pub const GROW_CHUNK_TOKENS: usize = 256;

    /// Construct an uncharged cache for low-level/tests compatibility.
    /// Managed inference must use [`Self::new_with_budget`].
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        let layout = Self::layout(n_layers, n_kv_heads, head_dim, max_seq_len)?;
        Self::allocate_from_layout(
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            layout,
        )
    }

    /// Construct the cache under a live budget reservation.
    ///
    /// The charge is established before any backing vector is reserved. If a
    /// fallible vector reservation fails, the charge is rolled back and no
    /// partially constructed cache is returned.
    pub fn new_with_budget(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        budget: &mut MemoryBudget,
        charge_name: &str,
    ) -> Result<Self, String> {
        let layout = Self::layout(n_layers, n_kv_heads, head_dim, max_seq_len)?;
        let total_bytes = u64::try_from(layout.total_bytes)
            .map_err(|_| "KV cache byte size exceeds u64".to_string())?;
        budget
            .allocate(charge_name, total_bytes)
            .map_err(|error| format!("failed to reserve KV cache: {error}"))?;
        match Self::allocate_from_layout(
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            layout,
        ) {
            Ok(cache) => Ok(cache),
            Err(error) => {
                let rollback = budget.release(charge_name);
                if let Err(rollback_error) = rollback {
                    return Err(format!(
                        "KV cache allocation failed: {error}; reservation rollback failed: {rollback_error}"
                    ));
                }
                Err(error)
            }
        }
    }

    fn layout(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<KvLayout, String> {
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err("invalid KV cache dimensions".to_string());
        }
        let elems_per_layer = max_seq_len
            .checked_mul(n_kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| "KV cache element count overflow".to_string())?;
        let bytes_per_layer = elems_per_layer
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| "KV cache per-layer byte size overflow".to_string())?;
        let total_bytes = bytes_per_layer
            .checked_mul(n_layers)
            .ok_or_else(|| "KV cache total byte size overflow".to_string())?;
        Ok(KvLayout {
            elems_per_layer,
            bytes_per_layer,
            total_bytes,
        })
    }

    fn zeroed_buffer(len: usize, label: &str) -> Result<Vec<f32>, String> {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(len)
            .map_err(|error| format!("failed to reserve {label}: {error}"))?;
        buffer.resize(len, 0.0);
        Ok(buffer)
    }

    fn allocate_from_layout(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        layout: KvLayout,
    ) -> Result<Self, String> {
        let mut k_caches = Vec::new();
        k_caches
            .try_reserve_exact(n_layers)
            .map_err(|error| format!("failed to reserve KV K-layer table: {error}"))?;
        let mut v_caches = Vec::new();
        v_caches
            .try_reserve_exact(n_layers)
            .map_err(|error| format!("failed to reserve KV V-layer table: {error}"))?;
        for _ in 0..n_layers {
            k_caches.push(Self::zeroed_buffer(layout.elems_per_layer, "KV K storage")?);
            v_caches.push(Self::zeroed_buffer(layout.elems_per_layer, "KV V storage")?);
        }

        Ok(Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            k_caches,
            v_caches,
            bytes_per_layer: layout.bytes_per_layer,
        })
    }

    /// Current capacity in tokens
    pub fn capacity_tokens(&self) -> usize {
        self.max_seq_len
    }

    /// Checked byte size for a cache with capacity `tokens` across all layers.
    pub fn try_bytes_for_tokens(&self, tokens: usize) -> Result<usize, String> {
        tokens
            .checked_mul(self.n_kv_heads)
            .and_then(|value| value.checked_mul(self.head_dim))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(self.n_layers))
            .ok_or_else(|| "KV cache byte size overflow".to_string())
    }

    /// Bytes a cache with capacity `tokens` would occupy across all layers.
    ///
    /// New accounting code should use [`Self::try_bytes_for_tokens`] so an
    /// invalid size can be returned to its caller. This compatibility helper
    /// panics rather than silently wrapping if the requested size is not
    /// representable.
    pub fn bytes_for_tokens(&self, tokens: usize) -> usize {
        self.try_bytes_for_tokens(tokens)
            .expect("KV cache byte size must be representable")
    }

    /// Checked smallest chunk-aligned capacity that can hold `required`
    /// tokens.
    pub fn try_chunk_aligned_capacity(&self, required: usize) -> Result<usize, String> {
        let chunk = Self::GROW_CHUNK_TOKENS;
        let rounded = required
            .checked_add(chunk - 1)
            .ok_or_else(|| "KV cache chunk capacity overflow".to_string())?;
        Ok((rounded / chunk)
            .checked_mul(chunk)
            .ok_or_else(|| "KV cache chunk capacity overflow".to_string())?)
    }

    /// Smallest chunk-aligned capacity that can hold `required` tokens.
    /// Panics rather than silently wrapping for an unrepresentable capacity.
    pub fn chunk_aligned_capacity(&self, required: usize) -> usize {
        self.try_chunk_aligned_capacity(required)
            .expect("KV cache chunk capacity must be representable")
    }

    fn allocate_grown_buffers(
        &self,
        new_capacity: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, KvLayout), String> {
        if new_capacity <= self.max_seq_len {
            let elems_per_layer = self
                .max_seq_len
                .checked_mul(self.n_kv_heads)
                .and_then(|value| value.checked_mul(self.head_dim))
                .ok_or_else(|| "existing KV cache element count overflow".to_string())?;
            let total_bytes = self.try_bytes_for_tokens(self.max_seq_len)?;
            return Ok((
                self.k_caches.clone(),
                self.v_caches.clone(),
                KvLayout {
                    elems_per_layer,
                    bytes_per_layer: self.bytes_per_layer,
                    total_bytes,
                },
            ));
        }
        let layout = Self::layout(
            self.n_layers,
            self.n_kv_heads,
            self.head_dim,
            new_capacity,
        )?;
        let old_elems_per_layer = self
            .max_seq_len
            .checked_mul(self.n_kv_heads)
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or_else(|| "existing KV cache element count overflow".to_string())?;
        let mut k_caches = Vec::new();
        k_caches
            .try_reserve_exact(self.n_layers)
            .map_err(|error| format!("failed to reserve grown KV K-layer table: {error}"))?;
        let mut v_caches = Vec::new();
        v_caches
            .try_reserve_exact(self.n_layers)
            .map_err(|error| format!("failed to reserve grown KV V-layer table: {error}"))?;
        for layer in 0..self.n_layers {
            let mut k = Self::zeroed_buffer(layout.elems_per_layer, "grown KV K storage")?;
            k[..old_elems_per_layer].copy_from_slice(&self.k_caches[layer]);
            k_caches.push(k);
            let mut v = Self::zeroed_buffer(layout.elems_per_layer, "grown KV V storage")?;
            v[..old_elems_per_layer].copy_from_slice(&self.v_caches[layer]);
            v_caches.push(v);
        }
        Ok((k_caches, v_caches, layout))
    }

    /// Grow the backing buffers to exactly `new_capacity` tokens.
    ///
    /// New storage is built separately, so an allocation failure leaves the
    /// existing cache unchanged. Budget-aware callers should use
    /// [`Self::grow_to_with_budget`] to reserve the old/new peak overlap before
    /// this replacement is committed.
    pub fn grow_to(&mut self, new_capacity: usize) -> Result<(), String> {
        if new_capacity <= self.max_seq_len {
            return Ok(());
        }
        let (k_caches, v_caches, layout) = self.allocate_grown_buffers(new_capacity)?;
        self.k_caches = k_caches;
        self.v_caches = v_caches;
        self.max_seq_len = new_capacity;
        self.bytes_per_layer = layout.bytes_per_layer;
        Ok(())
    }

    /// Grow with a transactional budget charge.
    ///
    /// The existing charge stays live while a distinct reservation for the
    /// replacement capacity is established. This models the real peak while
    /// old and new buffers overlap. The reservation is rolled back on any
    /// allocation failure; the old cache and charge remain unchanged.
    pub fn grow_to_with_budget(
        &mut self,
        new_capacity: usize,
        budget: &mut MemoryBudget,
        charge_name: &str,
    ) -> Result<(), String> {
        if new_capacity <= self.max_seq_len {
            return Ok(());
        }
        let old_bytes = u64::try_from(self.try_bytes_for_tokens(self.max_seq_len)?)
            .map_err(|_| "existing KV cache byte size exceeds u64".to_string())?;
        let new_bytes = u64::try_from(self.try_bytes_for_tokens(new_capacity)?)
            .map_err(|_| "grown KV cache byte size exceeds u64".to_string())?;
        let reservation_name = format!("{charge_name}:growth");
        budget
            .allocate(&reservation_name, new_bytes)
            .map_err(|error| format!("failed to reserve grown KV cache: {error}"))?;
        if budget.get(charge_name) != Some(old_bytes) {
            let rollback = budget.release(&reservation_name);
            return Err(match rollback {
                Ok(_) => format!(
                    "existing KV cache charge '{}' does not match physical cache size {}",
                    charge_name, old_bytes
                ),
                Err(rollback_error) => format!(
                    "existing KV cache charge '{}' does not match physical cache size {}; growth reservation rollback failed: {}",
                    charge_name, old_bytes, rollback_error
                ),
            });
        }
        let buffers = self.allocate_grown_buffers(new_capacity);
        let (k_caches, v_caches, layout) = match buffers {
            Ok(buffers) => buffers,
            Err(error) => {
                let rollback = budget.release(&reservation_name);
                if let Err(rollback_error) = rollback {
                    return Err(format!(
                        "KV cache growth failed: {error}; growth reservation rollback failed: {rollback_error}"
                    ));
                }
                return Err(error);
            }
        };
        // Swap the replacement into the cache and drop the old physical
        // buffers before releasing the old logical charge. The reservation
        // remains live throughout this handoff, so the new buffers are never
        // uncharged.
        let old_k_caches = std::mem::replace(&mut self.k_caches, k_caches);
        let old_v_caches = std::mem::replace(&mut self.v_caches, v_caches);
        self.max_seq_len = new_capacity;
        self.bytes_per_layer = layout.bytes_per_layer;
        drop(old_k_caches);
        drop(old_v_caches);
        budget
            .commit_resize(charge_name, &reservation_name, new_bytes)
            .unwrap_or_else(|error| {
                panic!("KV cache budget commit failed after physical replacement: {error}")
            });
        Ok(())
    }

    /// Total bytes for all layers (current capacity).
    pub fn total_bytes(&self) -> usize {
        self.bytes_per_layer
            .checked_mul(self.n_layers)
            .expect("KV cache total byte size must be representable")
    }

    /// Charge an already-created cache for compatibility with older callers.
    ///
    /// New managed code must use [`Self::new_with_budget`], which reserves
    /// before allocating. This legacy method cannot retroactively protect the
    /// construction that has already happened.
    pub fn allocate_from_budget(&self, budget: &mut MemoryBudget) -> Result<(), MemoryError> {
        let bytes = u64::try_from(self.total_bytes())
            .map_err(|_| MemoryError::InvalidSize(u64::MAX))?;
        budget.allocate("kv_cache", bytes)
    }

    /// Append K and V for the current token at position `seq_len`.
    ///
    /// `k` and `v` are one position in `[kv_head][head_dim]` layout. The
    /// cache does not advance `seq_len` here because the caller must append
    /// every layer before committing the next history length.
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        if layer >= self.n_layers {
            return Err(format!("layer {} out of bounds", layer));
        }
        if self.seq_len >= self.max_seq_len {
            return Err("KV cache full".to_string());
        }
        let expected = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "K/V element count overflow".to_string())?;
        if k.len() != expected || v.len() != expected {
            return Err(format!(
                "K/V size mismatch: expected {}, got k={}, v={}",
                expected,
                k.len(),
                v.len()
            ));
        }

        let offset = self
            .seq_len
            .checked_mul(expected)
            .ok_or_else(|| "K/V cache offset overflow".to_string())?;
        let end = offset
            .checked_add(expected)
            .ok_or_else(|| "K/V cache slice end overflow".to_string())?;
        self.k_caches[layer][offset..end].copy_from_slice(k);
        self.v_caches[layer][offset..end].copy_from_slice(v);

        Ok(())
    }

    /// Increment seq_len after appending for all layers
    pub fn increment_seq_len(&mut self) {
        debug_assert!(self.seq_len < self.max_seq_len);
        self.seq_len = self
            .seq_len
            .checked_add(1)
            .expect("KV cache sequence length overflow");
    }

    /// Get committed K history `[0, seq_len)` for a layer, flattened as
    /// `[position][kv_head][head_dim]`.
    pub fn get_k(&self, layer: usize) -> &[f32] {
        let expected = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .expect("K/V element count must be representable");
        let len = self
            .seq_len
            .checked_mul(expected)
            .expect("K cache history length must be representable");
        &self.k_caches[layer][..len]
    }

    /// Get committed V history `[0, seq_len)` for a layer, flattened as
    /// `[position][kv_head][head_dim]`.
    pub fn get_v(&self, layer: usize) -> &[f32] {
        let expected = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .expect("K/V element count must be representable");
        let len = self
            .seq_len
            .checked_mul(expected)
            .expect("V cache history length must be representable");
        &self.v_caches[layer][..len]
    }

    /// Read an explicit committed K position range `[start, end)`.
    pub fn get_k_range(&self, layer: usize, start: usize, end: usize) -> Result<&[f32], String> {
        self.validate_read_range(layer, start, end)?;
        let width = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "K/V element count overflow".to_string())?;
        let start_offset = start
            .checked_mul(width)
            .ok_or_else(|| "K cache range start overflow".to_string())?;
        let end_offset = end
            .checked_mul(width)
            .ok_or_else(|| "K cache range end overflow".to_string())?;
        Ok(&self.k_caches[layer][start_offset..end_offset])
    }

    /// Read an explicit committed V position range `[start, end)`.
    pub fn get_v_range(&self, layer: usize, start: usize, end: usize) -> Result<&[f32], String> {
        self.validate_read_range(layer, start, end)?;
        let width = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "K/V element count overflow".to_string())?;
        let start_offset = start
            .checked_mul(width)
            .ok_or_else(|| "V cache range start overflow".to_string())?;
        let end_offset = end
            .checked_mul(width)
            .ok_or_else(|| "V cache range end overflow".to_string())?;
        Ok(&self.v_caches[layer][start_offset..end_offset])
    }

    fn validate_read_range(&self, layer: usize, start: usize, end: usize) -> Result<(), String> {
        if layer >= self.n_layers {
            return Err(format!("layer {} out of bounds", layer));
        }
        if start > end || end > self.seq_len {
            return Err(format!(
                "KV read range {}..{} is outside committed history 0..{}",
                start, end, self.seq_len
            ));
        }
        Ok(())
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KvCache::new(2, 2, 4, 10).unwrap();
        assert_eq!(cache.seq_len(), 0);
        let k = vec![1.0; 8]; // 2*4
        let v = vec![2.0; 8];
        cache.append(0, &k, &v).unwrap();
        cache.append(1, &k, &v).unwrap();
        cache.increment_seq_len();
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.get_k(0).len(), 8);
    }

    #[test]
    fn test_kv_cache_explicit_read_ranges() {
        let mut cache = KvCache::new(1, 1, 2, 4).unwrap();
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.increment_seq_len();
        cache.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        cache.increment_seq_len();
        assert_eq!(cache.get_k_range(0, 1, 2).unwrap(), &[5.0, 6.0]);
        assert_eq!(cache.get_v_range(0, 0, 2).unwrap(), &[3.0, 4.0, 7.0, 8.0]);
        assert!(cache.get_k_range(0, 0, 3).is_err());
        assert!(cache.get_v_range(1, 0, 1).is_err());
    }

    #[test]
    fn test_kv_cache_memory() {
        let cache = KvCache::new(2, 2, 4, 10).unwrap();
        // 2 layers * 10 * 2 *4 *4*2 = 2*10*2*4*8 = 1280 bytes
        assert_eq!(cache.total_bytes(), 1280);
    }

    #[test]
    fn test_kv_cache_full() {
        let mut cache = KvCache::new(1, 1, 2, 1).unwrap();
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.increment_seq_len();
        let err = cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap_err();
        assert!(err.contains("full"));
    }

    #[test]
    fn test_kv_cache_chunked_growth_preserves_data() {
        let mut cache = KvCache::new(2, 1, 2, 1).unwrap();
        assert_eq!(cache.capacity_tokens(), 1);
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.append(1, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        cache.increment_seq_len();

        // Grow: the caller picks chunk-aligned granularity.
        let target = cache.chunk_aligned_capacity(2);
        assert_eq!(target, 256);
        cache.grow_to(target).unwrap();
        assert_eq!(cache.capacity_tokens(), 256);
        // Data preserved
        assert_eq!(cache.get_k(0), &[1.0, 2.0]);
        assert_eq!(cache.get_k(1), &[5.0, 6.0]);
        // Append at grown capacity works
        cache.append(0, &[9.0, 9.0], &[8.0, 8.0]).unwrap();
        cache.append(1, &[1.0, 1.0], &[2.0, 2.0]).unwrap();
        cache.increment_seq_len();
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 9.0, 9.0]);
        // no-op when shrinking
        cache.grow_to(1).unwrap();
        assert_eq!(cache.capacity_tokens(), 256);
    }

    #[test]
    fn test_kv_cache_bytes_for_tokens() {
        let cache = KvCache::new(2, 2, 4, 10).unwrap();
        // 10 tokens * 2 heads * 4 dim * 4B * K&V * 2 layers = 1280
        assert_eq!(cache.bytes_for_tokens(10), 1280);
        assert_eq!(cache.total_bytes(), 1280);
        assert_eq!(cache.bytes_for_tokens(256), 256 * 2 * 4 * 4 * 2 * 2);
    }

    #[test]
    fn test_initial_budget_reservation_precedes_cache_construction() {
        let mut budget = MemoryBudget::new(16).unwrap();
        let error = KvCache::new_with_budget(1, 1, 1, 4, &mut budget, "kv_cache")
            .unwrap_err();
        assert!(error.contains("reserve") || error.contains("memory"));
        assert_eq!(budget.used_bytes(), 0);
        assert!(budget.get("kv_cache").is_none());
    }

    #[test]
    fn test_budgeted_growth_keeps_old_state_when_peak_does_not_fit() {
        let mut budget = MemoryBudget::new(80).unwrap();
        let mut cache = KvCache::new_with_budget(1, 1, 1, 4, &mut budget, "kv_cache").unwrap();
        let old_bytes = cache.total_bytes() as u64;
        let old_capacity = cache.capacity_tokens();
        let error = cache
            .grow_to_with_budget(8, &mut budget, "kv_cache")
            .unwrap_err();
        assert!(error.contains("reserve") || error.contains("memory"));
        assert_eq!(cache.capacity_tokens(), old_capacity);
        assert_eq!(budget.get("kv_cache"), Some(old_bytes));
        assert_eq!(budget.used_bytes(), old_bytes);
        assert!(budget.get("kv_cache:growth").is_none());
    }

    #[test]
    fn test_budgeted_growth_commits_new_charge_after_old_storage_is_dropped() {
        let mut budget = MemoryBudget::new(4096).unwrap();
        let mut cache = KvCache::new_with_budget(1, 1, 1, 1, &mut budget, "kv_cache").unwrap();
        cache.append(0, &[1.0], &[2.0]).unwrap();
        cache.increment_seq_len();
        let old_bytes = cache.total_bytes() as u64;
        cache
            .grow_to_with_budget(256, &mut budget, "kv_cache")
            .unwrap();
        let new_bytes = cache.total_bytes() as u64;
        assert!(new_bytes > old_bytes);
        assert_eq!(budget.get("kv_cache"), Some(new_bytes));
        assert!(budget.get("kv_cache:growth").is_none());
        assert_eq!(budget.used_bytes(), new_bytes);
        assert_eq!(budget.peak_used_bytes(), old_bytes + new_bytes);
        assert_eq!(cache.get_k(0), &[1.0]);
        assert_eq!(cache.get_v(0), &[2.0]);
    }

    #[test]
    fn test_kv_size_and_chunk_arithmetic_rejects_overflow() {
        let cache = KvCache::new(1, 1, 1, 1).unwrap();
        assert!(cache.try_bytes_for_tokens(usize::MAX).is_err());
        assert!(cache.try_chunk_aligned_capacity(usize::MAX).is_err());
        assert!(KvCache::new(1, usize::MAX, 2, 1).is_err());
    }
}
