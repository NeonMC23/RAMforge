//! KV cache for autoregressive generation
//!
//! Explicitly represented, grows as tokens are generated, avoids recomputing
//! previous tokens, accounts for memory usage in RAMforge budget.

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
    // Memory accounting
    pub bytes_per_layer: usize,
}

impl KvCache {
    /// Growth granularity: capacity grows in chunks of this many tokens to
    /// avoid paying a realloc per generated token, without allocating the
    /// full possible context upfront.
    pub const GROW_CHUNK_TOKENS: usize = 256;

    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err("invalid KV cache dimensions".to_string());
        }

        let elems_per_layer = max_seq_len * n_kv_heads * head_dim;
        let bytes_per_layer = elems_per_layer * 4 * 2; // K and V, f32

        let k_caches = vec![vec![0.0f32; elems_per_layer]; n_layers];
        let v_caches = vec![vec![0.0f32; elems_per_layer]; n_layers];

        Ok(Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            k_caches,
            v_caches,
            bytes_per_layer,
        })
    }

    /// Current capacity in tokens
    pub fn capacity_tokens(&self) -> usize {
        self.max_seq_len
    }

    /// Bytes a cache with capacity `tokens` would occupy across all layers.
    pub fn bytes_for_tokens(&self, tokens: usize) -> usize {
        tokens * self.n_kv_heads * self.head_dim * 4 * 2 * self.n_layers
    }

    /// Smallest chunk-aligned capacity that can hold `required` tokens.
    pub fn chunk_aligned_capacity(&self, required: usize) -> usize {
        let chunk = Self::GROW_CHUNK_TOKENS;
        required.div_ceil(chunk) * chunk
    }

    /// Grow the backing buffers to exactly `new_capacity` tokens.
    ///
    /// Chunk granularity is the caller's responsibility (see
    /// `chunk_aligned_capacity`); exact accounting keeps the budget charge
    /// (`bytes_for_tokens`) in lockstep with the backing buffers.
    /// Callers must reconcile the byte delta with the memory budget BEFORE
    /// calling this; on success, existing entries are preserved. Shrinking
    /// is not supported.
    pub fn grow_to(&mut self, new_capacity: usize) -> Result<(), String> {
        if new_capacity <= self.max_seq_len {
            return Ok(()); // no shrink / no-op
        }
        let elems_per_layer = new_capacity * self.n_kv_heads * self.head_dim;
        for k in self.k_caches.iter_mut() {
            k.resize(elems_per_layer, 0.0);
        }
        for v in self.v_caches.iter_mut() {
            v.resize(elems_per_layer, 0.0);
        }
        self.max_seq_len = new_capacity;
        self.bytes_per_layer = elems_per_layer * 4 * 2;
        Ok(())
    }

    /// Total bytes for all layers (current capacity)
    pub fn total_bytes(&self) -> usize {
        self.bytes_per_layer * self.n_layers
    }

    /// Try to allocate KV cache memory from budget
    pub fn allocate_from_budget(&self, budget: &mut MemoryBudget) -> Result<(), MemoryError> {
        budget.allocate("kv_cache", self.total_bytes() as u64)
    }

    /// Append K and V for a new token at current seq_len position
    /// k and v are [n_kv_heads * head_dim] each
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        if layer >= self.n_layers {
            return Err(format!("layer {} out of bounds", layer));
        }
        if self.seq_len >= self.max_seq_len {
            return Err("KV cache full".to_string());
        }
        let expected = self.n_kv_heads * self.head_dim;
        if k.len() != expected || v.len() != expected {
            return Err(format!(
                "K/V size mismatch: expected {}, got k={}, v={}",
                expected,
                k.len(),
                v.len()
            ));
        }

        let offset = self.seq_len * expected;
        self.k_caches[layer][offset..offset + expected].copy_from_slice(k);
        self.v_caches[layer][offset..offset + expected].copy_from_slice(v);

        Ok(())
    }

    /// Increment seq_len after appending for all layers
    pub fn increment_seq_len(&mut self) {
        self.seq_len += 1;
    }

    /// Get K cache for a layer up to current seq_len (flattened)
    pub fn get_k(&self, layer: usize) -> &[f32] {
        let expected = self.n_kv_heads * self.head_dim;
        let len = self.seq_len * expected;
        &self.k_caches[layer][..len]
    }

    pub fn get_v(&self, layer: usize) -> &[f32] {
        let expected = self.n_kv_heads * self.head_dim;
        let len = self.seq_len * expected;
        &self.v_caches[layer][..len]
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
}
