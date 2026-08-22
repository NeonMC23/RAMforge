//! Residency observability for out-of-core layer streaming

#[derive(Debug, Clone, Default)]
pub struct ResidencyStats {
    /// Total model weight bytes (sum of all tensors)
    pub total_model_weight_bytes: u64,
    /// Current resident layer weight bytes
    pub current_resident_layer_bytes: u64,
    /// Peak resident layer weight bytes observed
    pub peak_resident_layer_bytes: u64,
    /// Number of layer loads
    pub num_layer_loads: u64,
    /// Number of layer releases/evictions
    pub num_layer_releases: u64,
    /// Peak RAMforge-managed bytes (budget used)
    pub peak_managed_bytes: u64,
    /// Current managed bytes
    pub current_managed_bytes: u64,
}

impl ResidencyStats {
    pub fn new(total_model_weight_bytes: u64) -> Self {
        Self {
            total_model_weight_bytes,
            ..Default::default()
        }
    }

    pub fn on_layer_load(&mut self, layer_bytes: u64, current_managed: u64) {
        self.num_layer_loads += 1;
        self.current_resident_layer_bytes = layer_bytes;
        if layer_bytes > self.peak_resident_layer_bytes {
            self.peak_resident_layer_bytes = layer_bytes;
        }
        self.current_managed_bytes = current_managed;
        if current_managed > self.peak_managed_bytes {
            self.peak_managed_bytes = current_managed;
        }
    }

    pub fn on_layer_release(&mut self, current_managed: u64) {
        self.num_layer_releases += 1;
        self.current_resident_layer_bytes = 0;
        self.current_managed_bytes = current_managed;
        if current_managed > self.peak_managed_bytes {
            self.peak_managed_bytes = current_managed;
        }
    }

    pub fn update_managed(&mut self, current_managed: u64) {
        self.current_managed_bytes = current_managed;
        if current_managed > self.peak_managed_bytes {
            self.peak_managed_bytes = current_managed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residency_tracking() {
        let mut stats = ResidencyStats::new(1000);
        stats.on_layer_load(100, 500);
        assert_eq!(stats.current_resident_layer_bytes, 100);
        assert_eq!(stats.peak_resident_layer_bytes, 100);
        assert_eq!(stats.num_layer_loads, 1);
        stats.on_layer_load(200, 600);
        assert_eq!(stats.peak_resident_layer_bytes, 200);
        assert_eq!(stats.num_layer_loads, 2);
        stats.on_layer_release(400);
        assert_eq!(stats.current_resident_layer_bytes, 0);
        assert_eq!(stats.num_layer_releases, 1);
        assert_eq!(stats.peak_managed_bytes, 600);
    }
}
