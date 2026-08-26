//! Lightweight optional generation profiler using only `std`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub(crate) enum ProfileEvent {
    Prompt,
    LayerLoad,
    LayerCompute,
    LayerRelease,
    TensorConstruction,
    Dequantization,
    FloatMatvec,
    QuantizedMatvec,
    Allocation,
    Logits,
    Sampling,
    Output,
    TokenLatency,
    Total,
}

#[derive(Debug, Default)]
struct ProfileCounters {
    enabled: AtomicBool,
    prompt_ns: AtomicU64,
    layer_load_ns: AtomicU64,
    layer_compute_ns: AtomicU64,
    layer_release_ns: AtomicU64,
    tensor_construction_ns: AtomicU64,
    dequantization_ns: AtomicU64,
    float_matvec_ns: AtomicU64,
    quantized_matvec_ns: AtomicU64,
    allocation_ns: AtomicU64,
    logits_ns: AtomicU64,
    sampling_ns: AtomicU64,
    output_ns: AtomicU64,
    token_latency_ns: AtomicU64,
    max_token_latency_ns: AtomicU64,
    total_ns: AtomicU64,
    tokens: AtomicU64,
    layer_loads: AtomicU64,
    layer_releases: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Profiler {
    counters: Arc<ProfileCounters>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProfileSnapshot {
    pub prompt: Duration,
    pub layer_load: Duration,
    pub layer_compute: Duration,
    pub layer_release: Duration,
    pub tensor_construction: Duration,
    pub dequantization: Duration,
    pub float_matvec: Duration,
    pub quantized_matvec: Duration,
    pub allocation: Duration,
    pub logits: Duration,
    pub sampling: Duration,
    pub output: Duration,
    pub token_latency_total: Duration,
    pub max_token_latency: Duration,
    pub total: Duration,
    pub tokens: u64,
    pub layer_loads: u64,
    pub layer_releases: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl Profiler {
    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.counters.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.counters.enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn reset(&self) {
        for counter in [
            &self.counters.prompt_ns,
            &self.counters.layer_load_ns,
            &self.counters.layer_compute_ns,
            &self.counters.layer_release_ns,
            &self.counters.tensor_construction_ns,
            &self.counters.dequantization_ns,
            &self.counters.float_matvec_ns,
            &self.counters.quantized_matvec_ns,
            &self.counters.allocation_ns,
            &self.counters.logits_ns,
            &self.counters.sampling_ns,
            &self.counters.output_ns,
            &self.counters.token_latency_ns,
            &self.counters.max_token_latency_ns,
            &self.counters.total_ns,
            &self.counters.tokens,
            &self.counters.layer_loads,
            &self.counters.layer_releases,
            &self.counters.cache_hits,
            &self.counters.cache_misses,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    pub(crate) fn start(&self) -> Option<Instant> {
        self.is_enabled().then(Instant::now)
    }

    pub(crate) fn record_since(&self, event: ProfileEvent, started: Option<Instant>) {
        if let Some(started) = started {
            self.record(event, started.elapsed());
        }
    }

    pub(crate) fn record(&self, event: ProfileEvent, elapsed: Duration) {
        if !self.is_enabled() {
            return;
        }
        let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        let target = match event {
            ProfileEvent::Prompt => &self.counters.prompt_ns,
            ProfileEvent::LayerLoad => &self.counters.layer_load_ns,
            ProfileEvent::LayerCompute => &self.counters.layer_compute_ns,
            ProfileEvent::LayerRelease => &self.counters.layer_release_ns,
            ProfileEvent::TensorConstruction => &self.counters.tensor_construction_ns,
            ProfileEvent::Dequantization => &self.counters.dequantization_ns,
            ProfileEvent::FloatMatvec => &self.counters.float_matvec_ns,
            ProfileEvent::QuantizedMatvec => &self.counters.quantized_matvec_ns,
            ProfileEvent::Allocation => &self.counters.allocation_ns,
            ProfileEvent::Logits => &self.counters.logits_ns,
            ProfileEvent::Sampling => &self.counters.sampling_ns,
            ProfileEvent::Output => &self.counters.output_ns,
            ProfileEvent::TokenLatency => {
                update_max(&self.counters.max_token_latency_ns, nanos);
                &self.counters.token_latency_ns
            }
            ProfileEvent::Total => &self.counters.total_ns,
        };
        target.fetch_add(nanos, Ordering::Relaxed);
    }

    pub(crate) fn record_token(&self) {
        if self.is_enabled() {
            self.counters.tokens.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_layer_load(&self) {
        if self.is_enabled() {
            self.counters.layer_loads.fetch_add(1, Ordering::Relaxed);
            // There is currently no layer-weight cache in the inference path;
            // every disk-backed layer load is therefore an explicit miss.
            self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_layer_release(&self) {
        if self.is_enabled() {
            self.counters.layer_releases.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn snapshot(&self) -> ProfileSnapshot {
        let load = |counter: &AtomicU64| Duration::from_nanos(counter.load(Ordering::Relaxed));
        ProfileSnapshot {
            prompt: load(&self.counters.prompt_ns),
            layer_load: load(&self.counters.layer_load_ns),
            layer_compute: load(&self.counters.layer_compute_ns),
            layer_release: load(&self.counters.layer_release_ns),
            tensor_construction: load(&self.counters.tensor_construction_ns),
            dequantization: load(&self.counters.dequantization_ns),
            float_matvec: load(&self.counters.float_matvec_ns),
            quantized_matvec: load(&self.counters.quantized_matvec_ns),
            allocation: load(&self.counters.allocation_ns),
            logits: load(&self.counters.logits_ns),
            sampling: load(&self.counters.sampling_ns),
            output: load(&self.counters.output_ns),
            token_latency_total: load(&self.counters.token_latency_ns),
            max_token_latency: load(&self.counters.max_token_latency_ns),
            total: load(&self.counters.total_ns),
            tokens: self.counters.tokens.load(Ordering::Relaxed),
            layer_loads: self.counters.layer_loads.load(Ordering::Relaxed),
            layer_releases: self.counters.layer_releases.load(Ordering::Relaxed),
            cache_hits: self.counters.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.counters.cache_misses.load(Ordering::Relaxed),
        }
    }
}

fn update_max(counter: &AtomicU64, value: u64) {
    let mut current = counter.load(Ordering::Relaxed);
    while value > current {
        match counter.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_disabled_is_noop_and_enabled_counts() {
        let profiler = Profiler::default();
        profiler.record(ProfileEvent::LayerLoad, Duration::from_millis(5));
        assert_eq!(profiler.snapshot(), ProfileSnapshot::default());

        profiler.set_enabled(true);
        profiler.reset();
        profiler.record(ProfileEvent::LayerLoad, Duration::from_millis(5));
        profiler.record(ProfileEvent::TokenLatency, Duration::from_millis(7));
        profiler.record_token();
        profiler.record_layer_load();
        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.layer_load, Duration::from_millis(5));
        assert_eq!(snapshot.token_latency_total, Duration::from_millis(7));
        assert_eq!(snapshot.max_token_latency, Duration::from_millis(7));
        assert_eq!(snapshot.tokens, 1);
        assert_eq!(snapshot.layer_loads, 1);
        assert_eq!(snapshot.cache_misses, 1);
    }
}
