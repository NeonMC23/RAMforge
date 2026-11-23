//! Bounded transition-history prediction for residency prefetch targets.
//!
//! The predictor never claims per-layer execution order. It observes committed
//! layer-set transitions and predicts the next set from repeat, alternating or
//! constant-delta patterns. The prefetch horizon remains one pending selection
//! because the native transaction has one pending slot.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionStrategy {
    Repeat,
    Alternating,
    ConstantDelta,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionConfidence {
    High,
    Medium,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyPrediction {
    pub layers: Vec<u32>,
    pub strategy: PredictionStrategy,
    pub confidence: PredictionConfidence,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionStatus {
    pub horizon: u32,
    pub predictions: u64,
    pub hits: u64,
    pub misses: u64,
    pub unknown: u64,
    pub hit_rate_permille: Option<u32>,
    pub last_predicted: Option<Vec<u32>>,
    pub last_actual: Option<Vec<u32>>,
    pub last_strategy: PredictionStrategy,
}

#[derive(Debug, Clone)]
pub struct TransitionPredictor {
    history: Vec<Vec<u32>>,
    max_history: usize,
    pending: Option<ResidencyPrediction>,
    predictions: u64,
    hits: u64,
    misses: u64,
    unknown: u64,
    last_actual: Option<Vec<u32>>,
}

impl TransitionPredictor {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 8,
            pending: None,
            predictions: 0,
            hits: 0,
            misses: 0,
            unknown: 0,
            last_actual: None,
        }
    }

    pub fn next_prediction(&self) -> Option<&ResidencyPrediction> {
        self.pending
            .as_ref()
            .filter(|prediction| prediction.strategy != PredictionStrategy::Unknown)
    }

    /// Record an actual committed selection, score the previous prediction,
    /// then compute the next bounded prediction.
    pub fn observe(&mut self, selection: &[u32]) {
        let normalized = normalize(selection);
        if let Some(prediction) = self.pending.take() {
            if prediction.strategy != PredictionStrategy::Unknown {
                self.predictions += 1;
                if prediction.layers == normalized {
                    self.hits += 1;
                } else {
                    self.misses += 1;
                }
            } else {
                self.unknown += 1;
            }
        }
        self.last_actual = Some(normalized.clone());
        self.history.push(normalized);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        self.pending = Some(self.compute_prediction());
    }

    fn compute_prediction(&self) -> ResidencyPrediction {
        let len = self.history.len();
        if len >= 2 && self.history[len - 1] == self.history[len - 2] {
            return ResidencyPrediction {
                layers: self.history[len - 1].clone(),
                strategy: PredictionStrategy::Repeat,
                confidence: PredictionConfidence::High,
                reason: "last two selections are identical".to_string(),
            };
        }
        if len >= 3 && self.history[len - 1] == self.history[len - 3] {
            return ResidencyPrediction {
                layers: self.history[len - 2].clone(),
                strategy: PredictionStrategy::Alternating,
                confidence: if len >= 4 {
                    PredictionConfidence::High
                } else {
                    PredictionConfidence::Medium
                },
                reason: "selection alternates between two sets".to_string(),
            };
        }
        if len >= 3 {
            let previous = constant_shift(&self.history[len - 3], &self.history[len - 2]);
            let latest = constant_shift(&self.history[len - 2], &self.history[len - 1]);
            if let (Some(previous), Some(latest)) = (previous, latest) {
                if previous == latest && latest != 0 {
                    let mut next = Vec::with_capacity(self.history[len - 1].len());
                    let mut valid = true;
                    for layer in &self.history[len - 1] {
                        let shifted = i64::from(*layer) + latest;
                        if shifted < 0 || shifted > i64::from(u32::MAX) {
                            valid = false;
                            break;
                        }
                        next.push(shifted as u32);
                    }
                    if valid {
                        return ResidencyPrediction {
                            layers: normalize(&next),
                            strategy: PredictionStrategy::ConstantDelta,
                            confidence: PredictionConfidence::Medium,
                            reason: "last two selections shift by the same constant".to_string(),
                        };
                    }
                }
            }
        }
        ResidencyPrediction {
            layers: Vec::new(),
            strategy: PredictionStrategy::Unknown,
            confidence: PredictionConfidence::None,
            reason: "no repeat, alternating or constant-delta pattern".to_string(),
        }
    }

    pub fn status(&self) -> PredictionStatus {
        let scoreable = self.predictions;
        PredictionStatus {
            horizon: 1,
            predictions: self.predictions,
            hits: self.hits,
            misses: self.misses,
            unknown: self.unknown,
            hit_rate_permille: (scoreable > 0)
                .then(|| ((self.hits.saturating_mul(1000)) / scoreable) as u32),
            last_predicted: self.pending.as_ref().map(|p| p.layers.clone()),
            last_actual: self.last_actual.clone(),
            last_strategy: self
                .pending
                .as_ref()
                .map(|p| p.strategy)
                .unwrap_or(PredictionStrategy::Unknown),
        }
    }
}

impl Default for TransitionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

fn normalize(selection: &[u32]) -> Vec<u32> {
    let set: BTreeSet<u32> = selection.iter().copied().collect();
    set.into_iter().collect::<Vec<_>>()
}

fn constant_shift(from: &[u32], to: &[u32]) -> Option<i64> {
    if from.len() != to.len() || from.is_empty() {
        return None;
    }
    let shift = i64::from(to[0]) - i64::from(from[0]);
    for (from_layer, to_layer) in from.iter().zip(to.iter()) {
        if i64::from(*to_layer) - i64::from(*from_layer) != shift {
            return None;
        }
    }
    Some(shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_alternating_selections() {
        let mut predictor = TransitionPredictor::new();
        predictor.observe(&[0, 1]);
        predictor.observe(&[1, 2]);
        predictor.observe(&[0, 1]);
        let prediction = predictor.next_prediction().expect("prediction");
        assert_eq!(prediction.layers, vec![1, 2]);
        assert_eq!(prediction.strategy, PredictionStrategy::Alternating);
        predictor.observe(&[1, 2]);
        assert_eq!(predictor.status().hits, 1);
        assert_eq!(predictor.status().misses, 0);
    }

    #[test]
    fn predicts_repeated_selection() {
        let mut predictor = TransitionPredictor::new();
        predictor.observe(&[2, 3, 4]);
        predictor.observe(&[2, 3, 4]);
        let prediction = predictor.next_prediction().expect("prediction");
        assert_eq!(prediction.layers, vec![2, 3, 4]);
        assert_eq!(prediction.strategy, PredictionStrategy::Repeat);
    }

    #[test]
    fn predicts_constant_delta_window() {
        let mut predictor = TransitionPredictor::new();
        predictor.observe(&[0, 1, 2]);
        predictor.observe(&[1, 2, 3]);
        predictor.observe(&[2, 3, 4]);
        let prediction = predictor.next_prediction().expect("prediction");
        assert_eq!(prediction.layers, vec![3, 4, 5]);
        assert_eq!(prediction.strategy, PredictionStrategy::ConstantDelta);
    }

    #[test]
    fn unknown_without_pattern() {
        let mut predictor = TransitionPredictor::new();
        predictor.observe(&[0]);
        let prediction = predictor.status();
        assert_eq!(prediction.last_strategy, PredictionStrategy::Unknown);
        assert_eq!(prediction.horizon, 1);
    }
}
