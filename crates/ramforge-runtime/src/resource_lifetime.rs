//! Explicit resource-lifetime state machine.
//!
//! Graph-node completion is deliberately absent from the path to
//! `ReleaseEligible`. Only an observed operation boundary after native
//! synchronization can promote a file-backed model resource to physical
//! release eligibility. `FinalUseProven` is reserved for a future native
//! contract that actually proves no future graph can reuse the resource.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourceLifetimeState {
    Discovered,
    Owned,
    Referenced,
    ExecutionSubmitted,
    ExecutionCompleted,
    FinalUseProven,
    ReleaseEligible,
    ReleaseRequested,
    ReleaseAccepted,
    Reconciled,
    SharedOrReused,
    Invalidated,
    ReleaseFailed,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifetimeEvent {
    ModelBound,
    ContextReferenced,
    ExecutionSubmitted,
    ExecutionCompleted,
    OperationBoundarySynchronized,
    ModelDestroyed,
    ReleaseSelected,
    ReleaseSubmitted,
    NativeAccepted,
    ReconciliationComplete,
    SharedAccessObserved,
    GenerationInvalidated,
    NativeFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidLifetimeTransition {
    pub from: ResourceLifetimeState,
    pub event: LifetimeEvent,
}

impl fmt::Display for InvalidLifetimeTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid resource lifetime transition: {:?} on {:?}",
            self.from, self.event
        )
    }
}

impl std::error::Error for InvalidLifetimeTransition {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLifetime {
    state: ResourceLifetimeState,
    transitions: u64,
    evidence: String,
}

impl Default for ResourceLifetime {
    fn default() -> Self {
        Self {
            state: ResourceLifetimeState::Discovered,
            transitions: 0,
            evidence: "inventory discovery".to_string(),
        }
    }
}

impl ResourceLifetime {
    pub fn state(&self) -> ResourceLifetimeState {
        self.state
    }

    pub fn transitions(&self) -> u64 {
        self.transitions
    }

    pub fn evidence(&self) -> &str {
        &self.evidence
    }

    pub fn apply(
        &mut self,
        event: LifetimeEvent,
        evidence: impl Into<String>,
    ) -> Result<ResourceLifetimeState, InvalidLifetimeTransition> {
        let next = match (self.state, event) {
            (ResourceLifetimeState::Discovered, LifetimeEvent::ModelBound) => {
                ResourceLifetimeState::Owned
            }
            (ResourceLifetimeState::Owned, LifetimeEvent::ContextReferenced) => {
                ResourceLifetimeState::Referenced
            }
            (ResourceLifetimeState::Owned, LifetimeEvent::OperationBoundarySynchronized) => {
                ResourceLifetimeState::ReleaseEligible
            }
            (ResourceLifetimeState::Referenced, LifetimeEvent::ExecutionSubmitted) => {
                ResourceLifetimeState::ExecutionSubmitted
            }
            (ResourceLifetimeState::ExecutionSubmitted, LifetimeEvent::ExecutionCompleted) => {
                ResourceLifetimeState::ExecutionCompleted
            }
            (
                ResourceLifetimeState::ExecutionCompleted,
                LifetimeEvent::OperationBoundarySynchronized,
            ) => ResourceLifetimeState::ReleaseEligible,
            (ResourceLifetimeState::ExecutionCompleted, LifetimeEvent::SharedAccessObserved) => {
                ResourceLifetimeState::SharedOrReused
            }
            (ResourceLifetimeState::ReleaseEligible, LifetimeEvent::ReleaseSelected) => {
                ResourceLifetimeState::ReleaseRequested
            }
            (ResourceLifetimeState::ReleaseRequested, LifetimeEvent::ReleaseSubmitted) => {
                ResourceLifetimeState::ReleaseRequested
            }
            (ResourceLifetimeState::ReleaseRequested, LifetimeEvent::NativeAccepted) => {
                ResourceLifetimeState::ReleaseAccepted
            }
            (ResourceLifetimeState::ReleaseRequested, LifetimeEvent::NativeFailure) => {
                ResourceLifetimeState::ReleaseFailed
            }
            (ResourceLifetimeState::ReleaseAccepted, LifetimeEvent::ReconciliationComplete) => {
                ResourceLifetimeState::Reconciled
            }
            (_, LifetimeEvent::GenerationInvalidated) => ResourceLifetimeState::Invalidated,
            // A native contract may eventually prove final use; the current
            // pinned callback does not. Keep the transition explicit without
            // making it reachable from graph completion alone.
            (ResourceLifetimeState::ExecutionCompleted, LifetimeEvent::ModelDestroyed) => {
                ResourceLifetimeState::FinalUseProven
            }
            _ => {
                return Err(InvalidLifetimeTransition {
                    from: self.state,
                    event,
                })
            }
        };
        self.state = next;
        self.transitions = self.transitions.saturating_add(1);
        self.evidence = evidence.into();
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_completion_alone_cannot_release() {
        let mut lifetime = ResourceLifetime::default();
        lifetime
            .apply(LifetimeEvent::ModelBound, "model")
            .expect("bound");
        lifetime
            .apply(LifetimeEvent::ContextReferenced, "context")
            .expect("referenced");
        lifetime
            .apply(LifetimeEvent::ExecutionSubmitted, "decode")
            .expect("submitted");
        lifetime
            .apply(LifetimeEvent::ExecutionCompleted, "callback")
            .expect("completed");
        assert!(lifetime
            .apply(LifetimeEvent::ReleaseSelected, "bad")
            .is_err());
        assert_eq!(lifetime.state(), ResourceLifetimeState::ExecutionCompleted);
        lifetime
            .apply(
                LifetimeEvent::OperationBoundarySynchronized,
                "context dropped and model observed quiescent",
            )
            .expect("operation boundary");
        assert_eq!(lifetime.state(), ResourceLifetimeState::ReleaseEligible);
    }

    #[test]
    fn release_failure_is_terminal_and_explicit() {
        let mut lifetime = ResourceLifetime::default();
        lifetime.apply(LifetimeEvent::ModelBound, "model").unwrap();
        lifetime
            .apply(LifetimeEvent::OperationBoundarySynchronized, "quiescent")
            .unwrap();
        lifetime
            .apply(LifetimeEvent::ReleaseSelected, "plan")
            .unwrap();
        lifetime
            .apply(LifetimeEvent::NativeFailure, "EBUSY")
            .unwrap();
        assert_eq!(lifetime.state(), ResourceLifetimeState::ReleaseFailed);
        assert!(lifetime
            .apply(LifetimeEvent::NativeAccepted, "late")
            .is_err());
    }

    #[test]
    fn invalidation_is_explicit() {
        let mut lifetime = ResourceLifetime::default();
        lifetime
            .apply(LifetimeEvent::GenerationInvalidated, "reload")
            .unwrap();
        assert_eq!(lifetime.state(), ResourceLifetimeState::Invalidated);
    }
}
