//! Conservative graph manifest and last-use analysis from bounded trace events.
//!
//! The trace exposes node names, op names, and source names. It does not prove
//! that a weight will not be reused by the next graph/token. Therefore last
//! use here is graph-order analysis only, never an intra-decode eviction
//! guarantee.

use std::collections::{BTreeMap, BTreeSet};

use forge_core::{TraceEvent, TracePhase};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNode {
    pub id: usize,
    pub native_id: u64,
    pub name: Option<String>,
    pub op: Option<String>,
    pub bytes: u64,
    pub dependencies: Vec<usize>,
    pub unresolved_dependencies: Vec<String>,
    pub layer: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastUseStatus {
    /// Graph order and known source edges identify the last consumer, but
    /// reuse by another graph/token is not disproven.
    GraphOrderOnly,
    /// The graph has unresolved/duplicate names or no source edges.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastUse {
    pub node_index: usize,
    pub node_name: Option<String>,
    pub status: LastUseStatus,
    pub completion_proven: bool,
    pub reuse_unknown: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphManifest {
    pub execution_id: Option<u64>,
    pub nodes: Vec<GraphNode>,
    pub tensor_to_nodes: BTreeMap<String, Vec<usize>>,
    pub last_use: BTreeMap<String, LastUse>,
    pub edges_known: bool,
    pub completion_proven: bool,
    /// Always false with the pinned callback: it does not prove next-graph
    /// reuse is impossible.
    pub intra_decode_safe: bool,
}

impl GraphManifest {
    pub fn from_trace(events: &[TraceEvent]) -> Self {
        let requested: Vec<&TraceEvent> = events
            .iter()
            .filter(|event| event.phase == TracePhase::Requested)
            .collect();
        let completed_ids: BTreeSet<u64> = events
            .iter()
            .filter(|event| event.phase == TracePhase::CompletedSynchronized)
            .map(|event| event.node_id)
            .collect();

        let mut id_to_node = BTreeMap::<u64, usize>::new();
        let mut edges_known = true;
        for (id, event) in requested.iter().enumerate() {
            if id_to_node.insert(event.node_id, id).is_some() {
                edges_known = false;
            }
        }

        let mut nodes = Vec::with_capacity(requested.len());
        let mut tensor_to_nodes = BTreeMap::<String, Vec<usize>>::new();
        for (id, event) in requested.iter().enumerate() {
            let mut dependencies = Vec::new();
            let mut unresolved = Vec::new();
            for (name, dependency_id) in event.dependencies.iter().zip(&event.dependency_ids) {
                if let Some(dep_id) = id_to_node.get(dependency_id) {
                    dependencies.push(*dep_id);
                } else {
                    unresolved.push(name.clone());
                }
                tensor_to_nodes.entry(name.clone()).or_default().push(id);
            }
            nodes.push(GraphNode {
                id,
                native_id: event.node_id,
                name: event.node_name.clone(),
                op: event.op_name.clone(),
                bytes: event.bytes,
                dependencies,
                unresolved_dependencies: unresolved,
                layer: event.node_name.as_deref().and_then(layer_from_name),
            });
        }

        let mut last_use = BTreeMap::new();
        for (tensor, consumers) in &tensor_to_nodes {
            if let Some(&node_index) = consumers.iter().max() {
                let node = &nodes[node_index];
                let completion_proven = completed_ids.contains(&node.native_id);
                let status = if edges_known && completion_proven {
                    LastUseStatus::GraphOrderOnly
                } else {
                    LastUseStatus::Unknown
                };
                last_use.insert(
                    tensor.clone(),
                    LastUse {
                        node_index,
                        node_name: node.name.clone(),
                        status,
                        completion_proven,
                        reuse_unknown: true,
                    },
                );
            }
        }

        let completion_proven = last_use.values().all(|use_| use_.completion_proven);
        Self {
            execution_id: requested.first().map(|event| event.execution_id),
            nodes,
            tensor_to_nodes,
            last_use,
            edges_known,
            completion_proven,
            intra_decode_safe: false,
        }
    }

    pub fn summary(&self) -> GraphManifestSummary {
        GraphManifestSummary {
            nodes: self.nodes.len(),
            tensors: self.tensor_to_nodes.len(),
            edges_known: self.edges_known,
            completion_proven: self.completion_proven,
            intra_decode_safe: self.intra_decode_safe,
            graph_order_last_use: self
                .last_use
                .values()
                .filter(|use_| use_.status == LastUseStatus::GraphOrderOnly)
                .count(),
            unknown_last_use: self
                .last_use
                .values()
                .filter(|use_| use_.status == LastUseStatus::Unknown)
                .count(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphManifestSummary {
    pub nodes: usize,
    pub tensors: usize,
    pub edges_known: bool,
    pub completion_proven: bool,
    pub intra_decode_safe: bool,
    pub graph_order_last_use: usize,
    pub unknown_last_use: usize,
}

fn layer_from_name(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (number, suffix) = rest.split_once('.')?;
    if suffix.is_empty() {
        return None;
    }
    number.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::TraceCompletion;

    fn node_id(name: &str) -> u64 {
        name.bytes().fold(0u64, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(byte as u64)
        })
    }

    fn event(name: &str, deps: &[&str], phase: TracePhase) -> TraceEvent {
        TraceEvent {
            execution_id: 1,
            sequence: 0,
            node_id: node_id(name),
            node_name: Some(name.to_string()),
            op_name: Some("MUL".to_string()),
            bytes: 4096,
            dependencies: deps.iter().map(|value| value.to_string()).collect(),
            dependency_ids: deps.iter().map(|value| node_id(value)).collect(),
            phase,
            completion: if phase == TracePhase::CompletedSynchronized {
                TraceCompletion::BackendSynchronized
            } else {
                TraceCompletion::NotCompleted
            },
        }
    }

    #[test]
    fn builds_edges_and_graph_order_last_use() {
        let events = vec![
            event("blk.0.attn", &["blk.0.weight"], TracePhase::Requested),
            event("blk.0.mlp", &["blk.0.weight"], TracePhase::Requested),
            event("blk.0.mlp", &[], TracePhase::CompletedSynchronized),
            event("blk.0.attn", &[], TracePhase::CompletedSynchronized),
        ];
        let manifest = GraphManifest::from_trace(&events);
        assert!(manifest.edges_known);
        assert_eq!(manifest.nodes.len(), 2);
        let last = manifest.last_use.get("blk.0.weight").expect("last use");
        assert_eq!(last.node_index, 1);
        assert_eq!(last.status, LastUseStatus::GraphOrderOnly);
        assert!(last.completion_proven);
        assert!(last.reuse_unknown);
        assert!(!manifest.intra_decode_safe);
    }

    #[test]
    fn duplicate_or_unresolved_names_are_unknown() {
        let events = vec![
            event("same", &["missing"], TracePhase::Requested),
            event("same", &[], TracePhase::Requested),
        ];
        let manifest = GraphManifest::from_trace(&events);
        assert!(!manifest.edges_known);
        assert!(manifest
            .last_use
            .values()
            .all(|use_| use_.status == LastUseStatus::Unknown));
    }

    #[test]
    fn empty_trace_is_empty_manifest() {
        let manifest = GraphManifest::from_trace(&[]);
        assert!(manifest.nodes.is_empty());
        assert!(!manifest.intra_decode_safe);
    }
}
