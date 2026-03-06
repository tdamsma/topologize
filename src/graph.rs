//! Graph construction: snap endpoints, extract maximal non-branching chains.
//! Adapted from svg-fixer/centerline-rs/src/graph.rs with added extract_chains().

use std::collections::{HashMap, HashSet};

type Pt = (f64, f64);
type Segment = (Pt, Pt);

pub struct Graph {
    pub nodes: Vec<Pt>,
    pub edges: Vec<(usize, usize)>,
}

/// Snap nearby endpoints into shared nodes using a grid-cell hash map.
pub fn segments_to_graph(segments: &[Segment], snap_tol: f64) -> Graph {
    let mut nodes: Vec<Pt> = Vec::new();
    let mut node_map: HashMap<(i64, i64), usize> = HashMap::new();

    let mut snap =
        |xy: Pt| -> usize {
            let cell = (
                (xy.0 / snap_tol).round() as i64,
                (xy.1 / snap_tol).round() as i64,
            );
            if let Some(&idx) = node_map.get(&cell) {
                return idx;
            }
            for dc in -1i64..=1 {
                for dr in -1i64..=1 {
                    let nc = (cell.0 + dc, cell.1 + dr);
                    if let Some(&idx) = node_map.get(&nc) {
                        node_map.insert(cell, idx);
                        return idx;
                    }
                }
            }
            let idx = nodes.len();
            nodes.push(xy);
            node_map.insert(cell, idx);
            idx
        };

    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();

    for &(start, end) in segments {
        let i = snap(start);
        let j = snap(end);
        if i == j {
            continue;
        }
        let key = (i.min(j), i.max(j));
        if seen.insert(key) {
            edges.push((i, j));
        }
    }

    Graph { nodes, edges }
}

/// A chain with flags indicating which endpoints are terminal (degree 1).
/// Terminal endpoints are safe to move during post-processing; junction
/// endpoints (degree > 2) are shared with other chains and must stay fixed.
pub struct Chain {
    pub pts: Vec<Pt>,
    pub start_terminal: bool,
    pub end_terminal: bool,
    pub start_node: usize,
    pub end_node: usize,
}

/// Iteratively remove terminal edges shorter than `min_tip_len`.
///
/// A "tip" is the chain from a degree-1 node through degree-2 nodes up to
/// the first junction (degree ≥ 3) or dead-end. If its arc length is less
/// than `min_tip_len`, all its edges are removed. This repeats until no
/// more short tips remain (removing a tip can expose a new terminal at the
/// former junction, which may itself be a short tip).
pub fn prune_short_tips(graph: &Graph, min_tip_len: f64) -> Graph {
    let n = graph.nodes.len();
    let mut active: Vec<bool> = vec![true; graph.edges.len()];

    let edge_len = |i: usize| -> f64 {
        let (a, b) = graph.edges[i];
        let (ax, ay) = graph.nodes[a];
        let (bx, by) = graph.nodes[b];
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
    };

    loop {
        // Recompute degree from active edges each iteration.
        let mut deg: Vec<usize> = vec![0; n];
        for (i, &(a, b)) in graph.edges.iter().enumerate() {
            if active[i] {
                deg[a] += 1;
                deg[b] += 1;
            }
        }

        let mut changed = false;

        for start in 0..n {
            if deg[start] != 1 {
                continue;
            }
            // Walk the tip from `start` through degree-2 nodes to junction.
            let mut path_edges: Vec<usize> = Vec::new();
            let mut arc = 0.0;
            let mut prev = usize::MAX; // sentinel: no previous node yet
            let mut cur = start;

            loop {
                // Find the one active neighbor of `cur` that isn't `prev`.
                let next = graph.edges.iter().enumerate().find(|&(i, &(a, b))| {
                    active[i] && ((a == cur && b != prev) || (b == cur && a != prev))
                });
                let (ei, &(ea, eb)) = match next {
                    Some(x) => x,
                    None => break,
                };
                path_edges.push(ei);
                arc += edge_len(ei);
                let nb = if ea == cur { eb } else { ea };
                if deg[nb] != 2 {
                    break; // reached junction or isolated end
                }
                prev = cur;
                cur = nb;
            }

            if !path_edges.is_empty() && arc < min_tip_len {
                for ei in path_edges {
                    active[ei] = false;
                }
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let new_edges = graph
        .edges
        .iter()
        .enumerate()
        .filter(|(i, _)| active[*i])
        .map(|(_, &e)| e)
        .collect();

    Graph { nodes: graph.nodes.clone(), edges: new_edges }
}

/// Extract maximal non-branching chains (polylines) from the graph.
/// Mirrors the Python _extract_chains() logic.
pub fn extract_chains(graph: &Graph) -> Vec<Chain> {
    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); graph.nodes.len()];
    for &(i, j) in &graph.edges {
        adj[i].push(j);
        adj[j].push(i);
    }

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut chains: Vec<Chain> = Vec::new();

    let traverse =
        |start: usize,
         nxt: usize,
         adj: &Vec<Vec<usize>>,
         visited: &mut HashSet<(usize, usize)>,
         nodes: &Vec<Pt>|
         -> Option<Chain> {
            let key = (start.min(nxt), start.max(nxt));
            if visited.contains(&key) {
                return None;
            }
            let mut path = vec![start, nxt];
            visited.insert(key);
            let mut prev = start;
            let mut cur = nxt;
            loop {
                if adj[cur].len() != 2 {
                    break;
                }
                let nb = adj[cur].iter().find(|&&nb| nb != prev).copied();
                let nb = match nb {
                    Some(n) => n,
                    None => break,
                };
                let k = (cur.min(nb), cur.max(nb));
                if visited.contains(&k) {
                    break;
                }
                visited.insert(k);
                path.push(nb);
                prev = cur;
                cur = nb;
            }
            let start_terminal = adj[start].len() == 1;
            let end_terminal = adj[cur].len() == 1;
            Some(Chain {
                pts: path.iter().map(|&i| nodes[i]).collect(),
                start_terminal,
                end_terminal,
                start_node: start,
                end_node: cur,
            })
        };

    // First pass: start from branch/terminal nodes (degree != 2)
    for start in 0..graph.nodes.len() {
        if adj[start].len() != 2 {
            for nxt_idx in 0..adj[start].len() {
                let nxt = adj[start][nxt_idx];
                if let Some(chain) =
                    traverse(start, nxt, &adj, &mut visited, &graph.nodes)
                {
                    chains.push(chain);
                }
            }
        }
    }

    // Second pass: pick up any remaining edges (pure cycles)
    for start in 0..graph.nodes.len() {
        for nxt_idx in 0..adj[start].len() {
            let nxt = adj[start][nxt_idx];
            if let Some(chain) =
                traverse(start, nxt, &adj, &mut visited, &graph.nodes)
            {
                chains.push(chain);
            }
        }
    }

    chains
}
