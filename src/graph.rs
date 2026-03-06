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
