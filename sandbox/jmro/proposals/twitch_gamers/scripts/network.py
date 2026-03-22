"""
Network analysis and feature engineering using scipy.sparse.

All heavy graph computations use sparse matrix operations (C/BLAS)
instead of pure-Python NetworkX, except k-core which is already O(m).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

from log import log, step_timer


@dataclass
class NetworkResult:
    """Holds network-level statistics computed during feature engineering."""

    n_nodes: int
    n_edges: int
    density: float
    n_components: int
    largest_component: int


def build_adjacency(
    edges_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[sparse.csr_matrix, int]:
    """Build a sparse symmetric adjacency matrix from the edge list.

    Returns (A, N) where N is the matrix dimension.
    """
    N = int(features_df["id"].max()) + 1

    with step_timer(f"Building sparse adjacency matrix ({N:,} x {N:,})"):
        src = edges_df["src_id"].values
        dst = edges_df["target_id"].values
        ones = np.ones(len(src), dtype=np.float64)
        A = sparse.csr_matrix((ones, (src, dst)), shape=(N, N))
        A = A + A.T  # symmetric (undirected)
        A.data = np.minimum(A.data, 1.0)  # binary

    log.info(f"  Nodes: {N:,}  |  Non-zero entries (nnz/2): {A.nnz // 2:,}")
    return A, N


def compute_components(A: sparse.csr_matrix, N: int) -> tuple[int, int]:
    """Return (n_components, largest_component_size)."""
    with step_timer("Connected components (scipy)"):
        n_comp, labels = sparse.csgraph.connected_components(A, directed=False)
    comp_sizes = np.sort(np.bincount(labels))[::-1]
    log.info(f"  Components: {n_comp}  |  Largest: {comp_sizes[0]:,} nodes")
    if n_comp > 1:
        log.info(f"  Top 5 sizes: {comp_sizes[:5].tolist()}")
    return n_comp, int(comp_sizes[0])


def compute_degree(
    A: sparse.csr_matrix,
    N: int,
    features_df: pd.DataFrame,
) -> np.ndarray:
    """Compute node degree from the adjacency matrix. Returns degree_array[0..N-1]."""
    with step_timer("Degree"):
        degree_array = np.asarray(A.sum(axis=1)).ravel()
        node_ids = features_df["id"].values
        features_df["degree"] = degree_array[node_ids].astype(int)
    log.info(
        f"    mean={features_df['degree'].mean():.2f}  "
        f"max={features_df['degree'].max()}  "
        f"median={features_df['degree'].median():.0f}"
    )
    return degree_array


def compute_clustering(
    A: sparse.csr_matrix,
    N: int,
    degree_array: np.ndarray,
    features_df: pd.DataFrame,
    chunk_size: int = 5000,
) -> None:
    """Clustering coefficient via chunked sparse row-slice multiplication.

    Instead of computing the full A^2 (which can blow up memory on dense
    graphs), we process rows in chunks:
        triangles[chunk] = (A[chunk] @ A).multiply(A[chunk]).sum(axis=1)

    This keeps memory bounded to chunk_size rows of A^2 at a time.
    """
    with step_timer("Clustering coefficient (chunked sparse)"):
        triangles_x2 = np.zeros(N)
        n_chunks = (N + chunk_size - 1) // chunk_size

        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            chunk_idx = i // chunk_size + 1
            if chunk_idx % 5 == 1 or chunk_idx == n_chunks:
                log.info(f"    chunk {chunk_idx}/{n_chunks} (rows {i:,}-{j:,})")
            # A[i:j] @ A gives rows i..j of A^2
            # element-wise multiply with A[i:j] keeps only entries where edge exists
            # row sums = 2 * triangles for each node in the chunk
            A_chunk = A[i:j]
            A2_chunk = A_chunk.dot(A)
            triangles_x2[i:j] = np.asarray(
                A_chunk.multiply(A2_chunk).sum(axis=1)
            ).ravel()

        denom = degree_array * (degree_array - 1)
        mask = denom > 0
        cc = np.zeros(N)
        cc[mask] = triangles_x2[mask] / denom[mask]
        node_ids = features_df["id"].values
        features_df["clustering_coeff"] = cc[node_ids]

    log.info(f"    mean={features_df['clustering_coeff'].mean():.4f}")


def compute_pagerank(
    A: sparse.csr_matrix,
    N: int,
    degree_array: np.ndarray,
    features_df: pd.DataFrame,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> None:
    """PageRank via sparse power iteration."""
    with step_timer("PageRank (sparse power iteration)"):
        out_deg = degree_array.copy()
        out_deg[out_deg == 0] = 1
        D_inv = sparse.diags(1.0 / out_deg)
        M = (D_inv @ A).T

        pr = np.ones(N) / N
        dangling = degree_array == 0

        for it in range(max_iter):
            pr_new = alpha * M.dot(pr) + alpha * dangling.dot(pr) / N + (1 - alpha) / N
            diff = np.abs(pr_new - pr).sum()
            pr = pr_new
            if diff < tol:
                log.info(f"    Converged at iteration {it + 1} (diff={diff:.2e})")
                break

        node_ids = features_df["id"].values
        features_df["pagerank"] = pr[node_ids]
        del M, D_inv

    log.info(
        f"    mean={features_df['pagerank'].mean():.8f}  "
        f"max={features_df['pagerank'].max():.8f}"
    )


def compute_core_number(
    edges_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> None:
    """K-core decomposition via NetworkX (already O(m), fast enough)."""
    with step_timer("Core numbers (NetworkX, O(m))"):
        G = nx.from_pandas_edgelist(
            edges_df, source="src_id", target="target_id"
        )
        all_nodes = set(features_df["id"].values)
        G.add_nodes_from(all_nodes - set(G.nodes()))
        core_dict = nx.core_number(G)
        features_df["core_number"] = (
            features_df["id"].map(core_dict).fillna(0).astype(int)
        )
        del G
    log.info(
        f"    mean={features_df['core_number'].mean():.2f}  "
        f"max={features_df['core_number'].max()}"
    )


def run_all(
    edges_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> NetworkResult:
    """Run the full network analysis pipeline. Mutates features_df in place."""
    A, N = build_adjacency(edges_df, features_df)
    density = A.nnz / (N * (N - 1.0))
    log.info(f"  Density: {density:.6f}")

    n_comp, largest = compute_components(A, N)
    degree_array = compute_degree(A, N, features_df)
    compute_clustering(A, N, degree_array, features_df)
    compute_pagerank(A, N, degree_array, features_df)
    compute_core_number(edges_df, features_df)

    return NetworkResult(
        n_nodes=N,
        n_edges=A.nnz // 2,
        density=density,
        n_components=n_comp,
        largest_component=largest,
    )
