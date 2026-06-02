# rsomics-sc-dendrogram

Hierarchical clustering of single-cell `groupby` categories — a Rust port of
scanpy's `sc.tl.dendrogram`.

Given a per-cell representation matrix (PCA components, an embedding, or
expression values) and a per-cell group label, it aggregates cell values to
per-group means, builds a Pearson / Spearman / Kendall correlation matrix
between groups, converts `1 - corr` to a condensed distance, runs SciPy-exact
linkage, and emits the linkage matrix plus the dendrogram leaf ordering.

## Input

A TSV with one row per cell: column 1 is the group label, the remaining columns
are the representation values. A header row is auto-detected (its first value
column does not parse as a number). Groups are ordered as the sorted set of
labels, matching pandas' default `Categorical` order that scanpy relies on.

## Usage

```
rsomics-sc-dendrogram rep.tsv \
    --cor-method pearson \
    --linkage-method complete \
    -o dendro.json \
    --linkage Z.tsv
```

`--cor-method`: `pearson` (default), `spearman`, `kendall`.
`--linkage-method`: `complete` (default), `average`, `single`, `weighted`,
`ward`, `centroid`, `median`.

The JSON output carries `categories`, `linkage` (the `n-1 × 4` SciPy matrix),
`categories_ordered`, `categories_idx_ordered` (the leaf order), and
`correlation_matrix`. `--linkage` additionally writes the bare linkage matrix as
TSV.

## Compatibility

Validated against scanpy 1.11.5 / scipy 1.15.3 across all 21 cor-method ×
linkage-method combinations: linkage structure is exact, heights agree to
≤2.2e-16, the correlation matrix to ≤1.7e-16, and the leaf order is identical.
`tests/compat.rs` runs a committed golden in CI and an optional live oracle diff
(set `RSOMICS_SCANPY_PY`).

## Origin

This crate reimplements scanpy's `sc.tl.dendrogram` and the SciPy/pandas
primitives it composes:

- scanpy `sc.tl.dendrogram` — group-mean aggregation, `1 - corr` distance,
  leaf-order extraction (Wolf, Angerer & Theis, *Genome Biology* 2018,
  doi:10.1186/s13059-017-1382-0; BSD-3-Clause).
- `scipy.cluster.hierarchy.linkage` — nearest-neighbour-chain
  (complete/average/weighted/ward), MST single linkage, and the Müllner generic
  algorithm (centroid/median), ported to match `_hierarchy.pyx` tie-breaking and
  relabelling (scipy 1.15.3; BSD-3-Clause).
- `pandas.DataFrame.corr` — Pearson, average-rank Spearman, Kendall tau-b
  (BSD-3-Clause).

License: MIT OR Apache-2.0.
Upstream credit: scanpy (BSD-3-Clause), SciPy (BSD-3-Clause), pandas
(BSD-3-Clause).
