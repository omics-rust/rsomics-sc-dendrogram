use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use rsomics_common::{Result, RsomicsError};
use serde::Serialize;

pub mod corr;
pub mod linkage;

pub use corr::{CorMethod, condensed_distance, corr_matrix};
pub use linkage::{Method, Node, leaf_order, linkage};

/// One row of the scipy linkage matrix.
#[derive(Serialize)]
pub struct LinkageRow {
    pub left: usize,
    pub right: usize,
    pub height: f64,
    pub size: usize,
}

#[derive(Serialize)]
pub struct Dendrogram {
    pub categories: Vec<String>,
    pub linkage: Vec<LinkageRow>,
    pub categories_ordered: Vec<String>,
    pub categories_idx_ordered: Vec<usize>,
    pub correlation_matrix: Vec<Vec<f64>>,
}

/// A cells×features representation with a per-cell group label. Group order is
/// the sorted set of labels — pandas' default Categorical order, which scanpy's
/// `.loc[categories]` reindex relies on.
pub struct GroupMeans {
    pub categories: Vec<String>,
    /// One mean-vector per category, in `categories` order.
    pub means: Vec<Vec<f64>>,
}

/// Read a TSV: column 0 = group label, columns 1.. = numeric features (one row
/// per cell). Aggregates per-group feature means. A leading header row is
/// detected when its first data column fails to parse as a float.
pub fn aggregate(input: &Path) -> Result<GroupMeans> {
    let file = std::fs::File::open(input)
        .map_err(|e| RsomicsError::InvalidInput(format!("{}: {e}", input.display())))?;
    let mut lines = BufReader::new(file).lines();

    let first = lines
        .next()
        .ok_or_else(|| RsomicsError::InvalidInput("empty input".into()))?
        .map_err(RsomicsError::Io)?;
    let n_features = {
        let cells: Vec<&str> = first.split('\t').collect();
        if cells.len() < 2 {
            return Err(RsomicsError::InvalidInput(
                "need a label column plus at least one feature column".into(),
            ));
        }
        cells.len() - 1
    };

    let mut sums: BTreeMap<String, (Vec<f64>, usize)> = BTreeMap::new();
    let mut add = |line: &str, lineno: usize| -> Result<()> {
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() != n_features + 1 {
            return Err(RsomicsError::InvalidInput(format!(
                "line {lineno}: {} columns, expected {}",
                cells.len(),
                n_features + 1
            )));
        }
        let entry = sums
            .entry(cells[0].to_string())
            .or_insert_with(|| (vec![0.0; n_features], 0));
        for (k, cell) in cells[1..].iter().enumerate() {
            entry.0[k] += cell.parse::<f64>().map_err(|e| {
                RsomicsError::InvalidInput(format!("line {lineno} col {}: {e}", k + 1))
            })?;
        }
        entry.1 += 1;
        Ok(())
    };

    let data_is_numeric = first
        .split('\t')
        .nth(1)
        .is_some_and(|s| s.parse::<f64>().is_ok());
    if data_is_numeric {
        add(&first, 1)?;
    }
    for (i, line) in lines.enumerate() {
        let line = line.map_err(RsomicsError::Io)?;
        if line.is_empty() {
            continue;
        }
        add(&line, i + 2)?;
    }

    if sums.len() < 2 {
        return Err(RsomicsError::InvalidInput(format!(
            "need at least 2 groups, found {}",
            sums.len()
        )));
    }

    let mut categories = Vec::with_capacity(sums.len());
    let mut means = Vec::with_capacity(sums.len());
    for (label, (sum, count)) in sums {
        categories.push(label);
        means.push(sum.into_iter().map(|s| s / count as f64).collect());
    }
    Ok(GroupMeans { categories, means })
}

pub fn compute(gm: &GroupMeans, cor: CorMethod, method: Method) -> Dendrogram {
    let g = gm.categories.len();
    let corr = corr_matrix(&gm.means, cor);
    let condensed = condensed_distance(&corr, g);
    let z = linkage(&condensed, g, method);
    let leaves = leaf_order(&z, g);

    let categories_ordered = leaves.iter().map(|&i| gm.categories[i].clone()).collect();
    let correlation_matrix = (0..g).map(|i| corr[i * g..i * g + g].to_vec()).collect();
    let linkage = z
        .iter()
        .map(|n| LinkageRow {
            left: n.left,
            right: n.right,
            height: n.height,
            size: n.size,
        })
        .collect();

    Dendrogram {
        categories: gm.categories.clone(),
        linkage,
        categories_ordered,
        categories_idx_ordered: leaves,
        correlation_matrix,
    }
}

pub fn write_json(d: &Dendrogram, out: &mut dyn Write) -> Result<()> {
    serde_json::to_writer_pretty(&mut *out, d)
        .map_err(|e| RsomicsError::ConfigError(e.to_string()))?;
    writeln!(out).map_err(RsomicsError::Io)
}

/// Tab-separated linkage matrix: `left right height size`, scipy column order,
/// height at full f64 precision.
pub fn write_linkage_tsv(d: &Dendrogram, out: &mut dyn Write) -> Result<()> {
    for r in &d.linkage {
        writeln!(out, "{}\t{}\t{:.17}\t{}", r.left, r.right, r.height, r.size)
            .map_err(RsomicsError::Io)?;
    }
    Ok(())
}
