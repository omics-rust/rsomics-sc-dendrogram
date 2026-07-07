//! Differential vs scanpy 1.11.5 / scipy 1.15.3. The committed golden
//! (`tests/golden/golden_expected.json`, captured from the upstream) always runs
//! in CI; a live `sc.tl.dendrogram` diff runs too when a scanpy venv is present
//! (loud-skip otherwise), via `RSOMICS_SCANPY_PY`.

use std::path::{Path, PathBuf};
use std::process::Command;

use rsomics_sc_dendrogram::{CorMethod, Method, aggregate, compute};
use serde_json::Value;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn parse_method(name: &str) -> (CorMethod, Method) {
    let (c, l) = name.split_once('_').unwrap();
    (CorMethod::parse(c).unwrap(), Method::parse(l).unwrap())
}

#[test]
fn matches_committed_golden() {
    let rep = golden_dir().join("golden_rep.tsv");
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir().join("golden_expected.json")).unwrap(),
    )
    .unwrap();

    let gm = aggregate(&rep).unwrap();
    let exp_cats: Vec<String> = expected["categories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(gm.categories, exp_cats, "category order");

    for (name, case) in expected["cases"].as_object().unwrap() {
        let (cor, method) = parse_method(name);
        let d = compute(&gm, cor, method).unwrap();

        let exp_link = case["linkage"].as_array().unwrap();
        assert_eq!(d.linkage.len(), exp_link.len(), "{name}: linkage rows");
        for (row, e) in d.linkage.iter().zip(exp_link) {
            let e = e.as_array().unwrap();
            assert_eq!(row.left as f64, e[0].as_f64().unwrap(), "{name}: left");
            assert_eq!(row.right as f64, e[1].as_f64().unwrap(), "{name}: right");
            assert_eq!(row.size as f64, e[3].as_f64().unwrap(), "{name}: size");
            let dh = (row.height - e[2].as_f64().unwrap()).abs();
            assert!(dh < 1e-9, "{name}: height diff {dh}");
        }

        let exp_leaves: Vec<usize> = case["leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(d.categories_idx_ordered, exp_leaves, "{name}: leaf order");

        let exp_corr = case["corr"].as_array().unwrap();
        for (i, erow) in exp_corr.iter().enumerate() {
            for (j, e) in erow.as_array().unwrap().iter().enumerate() {
                let dc = (d.correlation_matrix[i][j] - e.as_f64().unwrap()).abs();
                assert!(dc < 1e-9, "{name}: corr[{i}][{j}] diff {dc}");
            }
        }
    }
}

// Non-finite correlation distances. scanpy's pipeline (groupby-mean → corr →
// squareform → scipy.linkage) fails loud with
// ValueError('The condensed distance matrix must contain only finite values.')
// whenever a group's mean vector has zero variance (constant across features)
// or the input has a single feature — both make the group's correlations NaN.
// We mirror that ValueError instead of panicking on a corrupt distance matrix.
const SCIPY_NONFINITE_MSG: &str = "The condensed distance matrix must contain only finite values.";

#[test]
fn zero_variance_group_errors_loud() {
    // Group A's per-cell mean vector is [5,5,5] (constant) → NaN corr; scanpy raises.
    let gm = aggregate(&golden_dir().join("zero_variance_group.tsv")).unwrap();
    assert_eq!(gm.categories, ["A", "B", "C"]);
    for cor in [CorMethod::Pearson, CorMethod::Spearman, CorMethod::Kendall] {
        for m in [
            Method::Single,
            Method::Complete,
            Method::Average,
            Method::Weighted,
            Method::Ward,
            Method::Centroid,
            Method::Median,
        ] {
            let err = compute(&gm, cor, m).unwrap_err();
            assert!(
                err.to_string().contains(SCIPY_NONFINITE_MSG),
                "expected scipy non-finite error, got: {err}"
            );
        }
    }
}

#[test]
fn single_feature_errors_loud() {
    // One feature column → correlation of scalars is NaN; scanpy raises.
    let gm = aggregate(&golden_dir().join("single_feature.tsv")).unwrap();
    let err = compute(&gm, CorMethod::Pearson, Method::Complete).unwrap_err();
    assert!(
        err.to_string().contains(SCIPY_NONFINITE_MSG),
        "expected scipy non-finite error, got: {err}"
    );
}

#[test]
fn nonfinite_input_cell_rejected_at_parse() {
    // Rust's f64::parse silently accepts "inf"/"nan"; we reject at read time.
    for fixture in ["nonfinite_cell.tsv", "nan_cell.tsv"] {
        let err = aggregate(&golden_dir().join(fixture)).unwrap_err();
        assert!(
            err.to_string().contains("non-finite value"),
            "{fixture}: expected non-finite parse error, got: {err}"
        );
    }
}

#[test]
fn matches_live_scanpy() {
    let Ok(py) = std::env::var("RSOMICS_SCANPY_PY") else {
        eprintln!("SKIP matches_live_scanpy: set RSOMICS_SCANPY_PY to a scanpy python");
        return;
    };
    let rep = golden_dir().join("golden_rep.tsv");
    let scratch = std::env::temp_dir().join("rsomics_scd_live");
    std::fs::create_dir_all(&scratch).unwrap();
    let oracle = scratch.join("oracle.py");
    std::fs::write(&oracle, LIVE_ORACLE).unwrap();

    let gm = aggregate(&rep).unwrap();
    for cor in ["pearson", "spearman", "kendall"] {
        for link in ["complete", "average", "single", "weighted", "ward"] {
            let out = scratch.join(format!("{cor}_{link}.json"));
            let status = Command::new(&py)
                .arg(&oracle)
                .arg(&rep)
                .arg(cor)
                .arg(link)
                .arg(&out)
                .status()
                .unwrap();
            assert!(status.success(), "oracle failed for {cor}/{link}");

            let exp: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
            let d = compute(
                &gm,
                CorMethod::parse(cor).unwrap(),
                Method::parse(link).unwrap(),
            )
            .unwrap();

            for (row, e) in d.linkage.iter().zip(exp["linkage"].as_array().unwrap()) {
                let e = e.as_array().unwrap();
                assert_eq!(row.left as f64, e[0].as_f64().unwrap());
                assert_eq!(row.right as f64, e[1].as_f64().unwrap());
                assert!((row.height - e[2].as_f64().unwrap()).abs() < 1e-9);
            }
            let leaves: Vec<usize> = exp["leaves"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(d.categories_idx_ordered, leaves, "{cor}/{link} leaves");
        }
    }
}

const LIVE_ORACLE: &str = r#"
import sys, json, numpy as np, pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
tsv, cor, link, out = sys.argv[1:5]
labels=[]; rows=[]
for line in open(tsv):
    p=line.rstrip("\n").split("\t"); labels.append(p[0]); rows.append([float(x) for x in p[1:]])
rep=pd.DataFrame(np.asarray(rows)); rep.index=pd.Categorical(labels)
cats=rep.index.categories
mean=rep.groupby(level=0,observed=True).mean().loc[cats]
corr=mean.T.corr(method=cor).clip(-1,1)
Z=sch.linkage(distance.squareform(1-corr), method=link)
d=sch.dendrogram(Z, labels=list(cats), no_plot=True)
json.dump({"linkage":[[float(v) for v in r] for r in Z], "leaves":[int(i) for i in d["leaves"]]}, open(out,"w"))
"#;
