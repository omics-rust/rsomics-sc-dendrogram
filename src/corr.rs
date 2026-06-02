//! pandas `DataFrame.corr` reproduced for the three methods scanpy exposes:
//! pearson, spearman (average-rank then pearson), kendall tau-b. Input is the
//! group-mean matrix (one row per group); output is the symmetric group×group
//! correlation, matching pandas to ~1e-12 on no-NaN data.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorMethod {
    Pearson,
    Spearman,
    Kendall,
}

impl CorMethod {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pearson" => CorMethod::Pearson,
            "spearman" => CorMethod::Spearman,
            "kendall" => CorMethod::Kendall,
            _ => return None,
        })
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Average ranks (ties share the mean of the positions they span).
fn rank_average(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + j - 1) as f64) / 2.0 + 1.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

fn kendall_tau_b(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut tie_a = 0i64;
    let mut tie_b = 0i64;
    for i in 0..n {
        for j in i + 1..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            let s = (da * db).partial_cmp(&0.0);
            if da == 0.0 && db == 0.0 {
                tie_a += 1;
                tie_b += 1;
            } else if da == 0.0 {
                tie_a += 1;
            } else if db == 0.0 {
                tie_b += 1;
            } else {
                match s {
                    Some(std::cmp::Ordering::Greater) => concordant += 1,
                    _ => discordant += 1,
                }
            }
        }
    }
    let n0 = (n * (n - 1) / 2) as f64;
    let denom = ((n0 - tie_a as f64) * (n0 - tie_b as f64)).sqrt();
    (concordant - discordant) as f64 / denom
}

/// Group×group correlation matrix (row-major, `g*g`) for the `g`-row group-mean
/// matrix `means` (each row has `f` features).
#[must_use]
pub fn corr_matrix(means: &[Vec<f64>], method: CorMethod) -> Vec<f64> {
    let g = means.len();
    let rows: Vec<Vec<f64>> = match method {
        CorMethod::Spearman => means.iter().map(|r| rank_average(r)).collect(),
        _ => means.to_vec(),
    };
    let mut out = vec![0.0; g * g];
    for i in 0..g {
        out[i * g + i] = 1.0;
        for j in i + 1..g {
            let c = match method {
                CorMethod::Kendall => kendall_tau_b(&rows[i], &rows[j]),
                _ => pearson(&rows[i], &rows[j]),
            }
            .clamp(-1.0, 1.0);
            out[i * g + j] = c;
            out[j * g + i] = c;
        }
    }
    out
}

/// `1 - corr` over the strict upper triangle — scipy `squareform(1 - corr)`.
#[must_use]
pub fn condensed_distance(corr: &[f64], g: usize) -> Vec<f64> {
    let mut d = Vec::with_capacity(g * (g - 1) / 2);
    for i in 0..g {
        for j in i + 1..g {
            d.push(1.0 - corr[i * g + j]);
        }
    }
    d
}
