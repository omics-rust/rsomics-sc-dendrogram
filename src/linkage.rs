//! scipy.cluster.hierarchy.linkage, ported bit-for-bit (scipy 1.15.3 `_hierarchy.pyx`).
//! single → MST; complete/average/weighted/ward → nearest-neighbour chain;
//! centroid/median → Müllner generic algorithm with a binary heap. Tie-breaks,
//! the `x<y` merge convention, the mergesort-by-height relabel, and the
//! Lance-Williams updates all follow scipy so the linkage matrix matches to ~1e-12.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Single,
    Complete,
    Average,
    Centroid,
    Median,
    Ward,
    Weighted,
}

impl Method {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "single" => Method::Single,
            "complete" => Method::Complete,
            "average" => Method::Average,
            "centroid" => Method::Centroid,
            "median" => Method::Median,
            "ward" => Method::Ward,
            "weighted" => Method::Weighted,
            _ => return None,
        })
    }
}

/// A linkage-matrix row: clusters `left` and `right` merge at `height`, forming
/// a cluster of `size` leaves. After relabelling, `left < right`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub left: usize,
    pub right: usize,
    pub height: f64,
    pub size: usize,
}

#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    n * i - (i * (i + 1)) / 2 + (j - i - 1)
}

#[inline]
fn new_dist(m: Method, d_xi: f64, d_yi: f64, d_xy: f64, sx: usize, sy: usize, si: usize) -> f64 {
    let (sxf, syf, sif) = (sx as f64, sy as f64, si as f64);
    match m {
        Method::Single => d_xi.min(d_yi),
        Method::Complete => d_xi.max(d_yi),
        Method::Average => (sxf * d_xi + syf * d_yi) / (sxf + syf),
        Method::Weighted => 0.5 * (d_xi + d_yi),
        Method::Centroid => ((((sxf * d_xi * d_xi) + (syf * d_yi * d_yi))
            - (sxf * syf * d_xy * d_xy) / (sxf + syf))
            / (sxf + syf))
            .sqrt(),
        Method::Median => (0.5 * (d_xi * d_xi + d_yi * d_yi) - 0.25 * d_xy * d_xy).sqrt(),
        Method::Ward => {
            let t = 1.0 / (sxf + syf + sif);
            ((sif + sxf) * t * d_xi * d_xi + (sif + syf) * t * d_yi * d_yi - sif * t * d_xy * d_xy)
                .sqrt()
        }
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next_label: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..2 * n - 1).collect(),
            size: vec![1; 2 * n - 1],
            next_label: n,
        }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut p = x;
        while self.parent[p] != root {
            let nxt = self.parent[p];
            self.parent[p] = root;
            p = nxt;
        }
        root
    }
    fn merge(&mut self, x: usize, y: usize) -> usize {
        self.parent[x] = self.next_label;
        self.parent[y] = self.next_label;
        let s = self.size[x] + self.size[y];
        self.size[self.next_label] = s;
        self.next_label += 1;
        s
    }
}

/// argsort by height with mergesort (stable), then union-find relabel — scipy's
/// `np.argsort(kind='mergesort')` + `label()`.
fn sort_and_label(mut raw: Vec<(usize, usize, f64)>, n: usize) -> Vec<Node> {
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by(|&a, &b| raw[a].2.total_cmp(&raw[b].2));
    let sorted: Vec<(usize, usize, f64)> = order.into_iter().map(|i| raw[i]).collect();
    raw = sorted;

    let mut uf = UnionFind::new(n);
    raw.into_iter()
        .map(|(x, y, height)| {
            let xr = uf.find(x);
            let yr = uf.find(y);
            let (left, right) = if xr < yr { (xr, yr) } else { (yr, xr) };
            let size = uf.merge(xr, yr);
            Node {
                left,
                right,
                height,
                size,
            }
        })
        .collect()
}

fn nn_chain(condensed: &[f64], n: usize, m: Method) -> Vec<(usize, usize, f64)> {
    let mut d = condensed.to_vec();
    let mut size = vec![1usize; n];
    let mut chain = vec![0usize; n];
    let mut chain_len = 0usize;
    let mut out = Vec::with_capacity(n - 1);

    for _ in 0..n - 1 {
        if chain_len == 0 {
            chain_len = 1;
            chain[0] = (0..n).find(|&i| size[i] > 0).unwrap();
        }

        let mut x;
        let mut y = 0usize;
        let mut current_min;
        loop {
            x = chain[chain_len - 1];
            if chain_len > 1 {
                y = chain[chain_len - 2];
                current_min = d[condensed_index(n, x, y)];
            } else {
                current_min = f64::INFINITY;
            }
            for i in 0..n {
                if size[i] == 0 || x == i {
                    continue;
                }
                let dist = d[condensed_index(n, x, i)];
                if dist < current_min {
                    current_min = dist;
                    y = i;
                }
            }
            if chain_len > 1 && y == chain[chain_len - 2] {
                break;
            }
            chain[chain_len] = y;
            chain_len += 1;
        }

        chain_len -= 2;
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        let (nx, ny) = (size[x], size[y]);
        out.push((x, y, current_min));
        size[x] = 0;
        size[y] = nx + ny;

        for i in 0..n {
            let ni = size[i];
            if ni == 0 || i == y {
                continue;
            }
            let nd = new_dist(
                m,
                d[condensed_index(n, i, x)],
                d[condensed_index(n, i, y)],
                current_min,
                nx,
                ny,
                ni,
            );
            d[condensed_index(n, i, y)] = nd;
        }
    }
    out
}

fn mst_single(condensed: &[f64], n: usize) -> Vec<(usize, usize, f64)> {
    let mut merged = vec![false; n];
    let mut d = vec![f64::INFINITY; n];
    let mut out = Vec::with_capacity(n - 1);
    let mut x = 0usize;

    for _ in 0..n - 1 {
        let mut current_min = f64::INFINITY;
        let mut y = 0usize;
        merged[x] = true;
        for i in 0..n {
            if merged[i] {
                continue;
            }
            let dist = condensed[condensed_index(n, x, i)];
            if d[i] > dist {
                d[i] = dist;
            }
            if d[i] < current_min {
                y = i;
                current_min = d[i];
            }
        }
        out.push((x, y, current_min));
        x = y;
    }
    out
}

// Müllner generic algorithm with the same binary heap as scipy's _structures.pxi.
struct Heap {
    index_by_key: Vec<usize>,
    key_by_index: Vec<usize>,
    values: Vec<f64>,
    size: usize,
}

impl Heap {
    fn new(values: Vec<f64>) -> Self {
        let size = values.len();
        let mut h = Heap {
            index_by_key: (0..size).collect(),
            key_by_index: (0..size).collect(),
            values,
            size,
        };
        for i in (0..size / 2).rev() {
            h.sift_down(i);
        }
        h
    }
    fn get_min(&self) -> (usize, f64) {
        (self.key_by_index[0], self.values[0])
    }
    fn remove_min(&mut self) {
        self.swap(0, self.size - 1);
        self.size -= 1;
        self.sift_down(0);
    }
    fn change_value(&mut self, key: usize, value: f64) {
        let index = self.index_by_key[key];
        let old = self.values[index];
        self.values[index] = value;
        if value < old {
            self.sift_up(index);
        } else {
            self.sift_down(index);
        }
    }
    fn sift_up(&mut self, mut index: usize) {
        let mut parent = index.wrapping_sub(1) >> 1;
        while index > 0 && self.values[parent] > self.values[index] {
            self.swap(index, parent);
            index = parent;
            parent = index.wrapping_sub(1) >> 1;
        }
    }
    fn sift_down(&mut self, mut index: usize) {
        let mut child = (index << 1) + 1;
        while child < self.size {
            if child + 1 < self.size && self.values[child + 1] < self.values[child] {
                child += 1;
            }
            if self.values[index] > self.values[child] {
                self.swap(index, child);
                index = child;
                child = (index << 1) + 1;
            } else {
                break;
            }
        }
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.values.swap(i, j);
        let (ki, kj) = (self.key_by_index[i], self.key_by_index[j]);
        self.key_by_index[i] = kj;
        self.key_by_index[j] = ki;
        self.index_by_key[ki] = j;
        self.index_by_key[kj] = i;
    }
}

fn find_min_dist(n: usize, d: &[f64], size: &[usize], x: usize) -> (i64, f64) {
    let mut current_min = f64::INFINITY;
    let mut y: i64 = -1;
    for i in x + 1..n {
        if size[i] == 0 {
            continue;
        }
        let dist = d[condensed_index(n, x, i)];
        if dist < current_min {
            current_min = dist;
            y = i as i64;
        }
    }
    (y, current_min)
}

fn generic(condensed: &[f64], n: usize, m: Method) -> Vec<(usize, usize, f64, usize)> {
    let mut d = condensed.to_vec();
    let mut size = vec![1usize; n];
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut neighbor = vec![0i64; n - 1];
    let mut min_dist = vec![0f64; n - 1];
    let mut z: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n - 1);

    for x in 0..n - 1 {
        let (nb, dist) = find_min_dist(n, &d, &size, x);
        neighbor[x] = nb;
        min_dist[x] = dist;
    }
    let mut heap = Heap::new(min_dist.clone());

    for k in 0..n - 1 {
        let mut x;
        let mut y;
        let mut dist;
        loop {
            let (gx, gd) = heap.get_min();
            x = gx;
            dist = gd;
            y = neighbor[x] as usize;
            if dist == d[condensed_index(n, x, y)] {
                break;
            }
            let (ny2, nd2) = find_min_dist(n, &d, &size, x);
            neighbor[x] = ny2;
            min_dist[x] = nd2;
            heap.change_value(x, nd2);
        }
        heap.remove_min();

        let mut id_x = cluster_id[x];
        let mut id_y = cluster_id[y];
        let (nx, ny) = (size[x], size[y]);
        if id_x > id_y {
            std::mem::swap(&mut id_x, &mut id_y);
        }
        z.push((id_x, id_y, dist, nx + ny));

        size[x] = 0;
        size[y] = nx + ny;
        cluster_id[y] = n + k;

        for zc in 0..n {
            let nz = size[zc];
            if nz == 0 || zc == y {
                continue;
            }
            let nd = new_dist(
                m,
                d[condensed_index(n, zc, x)],
                d[condensed_index(n, zc, y)],
                dist,
                nx,
                ny,
                nz,
            );
            d[condensed_index(n, zc, y)] = nd;
        }

        for zc in 0..x {
            if size[zc] > 0 && neighbor[zc] == x as i64 {
                neighbor[zc] = y as i64;
            }
        }
        for zc in 0..y {
            if size[zc] == 0 {
                continue;
            }
            let dd = d[condensed_index(n, zc, y)];
            if dd < min_dist[zc] {
                neighbor[zc] = y as i64;
                min_dist[zc] = dd;
                heap.change_value(zc, dd);
            }
        }
        if y < n - 1 {
            let (zk, zd) = find_min_dist(n, &d, &size, y);
            if zk != -1 {
                neighbor[y] = zk;
                min_dist[y] = zd;
                heap.change_value(y, zd);
            }
        }
    }
    z
}

/// Linkage matrix for the condensed distance vector `condensed` of `n` observations.
#[must_use]
pub fn linkage(condensed: &[f64], n: usize, m: Method) -> Vec<Node> {
    match m {
        Method::Single => sort_and_label(mst_single(condensed, n), n),
        Method::Complete | Method::Average | Method::Weighted | Method::Ward => {
            sort_and_label(nn_chain(condensed, n, m), n)
        }
        // generic already emits the linkage matrix in agglomeration order with
        // final ids + sizes (no relabel), like scipy's fast_linkage.
        Method::Centroid | Method::Median => generic(condensed, n, m)
            .into_iter()
            .map(|(l, r, h, s)| Node {
                left: l,
                right: r,
                height: h,
                size: s,
            })
            .collect(),
    }
}

/// Pre-order leaf list: scipy `dendrogram(no_plot=True)["leaves"]` with default
/// sort — left child (`Z[i,0]`) before right (`Z[i,1]`), root first.
#[must_use]
pub fn leaf_order(z: &[Node], n: usize) -> Vec<usize> {
    let mut visited = vec![false; 2 * n - 1];
    let mut out = Vec::with_capacity(n);
    let mut stack = vec![2 * n - 2];

    while let Some(&node) = stack.last() {
        let root = node - n;
        let lc = z[root].left;
        if !visited[lc] {
            visited[lc] = true;
            if lc >= n {
                stack.push(lc);
                continue;
            } else {
                out.push(lc);
            }
        }
        let rc = z[root].right;
        if !visited[rc] {
            visited[rc] = true;
            if rc >= n {
                stack.push(rc);
                continue;
            } else {
                out.push(rc);
            }
        }
        stack.pop();
    }
    out
}
