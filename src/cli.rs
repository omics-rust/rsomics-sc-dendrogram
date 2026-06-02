use std::path::PathBuf;

use clap::Parser;
use rsomics_common::{CommonFlags, Result, RsomicsError, Tool, ToolMeta};
use rsomics_help::{Example, HelpSpec, Origin};

use rsomics_sc_dendrogram::{CorMethod, Method, aggregate, compute};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(name = "rsomics-sc-dendrogram", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// Cells×features TSV: column 1 = group label, columns 2.. = representation values.
    pub input: PathBuf,

    #[arg(short = 'o', long, default_value = "-")]
    output: String,

    /// Correlation method: pearson, spearman, kendall.
    #[arg(long = "cor-method", default_value = "pearson")]
    cor_method: String,

    /// Linkage method: complete, average, single, weighted, ward, centroid, median.
    #[arg(long = "linkage-method", default_value = "complete")]
    linkage_method: String,

    /// Also write the scipy-style linkage matrix (left right height size) here.
    #[arg(long = "linkage")]
    linkage: Option<PathBuf>,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Tool for Cli {
    fn meta() -> ToolMeta {
        META
    }
    fn common(&self) -> &CommonFlags {
        &self.common
    }

    fn execute(self) -> Result<()> {
        let cor = CorMethod::parse(&self.cor_method).ok_or_else(|| {
            RsomicsError::InvalidInput(format!("cor-method: {}", self.cor_method))
        })?;
        let method = Method::parse(&self.linkage_method).ok_or_else(|| {
            RsomicsError::InvalidInput(format!("linkage-method: {}", self.linkage_method))
        })?;

        let gm = aggregate(&self.input)?;
        let dendro = compute(&gm, cor, method);

        if let Some(path) = &self.linkage {
            let mut lf = std::fs::File::create(path).map_err(RsomicsError::Io)?;
            rsomics_sc_dendrogram::write_linkage_tsv(&dendro, &mut lf)?;
        }

        let mut out: Box<dyn std::io::Write> = if self.output == "-" {
            Box::new(std::io::stdout().lock())
        } else {
            Box::new(std::fs::File::create(&self.output).map_err(RsomicsError::Io)?)
        };
        rsomics_sc_dendrogram::write_json(&dendro, &mut out)
    }
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "Hierarchical clustering of single-cell groupby categories — scanpy sc.tl.dendrogram.",
    origin: Some(Origin {
        upstream: "scanpy sc.tl.dendrogram (scipy.cluster.hierarchy + pandas.DataFrame.corr)",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("10.1186/s13059-017-1382-0"),
    }),
    usage_lines: &[
        "<rep.tsv> [--cor-method pearson] [--linkage-method complete] [-o out.json] [--linkage Z.tsv]",
    ],
    sections: &[],
    examples: &[Example {
        description: "Cluster groupby categories from a representation matrix",
        command: "rsomics-sc-dendrogram rep.tsv --cor-method pearson --linkage-method complete -o dendro.json",
    }],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
