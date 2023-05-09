"""
An executable script to conduct epitope label-based contrastive learning on TCR
models using labelled TCR data.
"""

from src.pipelines import ECLPipeline

if __name__ == "__main__":
    ECLPipeline().run_from_clargs()
