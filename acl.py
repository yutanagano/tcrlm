"""
An executable script to conduct simple contrastive learning on TCR models using
unlabelled TCR data.
"""

from src.pipelines import ACLPipeline


if __name__ == "__main__":
    ACLPipeline().run_from_clargs()
