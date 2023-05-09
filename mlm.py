"""
An executable script to conduct masked-language modelling on TCR models.
"""
from src.pipelines import MLMPipeline

if __name__ == "__main__":
    MLMPipeline().run_from_clargs()
