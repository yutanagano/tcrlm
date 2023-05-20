"""
Benchmark models using beta-chain only data.
"""


from src.pipelines import BetaBenchmarkingPipeline


if __name__ == "__main__":
    BetaBenchmarkingPipeline.run_from_clargs()
