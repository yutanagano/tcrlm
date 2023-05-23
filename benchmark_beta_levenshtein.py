"""
Benchmark simple Levenshtein distances.
"""


from src.pipelines import CDR3BLevenshteinBenchmarkingPipeline, CDRBLevenshteinBenchmarkingPipeline


if __name__ == "__main__":
    CDR3BLevenshteinBenchmarkingPipeline.run_from_clargs()
    CDRBLevenshteinBenchmarkingPipeline.run_from_clargs()