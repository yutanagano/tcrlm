"""
Benchmark simple Levenshtein distances.
"""


from src.pipelines import CDR3BLevenshteinBenchmarkingPipeline


if __name__ == "__main__":
    CDR3BLevenshteinBenchmarkingPipeline.run_from_clargs()