import pytest
import source.benchmarking as benchmarking


class TestNegativeLevenshtein:
    def test_name(self):
        algo = benchmarking.NegativeLevenshtein()
        assert algo.name == 'Negative Levenshtein'


    def test_similarity(self):
        algo = benchmarking.NegativeLevenshtein()
        result = algo.similarity_func('CAS', 'CAT')
        expected = -1
        assert result == expected


class TestAtchleyCosineSimilarity:
    def test_name(self):
        algo = benchmarking.AtchleyCs()
        assert algo.name == 'Averaged Atchley Factors + Cosine Distance'

    
    def test_similarity(self):
        algo = benchmarking.AtchleyCs()
        result = algo.similarity_func('CAS', 'CAT')
        expected = -0.04713477936642105
        assert result == expected


class TestPretrainCdrBert:
    def test_name(self):
        algo = benchmarking.PretrainCdr3Bert(test_mode=True)
        assert algo.name == 'CDR3 BERT (Pretrained)'


    def test_similarity(self):
        algo = benchmarking.PretrainCdr3Bert(test_mode=True)
        result = algo.similarity_func('CAS', 'CAT')
        expected = 1.0
        assert result == expected


    def test_no_path_to_model(self):
        with pytest.raises(RuntimeError):
            benchmarking.PretrainCdr3Bert()


    def test_bad_path_to_model(self):
        with pytest.raises(FileNotFoundError):
            benchmarking.PretrainCdr3Bert(path_to_model='foobarbaz')