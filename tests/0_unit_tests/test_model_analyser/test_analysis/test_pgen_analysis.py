from src.model_analyser.analysis import PgenAnalysis

from tests.resources.analysis_result_checker import AnalysisResultChecker


def test_run(mock_bg_data, mock_pgens, mock_labelled_data_dict, beta_cdr3_levenshtein_model, tmp_path):
    print(mock_bg_data)
    analysis = PgenAnalysis(
        background_data=mock_bg_data,
        background_pgen=mock_pgens,
        labelled_data=mock_labelled_data_dict,
        tcr_model=beta_cdr3_levenshtein_model,
        working_directory=tmp_path
    )

    result = analysis.run()
    result_checker = AnalysisResultChecker(result)

    assert result_checker.name_is("pgen_vs_local_density")
    assert not result_checker.has_results_dict()
    assert result_checker.has_figures()