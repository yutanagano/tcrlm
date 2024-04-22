from pandas import DataFrame
from libtcrlm import schema
from libtcrlm.schema import TcrPmhcPair
from torch.utils.data import Dataset


class TcrDataset(Dataset):
    def __init__(self, data: DataFrame):
        super().__init__()
        self._tcr_pmhc_series = schema.generate_tcr_pmhc_series(data)

    def __len__(self) -> int:
        return len(self._tcr_pmhc_series)

    def __getitem__(self, index: int) -> TcrPmhcPair:
        return self._tcr_pmhc_series.iloc[index]


class EpitopeBalancedTcrDataset(Dataset):
    def __init__(self, data: DataFrame, num_samples_per_pmhc_per_epoch: int):
        super().__init__()
        tcr_pmhc_series = schema.generate_tcr_pmhc_series(data)
        tcr_pmhc_df = tcr_pmhc_series.to_frame().apply(
            lambda row: (row.item().tcr, row.item().pmhc),
            axis=1,
            result_type="expand"
        )
        tcr_pmhc_df.columns = ["tcr", "pmhc"]

        self._pmhcs = tcr_pmhc_df.pmhc.unique()
        self._tcrs_per_pmhc = {
            pmhc: tcr_pmhc_df.tcr.loc[tcr_pmhc_df.pmhc == pmhc].reset_index(drop=True)
            for pmhc in self._pmhcs
        }
        self._num_samples_per_specificity_per_epoch = num_samples_per_pmhc_per_epoch

    def __len__(self) -> int:
        return len(self._pmhcs) * self._num_samples_per_specificity_per_epoch

    def __getitem__(self, index: int) -> TcrPmhcPair:
        pmhc = self._pmhcs[index % len(self._pmhcs)]
        tcrs_against_pmhc = self._tcrs_per_pmhc[pmhc]
        adjusted_index = index // len(self._pmhcs) % len(tcrs_against_pmhc)
        tcr = tcrs_against_pmhc.iloc[adjusted_index]

        return TcrPmhcPair(tcr, pmhc)
    

class EpitopeBackgroundTcrDataset(Dataset):
    def __init__(self, data: DataFrame, num_samples_per_pmhc_per_epoch: int):
        super().__init__()
        tcr_pmhc_series = schema.generate_tcr_pmhc_series(data)
        tcr_pmhc_df = tcr_pmhc_series.to_frame().apply(
            lambda row: (row.item().tcr, row.item().pmhc),
            axis=1,
            result_type="expand"
        )
        tcr_pmhc_df.columns = ["tcr", "pmhc"]
        labelled_data_mask = tcr_pmhc_df.pmhc.map(lambda pmhc: pmhc.epitope_sequence is not None)
        labelled_data = tcr_pmhc_df.loc[labelled_data_mask]

        self._pmhcs = labelled_data.pmhc.unique()

        self._labelled_data_per_pmhc = {
            pmhc: labelled_data.tcr.loc[labelled_data.pmhc == pmhc].reset_index(drop=True)
            for pmhc in self._pmhcs
        }
        self._background_data = tcr_pmhc_df.tcr.loc[~labelled_data_mask].reset_index(drop=True)

        self._num_samples_per_specificity_per_epoch = num_samples_per_pmhc_per_epoch

    def __len__(self) -> int:
        return len(self._pmhcs) * self._num_samples_per_specificity_per_epoch * 2

    def __getitem__(self, index: int) -> TcrPmhcPair:
        index = index % len(self)
        return_labelled_sequence = index % 2 == 0

        if return_labelled_sequence:
            pmhc = self._pmhcs[(index // 2) % len(self._pmhcs)]
            specific_tcrs = self._labelled_data_per_pmhc[pmhc]
            adjusted_index = ((index // 2) // len(self._pmhcs)) % len(specific_tcrs)
            tcr = specific_tcrs.iloc[adjusted_index]
            return TcrPmhcPair(tcr, pmhc)
        
        adjusted_index = ((index - 1) // 2) % len(self._background_data)
        tcr = self._background_data.iloc[adjusted_index]
        pmhc = schema.make_pmhc_from_components(None, None, None)
        return TcrPmhcPair(tcr, pmhc)