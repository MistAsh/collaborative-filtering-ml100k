import numpy as np
import pandas as pd

from cf.base import BaseCF


class BaselineCF(BaseCF):

    def fit(self, rating_matrix: pd.DataFrame, verbose: bool = True):

        self.rating_matrix = rating_matrix.copy()
        self.pred_matrix = self.rating_matrix.copy()
        overall_mean = self.pred_matrix.values.flatten()
        overall_mean = np.nanmean(overall_mean)

        self.pred_matrix = self.pred_matrix.apply(
            lambda row: row.fillna(row.mean()) if not pd.isnull(row.mean())
            else row.fillna(overall_mean),
            axis=1
        )
