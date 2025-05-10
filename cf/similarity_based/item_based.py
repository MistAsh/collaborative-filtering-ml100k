import pandas as pd

from cf.similarity_based.user_based import UserBasedCF



class ItemBasedCF(UserBasedCF):

    def fit(self, rating_matrix: pd.DataFrame, verbose: bool = False) -> None:
        super().fit(rating_matrix.T, verbose)
        self.pred_matrix = self.pred_matrix.T
        self.rating_matrix = rating_matrix.copy()