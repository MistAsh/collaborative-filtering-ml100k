from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseCF(ABC):

    def __init__(
            self,
    ):
        self.pred_matrix = None
        self.rating_matrix = None

    @abstractmethod
    def fit(self, rating_matrix: pd.DataFrame, verbose: bool = False) -> None:
        pass

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if self.pred_matrix is None:
            raise RuntimeError("Model has not been fitted")

        if (
                user_id not in self.pred_matrix.index or
                item_id not in self.pred_matrix.columns
        ):
            return np.nan

        return self.pred_matrix.loc[user_id, item_id]

    def recommend_top_k(self, user_id, k=10):
        if self.pred_matrix is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        if user_id not in self.pred_matrix.index:
            print('empty pred_matrix')
            return []

        user_predictions = self.pred_matrix.loc[user_id].copy()
        if (
                self.rating_matrix is not None and
                user_id in self.rating_matrix.index
        ):
            known_items_ratings = self.rating_matrix.loc[user_id].dropna()
            user_predictions.drop(known_items_ratings.index, inplace=True,
                                  errors='ignore')

        return user_predictions.nlargest(k).index.tolist()
