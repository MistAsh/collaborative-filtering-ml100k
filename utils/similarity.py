from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityComputer(ABC):
    @abstractmethod
    def compute(self, matrix: pd.DataFrame) -> pd.DataFrame:
        pass


class PearsonSimilarity(SimilarityComputer):
    def compute(self, matrix: pd.DataFrame) -> pd.DataFrame:
        return matrix.T.corr(method='pearson')


class CosineSimilarityWithZeros(SimilarityComputer):
    def compute(self, matrix: pd.DataFrame) -> pd.DataFrame:
        similarity_matrix_np = cosine_similarity(matrix.fillna(0))
        return pd.DataFrame(
            similarity_matrix_np,
            index=matrix.index,
            columns=matrix.index
        )


class JaccardMeasureComputer(SimilarityComputer):

    def __init__(self, good_rating_threshold: float = 4):
        self.good_rating_threshold = good_rating_threshold

    def compute(self, matrix: pd.DataFrame) -> pd.DataFrame:
        matrix_np = matrix.to_numpy()
        n_rows, _ = matrix_np.shape
        result_matrix = np.zeros((n_rows, n_rows), dtype=float)
        nan_masks = np.isnan(matrix_np)

        for i in range(n_rows):
            for j in range(i + 1, n_rows):

                union_mask = ~nan_masks[i, :] | ~nan_masks[j, :]
                if not np.any(union_mask):
                    continue

                user_1_fav = (matrix_np[i, union_mask] >=
                              self.good_rating_threshold)
                user_2_fav = (matrix_np[j, union_mask] >=
                              self.good_rating_threshold)

                intersection = np.sum(user_1_fav & user_2_fav)
                union = np.sum(user_1_fav | user_2_fav)

                score = intersection / union if union != 0 else 0.0
                result_matrix[i, j] = result_matrix[j, i] = score

        return pd.DataFrame(
            result_matrix,
            index=matrix.index,
            columns=matrix.index
        )