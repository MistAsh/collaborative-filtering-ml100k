import numpy as np
import pandas as pd
from tqdm import tqdm

from cf.similarity_based.base import SimilarityBasedCF


class UserBasedCF(SimilarityBasedCF):

    def fit(self, rating_matrix: pd.DataFrame, verbose: bool = False) -> None:

        self.rating_matrix = rating_matrix.copy()
        self.pred_matrix = self.rating_matrix.copy()

        self.average_neighbors_per_user = 0
        mean_rating_by_user = self.rating_matrix.mean(axis=1)

        similarity_df = self.similarity_computer.compute(
            self.rating_matrix
        )

        ratings_diff_from_mean = self.rating_matrix.subtract(
            mean_rating_by_user,
            axis=0
        )

        for user_id in tqdm(self.rating_matrix.index,
                            desc="Fitting",
                            disable=not verbose):

            user_mean_r = mean_rating_by_user.loc[user_id]

            items_to_predict_mask = self.pred_matrix.loc[user_id].isna()
            if not items_to_predict_mask.any():
                continue

            item_ids_for_prediction = self.pred_matrix.columns[
                items_to_predict_mask]

            user_similarities = similarity_df.loc[user_id].copy()

            if user_id in user_similarities.index:
                user_similarities.drop(user_id, inplace=True)

            relevant_neighbors_sims = user_similarities[
                user_similarities > self.similarity_threshold
                ]

            self.average_neighbors_per_user += len(relevant_neighbors_sims)

            if relevant_neighbors_sims.empty:
                self.pred_matrix.loc[
                    user_id,
                    item_ids_for_prediction
                ] = user_mean_r
                continue

            neighbor_ids = relevant_neighbors_sims.index
            sim_scores = relevant_neighbors_sims.values

            neighbor_ratings_diff_subset = ratings_diff_from_mean.loc[
                neighbor_ids, item_ids_for_prediction]

            weighted_diffs = neighbor_ratings_diff_subset.multiply(
                sim_scores,
                axis=0
            )

            numerator = weighted_diffs.sum(axis=0)

            did_neighbor_rate_item_mask = self.rating_matrix.loc[
                neighbor_ids, item_ids_for_prediction].notna()

            abs_sim_scores = np.abs(sim_scores)

            denominator_terms = did_neighbor_rate_item_mask.multiply(
                abs_sim_scores,
                axis=0
            )

            denominator = denominator_terms.sum(axis=0)
            prediction_values = user_mean_r + (numerator / denominator)

            prediction_values[denominator == 0] = user_mean_r

            prediction_values.fillna(user_mean_r, inplace=True)

            self.pred_matrix.loc[
                user_id, item_ids_for_prediction] = prediction_values
        self.average_neighbors_per_user = (
                self.average_neighbors_per_user / len(self.rating_matrix.index)
        )
