from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from cf.base import BaseCF

class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(
            self,
            cf: BaseCF,
            test_data: pd.DataFrame,
            verbose: bool = True,
            **kwargs
    ) -> float:
        pass


class RatingEvaluator(BaseEvaluator):

    def __init__(self, metric: Callable):
        self.metric = metric

    def evaluate(
            self,
            cf: BaseCF,
            test_data: pd.DataFrame,
            verbose: bool = True,
            **kwargs
    ) -> float:

        pred_rating = []
        true_rating = test_data['rating']

        for row in tqdm(test_data.itertuples(),
                        desc='Evaluating',
                        disable=not verbose):
            user_id = row.user_id
            item_id = row.item_id
            predicted_rating = cf.predict_rating(user_id, item_id)
            pred_rating.append(predicted_rating)

        not_nan_mask = ~np.isnan(pred_rating)
        if verbose:
            print('coverage: {percent:.2f}%'.format(
                percent=float(np.sum(not_nan_mask) / len(true_rating) * 100)
            ))
        pred_rating = np.array(pred_rating)

        return self.metric(
            pred_rating[not_nan_mask],
            true_rating[not_nan_mask]
        )



class TopRecommenderEvaluator(BaseEvaluator):
    def __init__(self, good_rating_threshold=4):
        self.good_rating_threshold = good_rating_threshold

    def evaluate(
            self, cf: BaseCF,
            test_data: pd.DataFrame,
            verbose: bool = True,
            **kwargs
    ) -> float:
        mode = kwargs.get('mode', 'precision')
        top_k = kwargs.get('top_k', 10)

        test_truth = defaultdict(set)
        for row in tqdm(test_data.itertuples(),
                        desc="Preparing ground truth",
                        disable=not verbose):
            if row.rating >= self.good_rating_threshold:
                test_truth[int(row.user_id)].add(int(row.item_id))

        if mode not in ('precision', 'recall'):
            raise ValueError("Unsupported mode. Use 'precision' or 'recall'.")

        return self._compute_metric(cf, test_truth, top_k, mode, verbose)

    def _compute_metric(
            self,
            cf: BaseCF,
            test_truth: dict[int, set],
            top_k: int,
            mode: str,
            verbose: bool
    ) -> float:
        total_score = 0.0
        valid_users = 0

        for user_id in tqdm(test_truth.keys(),
                            desc=f"Evaluating {mode}",
                            disable=not verbose):
            predicted = set(cf.recommend_top_k(user_id, top_k))
            actual = test_truth[user_id]

            if not predicted or (mode == 'recall' and not actual):
                continue

            true_positives = len(predicted & actual)

            if mode == 'precision':
                score = true_positives / len(predicted)
            else:  # recall
                score = true_positives / len(actual)

            total_score += score
            valid_users += 1

        if valid_users == 0:
            return 0.0

        return total_score / valid_users