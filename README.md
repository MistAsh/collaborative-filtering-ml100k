# Collaborative Filtering (CF)

This repository contains an implementation of collaborative filtering algorithms based on similarity measures for recommendation systems.

## Implemented Models

* **Item-Based CF**
* **User-Based CF**
* **Baseline** — simple model using:

  * Mean rating per user
  * Or global mean if user not seen

## Similarity Metrics

The predicted rating is computed as:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} s(u, v) (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |s(u, v)|}
$$

Supported similarity computations:

* **Pearson Similarity**
* **Cosine Similarity** (with zero fill for NaN)
* **Jaccard Measure**

## Installation

This project uses [uv](https://github.com/astral-sh/uv) as the dependency manager.
```bash
uv pip install -r pyproject.toml
```

## Usage

### Fitting a Model

```python
import pandas as pd
from cf import ItemBasedCF
from utils.similarity import JaccardMeasureComputer

# rating_matrix: DataFrame with users as rows, items as columns
rating_matrix = pd.DataFrame()

cf_instance = ItemBasedCF(
    similarity_threshold=-0.33,
    similarity_computer=JaccardMeasureComputer()
)

cf_instance.fit(rating_matrix, verbose=True)
```

### Evaluation

```python
import pandas as pd
from cf import ItemBasedCF
from utils.similarity import JaccardMeasureComputer
from utils.evaluation import RatingEvaluator, TopRecommenderEvaluator
from sklearn.metrics import root_mean_squared_error

rating_matrix = pd.DataFrame()
test_data = pd.DataFrame()

cf_instance = ItemBasedCF(
    similarity_threshold=-0.33,
    similarity_computer=JaccardMeasureComputer()
)
cf_instance.fit(rating_matrix, verbose=True)

# RMSE Evaluation
rmse_evaluator = RatingEvaluator(metric=root_mean_squared_error)
rmse_evaluator.evaluate(cf_instance, test_data, verbose=True)

# Top-K Recommendation Evaluation
top_recommender_evaluator = TopRecommenderEvaluator()
top_recommender_evaluator.evaluate(cf_instance, test_data, top_k=10, mode='precision')
```

Implemented metrics:

* `precision@k`
* `recall@k`

## Metrics on [MovieLens 100k](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

Calculation in [cf_demo.ipynb](notebooks/cf_demo.ipynb)

| # | Algorithm  | Similarity Metric | RMSE   | MAE    |
| - | ---------- | ----------------- | ------ | ------ |
| 0 | Baseline   | N/A               | 1.0431 | 0.8326 |
| 1 | Item-Based | CosineWithZeros   | 0.9500 | 0.7500 |
| 2 | Item-Based | Jaccard           | 0.9500 | 0.7400 |
| 3 | Item-Based | Pearson           | 0.9500 | 0.7500 |
| 4 | User-Based | CosineWithZeros   | 0.9700 | 0.7600 |
| 5 | User-Based | Jaccard           | 0.9700 | 0.7600 |
| 6 | User-Based | Pearson           | 0.9600 | 0.7600 |

---


