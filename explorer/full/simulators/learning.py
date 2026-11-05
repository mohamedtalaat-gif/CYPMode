# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IV: machine learning on toy data that can be seen. Overfitting (4.1), the split of a decision tree (4.2), gradient boosting (4.3) and the average of many trees (4.4). The learners are scikit-learn's,
the same library as the project's models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def curve_data(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A smooth truth, sin(2 pi x), with noise: a training set and a large test set."""
    rng = np.random.default_rng(seed)
    x_train, x_test = np.sort(rng.random(n)), np.sort(rng.random(400))
    truth = lambda x: np.sin(2 * np.pi * x)
    return x_train, truth(x_train) + rng.normal(0, noise, n), x_test, truth(x_test) + rng.normal(0, noise, x_test.size)


def overfitting(max_degree: int, n: int, noise: float, seed: int) -> pd.DataFrame:
    """The error of a polynomial of each degree on the data it was fitted to and on new data (4.1)."""
    x_train, y_train, x_test, y_test = curve_data(n, noise, seed)
    rows = []
    for degree in range(max_degree + 1):
        coefficients = np.polyfit(x_train, y_train, degree)
        rows.append({"degree": degree, "training error": float(np.sqrt(np.mean((np.polyval(coefficients, x_train) - y_train) ** 2))), "error on new data": float(np.sqrt(np.mean((np.polyval(coefficients, x_test) - y_test) ** 2)))})
    return pd.DataFrame(rows)


def polynomial_curve(degree: int, n: int, noise: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, y_train, _, _ = curve_data(n, noise, seed)
    grid = np.linspace(0, 1, 200)
    return pd.DataFrame({"x": x_train, "y": y_train}), pd.DataFrame({"x": grid, "fit": np.polyval(np.polyfit(x_train, y_train, degree), grid), "truth": np.sin(2 * np.pi * grid)})


def gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    p = labels.mean()
    return float(2 * p * (1 - p))


def tree_split(separation: float, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """One feature, two classes whose means are ``separation`` apart: the impurity of the two groups that a question 'x below a?' makes, for every a, and the best a (4.2)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    x = rng.normal(separation * labels, 1.0)
    thresholds = np.linspace(x.min(), x.max(), 160)[1:-1]
    impurity = [(gini(labels[x <= a]) * (x <= a).sum() + gini(labels[x > a]) * (x > a).sum()) / n for a in thresholds]
    best = float(thresholds[int(np.argmin(impurity))])
    return pd.DataFrame({"x": x, "class": np.where(labels == 1, "positive", "negative")}), pd.DataFrame({"threshold": thresholds, "impurity after the split": impurity}), best


def boosting(n_trees: int, learning_rate: float, depth: int, noise: float, seed: int) -> dict:
    """Gradient boosting of small trees on the curve of ``curve_data``: the fit after ``n_trees`` trees and the errors at every stage (4.3)."""
    x_train, y_train, x_test, y_test = curve_data(80, noise, seed)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=learning_rate, max_depth=depth, random_state=seed).fit(x_train[:, None], y_train)
    stages = list(model.staged_predict(x_test[:, None]))
    train_stages = list(model.staged_predict(x_train[:, None]))
    grid = np.linspace(0, 1, 200)
    fit = list(model.staged_predict(grid[:, None]))[n_trees - 1]
    errors = pd.DataFrame({"trees": np.arange(1, 201), "training error": [float(np.sqrt(np.mean((p - y_train) ** 2))) for p in train_stages], "error on new data": [float(np.sqrt(np.mean((p - y_test) ** 2))) for p in stages]})
    return {"points": pd.DataFrame({"x": x_train, "y": y_train}), "fit": pd.DataFrame({"x": grid, "fit": fit, "truth": np.sin(2 * np.pi * grid)}), "errors": errors}


def forest(max_trees: int, noise: float, seed: int) -> pd.DataFrame:
    """The error on new data of one deep tree and of forests of 1 to ``max_trees`` trees (4.4): averaging trees that differ lowers the variance."""
    x_train, y_train, x_test, y_test = curve_data(80, noise, seed)
    single = DecisionTreeRegressor(random_state=seed).fit(x_train[:, None], y_train)
    base = float(np.sqrt(np.mean((single.predict(x_test[:, None]) - y_test) ** 2)))
    rows = []
    for k in sorted({1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 150, 200} | {max_trees}):
        if k > max_trees:
            continue
        model = RandomForestRegressor(n_estimators=k, random_state=seed, bootstrap=True).fit(x_train[:, None], y_train)
        rows.append({"trees": k, "error on new data": float(np.sqrt(np.mean((model.predict(x_test[:, None]) - y_test) ** 2))), "one deep tree": base})
    return pd.DataFrame(rows)
