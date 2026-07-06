import dataclasses
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Perceptron,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR
from sklearn.tree import BaseDecisionTree

ModelType = Literal["tree", "linear", "distance_based", "other"]


class ModelProfiler:
    # isinstance-based, so subclasses of any of these are also caught (unlike
    # an exact type(model).__name__ match, which misses subclasses entirely).
    _TREE_ENSEMBLE_CLASSES: tuple[type, ...] = (
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    _LINEAR_CLASSES: tuple[type, ...] = (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        LogisticRegression,
        LinearSVC,
        LinearSVR,
        SGDClassifier,
        SGDRegressor,
        Perceptron,
    )

    # Share linear models' scale-sensitivity (distance/margin-based, or
    # gradient-based on raw feature magnitudes) without being linear models
    # themselves — grouped with "linear" by the RFC as "Linear Models &
    # Distance-based Models" for scaling/outlier-handling purposes.
    _DISTANCE_BASED_CLASSES: tuple[type, ...] = (
        KNeighborsClassifier,
        KNeighborsRegressor,
        SVC,
        SVR,
        NuSVC,
        NuSVR,
        MLPClassifier,
        MLPRegressor,
    )

    # XGBoost/LightGBM/CatBoost aren't project dependencies, so they can't be
    # imported for isinstance checks without risking an ImportError for users
    # who don't have them installed. Falling back to exact class-name
    # matching only for these three is a deliberate, narrower exception —
    # sklearn's own models above all use isinstance.
    _THIRD_PARTY_TREE_CLASS_NAMES = frozenset(
        {
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "CatBoostClassifier",
            "CatBoostRegressor",
        }
    )

    # Meta-estimators that wrap a single inner estimator under one of these
    # attribute names (Bagging, AdaBoost, CalibratedClassifierCV, etc.) are
    # profiled by unwrapping to that inner estimator instead of defaulting to
    # "other", since their own inductive bias follows whatever they wrap.
    _WRAPPED_ESTIMATOR_ATTRS: tuple[str, ...] = ("estimator", "base_estimator")

    @classmethod
    def detect(cls, model: object) -> ModelType:
        if isinstance(model, (BaseDecisionTree, *cls._TREE_ENSEMBLE_CLASSES)):
            return "tree"
        if isinstance(model, cls._LINEAR_CLASSES):
            return "linear"
        if isinstance(model, cls._DISTANCE_BASED_CLASSES):
            return "distance_based"
        if type(model).__name__ in cls._THIRD_PARTY_TREE_CLASS_NAMES:
            return "tree"

        wrapped = cls._unwrap_single_estimator(model)
        if wrapped is not None:
            return cls.detect(wrapped)

        return "other"

    @classmethod
    def _unwrap_single_estimator(cls, model: object) -> object | None:
        for attr in cls._WRAPPED_ESTIMATOR_ATTRS:
            wrapped: object | None = getattr(model, attr, None)
            if wrapped is not None and wrapped != "deprecated":
                return wrapped
        return None


@dataclasses.dataclass(frozen=True)
class DataProfiler:
    """Data-derived profile used to adapt space-generator behaviour.

    All boolean flags are derived from class-level threshold constants so they
    are easy to reference and override in tests.
    """

    n_samples: int
    n_features: int
    n_classes: int
    class_imbalance_ratio: float
    is_high_dimensional: bool
    is_small_sample: bool
    is_large_sample: bool

    # Thresholds — class-level constants so they are easy to reference and
    # override in tests without touching instance fields.
    SMALL_SAMPLE_THRESHOLD: ClassVar[int] = 200
    LARGE_SAMPLE_THRESHOLD: ClassVar[int] = 10_000
    MAX_CLASSES_THRESHOLD: ClassVar[int] = 20

    @classmethod
    def from_data(cls, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "DataProfiler":
        """Construct a :class:`DataProfiler` from feature matrix and target.

        Args:
            X: Feature matrix.
            y: Target variable (Series or array).

        Returns:
            Frozen :class:`DataProfiler` instance; inputs are never mutated.
        """
        n_samples, n_features = X.shape
        y_series = pd.Series(y)
        n_unique = int(y_series.nunique())
        n_classes = n_unique if n_unique < cls.MAX_CLASSES_THRESHOLD else 0

        if n_classes >= 2:
            counts = y_series.value_counts()
            class_imbalance_ratio = float(counts.max() / counts.min())
        else:
            class_imbalance_ratio = 1.0

        return cls(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_imbalance_ratio=class_imbalance_ratio,
            is_high_dimensional=n_features > n_samples,
            is_small_sample=n_samples < cls.SMALL_SAMPLE_THRESHOLD,
            is_large_sample=n_samples >= cls.LARGE_SAMPLE_THRESHOLD,
        )
