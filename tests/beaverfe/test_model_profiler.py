import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from beaverfe.model_profiler import DataProfiler, ModelProfiler


class _SubclassedForest(RandomForestClassifier):
    pass


class TestModelProfilerDetect:
    def test_should_return_tree_for_decision_tree(self):
        assert ModelProfiler.detect(DecisionTreeClassifier()) == "tree"

    def test_should_return_tree_for_random_forest(self):
        assert ModelProfiler.detect(RandomForestClassifier()) == "tree"

    def test_should_return_tree_for_gradient_boosting(self):
        assert ModelProfiler.detect(GradientBoostingClassifier()) == "tree"

    def test_should_return_tree_for_hist_gradient_boosting(self):
        assert ModelProfiler.detect(HistGradientBoostingRegressor()) == "tree"

    def test_should_return_tree_for_third_party_class_name(self):
        class XGBClassifier:
            pass

        assert ModelProfiler.detect(XGBClassifier()) == "tree"

    def test_should_return_linear_for_logistic_regression(self):
        assert ModelProfiler.detect(LogisticRegression()) == "linear"

    def test_should_return_linear_for_ridge(self):
        assert ModelProfiler.detect(Ridge()) == "linear"

    def test_should_return_other_for_unlisted_model(self):
        assert ModelProfiler.detect(GaussianNB()) == "other"

    def test_should_return_distance_based_for_svc(self):
        assert ModelProfiler.detect(SVC()) == "distance_based"

    def test_should_return_distance_based_for_knn(self):
        assert ModelProfiler.detect(KNeighborsClassifier()) == "distance_based"

    def test_should_return_distance_based_for_mlp(self):
        assert ModelProfiler.detect(MLPClassifier()) == "distance_based"

    def test_should_return_tree_for_subclass_of_known_tree_class(self):
        """isinstance-based detection must catch subclasses, unlike an exact
        type(model).__name__ match."""
        assert ModelProfiler.detect(_SubclassedForest()) == "tree"

    def test_should_unwrap_bagging_with_explicit_tree_estimator(self):
        model = BaggingClassifier(estimator=DecisionTreeClassifier())
        assert ModelProfiler.detect(model) == "tree"

    def test_should_unwrap_bagging_with_explicit_linear_estimator(self):
        model = BaggingClassifier(estimator=LogisticRegression())
        assert ModelProfiler.detect(model) == "linear"

    def test_should_unwrap_adaboost_with_explicit_tree_estimator(self):
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier())
        assert ModelProfiler.detect(model) == "tree"

    def test_should_return_other_for_bagging_with_no_explicit_estimator(self):
        """Bagging's implicit default (a decision tree) can't be introspected
        from an unfitted instance's estimator attribute, so this stays
        "other" rather than guessing — same as before this rewrite."""
        assert ModelProfiler.detect(BaggingClassifier()) == "other"


class TestDataProfiler:
    def _make_X(self, n_rows: int, n_cols: int) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            rng.random((n_rows, n_cols)),
            columns=[f"f{i}" for i in range(n_cols)],
        )

    def test_should_detect_small_sample_when_fewer_than_200_rows(self):
        X = self._make_X(50, 5)
        y = np.zeros(50)
        profiler = DataProfiler.from_data(X, y)
        assert profiler.is_small_sample is True

    def test_should_detect_large_sample_when_10k_or_more_rows(self):
        X = self._make_X(10_000, 5)
        y = np.zeros(10_000)
        profiler = DataProfiler.from_data(X, y)
        assert profiler.is_large_sample is True

    def test_should_detect_high_dimensional_when_features_exceed_rows(self):
        X = self._make_X(10, 20)
        y = np.zeros(10)
        profiler = DataProfiler.from_data(X, y)
        assert profiler.is_high_dimensional is True

    def test_should_report_n_classes_for_classification_target(self):
        X = self._make_X(60, 3)
        y = np.array([0, 1, 2] * 20)
        profiler = DataProfiler.from_data(X, y)
        assert profiler.n_classes == 3

    def test_should_report_zero_n_classes_for_continuous_target(self):
        """Targets with >= 20 unique values are treated as regression (n_classes=0)."""
        X = self._make_X(100, 3)
        rng = np.random.default_rng(0)
        # 100 unique float values — well over the 20-class threshold.
        y = rng.uniform(0, 1, 100)
        profiler = DataProfiler.from_data(X, y)
        assert profiler.n_classes == 0

    def test_should_handle_empty_dataframe_zero_rows(self):
        X = pd.DataFrame({"a": [], "b": []})
        y = np.array([], dtype=float)
        profiler = DataProfiler.from_data(X, y)

        assert profiler.n_samples == 0
        assert profiler.n_features == 2
        assert profiler.is_small_sample is True

    def test_should_handle_single_row_dataframe(self):
        X = pd.DataFrame({"a": [1], "b": [2]})
        y = np.array([1])
        profiler = DataProfiler.from_data(X, y)

        assert profiler.n_samples == 1
        assert profiler.is_small_sample is True
        assert profiler.class_imbalance_ratio >= 1.0

    def test_should_accept_y_as_plain_numpy_array(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = np.array([0, 1])
        profiler = DataProfiler.from_data(X, y)

        assert profiler.n_samples == 2
        assert profiler.n_classes == 2

    def test_should_accept_y_as_pandas_series(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        profiler = DataProfiler.from_data(X, y)

        assert profiler.n_samples == 2
        assert profiler.n_classes == 2

    def test_should_handle_y_with_all_identical_values_zero_variance(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = np.array([0, 0, 0])
        profiler = DataProfiler.from_data(X, y)

        assert profiler.n_classes == 1
        assert profiler.class_imbalance_ratio == 1.0

    def test_should_not_be_high_dimensional_at_exact_boundary_n_features_equals_n_samples(
        self,
    ):
        """Per spec, is_high_dimensional = n_features > n_samples (strict >)."""
        X = pd.DataFrame({f"f{i}": range(10) for i in range(10)})
        y = np.zeros(10)
        profiler = DataProfiler.from_data(X, y)

        assert profiler.n_samples == 10
        assert profiler.n_features == 10
        assert profiler.is_high_dimensional is False

    def test_should_be_high_dimensional_when_one_more_feature_than_samples(self):
        X = pd.DataFrame({f"f{i}": range(10) for i in range(11)})
        y = np.zeros(10)
        profiler = DataProfiler.from_data(X, y)

        assert profiler.is_high_dimensional is True

    def test_should_be_small_sample_at_199_rows_but_not_at_200(self):
        X_199 = pd.DataFrame({f"f{i}": range(199) for i in range(5)})
        X_200 = pd.DataFrame({f"f{i}": range(200) for i in range(5)})
        y = np.zeros(200)

        profiler_199 = DataProfiler.from_data(X_199, y[:199])
        profiler_200 = DataProfiler.from_data(X_200, y)

        assert profiler_199.is_small_sample is True
        assert profiler_200.is_small_sample is False

    def test_should_be_large_sample_at_10000_rows_but_not_at_9999(self):
        X_9999 = pd.DataFrame({"a": range(9999)})
        X_10000 = pd.DataFrame({"a": range(10000)})
        y = np.zeros(10000)

        profiler_9999 = DataProfiler.from_data(X_9999, y[:9999])
        profiler_10000 = DataProfiler.from_data(X_10000, y)

        assert profiler_9999.is_large_sample is False
        assert profiler_10000.is_large_sample is True
