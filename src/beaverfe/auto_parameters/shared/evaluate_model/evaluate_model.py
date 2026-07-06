import numpy as np
import optuna
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from beaverfe.transformations.utils import dtypes


# Custom transformer to drop datetime columns
class DropDatetimeColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_drop_ = X.select_dtypes(
            include=["datetime64", "datetime64[ns]"]
        ).columns
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


# Casts bool/timedelta columns to numeric so they reach the "number" selector
# below instead of being silently dropped by ColumnTransformer's remainder="drop".
class CastAuxiliaryDtypes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.bool_columns_ = dtypes.bool_columns(X)
        self.timedelta_columns_ = dtypes.timedelta_columns(X)
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.bool_columns_:
            X[column] = X[column].astype("int8")
        for column in self.timedelta_columns_:
            X[column] = X[column].dt.total_seconds()
        return X


def build_pipeline(model, transformer=None, skip_safety_net: bool = False) -> Pipeline:
    steps = []

    # Step 1: Optional custom transformer (always the first step when present,
    # so it is fitted per CV fold — never pre-fitted outside the fold).
    if transformer:
        steps.append(("transformer", transformer))

    if skip_safety_net:
        # Caller guarantees the recipe already handles imputation and encoding;
        # omit the redundant DropDatetimeColumns / ColumnTransformer pass.
        steps.append(("model", model))
        return Pipeline(steps)

    # Step 2: Drop datetime columns, cast bool/timedelta columns to numeric
    steps.append(("drop_datetime", DropDatetimeColumns()))
    steps.append(("cast_auxiliary_dtypes", CastAuxiliaryDtypes()))

    # Step 3: Define imputers
    numeric_imputer = SimpleImputer(strategy="constant", fill_value=0)
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")

    # Step 4: Categorical pipeline: impute + encode
    categorical_pipeline = Pipeline(
        [
            ("imputer", categorical_imputer),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Step 5: Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_imputer, make_column_selector(dtype_include=["number"])),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ],
        remainder="drop",
    )

    # Step 6: Assemble full pipeline
    steps.append(("preprocessing", preprocessor))
    steps.append(("model", model))

    return Pipeline(steps)


def evaluate_model(
    x,
    y,
    model,
    scoring,
    cv=5,
    groups=None,
    transformer=None,
    trial=None,
    skip_safety_net: bool = False,
):
    pipe = build_pipeline(model, transformer, skip_safety_net=skip_safety_net)

    if trial is None:
        scores = cross_val_score(
            pipe, x, y, scoring=scoring, cv=cv, groups=groups, n_jobs=-1
        )
        return np.mean(scores)

    # Fold-by-fold so a clearly bad trial can be pruned before every
    # remaining fold runs. This trades away n_jobs=-1 fold parallelism (folds
    # must run sequentially to be prunable between them) for the ability to
    # stop early — worthwhile when trials, not folds, are the bottleneck.
    return _cross_val_score_with_pruning(pipe, x, y, model, scoring, cv, groups, trial)


def _cross_val_score_with_pruning(pipe, x, y, model, scoring, cv, groups, trial):
    scorer = get_scorer(scoring)
    splitter = check_cv(cv, y, classifier=is_classifier(model))
    y_arr = np.asarray(y)

    fold_scores = []
    for step, (train_idx, test_idx) in enumerate(splitter.split(x, y_arr, groups)):
        x_train = x.iloc[train_idx] if hasattr(x, "iloc") else x[train_idx]
        x_test = x.iloc[test_idx] if hasattr(x, "iloc") else x[test_idx]

        fold_pipe = clone(pipe)
        fold_pipe.fit(x_train, y_arr[train_idx])
        fold_scores.append(scorer(fold_pipe, x_test, y_arr[test_idx]))

        trial.report(float(np.mean(fold_scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))
