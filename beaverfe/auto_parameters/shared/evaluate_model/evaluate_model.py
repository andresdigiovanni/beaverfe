import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# Custom selector to exclude datetime columns
class ExcludeDatetimeColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.select_dtypes(
            exclude=["datetime64[ns]", "datetime64"]
        ).columns
        return self

    def transform(self, X):
        return X[self.columns_]


def build_pipeline(model, transformer=None):
    steps = []

    # Step 0: Drop datetime columns
    steps.append(("drop_datetime", ExcludeDatetimeColumns()))

    # Step 1: Optional custom transformer
    if transformer:
        steps.append(("transformer", transformer))

    # Step 2: Define imputers
    numeric_imputer = SimpleImputer(strategy="constant", fill_value=0)
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")

    # Step 3: Categorical pipeline: impute + encode
    categorical_pipeline = Pipeline(
        [
            ("imputer", categorical_imputer),
            (
                "label_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Step 4: Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_imputer, make_column_selector(dtype_include="number")),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ],
        remainder="drop",  # drop unhandled types like datetime
    )

    # Step 5: Add preprocessing and model
    steps.append(("preprocessing", preprocessor))
    steps.append(("model", model))

    return Pipeline(steps=steps)


def evaluate_model(
    x,
    y,
    model,
    scoring,
    cv=5,
    groups=None,
    transformer=None,
):
    pipe = build_pipeline(model, transformer)
    scores = cross_val_score(
        pipe, x, y, scoring=scoring, cv=cv, groups=groups, n_jobs=-1
    )

    return np.mean(scores)
