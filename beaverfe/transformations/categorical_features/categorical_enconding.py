import category_encoders as ce
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoding(BaseEstimator, TransformerMixin):
    def __init__(
        self, transformation_options=None, ordinal_orders=None, track_columns=False
    ):
        self.transformation_options = transformation_options or {}
        self.ordinal_orders = ordinal_orders
        self.track_columns = track_columns

        self.tracked_columns = {}
        self._encoders = {}

    def get_params(self, deep=True):
        return {
            "transformation_options": self.transformation_options,
            "ordinal_orders": self.ordinal_orders,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        self._encoders = {}
        X = X.copy()

        for column, transformation in self.transformation_options.items():
            self._fit_encoder(X, y, column, transformation)

        return self

    def _fit_encoder(self, X, y, column, transformation):
        encoder_classes = {
            "backward_diff": ce.BackwardDifferenceEncoder,
            "basen": ce.BaseNEncoder,
            "binary": ce.BinaryEncoder,
            "catboost": ce.CatBoostEncoder,
            "count": lambda: X[column].value_counts().to_dict(),
            "dummy": lambda: ce.OneHotEncoder(drop_invariant=True),
            "glmm": ce.GLMMEncoder,
            "gray": ce.GrayEncoder,
            "hashing": ce.HashingEncoder,
            "helmert": ce.HelmertEncoder,
            "james_stein": ce.JamesSteinEncoder,
            "label": ce.OrdinalEncoder,
            "loo": lambda: ce.LeaveOneOutEncoder() if y is not None else None,
            "m_estimate": ce.MEstimateEncoder,
            "onehot": ce.OneHotEncoder,
            "ordinal": lambda: ce.OrdinalEncoder(
                mapping=[
                    {
                        "col": column,
                        "mapping": {
                            k: v for v, k in enumerate(self.ordinal_orders[column])
                        },
                    }
                ]
            ),
            "polynomial": ce.PolynomialEncoder,
            "quantile": ce.QuantileEncoder,
            "rankhot": ce.RankHotEncoder,
            "sum": ce.SumEncoder,
            "target": lambda: ce.TargetEncoder() if y is not None else None,
            "woe": ce.WOEEncoder,
        }

        if transformation == "count":
            self._encoders[column] = X[column].value_counts().to_dict()

        else:
            encoder_class = encoder_classes.get(transformation)
            self._encoders[column] = (
                encoder_class().fit(X[[column]], y)
                if y is not None
                else encoder_class().fit(X[[column]])
            )

    def transform(self, X, y=None):
        X = X.copy()
        for column, transformation in self.transformation_options.items():
            X[column] = X[column].fillna("Unknown")
            X = self._transform_column(
                X, column, transformation, self._encoders[column]
            )
        return X

    def _transform_column(self, X, column, transformation, transformer):
        if transformation in ["label", "ordinal"]:
            X[column] = transformer.transform(X[[column]])

            if self.track_columns:
                self.tracked_columns[column] = [column]

        elif transformation in [
            "backward_diff",
            "basen",
            "binary",
            "catboost",
            "dummy",
            "glmm",
            "gray",
            "hashing",
            "helmert",
            "james_stein",
            "loo",
            "m_estimate",
            "onehot",
            "polynomial",
            "quantile",
            "rankhot",
            "sum",
            "target",
            "woe",
        ]:
            encoded_array = transformer.transform(X[[column]])
            columns = transformer.get_feature_names_out([column])

            if transformation in ["hashing"]:
                columns = [c.replace("col_", f"{column}_") for c in columns]

            encoded_df = pd.DataFrame(encoded_array, columns=columns, index=X.index)
            X = pd.concat([X.drop(columns=[column]), encoded_df], axis=1)

            if self.track_columns:
                for new_column in columns:
                    self.tracked_columns[new_column] = [column]

        elif transformation == "count":
            X[column] = X[column].map(self._encoders[column]).fillna(0)

            if self.track_columns:
                self.tracked_columns[column] = [column]

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
