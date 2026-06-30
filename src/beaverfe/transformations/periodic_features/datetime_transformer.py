from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, track_columns=False):
        self.features = features or []
        self.track_columns = track_columns

        self.tracked_columns = {}

    def get_params(self, deep=True):
        return {
            "features": self.features,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, X, y=None):
        return self  # No fitting necessary, but method required for compatibility

    def transform(self, X, y=None):
        X = X.copy()

        for column in self.features:
            X[f"{column}_year"] = X[column].dt.year
            X[f"{column}_month"] = X[column].dt.month
            X[f"{column}_day"] = X[column].dt.day
            X[f"{column}_weekday"] = X[column].dt.weekday
            X[f"{column}_hour"] = X[column].dt.hour
            X[f"{column}_minute"] = X[column].dt.minute
            X[f"{column}_second"] = X[column].dt.second

            if self.track_columns:
                self.tracked_columns[f"{column}_year"] = [column]
                self.tracked_columns[f"{column}_month"] = [column]
                self.tracked_columns[f"{column}_day"] = [column]
                self.tracked_columns[f"{column}_weekday"] = [column]
                self.tracked_columns[f"{column}_hour"] = [column]
                self.tracked_columns[f"{column}_minute"] = [column]
                self.tracked_columns[f"{column}_second"] = [column]

        X = X.drop(columns=self.features)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
