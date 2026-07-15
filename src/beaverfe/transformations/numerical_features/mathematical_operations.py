import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MathematicalOperations(BaseEstimator, TransformerMixin):
    def __init__(self, operations_options=None, track_columns=False):
        self.operations_options = operations_options
        self.track_columns = track_columns

        self.tracked_columns = {}

    def fit(self, X, y=None):
        # No fitting required for this transformer, maintaining compatibility with scikit-learn API
        return self

    @staticmethod
    def _apply_op(a: pd.Series, b: pd.Series, operation: str) -> pd.Series | None:
        """Apply a single binary operation between two raw (unprocessed) series.

        Returns None for unrecognized operations so callers can skip the tuple
        (mirrors the previous `continue` behavior of the inline if/elif chain).
        """
        if operation == "add":
            return a + b

        if operation == "subtract":
            return a - b

        if operation == "multiply":
            return a * b

        if operation == "divide":
            return a / b

        if operation == "modulus":
            return a % b

        if operation == "hypotenuse":
            return np.hypot(a, b)

        if operation == "mean":
            return (a + b) / 2

        if operation == "power":
            return np.power(a, b)

        if operation == "min":
            return np.minimum(a, b)

        if operation == "max":
            return np.maximum(a, b)

        if operation == "log_ratio":
            return np.log(a / b)

        return None

    @staticmethod
    def _apply_unary_op(a: pd.Series, operation: str) -> pd.Series | None:
        """Apply a single unary operation to a raw (unprocessed) series.

        Returns None for unrecognized operations, mirroring `_apply_op`.
        """
        # Suppress numpy's "invalid value encountered" warnings for sqrt of
        # negative values and similar domain issues: the resulting NaN is
        # handled uniformly by the __is_invalid flag machinery below, so a
        # console warning would just be noise.
        with np.errstate(invalid="ignore", divide="ignore"):
            if operation == "square":
                return a**2

            if operation == "cube":
                return a**3

            if operation == "sqrt":
                return np.sqrt(a)

            if operation == "cbrt":
                return np.cbrt(a)

            if operation == "reciprocal":
                return 1 / a

            if operation == "abs":
                return a.abs()

        return None

    @classmethod
    def _resolve_operand(
        cls, X: pd.DataFrame, expr: str | tuple
    ) -> tuple[pd.Series | None, list[str]]:
        """Resolve an expression to (series, root_columns).

        Unified recursive grammar (no depth limit — any sub-expression can be
        wrapped in a unary op or combined via a binary op with any other
        sub-expression, including composite ones):

            expr := column_name (str)
                  | (expr, unary_op)          # unary wraps ANY sub-expression
                  | (expr, expr, binary_op)   # binary combines two sub-expressions

        This single recursive definition applies uniformly whether `expr` is
        a top-level entry in `operations_options` or a nested operand inside
        another expression — there is no separate "operand" grammar anymore,
        which is what previously allowed a flat, ambiguous 5-element chained
        shape (operand1, op1, operand2, op2, operand3) to exist. Chained
        expressions must now be written as explicit nested tuples, e.g.
        `(("a", "b", "add"), "c", "multiply")` for `(a + b) * c`, which is
        unambiguous because the nesting itself encodes evaluation order.

        The generated column name is computed separately by
        `describe_operation` (the single source of truth for naming, also
        used by the space generator); this method only resolves values.
        """
        if isinstance(expr, str):
            return X[expr], [expr]

        if len(expr) == 2:
            sub_expr, unary_op = expr
            series, roots = cls._resolve_operand(X, sub_expr)
            result = None if series is None else cls._apply_unary_op(series, unary_op)
            return result, roots

        if len(expr) == 3:
            expr1, expr2, operation = expr
            series1, roots1 = cls._resolve_operand(X, expr1)
            series2, roots2 = cls._resolve_operand(X, expr2)
            result = (
                None
                if series1 is None or series2 is None
                else cls._apply_op(series1, series2, operation)
            )
            return result, roots1 + roots2

        raise ValueError(f"Unsupported expression shape: {expr!r}")

    @classmethod
    def describe_operation(cls, op_tuple: tuple) -> str:
        """Return the cosmetic column name `transform` generates for a given
        expression, without needing actual data. This is the single source
        of truth for the naming scheme, shared with the space generator so it
        can identify MI-scored candidate columns and never drift out of sync
        with what `transform` actually produces.

        Purely mirrors the recursive grammar handled by `_resolve_operand`
        (see its docstring), but requires no DataFrame since it only builds
        the name string.

        Composite sub-expressions (i.e. anything that isn't a bare column
        name) are wrapped in parentheses so the generated name preserves the
        same unambiguous nesting/grouping as the underlying tuple — without
        this, flattening everything with "__" would silently reintroduce the
        exact ambiguity the nested-tuple grammar was designed to eliminate.
        For example `(("hue", "flavanoids", "multiply"), ("color_intensity",
        "proanthocyanins", "divide"), "power")` becomes
        `(hue__multiply__flavanoids)__power__(color_intensity__divide__proanthocyanins)`,
        making the grouping explicit instead of a flat, unparseable
        `hue__multiply__flavanoids__power__color_intensity__divide__proanthocyanins`.
        """
        if isinstance(op_tuple, str):
            return op_tuple

        if len(op_tuple) == 2:
            sub_expr, unary_op = op_tuple
            return f"{cls._describe_wrapped(sub_expr)}__{unary_op}"

        if len(op_tuple) == 3:
            expr1, expr2, operation = op_tuple
            return (
                f"{cls._describe_wrapped(expr1)}__{operation}"
                f"__{cls._describe_wrapped(expr2)}"
            )

        raise ValueError(f"Unsupported operation tuple shape: {op_tuple!r}")

    @classmethod
    def _describe_wrapped(cls, expr: str | tuple) -> str:
        """Describe an expression, parenthesizing it if it is composite
        (i.e. not a bare column name), so its name unambiguously reads as a
        single grouped unit when embedded in a larger expression's name.
        """
        name = cls.describe_operation(expr)
        return f"({name})" if isinstance(expr, tuple) else name

    def transform(self, X, y=None):
        # Protect the caller's DataFrame — all mutations happen on this copy.
        X = X.copy()

        # Each operation is computed from the original input columns only (no
        # operation references a column generated by a previous operation in
        # the same call), so the collect-then-concat pattern is safe: we
        # accumulate result Series and join them once after the loop.
        new_cols: list[pd.Series] = []

        for op_tuple in self.operations_options or []:
            # Every entry in operations_options is itself a recursive
            # expression (see _resolve_operand's docstring for the grammar):
            # a raw column reference wrapped in 1+ unary/binary combinators,
            # nested to any depth, e.g.
            # ((("a", "square"), "b", "add"), ("c", "sqrt"), "multiply")
            # -> (a**2 + b) * sqrt(c).
            #
            # A bare column name (str) at the TOP level is intentionally
            # rejected here (though still valid as a nested sub-expression):
            # it applies no transformation at all, so describe_operation
            # would return the raw column's own name, generating a
            # duplicate-named column that silently corrupts the DataFrame.
            if isinstance(op_tuple, str):
                continue

            try:
                new_column = self.describe_operation(op_tuple)
            except (ValueError, TypeError):
                continue

            result, input_columns = self._resolve_operand(X, op_tuple)

            if result is None:
                continue

            # Replace inf with NaN so both cases are handled uniformly.
            result = result.replace([np.inf, -np.inf], np.nan)
            result.name = new_column

            # Inject an indicator column for cells that are structurally
            # undefined (inf or NaN after the operation), so the signal is not
            # silently erased by the fillna below. This mirrors the pattern
            # used by MissingValuesIndicator for upstream missing values.
            nan_mask = result.isna()
            if nan_mask.any():
                flag_series = nan_mask.astype("int8")
                flag_series.name = f"{new_column}__is_invalid"
                new_cols.append(flag_series)

            # Keep fillna(0) so downstream scalers do not encounter NaN.
            new_cols.append(result.fillna(0))

            if self.track_columns:
                self.tracked_columns[new_column] = input_columns

        # Single concat avoids per-iteration DataFrame fragmentation.
        if new_cols:
            X = pd.concat([X, *new_cols], axis=1)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
