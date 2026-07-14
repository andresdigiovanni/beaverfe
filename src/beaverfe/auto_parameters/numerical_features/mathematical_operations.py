import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from beaverfe.auto_parameters.shared.space_generator import SearchSpace, SpaceGenerator
from beaverfe.transformations import MathematicalOperations
from beaverfe.transformations.utils import dtypes


class MathematicalOperationsSpaceGenerator(SpaceGenerator):
    SYMMETRIC_OPERATIONS = ["add", "multiply", "mean", "hypotenuse", "min", "max"]
    NON_SYMMETRIC_OPERATIONS = [
        "subtract",
        "divide",
        "modulus",
        "power",
        "log_ratio",
    ]
    # Independent slots let Optuna add several generated features together instead
    # of at most one per pipeline (a single flat "math_ops" choice would collapse
    # the whole O(n^2) candidate space into one selectable value).
    MAX_SLOTS = 5
    # Candidate operations are scored in batches of this size rather than all
    # at once: materializing every O(n^2) candidate column into a single
    # DataFrame before running the MI filter causes a large one-shot memory
    # spike on wide datasets. Scoring in chunks bounds peak memory to one
    # chunk's width regardless of how many candidates there are in total.
    CHUNK_SIZE = 200
    # MRMR pre-filter: before running greedy selection, keep this many times
    # top_k candidates by raw MI relevance. Bounds the in-memory correlation
    # matrix to (PRE_FILTER_MULTIPLIER * top_k)^2 entries.
    PRE_FILTER_MULTIPLIER = 3

    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        numeric_columns = dtypes.numerical_columns(X)
        n_numeric = len(numeric_columns)
        if n_numeric == 0:
            return {}

        top_k = min(50, max(10, int(0.25 * n_numeric)))

        transformations_map, operation_candidates = self._generate_operations(
            X, numeric_columns
        )

        if not operation_candidates:
            return {}

        selected = self._select_top_k_mrmr(X, y, operation_candidates, top_k)

        encoded_ops: list[str] = []
        for col in selected:
            if col in transformations_map:
                col_a, col_b, op = transformations_map[col]
                encoded_ops.append(f"{col_a}__{op}__{col_b}")

        if not encoded_ops:
            return {}

        n_slots = min(self.MAX_SLOTS, len(encoded_ops))
        options = ["none", *encoded_ops]
        return {f"math_ops_{i + 1}": options for i in range(n_slots)}

    def _generate_operations(
        self, x: pd.DataFrame, columns: list[str]
    ) -> tuple[dict[str, tuple], list[tuple]]:
        transformations: dict[str, tuple] = {}
        operations: list[tuple] = []

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    continue
                for op in self._operation_definitions(i, j):
                    op_tuple = (col1, col2, op)
                    operations.append(op_tuple)
                    # Deterministic naming from MathematicalOperations.transform
                    # ("{col1}__{operation}__{col2}") — no need to actually run
                    # the transformer just to read the name back.
                    transformations[f"{col1}__{op}__{col2}"] = op_tuple

        return transformations, operations

    def _operation_definitions(self, i: int, j: int) -> list[str]:
        definitions: list[str] = []
        for op in self.SYMMETRIC_OPERATIONS:
            if i > j:
                definitions.append(op)
        for op in self.NON_SYMMETRIC_OPERATIONS:
            definitions.append(op)
        return definitions

    # LOOK-AHEAD NOTE (menu-curation bias, not label leakage into the shipped model):
    # Phase-1 MI ranking and phase-2 MRMR correlation both use the full dataset,
    # including rows that will later serve as CV validation folds. This makes the
    # Optuna search space mildly optimistic relative to true generalization.
    # Severity: LOW — only the offered menu is affected, not the final model's
    # feature values. Pass validation_fraction > 0 to limit this to a training slice.
    def _select_top_k_mrmr(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        operation_candidates: list[tuple],
        top_k: int,
    ) -> list[str]:
        """Two-phase MRMR selection.

        Phase 1 (chunked): score all candidates by MI(f, y); retain the top
        PRE_FILTER_MULTIPLIER * top_k as pre-filtered candidates. This bounds
        peak memory to one chunk's width regardless of total candidate count.

        Phase 2 (in-memory): materialize the pre-filtered candidates, then run
        greedy MRMR initialized with the original X columns. The redundancy set
        starts with the original features so that candidates that merely replicate
        existing columns are penalized from the first selection step.
        """
        pre_filter_k = min(
            top_k * self.PRE_FILTER_MULTIPLIER, len(operation_candidates)
        )
        relevance_scores: dict[str, float] = {}

        # --- Phase 1: relevance scoring in chunks ---
        for chunk_start in range(0, len(operation_candidates), self.CHUNK_SIZE):
            chunk = operation_candidates[chunk_start : chunk_start + self.CHUNK_SIZE]
            chunk_transformer = MathematicalOperations(chunk)
            try:
                X_chunk = chunk_transformer.fit_transform(X)
            except (TypeError, ValueError):
                continue

            new_cols = [
                c
                for c in X_chunk.columns
                if c not in X.columns and not c.endswith("__is_invalid")
            ]
            if new_cols:
                for name, score in self._score_columns(X_chunk[new_cols], y):
                    relevance_scores[name] = score

        if not relevance_scores:
            return []

        sorted_by_relevance = sorted(
            relevance_scores.items(), key=lambda item: item[1], reverse=True
        )
        pre_filtered = [
            name for name, score in sorted_by_relevance[:pre_filter_k] if score > 0
        ]
        if not pre_filtered:
            return []

        # --- Phase 2: greedy MRMR on pre-filtered candidates ---
        # Build a local name→tuple map to re-materialise only the pre-filtered set.
        local_map: dict[str, tuple] = {
            f"{col1}__{op}__{col2}": (col1, col2, op)
            for col1, col2, op in operation_candidates
        }
        pre_filtered_tuples = [
            local_map[name] for name in pre_filtered if name in local_map
        ]
        if not pre_filtered_tuples:
            return [
                name
                for name, _ in sorted_by_relevance[:top_k]
                if relevance_scores.get(name, 0) > 0
            ]

        try:
            X_pre = MathematicalOperations(pre_filtered_tuples).fit_transform(X)
        except (TypeError, ValueError):
            return [
                name
                for name, _ in sorted_by_relevance[:top_k]
                if relevance_scores.get(name, 0) > 0
            ]

        cand_cols = [
            c
            for c in pre_filtered
            if c in X_pre.columns and not c.endswith("__is_invalid")
        ]
        if not cand_cols:
            return [
                name
                for name, _ in sorted_by_relevance[:top_k]
                if relevance_scores.get(name, 0) > 0
            ]

        X_orig_vals = X.select_dtypes(include="number").values  # (n, n_orig)
        X_cand_vals = X_pre[cand_cols].values  # (n, n_cand)
        n_cand = len(cand_cols)
        n_orig = X_orig_vals.shape[1]

        # Correlation matrix over [candidates | original columns]
        combined = np.hstack([X_cand_vals, X_orig_vals]).T  # (n_cand+n_orig, n)
        corr_abs = np.abs(np.nan_to_num(np.corrcoef(combined), nan=0.0))
        corr_cand_cand = corr_abs[:n_cand, :n_cand]  # (n_cand, n_cand)
        corr_cand_orig = corr_abs[:n_cand, n_cand:]  # (n_cand, n_orig)

        relevance_arr = np.array(
            [relevance_scores.get(name, 0.0) for name in cand_cols]
        )

        # Greedy selection: redundancy set S is initialised with original X columns.
        accumulated_redundancy = corr_cand_orig.sum(axis=1).copy()
        n_in_set = max(n_orig, 1)
        remaining_mask = np.ones(n_cand, dtype=bool)
        selected_indices: list[int] = []

        while len(selected_indices) < top_k and remaining_mask.any():
            mrmr_scores = np.where(
                remaining_mask,
                relevance_arr - accumulated_redundancy / n_in_set,
                -np.inf,
            )
            best_idx = int(np.argmax(mrmr_scores))

            if relevance_arr[best_idx] <= 0:
                break

            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False
            accumulated_redundancy += corr_cand_cand[:, best_idx]
            n_in_set += 1

        return [cand_cols[i] for i in selected_indices]

    def _score_columns(
        self, X_block: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> list[tuple[str, float]]:
        try:
            y_arr = np.array(y)
            if len(np.unique(y_arr)) < 20:
                scores = mutual_info_classif(X_block, y_arr, discrete_features="auto")
            else:
                scores = mutual_info_regression(
                    X_block, y_arr, discrete_features="auto"
                )
            return list(zip(X_block.columns, scores, strict=True))
        except ValueError:
            # Fail-open for this chunk: treat its candidates as viable rather
            # than dropping them, since a downstream numeric issue in one
            # chunk shouldn't zero out the columns in it.
            return [(str(col), 1.0) for col in X_block.columns]
