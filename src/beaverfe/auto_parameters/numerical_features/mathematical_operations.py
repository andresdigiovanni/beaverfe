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
    ALL_OPERATIONS = SYMMETRIC_OPERATIONS + NON_SYMMETRIC_OPERATIONS
    UNARY_OPERATIONS = ["square", "cube", "sqrt", "cbrt", "reciprocal", "abs"]
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
    # 3-column candidates (col1 op1 col2) op2 col3 are NOT built from the full
    # O(n^3 * ops^2) cross product (prohibitively expensive on wide datasets).
    # Instead, only the top-relevance candidates from the previous depth are
    # expanded further, bounding the extra candidates per depth to roughly
    # TOP_CANDIDATES_PER_LEVEL * (n_columns * len(ALL_OPERATIONS) + len(UNARY_OPERATIONS)).
    TOP_CANDIDATES_PER_LEVEL = 20
    # Level 0 keeps this many top-scoring unary transforms per column instead
    # of only the single best one. A unary transform that is only useful in
    # combination with another column (interaction effect) can have a low or
    # zero univariate MI score and be wrongly discarded if only the top-1 is
    # kept; keeping a small top-K gives such transforms a chance to survive
    # into later expansion depths, where they are actually evaluated in
    # combination and can then win the final MRMR selection on their own
    # merits.
    UNARY_TOP_K_PER_COLUMN = 2
    # Maximum expression-nesting depth explored by the iterative expansion
    # loop in get_search_space. Depth 1 = raw column pairs and raw-column
    # unary transforms (cheap, exhaustive). Each additional depth combines
    # the previous depth's top-TOP_CANDIDATES_PER_LEVEL candidates (by MI
    # relevance) with a raw column (binary op) or wraps them in a unary op,
    # instead of exploring the full combinatorial expression space. Growth
    # per depth is bounded by TOP_CANDIDATES_PER_LEVEL, not exponential in
    # MAX_DEPTH, since only the single best-scoring depth is expanded next
    # (not the union of every depth seen so far).
    MAX_DEPTH = 3

    def get_search_space(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> SearchSpace:
        """Iterative, bounded expansion of the math-operations candidate pool.

        Building the full nested-expression space exhaustively is
        combinatorially infeasible (grows with ops^depth * n^depth). Instead
        this greedily expands in depths, each pruned to the most-promising
        candidates (by univariate MI relevance) before the next depth extends
        them:

          Depth 1: raw column pairs, and the top UNARY_TOP_K_PER_COLUMN unary
                    transforms per raw column (cheap, exhaustive over raw
                    columns only).
          Depth 2..MAX_DEPTH: the top-TOP_CANDIDATES_PER_LEVEL scoring
                    candidates from the previous depth (which may themselves
                    already be composite expressions), each combined with
                    every raw column via a binary op, or wrapped in a unary
                    op. This is what lets a Depth-1 candidate like
                    ("a", "square") be extended into
                    (("a", "square"), "b", "add") at Depth 2, and further into
                    ((("a", "square"), "b", "add"), "sqrt") at Depth 3, without
                    ever materializing the full cross product.

        The final MRMR selection runs once over the pooled candidates from
        every depth.
        """
        numeric_columns = dtypes.numerical_columns(X)
        n_numeric = len(numeric_columns)
        if n_numeric == 0:
            return {}

        top_k = min(50, max(10, int(0.25 * n_numeric)))

        # Depth 1: raw column pairs.
        transformations_map, pair_candidates = self._generate_operations(
            X, numeric_columns
        )
        if not pair_candidates:
            return {}

        # Depth 1: top-K unary transforms per raw column. best_unary already
        # holds each kept transform's MI score (computed internally by
        # _best_unary_per_column while picking the top-K), so those scores
        # are reused directly below instead of re-scoring the same
        # (col, unary_op) candidates a second time.
        best_unary = self._best_unary_per_column(X, y, numeric_columns)
        unary_candidates: list[tuple] = []
        unary_relevance: dict[str, float] = {}
        for col, options in best_unary.items():
            for op, score in options:
                op_tuple = (col, op)
                unary_candidates.append(op_tuple)
                unary_relevance[MathematicalOperations.describe_operation(op_tuple)] = (
                    score
                )
        for op_tuple in unary_candidates:
            name = MathematicalOperations.describe_operation(op_tuple)
            transformations_map[name] = op_tuple

        depth_candidates = pair_candidates + unary_candidates
        all_candidates: list[tuple] = list(depth_candidates)

        # Iteratively expand the best-scoring candidates from each depth into
        # the next, up to MAX_DEPTH. Stops early if a depth yields nothing
        # (e.g. fewer than 2 numeric columns leaves no room to expand).
        #
        # Every depth's relevance scores are accumulated into
        # `relevance_scores` (rather than discarded once the next depth is
        # picked) so the final MRMR selection can reuse them directly instead
        # of re-scoring every candidate from scratch: each candidate would
        # otherwise be fit_transform'd and MI-scored twice — once here, once
        # again in _select_top_k_mrmr — doubling the single most expensive
        # part of search-space generation for no benefit, since the scores
        # are deterministic given (X, y). The unary candidates' scores are
        # already known (see above), so only the raw pairs need scoring here.
        pair_relevance = self._score_candidates(X, y, pair_candidates)
        depth_relevance: dict[str, float] = {**pair_relevance, **unary_relevance}
        relevance_scores: dict[str, float] = dict(depth_relevance)
        for _depth in range(2, self.MAX_DEPTH + 1):
            expanded = self._expand_level(
                depth_relevance, numeric_columns, transformations_map
            )
            if not expanded:
                break

            for op_tuple in expanded:
                name = MathematicalOperations.describe_operation(op_tuple)
                transformations_map[name] = op_tuple

            all_candidates.extend(expanded)
            depth_relevance = self._score_candidates(X, y, expanded)
            relevance_scores.update(depth_relevance)

        selected = self._select_top_k_mrmr(
            X, y, all_candidates, top_k, transformations_map, relevance_scores
        )

        # Optuna choices are the repr() of the operation tuple itself (parsed
        # back via ast.literal_eval in PipelineAssembler._decode_math_ops).
        # Column names can no longer serve as the round-trip encoding once
        # expressions can be arbitrarily nested: e.g. a unary-wrapped
        # composite expression and a raw pair can legitimately produce column
        # names of the same "__"-split shape, which a split("__")-based
        # decoder cannot disambiguate reliably. The tuple `repr` has no such
        # ambiguity regardless of what characters appear in column names, and
        # also has no evaluation-order ambiguity, unlike the old flat
        # 5-element chained shape it replaces.
        encoded_ops: list[str] = [
            repr(transformations_map[name])
            for name in selected
            if name in transformations_map
        ]

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
                    # Deterministic naming from MathematicalOperations — no
                    # need to actually run the transformer to read it back.
                    name = MathematicalOperations.describe_operation(op_tuple)
                    transformations[name] = op_tuple

        return transformations, operations

    def _best_unary_per_column(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, columns: list[str]
    ) -> dict[str, list[tuple[str, float]]]:
        """Depth 1: score every (col, unary_op) candidate and keep the top
        UNARY_TOP_K_PER_COLUMN scoring unary transforms per column (sorted
        best-first), instead of only the single best one. Bounds cost to
        O(n_columns * len(UNARY_OPERATIONS)), independent of the (much
        larger) pair/expansion space.

        Keeping more than 1 candidate per column matters because a unary
        transform can be genuinely useful only in combination with another
        column (interaction effect) while having low/zero univariate MI on
        its own — a strict top-1 cut would silently discard it before it
        ever gets a chance to be tried in a later expansion depth. A small
        top-K keeps the pool bounded while giving such transforms a chance
        to be evaluated in combination, where the final MRMR selection can
        judge them on their actual merit.
        """
        unary_candidates = [
            (col, op) for col in columns for op in self.UNARY_OPERATIONS
        ]
        if not unary_candidates:
            return {}

        scores = self._score_candidates(X, y, unary_candidates)

        scored_per_column: dict[str, list[tuple[str, float]]] = {}
        for col, unary_op in unary_candidates:
            name = MathematicalOperations.describe_operation((col, unary_op))
            score = scores.get(name)
            if score is None or score <= 0:
                continue
            scored_per_column.setdefault(col, []).append((unary_op, score))

        best_per_column: dict[str, list[tuple[str, float]]] = {}
        for col, scored in scored_per_column.items():
            scored.sort(key=lambda item: item[1], reverse=True)
            best_per_column[col] = scored[: self.UNARY_TOP_K_PER_COLUMN]

        return best_per_column

    def _expand_level(
        self,
        prev_relevance: dict[str, float],
        columns: list[str],
        transformations_map: dict[str, tuple],
    ) -> list[tuple]:
        """Generic expansion from one depth to the next: take the
        top-TOP_CANDIDATES_PER_LEVEL scoring expressions from the previous
        depth and either (a) combine each with every raw column via a binary
        op, or (b) wrap each in a unary op — instead of exploring the full
        combinatorial expression space. This single mechanism subsumes what
        earlier phases handled as separate special cases ("augment pairs with
        unary", "extend pairs into triples"): a Depth-1 unary candidate like
        ("a", "square") can now be picked up here and combined with a raw
        column, and a Depth-2 pair can be wrapped in a unary op or extended
        with a third column at Depth 3, using the exact same code path.
        """
        if not prev_relevance:
            return []

        top_prev = sorted(
            prev_relevance.items(), key=lambda item: item[1], reverse=True
        )[: self.TOP_CANDIDATES_PER_LEVEL]

        expanded: list[tuple] = []
        for name, _ in top_prev:
            base_expr = transformations_map.get(name)
            if base_expr is None:
                continue

            base_roots = self._root_columns(base_expr)

            for col in columns:
                if col in base_roots:
                    continue
                for op in self.ALL_OPERATIONS:
                    expanded.append((base_expr, col, op))

            for unary_op in self.UNARY_OPERATIONS:
                expanded.append((base_expr, unary_op))

        return expanded

    @staticmethod
    def _root_columns(expr: str | tuple) -> set[str]:
        """Return the set of raw column names referenced anywhere inside an
        expression, used to avoid generating degenerate candidates that
        combine an expression with a column it already depends on (e.g.
        (a + b) - a).
        """
        if isinstance(expr, str):
            return {expr}
        if len(expr) == 2:
            return MathematicalOperationsSpaceGenerator._root_columns(expr[0])
        return MathematicalOperationsSpaceGenerator._root_columns(
            expr[0]
        ) | MathematicalOperationsSpaceGenerator._root_columns(expr[1])

    def _operation_definitions(self, i: int, j: int) -> list[str]:
        definitions: list[str] = []
        for op in self.SYMMETRIC_OPERATIONS:
            if i > j:
                definitions.append(op)
        for op in self.NON_SYMMETRIC_OPERATIONS:
            definitions.append(op)
        return definitions

    def _score_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        operation_candidates: list[tuple],
    ) -> dict[str, float]:
        """Chunked relevance (MI) scoring shared by every expansion level.
        Scoring in chunks bounds peak memory to one chunk's width regardless
        of total candidate count.
        """
        relevance_scores: dict[str, float] = {}

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

        return relevance_scores

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
        transformations_map: dict[str, tuple],
        relevance_scores: dict[str, float] | None = None,
    ) -> list[str]:
        """Two-phase MRMR selection.

        Phase 1 (chunked): score all candidates by MI(f, y); retain the top
        PRE_FILTER_MULTIPLIER * top_k as pre-filtered candidates. This bounds
        peak memory to one chunk's width regardless of total candidate count.

        Phase 2 (in-memory): materialize the pre-filtered candidates, then run
        greedy MRMR initialized with the original X columns. The redundancy set
        starts with the original features so that candidates that merely replicate
        existing columns are penalized from the first selection step.

        `relevance_scores` lets a caller that already scored every candidate
        while building it (e.g. get_search_space's depth-by-depth expansion)
        pass those scores straight through, instead of this method
        re-running the expensive fit_transform + MI pass on every candidate
        all over again. Only computed here (fallback) when not supplied, so
        direct/test callers that don't have precomputed scores keep working
        unchanged.
        """
        pre_filter_k = min(
            top_k * self.PRE_FILTER_MULTIPLIER, len(operation_candidates)
        )

        # --- Phase 1: relevance scoring (reused if already computed) ---
        if relevance_scores is None:
            relevance_scores = self._score_candidates(X, y, operation_candidates)

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
        pre_filtered_tuples = [
            transformations_map[name]
            for name in pre_filtered
            if name in transformations_map
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
