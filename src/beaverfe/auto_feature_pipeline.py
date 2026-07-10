import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import beaverfe.auto_parameters as ap
from beaverfe.auto_parameters.shared import (
    build_pipeline,
    evaluate_model,
)
from beaverfe.auto_parameters.shared.space_generator import SearchSpace
from beaverfe.beaver_pipeline import BeaverPipeline
from beaverfe.model_profiler import DataProfiler, ModelProfiler
from beaverfe.pipeline_assembler import PipelineAssembler
from beaverfe.pipeline_blocks import CANONICAL_ORDER

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def auto_feature_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    scoring: str,
    direction: str = "maximize",
    cv: int | Callable | None = 5,
    groups: np.ndarray | None = None,
    timeout: int | None = 600,
    n_trials: int | None = 100,
    verbose: bool = True,
) -> list[dict]:
    """
    Automatically discover the best feature engineering pipeline using Bayesian
    Optimisation (Optuna) over a globally declared search space.

    Args:
        X: Feature matrix.
        y: Target variable.
        model: Sklearn-compatible estimator.
        scoring: Scoring metric string for cross-validation.
        direction: "maximize" or "minimize". Defaults to "maximize".
        cv: Cross-validation strategy. Defaults to 5.
        groups: Group labels for cross-validation splitting. Defaults to None.
        timeout: Seconds budget for Optuna optimisation. Defaults to 600.
        n_trials: Maximum number of Optuna trials. Defaults to 100.
        verbose: Whether to print Optuna progress and relevant setup information.
            Third-party library warnings (sklearn, numpy, etc.) are always
            suppressed regardless of this flag. Defaults to True.
    Returns:
        List of transformer dicts with "name" and "params" keys ready for BeaverPipeline.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if verbose:
        if not logger.handlers or all(
            isinstance(h, logging.NullHandler) for h in logger.handlers
        ):
            _handler = logging.StreamHandler()
            _handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    n_features = X.shape[1]
    budget = f"{timeout}s" if n_trials is None else f"{n_trials} trials"
    logger.info("[BeaverFE] Starting auto feature pipeline")
    logger.info("[BeaverFE] Dataset: %d rows x %d features", X.shape[0], n_features)
    logger.info(
        "[BeaverFE] Model: %s | Scoring: %s (%s)",
        type(model).__name__,
        scoring,
        direction,
    )
    logger.info("[BeaverFE] Optimisation budget: %s", budget)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        X = X.copy()

        # 1. Detect model type
        model_type = ModelProfiler.detect(model)

        logger.info("[BeaverFE] Detected model type: %s", model_type)

        # 1b. Profile the data for data-aware search space generation.
        data_profiler = DataProfiler.from_data(X, y)

        # 2. Build global search space from all generators
        spaces: SearchSpace = {}
        generators = [
            ap.MissingValuesIndicatorSpaceGenerator(),
            ap.MissingValuesHandlerSpaceGenerator(),
            ap.DateTimeTransformerSpaceGenerator(),
            ap.CyclicalFeaturesTransformerSpaceGenerator(),
            ap.OutliersSpaceGenerator(),
            ap.MathematicalOperationsSpaceGenerator(),
            ap.SplineTransformationSpaceGenerator(),
            ap.NumericalBinningSpaceGenerator(),
            ap.CategoricalEncodingSpaceGenerator(),
            ap.NormalizerSpaceGenerator(),
            ap.ColumnSelectionSpaceGenerator(),
            ap.DimensionalityReductionSpaceGenerator(),
        ]
        for generator in generators:
            if hasattr(generator, "set_profiler"):
                generator.set_profiler(data_profiler)
            spaces.update(generator.get_search_space(X, y))

        logger.info("[BeaverFE] Search space: %d parameters", len(spaces))
        logger.info("[BeaverFE] Running Bayesian optimisation...")

        assembler = PipelineAssembler()

        # 3. Define Optuna objective
        def objective(trial: optuna.Trial) -> float:
            transformer_dicts = assembler.assemble(trial, spaces, model_type, X, y)
            pipeline = BeaverPipeline(transformer_dicts, order=CANONICAL_ORDER)
            try:
                # optuna.TrialPruned (raised by evaluate_model when a fold-by-fold
                # score triggers the pruner) isn't a ValueError/RuntimeError, so
                # it propagates past this except clause for Optuna to handle.
                score = float(
                    evaluate_model(
                        X,
                        y,
                        model,
                        scoring,
                        cv,
                        groups,
                        transformer=pipeline,
                        trial=trial,
                        skip_safety_net=_recipe_guarantees_clean_output(
                            transformer_dicts, spaces
                        ),
                    )
                )
            except (ValueError, RuntimeError):
                logger.warning(
                    "[BeaverFE] Trial evaluation failed; returning worst score."
                )
                score = float("-inf") if direction == "maximize" else float("inf")
            return score

        # 4. Run Optuna study.
        # multivariate+group: the search space is wide (one dimension per
        # column per block) and TPE's default per-parameter independence
        # assumption models that poorly; grouping lets it model each block's
        # parameters jointly instead of treating hundreds of dimensions as
        # independent.
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42, multivariate=True, group=True),
            pruner=optuna.pruners.MedianPruner(),
        )
        # Seed an all-"none" baseline trial so the study starts from a known
        # reference point instead of spending early trials rediscovering that
        # "do nothing" is a reasonable score to beat. Keys without a "none"
        # option (structural metadata, numeric-only choices, or blocks forced
        # non-"none" for this model type) are left for the sampler to fill in
        # normally.
        baseline_params = assembler.default_params(spaces, model_type)
        if baseline_params:
            study.enqueue_trial(baseline_params, skip_if_exists=True)

        # Safety floor: when the caller leaves both n_trials and timeout at None
        # the study would run indefinitely. Auto-scale n_trials from the search
        # space size so there is always a hard upper bound.
        effective_n_trials = n_trials
        if effective_n_trials is None and timeout is None:
            effective_n_trials = max(50, min(500, int(10 * len(spaces) ** 0.5)))
            logger.info(
                "[BeaverFE] n_trials and timeout both unset; "
                "auto-set n_trials=%d from search space size (%d params)",
                effective_n_trials,
                len(spaces),
            )

        # Early stopping: halt when TPE has converged, i.e. no improvement in
        # `patience` consecutive complete trials.  Patience scales with the
        # search space size so larger spaces get more exploration before giving
        # up.  Minimum of 20 avoids premature stopping on small spaces.
        patience = max(20, min(50, len(spaces) // 5))
        _no_improve_count = [0]
        _best_so_far: list[float] = [
            float("-inf") if direction == "maximize" else float("inf")
        ]

        def _early_stop_callback(
            study: optuna.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            improved = (
                direction == "maximize" and study.best_value > _best_so_far[0]
            ) or (direction == "minimize" and study.best_value < _best_so_far[0])
            if improved:
                _best_so_far[0] = study.best_value
                _no_improve_count[0] = 0
            else:
                _no_improve_count[0] += 1
                if _no_improve_count[0] >= patience:
                    logger.info(
                        "[BeaverFE] Early stopping: no improvement for %d "
                        "consecutive trials (patience=%d)",
                        _no_improve_count[0],
                        patience,
                    )
                    study.stop()

        callbacks: list = [_early_stop_callback]
        if verbose:

            def _progress_callback(
                study: optuna.Study, trial: optuna.trial.FrozenTrial
            ) -> None:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    logger.info(
                        "[BeaverFE] Trial %4d | score: %.6f | best: %.6f",
                        trial.number,
                        trial.value,
                        study.best_value,
                    )

            callbacks.append(_progress_callback)

        study.optimize(
            objective, timeout=timeout, n_trials=effective_n_trials, callbacks=callbacks
        )

        logger.info(
            "[BeaverFE] Optimisation complete: %d trials | best score: %.6f",
            len(study.trials),
            study.best_value,
        )

        # 5. Reconstruct best pipeline
        best_params = study.best_params
        best_dicts = assembler.assemble_from_params(best_params, spaces, model_type)
        best_pipeline = BeaverPipeline(best_dicts, order=CANONICAL_ORDER)

        # 6. Post-hoc feature selection via single-shot permutation importance.
        # Architecture: fit best_pipeline on full X to get X_transformed, then
        # select transformed features with positive permutation importance on a
        # held-out validation split. Prune the recipe directly from the selected
        # feature names (transformed space).
        #
        # Why single-shot perm importance instead of RFECV:
        #   - RFECV needs O(log n_features) model fits; single-shot needs one fit.
        #   - Post-Optuna we only need to *remove clearly useless features* — we
        #     don't need the precise optimal subset that iterative RFECV provides.
        #   - Single-shot is model-agnostic and handles correlated features well
        #     enough for this purpose.
        logger.info("[BeaverFE] Running post-hoc feature selection...")
        try:
            X_transformed = best_pipeline.fit_transform(X.copy(), y)
            n_before = X_transformed.shape[1]
            selected_transformed = _select_features_by_importance(
                X_transformed, np.array(y), model, scoring
            )

            if selected_transformed and len(selected_transformed) < n_before:
                best_dicts = _prune_by_selected_transformed(
                    best_dicts, selected_transformed
                )
                best_dicts.append(
                    {
                        "name": "ColumnSelection",
                        "params": {"features": sorted(selected_transformed)},
                    }
                )
                n_removed = n_before - len(selected_transformed)
                logger.info(
                    "[BeaverFE] Feature selection: removed %d of %d transformed "
                    "features (%d kept)",
                    n_removed,
                    n_before,
                    len(selected_transformed),
                )
            else:
                logger.info(
                    "[BeaverFE] Feature selection: all %d transformed features kept",
                    n_before,
                )
        except Exception:
            logger.warning("[BeaverFE] Post-hoc feature selection failed; skipping.")

    # 7. Strip internal "block" key before returning
    return [{"name": d["name"], "params": d["params"]} for d in best_dicts]

    # 7. Strip internal "block" key before returning
    return [{"name": d["name"], "params": d["params"]} for d in best_dicts]


def _select_features_by_importance(
    X_transformed: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    scoring: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> set[str]:
    """Select transformed features using single-shot permutation importance.

    Fits the model once on a training split, then evaluates permutation
    importance on a held-out validation split.  Features whose mean importance
    is positive (> 0) are kept; all others are discarded.  If every feature
    has non-positive importance the single best-scoring feature is kept as a
    minimum fallback.

    This is intentionally simpler and faster than iterative RFECV:
    post-Optuna selection only needs to *remove clearly useless features* —
    not find the theoretically optimal minimal subset.

    Args:
        X_transformed: Fully transformed feature matrix (output of best pipeline).
        y: Target array.
        model: Sklearn-compatible estimator.
        scoring: Scoring metric string (same one used during Optuna).
        n_repeats: Number of permutation repetitions (default: 10).
        random_state: Random seed.

    Returns:
        Set of column names whose permutation importance is > 0.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=0.25, random_state=random_state
    )
    estimator = build_pipeline(model)
    estimator.fit(X_train, y_train)

    result = permutation_importance(
        estimator,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    importances = result.importances_mean
    selected_mask = importances > 0.0

    if not selected_mask.any():
        selected_mask[int(np.argmax(importances))] = True

    return set(X_transformed.columns[selected_mask])


def _recipe_guarantees_clean_output(
    transformer_dicts: list[dict], spaces: SearchSpace
) -> bool:
    """Return True when the recipe is guaranteed to produce clean output for the model.

    Specifically, returns True when:
    - A MissingValuesHandler step is present, AND
    - Either a CategoricalEncoding step is present, OR there were no categorical
      columns to encode in the first place (detected by the absence of any
      ``cat_enc_``-prefixed keys in the search space).

    Returns False (keep safety net) when either condition is not satisfied, so
    build_pipeline continues to guard against crashes for incomplete recipes.
    """
    names = {d["name"] for d in transformer_dicts}
    has_missing_handler = "MissingValuesHandler" in names
    has_categorical_encoding = "CategoricalEncoding" in names
    no_categorical_columns = not any(key.startswith("cat_enc_") for key in spaces)
    return has_missing_handler and (has_categorical_encoding or no_categorical_columns)


def _prune_eliminated_columns(
    transformer_dicts: list[dict], keep: set[str]
) -> list[dict]:
    """Drop references to raw input columns permutation-importance elimination
    decided to remove, so no recipe step looks up a column that won't exist.

    Builds an extended keep set that grows as expansion steps are processed:
    generated column names (math-op results, binned columns, cyclical/datetime
    derivatives, missing indicators) are added whenever their source columns
    survive pruning, so downstream transformation steps retain per-column
    entries for those generated features.
    """
    # Grows to include column names generated by earlier pipeline steps whose
    # source columns are in `keep`.
    extended_keep: set[str] = set(keep)

    pruned: list[dict] = []

    for d in transformer_dicts:
        params = dict(d["params"])

        options = params.get("transformation_options")
        if isinstance(options, dict):
            params["transformation_options"] = {
                col: value for col, value in options.items() if col in extended_keep
            }
            if not params["transformation_options"]:
                continue

        if d["name"] == "MathematicalOperations":
            operations = params.get("operations_options") or []
            params["operations_options"] = [
                op for op in operations if op[0] in keep and op[1] in keep
            ]
            if not params["operations_options"]:
                continue

        features = params.get("features")
        if isinstance(features, list):
            params["features"] = [col for col in features if col in keep]
            if not params["features"]:
                continue

        pruned.append({**d, "params": params})

        # Expand extended_keep with generated column names from this step,
        # so downstream steps' transformation_options entries for those
        # columns are preserved.
        _update_extended_keep(d["name"], params, extended_keep)

    return pruned


def _update_extended_keep(name: str, params: dict, extended_keep: set[str]) -> None:
    """Add generated column names to ``extended_keep`` after an expansion step
    has been accepted into the pruned pipeline.

    Called with already-pruned ``params``, so only surviving columns and
    operations contribute new names.
    """
    if name == "MissingValuesIndicator":
        for col in params.get("features") or []:
            extended_keep.add(f"{col}__is_missing")

    elif name == "DateTimeTransformer":
        suffixes = (
            "_year",
            "_month",
            "_day",
            "_weekday",
            "_hour",
            "_minute",
            "_second",
        )
        for col in params.get("features") or []:
            for suffix in suffixes:
                extended_keep.add(f"{col}{suffix}")

    elif name == "CyclicalFeaturesTransformer":
        for col in params.get("transformation_options") or {}:
            extended_keep.add(f"{col}_sin")
            extended_keep.add(f"{col}_cos")

    elif name == "MathematicalOperations":
        for op in params.get("operations_options") or []:
            col1, col2, operation = op[0], op[1], op[2]
            generated = f"{col1}__{operation}__{col2}"
            extended_keep.add(generated)
            extended_keep.add(f"{generated}__is_invalid")

    elif name == "NumericalBinning":
        for col, val in (params.get("transformation_options") or {}).items():
            strategy, n_bins = val
            extended_keep.add(f"{col}__bins_{strategy}_{n_bins}")


def _prune_by_selected_transformed(
    transformer_dicts: list[dict],
    selected_transformed: set[str],
) -> list[dict]:
    """Prune the pipeline recipe using the RFECV-selected transformed feature names.

    Two passes:

    1. **Forward pass** over all expansion/reduction steps to compute
       ``needed_columns``: every column that must exist at some intermediate
       point — either because it appears in ``selected_transformed`` directly,
       or because it is an input to an expansion step or DimensionalityReduction
       whose output survived selection.

    2. **Pruning pass** (forward) that removes:
       - Expansion step entries (SplineTransformation, NumericalBinning,
         MathematicalOperations, CyclicalFeaturesTransformer,
         MissingValuesIndicator, DateTimeTransformer) when none of the
         generated columns are in ``selected_transformed``.
       - DimensionalityReduction when none of its output components survived.
       - In-place step entries (OutliersHandler, ScaleTransformation, etc.)
         for columns absent from ``needed_columns``.
    """
    # --- Pass 1: compute needed_columns ----------------------------------------
    needed_columns: set[str] = set(selected_transformed)

    for d in transformer_dicts:
        name = d["name"]
        params = d["params"]

        if name == "MathematicalOperations":
            for op in params.get("operations_options") or []:
                col1, col2, operation = op[0], op[1], op[2]
                generated = f"{col1}__{operation}__{col2}"
                if (
                    generated in selected_transformed
                    or f"{generated}__is_invalid" in selected_transformed
                ):
                    needed_columns.add(col1)
                    needed_columns.add(col2)

        elif name == "SplineTransformation":
            for col in params.get("transformation_options") or {}:
                if any(f.startswith(f"{col}__spline_") for f in selected_transformed):
                    needed_columns.add(col)

        elif name == "NumericalBinning":
            for col, val in (params.get("transformation_options") or {}).items():
                strategy, n_bins = val[0], val[1]
                if f"{col}__bins_{strategy}_{n_bins}" in selected_transformed:
                    needed_columns.add(col)

        elif name == "CyclicalFeaturesTransformer":
            for col in params.get("transformation_options") or {}:
                if (
                    f"{col}_sin" in selected_transformed
                    or f"{col}_cos" in selected_transformed
                ):
                    needed_columns.add(col)

        elif name == "MissingValuesIndicator":
            for col in params.get("features") or []:
                if f"{col}__is_missing" in selected_transformed:
                    needed_columns.add(col)

        elif name == "DateTimeTransformer":
            suffixes = (
                "_year",
                "_month",
                "_day",
                "_weekday",
                "_hour",
                "_minute",
                "_second",
            )
            for col in params.get("features") or []:
                if any(f"{col}{s}" in selected_transformed for s in suffixes):
                    needed_columns.add(col)

        elif name == "CategoricalEncoding":
            for col in params.get("transformation_options") or {}:
                if col in selected_transformed or any(
                    f.startswith(f"{col}_") for f in selected_transformed
                ):
                    needed_columns.add(col)

        elif name == "DimensionalityReduction":
            method = params.get("method", "")
            if any(f.startswith(f"{method}_") for f in selected_transformed):
                needed_columns.update(params.get("features") or [])

    # --- Pass 2: prune each transformer ----------------------------------------
    pruned: list[dict] = []

    for d in transformer_dicts:
        name = d["name"]
        params = dict(d["params"])

        if name == "MathematicalOperations":
            params["operations_options"] = [
                op
                for op in (params.get("operations_options") or [])
                if f"{op[0]}__{op[2]}__{op[1]}" in selected_transformed
                or f"{op[0]}__{op[2]}__{op[1]}__is_invalid" in selected_transformed
            ]
            if not params["operations_options"]:
                continue

        elif name == "SplineTransformation":
            params["transformation_options"] = {
                col: val
                for col, val in (params.get("transformation_options") or {}).items()
                if any(f.startswith(f"{col}__spline_") for f in selected_transformed)
            }
            if not params["transformation_options"]:
                continue

        elif name == "NumericalBinning":
            params["transformation_options"] = {
                col: val
                for col, val in (params.get("transformation_options") or {}).items()
                if f"{col}__bins_{val[0]}_{val[1]}" in selected_transformed
            }
            if not params["transformation_options"]:
                continue

        elif name == "CyclicalFeaturesTransformer":
            params["transformation_options"] = {
                col: period
                for col, period in (params.get("transformation_options") or {}).items()
                if (
                    f"{col}_sin" in selected_transformed
                    or f"{col}_cos" in selected_transformed
                )
            }
            if not params["transformation_options"]:
                continue

        elif name == "MissingValuesIndicator":
            params["features"] = [
                col
                for col in (params.get("features") or [])
                if f"{col}__is_missing" in selected_transformed
            ]
            if not params["features"]:
                continue

        elif name == "DateTimeTransformer":
            suffixes = (
                "_year",
                "_month",
                "_day",
                "_weekday",
                "_hour",
                "_minute",
                "_second",
            )
            params["features"] = [
                col
                for col in (params.get("features") or [])
                if any(f"{col}{s}" in selected_transformed for s in suffixes)
            ]
            if not params["features"]:
                continue

        elif name == "CategoricalEncoding":
            options = params.get("transformation_options")
            if isinstance(options, dict):
                params["transformation_options"] = {
                    col: enc_type
                    for col, enc_type in options.items()
                    if col in selected_transformed
                    or any(f.startswith(f"{col}_") for f in selected_transformed)
                }
                if not params["transformation_options"]:
                    continue

        elif name == "DimensionalityReduction":
            method = params.get("method", "")
            if not any(f.startswith(f"{method}_") for f in selected_transformed):
                continue
            params["features"] = [
                col for col in (params.get("features") or []) if col in needed_columns
            ]
            if not params["features"]:
                continue
            # Clamp n_components so it never exceeds the surviving feature count.
            n_comp = params.get("n_components")
            if n_comp is not None:
                params["n_components"] = min(n_comp, len(params["features"]))

        else:
            # In-place transformers: keep only entries for needed_columns.
            options = params.get("transformation_options")
            if isinstance(options, dict):
                params["transformation_options"] = {
                    col: val for col, val in options.items() if col in needed_columns
                }
                if not params["transformation_options"]:
                    continue

                # OutliersHandler stores extra per-column sub-dicts.
                for extra_key in ("thresholds", "iforest_params", "lof_params"):
                    extra = params.get(extra_key)
                    if isinstance(extra, dict):
                        params[extra_key] = {
                            col: val
                            for col, val in extra.items()
                            if col in needed_columns
                        }

            features = params.get("features")
            if isinstance(features, list):
                params["features"] = [col for col in features if col in needed_columns]
                if not params["features"]:
                    continue

        pruned.append({**d, "params": params})

    return pruned
