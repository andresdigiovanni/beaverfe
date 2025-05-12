from sklearn.inspection import permutation_importance


def feature_importance(model, x, y):
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        feature_importances = model.coef_

    else:
        result = permutation_importance(model, x, y, n_repeats=5, random_state=42)
        feature_importances = result.importances_mean

    return feature_importances
