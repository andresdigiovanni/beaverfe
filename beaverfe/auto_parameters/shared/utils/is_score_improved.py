def is_score_improved(score, best_score, direction, tol=0):
    return (direction == "maximize" and score > best_score + tol) or (
        direction == "minimize" and score < best_score - tol
    )
