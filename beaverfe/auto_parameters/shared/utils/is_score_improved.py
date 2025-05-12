def is_score_improved(score, best_score, direction):
    return (direction == "maximize" and score > best_score) or (
        direction == "minimize" and score < best_score
    )
