import numpy as np


def weighted_epic_score(p_epic, p_medium, p_thin):
    p_max = max(p_epic, p_medium, p_thin)

    if p_max < 0.40:
        return 40.0

    if p_thin == p_max:
        score = p_thin * 8 + p_epic * 100 + p_medium * 50
        if p_max < 0.7:
            score *= 0.92
    else:
        epic_medium_sum = p_epic + p_medium
        effective_anchor = (
            (p_epic * 100 + p_medium * 75) / epic_medium_sum
            if epic_medium_sum > 0 else 100
        )
        score = (p_epic ** 1.1) * 100 + p_medium * 75 - p_thin * effective_anchor * 0.5
        if p_thin > 0.05:
            score *= 0.92

    score = float(np.clip(score, 0, 100))
    score = 100 * (score / 100) ** 0.85
    return float(np.clip(score, 0, 100))


def compress_top_end(score, floor=95.0, new_floor=85.0, gamma=2.5):
    if score < floor:
        return score
    frac = (score - floor) / (100.0 - floor)
    curved = frac ** gamma
    return new_floor + (100.0 - new_floor) * curved


def classify_epicness(score):
    if score >= 95:
        return "Legendarisk mustasch"
    elif score >= 80:
        return "Episk mustasch"
    elif score >= 60:
        return "Respektabel mustasch"
    elif score >= 25:
        return "Lovande mustasch"
    else:
        return "Fjunig mustasch"
