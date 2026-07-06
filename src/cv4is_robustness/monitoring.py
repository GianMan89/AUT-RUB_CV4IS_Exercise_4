from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def ks_statistic(reference_values, current_values) -> float:
    return float(
        ks_2samp(
            np.asarray(reference_values),
            np.asarray(current_values),
            alternative="two-sided",
            mode="auto",
        ).statistic
    )
