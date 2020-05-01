from typing import Dict

import pandas as pd
from src.models.similarity import rv_coefficient, rbig_it_measures


def get_similarity_scores(
    X_ref: pd.DataFrame, Y_compare: pd.DataFrame, smoke_test: bool = False
) -> Dict:

    if smoke_test is True:
        X_ref = X_ref[:500]
        Y_compare = Y_compare[:500]

    # RV Coefficient
    rv_results = rv_coefficient(X_ref, Y_compare)

    # RBIG Coefficient
    rbig_results = rbig_it_measures(X_ref, Y_compare)

    results = {
        **rv_results,
        **rbig_results,
    }

    return results
