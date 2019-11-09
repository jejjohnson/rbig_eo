import sys

sys.path.insert(0, "/home/emmanuel/code/rbig")

from rbig import RBIG
from src.models.information.entropy import RBIGEstimator as RBIGENT
from src.models.information.mutual_information import RBIGEstimator as RBIGMI

import numpy as np
import time


def run_rbig_models(
    X1, X2=None, measure="t", verbose=None, random_state=123, batch_size=None
):

    # RBIG Parameters
    n_layers = 10000
    rotation_type = "PCA"
    zero_tolerance = 60
    pdf_extension = 10
    pdf_resolution = None
    tolerance = None

    if measure.lower() == "t":
        # RBIG MODEL 0
        rbig_tc_model = RBIGMI(
            n_layers=n_layers,
            rotation_type=rotation_type,
            random_state=random_state,
            zero_tolerance=zero_tolerance,
            tolerance=tolerance,
            pdf_extension=pdf_extension,
            pdf_resolution=pdf_resolution,
            verbose=None,
            batch_size=batch_size,
        )

        # fit model to the data
        t0 = time.time()
        rbig_tc_model.fit(X1)
        t1 = time.time() - t0

        if verbose:
            print(
                f"Trained RBIG TC ({X1.shape[0]:,} points, {X1.shape[1]:,} dimensions): {t1:.3f} secs"
            )

        tc = rbig_tc_model.score(X1)
        if verbose:
            print(f"TC: {tc:.3f}")

        return tc, t1

    elif measure.lower() == "h":
        # RBIG MODEL 0
        rbig_h_model = RBIGENT(
            n_layers=n_layers,
            rotation_type=rotation_type,
            random_state=random_state,
            zero_tolerance=zero_tolerance,
            tolerance=tolerance,
            pdf_extension=pdf_extension,
            pdf_resolution=pdf_resolution,
            verbose=None,
            batch_size=batch_size,
        )

        # fit model to the data
        t0 = time.time()
        rbig_h_model.fit(X1, X2)
        t1 = time.time() - t0

        if verbose:
            print(
                f"Trained RBIG ({X1.shape[0]:,} points, {X1.shape[1]:,} dimensions): {t1:.3f} secs"
            )

        h = rbig_h_model.score(X1)
        if verbose:
            print(f"H: {h:.3f}")

        return h, t1

    elif measure.lower() == "mi":
        # RBIG MODEL 0
        rbig_mi_model = RBIGMI(
            n_layers=n_layers,
            rotation_type=rotation_type,
            random_state=random_state,
            zero_tolerance=zero_tolerance,
            tolerance=tolerance,
            pdf_extension=pdf_extension,
            pdf_resolution=pdf_resolution,
            verbose=None,
            batch_size=batch_size,
        )

        # fit model to the data
        t0 = time.time()
        rbig_mi_model.fit(X1, X2)
        t1 = time.time() - t0

        if verbose:
            print(
                f"Trained RBIG1 MI ({X1.shape[0]:,} points, {X1.shape[1]:,} dimensions): {t1:.3f} secs"
            )

        mi = rbig_mi_model.score(X1)
        if verbose:
            print(f"MI: {mi:.3f}")

        return mi, t1

    else:
        raise ValueError(f"Unrecognized measure: {measure}.")
