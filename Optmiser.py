"""
is given - Re, angle of attach, targeet CL
search over the cadidate flap angle and pick the one that fits the target CL with the lowest drag (CD)
"""

import numpy as np
import pandas as pd
import joblib


FEATURE_COLUMNS = ["Re", "alpha", "flap_angle"] #order is IMPORTANT


def load_model(path="rf_model.pkl"):
    """Load a trained regressor that was saved with joblib.dump()."""
    model = joblib.load(path)

    # RandomForestRegressor was trained with n_jobs=-1, which is right for
    # fitting on the full dataset -- but at inference time here we only ever
    # predict tiny batches (1-61 rows per call). Measured directly: spinning
    # up joblib's thread pool on every .predict() call costs MORE than the
    # parallelism saves at this batch size (~27ms vs ~10ms per call for a
    # 61-row sweep). Force single-threaded prediction instead.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    return model


def predict_cl_cd(model, Re: float, alpha: float, flap_angle: float) -> tuple[float, float]:
    """
    Wraps model.predict for a single (Re, alpha, flap_angle) input.
    Returns (cl, cd) as plain floats, not arrays.
    """
    # Wrap the single row in a DataFrame (not a bare list) so the column
    # names match training -- avoids sklearn's "X has feature names" warning
    # and protects against accidentally swapping the column order.
    X = pd.DataFrame(
        [[Re, alpha, flap_angle]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(X)  # shape (1, 2): columns are [Cl, Cd]
    cl, cd = prediction[0]

    return float(cl), float(cd)


def find_best_flap_angle(model,target_cl: float,Re: float,alpha: float,flap_range=np.arange(-15, 15.5, 0.5),cl_tolerance: float = 0.05) -> dict:
    """
    For every candidate flap_angle in flap_range, predict (cl, cd) at the
    given (Re, alpha) and check how close cl lands to target_cl.

    Among the candidates within `cl_tolerance` of target_cl, returns the one
    with the lowest cd -- i.e. the least-drag way of achieving the required
    lift at this flight condition.

    If NONE of the candidates reach target_cl within tolerance (possible at
    low Re / extreme alpha, where the flap range simply can't produce enough
    lift), falls back to the candidate that gets closest to target_cl, and
    flags that the target wasn't actually reachable.

    Returns:
        {
            "flap_angle": best flap angle found (degrees),
            "cl": predicted Cl at that flap angle,
            "cd": predicted Cd at that flap angle,
            "cl_error": abs(cl - target_cl),
            "reachable": True if target_cl was hit within cl_tolerance, else False
        }
    """
    flap_range = np.asarray(flap_range)

    X_candidates = pd.DataFrame({
        "Re": Re,
        "alpha": alpha,
        "flap_angle": flap_range
    })[FEATURE_COLUMNS]

    predictions = model.predict(X_candidates)  # shape (n_candidates, 2)
    cl_values = predictions[:, 0] 
    cd_values = predictions[:, 1] #:, 1 or 0 allows to seperate cl, cd values from the predictions array
    cl_errors = np.abs(cl_values - target_cl)
    within_tolerance = cl_errors <= cl_tolerance
    if np.any(within_tolerance): # Among the candidates that hit the lift target, pick the lowest-drag one.
        masked_cd = np.where(within_tolerance, cd_values, np.inf) # to Mask out the Cd values that are not within tolerance by setting them to infinity, so they won't be considered in the argmin.
        best_index = int(np.argmin(masked_cd))
        reachable = True
    else: # target_cl isn't achievable anywhere in flap_range at this Re/alpha -  fall back to whichever flap angle gets closest to it.
        best_index = int(np.argmin(cl_errors))
        reachable = False
    return {
        "flap_angle": float(flap_range[best_index]),
        "cl": float(cl_values[best_index]),
        "cd": float(cd_values[best_index]),
        "cl_error": float(cl_errors[best_index]),
        "reachable": reachable
    }


def main():

    model = load_model()

    # Example: at Re=100,000 and alpha=4 deg, find the flap deflection that
    # delivers Cl = 0.6 with the least possible drag.
    result = find_best_flap_angle(
        model,
        target_cl=0.6,
        Re=100_000,
        alpha=4.0
    )

    print("Best flap angle found:")
    print(f"  flap_angle          = {result['flap_angle']:.2f} deg")
    print(f"  predicted Cl        = {result['cl']:.4f}")
    print(f"  predicted Cd        = {result['cd']:.5f}")
    print(f"  |cl - target_cl|    = {result['cl_error']:.4f}")
    print(f"  target reachable?   = {result['reachable']}")


if __name__ == "__main__":
    main()

