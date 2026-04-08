import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import Image as _IPyImage, display as _ipython_display

    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

from simulationTools import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    SIGMA_TO_FWHM,
    analyze_psf_image,
    compute_curve_of_growth,
    compute_ee_radius,
    compute_fwhm_from_profile,
    compute_second_moment_shape,
    count_best_models,
    fit_double_gaussian_image as _fit_double_gaussian_image,
    fit_elliptical_double_gaussian_image as _fit_elliptical_double_gaussian_image,
    fit_gauss_hermite as _fit_gauss_hermite,
    fit_gauss_hermite_image as _fit_gauss_hermite_image,
    fit_gaussian_image as _fit_gaussian_image,
    fit_moffat_image as _fit_moffat_image,
    fit_shapelet_image as _fit_shapelet_image,
    fit_shapelets as _fit_shapelets,
    image_center,
    radial_profile,
)

_PLOT_DPI = 72
SCIENCE_BAD_STAR_Z_THRESHOLD = 2.5
DETECTOR_FAILURE_FRACTION_THRESHOLD = 0.25
HIGH_NG_SCORE_THRESHOLD = 0.5
HIGH_EDG_DEVIATION_SCORE_THRESHOLD = 0.15


############################################
# 1. Thin wrappers around simulationTools
############################################


def fit_gaussian_image(*args, **kwargs):
    """Reuse the single-Gaussian fit helper from simulationTools."""
    return _fit_gaussian_image(*args, **kwargs)


def fit_double_gaussian_image(*args, **kwargs):
    """Reuse the Double Gaussian fit helper from simulationTools."""
    return _fit_double_gaussian_image(*args, **kwargs)


def fit_elliptical_double_gaussian_image(*args, **kwargs):
    """Reuse the EDG fit helper from simulationTools."""
    return _fit_elliptical_double_gaussian_image(*args, **kwargs)


def fit_gauss_hermite(image, center=None, **kwargs):
    """Reuse the Gauss-Hermite parameter fit helper from simulationTools."""
    return _fit_gauss_hermite(image, center=center, **kwargs)


def fit_gauss_hermite_image(*args, **kwargs):
    """Reuse the Gauss-Hermite image fit helper from simulationTools."""
    return _fit_gauss_hermite_image(*args, **kwargs)


def fit_moffat_image(*args, **kwargs):
    """Reuse the Moffat fit helper from simulationTools."""
    return _fit_moffat_image(*args, **kwargs)


def fit_shapelets(image, beta=2.0, nmax=6, center=None):
    """Reuse the shapelet coefficient fit helper from simulationTools."""
    return _fit_shapelets(image, beta=beta, nmax=nmax, center=center)


def fit_shapelet_image(*args, **kwargs):
    """Reuse the shapelet image fit helper from simulationTools."""
    return _fit_shapelet_image(*args, **kwargs)


def summarize_coefficients(coeffs, modes):
    """Group shapelet coefficients by total order."""
    summary = {}
    for coeff, (n1, n2) in zip(coeffs, modes):
        order = n1 + n2
        summary.setdefault(order, []).append(coeff)
    for order in summary:
        summary[order] = np.asarray(summary[order])
    return summary


############################################
# 2. Observed-stamp preprocessing
############################################


def validate_stamp(image, min_size=7, min_finite_fraction=0.95):
    """Validate that a cutout is usable for PSF analysis."""
    if image is None:
        return {"valid": False, "message": "stamp is None"}

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        return {"valid": False, "message": "stamp must be a 2D array"}

    ny, nx = image.shape
    if min(ny, nx) < min_size:
        return {"valid": False, "message": "stamp is too small"}

    finite_mask = np.isfinite(image)
    finite_fraction = float(np.mean(finite_mask))
    if finite_fraction < min_finite_fraction:
        return {
            "valid": False,
            "message": f"finite fraction too low ({finite_fraction:.2f})",
        }

    finite_values = image[finite_mask]
    if finite_values.size == 0:
        return {"valid": False, "message": "stamp has no finite pixels"}

    if np.nanmax(np.abs(finite_values)) == 0:
        return {"valid": False, "message": "stamp is all zeros"}

    return {
        "valid": True,
        "message": "stamp validation passed",
        "finite_fraction": finite_fraction,
        "shape": image.shape,
    }


def estimate_background_from_border(image, border_width=2):
    """Estimate local background from the cutout border using the median."""
    image = np.asarray(image, dtype=float)
    if border_width <= 0:
        return 0.0

    ny, nx = image.shape
    mask = np.zeros_like(image, dtype=bool)
    mask[:border_width, :] = True
    mask[-border_width:, :] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True

    border_values = image[mask & np.isfinite(image)]
    if border_values.size == 0:
        return 0.0

    return float(np.median(border_values))


def subtract_local_background(image, background=None, border_width=2):
    """Subtract a scalar local background estimate from a cutout."""
    image = np.asarray(image, dtype=float)
    if background is None:
        background = estimate_background_from_border(image, border_width=border_width)
    return image - background, float(background)


def normalize_stamp_flux(image, positive_only=True):
    """
    Normalize a cutout by its positive flux.

    This keeps the normalization stable after local background subtraction while
    preserving negative residual pixels in the normalized stamp.
    """
    image = np.asarray(image, dtype=float)
    if positive_only:
        flux = float(np.sum(np.clip(np.where(np.isfinite(image), image, 0.0), 0.0, None)))
    else:
        flux = float(np.nansum(image))

    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("stamp has non-positive flux after preprocessing")

    return image / flux, flux


def prepare_observed_star_stamp(
    stamp,
    center=None,
    border_width=2,
    normalize=True,
    positive_only=True,
):
    """Validate, background-subtract, and flux-normalize one observed star cutout."""
    raw_stamp = np.asarray(stamp, dtype=float)
    validation = validate_stamp(raw_stamp)
    if not validation["valid"]:
        return {
            "valid": False,
            "message": validation["message"],
            "raw_stamp": raw_stamp,
            "prepared_stamp": None,
            "background": np.nan,
            "positive_flux": np.nan,
            "center": center if center is not None else image_center(raw_stamp),
        }

    bgsub_stamp, background = subtract_local_background(raw_stamp, border_width=border_width)
    bg_validation = validate_stamp(bgsub_stamp, min_size=1, min_finite_fraction=0.95)
    if not bg_validation["valid"]:
        return {
            "valid": False,
            "message": f"background-subtracted stamp invalid: {bg_validation['message']}",
            "raw_stamp": raw_stamp,
            "prepared_stamp": None,
            "background": background,
            "positive_flux": np.nan,
            "center": center if center is not None else image_center(raw_stamp),
        }

    try:
        prepared_stamp, positive_flux = normalize_stamp_flux(bgsub_stamp, positive_only=positive_only) if normalize else (bgsub_stamp, np.nan)
    except ValueError as exc:
        return {
            "valid": False,
            "message": str(exc),
            "raw_stamp": raw_stamp,
            "prepared_stamp": None,
            "background": background,
            "positive_flux": np.nan,
            "center": center if center is not None else image_center(raw_stamp),
        }

    if center is None:
        center = image_center(prepared_stamp)

    return {
        "valid": True,
        "message": "prepared observed star stamp",
        "raw_stamp": raw_stamp,
        "bgsub_stamp": bgsub_stamp,
        "prepared_stamp": prepared_stamp,
        "background": background,
        "positive_flux": positive_flux,
        "center": center,
    }


############################################
# 3. Observed quick metrics
############################################


def compute_observed_star_metrics(image, center=None, total_flux=None):
    """Compute simple observed-star metrics on a prepared stamp."""
    image = np.asarray(image, dtype=float)
    if center is None:
        center = image_center(image)

    shape = compute_second_moment_shape(image, center=center)
    metrics = {
        "peak_value": float(np.nanmax(image)) if np.any(np.isfinite(image)) else np.nan,
        "total_flux": float(np.nansum(image)) if total_flux is None else float(total_flux),
        "fwhm_proxy": compute_fwhm_from_profile(image, center=center),
        "ee80_radius": compute_ee_radius(image, frac=0.8, center=center),
        "e1": shape["e1"],
        "e2": shape["e2"],
        "ellipticity": shape["ellipticity"],
        "Mxx": shape["Mxx"],
        "Myy": shape["Myy"],
        "Mxy": shape["Mxy"],
        "determinant_radius": shape["determinant_radius"],
    }
    return metrics


############################################
# 4. Analysis entry points
############################################


def _empty_model_analysis(center):
    """Return a failed-analysis shell for unusable observed cutouts."""
    result = {
        "center": center,
        "psf_array": None,
        "peak_value": np.nan,
        "total_flux": np.nan,
        "curve_of_growth_radius": np.array([], dtype=float),
        "curve_of_growth": np.array([], dtype=float),
        "ee80_radius": np.nan,
        "fwhm_proxy": np.nan,
        "shape_metrics": {},
        "Mxx": np.nan,
        "Myy": np.nan,
        "Mxy": np.nan,
        "e1": np.nan,
        "e2": np.nan,
        "ellipticity": np.nan,
        "determinant_radius": np.nan,
        "rp_radius": np.array([], dtype=float),
        "rp_psf": np.array([], dtype=float),
        "best_model": "none",
        "chi2_map": {},
        "ng_score": np.nan,
        "edg_deviation_score": np.nan,
        "best_non_edg_chi2": np.nan,
        "delta_edg_vs_dg": np.nan,
        "delta_edg_vs_moffat": np.nan,
        "delta_edg_vs_gh": np.nan,
        "delta_edg_vs_shapelet": np.nan,
    }

    for prefix, array_key in [
        ("gaussian", "gaussian_array"),
        ("dg", "dgauss_array"),
        ("edg", "edg_array"),
        ("gh", "gh_array"),
        ("moffat", "moffat_array"),
        ("shapelet", "shapelet_array"),
    ]:
        result[f"{prefix}_chi2"] = np.nan
        result[f"{prefix}_success"] = False
        result[f"{prefix}_message"] = "analysis skipped"
        result[f"{prefix}_fit_valid"] = False
        result[f"{prefix}_global_mse"] = np.nan
        result[f"{prefix}_core_mse"] = np.nan
        result[f"{prefix}_wing_mse"] = np.nan
        result[f"{prefix}_profile_mse"] = np.nan
        result[f"{prefix}_metrics"] = {}
        result[f"{prefix}_residual"] = None
        result[f"{prefix}_params"] = np.array([])
        result[f"{prefix}_params_named"] = {}
        result[f"rp_{prefix}"] = np.array([], dtype=float)
        result[f"rp_{prefix}_residual"] = np.array([], dtype=float)
        result[array_key] = None
    result["shapelet_result"] = None
    return result


def analyze_observed_star(
    stamp,
    x=None,
    y=None,
    visit=None,
    detector=None,
    band=None,
    day_obs=None,
    center=None,
    border_width=2,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """
    Preprocess one observed star cutout, compute quick metrics, and run model fits.
    """
    prep = prepare_observed_star_stamp(
        stamp,
        center=center,
        border_width=border_width,
        normalize=True,
        positive_only=True,
    )

    metadata = {
        "x": x,
        "y": y,
        "visit": visit,
        "detector": detector,
        "band": band,
        "day_obs": day_obs,
        "background_estimate": prep["background"],
        "preprocess_valid": prep["valid"],
        "preprocess_message": prep["message"],
        "analysis_valid": False,
        "raw_stamp": prep["raw_stamp"],
        "prepared_stamp": prep["prepared_stamp"],
    }

    if not prep["valid"]:
        result = _empty_model_analysis(prep["center"])
        result.update(metadata)
        return result

    quick_metrics = compute_observed_star_metrics(
        prep["prepared_stamp"],
        center=prep["center"],
        total_flux=prep["positive_flux"],
    )
    sigma_hint = (
        quick_metrics["fwhm_proxy"] / SIGMA_TO_FWHM
        if np.isfinite(quick_metrics["fwhm_proxy"])
        else None
    )

    result = analyze_psf_image(
        prep["prepared_stamp"],
        sigma_hint=sigma_hint,
        center=prep["center"],
        shapelet_beta=shapelet_beta,
        shapelet_nmax=shapelet_nmax,
        fit_center=fit_center,
        fit_background=fit_background,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )

    result.update(metadata)
    result.update(quick_metrics)
    result["analysis_valid"] = True
    result["positive_flux"] = prep["positive_flux"]
    result["fwhm_hint"] = sigma_hint * SIGMA_TO_FWHM if sigma_hint is not None else np.nan
    return result


def analyze_psf_models(
    image,
    x=None,
    y=None,
    fwhm=None,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    center=None,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """
    Backward-compatible wrapper for notebook calls on already-prepared 2D stamps.
    """
    image = np.asarray(image, dtype=float)
    if center is None:
        center = image_center(image)

    sigma_hint = fwhm / SIGMA_TO_FWHM if fwhm is not None and np.isfinite(fwhm) else None
    result = analyze_psf_image(
        image,
        sigma_hint=sigma_hint,
        center=center,
        shapelet_beta=shapelet_beta,
        shapelet_nmax=shapelet_nmax,
        fit_center=fit_center,
        fit_background=fit_background,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )
    result["x"] = x
    result["y"] = y
    result["fwhm"] = fwhm
    result["max_val"] = float(np.nanmax(np.abs(image))) if np.any(np.isfinite(image)) else np.nan
    result["analysis_valid"] = True
    return result


############################################
# 5. Tabular summaries and badness scoring
############################################


def _named_param(params_named, key):
    """Return one scalar named fit parameter or NaN when unavailable."""
    if not params_named:
        return np.nan
    value = params_named.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _shapelet_order_norm_summary(shapelet_result, max_order=4):
    """Summarize shapelet coefficients by total order with compact L2 norms."""
    summary = {f"shapelet_order{order}_norm": np.nan for order in range(max_order + 1)}
    summary["shapelet_coeff_l2_total"] = np.nan

    if not shapelet_result:
        return summary

    coeffs = np.asarray(shapelet_result.get("coeffs", []), dtype=float)
    modes = shapelet_result.get("modes", [])
    if coeffs.size == 0 or len(modes) == 0 or len(coeffs) != len(modes):
        return summary

    summary["shapelet_coeff_l2_total"] = float(np.linalg.norm(coeffs))
    order_groups = summarize_coefficients(coeffs, modes)
    for order in range(max_order + 1):
        coeff_group = order_groups.get(order)
        summary[f"shapelet_order{order}_norm"] = (
            float(np.linalg.norm(coeff_group)) if coeff_group is not None else 0.0
        )
    return summary


def _flatten_model_parameter_columns(result_dict):
    """Flatten per-model scalar parameters for the star master table."""
    gaussian_params = result_dict.get("gaussian_params_named", {}) or {}
    dg_params = result_dict.get("dg_params_named", {}) or {}
    edg_params = result_dict.get("edg_params_named", {}) or {}
    gh_params = result_dict.get("gh_params_named", {}) or {}
    moffat_params = result_dict.get("moffat_params_named", {}) or {}
    shapelet_params = result_dict.get("shapelet_params_named", {}) or {}

    return {
        "gaussian_A": _named_param(gaussian_params, "A"),
        "gaussian_sigma": _named_param(gaussian_params, "sigma"),
        "gaussian_x0": _named_param(gaussian_params, "x0"),
        "gaussian_y0": _named_param(gaussian_params, "y0"),
        "gaussian_B": _named_param(gaussian_params, "B"),
        "dg_a1": _named_param(dg_params, "a1"),
        "dg_sigma1": _named_param(dg_params, "sigma1"),
        "dg_a2": _named_param(dg_params, "a2"),
        "dg_sigma2": _named_param(dg_params, "sigma2"),
        "edg_A1": _named_param(edg_params, "A1"),
        "edg_sigma1_x": _named_param(edg_params, "sigma1_x"),
        "edg_sigma1_y": _named_param(edg_params, "sigma1_y"),
        "edg_A2": _named_param(edg_params, "A2"),
        "edg_sigma2_x": _named_param(edg_params, "sigma2_x"),
        "edg_sigma2_y": _named_param(edg_params, "sigma2_y"),
        "edg_theta": _named_param(edg_params, "theta"),
        "edg_x0": _named_param(edg_params, "x0"),
        "edg_y0": _named_param(edg_params, "y0"),
        "edg_B": _named_param(edg_params, "B"),
        "gh_A": _named_param(gh_params, "A"),
        "gh_x0": _named_param(gh_params, "x0"),
        "gh_y0": _named_param(gh_params, "y0"),
        "gh_sigma_x": _named_param(gh_params, "sigma_x"),
        "gh_sigma_y": _named_param(gh_params, "sigma_y"),
        "gh_h3x": _named_param(gh_params, "h3x"),
        "gh_h4x": _named_param(gh_params, "h4x"),
        "gh_h3y": _named_param(gh_params, "h3y"),
        "gh_h4y": _named_param(gh_params, "h4y"),
        "gh_B": _named_param(gh_params, "B"),
        "moffat_A": _named_param(moffat_params, "A"),
        "moffat_x0": _named_param(moffat_params, "x0"),
        "moffat_y0": _named_param(moffat_params, "y0"),
        "moffat_alpha": _named_param(moffat_params, "alpha"),
        "moffat_beta": _named_param(moffat_params, "beta"),
        "moffat_B": _named_param(moffat_params, "B"),
        "shapelet_beta": _named_param(shapelet_params, "beta"),
        "shapelet_nmax": _named_param(shapelet_params, "nmax"),
    }


def build_star_summary_row(result_dict):
    """Build one row for a star-level metrics table."""
    row = {
        "sourceId": result_dict.get("sourceId"),
        "visit": result_dict.get("visit"),
        "detector": result_dict.get("detector"),
        "band": result_dict.get("band"),
        "day_obs": result_dict.get("day_obs"),
        "x": result_dict.get("x"),
        "y": result_dict.get("y"),
        "peak_value": result_dict.get("peak_value"),
        "total_flux": result_dict.get("total_flux"),
        "fwhm_proxy": result_dict.get("fwhm_proxy"),
        "ee80_radius": result_dict.get("ee80_radius"),
        "e1": result_dict.get("e1"),
        "e2": result_dict.get("e2"),
        "ellipticity": result_dict.get("ellipticity"),
        "gaussian_chi2": result_dict.get("gaussian_chi2"),
        "dg_chi2": result_dict.get("dg_chi2"),
        "edg_chi2": result_dict.get("edg_chi2"),
        "gh_chi2": result_dict.get("gh_chi2"),
        "moffat_chi2": result_dict.get("moffat_chi2"),
        "shapelet_chi2": result_dict.get("shapelet_chi2"),
        "gaussian_core_mse": result_dict.get("gaussian_core_mse"),
        "gaussian_wing_mse": result_dict.get("gaussian_wing_mse"),
        "gaussian_profile_mse": result_dict.get("gaussian_profile_mse"),
        "dg_core_mse": result_dict.get("dg_core_mse"),
        "dg_wing_mse": result_dict.get("dg_wing_mse"),
        "dg_profile_mse": result_dict.get("dg_profile_mse"),
        "edg_core_mse": result_dict.get("edg_core_mse"),
        "edg_wing_mse": result_dict.get("edg_wing_mse"),
        "edg_profile_mse": result_dict.get("edg_profile_mse"),
        "moffat_core_mse": result_dict.get("moffat_core_mse"),
        "moffat_wing_mse": result_dict.get("moffat_wing_mse"),
        "moffat_profile_mse": result_dict.get("moffat_profile_mse"),
        "delta_gaussian_vs_dg": result_dict.get("delta_gaussian_vs_dg"),
        "delta_gaussian_vs_moffat": result_dict.get("delta_gaussian_vs_moffat"),
        "delta_edg_vs_dg": result_dict.get("delta_edg_vs_dg"),
        "delta_edg_vs_moffat": result_dict.get("delta_edg_vs_moffat"),
        "delta_edg_vs_gh": result_dict.get("delta_edg_vs_gh"),
        "delta_edg_vs_shapelet": result_dict.get("delta_edg_vs_shapelet"),
        "ng_score": result_dict.get("ng_score"),
        "edg_deviation_score": result_dict.get("edg_deviation_score"),
        "best_model": result_dict.get("best_model"),
        "gaussian_fit_valid": result_dict.get("gaussian_fit_valid"),
        "dg_fit_valid": result_dict.get("dg_fit_valid"),
        "edg_fit_valid": result_dict.get("edg_fit_valid"),
        "gh_fit_valid": result_dict.get("gh_fit_valid"),
        "moffat_fit_valid": result_dict.get("moffat_fit_valid"),
        "shapelet_fit_valid": result_dict.get("shapelet_fit_valid"),
        "gaussian_message": result_dict.get("gaussian_message"),
        "dg_message": result_dict.get("dg_message"),
        "edg_message": result_dict.get("edg_message"),
        "gh_message": result_dict.get("gh_message"),
        "moffat_message": result_dict.get("moffat_message"),
        "shapelet_message": result_dict.get("shapelet_message"),
        "preprocess_valid": result_dict.get("preprocess_valid", True),
        "analysis_valid": result_dict.get("analysis_valid", True),
        "failed_star_flag": result_dict.get("failed_star_flag"),
        "science_bad_star_flag": result_dict.get("science_bad_star_flag"),
        "high_ng_star_flag": result_dict.get("high_ng_star_flag"),
        "high_edg_deviation_star_flag": result_dict.get("high_edg_deviation_star_flag"),
        "background_estimate": result_dict.get("background_estimate"),
    }

    row.update(_flatten_model_parameter_columns(result_dict))
    row.update(_shapelet_order_norm_summary(result_dict.get("shapelet_result")))
    return row


def build_shapelet_coefficients_long_table(results_or_star_rows):
    """
    Build a long-format shapelet coefficient table from analysis results.

    The intended input is the raw `analysis_results` list from the notebook so
    the full coefficient vectors can live in a sidecar table instead of the
    master star table.
    """
    if isinstance(results_or_star_rows, pd.DataFrame):
        records = results_or_star_rows.to_dict("records")
    else:
        records = list(results_or_star_rows or [])

    rows = []
    for record in records:
        shapelet_result = record.get("shapelet_result")
        if not shapelet_result:
            continue

        coeffs = np.asarray(shapelet_result.get("coeffs", []), dtype=float)
        modes = shapelet_result.get("modes", [])
        if coeffs.size == 0 or len(modes) == 0:
            continue

        for mode_index, ((n1, n2), coeff) in enumerate(zip(modes, coeffs)):
            rows.append(
                {
                    "sourceId": record.get("sourceId"),
                    "visit": record.get("visit"),
                    "detector": record.get("detector"),
                    "band": record.get("band"),
                    "day_obs": record.get("day_obs"),
                    "mode_index": int(mode_index),
                    "n1": int(n1),
                    "n2": int(n2),
                    "coeff": float(coeff),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "sourceId",
            "visit",
            "detector",
            "band",
            "day_obs",
            "mode_index",
            "n1",
            "n2",
            "coeff",
        ],
    )


def compute_robust_zscore(series):
    """Compute a robust z-score using the median and MAD."""
    s = pd.Series(series, dtype=float)
    median = s.median(skipna=True)
    mad = np.median(np.abs(s.dropna() - median)) if s.notna().any() else np.nan

    if not np.isfinite(mad) or mad == 0:
        std = s.std(ddof=0, skipna=True)
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        return (s - median) / std

    return 0.6744897501960817 * (s - median) / mad


def _positive_robust_excursion(series):
    """Return one-sided robust excursions where only unusually large values count as bad."""
    return np.clip(compute_robust_zscore(series), 0, None)


def annotate_star_quality(
    star_df,
    science_bad_z_threshold=SCIENCE_BAD_STAR_Z_THRESHOLD,
    high_ng_score_threshold=HIGH_NG_SCORE_THRESHOLD,
    high_edg_deviation_score_threshold=HIGH_EDG_DEVIATION_SCORE_THRESHOLD,
):
    """
    Add star-level failure and science-bad flags using one-sided robust excursions.

    The science-side flag is intentionally separate from pipeline failure so later
    detector/visit badness can remain physically interpretable.
    """
    out = star_df.copy()
    if out.empty:
        out["failed_star_flag"] = pd.Series(dtype=bool)
        out["science_bad_star_flag"] = pd.Series(dtype=bool)
        out["high_ng_star_flag"] = pd.Series(dtype=bool)
        out["high_edg_deviation_star_flag"] = pd.Series(dtype=bool)
        return out

    preprocess_ok = out.get("preprocess_valid", pd.Series(True, index=out.index)).fillna(False)
    analysis_ok = out.get("analysis_valid", pd.Series(True, index=out.index)).fillna(False)
    best_model = out.get("best_model", pd.Series("none", index=out.index)).fillna("none")

    out["failed_star_flag"] = (~preprocess_ok) | (~analysis_ok) | (best_model == "none")
    out["high_ng_star_flag"] = (
        out.get("ng_score", pd.Series(np.nan, index=out.index)).fillna(-np.inf) >= high_ng_score_threshold
    ) & (~out["failed_star_flag"])
    out["high_edg_deviation_star_flag"] = (
        out.get("edg_deviation_score", pd.Series(np.nan, index=out.index)).fillna(-np.inf)
        >= high_edg_deviation_score_threshold
    ) & (~out["failed_star_flag"])

    metric_map = {
        "fwhm_proxy": "fwhm_proxy_bad_z",
        "ee80_radius": "ee80_radius_bad_z",
        "ellipticity": "ellipticity_bad_z",
        "ng_score": "ng_score_bad_z",
    }
    for src_col, z_col in metric_map.items():
        if src_col in out.columns:
            z = _positive_robust_excursion(out[src_col])
            z[out["failed_star_flag"]] = np.nan
            out[z_col] = z
        else:
            out[z_col] = np.nan

    if "edg_deviation_score" in out.columns:
        z = _positive_robust_excursion(out["edg_deviation_score"])
        z[out["failed_star_flag"]] = np.nan
        out["edg_deviation_score_bad_z"] = z
    else:
        out["edg_deviation_score_bad_z"] = np.nan

    bad_z_cols = list(metric_map.values())
    science_bad = pd.Series(False, index=out.index)
    for z_col in bad_z_cols:
        science_bad |= out[z_col].fillna(0) >= science_bad_z_threshold
    out["science_bad_star_flag"] = science_bad & (~out["failed_star_flag"])

    return out


def compute_visit_badness(df, score_col=None):
    """
    Compute a practical first-pass badness score using robust excursions.

    The function works on either detector-level or visit-level summary tables.
    """
    out = df.copy()

    if score_col is None:
        score_col = "detector_badness" if "detector" in out.columns else "visit_badness"

    size_cols = [col for col in ["median_fwhm_proxy", "median_ee80_radius", "median_detector_fwhm", "median_detector_ee80"] if col in out.columns]
    shape_cols = [col for col in ["median_ellipticity", "median_detector_ellipticity"] if col in out.columns]
    ng_cols = [col for col in ["median_ng_score", "median_detector_ng_score"] if col in out.columns]
    edg_cols = [col for col in ["median_edg_deviation_score"] if col in out.columns]
    spatial_cols = [col for col in ["detector_fwhm_spread", "detector_ellipticity_spread"] if col in out.columns]
    failure_cols = [col for col in ["frac_failed_stars", "frac_failed_detectors"] if col in out.columns]

    if size_cols:
        z = np.column_stack([_positive_robust_excursion(out[col]).to_numpy() for col in size_cols])
        out["size_excursion"] = np.nanmean(z, axis=1)
    else:
        out["size_excursion"] = 0.0

    if shape_cols:
        z = np.column_stack([_positive_robust_excursion(out[col]).to_numpy() for col in shape_cols])
        out["shape_excursion"] = np.nanmean(z, axis=1)
    else:
        out["shape_excursion"] = 0.0

    if ng_cols:
        z = np.column_stack([_positive_robust_excursion(out[col]).to_numpy() for col in ng_cols])
        out["non_gaussian_excursion"] = np.nanmean(z, axis=1)
    else:
        out["non_gaussian_excursion"] = 0.0

    if edg_cols:
        z = np.column_stack([_positive_robust_excursion(out[col]).to_numpy() for col in edg_cols])
        out["edg_deviation_excursion"] = np.nanmean(z, axis=1)
    else:
        out["edg_deviation_excursion"] = 0.0

    if spatial_cols:
        z = np.column_stack([_positive_robust_excursion(out[col]).to_numpy() for col in spatial_cols])
        out["spatial_nonuniformity"] = np.nanmean(z, axis=1)
    else:
        out["spatial_nonuniformity"] = 0.0

    if failure_cols:
        z = np.column_stack([_positive_robust_excursion(out[col]).to_numpy() for col in failure_cols])
        out["qa_failure_excursion"] = np.nanmean(z, axis=1)
    else:
        out["qa_failure_excursion"] = 0.0

    out[score_col] = (
        out["size_excursion"]
        + out["shape_excursion"]
        + out["non_gaussian_excursion"]
        + out["spatial_nonuniformity"]
        + out["edg_deviation_excursion"]
    )

    if score_col == "detector_badness":
        out["detector_bad_flag"] = out["detector_badness"] >= 3.0

    return out


def summarize_detector_metrics(star_df):
    """Aggregate star-level rows into detector-level summary metrics."""
    star_df = annotate_star_quality(star_df)
    if star_df.empty:
        return pd.DataFrame(
            columns=[
                "visit",
                "detector",
                "band",
                "day_obs",
                "n_stars",
                "median_fwhm_proxy",
                "median_ee80_radius",
                "median_ellipticity",
                "median_ng_score",
                "median_edg_deviation_score",
                "n_failed_stars",
                "frac_failed_stars",
                "n_science_bad_stars",
                "frac_science_bad_stars",
                "n_high_ng_stars",
                "frac_high_ng_stars",
                "n_high_edg_deviation_stars",
                "frac_high_edg_deviation_stars",
                "detector_fwhm_spread",
                "detector_ellipticity_spread",
                "edg_deviation_excursion",
                "detector_badness",
            ]
        )

    rows = []
    group_cols = [col for col in ["visit", "detector", "band", "day_obs"] if col in star_df.columns]
    for keys, group in star_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_stars"] = len(group)
        row["median_fwhm_proxy"] = group["fwhm_proxy"].median(skipna=True)
        row["median_ee80_radius"] = group["ee80_radius"].median(skipna=True)
        row["median_ellipticity"] = group["ellipticity"].median(skipna=True)
        row["median_ng_score"] = group["ng_score"].median(skipna=True)
        row["median_edg_deviation_score"] = group["edg_deviation_score"].median(skipna=True)
        row["detector_fwhm_spread"] = group["fwhm_proxy"].quantile(0.9) - group["fwhm_proxy"].quantile(0.1)
        row["detector_ellipticity_spread"] = group["ellipticity"].quantile(0.9) - group["ellipticity"].quantile(0.1)
        row["n_failed_stars"] = int(group["failed_star_flag"].sum())
        row["frac_failed_stars"] = float(group["failed_star_flag"].mean()) if len(group) > 0 else np.nan
        row["n_science_bad_stars"] = int(group["science_bad_star_flag"].sum())
        row["frac_science_bad_stars"] = float(group["science_bad_star_flag"].mean()) if len(group) > 0 else np.nan
        row["n_high_ng_stars"] = int(group["high_ng_star_flag"].sum())
        row["frac_high_ng_stars"] = float(group["high_ng_star_flag"].mean()) if len(group) > 0 else np.nan
        row["n_high_edg_deviation_stars"] = int(group["high_edg_deviation_star_flag"].sum())
        row["frac_high_edg_deviation_stars"] = float(group["high_edg_deviation_star_flag"].mean()) if len(group) > 0 else np.nan
        rows.append(row)

    detector_df = pd.DataFrame(rows)
    detector_df = compute_visit_badness(detector_df, score_col="detector_badness")
    detector_df["detector_failed_flag"] = detector_df["frac_failed_stars"].fillna(0) > DETECTOR_FAILURE_FRACTION_THRESHOLD
    detector_df["detector_science_bad_flag"] = detector_df["detector_bad_flag"].fillna(False)
    detector_df["detector_high_ng_flag"] = detector_df["frac_high_ng_stars"].fillna(0) > 0
    detector_df["detector_high_edg_deviation_flag"] = detector_df["frac_high_edg_deviation_stars"].fillna(0) > 0
    return detector_df


def summarize_visit_metrics(detector_df):
    """Aggregate detector-level rows into visit-level summary metrics."""
    if detector_df.empty:
        return pd.DataFrame(
            columns=[
                "visit",
                "band",
                "day_obs",
                "n_detectors",
                "n_stars",
                "median_detector_fwhm",
                "p90_detector_fwhm",
                "detector_fwhm_spread",
                "median_detector_ee80",
                "median_detector_ellipticity",
                "p90_detector_ellipticity",
                "detector_ellipticity_spread",
                "median_detector_ng_score",
                "median_edg_deviation_score",
                "n_failed_detectors",
                "frac_failed_detectors",
                "n_science_bad_detectors",
                "frac_science_bad_detectors",
                "n_high_ng_detectors",
                "frac_high_ng_detectors",
                "n_high_ng_stars",
                "frac_high_ng_stars",
                "n_high_edg_deviation_detectors",
                "frac_high_edg_deviation_detectors",
                "n_high_edg_deviation_stars",
                "frac_high_edg_deviation_stars",
                "size_excursion",
                "shape_excursion",
                "non_gaussian_excursion",
                "edg_deviation_excursion",
                "spatial_nonuniformity",
                "qa_failure_excursion",
                "visit_badness",
                "bad_visit_flag",
                "bad_visit_rank",
            ]
        )

    rows = []
    group_cols = [col for col in ["visit", "band", "day_obs"] if col in detector_df.columns]
    for keys, group in detector_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_detectors"] = len(group)
        row["n_stars"] = group["n_stars"].sum()
        row["median_detector_fwhm"] = group["median_fwhm_proxy"].median(skipna=True)
        row["p90_detector_fwhm"] = group["median_fwhm_proxy"].quantile(0.9)
        row["detector_fwhm_spread"] = group["median_fwhm_proxy"].quantile(0.9) - group["median_fwhm_proxy"].quantile(0.1)
        row["median_detector_ee80"] = group["median_ee80_radius"].median(skipna=True)
        row["median_detector_ellipticity"] = group["median_ellipticity"].median(skipna=True)
        row["p90_detector_ellipticity"] = group["median_ellipticity"].quantile(0.9)
        row["detector_ellipticity_spread"] = group["median_ellipticity"].quantile(0.9) - group["median_ellipticity"].quantile(0.1)
        row["median_detector_ng_score"] = group["median_ng_score"].median(skipna=True)
        row["median_edg_deviation_score"] = group["median_edg_deviation_score"].median(skipna=True)
        row["n_failed_detectors"] = int(group["detector_failed_flag"].sum()) if "detector_failed_flag" in group else 0
        row["frac_failed_detectors"] = float(group["detector_failed_flag"].mean()) if "detector_failed_flag" in group else np.nan
        row["n_science_bad_detectors"] = int(group["detector_science_bad_flag"].sum()) if "detector_science_bad_flag" in group else 0
        row["frac_science_bad_detectors"] = float(group["detector_science_bad_flag"].mean()) if "detector_science_bad_flag" in group else np.nan
        row["n_high_ng_detectors"] = int(group["detector_high_ng_flag"].sum()) if "detector_high_ng_flag" in group else 0
        row["frac_high_ng_detectors"] = float(group["detector_high_ng_flag"].mean()) if "detector_high_ng_flag" in group else np.nan
        row["n_high_ng_stars"] = int(group["n_high_ng_stars"].sum()) if "n_high_ng_stars" in group else 0
        row["frac_high_ng_stars"] = (
            row["n_high_ng_stars"] / row["n_stars"] if row["n_stars"] > 0 else np.nan
        )
        row["n_high_edg_deviation_detectors"] = int(group["detector_high_edg_deviation_flag"].sum()) if "detector_high_edg_deviation_flag" in group else 0
        row["frac_high_edg_deviation_detectors"] = (
            float(group["detector_high_edg_deviation_flag"].mean()) if "detector_high_edg_deviation_flag" in group else np.nan
        )
        row["n_high_edg_deviation_stars"] = int(group["n_high_edg_deviation_stars"].sum()) if "n_high_edg_deviation_stars" in group else 0
        row["frac_high_edg_deviation_stars"] = (
            row["n_high_edg_deviation_stars"] / row["n_stars"] if row["n_stars"] > 0 else np.nan
        )
        rows.append(row)

    visit_df = pd.DataFrame(rows)
    visit_df = compute_visit_badness(visit_df, score_col="visit_badness")
    visit_df = flag_bad_visits(visit_df)
    return visit_df


def build_known_visit_validation_table(
    visit_df,
    known_bad_visits=None,
    known_good_visits=None,
    known_moderate_visits=None,
):
    """
    Build a compact validation table for manually labeled visits.

    Missing configured visits are retained in the output with NaN science
    columns so notebook validation cells can immediately show what was absent
    from the analyzed sample.
    """
    known_bad_visits = list(known_bad_visits or [])
    known_good_visits = list(known_good_visits or [])
    known_moderate_visits = list(known_moderate_visits or [])

    requested_rows = []
    for visit in known_bad_visits:
        requested_rows.append({"visit": visit, "expected_label": "known_bad"})
    for visit in known_moderate_visits:
        requested_rows.append({"visit": visit, "expected_label": "known_moderate"})
    for visit in known_good_visits:
        requested_rows.append({"visit": visit, "expected_label": "known_good"})

    requested_df = pd.DataFrame(requested_rows)
    if requested_df.empty:
        return pd.DataFrame(
            columns=[
                "visit",
                "expected_label",
                "visit_badness",
                "bad_visit_rank",
                "bad_visit_flag",
                "size_excursion",
                "shape_excursion",
                "non_gaussian_excursion",
                "edg_deviation_excursion",
                "spatial_nonuniformity",
                "frac_failed_detectors",
                "frac_science_bad_detectors",
                "frac_high_ng_detectors",
                "frac_high_ng_stars",
                "frac_high_edg_deviation_detectors",
                "frac_high_edg_deviation_stars",
            ]
        )

    keep_cols = [
        col
        for col in [
            "visit",
            "day_obs",
            "visit_badness",
            "bad_visit_rank",
            "bad_visit_flag",
            "size_excursion",
            "shape_excursion",
                "non_gaussian_excursion",
                "edg_deviation_excursion",
                "spatial_nonuniformity",
                "frac_failed_detectors",
                "frac_science_bad_detectors",
                "frac_high_ng_detectors",
                "frac_high_ng_stars",
                "frac_high_edg_deviation_detectors",
                "frac_high_edg_deviation_stars",
                "n_detectors",
                "n_stars",
            ]
        if col in visit_df.columns
    ]

    validation_df = requested_df.merge(visit_df[keep_cols], on="visit", how="left")
    label_order = {"known_bad": 0, "known_moderate": 1, "known_good": 2}
    validation_df["_label_order"] = validation_df["expected_label"].map(label_order).fillna(99)
    validation_df = validation_df.sort_values(
        ["_label_order", "visit_badness", "visit"],
        ascending=[True, False, True],
        na_position="last",
    ).drop(columns="_label_order")
    return validation_df


def build_night_summary_table(visit_df):
    """
    Build one compact row per day_obs for advisor-facing screening summaries.

    The `frac_visits_with_high_ng` and `frac_visits_with_high_edg_deviation`
    columns use a practical first-pass definition: a visit counts as high if
    any sampled star or detector crosses the corresponding screening flag.
    """
    if visit_df.empty:
        return pd.DataFrame(
            columns=[
                "day_obs",
                "n_visits",
                "n_flagged_bad_visits",
                "frac_flagged_bad_visits",
                "median_visit_badness",
                "median_size_excursion",
                "median_shape_excursion",
                "median_non_gaussian_excursion",
                "median_edg_deviation_excursion",
                "median_spatial_nonuniformity",
                "frac_visits_with_high_ng",
                "frac_visits_with_high_edg_deviation",
            ]
        )

    rows = []
    for day_obs, group in visit_df.groupby("day_obs", dropna=False):
        visit_has_high_ng = (
            group.get("frac_high_ng_stars", pd.Series(0.0, index=group.index)).fillna(0) > 0
        ) | (
            group.get("frac_high_ng_detectors", pd.Series(0.0, index=group.index)).fillna(0) > 0
        )
        visit_has_high_edg = (
            group.get("frac_high_edg_deviation_stars", pd.Series(0.0, index=group.index)).fillna(0) > 0
        ) | (
            group.get("frac_high_edg_deviation_detectors", pd.Series(0.0, index=group.index)).fillna(0) > 0
        )

        rows.append(
            {
                "day_obs": day_obs,
                "n_visits": int(len(group)),
                "n_flagged_bad_visits": int(group.get("bad_visit_flag", pd.Series(False, index=group.index)).fillna(False).sum()),
                "frac_flagged_bad_visits": float(group.get("bad_visit_flag", pd.Series(False, index=group.index)).fillna(False).mean()),
                "median_visit_badness": float(group["visit_badness"].median(skipna=True)),
                "median_size_excursion": float(group["size_excursion"].median(skipna=True)),
                "median_shape_excursion": float(group["shape_excursion"].median(skipna=True)),
                "median_non_gaussian_excursion": float(group["non_gaussian_excursion"].median(skipna=True)),
                "median_edg_deviation_excursion": float(group["edg_deviation_excursion"].median(skipna=True)),
                "median_spatial_nonuniformity": float(group["spatial_nonuniformity"].median(skipna=True)),
                "frac_visits_with_high_ng": float(visit_has_high_ng.mean()) if len(group) > 0 else np.nan,
                "frac_visits_with_high_edg_deviation": float(visit_has_high_edg.mean()) if len(group) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("day_obs").reset_index(drop=True)


def flag_bad_visits(visit_df, threshold=None, top_n=None, score_col="visit_badness"):
    """Flag bad visits either by threshold or by taking the largest scores."""
    out = visit_df.copy()
    scores = out[score_col].fillna(-np.inf)

    if top_n is not None:
        order = scores.sort_values(ascending=False).index
        bad_index = set(order[:top_n])
        out["bad_visit_flag"] = out.index.isin(bad_index)
    else:
        if threshold is None:
            threshold = 3.0
        out["bad_visit_flag"] = scores >= threshold

    out["bad_visit_rank"] = scores.rank(ascending=False, method="dense")
    return out


############################################
# 6. Plotting utilities
############################################


def _render_and_display(fig):
    """Render a figure efficiently in notebooks."""
    if _IPYTHON_AVAILABLE:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        _ipython_display(_IPyImage(buf.getvalue()))
    else:
        plt.show()
        plt.close(fig)


def plot_model_comparison_page(results, start, page_size, visit_id, detector_id, band):
    """Optional debugging gallery for per-star model comparisons."""
    end = min(start + page_size, len(results))
    n = end - start
    if n <= 0:
        return

    fig, axes = plt.subplots(n, 15, figsize=(56, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Observed star vs Gaussian / Double Gaussian / Gauss-Hermite / Moffat / Shapelet - Visit %s, Detector %s, Band %s (stars %d-%d of %d)"
        % (visit_id, detector_id, band, start + 1, end, len(results)),
        fontsize=14,
        y=1.01,
    )

    model_specs = [
        ("Gaussian", "gaussian_array", "gaussian_residual", "gaussian_chi2", "gaussian_params_named"),
        ("DG", "dgauss_array", "dg_residual", "dg_chi2", "dg_params_named"),
        ("EDG", "edg_array", "edg_residual", "edg_chi2", "edg_params_named"),
        ("GH", "gh_array", "gh_residual", "gh_chi2", "gh_params_named"),
        ("Moffat", "moffat_array", "moffat_residual", "moffat_chi2", "moffat_params_named"),
        ("Shapelet", "shapelet_array", "shapelet_residual", "shapelet_chi2", "shapelet_params_named"),
    ]

    for row_idx, i in enumerate(range(start, end)):
        r = results[i]
        image = r["psf_array"]
        max_val = float(np.nanmax(np.abs(image))) if image is not None else 1.0

        im0 = axes[row_idx, 0].imshow(
            image,
            vmin=np.nanmin(image),
            vmax=np.nanmax(image),
            cmap="viridis",
            origin="lower",
        )
        axes[row_idx, 0].set_title(
            "Star %d: Observed star\n(x=%.0f, y=%.0f)\nFWHM=%.2f pix\nBest=%s"
            % (
                i + 1,
                r.get("x", np.nan),
                r.get("y", np.nan),
                r.get("fwhm_proxy", np.nan),
                MODEL_LABELS.get(r.get("best_model", "gaussian"), r.get("best_model", "none")),
            ),
            fontsize=9,
        )
        fig.colorbar(im0, ax=axes[row_idx, 0], shrink=0.75)

        for col_offset, (label, array_key, residual_key, chi2_key, params_key) in enumerate(model_specs, start=1):
            model = r.get(array_key)
            residual = r.get(residual_key)
            chi2 = r.get(chi2_key, np.nan)
            params_named = r.get(params_key, {})

            im_model = axes[row_idx, 2 * col_offset - 1].imshow(
                model,
                vmin=np.nanmin(image),
                vmax=np.nanmax(image),
                cmap="viridis",
                origin="lower",
            )
            param_text = ", ".join(f"{k}={v:.2f}" for k, v in list(params_named.items())[:3])
            axes[row_idx, 2 * col_offset - 1].set_title(f"{label}\n{param_text}", fontsize=8)
            fig.colorbar(im_model, ax=axes[row_idx, 2 * col_offset - 1], shrink=0.75)

            res_scale = np.nanmax(np.abs(residual)) if residual is not None and np.any(np.isfinite(residual)) else max_val / 10.0
            res_scale = max(float(res_scale), 1e-12)
            im_res = axes[row_idx, 2 * col_offset].imshow(
                residual,
                vmin=-res_scale,
                vmax=res_scale,
                cmap="RdBu_r",
                origin="lower",
            )
            axes[row_idx, 2 * col_offset].set_title(f"{label} residual\nchi^2={chi2:.4e}", fontsize=8)
            fig.colorbar(im_res, ax=axes[row_idx, 2 * col_offset], shrink=0.75)

        axes[row_idx, 13].plot(r["rp_radius"], r["rp_psf"], "o-", ms=2.5, lw=1.0, label="Observed star")
        axes[row_idx, 13].plot(r["rp_radius"], r["rp_gaussian"], "o--", ms=2.5, lw=1.0, label="Gaussian")
        axes[row_idx, 13].plot(r["rp_radius"], r["rp_dg"], "s--", ms=2.5, lw=1.0, label="Double Gaussian")
        axes[row_idx, 13].plot(r["rp_radius"], r["rp_edg"], "P--", ms=2.5, lw=1.0, label="Elliptical DG")
        axes[row_idx, 13].plot(r["rp_radius"], r["rp_gh"], "^--", ms=2.5, lw=1.0, label="Gauss-Hermite")
        axes[row_idx, 13].plot(r["rp_radius"], r["rp_moffat"], "v--", ms=2.5, lw=1.0, label="Moffat")
        axes[row_idx, 13].plot(r["rp_radius"], r["rp_shapelet"], "d--", ms=2.5, lw=1.0, label="Shapelet")
        axes[row_idx, 13].set_title("Radial profile", fontsize=9)
        axes[row_idx, 13].set_xlabel("Radius [pixel]")
        axes[row_idx, 13].set_ylabel("Mean intensity")
        axes[row_idx, 13].grid(alpha=0.3)
        axes[row_idx, 13].legend(fontsize=6)

        axes[row_idx, 14].axhline(0.0, color="k", lw=1)
        axes[row_idx, 14].plot(r["rp_radius"], r["rp_gaussian_residual"], "o-", ms=2.5, lw=1.0, label="Data - Gaussian")
        axes[row_idx, 14].plot(r["rp_radius"], r["rp_dg_residual"], "s-", ms=2.5, lw=1.0, label="Data - DG")
        axes[row_idx, 14].plot(r["rp_radius"], r["rp_edg_residual"], "P-", ms=2.5, lw=1.0, label="Data - EDG")
        axes[row_idx, 14].plot(r["rp_radius"], r["rp_gh_residual"], "^-", ms=2.5, lw=1.0, label="Data - GH")
        axes[row_idx, 14].plot(r["rp_radius"], r["rp_moffat_residual"], "v-", ms=2.5, lw=1.0, label="Data - Moffat")
        axes[row_idx, 14].plot(r["rp_radius"], r["rp_shapelet_residual"], "d-", ms=2.5, lw=1.0, label="Data - Shapelet")
        axes[row_idx, 14].set_title("Profile residuals", fontsize=9)
        axes[row_idx, 14].set_xlabel("Radius [pixel]")
        axes[row_idx, 14].set_ylabel("Profile residual")
        axes[row_idx, 14].grid(alpha=0.3)
        axes[row_idx, 14].legend(fontsize=6)

    plt.tight_layout()
    _render_and_display(fig)


def plot_model_comparison_pages(results, page_size, visit_id, detector_id, band):
    """Render the optional all-stars debugging gallery in pages."""
    for page_start in range(0, len(results), page_size):
        plot_model_comparison_page(results, page_start, page_size, visit_id, detector_id, band)


def plot_best_model_counts(results):
    """Plot how often each model wins across a set of observed-star analyses."""
    counts = count_best_models(results)
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    values = [counts[name] for name in MODEL_ORDER]
    colors = [MODEL_COLORS[name] for name in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Best-Fit Model Counts")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of observed star stamps")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    _render_and_display(fig)
    return counts


def plot_visit_badness(visit_df, sort_by="visit_badness", top_n=None):
    """Plot visit badness as a compact ranking/time-series view."""
    if visit_df.empty:
        return

    plot_df = visit_df.copy()
    if top_n is not None:
        plot_df = plot_df.sort_values(sort_by, ascending=False).head(top_n)
    elif sort_by in plot_df.columns:
        plot_df = plot_df.sort_values(sort_by, ascending=True)

    x_labels = plot_df["day_obs"].astype(str) if "day_obs" in plot_df.columns and plot_df["day_obs"].notna().any() else plot_df["visit"].astype(str)
    y = plot_df["visit_badness"]
    colors = np.where(plot_df.get("bad_visit_flag", False), "tab:red", "tab:blue")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x_labels, y, color=colors)
    ax.set_title("Visit badness ranking")
    ax.set_xlabel("Visit" if "day_obs" not in plot_df.columns else "day_obs / visit")
    ax.set_ylabel("visit_badness")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _render_and_display(fig)


def plot_bad_visit_gallery(results, visit_df, top_n_visits=3, n_examples=6):
    """Plot representative star cutouts for the worst or flagged visits."""
    if not results:
        return

    flagged = visit_df[visit_df.get("bad_visit_flag", False)]
    if flagged.empty:
        flagged = visit_df.sort_values("visit_badness", ascending=False).head(top_n_visits)
    else:
        flagged = flagged.sort_values("visit_badness", ascending=False).head(top_n_visits)

    selected_visits = flagged["visit"].tolist()
    if not selected_visits:
        return
    visit_lookup = flagged.set_index("visit")

    star_df = pd.DataFrame([build_star_summary_row(result) for result in results])
    if not star_df.empty:
        star_df = annotate_star_quality(star_df)

    def _build_display_priority_table(visit_results):
        priority_rows = []
        source_lookup = {}
        if not star_df.empty and "sourceId" in star_df.columns:
            source_subset = star_df.dropna(subset=["sourceId"]).copy()
            if not source_subset.empty:
                source_subset["sourceId"] = source_subset["sourceId"].astype(str)
                source_lookup = source_subset.drop_duplicates("sourceId").set_index("sourceId")[
                    ["science_bad_star_flag", "high_ng_star_flag", "high_edg_deviation_star_flag"]
                ].to_dict("index")

        for idx, result in enumerate(visit_results):
            source_id = result.get("sourceId")
            lookup = source_lookup.get(str(source_id), {})
            priority_rows.append(
                {
                    "result_index": idx,
                    "fwhm_proxy": result.get("fwhm_proxy", np.nan),
                    "ee80_radius": result.get("ee80_radius", np.nan),
                    "ellipticity": result.get("ellipticity", np.nan),
                    "ng_score": result.get("ng_score", np.nan),
                    "edg_deviation_score": result.get("edg_deviation_score", np.nan),
                    "science_bad_star_flag": bool(
                        lookup.get("science_bad_star_flag", result.get("science_bad_star_flag", False))
                    ),
                    "high_ng_star_flag": bool(
                        lookup.get("high_ng_star_flag", result.get("high_ng_star_flag", False))
                    ),
                    "high_edg_deviation_star_flag": bool(
                        lookup.get("high_edg_deviation_star_flag", result.get("high_edg_deviation_star_flag", False))
                    ),
                }
            )

        priority_df = pd.DataFrame(priority_rows)
        if priority_df.empty:
            return priority_df

        for metric in ["fwhm_proxy", "ee80_radius", "ellipticity", "ng_score", "edg_deviation_score"]:
            priority_df[f"{metric}_display_z"] = _positive_robust_excursion(priority_df[metric]).fillna(0.0)

        priority_df["gallery_priority"] = (
            1.0 * priority_df["fwhm_proxy_display_z"]
            + 1.0 * priority_df["ee80_radius_display_z"]
            + 1.0 * priority_df["ellipticity_display_z"]
            + 1.25 * priority_df["ng_score_display_z"]
            + 1.25 * priority_df["edg_deviation_score_display_z"]
            + 1.0 * priority_df["science_bad_star_flag"].astype(float)
            + 0.5 * priority_df["high_ng_star_flag"].astype(float)
            + 0.5 * priority_df["high_edg_deviation_star_flag"].astype(float)
        )
        return priority_df

    rows = []
    for visit in selected_visits:
        visit_results = [r for r in results if r.get("visit") == visit]
        priority_df = _build_display_priority_table(visit_results)
        if not priority_df.empty:
            chosen_indices = (
                priority_df.sort_values(
                    ["gallery_priority", "edg_deviation_score", "ng_score", "ellipticity", "fwhm_proxy"],
                    ascending=False,
                    na_position="last",
                )["result_index"]
                .head(n_examples)
                .tolist()
            )
            visit_results = [visit_results[idx] for idx in chosen_indices]
        else:
            visit_results = visit_results[:n_examples]
        rows.append(visit_results)

    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, visit_results in enumerate(rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(visit_results):
                ax.axis("off")
                continue

            result = visit_results[col_idx]
            image = result.get("psf_array")
            im = ax.imshow(image, origin="lower", cmap="viridis")
            visit_meta = visit_lookup.loc[result.get("visit")]
            title_prefix = ""
            if col_idx == 0:
                title_prefix = (
                    "VB={:.2f} S={:.2f} Sh={:.2f}\nNG={:.2f} EDG={:.2f} Sp={:.2f}\n".format(
                        visit_meta.get("visit_badness", np.nan),
                        visit_meta.get("size_excursion", np.nan),
                        visit_meta.get("shape_excursion", np.nan),
                        visit_meta.get("non_gaussian_excursion", np.nan),
                        visit_meta.get("edg_deviation_excursion", np.nan),
                        visit_meta.get("spatial_nonuniformity", np.nan),
                    )
                )
            ax.set_title(
                title_prefix + "visit=%s  best=%s\n(x=%.0f, y=%.0f)\nng=%.2f  edg=%.2f  e=%.2f"
                % (
                    result.get("visit"),
                    result.get("best_model", "none"),
                    result.get("x", np.nan),
                    result.get("y", np.nan),
                    result.get("ng_score", np.nan),
                    result.get("edg_deviation_score", np.nan),
                    result.get("ellipticity", np.nan),
                ),
                fontsize=8,
            )
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Representative stars for flagged / worst visits", fontsize=13, y=1.02)
    plt.tight_layout()
    _render_and_display(fig)


if __name__ == "__main__":
    print("fittingTools.py is intended to be imported from a notebook or script.")
    print("It now wraps simulationTools.py for observed-star preprocessing and aggregation.")
