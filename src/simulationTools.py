from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import factorial, hermite

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
MODEL_ORDER = [
    "gaussian",
    "double_gaussian",
    "elliptical_double_gaussian",
    "gauss_hermite",
    "moffat",
    "shapelet",
]
MODEL_LABELS = {
    "gaussian": "Gaussian",
    "double_gaussian": "Double Gaussian",
    "elliptical_double_gaussian": "Elliptical Double Gaussian",
    "gauss_hermite": "Gauss-Hermite",
    "moffat": "Moffat",
    "shapelet": "Shapelet",
}
MODEL_COLORS = {
    "gaussian": "tab:purple",
    "double_gaussian": "tab:blue",
    "elliptical_double_gaussian": "tab:brown",
    "gauss_hermite": "tab:orange",
    "moffat": "tab:red",
    "shapelet": "tab:green",
}
MODEL_MARKERS = {
    "gaussian": "o",
    "double_gaussian": "s",
    "elliptical_double_gaussian": "P",
    "gauss_hermite": "^",
    "moffat": "v",
    "shapelet": "d",
}


############################################
# 1. Generic utilities
############################################


def image_center_from_shape(shape):
    """Return the geometric center used for synthetic PSF generation."""
    ny, nx = shape
    return ((nx - 1) / 2.0, (ny - 1) / 2.0)


def image_center(image):
    """Return the geometric center of a 2D image."""
    return image_center_from_shape(image.shape)


def normalize_flux(image):
    """Normalize an image to unit total flux."""
    image = np.asarray(image, dtype=float)
    total_flux = np.sum(image)

    if total_flux <= 0:
        raise ValueError("Total flux must be positive for normalization.")

    return image / total_flux


def radial_profile(image, center=None):
    """Compute an azimuthally averaged radial profile for a 2D image."""
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    x0, y0 = center
    y, x = np.indices(image.shape)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    r_bin = np.floor(r).astype(int)

    sum_per_bin = np.bincount(r_bin.ravel(), weights=image.ravel())
    count_per_bin = np.bincount(r_bin.ravel())
    radius_per_bin = np.bincount(r_bin.ravel(), weights=r.ravel())

    valid = count_per_bin > 0
    profile = sum_per_bin[valid] / count_per_bin[valid]
    radius = radius_per_bin[valid] / count_per_bin[valid]

    return radius, profile


def _radial_distance_grid(shape, center):
    """Return the radial distance grid for a given image shape and center."""
    y, x = np.indices(shape)
    x0, y0 = center
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def compute_curve_of_growth(image, center=None):
    """
    Compute the cumulative enclosed-flux curve for a 2D PSF image.

    The curve is based on non-negative finite pixel values sorted by radius.
    This keeps the helper useful for both synthetic PSFs and background-subtracted
    observed cutouts where a few negative pixels may exist.
    """
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    weights = np.clip(np.where(np.isfinite(image), image, 0.0), 0.0, None)
    total_flux = np.sum(weights)

    if total_flux <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    radius = _radial_distance_grid(image.shape, center).ravel()
    weights = weights.ravel()
    order = np.argsort(radius)
    radius_sorted = radius[order]
    flux_sorted = weights[order]
    cumulative = np.cumsum(flux_sorted) / total_flux

    return radius_sorted, cumulative


def compute_ee_radius(image, frac=0.8, center=None):
    """Return the enclosed-energy radius containing a given flux fraction."""
    frac = float(frac)
    if frac <= 0 or frac > 1:
        raise ValueError("frac must be in the interval (0, 1].")

    radius, cumulative = compute_curve_of_growth(image, center=center)
    if len(radius) == 0:
        return np.nan

    idx = np.searchsorted(cumulative, frac, side="left")
    if idx == 0:
        return float(radius[0])
    if idx >= len(radius):
        return float(radius[-1])

    r0, r1 = radius[idx - 1], radius[idx]
    c0, c1 = cumulative[idx - 1], cumulative[idx]
    if c1 <= c0:
        return float(r1)

    alpha = (frac - c0) / (c1 - c0)
    return float(r0 + alpha * (r1 - r0))


def compute_fwhm_from_profile(image, center=None):
    """Estimate FWHM from the azimuthally averaged radial profile."""
    image = np.asarray(image, dtype=float)
    safe_image = np.clip(np.where(np.isfinite(image), image, 0.0), 0.0, None)

    radius, profile = radial_profile(safe_image, center=center)
    if len(radius) == 0 or not np.any(np.isfinite(profile)):
        return np.nan

    peak_idx = int(np.nanargmax(profile))
    peak_value = profile[peak_idx]
    if not np.isfinite(peak_value) or peak_value <= 0:
        return np.nan

    half_max = 0.5 * peak_value
    for i in range(max(peak_idx + 1, 1), len(profile)):
        if profile[i] <= half_max:
            r0, r1 = radius[i - 1], radius[i]
            p0, p1 = profile[i - 1], profile[i]
            if p1 == p0:
                half_radius = r1
            else:
                alpha = (half_max - p0) / (p1 - p0)
                half_radius = r0 + alpha * (r1 - r0)
            return float(2.0 * half_radius)

    return np.nan


def compute_second_moment_shape(image, center=None):
    """
    Compute simple second-moment shape diagnostics around a supplied center.

    The moments use non-negative finite weights so the helper remains stable for
    both synthetic PSFs and lightly background-subtracted observed stamps.
    """
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    weights = np.clip(np.where(np.isfinite(image), image, 0.0), 0.0, None)
    total_flux = np.sum(weights)
    if total_flux <= 0:
        return {
            "Mxx": np.nan,
            "Myy": np.nan,
            "Mxy": np.nan,
            "e1": np.nan,
            "e2": np.nan,
            "ellipticity": np.nan,
            "determinant_radius": np.nan,
        }

    y, x = np.indices(image.shape)
    x0, y0 = center
    dx = x - x0
    dy = y - y0

    Mxx = float(np.sum(weights * dx**2) / total_flux)
    Myy = float(np.sum(weights * dy**2) / total_flux)
    Mxy = float(np.sum(weights * dx * dy) / total_flux)

    trace = Mxx + Myy
    if trace > 0:
        e1 = (Mxx - Myy) / trace
        e2 = (2.0 * Mxy) / trace
        ellipticity = float(np.sqrt(e1**2 + e2**2))
    else:
        e1 = np.nan
        e2 = np.nan
        ellipticity = np.nan

    determinant = Mxx * Myy - Mxy**2
    determinant_radius = float(determinant ** 0.25) if determinant > 0 else np.nan

    return {
        "Mxx": Mxx,
        "Myy": Myy,
        "Mxy": Mxy,
        "e1": float(e1) if np.isfinite(e1) else np.nan,
        "e2": float(e2) if np.isfinite(e2) else np.nan,
        "ellipticity": ellipticity,
        "determinant_radius": determinant_radius,
    }


def compute_fit_metrics(image, model, center, core_radius=2.0, wing_radius=3.0):
    """
    Compute several fit-quality diagnostics for a synthetic PSF benchmark.

    Metrics:
    - global pixel MSE
    - core pixel MSE for r < core_radius
    - wing pixel MSE for r >= wing_radius
    - radial profile MSE
    """
    image = np.asarray(image, dtype=float)
    model = np.asarray(model, dtype=float)

    if not np.all(np.isfinite(model)):
        return {
            "global_mse": np.inf,
            "core_mse": np.inf,
            "wing_mse": np.inf,
            "profile_mse": np.inf,
        }

    residual = image - model
    r = _radial_distance_grid(image.shape, center)

    global_mse = np.mean(residual**2)

    core_mask = r < core_radius
    wing_mask = r >= wing_radius

    core_mse = np.mean(residual[core_mask] ** 2) if np.any(core_mask) else np.nan
    wing_mse = np.mean(residual[wing_mask] ** 2) if np.any(wing_mask) else np.nan

    radius_image, rp_image = radial_profile(image, center=center)
    radius_model, rp_model = radial_profile(model, center=center)
    n_profile = min(len(radius_image), len(radius_model))
    if n_profile > 0:
        profile_mse = np.mean((rp_image[:n_profile] - rp_model[:n_profile]) ** 2)
    else:
        profile_mse = np.nan

    return {
        "global_mse": global_mse,
        "core_mse": core_mse,
        "wing_mse": wing_mse,
        "profile_mse": profile_mse,
    }


def _run_curve_fit(model_func, coords, data, p0, bounds, maxfev=10000):
    """Wrap curve_fit and return status metadata instead of failing silently."""
    try:
        params, cov = curve_fit(
            model_func,
            coords,
            data,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        fit_valid = np.all(np.isfinite(params))
        message = "curve_fit converged" if fit_valid else "curve_fit returned non-finite parameters"
        return {
            "params": params,
            "cov": cov,
            "success": True,
            "message": message,
            "fit_valid": bool(fit_valid),
            "nfev": None,
        }
    except Exception as exc:
        return {
            "params": np.asarray(p0, dtype=float),
            "cov": None,
            "success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "fit_valid": False,
            "nfev": None,
        }


def _finalize_fit_result(
    image,
    center,
    params,
    cov,
    success,
    message,
    fit_valid,
    model,
    core_radius=2.0,
    wing_radius=3.0,
    params_named=None,
    nfev=None,
    extra=None,
):
    """Build the common result dictionary shared by all fit helpers."""
    image = np.asarray(image, dtype=float)
    radius, image_profile = radial_profile(image, center=center)

    if not fit_valid or model is None or not np.all(np.isfinite(model)):
        model = np.full_like(image, np.nan, dtype=float)
        residual = np.full_like(image, np.nan, dtype=float)
        model_profile = np.full_like(radius, np.nan, dtype=float)
        profile_residual = np.full_like(radius, np.nan, dtype=float)
        metrics = {
            "global_mse": np.inf,
            "core_mse": np.inf,
            "wing_mse": np.inf,
            "profile_mse": np.inf,
        }
        chi2 = np.inf
    else:
        residual = image - model
        _, model_profile = radial_profile(model, center=center)
        n_profile = min(len(radius), len(model_profile))
        aligned_model_profile = np.full_like(radius, np.nan, dtype=float)
        aligned_model_profile[:n_profile] = model_profile[:n_profile]
        model_profile = aligned_model_profile
        profile_residual = image_profile - model_profile
        metrics = compute_fit_metrics(
            image,
            model,
            center,
            core_radius=core_radius,
            wing_radius=wing_radius,
        )
        chi2 = metrics["global_mse"]

    result = {
        "params": params,
        "params_named": params_named or {},
        "cov": cov,
        "model": model,
        "residual": residual,
        "chi2": chi2,
        "center": center,
        "rp_radius": radius,
        "rp_image": image_profile,
        "rp_model": model_profile,
        "rp_residual": profile_residual,
        "success": success,
        "message": message,
        "fit_valid": fit_valid,
        "nfev": nfev,
        "metrics": metrics,
        "global_mse": metrics["global_mse"],
        "core_mse": metrics["core_mse"],
        "wing_mse": metrics["wing_mse"],
        "profile_mse": metrics["profile_mse"],
    }

    if extra:
        result.update(extra)

    return result


############################################
# 2. Synthetic PSF generation
############################################


def make_gaussian_psf(size, sigma, center=None):
    """Generate a centered, unit-flux 2D Gaussian PSF."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    if center is None:
        center = image_center_from_shape((ny, nx))

    x0, y0 = center
    yy, xx = np.indices((ny, nx))
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    image = np.exp(-0.5 * r2 / sigma**2)

    return normalize_flux(image)


def make_heavy_wing_psf(size, sigma_core, wing_strength, wing_scale, center=None):
    """
    Generate a unit-flux PSF with a Gaussian-like core and non-Gaussian heavy wings.

    The wing component is exponential in radius, not a second Gaussian, so this does
    not exactly match the double-Gaussian fitting family.
    """
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    if center is None:
        center = image_center_from_shape((ny, nx))

    x0, y0 = center
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    core = np.exp(-0.5 * r**2 / sigma_core**2)
    wings = np.exp(-r / wing_scale)
    image = core + wing_strength * wings

    return normalize_flux(image)


def add_poisson_like_gaussian_noise(
    image,
    total_counts,
    rng=None,
    clip_negative=True,
    renormalize=True,
):
    """
    Add a simple Poisson-like Gaussian noise realization to a unit-flux PSF image.

    The image is first interpreted as an expectation map with total counts
    `total_counts`. Each pixel with expected count K receives Gaussian noise with
    sigma sqrt(K). By default negative noisy counts are clipped to zero and the
    output is renormalized back to unit total flux so it can be fed directly into
    the single-stamp analysis pipeline.
    """
    image = normalize_flux(np.asarray(image, dtype=float))
    rng = np.random.default_rng(rng)

    expected_counts = np.clip(image, 0.0, None) * float(total_counts)
    noisy_counts = expected_counts + rng.normal(
        loc=0.0,
        scale=np.sqrt(expected_counts),
        size=expected_counts.shape,
    )

    if clip_negative:
        noisy_counts = np.clip(noisy_counts, 0.0, None)

    if renormalize:
        return normalize_flux(noisy_counts)

    return noisy_counts


def _elliptical_coordinates(shape, center, theta=0.0):
    """Return rotated coordinates relative to a common image center."""
    y, x = np.indices(shape)
    x0, y0 = center
    dx = x - x0
    dy = y - y0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy
    return x_rot, y_rot


def make_elliptical_gaussian_psf(size, sigma_x, sigma_y, theta=0.0, center=None):
    """Generate a centered unit-flux elliptical Gaussian PSF."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    if center is None:
        center = image_center_from_shape((ny, nx))

    x_rot, y_rot = _elliptical_coordinates((ny, nx), center, theta=theta)
    image = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    return normalize_flux(image)


def make_elliptical_heavy_wing_psf(
    size,
    sigma_x,
    sigma_y,
    wing_strength,
    wing_scale,
    theta=0.0,
    center=None,
):
    """
    Generate an elliptical PSF with a Gaussian-like core and heavier wings.

    This future-ready helper mirrors the radial heavy-wing family but allows
    simple ellipticity benchmarks without introducing notebook-specific logic.
    """
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    if center is None:
        center = image_center_from_shape((ny, nx))

    x_rot, y_rot = _elliptical_coordinates((ny, nx), center, theta=theta)
    elliptical_radius = np.sqrt((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    core = np.exp(-0.5 * elliptical_radius**2)
    wings = np.exp(-elliptical_radius / wing_scale)
    image = core + wing_strength * wings
    return normalize_flux(image)


def make_elliptical_double_gaussian_psf(
    size,
    sigma1_x,
    sigma1_y,
    sigma2_x,
    sigma2_y,
    amp1=1.0,
    amp2=0.2,
    theta=0.0,
    center=None,
):
    """
    Generate a centered, normalized elliptical double-Gaussian PSF.

    This is a controlled truth model for later EDG-vs-deviant synthetic tests.
    Both Gaussian components share the same center and rotation.
    """
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    if center is None:
        center = image_center_from_shape((ny, nx))

    x_rot, y_rot = _elliptical_coordinates((ny, nx), center, theta=theta)
    g1 = amp1 * np.exp(-0.5 * ((x_rot / sigma1_x) ** 2 + (y_rot / sigma1_y) ** 2))
    g2 = amp2 * np.exp(-0.5 * ((x_rot / sigma2_x) ** 2 + (y_rot / sigma2_y) ** 2))
    return normalize_flux(g1 + g2)


def show_psf(image, title=None, ax=None, cmap="viridis"):
    """Display a simulated PSF image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(image, origin="lower", cmap=cmap)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


############################################
# 3. Gaussian baseline
############################################


def gaussian_2d_fixed_center(coords, A, sigma, center):
    """Circular 2D Gaussian model with fixed center and zero background."""
    x, y = coords
    x0, y0 = center
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return A * np.exp(-0.5 * r2 / sigma**2)


def fit_gaussian_image(
    image,
    sigma_hint=None,
    center=None,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Fit a circular Gaussian baseline to one synthetic PSF image."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    yy, xx = np.indices(image.shape)
    coords = (xx.ravel(), yy.ravel())
    data = image.ravel()

    if center is None:
        center = image_center(image)

    x0, y0 = center
    sigma0 = sigma_hint if sigma_hint is not None else min(nx, ny) / 4.0
    A0 = np.max(data)
    B0 = 0.0

    if fit_center and fit_background:
        model_func = lambda xy, A, sigma, x0_fit, y0_fit, B: gaussian_2d_fixed_center(xy, A, sigma, (x0_fit, y0_fit)) + B
        p0 = [A0, sigma0, x0, y0, B0]
        bounds = ([0.0, 0.05, 0.0, 0.0, -0.1], [np.inf, max(nx, ny), nx - 1.0, ny - 1.0, 0.1])
        param_names = ["A", "sigma", "x0", "y0", "B"]
    elif fit_center and not fit_background:
        model_func = lambda xy, A, sigma, x0_fit, y0_fit: gaussian_2d_fixed_center(xy, A, sigma, (x0_fit, y0_fit))
        p0 = [A0, sigma0, x0, y0]
        bounds = ([0.0, 0.05, 0.0, 0.0], [np.inf, max(nx, ny), nx - 1.0, ny - 1.0])
        param_names = ["A", "sigma", "x0", "y0"]
    elif not fit_center and fit_background:
        model_func = lambda xy, A, sigma, B: gaussian_2d_fixed_center(xy, A, sigma, center) + B
        p0 = [A0, sigma0, B0]
        bounds = ([0.0, 0.05, -0.1], [np.inf, max(nx, ny), 0.1])
        param_names = ["A", "sigma", "B"]
    else:
        model_func = lambda xy, A, sigma: gaussian_2d_fixed_center(xy, A, sigma, center)
        p0 = [A0, sigma0]
        bounds = ([0.0, 0.05], [np.inf, max(nx, ny)])
        param_names = ["A", "sigma"]

    fit = _run_curve_fit(model_func, coords, data, p0, bounds, maxfev=10000)

    if fit["fit_valid"]:
        try:
            model = model_func(coords, *fit["params"]).reshape(image.shape)
        except Exception as exc:
            fit["success"] = False
            fit["fit_valid"] = False
            fit["message"] = f"Model evaluation failed: {type(exc).__name__}: {exc}"
            model = None
    else:
        model = None

    params_named = {name: float(value) for name, value in zip(param_names, fit["params"])}

    return _finalize_fit_result(
        image,
        center,
        fit["params"],
        fit["cov"],
        fit["success"],
        fit["message"],
        fit["fit_valid"],
        model,
        core_radius=core_radius,
        wing_radius=wing_radius,
        params_named=params_named,
        nfev=fit["nfev"],
    )


############################################
# 4. Double Gaussian
############################################


def double_gaussian(r, a1, sigma1, a2, sigma2):
    g1 = a1 * np.exp(-r**2 / (2.0 * sigma1**2))
    g2 = a2 * np.exp(-r**2 / (2.0 * sigma2**2))
    return g1 + g2


def fit_double_gaussian_image(
    image,
    sigma_hint=None,
    center=None,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Fit a radial double-Gaussian model to one synthetic PSF image."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape

    if center is None:
        center = image_center(image)

    x0, y0 = center
    yy, xx = np.indices(image.shape)
    distances = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    data = image.ravel()

    sigma_guess = sigma_hint if sigma_hint is not None else min(nx, ny) / 4.0
    p0 = [np.max(data), sigma_guess, np.max(data) * 0.1, sigma_guess * 2.0]
    bounds = ([0.0, 0.05, 0.0, 0.05], [np.inf, max(nx, ny), np.inf, max(nx, ny)])

    fit = _run_curve_fit(
        double_gaussian,
        distances.ravel(),
        data,
        p0,
        bounds,
        maxfev=10000,
    )

    if fit["fit_valid"]:
        try:
            model = double_gaussian(distances, *fit["params"]).reshape(image.shape)
        except Exception as exc:
            fit["success"] = False
            fit["fit_valid"] = False
            fit["message"] = f"Model evaluation failed: {type(exc).__name__}: {exc}"
            model = None
    else:
        model = None

    params_named = {
        "a1": float(fit["params"][0]),
        "sigma1": float(fit["params"][1]),
        "a2": float(fit["params"][2]),
        "sigma2": float(fit["params"][3]),
    }

    return _finalize_fit_result(
        image,
        center,
        fit["params"],
        fit["cov"],
        fit["success"],
        fit["message"],
        fit["fit_valid"],
        model,
        core_radius=core_radius,
        wing_radius=wing_radius,
        params_named=params_named,
        nfev=fit["nfev"],
    )


############################################
# 5. Elliptical Double Gaussian
############################################


def elliptical_double_gaussian_2d(coords, A1, x0, y0, sigma1_x, sigma1_y, A2, sigma2_x, sigma2_y, theta, B):
    """Two co-centered elliptical Gaussians with a shared rotation angle."""
    x, y = coords
    dx = x - x0
    dy = y - y0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy

    g1 = A1 * np.exp(-0.5 * ((x_rot / sigma1_x) ** 2 + (y_rot / sigma1_y) ** 2))
    g2 = A2 * np.exp(-0.5 * ((x_rot / sigma2_x) ** 2 + (y_rot / sigma2_y) ** 2))
    return g1 + g2 + B


def _edg_initial_guess(image, sigma_hint=None, center=None):
    """Build a stable first-pass parameter guess for the EDG fit."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape

    if center is None:
        center = image_center(image)

    x0, y0 = center
    sigma0 = sigma_hint if sigma_hint is not None else min(nx, ny) / 4.0
    sigma0 = float(max(sigma0, 0.5))

    shape = compute_second_moment_shape(image, center=center)
    Mxx = shape.get("Mxx", np.nan)
    Myy = shape.get("Myy", np.nan)
    Mxy = shape.get("Mxy", np.nan)

    if np.all(np.isfinite([Mxx, Myy, Mxy])) and (Mxx + Myy) > 0:
        moment_matrix = np.array([[Mxx, Mxy], [Mxy, Myy]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(moment_matrix)
        eigvals = np.clip(eigvals, 0.05**2, None)
        sigma_minor = float(np.sqrt(eigvals[0]))
        sigma_major = float(np.sqrt(eigvals[1]))
        major_vec = eigvecs[:, 1]
        theta0 = float(np.arctan2(major_vec[1], major_vec[0]))
    else:
        sigma_major = 1.1 * sigma0
        sigma_minor = 0.9 * sigma0
        theta0 = 0.0

    theta0 = ((theta0 + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
    A0 = float(np.nanmax(image)) if np.any(np.isfinite(image)) else 1.0

    return {
        "A1": A0,
        "x0": float(x0),
        "y0": float(y0),
        "sigma1_x": max(sigma_major, 0.05),
        "sigma1_y": max(sigma_minor, 0.05),
        "A2": 0.2 * A0,
        "sigma2_x": max(2.0 * sigma_major, 0.05),
        "sigma2_y": max(2.0 * sigma_minor, 0.05),
        "theta": theta0,
        "B": 0.0,
    }


def _canonicalize_edg_params(params_named):
    """Sort EDG components from narrower to broader and wrap theta consistently."""
    params_named = dict(params_named)
    theta = params_named.get("theta", 0.0)
    if np.isfinite(theta):
        params_named["theta"] = float(((theta + 0.5 * np.pi) % np.pi) - 0.5 * np.pi)

    sigma1_radius = np.sqrt(
        max(params_named.get("sigma1_x", np.nan), 0.0) * max(params_named.get("sigma1_y", np.nan), 0.0)
    )
    sigma2_radius = np.sqrt(
        max(params_named.get("sigma2_x", np.nan), 0.0) * max(params_named.get("sigma2_y", np.nan), 0.0)
    )

    if np.isfinite(sigma1_radius) and np.isfinite(sigma2_radius) and sigma2_radius < sigma1_radius:
        params_named["A1"], params_named["A2"] = params_named.get("A2", np.nan), params_named.get("A1", np.nan)
        params_named["sigma1_x"], params_named["sigma2_x"] = params_named.get("sigma2_x", np.nan), params_named.get("sigma1_x", np.nan)
        params_named["sigma1_y"], params_named["sigma2_y"] = params_named.get("sigma2_y", np.nan), params_named.get("sigma1_y", np.nan)

    return params_named


def fit_elliptical_double_gaussian_image(
    image,
    sigma_hint=None,
    center=None,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Fit an elliptical double-Gaussian model with shared center and rotation."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    yy, xx = np.indices(image.shape)
    coords = (xx.ravel(), yy.ravel())
    data = image.ravel()

    if center is None:
        center = image_center(image)

    guess = _edg_initial_guess(image, sigma_hint=sigma_hint, center=center)
    sigma_upper = max(nx, ny)
    theta_bounds = (-0.5 * np.pi, 0.5 * np.pi)

    if fit_center and fit_background:
        model_func = lambda xy, A1, x0_fit, y0_fit, s1x, s1y, A2, s2x, s2y, theta, B: elliptical_double_gaussian_2d(
            xy, A1, x0_fit, y0_fit, s1x, s1y, A2, s2x, s2y, theta, B
        )
        p0 = [guess["A1"], guess["x0"], guess["y0"], guess["sigma1_x"], guess["sigma1_y"], guess["A2"], guess["sigma2_x"], guess["sigma2_y"], guess["theta"], guess["B"]]
        bounds = (
            [0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.05, 0.05, theta_bounds[0], -0.1],
            [np.inf, nx - 1.0, ny - 1.0, sigma_upper, sigma_upper, np.inf, sigma_upper, sigma_upper, theta_bounds[1], 0.1],
        )
        param_names = ["A1", "x0", "y0", "sigma1_x", "sigma1_y", "A2", "sigma2_x", "sigma2_y", "theta", "B"]
    elif fit_center and not fit_background:
        model_func = lambda xy, A1, x0_fit, y0_fit, s1x, s1y, A2, s2x, s2y, theta: elliptical_double_gaussian_2d(
            xy, A1, x0_fit, y0_fit, s1x, s1y, A2, s2x, s2y, theta, 0.0
        )
        p0 = [guess["A1"], guess["x0"], guess["y0"], guess["sigma1_x"], guess["sigma1_y"], guess["A2"], guess["sigma2_x"], guess["sigma2_y"], guess["theta"]]
        bounds = (
            [0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.05, 0.05, theta_bounds[0]],
            [np.inf, nx - 1.0, ny - 1.0, sigma_upper, sigma_upper, np.inf, sigma_upper, sigma_upper, theta_bounds[1]],
        )
        param_names = ["A1", "x0", "y0", "sigma1_x", "sigma1_y", "A2", "sigma2_x", "sigma2_y", "theta"]
    elif not fit_center and fit_background:
        x0, y0 = center
        model_func = lambda xy, A1, s1x, s1y, A2, s2x, s2y, theta, B: elliptical_double_gaussian_2d(
            xy, A1, x0, y0, s1x, s1y, A2, s2x, s2y, theta, B
        )
        p0 = [guess["A1"], guess["sigma1_x"], guess["sigma1_y"], guess["A2"], guess["sigma2_x"], guess["sigma2_y"], guess["theta"], guess["B"]]
        bounds = (
            [0.0, 0.05, 0.05, 0.0, 0.05, 0.05, theta_bounds[0], -0.1],
            [np.inf, sigma_upper, sigma_upper, np.inf, sigma_upper, sigma_upper, theta_bounds[1], 0.1],
        )
        param_names = ["A1", "sigma1_x", "sigma1_y", "A2", "sigma2_x", "sigma2_y", "theta", "B"]
    else:
        x0, y0 = center
        model_func = lambda xy, A1, s1x, s1y, A2, s2x, s2y, theta: elliptical_double_gaussian_2d(
            xy, A1, x0, y0, s1x, s1y, A2, s2x, s2y, theta, 0.0
        )
        p0 = [guess["A1"], guess["sigma1_x"], guess["sigma1_y"], guess["A2"], guess["sigma2_x"], guess["sigma2_y"], guess["theta"]]
        bounds = (
            [0.0, 0.05, 0.05, 0.0, 0.05, 0.05, theta_bounds[0]],
            [np.inf, sigma_upper, sigma_upper, np.inf, sigma_upper, sigma_upper, theta_bounds[1]],
        )
        param_names = ["A1", "sigma1_x", "sigma1_y", "A2", "sigma2_x", "sigma2_y", "theta"]

    fit = _run_curve_fit(model_func, coords, data, p0, bounds, maxfev=40000)
    params_named = {name: float(value) for name, value in zip(param_names, fit["params"])}
    params_named = _canonicalize_edg_params(params_named)
    params_named.setdefault("A1", np.nan)
    params_named.setdefault("sigma1_x", np.nan)
    params_named.setdefault("sigma1_y", np.nan)
    params_named.setdefault("A2", np.nan)
    params_named.setdefault("sigma2_x", np.nan)
    params_named.setdefault("sigma2_y", np.nan)
    params_named.setdefault("theta", 0.0)

    # Preserve fixed-center / fixed-background values explicitly so downstream
    # master-table columns stay populated even when those terms were not fit.
    if fit_center:
        params_named.setdefault("x0", guess["x0"])
        params_named.setdefault("y0", guess["y0"])
    else:
        params_named["x0"] = float(center[0])
        params_named["y0"] = float(center[1])

    if fit_background:
        params_named.setdefault("B", guess["B"])
    else:
        params_named["B"] = 0.0

    fit["params"] = np.array([params_named[name] for name in param_names], dtype=float)

    if fit["fit_valid"]:
        try:
            model = model_func(coords, *fit["params"]).reshape(image.shape)
        except Exception as exc:
            fit["success"] = False
            fit["fit_valid"] = False
            fit["message"] = f"Model evaluation failed: {type(exc).__name__}: {exc}"
            model = None
    else:
        model = None

    return _finalize_fit_result(
        image,
        center,
        fit["params"],
        fit["cov"],
        fit["success"],
        fit["message"],
        fit["fit_valid"],
        model,
        core_radius=core_radius,
        wing_radius=wing_radius,
        params_named=params_named,
        nfev=fit["nfev"],
    )


############################################
# 6. Gauss-Hermite
############################################


def hermite_1d(x, order):
    return hermite(order)(x)


def gauss_hermite_2d(coords, A, x0, y0, sigma_x, sigma_y, h3x, h4x, h3y, h4y, B):
    x, y = coords
    dx = (x - x0) / sigma_x
    dy = (y - y0) / sigma_y
    gaussian = np.exp(-0.5 * (dx**2 + dy**2))

    modifier = (
        1.0
        + h3x * hermite_1d(dx, 3)
        + h4x * hermite_1d(dx, 4)
        + h3y * hermite_1d(dy, 3)
        + h4y * hermite_1d(dy, 4)
    )

    return A * gaussian * modifier + B


def fit_gauss_hermite(
    image,
    sigma_hint=None,
    center=None,
    fit_center=False,
    fit_background=False,
):
    """Fit a Gauss-Hermite model to one synthetic PSF image."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    yy, xx = np.indices(image.shape)
    coords = (xx.ravel(), yy.ravel())
    data = image.ravel()

    if center is None:
        center = image_center(image)

    x0, y0 = center
    sigma0 = sigma_hint if sigma_hint is not None else min(nx, ny) / 4.0
    A0 = np.max(data)
    B0 = 0.0
    h0 = 0.0

    h_bound = 0.5
    sigma_upper = max(nx, ny)

    if fit_center and fit_background:
        model_func = lambda xy, A, x0_fit, y0_fit, sx, sy, h3x, h4x, h3y, h4y, B: gauss_hermite_2d(
            xy, A, x0_fit, y0_fit, sx, sy, h3x, h4x, h3y, h4y, B
        )
        p0 = [A0, x0, y0, sigma0, sigma0, h0, h0, h0, h0, B0]
        bounds = (
            [0.0, 0.0, 0.0, 0.05, 0.05, -h_bound, -h_bound, -h_bound, -h_bound, -0.1],
            [np.inf, nx - 1.0, ny - 1.0, sigma_upper, sigma_upper, h_bound, h_bound, h_bound, h_bound, 0.1],
        )
        param_names = ["A", "x0", "y0", "sigma_x", "sigma_y", "h3x", "h4x", "h3y", "h4y", "B"]
    elif fit_center and not fit_background:
        model_func = lambda xy, A, x0_fit, y0_fit, sx, sy, h3x, h4x, h3y, h4y: gauss_hermite_2d(
            xy, A, x0_fit, y0_fit, sx, sy, h3x, h4x, h3y, h4y, 0.0
        )
        p0 = [A0, x0, y0, sigma0, sigma0, h0, h0, h0, h0]
        bounds = (
            [0.0, 0.0, 0.0, 0.05, 0.05, -h_bound, -h_bound, -h_bound, -h_bound],
            [np.inf, nx - 1.0, ny - 1.0, sigma_upper, sigma_upper, h_bound, h_bound, h_bound, h_bound],
        )
        param_names = ["A", "x0", "y0", "sigma_x", "sigma_y", "h3x", "h4x", "h3y", "h4y"]
    elif not fit_center and fit_background:
        model_func = lambda xy, A, sx, sy, h3x, h4x, h3y, h4y, B: gauss_hermite_2d(
            xy, A, x0, y0, sx, sy, h3x, h4x, h3y, h4y, B
        )
        p0 = [A0, sigma0, sigma0, h0, h0, h0, h0, B0]
        bounds = (
            [0.0, 0.05, 0.05, -h_bound, -h_bound, -h_bound, -h_bound, -0.1],
            [np.inf, sigma_upper, sigma_upper, h_bound, h_bound, h_bound, h_bound, 0.1],
        )
        param_names = ["A", "sigma_x", "sigma_y", "h3x", "h4x", "h3y", "h4y", "B"]
    else:
        model_func = lambda xy, A, sx, sy, h3x, h4x, h3y, h4y: gauss_hermite_2d(
            xy, A, x0, y0, sx, sy, h3x, h4x, h3y, h4y, 0.0
        )
        p0 = [A0, sigma0, sigma0, h0, h0, h0, h0]
        bounds = (
            [0.0, 0.05, 0.05, -h_bound, -h_bound, -h_bound, -h_bound],
            [np.inf, sigma_upper, sigma_upper, h_bound, h_bound, h_bound, h_bound],
        )
        param_names = ["A", "sigma_x", "sigma_y", "h3x", "h4x", "h3y", "h4y"]

    fit = _run_curve_fit(model_func, coords, data, p0, bounds, maxfev=20000)
    if fit["fit_valid"]:
        try:
            model = model_func(coords, *fit["params"]).reshape(image.shape)
        except Exception as exc:
            fit["success"] = False
            fit["fit_valid"] = False
            fit["message"] = f"Model evaluation failed: {type(exc).__name__}: {exc}"
            model = None
    else:
        model = None

    params_named = {name: float(value) for name, value in zip(param_names, fit["params"])}

    return fit["params"], fit["cov"], fit["success"], fit["message"], fit["fit_valid"], fit["nfev"], params_named, model


def fit_gauss_hermite_image(
    image,
    sigma_hint=None,
    center=None,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Fit and evaluate Gauss-Hermite on one synthetic PSF image."""
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    params, cov, success, message, fit_valid, nfev, params_named, model = fit_gauss_hermite(
        image,
        sigma_hint=sigma_hint,
        center=center,
        fit_center=fit_center,
        fit_background=fit_background,
    )

    return _finalize_fit_result(
        image,
        center,
        params,
        cov,
        success,
        message,
        fit_valid,
        model,
        core_radius=core_radius,
        wing_radius=wing_radius,
        params_named=params_named,
        nfev=nfev,
    )


############################################
# 6. Moffat
############################################


def moffat_2d(coords, A, x0, y0, alpha, beta, B):
    x, y = coords
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return A * (1 + r2 / alpha**2) ** (-beta) + B


def fit_moffat_image(
    image,
    sigma_hint=None,
    center=None,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Fit a Moffat profile to one synthetic PSF image."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    yy, xx = np.indices(image.shape)
    coords = (xx.ravel(), yy.ravel())
    data = image.ravel()

    if center is None:
        center = image_center(image)

    x0, y0 = center
    alpha0 = sigma_hint if sigma_hint is not None else min(nx, ny) / 4.0
    A0 = np.max(data)
    B0 = 0.0

    if fit_center and fit_background:
        model_func = lambda xy, A, x0_fit, y0_fit, alpha, beta, B: moffat_2d(xy, A, x0_fit, y0_fit, alpha, beta, B)
        p0 = [A0, x0, y0, alpha0, 3.0, B0]
        bounds = (
            [0.0, 0.0, 0.0, 0.05, 1.01, -0.1],
            [np.inf, nx - 1.0, ny - 1.0, max(nx, ny), 20.0, 0.1],
        )
        param_names = ["A", "x0", "y0", "alpha", "beta", "B"]
    elif fit_center and not fit_background:
        model_func = lambda xy, A, x0_fit, y0_fit, alpha, beta: moffat_2d(xy, A, x0_fit, y0_fit, alpha, beta, 0.0)
        p0 = [A0, x0, y0, alpha0, 3.0]
        bounds = (
            [0.0, 0.0, 0.0, 0.05, 1.01],
            [np.inf, nx - 1.0, ny - 1.0, max(nx, ny), 20.0],
        )
        param_names = ["A", "x0", "y0", "alpha", "beta"]
    elif not fit_center and fit_background:
        model_func = lambda xy, A, alpha, beta, B: moffat_2d(xy, A, x0, y0, alpha, beta, B)
        p0 = [A0, alpha0, 3.0, B0]
        bounds = ([0.0, 0.05, 1.01, -0.1], [np.inf, max(nx, ny), 20.0, 0.1])
        param_names = ["A", "alpha", "beta", "B"]
    else:
        model_func = lambda xy, A, alpha, beta: moffat_2d(xy, A, x0, y0, alpha, beta, 0.0)
        p0 = [A0, alpha0, 3.0]
        bounds = ([0.0, 0.05, 1.01], [np.inf, max(nx, ny), 20.0])
        param_names = ["A", "alpha", "beta"]

    fit = _run_curve_fit(model_func, coords, data, p0, bounds, maxfev=20000)

    if fit["fit_valid"]:
        try:
            model = model_func(coords, *fit["params"]).reshape(image.shape)
        except Exception as exc:
            fit["success"] = False
            fit["fit_valid"] = False
            fit["message"] = f"Model evaluation failed: {type(exc).__name__}: {exc}"
            model = None
    else:
        model = None

    params_named = {name: float(value) for name, value in zip(param_names, fit["params"])}

    return _finalize_fit_result(
        image,
        center,
        fit["params"],
        fit["cov"],
        fit["success"],
        fit["message"],
        fit["fit_valid"],
        model,
        core_radius=core_radius,
        wing_radius=wing_radius,
        params_named=params_named,
        nfev=fit["nfev"],
    )


############################################
# 7. Shapelets
############################################


def shapelet_1d(n, x, beta):
    h_n = hermite(n)
    norm = 1.0 / np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi) * beta)
    return norm * h_n(x / beta) * np.exp(-0.5 * (x / beta) ** 2)


def shapelet_2d(n1, n2, x, y, beta):
    return shapelet_1d(n1, x, beta) * shapelet_1d(n2, y, beta)


def build_design_matrix(x, y, beta, nmax):
    modes = []

    for n1 in range(nmax + 1):
        for n2 in range(nmax + 1 - n1):
            modes.append((n1, n2))

    npix = x.size
    ncoeff = len(modes)
    phi = np.zeros((npix, ncoeff))

    for i, (n1, n2) in enumerate(modes):
        phi[:, i] = shapelet_2d(n1, n2, x, y, beta).ravel()

    return phi, modes


def fit_shapelets(image, beta=2.0, nmax=6, center=None):
    """Fit shapelets to one synthetic PSF image."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]

    if center is None:
        center = image_center(image)

    x0, y0 = center
    xx = xx - x0
    yy = yy - y0

    phi, modes = build_design_matrix(xx, yy, beta, nmax)
    data = image.ravel()
    coeffs, *_ = np.linalg.lstsq(phi, data, rcond=None)
    model = (phi @ coeffs).reshape(image.shape)

    fit_valid = bool(np.all(np.isfinite(coeffs)) and np.all(np.isfinite(model)))

    return {
        "coeffs": coeffs,
        "modes": modes,
        "model": model,
        "center": center,
        "beta": beta,
        "nmax": nmax,
        "success": True,
        "message": "linear least squares completed",
        "fit_valid": fit_valid,
        "nfev": None,
    }


def fit_shapelet_image(
    image,
    center=None,
    beta=2.0,
    nmax=6,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Fit and evaluate shapelets on one synthetic PSF image."""
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    shapelet_result = fit_shapelets(image, beta=beta, nmax=nmax, center=center)
    model = shapelet_result["model"] if shapelet_result["fit_valid"] else None

    return _finalize_fit_result(
        image,
        center,
        params=np.array([]),
        cov=None,
        success=shapelet_result["success"],
        message=shapelet_result["message"],
        fit_valid=shapelet_result["fit_valid"],
        model=model,
        core_radius=core_radius,
        wing_radius=wing_radius,
        params_named={"beta": beta, "nmax": nmax},
        nfev=shapelet_result["nfev"],
        extra={"result": shapelet_result},
    )


############################################
# 8. Benchmark helpers
############################################


def _safe_chi2(result, key):
    """Return a model chi2 if valid, else infinity so failures cannot win."""
    fit_valid_key = f"{key}_fit_valid"
    chi2_key = f"{key}_chi2"
    if not result.get(fit_valid_key, False):
        return np.inf
    return result[chi2_key]


def pick_best_model(result):
    """Choose the best valid model using the smallest chi-like score."""
    chi2_map = {
        "gaussian": _safe_chi2(result, "gaussian"),
        "double_gaussian": _safe_chi2(result, "dg"),
        "elliptical_double_gaussian": _safe_chi2(result, "edg"),
        "gauss_hermite": _safe_chi2(result, "gh"),
        "moffat": _safe_chi2(result, "moffat"),
        "shapelet": _safe_chi2(result, "shapelet"),
    }

    finite_items = {key: value for key, value in chi2_map.items() if np.isfinite(value)}
    if not finite_items:
        return "none", chi2_map

    best_key = min(finite_items, key=finite_items.get)
    return best_key, chi2_map


def _analyze_psf_core(
    image,
    sigma_hint=None,
    center=None,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Run the shared PSF model-comparison pipeline on one 2D stamp."""
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    radius, image_profile = radial_profile(image, center=center)
    curve_radius, curve_of_growth = compute_curve_of_growth(image, center=center)
    second_moment_shape = compute_second_moment_shape(image, center=center)

    gaussian = fit_gaussian_image(
        image,
        sigma_hint=sigma_hint,
        center=center,
        fit_center=fit_center,
        fit_background=fit_background,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )
    dg = fit_double_gaussian_image(
        image,
        sigma_hint=sigma_hint,
        center=center,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )
    edg = fit_elliptical_double_gaussian_image(
        image,
        sigma_hint=sigma_hint,
        center=center,
        fit_center=fit_center,
        fit_background=fit_background,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )
    gh = fit_gauss_hermite_image(
        image,
        sigma_hint=sigma_hint,
        center=center,
        fit_center=fit_center,
        fit_background=fit_background,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )
    moffat = fit_moffat_image(
        image,
        sigma_hint=sigma_hint,
        center=center,
        fit_center=fit_center,
        fit_background=fit_background,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )
    shapelet = fit_shapelet_image(
        image,
        center=center,
        beta=shapelet_beta,
        nmax=shapelet_nmax,
        core_radius=core_radius,
        wing_radius=wing_radius,
    )

    result = {
        "center": center,
        "psf_array": image,
        "peak_value": float(np.nanmax(image)) if np.any(np.isfinite(image)) else np.nan,
        "total_flux": float(np.nansum(image)),
        "curve_of_growth_radius": curve_radius,
        "curve_of_growth": curve_of_growth,
        "ee80_radius": compute_ee_radius(image, frac=0.8, center=center),
        "fwhm_proxy": compute_fwhm_from_profile(image, center=center),
        "shape_metrics": second_moment_shape,
        "Mxx": second_moment_shape["Mxx"],
        "Myy": second_moment_shape["Myy"],
        "Mxy": second_moment_shape["Mxy"],
        "e1": second_moment_shape["e1"],
        "e2": second_moment_shape["e2"],
        "ellipticity": second_moment_shape["ellipticity"],
        "determinant_radius": second_moment_shape["determinant_radius"],
        "rp_radius": radius,
        "rp_psf": image_profile,
        "gaussian_array": gaussian["model"],
        "gaussian_params": gaussian["params"],
        "gaussian_params_named": gaussian["params_named"],
        "gaussian_residual": gaussian["residual"],
        "gaussian_chi2": gaussian["chi2"],
        "gaussian_success": gaussian["success"],
        "gaussian_message": gaussian["message"],
        "gaussian_fit_valid": gaussian["fit_valid"],
        "gaussian_metrics": gaussian["metrics"],
        "gaussian_global_mse": gaussian["global_mse"],
        "gaussian_core_mse": gaussian["core_mse"],
        "gaussian_wing_mse": gaussian["wing_mse"],
        "gaussian_profile_mse": gaussian["profile_mse"],
        "rp_gaussian": gaussian["rp_model"],
        "rp_gaussian_residual": gaussian["rp_residual"],
        "dgauss_array": dg["model"],
        "dg_params": dg["params"],
        "dg_params_named": dg["params_named"],
        "dg_residual": dg["residual"],
        "dg_chi2": dg["chi2"],
        "dg_success": dg["success"],
        "dg_message": dg["message"],
        "dg_fit_valid": dg["fit_valid"],
        "dg_metrics": dg["metrics"],
        "dg_global_mse": dg["global_mse"],
        "dg_core_mse": dg["core_mse"],
        "dg_wing_mse": dg["wing_mse"],
        "dg_profile_mse": dg["profile_mse"],
        "rp_dg": dg["rp_model"],
        "rp_dg_residual": dg["rp_residual"],
        "edg_params": edg["params"],
        "edg_params_named": edg["params_named"],
        "edg_cov": edg["cov"],
        "edg_array": edg["model"],
        "edg_residual": edg["residual"],
        "edg_chi2": edg["chi2"],
        "edg_success": edg["success"],
        "edg_message": edg["message"],
        "edg_fit_valid": edg["fit_valid"],
        "edg_metrics": edg["metrics"],
        "edg_global_mse": edg["global_mse"],
        "edg_core_mse": edg["core_mse"],
        "edg_wing_mse": edg["wing_mse"],
        "edg_profile_mse": edg["profile_mse"],
        "rp_edg": edg["rp_model"],
        "rp_edg_residual": edg["rp_residual"],
        "gh_params": gh["params"],
        "gh_params_named": gh["params_named"],
        "gh_cov": gh["cov"],
        "gh_array": gh["model"],
        "gh_residual": gh["residual"],
        "gh_chi2": gh["chi2"],
        "gh_success": gh["success"],
        "gh_message": gh["message"],
        "gh_fit_valid": gh["fit_valid"],
        "gh_metrics": gh["metrics"],
        "gh_global_mse": gh["global_mse"],
        "gh_core_mse": gh["core_mse"],
        "gh_wing_mse": gh["wing_mse"],
        "gh_profile_mse": gh["profile_mse"],
        "rp_gh": gh["rp_model"],
        "rp_gh_residual": gh["rp_residual"],
        "moffat_params": moffat["params"],
        "moffat_params_named": moffat["params_named"],
        "moffat_cov": moffat["cov"],
        "moffat_array": moffat["model"],
        "moffat_residual": moffat["residual"],
        "moffat_chi2": moffat["chi2"],
        "moffat_success": moffat["success"],
        "moffat_message": moffat["message"],
        "moffat_fit_valid": moffat["fit_valid"],
        "moffat_metrics": moffat["metrics"],
        "moffat_global_mse": moffat["global_mse"],
        "moffat_core_mse": moffat["core_mse"],
        "moffat_wing_mse": moffat["wing_mse"],
        "moffat_profile_mse": moffat["profile_mse"],
        "rp_moffat": moffat["rp_model"],
        "rp_moffat_residual": moffat["rp_residual"],
        "shapelet_result": shapelet.get("result"),
        "shapelet_params_named": shapelet["params_named"],
        "shapelet_array": shapelet["model"],
        "shapelet_residual": shapelet["residual"],
        "shapelet_chi2": shapelet["chi2"],
        "shapelet_success": shapelet["success"],
        "shapelet_message": shapelet["message"],
        "shapelet_fit_valid": shapelet["fit_valid"],
        "shapelet_metrics": shapelet["metrics"],
        "shapelet_global_mse": shapelet["global_mse"],
        "shapelet_core_mse": shapelet["core_mse"],
        "shapelet_wing_mse": shapelet["wing_mse"],
        "shapelet_profile_mse": shapelet["profile_mse"],
        "rp_shapelet": shapelet["rp_model"],
        "rp_shapelet_residual": shapelet["rp_residual"],
    }

    best_model, chi2_map = pick_best_model(result)
    result["best_model"] = best_model
    result["chi2_map"] = chi2_map

    result["delta_gaussian_vs_dg"] = result["gaussian_chi2"] - result["dg_chi2"]
    result["delta_gaussian_vs_moffat"] = result["gaussian_chi2"] - result["moffat_chi2"]
    result["delta_gaussian_vs_gh"] = result["gaussian_chi2"] - result["gh_chi2"]
    result["delta_gaussian_vs_shapelet"] = result["gaussian_chi2"] - result["shapelet_chi2"]
    result["delta_edg_vs_dg"] = result["edg_chi2"] - result["dg_chi2"]
    result["delta_edg_vs_moffat"] = result["edg_chi2"] - result["moffat_chi2"]
    result["delta_edg_vs_gh"] = result["edg_chi2"] - result["gh_chi2"]
    result["delta_edg_vs_shapelet"] = result["edg_chi2"] - result["shapelet_chi2"]

    eps = 1e-30
    reference_candidates = []
    if result["dg_fit_valid"]:
        reference_candidates.append(result["dg_chi2"])
    if result["moffat_fit_valid"]:
        reference_candidates.append(result["moffat_chi2"])

    if result["gaussian_fit_valid"] and reference_candidates:
        best_non_gaussian = min(reference_candidates)
        result["ng_score"] = np.log10((result["gaussian_chi2"] + eps) / (best_non_gaussian + eps))
    else:
        result["ng_score"] = np.nan

    non_edg_candidates = []
    for prefix in ["dg", "gh", "moffat", "shapelet"]:
        if result.get(f"{prefix}_fit_valid", False):
            non_edg_candidates.append(result[f"{prefix}_chi2"])

    if result["edg_fit_valid"] and non_edg_candidates:
        best_non_edg = min(non_edg_candidates)
        edg_floor = 1e-8
        raw_ratio = np.log10((result["edg_chi2"] + edg_floor) / (best_non_edg + edg_floor))
        result["edg_deviation_score"] = max(0.0, raw_ratio)
        result["best_non_edg_chi2"] = best_non_edg
    else:
        result["edg_deviation_score"] = np.nan
        result["best_non_edg_chi2"] = np.nan

    return result


def analyze_psf_image(
    image,
    sigma_hint=None,
    center=None,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """
    Generic PSF-stamp analysis entry point shared by synthetic and observed workflows.

    The fitting math, residual metrics, and model comparison logic live here so
    downstream wrappers can stay thin.
    """
    return _analyze_psf_core(
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


def analyze_simulated_psf(
    image,
    sigma_hint=None,
    center=None,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Backward-compatible wrapper for the synthetic benchmark notebook."""
    return analyze_psf_image(
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


def print_available_chi2(result):
    """Print the chi-like scores and fit validity for one synthetic benchmark case."""
    lines = [
        ("Gaussian", result["gaussian_chi2"], result["gaussian_fit_valid"], result["gaussian_message"]),
        ("Double Gaussian", result["dg_chi2"], result["dg_fit_valid"], result["dg_message"]),
        ("Elliptical DG", result["edg_chi2"], result["edg_fit_valid"], result["edg_message"]),
        ("Gauss-Hermite", result["gh_chi2"], result["gh_fit_valid"], result["gh_message"]),
        ("Moffat", result["moffat_chi2"], result["moffat_fit_valid"], result["moffat_message"]),
        ("Shapelet", result["shapelet_chi2"], result["shapelet_fit_valid"], result["shapelet_message"]),
    ]

    for label, chi2, fit_valid, message in lines:
        status = "valid" if fit_valid else "invalid"
        print(f"{label:16s} chi-like score: {chi2:.4e}   [{status}]")
        if not fit_valid:
            print(f"    message: {message}")

    print(f"Best model: {result['best_model']}")
    print(f"Non-Gaussian score: {result['ng_score']:.4f}")
    print(f"EDG deviation score: {result['edg_deviation_score']:.4f}")


def build_reference_metric_row(case_name, deviation_type, result):
    """Build a compact synthetic-benchmark row for later calibration tables."""
    return {
        "case_name": case_name,
        "deviation_type": deviation_type,
        "best_model": result.get("best_model"),
        "gaussian_chi2": result.get("gaussian_chi2"),
        "dg_chi2": result.get("dg_chi2"),
        "edg_chi2": result.get("edg_chi2"),
        "gh_chi2": result.get("gh_chi2"),
        "moffat_chi2": result.get("moffat_chi2"),
        "shapelet_chi2": result.get("shapelet_chi2"),
        "ng_score": result.get("ng_score"),
        "edg_deviation_score": result.get("edg_deviation_score"),
        "ellipticity": result.get("ellipticity"),
        "fwhm_proxy": result.get("fwhm_proxy"),
        "ee80_radius": result.get("ee80_radius"),
        "gaussian_wing_mse": result.get("gaussian_wing_mse"),
        "dg_wing_mse": result.get("dg_wing_mse"),
        "edg_wing_mse": result.get("edg_wing_mse"),
        "moffat_wing_mse": result.get("moffat_wing_mse"),
    }


def run_named_synthetic_case(
    image,
    case_name,
    deviation_type,
    sigma_hint=None,
    center=None,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """
    Analyze one named synthetic PSF stamp and return a compact benchmark row.

    This keeps future synthetic families lightweight: a notebook can generate a
    new PSF image, run the canonical analysis once, and immediately get a tidy
    calibration row without re-implementing table-building logic.
    """
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

    row = build_reference_metric_row(case_name, deviation_type, result)
    row["result"] = result
    return row


def build_summary_row(case, result):
    """Collect one summary row for the synthetic benchmark summary table."""
    return {
        "case_name": case["case_name"],
        "deviation_type": case["deviation_type"],
        "gaussian_chi2": result["gaussian_chi2"],
        "dg_chi2": result["dg_chi2"],
        "edg_chi2": result["edg_chi2"],
        "gh_chi2": result["gh_chi2"],
        "moffat_chi2": result["moffat_chi2"],
        "shapelet_chi2": result["shapelet_chi2"],
        "gaussian_core_mse": result["gaussian_core_mse"],
        "gaussian_wing_mse": result["gaussian_wing_mse"],
        "dg_wing_mse": result["dg_wing_mse"],
        "edg_wing_mse": result["edg_wing_mse"],
        "moffat_wing_mse": result["moffat_wing_mse"],
        "gaussian_profile_mse": result["gaussian_profile_mse"],
        "dg_profile_mse": result["dg_profile_mse"],
        "edg_profile_mse": result["edg_profile_mse"],
        "moffat_profile_mse": result["moffat_profile_mse"],
        "gaussian_fit_valid": result["gaussian_fit_valid"],
        "dg_fit_valid": result["dg_fit_valid"],
        "edg_fit_valid": result["edg_fit_valid"],
        "gh_fit_valid": result["gh_fit_valid"],
        "moffat_fit_valid": result["moffat_fit_valid"],
        "shapelet_fit_valid": result["shapelet_fit_valid"],
        "gaussian_message": result["gaussian_message"],
        "dg_message": result["dg_message"],
        "edg_message": result["edg_message"],
        "gh_message": result["gh_message"],
        "moffat_message": result["moffat_message"],
        "shapelet_message": result["shapelet_message"],
        "delta_gaussian_vs_dg": result["delta_gaussian_vs_dg"],
        "delta_gaussian_vs_moffat": result["delta_gaussian_vs_moffat"],
        "delta_gaussian_vs_gh": result["delta_gaussian_vs_gh"],
        "delta_gaussian_vs_shapelet": result["delta_gaussian_vs_shapelet"],
        "delta_edg_vs_dg": result["delta_edg_vs_dg"],
        "delta_edg_vs_moffat": result["delta_edg_vs_moffat"],
        "delta_edg_vs_gh": result["delta_edg_vs_gh"],
        "delta_edg_vs_shapelet": result["delta_edg_vs_shapelet"],
        "ng_score": result["ng_score"],
        "edg_deviation_score": result["edg_deviation_score"],
        "best_model": result["best_model"],
    }


def count_best_models(results):
    """Count how often each model wins across synthetic benchmark cases."""
    counts = Counter(r["best_model"] for r in results)
    return {name: counts.get(name, 0) for name in MODEL_ORDER}


def run_heavy_wing_parameter_scan(
    size,
    sigma_core,
    wing_strength_values,
    wing_scale_values,
    center=None,
    shapelet_beta=2.0,
    shapelet_nmax=6,
    fit_center=False,
    fit_background=False,
    core_radius=2.0,
    wing_radius=3.0,
):
    """Run a 2D heavy-wing scan and return a tidy DataFrame of benchmark results."""
    if center is None:
        center = image_center_from_shape((size, size) if isinstance(size, int) else size)

    rows = []
    for wing_strength in wing_strength_values:
        for wing_scale in wing_scale_values:
            image = make_heavy_wing_psf(
                size=size,
                sigma_core=sigma_core,
                wing_strength=wing_strength,
                wing_scale=wing_scale,
                center=center,
            )
            result = analyze_simulated_psf(
                image,
                sigma_hint=sigma_core,
                center=center,
                shapelet_beta=shapelet_beta,
                shapelet_nmax=shapelet_nmax,
                fit_center=fit_center,
                fit_background=fit_background,
                core_radius=core_radius,
                wing_radius=wing_radius,
            )

            rows.append(
                {
                    "wing_strength": wing_strength,
                    "wing_scale": wing_scale,
                    "gaussian_chi2": result["gaussian_chi2"],
                    "dg_chi2": result["dg_chi2"],
                    "edg_chi2": result["edg_chi2"],
                    "gh_chi2": result["gh_chi2"],
                    "moffat_chi2": result["moffat_chi2"],
                    "shapelet_chi2": result["shapelet_chi2"],
                    "gaussian_core_mse": result["gaussian_core_mse"],
                    "gaussian_wing_mse": result["gaussian_wing_mse"],
                    "gaussian_profile_mse": result["gaussian_profile_mse"],
                    "dg_core_mse": result["dg_core_mse"],
                    "dg_wing_mse": result["dg_wing_mse"],
                    "dg_profile_mse": result["dg_profile_mse"],
                    "edg_core_mse": result["edg_core_mse"],
                    "edg_wing_mse": result["edg_wing_mse"],
                    "edg_profile_mse": result["edg_profile_mse"],
                    "moffat_core_mse": result["moffat_core_mse"],
                    "moffat_wing_mse": result["moffat_wing_mse"],
                    "moffat_profile_mse": result["moffat_profile_mse"],
                    "gh_core_mse": result["gh_core_mse"],
                    "gh_wing_mse": result["gh_wing_mse"],
                    "shapelet_core_mse": result["shapelet_core_mse"],
                    "shapelet_wing_mse": result["shapelet_wing_mse"],
                    "gaussian_fit_valid": result["gaussian_fit_valid"],
                    "dg_fit_valid": result["dg_fit_valid"],
                    "edg_fit_valid": result["edg_fit_valid"],
                    "gh_fit_valid": result["gh_fit_valid"],
                    "moffat_fit_valid": result["moffat_fit_valid"],
                    "shapelet_fit_valid": result["shapelet_fit_valid"],
                    "best_model": result["best_model"],
                    "ng_score": result["ng_score"],
                    "edg_deviation_score": result["edg_deviation_score"],
                    "delta_gaussian_vs_dg": result["delta_gaussian_vs_dg"],
                    "delta_gaussian_vs_moffat": result["delta_gaussian_vs_moffat"],
                    "delta_edg_vs_dg": result["delta_edg_vs_dg"],
                    "delta_edg_vs_moffat": result["delta_edg_vs_moffat"],
                }
            )

    return pd.DataFrame(rows)


############################################
# 9. Plotting
############################################


def plot_case_profiles(result, case_title, log_y=False):
    """Plot the combined radial profile and profile residuals for one case."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    profile_series = [
        ("input", result["rp_psf"], "k", "o"),
        ("Gaussian", result["rp_gaussian"], MODEL_COLORS["gaussian"], MODEL_MARKERS["gaussian"]),
        ("Double Gaussian", result["rp_dg"], MODEL_COLORS["double_gaussian"], MODEL_MARKERS["double_gaussian"]),
        ("Elliptical DG", result["rp_edg"], MODEL_COLORS["elliptical_double_gaussian"], MODEL_MARKERS["elliptical_double_gaussian"]),
        ("Gauss-Hermite", result["rp_gh"], MODEL_COLORS["gauss_hermite"], MODEL_MARKERS["gauss_hermite"]),
        ("Moffat", result["rp_moffat"], MODEL_COLORS["moffat"], MODEL_MARKERS["moffat"]),
        ("Shapelet", result["rp_shapelet"], MODEL_COLORS["shapelet"], MODEL_MARKERS["shapelet"]),
    ]
    for label, profile, color, marker in profile_series:
        axes[0].plot(result["rp_radius"], profile, marker=marker, ms=4, lw=1.2, linestyle="--" if label != "input" else "-", color=color, label=label)

    axes[0].set_title(f"Radial profile: {case_title}")
    axes[0].set_xlabel("Radius [pixel]")
    axes[0].set_ylabel("Mean intensity")
    if log_y:
        axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    residual_series = [
        ("input - Gaussian", result["rp_gaussian_residual"], MODEL_COLORS["gaussian"], MODEL_MARKERS["gaussian"]),
        ("input - DG", result["rp_dg_residual"], MODEL_COLORS["double_gaussian"], MODEL_MARKERS["double_gaussian"]),
        ("input - EDG", result["rp_edg_residual"], MODEL_COLORS["elliptical_double_gaussian"], MODEL_MARKERS["elliptical_double_gaussian"]),
        ("input - GH", result["rp_gh_residual"], MODEL_COLORS["gauss_hermite"], MODEL_MARKERS["gauss_hermite"]),
        ("input - Moffat", result["rp_moffat_residual"], MODEL_COLORS["moffat"], MODEL_MARKERS["moffat"]),
        ("input - Shapelet", result["rp_shapelet_residual"], MODEL_COLORS["shapelet"], MODEL_MARKERS["shapelet"]),
    ]
    axes[1].axhline(0.0, color="k", lw=1)
    for label, residual, color, marker in residual_series:
        axes[1].plot(result["rp_radius"], residual, marker=marker, ms=4, lw=1.2, color=color, label=label)

    axes[1].set_title(f"Profile residuals: {case_title}")
    axes[1].set_xlabel("Radius [pixel]")
    axes[1].set_ylabel("Profile residual")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_residual_comparison(result, case_title):
    """Show the input image and the residual image for each model."""
    max_val = np.nanmax(np.abs(result["psf_array"]))

    fig, axes = plt.subplots(1, 7, figsize=(24, 3.6))

    im0 = axes[0].imshow(result["psf_array"], cmap="viridis", vmin=0, vmax=max_val)
    axes[0].set_title(f"Input\n{case_title}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    residual_items = [
        ("Gaussian", result["gaussian_residual"], result["gaussian_chi2"], result["gaussian_fit_valid"]),
        ("DG", result["dg_residual"], result["dg_chi2"], result["dg_fit_valid"]),
        ("EDG", result["edg_residual"], result["edg_chi2"], result["edg_fit_valid"]),
        ("GH", result["gh_residual"], result["gh_chi2"], result["gh_fit_valid"]),
        ("Moffat", result["moffat_residual"], result["moffat_chi2"], result["moffat_fit_valid"]),
        ("Shapelet", result["shapelet_residual"], result["shapelet_chi2"], result["shapelet_fit_valid"]),
    ]

    for ax, (label, residual, chi2, fit_valid) in zip(axes[1:], residual_items):
        if fit_valid and np.all(np.isfinite(residual)):
            local_scale = np.max(np.abs(residual))
            local_scale = max(local_scale, 1e-12)
            im = ax.imshow(residual, cmap="RdBu_r", vmin=-local_scale, vmax=local_scale)
            title = f"{label} residual\nchi={chi2:.3e}"
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            blank = np.zeros_like(result["psf_array"])
            im = ax.imshow(blank, cmap="Greys", vmin=0, vmax=1)
            title = f"{label} residual\nfit invalid"
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    plt.tight_layout()
    plt.show()


def plot_chi_summary(summary_df):
    """Plot the chi-like metric for each model across benchmark cases."""
    chi_plot_map = {
        "gaussian_chi2": "Gaussian",
        "dg_chi2": "Double Gaussian",
        "edg_chi2": "Elliptical Double Gaussian",
        "gh_chi2": "Gauss-Hermite",
        "moffat_chi2": "Moffat",
        "shapelet_chi2": "Shapelet",
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, label in chi_plot_map.items():
        model_key = {
            "gaussian_chi2": "gaussian",
            "dg_chi2": "double_gaussian",
            "edg_chi2": "elliptical_double_gaussian",
            "gh_chi2": "gauss_hermite",
            "moffat_chi2": "moffat",
            "shapelet_chi2": "shapelet",
        }[col]
        ax.plot(
            summary_df["case_name"],
            summary_df[col],
            marker=MODEL_MARKERS[model_key],
            lw=1.5,
            label=label,
            color=MODEL_COLORS[model_key],
        )

    ax.set_yscale("log")
    ax.set_title("Chi-like metric across synthetic benchmark cases")
    ax.set_xlabel("Case")
    ax.set_ylabel("mean(residual^2)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


def plot_best_model_counts(results):
    """Plot how often each model wins across synthetic benchmark cases."""
    best_counts = count_best_models(results)
    best_labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    best_values = [best_counts[name] for name in MODEL_ORDER]
    colors = [MODEL_COLORS[name] for name in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(best_labels, best_values, color=colors)
    ax.set_title("Best-model counts across synthetic benchmark cases")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of synthetic PSFs")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, best_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    return best_counts


def _scan_grid(df, value_col):
    """Build a pivot table for scan heatmaps."""
    pivot = df.pivot(index="wing_scale", columns="wing_strength", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def plot_best_model_heatmap(scan_df):
    """Plot the best-model classification heatmap over the heavy-wing scan."""
    best_model_to_index = {name: i for i, name in enumerate(MODEL_ORDER)}
    mapped = scan_df.copy()
    mapped["best_model_index"] = mapped["best_model"].map(best_model_to_index)
    pivot = _scan_grid(mapped, "best_model_index")

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="tab10")
    ax.set_title("Best-model heatmap")
    ax.set_xlabel("wing_strength")
    ax.set_ylabel("wing_scale")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(len(MODEL_ORDER)))
    cbar.set_ticklabels([MODEL_LABELS[name] for name in MODEL_ORDER])

    plt.tight_layout()
    plt.show()


def plot_ng_score_heatmap(scan_df):
    """Plot the non-Gaussian score heatmap over the heavy-wing scan."""
    pivot = _scan_grid(scan_df, "ng_score")

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Non-Gaussian score heatmap")
    ax.set_xlabel("wing_strength")
    ax.set_ylabel("wing_scale")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ng_score")
    plt.tight_layout()
    plt.show()
