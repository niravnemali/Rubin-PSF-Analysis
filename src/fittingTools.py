from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial, hermite

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
MODEL_ORDER = ["double_gaussian", "gauss_hermite", "shapelet"]
MODEL_LABELS = {
    "double_gaussian": "Double Gaussian",
    "gauss_hermite": "Gauss-Hermite",
    "shapelet": "Shapelet",
}


############################################
# 1. Generic utilities
############################################


def image_center(image):
    ny, nx = image.shape
    return ((nx - 1) / 2.0, (ny - 1) / 2.0)


def radial_profile(image, center=None):
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


############################################
# 2. Double Gaussian
############################################


def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, rho, B):
    x, y = coords

    dx = x - x0
    dy = y - y0

    expo = (
        dx**2 / sigma_x**2
        + dy**2 / sigma_y**2
        + 2 * rho * dx * dy / (sigma_x * sigma_y)
    )

    return A * np.exp(-0.5 * expo) + B


def double_gaussian(r, a1, sigma1, a2, sigma2):
    g1 = a1 * np.exp(-r**2 / (2.0 * sigma1**2))
    g2 = a2 * np.exp(-r**2 / (2.0 * sigma2**2))
    return g1 + g2


def fit_double_gaussian_image(image, sigma_guess=None, center=None):
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape

    if center is None:
        center = image_center(image)

    yy, xx = np.indices(image.shape)

    # Preserve the notebook's original fitting convention.
    distances = np.sqrt((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2)
    data = image.ravel()

    if sigma_guess is None:
        sigma_guess = min(nx, ny) / 4.0

    p0 = [
        np.max(data),
        sigma_guess,
        np.max(data) * 0.1,
        sigma_guess * 2.0,
    ]
    bounds_low = [0, 0.1, 0, 0.1]
    bounds_high = [np.inf, nx, np.inf, nx]

    try:
        params, cov = curve_fit(
            double_gaussian,
            distances.ravel(),
            data,
            p0=p0,
            bounds=(bounds_low, bounds_high),
            maxfev=5000,
        )
    except RuntimeError:
        params = np.asarray(p0, dtype=float)
        cov = None

    model = double_gaussian(distances, *params).reshape(image.shape)
    residual = image - model

    radius, image_profile = radial_profile(image, center=center)
    _, model_profile = radial_profile(model, center=center)

    return {
        "params": params,
        "cov": cov,
        "model": model,
        "residual": residual,
        "chi2": np.mean(residual**2),
        "rp_radius": radius,
        "rp_image": image_profile,
        "rp_model": model_profile,
        "rp_residual": image_profile - model_profile,
    }


############################################
# 3. Gauss-Hermite
############################################


def hermite_1d(x, order):
    H = hermite(order)
    return H(x)


def gauss_hermite_2d(
    coords,
    A,
    x0,
    y0,
    sigma_x,
    sigma_y,
    h3x,
    h4x,
    h3y,
    h4y,
    B,
):
    x, y = coords

    dx = (x - x0) / sigma_x
    dy = (y - y0) / sigma_y

    G = np.exp(-0.5 * (dx**2 + dy**2))

    H3x = hermite_1d(dx, 3)
    H4x = hermite_1d(dx, 4)
    H3y = hermite_1d(dy, 3)
    H4y = hermite_1d(dy, 4)

    modifier = (
        1
        + h3x * H3x
        + h4x * H4x
        + h3y * H3y
        + h4y * H4y
    )

    return A * G * modifier + B


def fit_gauss_hermite(image):
    ny, nx = image.shape

    y, x = np.mgrid[:ny, :nx]
    coords = (x.ravel(), y.ravel())
    data = image.ravel()

    A0 = image.max()
    x0 = nx / 2
    y0 = ny / 2

    p0 = [
        A0,
        x0,
        y0,
        2.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.median(image),
    ]

    popt, pcov = curve_fit(
        gauss_hermite_2d,
        coords,
        data,
        p0=p0,
    )

    return popt, pcov


def fit_gauss_hermite_image(image, center=None):
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    yy, xx = np.indices(image.shape)
    params, cov = fit_gauss_hermite(image)
    model = gauss_hermite_2d((xx.ravel(), yy.ravel()), *params).reshape(image.shape)
    residual = image - model

    radius, image_profile = radial_profile(image, center=center)
    _, model_profile = radial_profile(model, center=center)

    return {
        "params": params,
        "cov": cov,
        "model": model,
        "residual": residual,
        "chi2": np.mean(residual**2),
        "rp_radius": radius,
        "rp_image": image_profile,
        "rp_model": model_profile,
        "rp_residual": image_profile - model_profile,
    }


############################################
# 4. Shapelets
############################################


def shapelet_1d(n, x, beta):
    Hn = hermite(n)
    norm = 1.0 / np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi) * beta)
    return norm * Hn(x / beta) * np.exp(-0.5 * (x / beta) ** 2)


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


def fit_shapelets(image, beta=2.0, nmax=6):
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]

    x0 = np.sum(x * image) / np.sum(image)
    y0 = np.sum(y * image) / np.sum(image)

    x = x - x0
    y = y - y0

    phi, modes = build_design_matrix(x, y, beta, nmax)
    data = image.ravel()
    coeffs, *_ = np.linalg.lstsq(phi, data, rcond=None)
    model = (phi @ coeffs).reshape(image.shape)

    return {
        "coeffs": coeffs,
        "modes": modes,
        "model": model,
        "center": (x0, y0),
        "beta": beta,
        "nmax": nmax,
    }


def fit_shapelet_image(image, center=None, beta=2.0, nmax=6):
    image = np.asarray(image, dtype=float)

    if center is None:
        center = image_center(image)

    shapelet_result = fit_shapelets(image, beta=beta, nmax=nmax)
    model = shapelet_result["model"]
    residual = image - model

    radius, image_profile = radial_profile(image, center=center)
    _, model_profile = radial_profile(model, center=center)

    return {
        "result": shapelet_result,
        "model": model,
        "residual": residual,
        "chi2": np.mean(residual**2),
        "rp_radius": radius,
        "rp_image": image_profile,
        "rp_model": model_profile,
        "rp_residual": image_profile - model_profile,
    }


def summarize_coefficients(coeffs, modes):
    summary = {}

    for c, (n1, n2) in zip(coeffs, modes):
        order = n1 + n2
        summary.setdefault(order, []).append(c)

    for k in summary:
        summary[k] = np.array(summary[k])

    return summary


############################################
# 5. Model comparison helpers
############################################


def pick_best_model(result):
    chi2_map = {
        "double_gaussian": result["dg_chi2"],
        "gauss_hermite": result["gh_chi2"],
        "shapelet": result["shapelet_chi2"],
    }
    best_key = min(chi2_map, key=chi2_map.get)
    return best_key, chi2_map


def analyze_psf_models(image, x=None, y=None, fwhm=None, shapelet_beta=2.0, shapelet_nmax=6):
    image = np.asarray(image, dtype=float)
    center = image_center(image)
    radius, psf_profile = radial_profile(image, center=center)

    sigma_guess = None
    if fwhm is not None:
        sigma_guess = fwhm / SIGMA_TO_FWHM

    dg = fit_double_gaussian_image(image, sigma_guess=sigma_guess, center=center)
    gh = fit_gauss_hermite_image(image, center=center)
    shapelet = fit_shapelet_image(
        image,
        center=center,
        beta=shapelet_beta,
        nmax=shapelet_nmax,
    )

    result = {
        "x": x,
        "y": y,
        "psf_array": image,
        "fwhm": fwhm,
        "max_val": np.max(np.abs(image)),
        "rp_radius": radius,
        "rp_psf": psf_profile,
        "dgauss_array": dg["model"],
        "dg_params": dg["params"],
        "dg_residual": dg["residual"],
        "dg_chi2": dg["chi2"],
        "rp_dg": dg["rp_model"],
        "rp_dg_residual": dg["rp_residual"],
        "gh_params": gh["params"],
        "gh_cov": gh["cov"],
        "gh_array": gh["model"],
        "gh_residual": gh["residual"],
        "gh_chi2": gh["chi2"],
        "rp_gh": gh["rp_model"],
        "rp_gh_residual": gh["rp_residual"],
        "shapelet_result": shapelet["result"],
        "shapelet_array": shapelet["model"],
        "shapelet_residual": shapelet["residual"],
        "shapelet_chi2": shapelet["chi2"],
        "rp_shapelet": shapelet["rp_model"],
        "rp_shapelet_residual": shapelet["rp_residual"],
    }

    best_model, chi2_map = pick_best_model(result)
    result["best_model"] = best_model
    result["chi2_map"] = chi2_map

    return result


def count_best_models(results):
    counts = Counter(r["best_model"] for r in results)
    return {name: counts.get(name, 0) for name in MODEL_ORDER}


############################################
# 6. Plotting
############################################


def plot_model_comparison_page(results, start, page_size, visit_id, detector_id, band):
    end = min(start + page_size, len(results))
    n = end - start
    if n <= 0:
        return

    fig, axes = plt.subplots(n, 9, figsize=(34, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "PSF vs Double Gaussian / Gauss-Hermite / Shapelet - Visit %s, Detector %s, Band %s (stars %d-%d of %d)"
        % (visit_id, detector_id, band, start + 1, end, len(results)),
        fontsize=14,
        y=1.01,
    )

    for idx, i in enumerate(range(start, end)):
        r = results[i]
        max_val = r["max_val"]
        res_scale = max_val / 10
        gh = r["gh_params"]
        shapelet_meta = r["shapelet_result"]

        title_psf = "Star %d: PSF\n(x=%.0f, y=%.0f)\nFWHM=%.2f pix\nBest=%s" % (
            i + 1,
            r["x"],
            r["y"],
            r["fwhm"],
            MODEL_LABELS[r["best_model"]],
        )
        im0 = axes[idx, 0].imshow(
            r["psf_array"],
            vmin=-max_val,
            vmax=max_val,
            cmap="viridis",
            origin="lower",
        )
        axes[idx, 0].set_title(title_psf, fontsize=9)
        fig.colorbar(im0, ax=axes[idx, 0], shrink=0.75)

        title_dg = "Double Gaussian\nA1=%.4f, s1=%.2f\nA2=%.4f, s2=%.2f" % (
            r["dg_params"][0],
            r["dg_params"][1],
            r["dg_params"][2],
            r["dg_params"][3],
        )
        im1 = axes[idx, 1].imshow(
            r["dgauss_array"],
            vmin=-max_val,
            vmax=max_val,
            cmap="viridis",
            origin="lower",
        )
        axes[idx, 1].set_title(title_dg, fontsize=9)
        fig.colorbar(im1, ax=axes[idx, 1], shrink=0.75)

        im2 = axes[idx, 2].imshow(
            r["dg_residual"],
            vmin=-res_scale,
            vmax=res_scale,
            cmap="RdBu_r",
            origin="lower",
        )
        axes[idx, 2].set_title("DG Residual\nchi^2=%.4e" % r["dg_chi2"], fontsize=9)
        fig.colorbar(im2, ax=axes[idx, 2], shrink=0.75)

        axes[idx, 3].plot(r["rp_radius"], r["rp_psf"], "o-", ms=2.5, lw=1.0, label="PSF")
        axes[idx, 3].plot(r["rp_radius"], r["rp_dg"], "s--", ms=2.5, lw=1.0, label="DG")
        title_gh = "Gauss-Hermite\nh3x=%.4f, h4x=%.4f\nh3y=%.4f, h4y=%.4f" % (
            gh[5],
            gh[6],
            gh[7],
            gh[8],
        )
        im3 = axes[idx, 3].imshow(
            r["gh_array"],
            vmin=-max_val,
            vmax=max_val,
            cmap="viridis",
            origin="lower",
        )
        axes[idx, 3].set_title(title_gh, fontsize=9)
        fig.colorbar(im3, ax=axes[idx, 3], shrink=0.75)

        im4 = axes[idx, 4].imshow(
            r["gh_residual"],
            vmin=-res_scale,
            vmax=res_scale,
            cmap="RdBu_r",
            origin="lower",
        )
        axes[idx, 4].set_title("GH Residual\nchi^2=%.4e" % r["gh_chi2"], fontsize=9)
        fig.colorbar(im4, ax=axes[idx, 4], shrink=0.75)

        title_shapelet = "Shapelet\nbeta=%.2f, nmax=%d" % (
            shapelet_meta["beta"],
            shapelet_meta["nmax"],
        )
        im5 = axes[idx, 5].imshow(
            r["shapelet_array"],
            vmin=-max_val,
            vmax=max_val,
            cmap="viridis",
            origin="lower",
        )
        axes[idx, 5].set_title(title_shapelet, fontsize=9)
        fig.colorbar(im5, ax=axes[idx, 5], shrink=0.75)

        im6 = axes[idx, 6].imshow(
            r["shapelet_residual"],
            vmin=-res_scale,
            vmax=res_scale,
            cmap="RdBu_r",
            origin="lower",
        )
        axes[idx, 6].set_title(
            "Shapelet Residual\nchi^2=%.4e" % r["shapelet_chi2"],
            fontsize=9,
        )
        fig.colorbar(im6, ax=axes[idx, 6], shrink=0.75)

        axes[idx, 7].plot(
            r["rp_radius"],
            r["rp_psf"],
            "o-",
            ms=2.5,
            lw=1.0,
            label="PSF",
        )
        axes[idx, 7].plot(
            r["rp_radius"],
            r["rp_dg"],
            "s--",
            ms=2.5,
            lw=1.0,
            label="Double Gaussian",
        )
        axes[idx, 7].plot(
            r["rp_radius"],
            r["rp_gh"],
            "^--",
            ms=2.5,
            lw=1.0,
            label="Gauss-Hermite",
        )
        axes[idx, 7].plot(
            r["rp_radius"],
            r["rp_shapelet"],
            "d--",
            ms=2.5,
            lw=1.0,
            label="Shapelet",
        )
        axes[idx, 7].set_title("Radial profile", fontsize=9)
        axes[idx, 7].set_xlabel("Radius [pixel]")
        axes[idx, 7].set_ylabel("Mean intensity")
        axes[idx, 7].grid(alpha=0.3)
        axes[idx, 7].legend(fontsize=7)

        axes[idx, 8].axhline(0.0, color="k", lw=1)
        axes[idx, 8].plot(
            r["rp_radius"],
            r["rp_dg_residual"],
            "s-",
            ms=2.5,
            lw=1.0,
            label="PSF - DG",
        )
        axes[idx, 8].plot(
            r["rp_radius"],
            r["rp_gh_residual"],
            "^-",
            ms=2.5,
            lw=1.0,
            label="PSF - GH",
        )
        axes[idx, 8].plot(
            r["rp_radius"],
            r["rp_shapelet_residual"],
            "d-",
            ms=2.5,
            lw=1.0,
            label="PSF - Shapelet",
        )
        axes[idx, 8].set_title("Profile residuals", fontsize=9)
        axes[idx, 8].set_xlabel("Radius [pixel]")
        axes[idx, 8].set_ylabel("Profile residual")
        axes[idx, 8].grid(alpha=0.3)
        axes[idx, 8].legend(fontsize=7)

        for ax in axes[idx, :3]:
            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("y (pixel)")
        axes[idx, 3].set_xlabel("x (pixel)")
        axes[idx, 3].set_ylabel("y (pixel)")
        axes[idx, 4].set_xlabel("x (pixel)")
        axes[idx, 4].set_ylabel("y (pixel)")
        axes[idx, 5].set_xlabel("x (pixel)")
        axes[idx, 5].set_ylabel("y (pixel)")
        axes[idx, 6].set_xlabel("x (pixel)")
        axes[idx, 6].set_ylabel("y (pixel)")

    plt.tight_layout()
    plt.show()


def plot_model_comparison_pages(results, page_size, visit_id, detector_id, band):
    for page_start in range(0, len(results), page_size):
        plot_model_comparison_page(results, page_start, page_size, visit_id, detector_id, band)


def plot_best_model_counts(results):
    counts = count_best_models(results)
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    values = [counts[name] for name in MODEL_ORDER]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Best-Fit Model Counts")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of PSF stamps")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            str(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    return counts


if __name__ == "__main__":
    print("fittingTools.py is intended to be imported from a notebook or script.")
    print("Example:")
    print("  from fittingTools import fit_gauss_hermite, fit_shapelets")
