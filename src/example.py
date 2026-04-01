import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
from scipy.optimize import curve_fit

########################################################
# Shapelet implementation (same as before)
########################################################

def shapelet_1d(n, x, beta):
    Hn = hermite(n)
    norm = 1.0 / np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi) * beta)
    return norm * Hn(x / beta) * np.exp(-0.5 * (x / beta)**2)

def shapelet_2d(n1, n2, x, y, beta):
    return shapelet_1d(n1, x, beta) * shapelet_1d(n2, y, beta)

def build_design_matrix(x, y, beta, nmax):
    modes = []
    for n1 in range(nmax + 1):
        for n2 in range(nmax + 1 - n1):
            modes.append((n1, n2))

    Npix = x.size
    Phi = np.zeros((Npix, len(modes)))

    for i, (n1, n2) in enumerate(modes):
        Phi[:, i] = shapelet_2d(n1, n2, x, y, beta).ravel()

    return Phi, modes

def fit_shapelets(image, beta=2.0, nmax=6):
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]

    # centroid
    x0 = np.sum(x * image) / np.sum(image)
    y0 = np.sum(y * image) / np.sum(image)

    x = x - x0
    y = y - y0

    Phi, modes = build_design_matrix(x, y, beta, nmax)
    data = image.ravel()

    coeffs, *_ = np.linalg.lstsq(Phi, data, rcond=None)
    model = (Phi @ coeffs).reshape(image.shape)

    return coeffs, modes, model

def gaussianity_metric(coeffs, modes):
    total = np.sum(coeffs**2)
    non_gauss = sum(
        c**2 for c, (n1, n2) in zip(coeffs, modes)
        if not (n1 == 0 and n2 == 0)
    )
    return np.sqrt(non_gauss / total)

def plot_diagnostics(image, model):
    residual = image - model

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, origin='lower')
    axes[0].set_title("Data")

    axes[1].imshow(model, origin='lower')
    axes[1].set_title("Model")

    axes[2].imshow(residual, origin='lower')
    axes[2].set_title("Residual")

    plt.tight_layout()
    plt.show()


########################################################
# 1. Generate synthetic Gaussian star
########################################################

def make_gaussian_image(
    size=25,
    A=1000,
    x0=12.3,
    y0=11.7,
    sigma=2.0,
    background=10,
    noise_std=2.0
):
    y, x = np.mgrid[:size, :size]

    image = A * np.exp(
        -((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)
    ) + background

    # add noise
    image += np.random.normal(0, noise_std, image.shape)

    ## add coma (asymmetry)
    image += 0.05 * (x - x0)**3
    # → h₃ / odd modes increase
    # → residual shows skew

    ## add heavy wings
    image += 200 * np.exp(-np.sqrt((x-x0)**2 + (y-y0)**2)/3)
    # → higher-order even modes increase
    # → G rises significantly

    # elliptical PSF
    sigma_x = 2.0
    sigma_y = 3.0
    # → order-2 modes dominate

    return image


########################################################
# Gaussian model
########################################################

def gaussian_2d(coords, A, x0, y0, sigma, B):
    x, y = coords
    return A * np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2)) + B


########################################################
# Moffat model
########################################################

def moffat_2d(coords, A, x0, y0, alpha, beta, B):
    x, y = coords
    r2 = (x-x0)**2 + (y-y0)**2
    return A * (1 + r2/alpha**2)**(-beta) + B


########################################################
# Fit helpers
########################################################

def fit_model(image, model_func, p0):
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]

    coords = (x.ravel(), y.ravel())
    data = image.ravel()

    popt, _ = curve_fit(model_func, coords, data, p0=p0, maxfev=10000)
    model = model_func(coords, *popt).reshape(image.shape)

    return popt, model


########################################################
# Radial profile
########################################################

def radial_profile(image, center):
    y, x = np.indices(image.shape)
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)

    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), image.ravel())
    nr = np.bincount(r_int.ravel())

    return tbin / np.maximum(nr, 1)


########################################################
# Comparison routine
########################################################

def compare_models(image):

    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]

    # centroid
    x0 = np.sum(x * image) / np.sum(image)
    y0 = np.sum(y * image) / np.sum(image)

    ##################################
    # Gaussian fit
    ##################################
    p0_gauss = [image.max(), x0, y0, 2.0, np.median(image)]
    pg, model_g = fit_model(image, gaussian_2d, p0_gauss)

    ##################################
    # Moffat fit
    ##################################
    p0_moff = [image.max(), x0, y0, 2.0, 3.0, np.median(image)]
    pm, model_m = fit_model(image, moffat_2d, p0_moff)

    ##################################
    # Shapelet (reuse your code)
    ##################################
    coeffs, modes, model_s = fit_shapelets(image, beta=0.8, nmax=6)

    ##################################
    # Residuals
    ##################################
    res_g = image - model_g
    res_m = image - model_m
    res_s = image - model_s

    ##################################
    # χ² (relative)
    ##################################
    chi_g = np.mean(res_g**2)
    chi_m = np.mean(res_m**2)
    chi_s = np.mean(res_s**2)

    print("\n=== Model comparison ===")
    print(f"Gaussian χ²:  {chi_g:.3f}")
    print(f"Moffat  χ²:  {chi_m:.3f}")
    print(f"Shapelet χ²: {chi_s:.3f}")

    ##################################
    # Plot images
    ##################################
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    titles = ["Data", "Gaussian", "Moffat"]
    imgs = [image, model_g, model_m]

    for i in range(3):
        axes[0, i].imshow(imgs[i], origin='lower')
        axes[0, i].set_title(titles[i])

    axes[1, 0].imshow(res_g, origin='lower')
    axes[1, 0].set_title("Residual (Gaussian)")

    axes[1, 1].imshow(res_m, origin='lower')
    axes[1, 1].set_title("Residual (Moffat)")

    axes[1, 2].imshow(res_s, origin='lower')
    axes[1, 2].set_title("Residual (Shapelet)")

    axes[2, 0].axis("off")

    ##################################
    # Radial profiles
    ##################################
    center = (x0, y0)

    rp_data = radial_profile(image, center)
    rp_g = radial_profile(model_g, center)
    rp_m = radial_profile(model_m, center)
    rp_s = radial_profile(model_s, center)

    r = np.arange(len(rp_data))

    axes[2, 1].plot(r, rp_data, label="data")
    axes[2, 1].plot(r, rp_g, label="gaussian")
    axes[2, 1].plot(r, rp_m, label="moffat")
    axes[2, 1].plot(r, rp_s, label="shapelet")
    axes[2, 1].legend()
    axes[2, 1].set_title("Radial profile")

    ##################################
    # Log residual profile (wings!)
    ##################################
    axes[2, 2].plot(r, rp_data - rp_g, label="G residual")
    axes[2, 2].plot(r, rp_data - rp_m, label="M residual")
    axes[2, 2].plot(r, rp_data - rp_s, label="S residual")
    axes[2, 2].legend()
    axes[2, 2].set_title("Profile residuals")

    plt.tight_layout()
    plt.show()

    return {
        "gaussian": pg,
        "moffat": pm,
        "shapelet_coeffs": coeffs
    }

 
