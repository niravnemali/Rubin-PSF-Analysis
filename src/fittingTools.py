import numpy as np
from scipy.optimize import curve_fit
from scipy.special import hermite

############################################
# 1. 2D Gaussian model
############################################

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, rho, B):
    x, y = coords
    
    dx = x - x0
    dy = y - y0
    
    expo = (
        dx**2 / sigma_x**2
        + dy**2 / sigma_y**2
        + 2*rho*dx*dy/(sigma_x*sigma_y)
    )
    
    return A * np.exp(-0.5 * expo) + B


############################################
# 2. Gauss-Hermite expansion
############################################

def hermite_1d(x, order):
    H = hermite(order)
    return H(x)

def gauss_hermite_2d(coords,
                     A, x0, y0,
                     sigma_x, sigma_y,
                     h3x, h4x,
                     h3y, h4y,
                     B):

    x, y = coords
    
    dx = (x - x0) / sigma_x
    dy = (y - y0) / sigma_y
    
    G = np.exp(-0.5*(dx**2 + dy**2))
    
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


############################################
# 3. Fit routine
############################################

def fit_gauss_hermite(image):

    ny, nx = image.shape
    
    y, x = np.mgrid[:ny, :nx]
    
    coords = (x.ravel(), y.ravel())
    data = image.ravel()
    
    # initial guesses
    A0 = image.max()
    x0 = nx/2
    y0 = ny/2
    
    p0 = [
        A0,
        x0,
        y0,
        2.0,   # sigma_x
        2.0,   # sigma_y
        0.0,   # h3x
        0.0,   # h4x
        0.0,   # h3y
        0.0,   # h4y
        np.median(image)
    ]
    
    popt, pcov = curve_fit(
        gauss_hermite_2d,
        coords,
        data,
        p0=p0
    )
    
    return popt, pcov


### Shapelet formalism (Refregier 2003 recap)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

########################################################
# 1. Shapelet basis functions
########################################################

def shapelet_1d(n, x, beta):
    """
    1D shapelet basis function (Refregier 2003)
    """
    Hn = hermite(n)
    
    norm = (
        1.0 /
        np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi) * beta)
    )
    
    return norm * Hn(x / beta) * np.exp(-0.5 * (x / beta)**2)


def shapelet_2d(n1, n2, x, y, beta):
    return shapelet_1d(n1, x, beta) * shapelet_1d(n2, y, beta)


########################################################
# 2. Build design matrix
########################################################

def build_design_matrix(x, y, beta, nmax):
    """
    Returns:
        Phi : (Npix, Ncoeff)
        modes : list of (n1, n2)
    """
    modes = []
    
    for n1 in range(nmax + 1):
        for n2 in range(nmax + 1 - n1):
            modes.append((n1, n2))
    
    Npix = x.size
    Ncoeff = len(modes)
    
    Phi = np.zeros((Npix, Ncoeff))
    
    for i, (n1, n2) in enumerate(modes):
        Phi[:, i] = shapelet_2d(n1, n2, x, y, beta).ravel()
    
    return Phi, modes


########################################################
# 3. Fit shapelet coefficients
########################################################

def fit_shapelets(image, beta=2.0, nmax=6):
    
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]
    
    # center coordinates
    x0 = np.sum(x * image) / np.sum(image)
    y0 = np.sum(y * image) / np.sum(image)
    
    x = x - x0
    y = y - y0
    
    Phi, modes = build_design_matrix(x, y, beta, nmax)
    
    data = image.ravel()
    
    # linear least squares
    coeffs, *_ = np.linalg.lstsq(Phi, data, rcond=None)
    
    model = (Phi @ coeffs).reshape(image.shape)
    
    return {
        "coeffs": coeffs,
        "modes": modes,
        "model": model,
        "center": (x0, y0),
        "beta": beta,
        "nmax": nmax
    }



## PSFEx-compatible interpretation

def summarize_coefficients(coeffs, modes):

    summary = {}

    for c, (n1, n2) in zip(coeffs, modes):
        order = n1 + n2
        summary.setdefault(order, []).append(c)

    for k in summary:
        summary[k] = np.array(summary[k])

    return summary


############################################
# 4. Example usage
############################################

# image = 2D numpy array cutout of star

params, cov = fit_gauss_hermite(image)

(
    A,
    x0,
    y0,
    sx,
    sy,
    h3x,
    h4x,
    h3y,
    h4y,
    B
) = params

print("Gaussianity diagnostics")
print("h3x =", h3x)
print("h3y =", h3y)
print("h4x =", h4x)
print("h4y =", h4y)

D = np.sqrt(h3x**2 + h3y**2 + h4x**2 + h4y**2)

print("overall deviation =", D)
