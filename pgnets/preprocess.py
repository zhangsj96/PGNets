"""Turn a dust-continuum image into the 64x64 input that the PGNets models expect.

The networks were trained on simulated images that are

* centred on the star,
* cropped to +/- 2.1 planet radii (r_p) on each side,
* sampled onto a 64 x 64 grid, and
* log-scaled as ``(log10 I - log10 I_max + 2) / 2``, clipped to [0, 1]
  (i.e. two decades of dynamic range below the peak).

``preprocess_image`` reproduces ``data/LHS43_45incl_ALMA.npy`` from
``data/LHS43_incl45.image.fits`` to a mean absolute difference of < 0.01.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

NPIX = 64
HALF_WIDTH_RP = 2.1  # crop half-width in units of the planet (gap) radius


def log_normalize(img: np.ndarray) -> np.ndarray:
    """Log-scale an intensity map to [0, 1] with two decades of dynamic range."""
    img = np.asarray(img, dtype=float)
    peak = np.nanmax(img)
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("image has no positive pixels")
    img = np.clip(np.nan_to_num(img, nan=0.0), peak * 1e-30, None)
    x = (np.log10(img) - np.log10(peak) + 2.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def read_fits(path: str):
    """Return (2-D image, pixel scale in arcsec, header) from a FITS file."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = np.squeeze(np.asarray(hdu.data, dtype=float))
        header = hdu.header
    if data.ndim != 2:
        raise ValueError(f"expected a single 2-D image, got shape {data.shape}")
    if "CDELT2" in header:
        pixscale = abs(header["CDELT2"]) * 3600.0
    elif "CD2_2" in header:
        pixscale = abs(header["CD2_2"]) * 3600.0
    else:
        raise ValueError("FITS header has no CDELT2/CD2_2; pass --pixscale")
    return data, pixscale, header


def find_center(img: np.ndarray, how: str = "peak"):
    """Pixel (y, x) of the star. ``how`` is 'peak', 'image', or 'x,y' (0-based)."""
    if how == "image":
        return (img.shape[0] - 1) / 2.0, (img.shape[1] - 1) / 2.0
    if how == "peak":
        y, x = np.unravel_index(np.nanargmax(img), img.shape)
        return float(y), float(x)
    x, y = (float(v) for v in how.split(","))
    return y, x


def crop_regrid(img, pixscale, rp_arcsec, center=None, half_width_rp=HALF_WIDTH_RP):
    """Sample ``img`` onto 64x64 pixels spanning +/- half_width_rp * r_p around ``center``."""
    if center is None:
        center = find_center(img, "peak")
    half = half_width_rp * rp_arcsec / pixscale  # half-width in input pixels
    g = np.linspace(-half, half, NPIX)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    return map_coordinates(np.nan_to_num(img), [center[0] + yy, center[1] + xx], order=1)


def preprocess_image(img, pixscale, rp_arcsec, center="peak"):
    """Full pipeline: crop to +/-2.1 r_p, regrid to 64x64, log-normalize."""
    c = find_center(img, center) if isinstance(center, str) else center
    return log_normalize(crop_regrid(img, pixscale, rp_arcsec, c))


def load_input(path, rp_arcsec=None, center="peak", pixscale=None):
    """Load a FITS image (needs r_p) or an already-preprocessed 64x64 .npy file."""
    if str(path).endswith(".npy"):
        x = np.load(path).astype("float32")
        if x.shape != (NPIX, NPIX):
            raise ValueError(f".npy input must already be 64x64, got {x.shape}")
        return x
    if rp_arcsec is None:
        raise ValueError("for a FITS image, give the gap radius (--rp-arcsec or --rp-au + --dist-pc)")
    img, ps, _ = read_fits(path)
    return preprocess_image(img, pixscale or ps, rp_arcsec, center).astype("float32")
