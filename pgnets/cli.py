"""Command line entry point:  pgnets image.fits --rp-arcsec 0.48 --mstar 0.83 -o result.png"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pgnets",
        description="Estimate the mass of a gap-opening planet (and the disc viscosity alpha) "
                    "from a dust-continuum image with the PGNets neural networks.")
    p.add_argument("image", nargs="?", help="FITS image, or an already-preprocessed 64x64 .npy file")
    p.add_argument("--download-models", action="store_true",
                   help="fetch the trained weights (only needed if you cloned without git-lfs)")
    g = p.add_argument_group("gap location (FITS input only)")
    g.add_argument("--rp-arcsec", type=float, help="radius of the gap / planet orbit in arcsec")
    g.add_argument("--rp-au", type=float, help="radius of the gap in au (use with --dist-pc)")
    g.add_argument("--dist-pc", type=float, help="distance to the source in pc")
    g.add_argument("--center", default="peak",
                   help="star position: 'peak' (brightest pixel, default), 'image' (image centre), or 'x,y' in 0-based pixels")
    g.add_argument("--pixscale", type=float, help="override pixel scale (arcsec/pixel)")
    p.add_argument("--mstar", type=float, help="stellar mass in M_sun (to report Mp in Jupiter masses)")
    p.add_argument("-o", "--output", help="output figure (default: <image>_pgnets.png)")
    p.add_argument("--save-input", help="also save the 64x64 network input as .npy")
    p.add_argument("--single-orientation", action="store_true",
                   help="use the image as given instead of averaging over its 8 rotations/reflections "
                        "(this is what the paper and notebooks did)")
    p.add_argument("--no-plot", action="store_true", help="print numbers only")
    a = p.parse_args(argv)

    if a.download_models:
        from .models import download_models
        download_models()
        if a.image is None:
            return 0
    if a.image is None:
        p.error("give an image (FITS or 64x64 .npy)")

    rp = a.rp_arcsec
    if rp is None and a.rp_au is not None:
        if a.dist_pc is None:
            p.error("--rp-au needs --dist-pc")
        rp = a.rp_au / a.dist_pc

    from .models import predict, load_model
    from .preprocess import load_input

    try:
        x = load_input(a.image, rp_arcsec=rp, center=a.center, pixscale=a.pixscale)
    except ValueError as e:
        p.error(str(e))
    if a.save_input:
        np.save(a.save_input, x)

    pred = predict(x, mstar=a.mstar, average_orientations=not a.single_orientation)
    print(f"PGNets prediction for {a.image}")
    print(pred.summary())

    if not a.no_plot:
        from .gradcam import gradcam
        from .plot import summary_figure

        cam = gradcam(load_model("regression"), x, output_index=0)  # d(Mp)/d(features)
        out = a.output or str(Path(a.image).with_suffix("")) + "_pgnets.png"
        summary_figure(x, pred, cam=cam, title=f"PGNets: {Path(a.image).stem}", path=out)
        print(f"Figure written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
