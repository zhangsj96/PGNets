"""Load the trained PGNets networks and convert their outputs to physical quantities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_DIR = Path(os.environ.get("PGNETS_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_PATHS = {
    "resnet": MODEL_DIR / "resnet" / "model",          # 5-class classifier (softmax)
    "vgg": MODEL_DIR / "VGG_like" / "model",           # 5-class classifier (logits)
    "regression": MODEL_DIR / "resnet_regression",     # ResNet regressor -> (Mp, alpha)
}

# Classification grid of Zhang et al. (2018): planet-to-star mass ratios
MASS_RATIOS = np.array([3.3e-5, 1e-4, 3.3e-4, 1e-3, 3.3e-3])
MASS_LABELS = ["11 M⊕", "33 M⊕", "0.33 MJ", "1 MJ", "3.3 MJ"]  # for a 1 M_sun star

# Regression normalization used in training
LOG_Q_MIN, LOG_Q_MAX = np.log10(3.3e-5), np.log10(3.3e-3)
LOG_ALPHA_MIN, LOG_ALPHA_MAX = -4.0, -2.0

M_JUP_IN_MSUN = 9.54265748e-4
M_EARTH_IN_MSUN = 3.0034896e-6


LFS_URL = "https://media.githubusercontent.com/media/zhangsj96/PGNets/master/"


def _is_lfs_pointer(f: Path) -> bool:
    return f.exists() and f.stat().st_size < 1000 and f.read_bytes().startswith(b"version https://git-lfs")


def download_models(force: bool = False):
    """Replace Git-LFS pointer files with the real weights (for clones made without git-lfs)."""
    import urllib.request

    root = MODEL_DIR.parent
    for f in sorted(MODEL_DIR.glob("*/model/variables/variables.*")):
        if force or _is_lfs_pointer(f):
            rel = f.relative_to(root).as_posix()
            print(f"downloading {rel} ...")
            tmp = f.with_suffix(f.suffix + ".part")
            urllib.request.urlretrieve(LFS_URL + rel, tmp)
            tmp.replace(f)
    print("model weights ready in", MODEL_DIR)


@lru_cache(maxsize=None)
def load_model(name: str):
    import tensorflow as tf

    path = MODEL_PATHS[name]
    if path.is_dir() and _is_lfs_pointer(path / "variables" / "variables.index"):
        raise RuntimeError(
            f"{path} contains Git LFS pointers, not weights. Run `git lfs pull` "
            "or `pgnets --download-models`.")
    return tf.keras.models.load_model(str(path), compile=False)


@dataclass
class Prediction:
    resnet_probs: np.ndarray
    vgg_probs: np.ndarray
    q: float                 # regression planet-to-star mass ratio
    alpha: float             # regression disc viscosity
    mstar: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def resnet_class(self) -> int:
        return int(np.argmax(self.resnet_probs))

    @property
    def vgg_class(self) -> int:
        return int(np.argmax(self.vgg_probs))

    def mass_str(self, q: float) -> str:
        if self.mstar is None:
            return f"Mp/M* = {q:.2e}"
        mp = q * self.mstar
        return f"Mp = {mp / M_JUP_IN_MSUN:.2f} M_J ({mp / M_EARTH_IN_MSUN:.0f} M⊕)"

    def summary(self) -> str:
        star = "" if self.mstar is None else f" (M* = {self.mstar} M_sun)"
        sq = self.extra.get("logq_std", 0.0)
        sa = self.extra.get("logalpha_std", 0.0)
        n = self.extra.get("n_orientations", 1)
        lines = [
            f"Regression (ResNet): Mp/M* = {self.q:.2e}"
            + ("" if self.mstar is None else f"  ->  {self.mass_str(self.q)}{star}"),
            f"                     alpha = {self.alpha:.1e}",
            (f"                     scatter over {n} image orientations: {sq:.2f} dex (Mp), {sa:.2f} dex (alpha)"
             if n > 1 else "                     (single orientation)"),
            "                     network test-set error: 0.16 dex (Mp), 0.23 dex (alpha)",
            f"Classifier (ResNet): {MASS_LABELS[self.resnet_class]:>8s}-class "
            f"(Mp/M* = {MASS_RATIOS[self.resnet_class]:.1e}), p = {self.resnet_probs[self.resnet_class]:.2f}",
            f"Classifier (VGG):    {MASS_LABELS[self.vgg_class]:>8s}-class "
            f"(Mp/M* = {MASS_RATIOS[self.vgg_class]:.1e}), p = {self.vgg_probs[self.vgg_class]:.2f}",
        ]
        return "\n".join(lines)


def _softmax(z):
    z = np.asarray(z, dtype=float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def decode_regression(y):
    q = 10 ** (y[0] * (LOG_Q_MAX - LOG_Q_MIN) + LOG_Q_MIN)
    alpha = 10 ** (y[1] * (LOG_ALPHA_MAX - LOG_ALPHA_MIN) + LOG_ALPHA_MIN)
    return float(q), float(alpha)


def orientations(x):
    """The 8 rotations/reflections of a square image (the dihedral group D4)."""
    out = [np.rot90(x, k) for k in range(4)]
    return out + [np.fliplr(o) for o in out]


def predict(x64: np.ndarray, mstar: float | None = None, average_orientations: bool = True) -> Prediction:
    """Run all three networks on one preprocessed 64x64 image.

    The networks are not exactly rotation/reflection invariant (the prediction for one
    disc can move by ~0.1-0.2 dex when the image is rotated). By default we therefore
    evaluate all 8 orientations and report the mean (in log space) and the scatter.
    """
    views = orientations(x64) if average_orientations else [x64]
    batch = np.stack([np.asarray(v, dtype="float32") for v in views])[..., None]
    resnet = load_model("resnet").predict(batch, verbose=0).mean(axis=0)
    vgg = np.mean([_softmax(z) for z in load_model("vgg").predict(batch, verbose=0)], axis=0)
    reg = np.array([decode_regression(y) for y in load_model("regression").predict(batch, verbose=0)])
    logq, loga = np.log10(reg[:, 0]), np.log10(reg[:, 1])
    extra = {"logq_std": float(logq.std()), "logalpha_std": float(loga.std()), "n_orientations": len(views),
             "q_each": reg[:, 0].tolist(), "alpha_each": reg[:, 1].tolist()}
    return Prediction(resnet_probs=np.asarray(resnet, float), vgg_probs=vgg,
                      q=float(10 ** logq.mean()), alpha=float(10 ** loga.mean()), mstar=mstar, extra=extra)
