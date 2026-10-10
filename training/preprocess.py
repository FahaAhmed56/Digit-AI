"""Shared preprocessing: the exact same steps run here (for training) and in web/nn.js (in the app).

Using one pipeline for both avoids "train/serve skew": a model that is excellent on its
training data but fails on real input because that input was prepared differently.
"""
import numpy as np
from scipy import ndimage

BLUR = 1.5   # must match BLUR in web/nn.js


def normalize(img: np.ndarray) -> np.ndarray:
    """Any-size grayscale image (0..1) -> 64 values: crop to ink, fit longest side to 28px
    inside a 32x32 box, blur slightly, average 4x4 blocks into 8x8, then scale so the darkest cell is 1."""
    ys, xs = np.nonzero(img > 0.1)
    if len(xs) == 0:
        return np.zeros(64)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 28 / max(crop.shape)
    small = np.clip(ndimage.zoom(crop, scale, order=1), 0, 1)
    h, w = small.shape
    box = np.zeros((32, 32))
    top, left = (32 - h) // 2, (32 - w) // 2
    box[top:top + h, left:left + w] = small[:32, :32]
    box = ndimage.gaussian_filter(box, BLUR)        # thickens thin pen strokes to match the dataset
    grid = box.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    return (grid / grid.max()).ravel()


def augment(img8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """8x8 dataset digit -> randomly distorted 48x48 image, imitating how people draw:
    different widths, slants, rotations and pen thicknesses."""
    big = np.clip(ndimage.zoom(img8.reshape(8, 8), 4, order=1), 0, 1)     # 32x32
    canvas = np.pad(big, 8)                                               # 48x48, room to move
    # width range goes down to 0.4 so the model also sees thin shapes, e.g. a "1" drawn as one line
    sx, sy = rng.uniform(0.4, 1.25), rng.uniform(0.85, 1.15)
    shear, angle = rng.uniform(-0.3, 0.3), np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = np.linalg.inv(rot @ np.array([[sy, 0], [shear, sx]]))
    center = np.array([24, 24])
    out = ndimage.affine_transform(canvas, matrix, offset=center - matrix @ center, order=1)
    r = rng.random()
    if r < 0.3:
        out = ndimage.grey_dilation(out, size=(3, 3))      # thicker pen
    elif r < 0.5:
        out = ndimage.grey_erosion(out, size=(2, 2))       # thinner pen
    return np.clip(out, 0, 1)
