"""Preprocessing used at inference. This is a verbatim port of the training notebook
(INTER_AREA resize -> Macenko stain normalisation with the fixed reference). Do not
change it independently of the notebook: any drift silently degrades the model."""
import io
import warnings
import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

# scikit-image 0.26 renames these arguments; behaviour for our calls is unchanged. Keep the log readable.
warnings.filterwarnings("ignore", message=".*deprecated since version 0.26.0.*", category=FutureWarning)

HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])


class InvalidImage(ValueError):
    pass


def tissue_mask(img):
    """Otsu on saturation (glass is unsaturated). True = tissue."""
    sat = rgb2hsv(img)[..., 1]
    if (sat < 0.05).mean() < 0.02:  # essentially no glass in the field
        return np.ones(sat.shape, bool)
    m = sat > threshold_otsu(sat)
    m = remove_small_objects(m, 64)
    m = remove_small_holes(m, 64)
    return m


def macenko(img, Io=240, alpha=1, beta=0.15):
    try:
        I = img.reshape(-1, 3).astype(np.float64)
        OD = -np.log((I + 1) / Io)
        ODhat = OD[(OD > beta).all(axis=1)]
        if len(ODhat) < 200:
            return img
        _, V = np.linalg.eigh(np.cov(ODhat.T))
        V = V[:, 1:3]
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        That = ODhat @ V
        phi = np.arctan2(That[:, 1], That[:, 0])
        a, b = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
        v1 = V @ np.array([np.cos(a), np.sin(a)])
        v2 = V @ np.array([np.cos(b), np.sin(b)])
        HE = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        maxC = np.percentile(C, 99, axis=1)
        C2 = C / (maxC / MAXC_REF)[:, None]
        out = Io * np.exp(-HE_REF @ C2)
        return np.clip(out.T.reshape(img.shape), 0, 255).astype(np.uint8)
    except Exception:
        return img


def preprocess_image(img_rgb_uint8, hw):
    h, w = hw
    img = cv2.resize(img_rgb_uint8, (w, h), interpolation=cv2.INTER_AREA)
    tissue_fraction = float(tissue_mask(img).mean())
    return macenko(img), tissue_fraction


def decode_image(data: bytes, hw, max_pixels: int):
    """Safely decode untrusted bytes to RGB uint8. Returns (array, original_size)."""
    h, w = hw
    try:
        im = Image.open(io.BytesIO(data))
        fmt = im.format
        if fmt not in ("JPEG", "PNG"):
            raise InvalidImage("unsupported_format")
        ow, oh = im.size
        if ow * oh > max_pixels:
            raise InvalidImage("image_too_large")
        im.draft("RGB", (w * 2, h * 2))  # same as training loader (JPEG DCT downscale)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        arr = np.asarray(im.convert("RGB"))
    except InvalidImage:
        raise
    except Exception as e:  # corrupt/truncated/bomb -> generic error, no detail leak
        raise InvalidImage("undecodable_image") from e
    return arr, (ow, oh)
