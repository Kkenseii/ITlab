from abc import ABC, abstractmethod

import numpy as np


def to_uint8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def get_brightness(img):
    f = img.astype(np.float64)
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def box_blur(a, r):
    if r <= 0:
        return a.astype(np.float64)
    k = 2 * r + 1
    h, w = a.shape

    p = np.pad(a.astype(np.float64), ((r, r), (0, 0)), mode="edge")
    a = sum(p[i:i + h] for i in range(k)) / k

    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    a = sum(p[:, i:i + w] for i in range(k)) / k
    return a


def make_sepia(img):
    m = np.array([[0.131, 0.534, 0.272],
                  [0.168, 0.686, 0.349],
                  [0.189, 0.769, 0.393]])
    return img.astype(np.float64) @ m.T


class ImageFilter(ABC):

    @staticmethod
    def get_filter(name, **params):
        if name not in FILTERS:
            raise ValueError(f"Неизвестный фильтр '{name}'. "
                             f"Доступные: {', '.join(sorted(FILTERS))}")

        filter_class, param_names = FILTERS[name]

        kwargs = {}
        for p in param_names:
            if params.get(p) is not None:
                kwargs[p] = params[p]

        return filter_class(**kwargs)

    @abstractmethod
    def apply_filter(self, image):
        pass


class Resize(ImageFilter):
    def __init__(self, width=None, height=None, scale=None, method="bilinear"):
        if width is None and height is None and scale is None:
            raise ValueError("Resize: укажите --width и/или --height, либо --scale")
        if method not in ("bilinear", "nearest"):
            raise ValueError("Resize: method должен быть 'bilinear' или 'nearest'")
        self.width = width
        self.height = height
        self.scale = scale
        self.method = method

    def get_new_size(self, h, w):
        if self.scale is not None:
            new_w, new_h = round(w * self.scale), round(h * self.scale)
        elif self.width is not None and self.height is not None:
            new_w, new_h = self.width, self.height
        elif self.width is not None:
            new_w, new_h = self.width, round(h * self.width / w)
        else:
            new_w, new_h = round(w * self.height / h), self.height

        if new_w < 1 or new_h < 1:
            raise ValueError("Resize: итоговый размер должен быть не меньше 1x1")
        return int(new_h), int(new_w)

    def apply_filter(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.get_new_size(h, w)

        ys = (np.arange(new_h) + 0.5) * h / new_h - 0.5
        xs = (np.arange(new_w) + 0.5) * w / new_w - 0.5

        if self.method == "nearest":
            y = np.clip(np.rint(ys), 0, h - 1).astype(int)
            x = np.clip(np.rint(xs), 0, w - 1).astype(int)
            return image[y][:, x]

        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)

        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)

        wy = (ys - y0)[:, None, None]
        wx = (xs - x0)[None, :, None]

        f = image.astype(np.float64)
        top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
        bottom = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
        return to_uint8(top * (1 - wy) + bottom * wy)


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        return to_uint8(get_brightness(image))


class Antique(ImageFilter):
    def __init__(self, intensity=1.0):
        if not 0 <= intensity <= 1:
            raise ValueError("Antique: intensity должен лежать в [0, 1]")
        self.intensity = intensity

    def apply_filter(self, image):
        original = image.astype(np.float64)
        sepia = make_sepia(image)
        return to_uint8((1 - self.intensity) * original + self.intensity * sepia)


class FadeColor(ImageFilter):
    def __init__(self, alpha=0.4, desaturation=0.5, level=210):
        if not 0 <= alpha <= 1 or not 0 <= desaturation <= 1:
            raise ValueError("FadeColor: alpha и desaturation должны лежать в [0, 1]")
        self.alpha = alpha
        self.desaturation = desaturation
        self.level = level

    def apply_filter(self, image):
        f = image.astype(np.float64)
        gray = get_brightness(image)[..., None]

        faded = gray + (1 - self.desaturation) * (f - gray)
        result = (1 - self.alpha) * faded + self.alpha * self.level
        return to_uint8(result)


class FilmEffect(ImageFilter):
    def __init__(self, grain=10.0, contrast=1.2, seed=None):
        self.grain = grain
        self.contrast = contrast
        self.seed = seed

    def apply_filter(self, image):
        f = image.astype(np.float64)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]

        nir = np.clip(0.2 * r + 0.9 * g - 0.1 * b, 0, 255)

        result = np.stack([g, r, nir], axis=-1)

        result = (result - 128.0) * self.contrast + 128.0

        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.grain, size=result.shape[:2])
        result = result + noise[..., None]
        return to_uint8(result)


class Matte(ImageFilter):
    def __init__(self, radius=0.65, feather=0.35):
        if radius < 0 or feather <= 0:
            raise ValueError("Matte: radius >= 0, feather > 0")
        self.radius = radius
        self.feather = feather

    def apply_filter(self, image):
        h, w = image.shape[:2]

        y = (np.arange(h) + 0.5 - h / 2) / (h / 2)
        x = (np.arange(w) + 0.5 - w / 2) / (w / 2)

        dist = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)

        t = np.clip((dist - self.radius) / self.feather, 0, 1)
        mask = t * t * (3 - 2 * t)

        mask = mask[..., None]
        result = image.astype(np.float64) * (1 - mask) + 255.0 * mask
        return to_uint8(result)


class AgedPhoto(ImageFilter):
    def __init__(self, noise=12.0, scratches=40, dust=300, seed=None):
        self.noise = noise
        self.scratches = int(scratches)
        self.dust = int(dust)
        self.seed = seed

    def apply_filter(self, image):
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]

        result = make_sepia(image)
        result = (result - 128.0) * 0.85 + 128.0 + 10.0

        result = result + rng.normal(0.0, self.noise, size=(h, w))[..., None]

        for _ in range(self.scratches):
            length = int(rng.integers(max(2, h // 10), max(3, int(h * 0.6))))
            y_start = int(rng.integers(0, max(1, h - length + 1)))
            x_start = int(rng.integers(0, w))

            ys = np.arange(y_start, min(y_start + length, h))
            drift = np.cumsum(rng.choice([-1, 0, 0, 1], size=len(ys))) // 2
            xs = np.clip(x_start + drift, 0, w - 1)

            color = 255.0 if rng.random() < 0.7 else 20.0
            a = rng.uniform(0.3, 0.9)
            result[ys, xs] = result[ys, xs] * (1 - a) + color * a

        if self.dust > 0:
            py = rng.integers(0, h, self.dust)
            px = rng.integers(0, w, self.dust)
            colors = np.where(rng.random(self.dust) < 0.6, 255.0, 15.0)[:, None]
            result[py, px] = result[py, px] * 0.3 + colors * 0.7

        return to_uint8(result)


class Neon(ImageFilter):
    def __init__(self, color=(255, 0, 255), glow_radius=6, glow_gain=1.5, edge_threshold=0.1):
        if not 0 <= edge_threshold < 1:
            raise ValueError("Neon: edge_threshold должен лежать в [0, 1)")
        self.color = np.array(color[::-1], dtype=np.float64)
        self.glow_radius = int(glow_radius)
        self.glow_gain = glow_gain
        self.edge_threshold = edge_threshold

    @staticmethod
    def sobel(gray):
        p = np.pad(gray, 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        return np.sqrt(gx ** 2 + gy ** 2)

    def apply_filter(self, image):
        gray = box_blur(get_brightness(image), 1)
        strength = self.sobel(gray)

        edges = np.clip(strength / (np.percentile(strength, 99) + 1e-9), 0, 1)
        edges = np.clip((edges - self.edge_threshold) / (1 - self.edge_threshold), 0, 1)

        glow = box_blur(box_blur(edges, self.glow_radius), self.glow_radius)
        glow = np.clip(glow * self.glow_gain * 3, 0, 1)

        light = np.clip(edges + 0.6 * glow, 0, 1)[..., None]
        core = (edges ** 3)[..., None]
        result = 0.12 * image.astype(np.float64) + light * self.color + core * 0.5 * (255.0 - self.color)
        return to_uint8(result)


FILTERS = {
    "resize":  (Resize,        ["width", "height", "scale", "method"]),
    "gray":    (RGB2GrayScale, []),
    "antique": (Antique,       ["intensity"]),
    "fade":    (FadeColor,     ["alpha", "desaturation", "level"]),
    "film":    (FilmEffect,    ["grain", "contrast", "seed"]),
    "matte":   (Matte,         ["radius", "feather"]),
    "aged":    (AgedPhoto,     ["noise", "scratches", "dust", "seed"]),
    "neon":    (Neon,          ["color", "glow_radius", "glow_gain", "edge_threshold"]),
}