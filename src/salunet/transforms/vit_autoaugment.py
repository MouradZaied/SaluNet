"""CIFAR-10 AutoAugment policy used by omihub777/ViT-CIFAR.

Adapted from ViT-CIFAR's MIT-licensed ``autoaugment.py`` and ``ops.py``.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class _Operation:
    def __init__(self, name: str, fillcolor=(128, 128, 128)) -> None:
        self.name = name
        self.fillcolor = fillcolor

    def __call__(self, image: Image.Image, magnitude):
        if self.name == "shearX":
            sign = random.choice((-1, 1))
            return image.transform(
                image.size, Image.AFFINE, (1, magnitude * sign, 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=self.fillcolor,
            )
        if self.name == "shearY":
            sign = random.choice((-1, 1))
            return image.transform(
                image.size, Image.AFFINE, (1, 0, 0, magnitude * sign, 1, 0),
                Image.BICUBIC, fillcolor=self.fillcolor,
            )
        if self.name == "translateX":
            sign = random.choice((-1, 1))
            return image.transform(
                image.size, Image.AFFINE,
                (1, 0, magnitude * image.size[0] * sign, 0, 1, 0),
                fillcolor=self.fillcolor,
            )
        if self.name == "translateY":
            sign = random.choice((-1, 1))
            return image.transform(
                image.size, Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * image.size[1] * sign),
                fillcolor=self.fillcolor,
            )
        if self.name == "rotate":
            rotated = image.convert("RGBA").rotate(magnitude)
            background = Image.new("RGBA", rotated.size, (128,) * 4)
            return Image.composite(rotated, background, rotated).convert(image.mode)
        if self.name == "color":
            sign = random.choice((-1, 1))
            return ImageEnhance.Color(image).enhance(1 + magnitude * sign)
        if self.name == "posterize":
            return ImageOps.posterize(image, int(magnitude))
        if self.name == "solarize":
            return ImageOps.solarize(image, magnitude)
        if self.name == "contrast":
            sign = random.choice((-1, 1))
            return ImageEnhance.Contrast(image).enhance(1 + magnitude * sign)
        if self.name == "sharpness":
            sign = random.choice((-1, 1))
            return ImageEnhance.Sharpness(image).enhance(1 + magnitude * sign)
        if self.name == "brightness":
            sign = random.choice((-1, 1))
            return ImageEnhance.Brightness(image).enhance(1 + magnitude * sign)
        if self.name == "autocontrast":
            return ImageOps.autocontrast(image)
        if self.name == "equalize":
            return ImageOps.equalize(image)
        if self.name == "invert":
            return ImageOps.invert(image)
        raise ValueError(f"Unknown AutoAugment operation: {self.name}")


class _SubPolicy:
    _RANGES = {
        "shearX": np.linspace(0, 0.3, 10),
        "shearY": np.linspace(0, 0.3, 10),
        "translateX": np.linspace(0, 150 / 331, 10),
        "translateY": np.linspace(0, 150 / 331, 10),
        "rotate": np.linspace(0, 30, 10),
        "color": np.round(np.linspace(0.0, 0.9, 10), 2).astype(np.float32),
        "posterize": np.round(np.linspace(8, 4, 10)).astype(int),
        "solarize": np.round(np.linspace(256, 0, 10), 0).astype(int),
        "contrast": np.round(np.linspace(0.0, 0.9, 10), 2).astype(np.float32),
        "sharpness": np.round(np.linspace(0.0, 0.9, 10), 2).astype(np.float32),
        "brightness": np.round(np.linspace(0.0, 0.9, 10), 2).astype(np.float32),
        "autocontrast": [0] * 10,
        "equalize": [0] * 10,
        "invert": [0] * 10,
    }

    def __init__(self, p1, op1, mag1, p2, op2, mag2, fillcolor=(128, 128, 128)):
        self.p1, self.p2 = p1, p2
        self.op1, self.op2 = _Operation(op1, fillcolor), _Operation(op2, fillcolor)
        self.mag1, self.mag2 = self._RANGES[op1][mag1], self._RANGES[op2][mag2]

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p1:
            image = self.op1(image, self.mag1)
        if random.random() < self.p2:
            image = self.op2(image, self.mag2)
        return image


class ViTCIFAR10Policy:
    """The 25 sub-policies used by the original ViT-CIFAR repository."""

    def __init__(self, fillcolor=(128, 128, 128)) -> None:
        specs = [
            (0.1,"invert",7,0.2,"contrast",6), (0.7,"rotate",2,0.3,"translateX",9),
            (0.8,"sharpness",1,0.9,"sharpness",3), (0.5,"shearY",8,0.7,"translateY",9),
            (0.5,"autocontrast",8,0.9,"equalize",2), (0.2,"shearY",7,0.3,"posterize",7),
            (0.4,"color",3,0.6,"brightness",7), (0.3,"sharpness",9,0.7,"brightness",9),
            (0.6,"equalize",5,0.5,"equalize",1), (0.6,"contrast",7,0.6,"sharpness",5),
            (0.7,"color",7,0.5,"translateX",8), (0.3,"equalize",7,0.4,"autocontrast",8),
            (0.4,"translateY",3,0.2,"sharpness",6), (0.9,"brightness",6,0.2,"color",8),
            (0.5,"solarize",2,0.0,"invert",3), (0.2,"equalize",0,0.6,"autocontrast",0),
            (0.2,"equalize",8,0.6,"equalize",4), (0.9,"color",9,0.6,"equalize",6),
            (0.8,"autocontrast",4,0.2,"solarize",8), (0.1,"brightness",3,0.7,"color",0),
            (0.4,"solarize",5,0.9,"autocontrast",3), (0.9,"translateY",9,0.7,"translateY",9),
            (0.9,"autocontrast",2,0.8,"solarize",3), (0.8,"equalize",8,0.1,"invert",3),
            (0.7,"translateY",9,0.9,"autocontrast",1),
        ]
        self.policies = [_SubPolicy(*spec, fillcolor=fillcolor) for spec in specs]

    def __call__(self, image: Image.Image) -> Image.Image:
        policy_index = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_index](image)

    def __repr__(self) -> str:
        return "ViT-CIFAR AutoAugment CIFAR10 Policy"


__all__ = ["ViTCIFAR10Policy"]
