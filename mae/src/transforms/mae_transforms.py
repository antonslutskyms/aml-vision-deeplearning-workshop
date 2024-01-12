import numpy as np
import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        """
        Square pads an image using whitespace.

        Returns:
            PIL Image or Tensor
        """

        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        return F.pad(image, padding, 0, "constant")