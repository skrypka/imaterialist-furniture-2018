import numbers

import torchvision.transforms.functional as F
from torchvision.transforms import transforms


class HorizontalFlip(object):
    """Horizontally flip the given PIL Image."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Flipped image.
        """
        return F.hflip(img)


def five_crop(img, size, crop_pos):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    if crop_pos == 0:
        return img.crop((0, 0, crop_w, crop_h))
    elif crop_pos == 1:
        return img.crop((w - crop_w, 0, w, crop_h))
    elif crop_pos == 2:
        return img.crop((0, h - crop_h, crop_w, h))
    elif crop_pos == 3:
        return img.crop((w - crop_w, h - crop_h, w, h))
    else:
        return F.center_crop(img, (crop_h, crop_w))


class FiveCropParametrized(object):
    def __init__(self, size, crop_pos):
        self.size = size
        self.crop_pos = crop_pos
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return five_crop(img, self.size, self.crop_pos)


def five_crops(size):
    return [FiveCropParametrized(size, i) for i in range(5)]


def make_transforms(first_part, second_part, inners):
    return [transforms.Compose(first_part + [inner] + second_part) for inner in inners]
