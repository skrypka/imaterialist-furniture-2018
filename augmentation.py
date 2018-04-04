import torchvision.transforms.functional as F


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
