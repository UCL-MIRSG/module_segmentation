import numpy as np
import numpy.typing as npt
from skimage.io import imread

from utils import local_minima_seeded_watershed


def segment_image(
    projected_image: npt.NDArray[np.uint8],
    spot_sigma: float = 10,
    outline_sigma: float = 0,
) -> None:
    """
    read in the projected image and segment it
    """
    local_minima_seeded_watershed(projected_image, spot_sigma, outline_sigma)


if __name__ == "__main__":
    image = imread("projected_image.tif")
    segment_image(image)
