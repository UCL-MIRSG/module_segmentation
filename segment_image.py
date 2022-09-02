import numpy as np
import numpy.typing as npt
from skimage.io import imread

from utils import thresholded_local_minima_seeded_watershed, unlabel_poor_seeds_in_frame


def segment_image(
    projected_image: npt.NDArray[np.uint8],
    spot_sigma: float = 3,
    outline_sigma: float = 0,
    minimum_intensity: float = 30,
    min_seed_boundary_ratio: float = 0.1,
) -> None:
    """
    read in the projected image and segment it
    """
    seeds, labels = thresholded_local_minima_seeded_watershed(
        projected_image,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        minimum_intensity=minimum_intensity,
    )
    seeds, labels = unlabel_poor_seeds_in_frame(
        projected_image,
        seeds,
        labels,
        outline_sigma=outline_sigma,
        minimum_intensity=minimum_intensity,
        min_seed_boundary_ratio=min_seed_boundary_ratio,
    )
    np.testing.assert_equal(seeds.sum(), 140)
    print()


if __name__ == "__main__":
    image = imread("example_projected_image.tif")
    segment_image(image)
