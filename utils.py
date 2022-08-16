# Functions in this file have been taken from
# https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/napari_segment_blobs_and_things_with_membranes/__init__.py
import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import local_minima
from skimage.segmentation import watershed


def local_minima_seeded_watershed(
    image: npt.NDArray[np.uint8], spot_sigma: float, outline_sigma
) -> npt.NDArray[np.int32]:
    """
    Segment cells in images with fluorescently marked membranes. The two sigma
    parameters allow tuning the segmentation result.The first sigma controls how
    close detected cells can be (spot_sigma) and the second controls how precise
    segmented objects are outlined (outline_sigma). Under the hood, this filter
    applies two Gaussian blurs, local minima detection and a seeded watershed.

    See also
    --------
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    """
    spot_blurred = gaussian(image, sigma=spot_sigma)

    spots = label(local_minima(spot_blurred))

    outline_blurred = (
        spot_blurred
        if outline_sigma == spot_sigma
        else gaussian(image, sigma=outline_sigma)
    )

    return watershed(outline_blurred, spots)
