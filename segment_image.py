import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import local_minima
from skimage.segmentation import watershed


def segment_image(
    projected_image: npt.NDArray[np.uint8],
    spot_sigma: float = 10,
    outline_sigma: float = 0,
) -> None:
    """
    read in the projected image and segment it
    """
    pass


def local_minima_seeded_watershed(
    image: npt.NDArray[np.uint8], spot_sigma: float, outline_sigma
) -> None:
    """
    Segment cells in images with fluorescently marked membranes. The two sigma
    parameters allow tuning the segmentation result.The first sigma controls how
    close detected cells can be (spot_sigma) and the second controls how precise
    segmented objects are outlined (outline_sigma). Under the hood, this filter
    applies two Gaussian blurs, local minima detection and a seeded watershed.

    Taken from
    https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/b57762283a63517ecd3dfe0ff56ec5a051977ee1/napari_segment_blobs_and_things_with_membranes/__init__.py#L583-L607. # noqa: E501

    See also
    --------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    """

    image = np.asarray(image)

    spot_blurred = gaussian(image, sigma=spot_sigma)

    spots = label(local_minima(spot_blurred))

    if outline_sigma == spot_sigma:
        outline_blurred = spot_blurred
    else:
        outline_blurred = gaussian(image, sigma=outline_sigma)

    return watershed(outline_blurred, spots)
