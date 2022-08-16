# Functions in this file have been taken from
# https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/napari_segment_blobs_and_things_with_membranes/__init__.py
import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import local_minima
from skimage.segmentation import relabel_sequential, watershed


def _local_minima_seeded_watershed(
    image: npt.NDArray[np.uint8], *, spot_sigma: float, outline_sigma
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
    """
    Segment cells in images with fluorescently marked membranes.

    The two sigma parameters allow tuning the segmentation result.The first
    sigma controls how close detected cells can be (spot_sigma) and the second
    controls how precise segmented objects are outlined (outline_sigma). Under
    the hood, this filter applies two Gaussian blurs, local minima detection and
    a seeded watershed.

    See also
    --------
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

    Taken from
    --------
    https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/napari_segment_blobs_and_things_with_membranes/__init__.py
    """
    spot_blurred = gaussian(image, sigma=spot_sigma)

    seeds = local_minima(spot_blurred)

    spots = label(seeds)

    outline_blurred = (
        spot_blurred
        if outline_sigma == spot_sigma
        else gaussian(image, sigma=outline_sigma)
    )

    return watershed(outline_blurred, spots), seeds


def thresholded_local_minima_seeded_watershed(
    image: npt.NDArray[np.uint8],
    *,
    spot_sigma: float,
    outline_sigma: float,
    minimum_intensity: float,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.bool_]]:
    """
    Segment cells in images with marked membranes that have a high signal intensity.

    The two sigma parameters allow tuning the segmentation result. The first
    sigma controls how close detected cells can be (spot_sigma) and the second
    controls how precise segmented objects are outlined (outline_sigma). Under
    the hood, this filter applies two Gaussian blurs, local minima detection and
    a seeded watershed. Afterwards, all objects are removed that have an average
    intensity below a given minimum_intensity

    Taken from
    --------
    https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/napari_segment_blobs_and_things_with_membranes/__init__.py
    """
    labels, seeds = _local_minima_seeded_watershed(
        image, spot_sigma=spot_sigma, outline_sigma=outline_sigma
    )

    # measure intensities
    stats = regionprops(labels, image)
    intensities = np.array([r.mean_intensity for r in stats])

    # filter labels with low intensity
    new_label_indices, _, _ = relabel_sequential(
        (intensities > minimum_intensity) * np.arange(labels.max())
    )
    new_label_indices = np.insert(new_label_indices, 0, 0)
    new_labels = np.take(np.asarray(new_label_indices, np.uint32), labels)

    return new_labels, seeds
