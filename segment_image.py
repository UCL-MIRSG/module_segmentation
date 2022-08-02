import numpy as np
import numpy.typing as npt

from utils import (
    create_color_boundaries,
    do_initial_seeding,
    grow_cells_in_frame,
    merge_seeds_from_labels,
    neutralise_pts_not_under_label_in_frame,
    unlabel_poor_seeds_in_frame,
)


def segment_image(
    image: npt.NDArray[np.uint8],
    *,
    sigma_1: float,
    min_cell_size: float,
    threshold: int,
    merge_criteria: float,
    sigma_3: float,
    large_cell_size_thres: float,
    i_bound_max_pcnt: float,
    show_feedback: bool = False
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16], npt.NDArray[np.uint8]]:
    """
    Segments a single frame extracting the cell outlines.

    Args:
      image:
        increasing intesity for membrane
      sigma_1:
        size px of gaussian for smoothing image [0+]
      min_cell_size:
        size px of smallest cell expected [0+]
      threshold:
        minimum value for membrane signal [0-255]
      merge_criteria:
        minimum ratio of low intensity pxs upon which to merge cells [0-1]
      sigma_3:
        size px of gaussian for smoothing image [0+]
      large_cell_size_thres:
        size px of largest cell expected [0+]
      i_bound_max_pcnt:
        minimum ratio for seed and membrane intensity [0-1]
      show_feedback:
        show feedback for segmentation

    Returns:
      segmented_image:
        Im with seeds [255] and cell outlines [0]
      cell_seeds:
        Rescaled Image to fit the range [0,252]
        253/254/255 are used for seed information
      cell_labels:
        bitmap of cells colored with 16bit id
    """
    # initialise
    cell_seeds = np.zeros(image.shape, dtype=np.uint8)
    # TODO: why using 16 bit for labels?
    cell_labels = np.zeros(image.shape, dtype=np.uint16)

    # TODO: check image casting
    image = image.astype(float)
    image *= 252 / image.max()
    image = image.astype(np.uint8)

    do_initial_seeding(sigma_1, min_cell_size, threshold)

    merge_seeds_from_labels(merge_criteria, sigma_3)

    grow_cells_in_frame(sigma_3)

    unlabel_poor_seeds_in_frame(threshold, sigma_3)

    neutralise_pts_not_under_label_in_frame(cell_labels, cell_seeds, value=253)

    create_color_boundaries()
