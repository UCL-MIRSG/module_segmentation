import numpy as np
import numpy.typing as npt


def create_color_boundaries(image: npt.NDArray[np.uint8], cell_labels: npt.NDArray[np.uint8]) -> None:
    """
    Generate final colored image (RGB) to represent the segmentation results
    """
    # given that every cell has a different label we can compute
    # the boundaries by computing where the gradient changes
    cell_lables = cell_lables.astype(float)
    [gx,gy] = gradient(cell_lables);
    cell_outlines = (cell_lables > 0) & ((gx**2+gy**2)>0)

    cell_seeds = CellSeeds(:,:) > 253;

    ColIm = Im
    ColIm(cell_outlines) = 0
    ColIm(cell_seeds) = 255


def do_initial_seeding(sigma_1: float, min_cell_size: float, threshold: int) -> None:
    """
    Find the initial cell seeds
    """
    pass


def grow_cells_in_frame(sigma_3: float) -> None:
    """
    Growing cells from seeds TODO: add paramters in Name description!
    """
    pass


def unlabel_poor_seeds_in_frame(threshold: int, sigma_3: float) -> None:
    """
    Eliminate labels from seeds which have poor boundary intensity
    """
    pass


def delabel_very_large_areas() -> None:
    """ """
    pass


def merge_seeds_from_labels(merge_criteria: float, sigma_3: float) -> None:
    """
    Remove initial cell regions which touch & whose boundary is insufficient
    """
    pass


def merge_labels() -> None:
    """ """
    pass


def neutralise_pts_not_under_label_in_frame(
    cell_labels: npt.NDArray[np.uint16],
    cell_seeds: npt.NDArray[np.uint8],
    *,
    value: int = 253
) -> npt.NDArray[np.uint8]:
    """
    Seeds whose label has been eliminated are converted to neutral seeds

    the idea here is to set seeds not labelled to a value
    ie invisible to retracking (and to growing, caution!)
    """
    cell_seeds_copy = cell_seeds.copy()
    cell_seeds_copy[cell_labels != 0] = 0
    cell_seeds[cell_seeds_copy > 252] = value
    return cell_seeds
