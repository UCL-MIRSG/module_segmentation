from pathlib import Path

from scipy import io as sio
from segment_image import segment_image

_file_location = Path(__file__).resolve()
_matfile = _file_location.parent / "ProjIm.mat"


def main() -> None:
    """
    implements the demo in the README in python
    """
    # load example image (Projected Drosophila Wing Disc - Ecad:GFP)
    proj_im = sio.loadmat(_matfile)["ProjIm"]

    # crop image for testing
    crop = proj_im[300:500, 400:600]

    # compute segmentation and output segmentation feedback
    first_smoothing_seeding = 0.5
    min_cell_area = 2
    min_membrane_intensity = 20
    min_border_intensity_ratio = 0.0
    second_smoothing_segmentation = 0.5
    max_cell_area = 3000
    min_seed_per_boundary = 0.1
    segmentation_feedback_plot = True

    segmentation, seeds, labels = segment_image(
        crop,
        first_smoothing_seeding,
        min_cell_area,
        min_membrane_intensity,
        min_border_intensity_ratio,
        second_smoothing_segmentation,
        max_cell_area,
        min_seed_per_boundary,
        segmentation_feedback_plot,
    )


if __name__ == "__main__":
    main()
