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
    cropped_image = proj_im[300:500, 400:600]

    segmentation, seeds, labels = segment_image(
        cropped_image,
        sigma_1=0.5,
        min_cell_size=2,
        threshold=20,
        merge_criteria=0.0,
        sigma_3=0.5,
        large_cell_size_thres=3000,
        I_bound_max_pcnt=0.1,
        show_feedback=True,
    )


if __name__ == "__main__":
    main()
