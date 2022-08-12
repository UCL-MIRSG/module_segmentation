import numpy as np
import numpy.typing as npt

TWO_FIVE_FIVE = 255


def calculate_cell_positions(
    image: npt.NDArray[np.float64], labelled_cells: npt.NDArray[np.uint16], type: bool
) -> None:
    """
    This function calculates the min intensity position of each
    labelled cell or the centroid position of each labelled region
    """
    no_cells = labelled_cells.max()

    pos = np.zeros((no_cells, 2))

    I2 = TWO_FIVE_FIVE - image

    for n in range(no_cells):

        sy, sx = np.argwhere(labelled_cells == n).T

        if type:
            _place_at_lowest_int(pos, I2, sx, sy, n)

        else:
            _place_at_centroid(pos, image, sx, sy, n)

            if ~np.isnan(pos[n, 1]) and labelled_cells[pos[n, 1], pos[n, 0]] != n:
                # every so often the centroid is actually not in the label!
                _place_at_lowest_int(pos, image, sx, sy, n)


def _place_at_centroid(pos, image, sx, sy, n) -> None:
    """
    calculating the centroid from intensities
    """
    sum_x = 0
    sum_y = 0
    sum_I = 0

    for m in range(len(sy)):
        sum_x += sx[m] * image[sy[m], sx[m]]
        sum_y += sy[m] * image[sy[m], sx[m]]
        sum_I += image[sy[m], sx[m]]

    pos[n, 1] = round(sum_y / sum_I)
    pos[n, 0] = round(sum_x / sum_I)


def _place_at_lowest_int(pos, image, sx, sy, n) -> None:
    """
    looking for the lowest intensity
    """
    val = TWO_FIVE_FIVE

    for m in range(len(sy)):

        if image[sy[m], sx[m]] < val:

            val = image[sy[m], sx[m]]
            pos[n, 2] = sy[m]
            pos[n, 1] = sx[m]
