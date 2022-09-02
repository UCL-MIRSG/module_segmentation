import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk, local_minima
from skimage.segmentation import relabel_sequential, watershed

# TODO: work out what these are for and give better names
TWO_FIVE_FIVE = 255
TWO_FIVE_TWO = 252


def unlabel_poor_seeds_in_frame(
    image: npt.NDArray[np.uint8],
    seeds: list[tuple[float, float]],
    labels: npt.NDArray[np.uint32],
    *,
    outline_sigma: float,
    minimum_intensity: float,
    min_seed_boundary_ratio: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Eliminate labels from seeds which have poor boundary intensity
    """
    # structuring element, SE, used for morphological operations
    se = disk(2)

    L = labels.copy()

    # the multiplication ensures similar results to the MATLAB
    smoothed_image = gaussian(image, sigma=outline_sigma) * TWO_FIVE_TWO

    # i.e. every cell is marked by one unique integer label
    label_list = np.unique(L)
    label_list = label_list[label_list != 0]
    IBounds = np.zeros(len(label_list))
    decisions = [0, 0, 0, 0, 0]

    for c in range(len(label_list)):
        mask = L == label_list[c]
        cpy, cpx = np.argwhere(mask).T

        # find region of that label
        minx = max(cpx.min() - 5, 0)
        miny = max(cpy.min() - 5, 0)
        maxx = min(cpx.max() + 5, image.shape[1] - 1)
        maxy = min(cpy.max() + 5, image.shape[0] - 1)

        # reduced to region of the boundary
        reduced_mask = mask[miny : maxy + 1, minx : maxx + 1]
        reduced_image = smoothed_image[miny : maxy + 1, minx : maxx + 1]
        dilated_mask = binary_dilation(reduced_mask, se)
        eroded_mask = binary_erosion(reduced_mask, se)
        boundary_mask = dilated_mask ^ eroded_mask
        boundary_intensities = reduced_image[boundary_mask > 0]
        H = reduced_image[boundary_mask > 0]
        I_bound = (boundary_intensities).mean()
        IBounds[c] = I_bound

        # cell seed information is retrieved as comparison
        F2 = seeds.copy()
        F2[~mask] = 0
        cpy, cpx = np.argwhere(F2 > TWO_FIVE_TWO).T
        ICentre = smoothed_image[cpy, cpx][0]

        I_bound_max = TWO_FIVE_FIVE * min_seed_boundary_ratio

        # Figure out which conditions make the label invalid
        # 1. I_bound_max, gives the Lower bound to the mean intensity
        # 1.b condition upon that the cell seed has less
        #     than 20% intensity difference to the mean
        #   => If the cell boundary is low and not very different
        #      from the seed, cancel the region
        first_condition = (I_bound < I_bound_max) & (I_bound / ICentre < 1.2)
        # 2. W/o (1.b) the lower bound is reduced by ~17% (1 - 0.833) to be decisive
        second_condition = I_bound < I_bound_max * 5 / 6
        # 3. If the minimum retrieved in the boundary mask is 0 (dangerous!)
        third_condition = boundary_intensities.min() == 0
        # 4. If the amount of low intensity signal (i.e. < 20) is more than 10%
        fourth_condition = (H < minimum_intensity).sum() / len(H) > 0.1

        if first_condition | second_condition | third_condition | fourth_condition:
            # The label is cancelled (inverted mask multiplication.)
            labels *= (mask == 0).astype(np.uint16)

            # record the removal decisions
            if first_condition:
                decisions[1] += 1
            elif second_condition:
                decisions[2] += 1
            elif third_condition:
                decisions[3] += 1
            elif fourth_condition:
                decisions[4] += 1
            else:
                # should not happen
                decisions[5] += 1

    return labels, seeds


def _local_minima_seeded_watershed(
    image: npt.NDArray[np.uint8], *, spot_sigma: float, outline_sigma: float
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
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

    spots = label(local_minima(spot_blurred))

    outline_blurred = (
        spot_blurred
        if outline_sigma == spot_sigma
        else gaussian(image, sigma=outline_sigma)
    )

    return spots, watershed(outline_blurred, spots)


def thresholded_local_minima_seeded_watershed(
    image: npt.NDArray[np.uint8],
    *,
    spot_sigma: float,
    outline_sigma: float,
    minimum_intensity: float,
) -> tuple[list[tuple[float, float]], npt.NDArray[np.uint32]]:
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
    spots, labels = _local_minima_seeded_watershed(
        image, spot_sigma=spot_sigma, outline_sigma=outline_sigma
    )

    # get seeds
    spots_stats = regionprops(spots)
    seeds = [r.centroid for r in spots_stats]

    # measure intensities
    stats = regionprops(labels, image)
    intensities = np.array([r.mean_intensity for r in stats])

    # filter labels with low intensity
    new_label_indices, _, _ = relabel_sequential(
        (intensities > minimum_intensity) * np.arange(labels.max())
    )
    new_label_indices = np.insert(new_label_indices, 0, 0)
    new_labels = np.take(np.asarray(new_label_indices, np.uint32), labels)

    return seeds, new_labels
