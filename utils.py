import numpy as np
import numpy.typing as npt
from skimage.measure import regionprops


def create_color_boundaries(
    image: npt.NDArray[np.uint8],
    cell_seeds: npt.NDArray[np.uint8],
    cell_labels: npt.NDArray[np.uint16],
) -> npt.NDArray[np.uint8]:
    """
    Generate final colored image (RGB) to represent the segmentation results
    """
    # given that every cell has a different label we can compute
    # the boundaries by computing where the gradient changes
    cell_labels = cell_labels.astype(float)
    gx, gy = np.gradient(cell_labels)
    cell_outlines = (cell_labels > 0) & ((gx**2 + gy**2) > 0)

    cell_seeds = cell_seeds > 253

    segmented_image = image.copy()
    segmented_image[cell_outlines] = 0
    segmented_image[cell_seeds] = 255
    return segmented_image


def do_initial_seeding(
    image: npt.NDArray[np.uint8],
    sigma_1: float,
    min_cell_size: float,
    threshold: int,
    large_cell_size_thres: float,
) -> None:
    """
    Find the initial cell seeds
    """
    # Create gaussian filter
    if sigma_1 > 0.01:
        f1 = _matlab_style_gauss2D(image.shape, sigma_1)

        # Gaussian smoothing for the segmentation of individual cells
        smoothed_image = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(f1))
        ).real
    else:
        smoothed_image = image.astype(float)

    smoothed_image /= smoothed_image.max() * 252

    # Use external c-code to find initial seeds
    initial_labelling = findcellsfromregiongrowing(
        smoothed_image, min_cell_size, threshold
    )

    # set unallocated pixels to 0
    initial_labelling[initial_labelling == 1] = 0

    # Generate cell_labels from inital_labelling
    cell_labels = initial_labelling.astype(np.uint16)

    # eliminate very large areas
    delabel_very_large_areas(cell_labels, large_cell_size_thres)

    # Use true centre of cells as labels
    centroids = np.round(calculateCellPositions(smoothed_image, cell_labels, False))
    centroids = centroids[~np.isnan(centroids.T[0])]
    for n in range(len(centroids)):
        smoothed_image[centroids[n, 1], centroids[n, 0]] = 255

    # cell_seeds contains the position of the true cell center.
    cell_seeds = smoothed_image.astype(np.uint8)


def grow_cells_in_frame(
    image: npt.NDArray[np.uint8], cell_seeds: npt.NDArray[np.uint8], sigma_3: float
) -> None:
    """
    Growing cells from seeds TODO: add paramters in Name description!
    """
    # find labels
    bw = (cell_seeds > 252).astype(float)

    if sigma_3 > 0.01:
        f1 = _matlab_style_gauss2D(image.shape, sigma_3)
        smoothed_image = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(f1))
        ).real
    else:
        smoothed_image = image.astype(float)

    # mark labels on image
    image_with_seeds = (smoothed_image).astype(float) * (1 - bw) + 255 * bw
    cell_labels = growcellsfromseeds3(image_with_seeds, 253).astype(np.uint16)


def unlabel_poor_seeds_in_frame(
    image: npt.NDArray[np.uint8],
    cell_seeds: npt.NDArray[np.uint8],
    cell_labels: npt.NDArray[np.uint16],
    threshold: int,
    sigma_3: float,
    I_bound_max_pcnt: float,
) -> None:
    """
    Eliminate labels from seeds which have poor boundary intensity
    """
    L = cell_labels

    if sigma_3 > 0.01:
        f1 = _matlab_style_gauss2D(image.shape, sigma_3)
        smoothed_image = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(f1))
        ).real
    else:
        smoothed_image = image.astype(float)

    # i.e. every cell is marked by one unique integer label
    label_list = np.unique(L)
    label_list = label_list[label_list != 0]
    IBounds = np.zeros(len(label_list))
    decisions = [0, 0, 0, 0, 0]

    for c in range(len(label_list)):
        mask = L == label_list[c]
        cpy, cpx = np.argwhere(mask > 0)
        # find region of that label
        minx = np.min(cpx)
        maxx = np.max(cpx)
        miny = np.min(cpy)
        maxy = np.max(cpy)
        minx = np.max(minx - 5, 1)
        miny = np.max(miny - 5, 1)
        maxx = np.min(maxx + 5, image.shape[1])
        maxy = np.min(maxy + 5, image.shape[0])
        # reduced to region of the boundary
        reduced_mask = mask[miny:maxy, minx:maxx]
        reduced_image = smoothed_image[miny:maxy, minx:maxx]
        dilated_mask = imdilate(reduced_mask, se)
        eroded_mask = imerode(reduced_mask, se)
        boundary_mask = dilated_mask - eroded_mask
        boundary_intensities = reduced_image[boundary_mask > 0]
        H = reduced_image[boundary_mask > 0]
        IEr = reduced_image[eroded_mask > 0]
        I_bound = np.mean(boundary_intensities)
        IBounds[c] = I_bound

        # cell seed information is retrieved as comparison
        F2 = cell_seeds
        F2[~mask] = 0
        cpy, cpx = np.argwhere(F2 > 252)
        ICentre = smoothed_image[cpy, cpx]

        I_bound_max = 255 * I_bound_max_pcnt

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
        third_condition = np.min(boundary_intensities) == 0
        # 4. If the amount of low intensity signal (i.e. < 20) is more than 10%
        fourth_condition = np.sum(H < threshold) / len(H) > 0.1
        if first_condition | second_condition | third_condition | fourth_condition:

            # The label is cancelled (inverted mask multiplication.)
            cell_labels *= (mask == 0).astype(np.uint16)

            # record the removal decisions
            if first_condition:
                decisions[1] = decisions[1] + 1
            elif second_condition:
                decisions[2] = decisions[2] + 1
            elif third_condition:
                decisions[3] = decisions[3] + 1
            elif fourth_condition:
                decisions[4] = decisions[4] + 1
            else:
                # should not happen
                decisions[5] = decisions[5] + 1


def delabel_very_large_areas(
    cell_labels: npt.NDArray[np.uint16], large_cell_size_thres: float
) -> npt.NDArray[np.uint16]:
    """
    remove cells which are bigger than large_cell_size_thres
    """
    As = [r.area for r in regionprops(cell_labels, cache=False)]
    ls = np.unique(cell_labels)
    for l in range(len(ls)):
        if l == 0:
            continue

        A = As[l]
        if A > large_cell_size_thres:
            cell_labels[cell_labels == l] = 0

    return cell_labels


def merge_seeds_from_labels(
    image: npt.NDArray[np.uint8],
    cell_seeds: npt.NDArray[np.uint8],
    cell_labels: npt.NDArray[np.uint16],
    merge_criteria: float,
    sigma_3: float,
) -> None:
    """
    Remove initial cell regions which touch & whose boundary is insufficient
    """
    # smoothing
    if sigma_3 > 0.01:
        f1 = _matlab_style_gauss2D(image.shape, sigma_3)
        smoothed_image = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(f1))
        ).real
    else:
        smoothed_image = image.astype(float)

    label_list = np.unique(cell_labels)
    label_list = label_list[label_list != 0]
    c = 1

    merge_intensity_distro = []
    merge_decisions = 0

    # loop over labels
    while True:
        label_mask = cell_labels == label_list[c]
        label = label_list[c]

        cpy, cpx = np.argwhere(label_mask > 0)

        # find region of that label
        minx = np.min(cpx)
        maxx = np.max(cpx)
        miny = np.min(cpy)
        maxy = np.max(cpy)
        minx = np.max(minx - 5, 1)
        miny = np.max(miny - 5, 1)
        maxx = np.min(maxx + 5, image.shape[1])
        maxy = np.min(maxy + 5, image.shape[0])

        # reduce data to that region
        reduced_label_mask = label_mask[miny:maxy, minx:maxx]
        reduced_image = smoothed_image[miny:maxy, minx:maxx]
        reduced_labels = cell_labels[miny:maxy, minx:maxx]

        # now find boundaries
        dilated_mask = imdilate(reduced_label_mask, se)
        eroded_mask = imerode(reduced_label_mask, se)
        border_mask = dilated_mask - eroded_mask
        border_intensities = reduced_image[border_mask > 0]
        central_intensity = reduced_image[eroded_mask > 0]

        F2 = cell_seeds
        F2[~label_mask] = 0
        cpy, cpx = np.argwhere(F2 > 253)
        ICentre = smoothed_image[cpy, cpx]

        background_std = np.std(central_intensity.astype(float))

        # get labels of surrounding cells (neighbours)
        neighbour_labels = np.unique(reduced_labels[dilated_mask > 0])
        neighbour_labels = neighbour_labels[neighbour_labels != label]

        low_intensity_ratios = []
        for i in range(len(neighbour_labels)):
            neighb_label = neighbour_labels(i)
            neighbor_border = dilated_mask
            # slice of neighbour around cell
            neighbor_border[reduced_labels != neighb_label] = 0
            cell_border = imdilate(neighbor_border, se)
            # slice of cell closest to neighbour
            cell_border[reduced_labels != label] = 0

            # combination of both creating boundary region
            joint_border = (cell_border + neighbor_border) > 0
            border_intensities = reduced_image
            # intensities at boundary
            border_intensities[~joint_border] = 0

            # average number of points in boundary where intensity is
            # of low quality (dodgy)
            low_intensity_threshold = ICentre + (background_std / 2)
            low_intensity_pixels = (
                border_intensities[joint_border] < low_intensity_threshold
            )

            low_intensity_ratio = np.sum(low_intensity_pixels) / np.shape(
                border_intensities[joint_border]
            )

            low_intensity_ratios = [low_intensity_ratios, low_intensity_ratio]

        # Find out which is border with the lowest intensity ratio
        worst_intensity_ratio, worst_neighbor_index = np.max(low_intensity_ratios)
        neighb_label = neighbour_labels(worst_neighbor_index)

        # if the label value is of poor quality, then recursively check the merge
        # criteria in order to add it as a potential label in the label set.

        merge_intensity_distro = [merge_intensity_distro, worst_intensity_ratio]

        if (worst_intensity_ratio > merge_criteria) & label != 0 & neighb_label != 0:

            merge_labels(cell_seeds, cell_labels, label, neighb_label)
            label_list = np.unique(cell_labels)
            label_list = label_list[label_list != 0]
            c -= 1
            # reanalyze the same cell for more possible mergings
            merge_decisions += 1

        c += 1

        # Condition to break the while cycle -> as soon as all the
        # labels are processed, then exit
        if c > len(label_list):
            break


def merge_labels(
    cell_seeds: npt.NDArray[np.uint8], cell_labels: npt.NDArray[np.uint16], l1, l2
) -> None:
    """ """
    Cl = cell_labels
    Il = cell_seeds
    m1 = Cl == l1
    m2 = Cl == l2
    Il1 = Il
    Il1[~m1] = 0
    Il2 = Il
    Il2[~m2] = 0
    cpy1, cpx1 = np.argwhere(Il1 > 253)
    cpy2, cpx2 = np.argwhere(Il2 > 253)
    cpx = round((cpx1 + cpx2) / 2)
    cpy = round((cpy1 + cpy2) / 2)

    # background level
    cell_seeds[cpy1, cpx1] = 20
    cell_seeds[cpy2, cpx2] = 20
    if (cell_labels[cpy, cpx] == l1) | (cell_labels[cpy, cpx] == l2):
        cell_seeds[cpy, cpx] = 255
    elif np.sum(m1) > np.sum(m2):
        cell_seeds[cpy1, cpx1] = 255
    else:
        cell_seeds[cpy2, cpx2] = 255

    Cl[m2] = l1
    cell_labels = Cl


def neutralise_pts_not_under_label_in_frame(
    cell_seeds: npt.NDArray[np.uint8],
    cell_labels: npt.NDArray[np.uint16],
) -> None:
    """
    Seeds whose label has been eliminated are converted to neutral seeds

    the idea here is to set seeds not labelled to a value
    ie invisible to retracking (and to growing, caution!)
    """
    L = cell_labels
    F = cell_seeds
    F2 = F
    F2[L != 0] = 0
    F[F2 > 252] = 253
    cell_seeds = F


def growcellsfromseeds3():
    pass


def findcellsfromregiongrowing():
    pass


def calculateCellPositions():
    pass


def _matlab_style_gauss2D(shape: tuple[int, int], sigma: float):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian', [shape], [sigma])
    """
    m, n = [(ss - 1) / 2 for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
