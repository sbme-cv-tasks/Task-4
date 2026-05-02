import numpy as np
import cv2

def otsu_threshold(image):
    """
    Compute optimal threshold using Otsu's method.
    Parameters:
        image : grayscale image (2D numpy array)
    Returns:
        threshold value (int)
    """

    # check if grayscale
    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")

    # histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()

    total_pixels = image.shape[0] * image.shape[1]

    # probabilities
    probabilities = histogram / total_pixels

    best_threshold = 0
    best_variance = 0

    # total mean (mG)
    intensities = np.arange(256)
    total_mean = np.sum(intensities * probabilities)

    weight_bg = 0
    sum_bg = 0

    for t in range(256):

        weight_bg += probabilities[t]
        if weight_bg == 0:
            continue

        weight_fg = 1 - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * probabilities[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (total_mean - sum_bg) / weight_fg

        # between-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    return int(best_threshold)