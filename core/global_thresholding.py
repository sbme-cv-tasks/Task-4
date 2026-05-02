import numpy as np


def apply_global_threshold(image, threshold_function):
    """
    Apply global thresholding using any threshold function.

    Parameters:
        image : grayscale image (2D numpy array)
        threshold_function : function that returns threshold value

    Returns:
        binary image
    """

    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")

    threshold = threshold_function(image)

    rows, cols = image.shape
    binary = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if image[i, j] > threshold:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary