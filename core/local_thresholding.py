import numpy as np


def apply_local_threshold(image, threshold_function, block_size=64):
    """
    Apply local thresholding using blocks.

    Parameters:
        image : grayscale image
        threshold_function : function returns threshold for block
        block_size : block size

    Returns:
        binary image
    """

    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")

    rows, cols = image.shape
    binary = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):

            row_end = min(row + block_size, rows)
            col_end = min(col + block_size, cols)

            block = image[row:row_end, col:col_end]

            threshold = threshold_function(block)

            for i in range(block.shape[0]):
                for j in range(block.shape[1]):

                    if block[i, j] > threshold:
                        binary[row + i, col + j] = 255
                    else:
                        binary[row + i, col + j] = 0

    return binary