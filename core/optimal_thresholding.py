# optimal_thresholding.py

import numpy as np


def get_corner_pixels(image):
    rows, cols = image.shape

    corners = [
        image[0, 0],
        image[0, cols - 1],
        image[rows - 1, 0],
        image[rows - 1, cols - 1]
    ]

    return np.array(corners, dtype=np.float32)


def compute_optimal_threshold(image, epsilon=0.001):
    """
    Compute optimal iterative threshold from scratch.
    """

    rows, cols = image.shape

    # initial background = corners
    background = get_corner_pixels(image)

    # all pixels
    all_pixels = image.flatten()

    # initial object pixels
    object_pixels = []

    for pixel in all_pixels:
        found = False
        for c in background:
            if pixel == c:
                found = True
                break

        if not found:
            object_pixels.append(pixel)

    object_pixels = np.array(object_pixels, dtype=np.float32)

    previous_threshold = 0

    while True:

        # means
        bg_mean = np.sum(background) / len(background) if len(background) > 0 else 0
        obj_mean = np.sum(object_pixels) / len(object_pixels) if len(object_pixels) > 0 else 0

        current_threshold = (bg_mean + obj_mean) / 2

        if abs(current_threshold - previous_threshold) < epsilon:
            break

        # recalculate groups
        new_background = []
        new_object = []

        for i in range(rows):
            for j in range(cols):

                pixel = image[i, j]

                if pixel <= current_threshold:
                    new_background.append(pixel)
                else:
                    new_object.append(pixel)

        background = np.array(new_background, dtype=np.float32)
        object_pixels = np.array(new_object, dtype=np.float32)

        previous_threshold = current_threshold

    return current_threshold