import cv2

from core.global_thresholding import apply_global_threshold
from core.local_thresholding import apply_local_threshold
from core.optimal_thresholding import compute_optimal_threshold


class ImageModel:
    def __init__(self):
        self.original_image = None  # Stores the cv2 numpy array
        self.processed_image = None

    def threshold_image(self, mode, technique, window_size=3):
        if self.original_image is None:
            raise ValueError("No image loaded")

        gray_image = self._to_grayscale(self.original_image)
        threshold_function = self._get_threshold_function(technique)

        if mode == "Global":
            return apply_global_threshold(gray_image, threshold_function)
        elif mode == "Local":
            block_size = max(3, window_size)
            return apply_local_threshold(gray_image, threshold_function, block_size=block_size)
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")

    def _to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _get_threshold_function(self, technique):
        technique = technique.lower()

        if technique == "otsu":
            return self._otsu_threshold
        if technique == "optimal":
            return self._optimal_threshold
        if technique == "spectral":
            return self._spectral_threshold

        raise ValueError(f"Unknown threshold technique: {technique}")

    def _otsu_threshold(self, image):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def _optimal_threshold(self, image):
        return compute_optimal_threshold(image)

    def _spectral_threshold(self, image):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return threshold
