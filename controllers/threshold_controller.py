import cv2

from core.global_thresholding import apply_global_threshold
from core.local_thresholding import apply_local_threshold


class ThresholdController:
    def __init__(self, ui, model, statusbar=None):
        self.ui = ui
        self.model = model
        self.statusbar = statusbar

        self._connect_signals()
        self.on_mode_changed(self.ui.comboThresholdMode.currentText())

    def _connect_signals(self):
        self.ui.comboThresholdMode.currentTextChanged.connect(self.on_mode_changed)

    def on_mode_changed(self, mode):
        local_mode = mode == "Local"
        self.ui.windowSize.setEnabled(local_mode)

        if self.statusbar is not None:
            if local_mode:
                self.statusbar.showMessage("Using local thresholding: choose window size", 3000)
            else:
                self.statusbar.showMessage("Using global thresholding", 3000)

    def apply_thresholding(self):
        mode = self.ui.comboThresholdMode.currentText()
        technique = self.ui.comboThresholdTechnique.currentText()
        window_size = self.ui.windowSize.value()

        if self.model.original_image is None:
            raise ValueError("No image loaded")

        if mode not in {"Global", "Local"}:
            raise ValueError(f"Unsupported mode: {mode}")

        gray = self._to_grayscale(self.model.original_image)
        threshold_function = self._select_threshold_function(technique)

        if mode == "Global":
            return apply_global_threshold(gray, threshold_function)

        return apply_local_threshold(gray, threshold_function, block_size=window_size)

    def _to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _select_threshold_function(self, technique):
        technique = technique.lower()

        if technique == "otsu":
            return self._otsu_threshold
        if technique == "optimal":
            return self._optimal_threshold
        if technique == "spectral":
            return self._spectral_threshold

        raise ValueError(f"Unsupported threshold technique: {technique}")

    def _otsu_threshold(self, image):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def _optimal_threshold(self, image):
        from core.optimal_thresholding import compute_optimal_threshold
        return compute_optimal_threshold(image)

    def _spectral_threshold(self, image):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return threshold
