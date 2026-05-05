import cv2

from core.global_thresholding import apply_global_threshold
from core.local_thresholding import apply_local_threshold
from core.spectral_thresholding import spectral_thresholding
from core.Otsu_thersholding import otsu_threshold


class ThresholdController:
    def __init__(self, ui, model, statusbar=None):
        self.ui = ui
        self.model = model
        self.statusbar = statusbar

        self._connect_signals()  
        
        self.ui.comboThresholdMode.setCurrentText("Local")
        self.on_mode_changed("Local")

    def _connect_signals(self):
        self.ui.comboThresholdMode.currentTextChanged.connect(self.on_mode_changed)
        self.ui.comboThresholdTechnique.currentTextChanged.connect(
            lambda _: self._update_k_visibility()
        )
        self._update_k_visibility()  

    def _update_k_visibility(self):
        technique = self.ui.comboThresholdTechnique.currentText()
        is_spectral = technique.lower() == "spectral"
        self.ui.spinSpectralK.setVisible(is_spectral)
        self.ui.labelSpectralK.setVisible(is_spectral)

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
        k = self.ui.spinSpectralK.value()  
        
        if self.model.original_image is None:
            raise ValueError("No image loaded")

        gray = self._to_grayscale(self.model.original_image)

        if technique.lower() == "spectral":
            if mode == "Global":
                output, _ = spectral_thresholding(gray, k=k, sigma=5)
                return output
            else:
                return self._apply_spectral_local(gray, window_size, k=k)

        threshold_function = self._select_threshold_function(technique)
        if mode == "Global":
            return apply_global_threshold(gray, threshold_function)
        return apply_local_threshold(gray, threshold_function, block_size=window_size)

    def _apply_spectral_local(self, gray, window_size, k=3):
        import numpy as np
        h, w = gray.shape
        output = np.zeros_like(gray, dtype=np.uint8)

        # Spectral needs larger blocks to detect intensity differences
        effective_size = window_size * 20

        for y in range(0, h, effective_size):
            for x in range(0, w, effective_size):
                block = gray[y:y+effective_size, x:x+effective_size]
                if block.shape[0] < 4 or block.shape[1] < 4:
                    output[y:y+effective_size, x:x+effective_size] = block
                    continue
                result, _ = spectral_thresholding(block, k=k, sigma=5)
                output[y:y+effective_size, x:x+effective_size] = result

        return output
    
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
        return otsu_threshold(image)

    def _optimal_threshold(self, image):
        from core.optimal_thresholding import compute_optimal_threshold
        return compute_optimal_threshold(image)

    def _spectral_threshold(self, image):
        # This should not be called directly since spectral is handled in apply_thresholding
        raise NotImplementedError("Spectral thresholding is handled separately in apply_thresholding()")
