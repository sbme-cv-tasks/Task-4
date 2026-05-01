import cv2
from PySide6.QtGui import QImage, QPixmap


def cv_to_pixmap(cv_img):
    """Converts an OpenCV image (numpy array) to a QPixmap."""
    if cv_img is None:
        return None

    if len(cv_img.shape) == 2:  # Grayscale image
        h, w = cv_img.shape
        bytes_per_line = w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    else:  # BGR image (Standard cv2.imread)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

    return QPixmap.fromImage(q_img)