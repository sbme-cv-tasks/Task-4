import sys

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QLocale

from models.image_model import ImageModel
from views.ui_main_window import Ui_MainWindow
from controllers.main_controller import MainController


QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ================= UI =================
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    # ================= WINDOW =================
    window = MainWindow()

    # ================= MODEL =================
    model = ImageModel()

    # ================= CONTROLLER =================
    controller = MainController(window.ui, model, window)

    # ================= SHOW =================
    window.show()

    sys.exit(app.exec())