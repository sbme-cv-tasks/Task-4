from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFile, QMetaObject
from PySide6.QtGui import QAction
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QMenu, QMenuBar, QStatusBar,
    QWidget, QComboBox, QSpinBox, QPushButton,
    QTabWidget, QStackedWidget
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        loader = QUiLoader()
        ui_path = Path(__file__).with_name("main_window.ui")

        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.OpenModeFlag.ReadOnly):
            raise FileNotFoundError(ui_path)

        self.centralwidget = loader.load(ui_file, MainWindow)
        ui_file.close()

        if self.centralwidget is None:
            raise RuntimeError("Failed to load UI")

        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.resize(1366, 768)

        # ================= MENU =================
        # self.actionOpen_Image = QAction(MainWindow)
        # self.actionSave_Result = QAction(MainWindow)
        # self.actionExit = QAction(MainWindow)

        # self.menubar = QMenuBar(MainWindow)
        # self.menuFile = QMenu("File", self.menubar)
        # self.menubar.addMenu(self.menuFile)
        # MainWindow.setMenuBar(self.menubar)

        # self.menuFile.addAction(self.actionOpen_Image)
        # self.menuFile.addAction(self.actionSave_Result)
        # self.menuFile.addAction(self.actionExit)

        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # ================= WIDGETS FROM UI =================
        # Top bar buttons (now on the LEFT side)
        self.btnUploadOriginal = self.centralwidget.findChild(QPushButton, "btnUploadOriginal")
        self.btnReset = self.centralwidget.findChild(QPushButton, "btnReset")

        # Image labels
        self.lblOriginal = self.centralwidget.findChild(QWidget, "lblOriginal")
        self.lblProcessed = self.centralwidget.findChild(QWidget, "lblProcessed")

        # Sidebar: parametersStack + tabs (embedded in left sidebar of pageSingleView)
        self.parametersStack = self.centralwidget.findChild(QStackedWidget, "parametersStack")
        self.tabs = self.centralwidget.findChild(QTabWidget, "tabs")

        # Tab controls
        self.comboThresholdMode = self.centralwidget.findChild(QComboBox, "comboThresholdMode")
        self.comboThresholdTechnique = self.centralwidget.findChild(QComboBox, "comboThresholdTechnique")
        self.windowSize = self.centralwidget.findChild(QSpinBox, "windowSize")
        self.btnApplyThreshold = self.centralwidget.findChild(QPushButton, "btnApplyThreshold")
        self.btnApplySegmentation = self.centralwidget.findChild(QPushButton, "btnApplySegmentation")

        QMetaObject.connectSlotsByName(MainWindow)
