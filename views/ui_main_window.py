from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFile, QMetaObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QMenu, QMenuBar, QStatusBar, QTabWidget,
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox
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
        self.actionOpen_Image = QAction(MainWindow)
        self.actionSave_Result = QAction(MainWindow)
        self.actionExit = QAction(MainWindow)

        self.menubar = QMenuBar(MainWindow)
        self.menuFile = QMenu("File", self.menubar)
        self.menubar.addMenu(self.menuFile)
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.menuFile.addAction(self.actionOpen_Image)
        self.menuFile.addAction(self.actionSave_Result)
        self.menuFile.addAction(self.actionExit)

        # ================= EXISTING UI =================
        self.lblOriginal = self.centralwidget.findChild(type(self.centralwidget), "lblOriginal")
        self.lblProcessed = self.centralwidget.findChild(type(self.centralwidget), "lblProcessed")

        self.btnUploadOriginal = self.centralwidget.findChild(type(self.centralwidget), "btnUploadOriginal")
        self.btnReset = self.centralwidget.findChild(type(self.centralwidget), "btnReset")

        self.parametersStack = self.centralwidget.findChild(type(self.centralwidget), "parametersStack")

        # ======================================================
        # 🆕 REPLACE OLD PARAMETERS WITH TABS
        # ======================================================

        self.tabs = QTabWidget()

        # ================= TAB 1: THRESHOLDING =================
        self.tabThreshold = QWidget()
        th_layout = QVBoxLayout(self.tabThreshold)

        self.comboThresholdMode = QComboBox()
        self.comboThresholdMode.addItems(["Global", "Local"])

        self.comboThresholdTechnique = QComboBox()
        self.comboThresholdTechnique.addItems(["Otsu", "Spectral", "Optimal"])

        self.windowSize = QSpinBox()
        self.windowSize.setRange(3, 51)
        self.windowSize.setValue(3)

        self.btnApplyThreshold = QPushButton("Apply Thresholding")

        th_layout.addWidget(QLabel("Mode"))
        th_layout.addWidget(self.comboThresholdMode)

        th_layout.addWidget(QLabel("Technique"))
        th_layout.addWidget(self.comboThresholdTechnique)

        th_layout.addWidget(QLabel("Window Size"))
        th_layout.addWidget(self.windowSize)

        th_layout.addWidget(self.btnApplyThreshold)

        # ================= TAB 2: SEGMENTATION =================
        self.tabSegmentation = QWidget()
        seg_layout = QVBoxLayout(self.tabSegmentation)

        self.btnApplySegmentation = QPushButton("Apply Segmentation")
        seg_layout.addWidget(self.btnApplySegmentation)

        # ================= ADD TABS =================
        self.tabs.addTab(self.tabThreshold, "Thresholding")
        self.tabs.addTab(self.tabSegmentation, "Segmentation")

        # inject into old stack (same UI structure preserved)
        if self.parametersStack and self.parametersStack.layout():
            layout = self.parametersStack.layout()

            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            layout.addWidget(self.tabs)

        QMetaObject.connectSlotsByName(MainWindow)