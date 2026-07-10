import os

from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ProgressBar, TransparentToolButton
from qfluentwidgets.common.style_sheet import setCustomStyleSheet
from qfluentwidgets.components.widgets.combo_box import ComboBoxMenu

COMPACT_CONTROL_HEIGHT = 28

_COMBO_BORDERLESS_QSS = """
ComboBox, ModelComboBox {
    border: none;
    border-top: none;
    outline: none;
}
ComboBox:pressed, ModelComboBox:pressed {
    border: none;
    border-top: none;
}
ComboBox:disabled, ModelComboBox:disabled {
    border: none;
    border-top: none;
}
"""

_COMBO_MENU_BORDERLESS_QSS = """
MenuActionListWidget {
    border: none;
    outline: none;
}
"""


class BorderlessComboBoxMenu(ComboBoxMenu):
    def __init__(self, parent=None):
        super().__init__(parent)
        setCustomStyleSheet(self, _COMBO_MENU_BORDERLESS_QSS, _COMBO_MENU_BORDERLESS_QSS)
        self.view.setGraphicsEffect(None)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)


class BorderlessComboBox(ComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        setCustomStyleSheet(self, _COMBO_BORDERLESS_QSS, _COMBO_BORDERLESS_QSS)

    def _createComboMenu(self):
        return BorderlessComboBoxMenu(self)


def create_compact_combo(
    parent=None,
    *,
    min_width: int = 88,
    max_width: int = 180,
) -> BorderlessComboBox:
    combo = BorderlessComboBox(parent)
    combo.setFixedHeight(COMPACT_CONTROL_HEIGHT)
    combo.setMinimumWidth(min_width)
    combo.setMaximumWidth(max_width)
    return combo


class DragLineEdit(LineEdit):
    """接受拖入单个文件或文件夹，把绝对路径写入文本框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)


def collect_midi_files(root: str) -> list[str]:
    """递归收集目录下所有 .mid / .midi 文件（按路径升序）。"""
    hits: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            low = name.lower()
            if low.endswith(".mid") or low.endswith(".midi"):
                hits.append(os.path.abspath(os.path.join(dirpath, name)))
    hits.sort()
    return hits


class BatchProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progressBar = ProgressBar(self)
        self.statusLabel = BodyLabel("就绪", self)
        self.openBtn = TransparentToolButton(self)
        self.openBtn.setToolTip("打开输出目录")
        self.openBtn.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.statusLabel, 1)
        row.addWidget(self.openBtn)
        layout.addLayout(row)
        layout.addWidget(self.progressBar)

    def reset(self):
        self.progressBar.setValue(0)
        self.statusLabel.setText("就绪")
        self.openBtn.setEnabled(False)

    def set_progress(self, value: int, text: str):
        self.progressBar.setValue(max(0, min(100, value)))
        self.statusLabel.setText(text)
