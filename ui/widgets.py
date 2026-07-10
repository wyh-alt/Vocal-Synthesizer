import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ProgressBar, PushButton, TransparentToolButton
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)


# --- 状态标签用 ---
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_STATUS_TEXT = {
    STATUS_PENDING: "· 等待中",
    STATUS_RUNNING: "▶ 生成中",
    STATUS_DONE: "✓ 完成",
    STATUS_FAILED: "✗ 失败",
}

_STATUS_COLOR = {
    STATUS_PENDING: "#8a8a8a",
    STATUS_RUNNING: "#5b81f4",
    STATUS_DONE: "#3fa876",
    STATUS_FAILED: "#d94a4a",
}


def _walk_midi_files(root: str) -> list[str]:
    hits: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            low = name.lower()
            if low.endswith(".mid") or low.endswith(".midi"):
                hits.append(os.path.join(dirpath, name))
    hits.sort()
    return hits


class MidiQueueWidget(QWidget):
    """批量 MIDI 输入队列。支持拖入单个/多个文件或整个文件夹。"""

    filesChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self._status: dict[str, str] = {}
        self._messages: dict[str, str] = {}

        self.setAcceptDrops(True)

        self.list = QListWidget(self)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setAlternatingRowColors(True)
        self.list.setUniformItemSizes(True)
        self.list.setMinimumHeight(120)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.setStyleSheet(
            "QListWidget { border: 1px solid rgba(0,0,0,0.08); border-radius: 6px; }"
            "QListWidget::item { padding: 4px 8px; }"
        )

        self._hintLabel = BodyLabel(
            "拖入单/多个 .mid 文件，或整个文件夹（会自动搜索子目录）",
            self,
        )
        self._hintLabel.setStyleSheet("color: #8a8a8a;")
        self._hintLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 覆盖在 list 上的占位提示
        self._hintLabel.setParent(self.list)
        self._hintLabel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        add_files_btn = PushButton("添加文件", self)
        add_files_btn.clicked.connect(self._on_add_files)
        add_folder_btn = PushButton("添加文件夹", self)
        add_folder_btn.clicked.connect(self._on_add_folder)
        remove_btn = PushButton("移除选中", self)
        remove_btn.clicked.connect(self._on_remove_selected)
        clear_btn = PushButton("清空", self)
        clear_btn.clicked.connect(self._on_clear)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)
        toolbar.addWidget(add_files_btn)
        toolbar.addWidget(add_folder_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(remove_btn)
        toolbar.addWidget(clear_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.list, 1)
        layout.addLayout(toolbar)

        self._refresh_hint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        vp = self.list.viewport()
        self._hintLabel.setGeometry(0, 0, vp.width(), vp.height())

    # --- 拖拽 ---
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
        if not urls:
            return
        added = 0
        for url in urls:
            local = url.toLocalFile()
            if not local:
                continue
            added += self._add_path(local)
        if added:
            self._emit_changed()

    # --- 增删 ---
    def _add_path(self, path: str) -> int:
        if not path:
            return 0
        if os.path.isdir(path):
            found = _walk_midi_files(path)
            n = 0
            for f in found:
                if self._add_single(f):
                    n += 1
            return n
        if os.path.isfile(path):
            low = path.lower()
            if low.endswith(".mid") or low.endswith(".midi"):
                return 1 if self._add_single(path) else 0
        return 0

    def _add_single(self, path: str) -> bool:
        norm = os.path.abspath(path)
        if norm in self._paths:
            return False
        self._paths.append(norm)
        self._status[norm] = STATUS_PENDING
        self._messages[norm] = ""
        self._append_item(norm)
        return True

    def _append_item(self, path: str) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.list.addItem(item)
        self._refresh_item(item)

    def _refresh_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        status = self._status.get(path, STATUS_PENDING)
        msg = self._messages.get(path, "")
        base = os.path.basename(path)
        tail = f"  ·  {msg}" if msg else ""
        item.setText(f"{_STATUS_TEXT[status]}   {base}{tail}")
        item.setToolTip(path)
        item.setForeground(Qt.GlobalColor.black)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择 MIDI 文件", "", "MIDI 文件 (*.mid *.midi)"
        )
        added = 0
        for p in paths:
            added += self._add_path(p)
        if added:
            self._emit_changed()

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含 MIDI 的文件夹")
        if not folder:
            return
        added = self._add_path(folder)
        if added:
            self._emit_changed()

    def _on_remove_selected(self) -> None:
        selected = self.list.selectedItems()
        if not selected:
            return
        for item in selected:
            path = item.data(Qt.ItemDataRole.UserRole)
            self._paths = [p for p in self._paths if p != path]
            self._status.pop(path, None)
            self._messages.pop(path, None)
            self.list.takeItem(self.list.row(item))
        self._emit_changed()

    def _on_clear(self) -> None:
        self._paths.clear()
        self._status.clear()
        self._messages.clear()
        self.list.clear()
        self._emit_changed()

    def _context_menu(self, pos):
        item = self.list.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            open_act = QAction("打开所在文件夹", self)
            open_act.triggered.connect(
                lambda: os.startfile(os.path.dirname(item.data(Qt.ItemDataRole.UserRole)))
            )
            menu.addAction(open_act)
            remove_act = QAction("从队列移除", self)
            remove_act.triggered.connect(self._on_remove_selected)
            menu.addAction(remove_act)
            menu.addSeparator()
        clear_act = QAction("清空队列", self)
        clear_act.triggered.connect(self._on_clear)
        menu.addAction(clear_act)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    # --- 状态 ---
    def paths(self) -> list[str]:
        return list(self._paths)

    def pending_paths(self) -> list[str]:
        return [p for p in self._paths if self._status.get(p) != STATUS_DONE]

    def reset_status(self) -> None:
        for p in self._paths:
            self._status[p] = STATUS_PENDING
            self._messages[p] = ""
        self._refresh_all()

    def set_status(self, path: str, status: str, message: str = "") -> None:
        norm = os.path.abspath(path)
        if norm not in self._status:
            return
        self._status[norm] = status
        if message:
            self._messages[norm] = message
        elif status == STATUS_PENDING:
            self._messages[norm] = ""
        self._refresh_all()

    def _refresh_all(self) -> None:
        for row in range(self.list.count()):
            self._refresh_item(self.list.item(row))

    def _emit_changed(self) -> None:
        self._refresh_hint()
        self.filesChanged.emit(len(self._paths))

    def _refresh_hint(self) -> None:
        self._hintLabel.setVisible(not self._paths)


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


