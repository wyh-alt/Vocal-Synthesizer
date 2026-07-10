import os
import webbrowser
from dataclasses import replace

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon as FIF,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    NavigationItemPosition,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    DoubleSpinBox,
    SpinBox,
    StrongBodyLabel,
    TextEdit,
    TitleLabel,
)

from core.config import AppConfig, load_config, save_config
from core.pipeline import GuideVocalPipeline, PipelineResult
from ui.widgets import (
    BatchProgressPanel,
    DragLineEdit,
    collect_midi_files,
    create_compact_combo,
)

SPEAKER_OPTIONS = [
    ("Standard（默认）", "Standard"),
    ("Sweet", "Sweet"),
    ("Soft", "Soft"),
    ("Power", "Power"),
    ("Emotional", "Emotional"),
    ("2P (Kayama)", "2P"),
]

LANGUAGE_OPTIONS = [
    ("自动识别（Auto）", "auto"),
    ("English", "en"),
    ("中文", "zh"),
    ("日本語", "ja"),
    ("한국어", "ko"),
]


class BatchWorker(QThread):
    """批量顺序合成：每个文件产生 itemStarted / itemProgress / itemFinished 信号，
    最后再发一次 allFinished(success_count, failed_count)。"""

    itemStarted = pyqtSignal(str, int, int)              # path, index (1-based), total
    itemProgress = pyqtSignal(str, str)                  # path, message
    itemFinished = pyqtSignal(str, object)               # path, PipelineResult
    allFinished = pyqtSignal(int, int)                   # success_count, failed_count

    def __init__(
        self,
        midi_paths: list[str],
        output_dir: str,
        config: AppConfig,
        parent=None,
    ):
        super().__init__(parent)
        self.midi_paths = list(midi_paths)
        self.output_dir = output_dir.strip()
        self.config = config
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _output_for(self, midi_path: str) -> str | None:
        base = os.path.splitext(os.path.basename(midi_path))[0]
        if self.output_dir:
            return os.path.join(self.output_dir, f"{base}_导唱.wav")
        return None  # 交给 pipeline 使用 MIDI 同目录

    def run(self):
        total = len(self.midi_paths)
        success = 0
        failed = 0
        pipeline = GuideVocalPipeline(self.config)
        for idx, midi_path in enumerate(self.midi_paths, start=1):
            if self._cancel:
                break
            self.itemStarted.emit(midi_path, idx, total)
            try:
                result = pipeline.run(
                    midi_path,
                    self._output_for(midi_path),
                    log=lambda msg, p=midi_path: self.itemProgress.emit(p, msg),
                )
            except Exception as exc:
                result = PipelineResult(False, message=f"未捕获异常: {exc}")
            self.itemFinished.emit(midi_path, result)
            if result.success:
                success += 1
            else:
                failed += 1
        self.allFinished.emit(success, failed)


class GeneratePage(QWidget):
    def __init__(self, app_dir: str, user_dir: str, parent=None):
        super().__init__(parent)
        self.app_dir = app_dir
        self.user_dir = user_dir
        self.config = load_config(app_dir, user_dir)
        self._worker: BatchWorker | None = None
        self._first_success_dir = ""
        self._current_index = 0
        self._current_total = 0

        self.setObjectName("generatePage")
        self._build_ui()
        self._load_config_to_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = TitleLabel("MIDI 导唱生成", self)
        subtitle = BodyLabel(
            "读取 ACE Studio 风格 MIDI（音符 + 歌词），通过 DiffSinger 合成自然导唱 WAV。",
            self,
        )
        root.addWidget(title)
        root.addWidget(subtitle)

        scroll = ScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.viewport().setStyleSheet("background: transparent;")
        container = QWidget(scroll)
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet("background: transparent;")
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        io_card = CardWidget(container)
        io_layout = QVBoxLayout(io_card)
        io_layout.setSpacing(10)
        io_layout.addWidget(StrongBodyLabel("输入 / 输出", io_card))

        in_row = QHBoxLayout()
        self.inputEdit = DragLineEdit(io_card)
        self.inputEdit.setPlaceholderText("将 MIDI 文件或文件夹拖到这里")
        in_browse = PushButton("浏览...", io_card)
        in_browse.clicked.connect(self._browse_input)
        in_row.addWidget(self.inputEdit, 1)
        in_row.addWidget(in_browse)
        io_layout.addLayout(in_row)

        out_row = QHBoxLayout()
        self.outputDirEdit = DragLineEdit(io_card)
        self.outputDirEdit.setPlaceholderText("输出目录（留空则与 MIDI 同目录）")
        out_browse = PushButton("浏览...", io_card)
        out_browse.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self.outputDirEdit, 1)
        out_row.addWidget(out_browse)
        io_layout.addLayout(out_row)

        layout.addWidget(io_card)

        ds_card = CardWidget(container)
        ds_layout = QVBoxLayout(ds_card)
        ds_layout.setSpacing(10)
        ds_layout.addWidget(StrongBodyLabel("Nishiren / ONNX 声库", ds_card))

        vb_row = QHBoxLayout()
        self.voicebankEdit = DragLineEdit(ds_card)
        self.voicebankEdit.setPlaceholderText("声库目录（Nishiren Diffsinger v2.0）")
        vb_btn = PushButton("浏览", ds_card)
        vb_btn.clicked.connect(self._browse_voicebank)
        vb_row.addWidget(self.voicebankEdit, 1)
        vb_row.addWidget(vb_btn)
        ds_layout.addLayout(vb_row)

        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(12)
        param_grid.setVerticalSpacing(10)
        param_grid.setColumnStretch(1, 1)
        param_grid.setColumnStretch(3, 1)

        def _param_label(text: str) -> BodyLabel:
            label = BodyLabel(text, ds_card)
            label.setMinimumWidth(96)
            return label

        self.speakerCombo = create_compact_combo(ds_card, min_width=140, max_width=220)
        for label, value in SPEAKER_OPTIONS:
            self.speakerCombo.addItem(label, userData=value)
        param_grid.addWidget(_param_label("说话人"), 0, 0)
        param_grid.addWidget(self.speakerCombo, 0, 1)

        self.langCombo = create_compact_combo(ds_card, min_width=140, max_width=220)
        for label, value in LANGUAGE_OPTIONS:
            self.langCombo.addItem(label, userData=value)
        param_grid.addWidget(_param_label("语言"), 0, 2)
        param_grid.addWidget(self.langCombo, 0, 3)

        self.stepsSpin = SpinBox(ds_card)
        self.stepsSpin.setRange(5, 50)
        self.stepsSpin.setValue(20)
        param_grid.addWidget(_param_label("Acoustic 步数"), 1, 0)
        param_grid.addWidget(self.stepsSpin, 1, 1)

        self.velocitySpin = DoubleSpinBox(ds_card)
        self.velocitySpin.setRange(50.0, 200.0)
        self.velocitySpin.setSingleStep(5.0)
        self.velocitySpin.setValue(100.0)
        param_grid.addWidget(_param_label("咬字速度 %"), 1, 2)
        param_grid.addWidget(self.velocitySpin, 1, 3)

        ds_layout.addLayout(param_grid)

        help_label = CaptionLabel(
            "推荐 10 步快速试听，20–30 步最终导出质量。"
            "语言选择「自动识别」时按 MIDI 歌词内容自动判断，支持中英日韩混合歌词。"
            "咬字若含糊，可调整咬字速度：偏快的歌曲试试 110–130，偏慢的试试 90–100。",
            ds_card,
        )
        help_label.setWordWrap(True)
        ds_layout.addWidget(help_label)

        layout.addWidget(ds_card)

        self.progressPanel = BatchProgressPanel(container)
        self.generateBtn = PrimaryPushButton(FIF.PLAY, "生成导唱", container)
        self.generateBtn.clicked.connect(self._generate)
        self.generateBtn.setMinimumHeight(40)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        bottom_row.addWidget(self.progressPanel, 1)
        bottom_row.addWidget(self.generateBtn, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(bottom_row)

        layout.addStretch(1)
        root.addWidget(scroll, 1)

    def _load_config_to_ui(self):
        self.outputDirEdit.setText(self.config.output_dir or "")
        self.voicebankEdit.setText(self.config.voicebank_path)
        self.stepsSpin.setValue(self.config.diffusion_steps or 20)
        self.velocitySpin.setValue((self.config.velocity or 1.0) * 100.0)
        speaker = self.config.speaker or "Standard"
        for index in range(self.speakerCombo.count()):
            if self.speakerCombo.itemData(index) == speaker:
                self.speakerCombo.setCurrentIndex(index)
                break
        language = (self.config.language or "auto").strip().lower()
        for index in range(self.langCombo.count()):
            if self.langCombo.itemData(index) == language:
                self.langCombo.setCurrentIndex(index)
                break
        else:
            self.langCombo.setCurrentIndex(0)

    def _collect_config(self) -> AppConfig:
        return AppConfig(
            engine="onnx",
            voicebank_path=self.voicebankEdit.text().strip(),
            diffsinger_root=self.config.diffsinger_root,
            variance_exp=self.config.variance_exp,
            acoustic_exp=self.config.acoustic_exp,
            speaker=str(self.speakerCombo.currentData() or "Standard"),
            language=str(self.langCombo.currentData() or "auto"),
            melody_track=self.config.melody_track,
            diffusion_steps=self.stepsSpin.value(),
            pitch_steps=self.config.pitch_steps,
            variance_steps=self.config.variance_steps,
            shallow_depth=self.config.shallow_depth,
            seed=self.config.seed,
            velocity=self.velocitySpin.value() / 100.0,
            output_dir=self.outputDirEdit.text().strip(),
            last_midi_dir=self.config.last_midi_dir,
            last_output_dir=self.config.last_output_dir,
        )

    def _save_config(self):
        self.config = self._collect_config()
        save_config(self.user_dir, self.config)

    def _browse_input(self):
        """按用户要求：仅打开目录选择器。要单个文件的话拖进来即可。"""
        start = self.inputEdit.text().strip() or self.config.last_midi_dir or self.app_dir
        if not os.path.isdir(start):
            start = os.path.dirname(start) if os.path.isfile(start) else self.app_dir
        path = QFileDialog.getExistingDirectory(self, "选择 MIDI 所在文件夹", start)
        if path:
            self.inputEdit.setText(path)
            self.config = replace(self.config, last_midi_dir=path)
            save_config(self.user_dir, self.config)

    def _browse_output_dir(self):
        start = self.outputDirEdit.text() or self.config.last_output_dir or self.app_dir
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if path:
            self.outputDirEdit.setText(path)
            self.config = replace(
                self.config, output_dir=path, last_output_dir=path
            )
            save_config(self.user_dir, self.config)

    def _browse_voicebank(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "选择 ONNX 声库目录",
            self.voicebankEdit.text() or self.app_dir,
        )
        if path:
            self.voicebankEdit.setText(path)

    def _resolve_input_paths(self, raw: str) -> tuple[list[str], str]:
        """把输入框文本解析为具体 MIDI 文件列表。
        - 空 → 错误提示
        - 单个 .mid/.midi 文件 → [该文件]
        - 目录 → 递归收集其中所有 .mid/.midi
        - 其他 → 报错
        返回 (列表, 错误信息)，两者互斥。"""
        raw = (raw or "").strip().strip('"').strip("'")
        if not raw:
            return [], "请先选择 MIDI 文件或包含 MIDI 的文件夹"
        if not os.path.exists(raw):
            return [], f"路径不存在: {raw}"
        if os.path.isfile(raw):
            if not raw.lower().endswith((".mid", ".midi")):
                return [], "文件必须是 .mid 或 .midi"
            return [os.path.abspath(raw)], ""
        if os.path.isdir(raw):
            files = collect_midi_files(raw)
            if not files:
                return [], f"该文件夹（含子目录）内未找到 .mid / .midi 文件"
            return files, ""
        return [], f"无法识别的路径: {raw}"

    def _generate(self):
        paths, err = self._resolve_input_paths(self.inputEdit.text())
        if err:
            InfoBar.warning(
                "无法生成", err,
                parent=self, position=InfoBarPosition.TOP, duration=4000,
            )
            return

        self._save_config()
        self.generateBtn.setEnabled(False)
        self._first_success_dir = ""
        self._failed_items: list[tuple[str, str]] = []

        output_dir = self.outputDirEdit.text().strip()
        self.progressPanel.set_progress(1, f"开始批处理 {len(paths)} 个文件...")

        self._worker = BatchWorker(paths, output_dir, self.config, self)
        self._worker.itemStarted.connect(self._on_item_started)
        self._worker.itemProgress.connect(self._on_item_progress)
        self._worker.itemFinished.connect(self._on_item_finished)
        self._worker.allFinished.connect(self._on_batch_finished)
        self._worker.start()

    def _on_item_started(self, midi_path: str, index: int, total: int):
        self._current_total = total
        self._current_index = index
        base = os.path.basename(midi_path)
        pct = int((index - 1) / max(1, total) * 100)
        self.progressPanel.set_progress(pct, f"[{index}/{total}] {base} —— 解析中")

    def _on_item_progress(self, midi_path: str, message: str):
        first_line = message.strip().splitlines()[0] if message else ""
        if not first_line:
            return
        base = os.path.basename(midi_path)
        pct = int((self._current_index - 1) / max(1, self._current_total) * 100)
        self.progressPanel.set_progress(
            pct, f"[{self._current_index}/{self._current_total}] {base} —— {first_line[:60]}"
        )

    def _on_item_finished(self, midi_path: str, result: PipelineResult):
        if result.success:
            if not self._first_success_dir:
                self._first_success_dir = os.path.dirname(result.output_wav)
        else:
            first_line = (result.message or "失败").strip().splitlines()[0]
            self._failed_items.append((os.path.basename(midi_path), first_line[:200]))

    def _on_batch_finished(self, success: int, failed: int):
        self._worker = None
        total = success + failed
        self.progressPanel.set_progress(100, f"完成：成功 {success} / 失败 {failed}（共 {total}）")
        self.generateBtn.setEnabled(True)

        if self._first_success_dir:
            self.progressPanel.openBtn.setEnabled(True)
            try:
                self.progressPanel.openBtn.clicked.disconnect()
            except TypeError:
                pass
            self.progressPanel.openBtn.clicked.connect(
                lambda: os.startfile(self._first_success_dir)
            )

        if failed == 0 and success > 0:
            InfoBar.success(
                "全部完成", f"成功生成 {success} 个 WAV",
                parent=self, position=InfoBarPosition.TOP, duration=5000,
            )
        elif success > 0:
            self._show_failed_dialog(f"部分完成：成功 {success}，失败 {failed}")
        else:
            self._show_failed_dialog("全部失败")

    def _show_failed_dialog(self, title: str):
        if not self._failed_items:
            return
        lines = [f"• {name}\n    {msg}" for name, msg in self._failed_items[:20]]
        if len(self._failed_items) > 20:
            lines.append(f"...还有 {len(self._failed_items) - 20} 个")
        box = MessageBox(title, "\n".join(lines), self)
        box.yesButton.setText("确定")
        box.cancelButton.hide()
        box.exec()


class HelpPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("helpPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(TitleLabel("使用说明", self))

        text = TextEdit(self)
        text.setReadOnly(True)
        text.setMarkdown(
            """## 功能

- 读取 ACE Studio 导出的 MIDI（同轨音符 + 歌词）
- 支持歌词格式：
  - **多音节**：`baby#1` / `baby#2`
  - **延音**：`baby` + `-`（不换字）
- 使用 **DiffSinger** 合成导唱 WAV，时长与 MIDI 对齐

## DiffSinger 环境

1. 克隆 [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger)
2. 准备 variance + acoustic 模型 checkpoint
3. 下载 NSF-HiFiGAN 声码器
4. 在本程序填写 DiffSinger 根目录与实验名

## 歌词格式

| MIDI 歌词 | 含义 |
|-----------|------|
| `baby#1` | 第 1 音节 |
| `baby#2` | 第 2 音节 |
| `-` | 延音 / slur |
"""
        )
        layout.addWidget(text)

        link_row = QHBoxLayout()
        ds_btn = PushButton(FIF.LINK, "DiffSinger 文档", self)
        ds_btn.clicked.connect(
            lambda: webbrowser.open(
                "https://github.com/openvpi/DiffSinger/blob/main/docs/GettingStarted.md"
            )
        )
        link_row.addWidget(ds_btn)
        link_row.addStretch(1)
        layout.addLayout(link_row)


class MainWindow(FluentWindow):
    def __init__(self, app_dir: str, user_dir: str | None = None):
        super().__init__()
        self.app_dir = app_dir
        self.user_dir = user_dir or app_dir
        self.setWindowTitle("MIDI 导唱生成 · Vocal Synthesizer v1.0")
        self.resize(900, 600)
        self.setMinimumSize(900, 600)

        icon_path = os.path.join(app_dir, "icon.ico")
        if os.path.exists(icon_path):
            from PyQt6.QtGui import QIcon

            self.setWindowIcon(QIcon(icon_path))

        self.generatePage = GeneratePage(app_dir, self.user_dir, self)
        self.helpPage = HelpPage(self)

        self.addSubInterface(
            self.generatePage, FIF.MUSIC, "导唱生成", NavigationItemPosition.TOP
        )
        self.addSubInterface(
            self.helpPage, FIF.INFO, "说明", NavigationItemPosition.BOTTOM
        )
