"""启动画面。"""

from __future__ import annotations

import os

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QSplashScreen


APP_VERSION = "1.0.0"

_WIDTH = 480
_HEIGHT = 300


def _build_pixmap(app_dir: str) -> QPixmap:
    pix = QPixmap(_WIDTH, _HEIGHT)
    pix.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    grad = QLinearGradient(0, 0, _WIDTH, _HEIGHT)
    grad.setColorAt(0.0, QColor(94, 129, 244))
    grad.setColorAt(1.0, QColor(66, 84, 214))
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, _WIDTH, _HEIGHT), 18, 18)
    painter.fillPath(path, QBrush(grad))

    icon_path = os.path.join(app_dir, "icon.png")
    if os.path.isfile(icon_path):
        icon_pix = QPixmap(icon_path).scaled(
            96, 96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap((_WIDTH - 96) // 2, 46, icon_pix)

    title_font = QFont("Microsoft YaHei UI", 18, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(
        QRectF(0, 156, _WIDTH, 32),
        int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
        "MIDI 导唱生成",
    )

    sub_font = QFont("Microsoft YaHei UI", 9)
    painter.setFont(sub_font)
    painter.setPen(QColor(220, 230, 255))
    painter.drawText(
        QRectF(0, 194, _WIDTH, 20),
        int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
        f"Vocal Synthesizer  ·  v{APP_VERSION}",
    )

    ver_font = QFont("Microsoft YaHei UI", 8)
    painter.setFont(ver_font)
    painter.setPen(QColor(255, 255, 255, 160))
    painter.drawText(
        QRectF(0, _HEIGHT - 30, _WIDTH - 18, 18),
        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
        "Powered by DiffSinger",
    )

    painter.end()
    return pix


class Splash(QSplashScreen):
    def __init__(self, app_dir: str):
        pix = _build_pixmap(app_dir)
        super().__init__(pix, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._message_color = QColor(255, 255, 255, 220)
        self.showMessage("正在启动...", color=self._message_color)

    def showMessage(self, message: str, color: QColor | None = None) -> None:
        super().showMessage(
            message,
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
            color or self._message_color,
        )
