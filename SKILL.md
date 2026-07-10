---
name: "pyqt6-fluent-ui"
description: "Generates modern Windows 11 style GUI applications using PyQt6 and qfluentwidgets. Invoke when the user asks to build, design, or update a desktop UI."
---

# PyQt6 Fluent UI Design Guidelines

当需要为应用程序开发GUI（图形用户界面）时，请默认遵循以下UI设计逻辑与规范，以保持与Windows 11及现有工具的一致性。

## 1. 核心技术栈与整体风格
- **基础框架**: `PyQt6` 
- **UI组件库**: `qfluentwidgets` (提供原生Windows 11 Fluent Design风格)
- **主窗口**: 继承 `FluentWindow`，并使用 `addSubInterface` 添加侧边栏导航页面。
- **自适应与主题**: 
  - 必须包含 `QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)` 以支持高DPI屏幕。
  - 使用 `setTheme(Theme.AUTO)` 跟随系统亮暗主题。
- **布局容器**: 页面主要使用 `ScrollArea` 配合内部 `QWidget`，使用 `CardWidget` 对功能模块进行分组，标签使用 `TitleLabel`、`StrongBodyLabel` 和 `BodyLabel` 以区分层级。

## 2. 拖拽上传功能的实现 (Drag & Drop)
对于涉及文件/文件夹路径输入的组件，**必须支持拖拽上传**。通过重写 `LineEdit` 实现一个通用的 `DragLineEdit` 类：

```python
from qfluentwidgets import LineEdit

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
```
- **使用方式**: 替换原有的 `LineEdit` 为 `DragLineEdit`，并监听 `textChanged` 信号来解析路径。同时建议保留一个“浏览”按钮 (使用 `PushButton`) 提供传统的文件选择方式。

## 3. 应用程序图标配置 (Icon Design)
- **图标生成**: 当创建新程序时，需根据程序功能要求生成或配置一个相应的 `icon.ico` 文件（放置在与主程序同级的目录）。
- **图标加载逻辑**: 必须处理Windows任务栏图标显示问题，并设置全局及窗口图标：
```python
import os
import ctypes
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

# 1. 解决 Windows 任务栏图标可能不显示的问题
try:
    myappid = 'mycompany.myproduct.subproduct.version' # 根据具体项目修改
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

# 2. 设置应用程序全局图标
app = QApplication(sys.argv)
icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
if os.path.exists(icon_path):
    app.setWindowIcon(QIcon(icon_path))

# 3. 设置主窗口图标 (在 MainWindow 的 __init__ 中)
if os.path.exists(icon_path):
    self.setWindowIcon(QIcon(icon_path))
```

## 4. 交互与反馈
- **耗时操作**: 所有数据处理或耗时操作必须通过继承 `QThread` 在后台执行，通过 `pyqtSignal` 与主线程通信，避免界面卡顿。
- **提示信息**: 
  - 轻量提示（如"开始处理"）使用 `InfoBar.info()` / `InfoBar.success()` / `InfoBar.warning()`。
  - 详细总结或严重错误使用 `MessageBox`。
- **动作按钮**: 页面主要执行按钮需使用 `PrimaryPushButton` 以突出显示，次要操作使用普通的 `PushButton`。
