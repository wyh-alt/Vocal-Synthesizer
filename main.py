import ctypes
import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _user_data_dir() -> str:
    """打包后 sys._MEIPASS 每次启动都是新的临时目录，把 config.json 放在
    可写的用户数据目录，避免被冲掉；开发模式下就用项目根目录。"""
    if getattr(sys, "frozen", False):
        base = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "VocalSynthesizer")
        os.makedirs(base, exist_ok=True)
        return base
    return os.path.dirname(os.path.abspath(__file__))


def main():
    try:
        myappid = "midiguide.diffsinger.vocal.v1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    setTheme(Theme.AUTO)

    app_dir = _app_dir()
    user_dir = _user_data_dir()
    icon_path = os.path.join(app_dir, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    from ui.splash import Splash

    splash = Splash(app_dir)
    splash.show()
    app.processEvents()

    splash.showMessage("加载配置...")
    app.processEvents()
    from ui.main_window import MainWindow

    splash.showMessage("初始化界面...")
    app.processEvents()
    window = MainWindow(app_dir, user_dir)

    splash.showMessage("就绪")
    app.processEvents()
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
