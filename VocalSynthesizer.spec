# PyInstaller spec for Vocal Synthesizer v1.0
# 构建： pyinstaller --clean --noconfirm VocalSynthesizer.spec
#
# onedir 模式：产出 dist/VocalSynthesizer/ 目录 —— 包含 exe + 所有依赖 +
# 声库。整个目录压缩后即可分发，用户解压任意位置双击 exe 运行，不需要
# 安装 Python 或 pip 包。onedir 相比 onefile 启动更快，且避免每次运行
# 都把 1GB 声库解压到 %TEMP%。

from pathlib import Path

APP_NAME = "VocalSynthesizer"
ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "icon.ico"), "."),
    (str(ROOT / "icon.png"), "."),
    (str(ROOT / "config.json"), "."),
    (str(ROOT / "README.md"), "."),
    # 默认 Nishiren 声库整包
    (str(ROOT / "Nishiren Diffsinger v2.0"), "Nishiren Diffsinger v2.0"),
]

hiddenimports = [
    "onnxruntime",
    "onnx",
    "yaml",
    "mido",
    "soundfile",
    "g2p_en",
    "pypinyin",
    "qfluentwidgets",
    "core.chinese_g2p",
    "core.korean_g2p",
]

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "matplotlib",
    "pandas",
    "scipy",
    "sklearn",
    "notebook",
    "IPython",
    "tkinter",
    "PySide2",
    "PySide6",
    "PyQt5",
    "test",
    "tests",
    "unittest",
]

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX 压缩会拖慢启动 + 部分杀软误报，关掉
    console=False,           # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
