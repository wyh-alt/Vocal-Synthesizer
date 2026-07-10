"""应用配置读写。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal


EngineMode = Literal["onnx", "pytorch"]


@dataclass
class AppConfig:
    engine: EngineMode = "onnx"
    voicebank_path: str = ""
    diffsinger_root: str = ""
    variance_exp: str = ""
    acoustic_exp: str = ""
    speaker: str = "Standard"
    language: str = "auto"
    melody_track: str = "auto"
    diffusion_steps: int = 20
    pitch_steps: int = 10
    variance_steps: int = 10
    shallow_depth: float = 0.6
    seed: int = 42
    velocity: float = 1.0
    output_dir: str = ""
    last_midi_dir: str = ""
    last_output_dir: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_path(app_dir: str) -> str:
    return os.path.join(app_dir, "config.json")


def default_voicebank_path(app_dir: str) -> str:
    candidate = os.path.join(app_dir, "Nishiren Diffsinger v2.0")
    if os.path.isdir(candidate) and os.path.isfile(
        os.path.join(candidate, "dsconfig.yaml")
    ):
        return candidate
    return ""


def default_diffsinger_root(app_dir: str) -> str:
    candidate = os.path.join(app_dir, "DiffSinger-2.5.1")
    if os.path.isdir(candidate) and os.path.isfile(
        os.path.join(candidate, "scripts", "infer.py")
    ):
        return candidate
    return ""


def load_config(app_dir: str, user_dir: str | None = None) -> AppConfig:
    """从 user_dir/config.json 读取用户配置；找不到则回退到 app_dir 的默认
    模板。声库/DiffSinger 根目录若未设置，在 app_dir 下自动探测（打包后
    app_dir 是解压出的 _MEIPASS，声库正好被 PyInstaller 释放在那里）。"""
    user_dir = user_dir or app_dir
    path = config_path(user_dir)
    if not os.path.isfile(path):
        # 首次启动：如果用户目录没有 config.json，尝试用 app_dir 里的模板
        template = config_path(app_dir)
        if os.path.isfile(template) and template != path:
            try:
                with open(template, "r", encoding="utf-8") as f:
                    config = AppConfig.from_dict(json.load(f))
            except (OSError, json.JSONDecodeError, TypeError):
                config = AppConfig()
        else:
            config = AppConfig()
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = AppConfig.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError, TypeError):
            config = AppConfig()

    if not config.voicebank_path or not os.path.isdir(config.voicebank_path):
        detected = default_voicebank_path(app_dir)
        if detected:
            config.voicebank_path = detected
    if not config.diffsinger_root or not os.path.isdir(config.diffsinger_root):
        detected = default_diffsinger_root(app_dir)
        if detected:
            config.diffsinger_root = detected
    if not config.speaker:
        config.speaker = "Standard"
    return config


def save_config(user_dir: str, config: AppConfig) -> None:
    path = config_path(user_dir)
    os.makedirs(user_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
