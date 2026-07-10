"""调用本地 DiffSinger 安装进行 variance + acoustic 推理。"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DiffSingerSettings:
    root_dir: str
    variance_exp: str
    acoustic_exp: str
    speaker: str = ""
    language: str = "en"
    diffusion_steps: int = 50
    shallow_depth: float = 0.6


class DiffSingerEngine:
    def __init__(self, settings: DiffSingerSettings):
        self.settings = settings

    def validate(self) -> list[str]:
        errors: list[str] = []
        root = self.settings.root_dir
        if not root:
            errors.append("未配置 DiffSinger 根目录")
            return errors
        if not os.path.isdir(root):
            errors.append(f"DiffSinger 目录不存在: {root}")
            return errors

        infer_py = os.path.join(root, "scripts", "infer.py")
        if not os.path.isfile(infer_py):
            errors.append(f"未找到 scripts/infer.py: {infer_py}")

        ckpt_root = os.path.join(root, "checkpoints")
        if not os.path.isdir(ckpt_root):
            errors.append(f"未找到 checkpoints 目录: {ckpt_root}")
        else:
            if self.settings.variance_exp and not self._find_exp_dir(
                self.settings.variance_exp
            ):
                errors.append(
                    f"未找到 variance 模型: {self.settings.variance_exp}"
                )
            if self.settings.acoustic_exp and not self._find_exp_dir(
                self.settings.acoustic_exp
            ):
                errors.append(
                    f"未找到 acoustic 模型: {self.settings.acoustic_exp}"
                )
        return errors

    def _find_exp_dir(self, exp_name: str) -> Optional[str]:
        direct = os.path.join(self.settings.root_dir, "checkpoints", exp_name)
        if os.path.isdir(direct):
            return direct
        pattern = os.path.join(
            self.settings.root_dir, "checkpoints", f"{exp_name}*"
        )
        matches = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        return matches[0] if matches else None

    def _python_cmd(self) -> list[str]:
        venv_python = os.path.join(self.settings.root_dir, ".venv", "Scripts", "python.exe")
        if os.path.isfile(venv_python):
            return [venv_python]
        return [sys.executable]

    def _run(
        self,
        args: list[str],
        log: Callable[[str], None],
        cwd: Optional[str] = None,
    ) -> None:
        cmd = self._python_cmd() + args
        log(f"$ {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=cwd or self.settings.root_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout:
            for line in proc.stdout.splitlines():
                log(line)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                log(line)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DiffSinger 命令失败 (code={proc.returncode})"
            )

    def run_variance(
        self,
        ds_path: str,
        out_dir: str,
        log: Callable[[str], None],
    ) -> str:
        infer_py = os.path.join(self.settings.root_dir, "scripts", "infer.py")
        args = [
            infer_py,
            "variance",
            ds_path,
            "--exp",
            self.settings.variance_exp,
            "--out",
            out_dir,
        ]
        if self.settings.speaker:
            args.extend(["--spk", self.settings.speaker])
        if self.settings.language:
            args.extend(["--lang", self.settings.language])
        if self.settings.diffusion_steps:
            args.extend(["--steps", str(self.settings.diffusion_steps)])

        self._run(args, log)
        base = os.path.splitext(os.path.basename(ds_path))[0]
        for candidate in (
            os.path.join(out_dir, f"{base}_variance.ds"),
            os.path.join(out_dir, f"{base}.ds"),
            ds_path,
        ):
            if os.path.isfile(candidate):
                return candidate
        return ds_path

    def run_acoustic(
        self,
        ds_path: str,
        out_dir: str,
        title: str,
        log: Callable[[str], None],
    ) -> str:
        infer_py = os.path.join(self.settings.root_dir, "scripts", "infer.py")
        args = [
            infer_py,
            "acoustic",
            ds_path,
            "--exp",
            self.settings.acoustic_exp,
            "--out",
            out_dir,
            "--title",
            title,
        ]
        if self.settings.speaker:
            args.extend(["--spk", self.settings.speaker])
        if self.settings.language:
            args.extend(["--lang", self.settings.language])
        if self.settings.diffusion_steps:
            args.extend(["--steps", str(self.settings.diffusion_steps)])
        if self.settings.shallow_depth is not None:
            args.extend(["--depth", str(self.settings.shallow_depth)])

        self._run(args, log)

        candidates = sorted(
            glob.glob(os.path.join(out_dir, f"{title}*.wav")),
            key=os.path.getmtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

        candidates = sorted(
            glob.glob(os.path.join(out_dir, "*.wav")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"未在 {out_dir} 找到生成的 wav 文件")
        return candidates[0]

    def synthesize(
        self,
        ds_path: str,
        output_wav: str,
        log: Callable[[str], None],
    ) -> str:
        errors = self.validate()
        if errors:
            raise RuntimeError("\n".join(errors))

        work_dir = tempfile.mkdtemp(prefix="ds_infer_")
        title = os.path.splitext(os.path.basename(output_wav))[0]

        log("运行 variance 模型（预测时长与 F0）...")
        variance_ds = self.run_variance(ds_path, work_dir, log)

        log("运行 acoustic 模型（合成人声）...")
        raw_wav = self.run_acoustic(variance_ds, work_dir, title, log)

        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        if os.path.abspath(raw_wav) != os.path.abspath(output_wav):
            import shutil

            shutil.copy2(raw_wav, output_wav)
        return output_wav
