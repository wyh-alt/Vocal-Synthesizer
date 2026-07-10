"""将合成音频对齐到 MIDI 总时长。"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf


def load_wav(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32), int(sr)


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def align_duration(
    input_wav: str,
    output_wav: str,
    target_duration_sec: float,
    *,
    tolerance_sec: float = 0.05,
    stretch_if_needed: bool = False,
) -> dict:
    audio, sr = load_wav(input_wav)
    current_duration = audio.shape[0] / sr
    target_samples = max(int(round(target_duration_sec * sr)), 1)

    method = "unchanged"
    diff = target_duration_sec - current_duration

    if abs(diff) <= tolerance_sec:
        aligned = audio[:target_samples] if audio.shape[0] >= target_samples else audio
        if aligned.shape[0] < target_samples:
            pad = target_samples - aligned.shape[0]
            aligned = np.pad(aligned, ((0, pad), (0, 0)), mode="constant")
            method = "pad"
    elif diff > 0:
        pad = target_samples - audio.shape[0]
        aligned = np.pad(audio, ((0, pad), (0, 0)), mode="constant")
        method = "pad"
    elif stretch_if_needed and current_duration > 0:
        ratio = target_duration_sec / current_duration
        if 0.85 <= ratio <= 1.15:
            indices = np.linspace(0, audio.shape[0] - 1, target_samples)
            aligned = np.stack(
                [np.interp(indices, np.arange(audio.shape[0]), audio[:, ch]) for ch in range(audio.shape[1])],
                axis=1,
            ).astype(np.float32)
            method = "stretch"
        else:
            aligned = audio[:target_samples]
            method = "trim"
    else:
        aligned = audio[:target_samples]
        method = "trim"

    save_wav(output_wav, aligned, sr)
    return {
        "input_duration": current_duration,
        "target_duration": target_duration_sec,
        "output_duration": aligned.shape[0] / sr,
        "method": method,
        "sample_rate": sr,
    }
