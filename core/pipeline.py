"""MIDI 导唱生成完整流水线。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from core.audio_aligner import align_duration
from core.config import AppConfig
from core.diffsinger_engine import DiffSingerEngine, DiffSingerSettings
from core.ds_builder import build_ds_segment, format_ds_preview, write_ds_file
from core.lyric_phoneme import detect_dominant_language
from core.midi_parser import ParsedMidi, parse_midi_file, summarize_parsed_midi
from core.onnx_voicebank_engine import NishirenOnnxEngine, VoicebankSettings

_AUTO_LANGUAGE_VALUES = {"", "auto"}


@dataclass
class PipelineResult:
    success: bool
    output_wav: str = ""
    ds_path: str = ""
    message: str = ""
    parsed_summary: str = ""
    ds_preview: str = ""
    align_info: dict | None = None


class GuideVocalPipeline:
    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def _language_is_auto(self) -> bool:
        return (self.config.language or "").strip().lower() in _AUTO_LANGUAGE_VALUES

    def _effective_language(self, parsed: ParsedMidi) -> str:
        """"auto"/空 时，按 MIDI 歌词内容自动选出出现最多的语种作为默认
        语言；每个音符自身仍按其文本各自检测语种（见
        core.lyric_phoneme.phonemes_for_note），混合中/英/韩歌词的歌曲
        无需用户手动指定、也不会被强制统一成一种语言。"""
        if not self._language_is_auto:
            return self.config.language
        return detect_dominant_language(parsed)

    def _pytorch_engine(self, language: str) -> DiffSingerEngine:
        return DiffSingerEngine(
            DiffSingerSettings(
                root_dir=self.config.diffsinger_root,
                variance_exp=self.config.variance_exp,
                acoustic_exp=self.config.acoustic_exp,
                speaker=self.config.speaker,
                language=language,
                diffusion_steps=self.config.diffusion_steps,
                shallow_depth=self.config.shallow_depth,
            )
        )

    def _onnx_engine(self, language: str) -> NishirenOnnxEngine:
        return NishirenOnnxEngine(
            VoicebankSettings(
                voicebank_path=self.config.voicebank_path,
                speaker=self.config.speaker or "Standard",
                language=language,
                acoustic_steps=self.config.diffusion_steps,
                pitch_steps=self.config.pitch_steps,
                variance_steps=self.config.variance_steps,
                seed=self.config.seed,
                velocity=self.config.velocity,
            )
        )

    def preview(self, midi_path: str) -> tuple[str, str]:
        parsed = parse_midi_file(midi_path, self.config.melody_track)
        language = self._effective_language(parsed)
        segment = build_ds_segment(
            parsed,
            language,
            self.config.voicebank_path,
        )
        preview = format_ds_preview(segment)
        if self.config.engine == "onnx" and self.config.voicebank_path:
            preview += f"\n\n声库: {os.path.basename(self.config.voicebank_path)}"
            preview += f"\n说话人: {self.config.speaker or 'Standard'}"
        preview += f"\n语言: {language}" + (" (自动检测)" if self._language_is_auto else "")
        return summarize_parsed_midi(parsed), preview

    def run(
        self,
        midi_path: str,
        output_wav: Optional[str] = None,
        log: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        log = log or (lambda _msg: None)
        midi_path = os.path.abspath(midi_path)

        if not os.path.isfile(midi_path):
            return PipelineResult(False, message=f"MIDI 文件不存在: {midi_path}")

        if output_wav is None:
            base = os.path.splitext(os.path.basename(midi_path))[0]
            out_dir = self.config.output_dir or os.path.dirname(midi_path)
            output_wav = os.path.join(out_dir, f"{base}_导唱.wav")

        output_wav = os.path.abspath(output_wav)
        out_dir = os.path.dirname(output_wav)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        log("解析 MIDI...")
        parsed = parse_midi_file(midi_path, self.config.melody_track)
        summary = summarize_parsed_midi(parsed)
        log(summary)

        if not parsed.notes:
            return PipelineResult(False, message="MIDI 中未找到音符", parsed_summary=summary)

        if parsed.lyric_note_count == 0:
            log("警告: 未检测到歌词，将按 SP 休止处理")

        language = self._effective_language(parsed)
        if self._language_is_auto:
            log(f"自动检测演唱语言: {language}")

        segment = build_ds_segment(
            parsed,
            language,
            self.config.voicebank_path,
        )
        ds_preview = format_ds_preview(segment)
        log(ds_preview)

        work_dir = tempfile.mkdtemp(prefix="midi_guide_")
        ds_path = os.path.join(work_dir, "input.ds")
        write_ds_file(ds_path, segment)

        raw_wav = os.path.join(work_dir, "raw.wav")

        if self.config.engine == "onnx":
            engine = self._onnx_engine(language)
            errors = engine.validate()
            if errors:
                return PipelineResult(
                    False,
                    ds_path=ds_path,
                    message="ONNX 声库配置不完整:\n" + "\n".join(errors),
                    parsed_summary=summary,
                    ds_preview=ds_preview,
                )
            log(f"使用 ONNX 声库: {self.config.voicebank_path}")
            log(f"说话人: {self.config.speaker or 'Standard'}")
            try:
                engine.synthesize(parsed, raw_wav, log)
            except Exception as exc:
                return PipelineResult(
                    False,
                    ds_path=ds_path,
                    message=str(exc),
                    parsed_summary=summary,
                    ds_preview=ds_preview,
                )
        else:
            pt_engine = self._pytorch_engine(language)
            errors = pt_engine.validate()
            if errors:
                return PipelineResult(
                    False,
                    ds_path=ds_path,
                    message="DiffSinger PyTorch 配置不完整:\n" + "\n".join(errors),
                    parsed_summary=summary,
                    ds_preview=ds_preview,
                )
            log("开始 DiffSinger PyTorch 合成...")
            try:
                pt_engine.synthesize(ds_path, raw_wav, log)
            except Exception as exc:
                return PipelineResult(
                    False,
                    ds_path=ds_path,
                    message=str(exc),
                    parsed_summary=summary,
                    ds_preview=ds_preview,
                )

        log("对齐音频时长到 MIDI...")
        align_info = align_duration(raw_wav, output_wav, parsed.duration_sec)
        log(
            f"时长: {align_info['input_duration']:.2f}s -> "
            f"{align_info['output_duration']:.2f}s ({align_info['method']})"
        )

        return PipelineResult(
            success=True,
            output_wav=output_wav,
            ds_path=ds_path,
            message="生成完成",
            parsed_summary=summary,
            ds_preview=ds_preview,
            align_info=align_info,
        )
