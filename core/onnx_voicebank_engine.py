"""OpenUTAU 风格 ONNX 声库推理（Nishiren 等）。"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import yaml

from core.midi_parser import ParsedMidi
from core.lyric_phoneme import (
    AP,
    SP,
    _detect_language,
    phonemes_for_note,
)

HEAD_FRAMES = 8
TAIL_FRAMES = 8

# 超过这个时长的连续静音（无歌词间奏/前奏）直接在波形上静音处理。
# 扩散模型的训练数据里很少出现这么长的连续静音，缺乏条件约束时容易
# "凭空"生成幻觉噪音/哼唱；既然从歌词时间轴已确知这段绝对没有演唱内容，
# 直接静音比让模型自由发挥更可靠。正常乐句间的换气停顿通常在 1~2 秒
# 以内，不会被这个阈值误伤。
_LONG_REST_SILENCE_THRESHOLD_SEC = 1.5
_SILENCE_FADE_SEC = 0.05

_RANDOM_OP_TYPES = (
    "RandomNormalLike",
    "RandomNormal",
    "RandomUniform",
    "RandomUniformLike",
)


def _seeded_model_bytes(path: str, seed: int) -> bytes:
    """pitch/variance/acoustic 是扩散模型，图里带无 seed 的随机采样节点
    （RandomNormalLike）。不固定 seed 时，同一份 MIDI 每次合成都会用不同
    的随机噪声起点，导致同一段转音有时唱得出来、有时被"抹平"成一个音高
    —— 结果完全靠运气、不可复现。这里在建 session 前给随机节点写死 seed，
    让同样的输入始终得到同样的输出。"""
    model = onnx.load(path)
    changed = False
    for node in model.graph.node:
        if node.op_type in _RANDOM_OP_TYPES:
            del_idx = [i for i, a in enumerate(node.attribute) if a.name == "seed"]
            for i in reversed(del_idx):
                del node.attribute[i]
            node.attribute.append(onnx.helper.make_attribute("seed", float(seed)))
            changed = True
    if not changed:
        with open(path, "rb") as f:
            return f.read()
    return model.SerializeToString()


_EN_VOWEL_PHONES = {
    "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw",
}
_KO_VOWEL_PHONES = {"a", "e", "y", "eo", "o", "w", "u", "eu", "i"}
_ZH_VOWEL_LETTERS = {
    # 单元音
    "a", "e", "i", "o", "u", "v",
    # 复合韵母/介音+主元音组合（Nishiren 声库把这些整体当作一个音素，
    # 不是单字母；漏掉这些会把 "ia"/"ie" 这类韵母误判成辅音，导致它们
    # 被当作"辅音时长上限"的裁剪对象，而不是本该承载延音的元音，
    # 出现"辅音被拖长、真正的韵母一闪而过"的错误（如"翔"被唱成拖长
    # 的"xi"再匆匆带过"ang"）。
    "ai", "ao", "ei", "ou", "er", "ia", "ie", "io",
    "ua", "uo", "ue", "ve", "i0", "ir",
}

_ANTICIPATION_MAX_LEAD_PHONES = 3
# 静音间隙可借用的比例（间隙内容为无声，损失可以忽略）
_ANTICIPATION_GAP_BORROW_RATIO = 0.85
# 相邻音节尾部（元音收尾/浊辅音）可借用的比例。
# 只截取上一音节最后一个音素的一小段尾巴，保证听感自然
# —— 元音尾的一小截被截掉几乎察觉不到，却能让当前音节
# 的元音准确落在下一个音符起点，避免整句人声随着每个
# 音节的辅音时长逐字累积滞后。
_ANTICIPATION_VOICED_BORROW_RATIO = 0.30
# 从相邻音节尾部借用的绝对上限（秒），避免元音尾被过度截断。
_ANTICIPATION_VOICED_BORROW_MAX_SEC = 0.08

# 辅音的自然发声时长基本是固定的，不会随着音符本身有多长而等比拉长。
# 音符很长（比如乐句结尾的长音）时，时长预测模型有时会把过多时间分给
# 辅音而不是元音，导致"辅音被拖长、元音反而一闪而过"的听感（例如"伤"
# 被唱成拖长的"sh"再匆匆带过"ang"）。超出这个上限的部分退还给本词内
# 最近的元音。
_MAX_CONSONANT_DURATION_SEC = 0.18


def _is_vowel_phone(mapped_phone: str) -> bool:
    """判断音素是否为元音（用于识别音节开头的辅音串，实现"辅音提前"）。
    无法识别的语言/写法一律当作"元音"处理（即立刻停止提前搜索），
    宁可少提前也不要因误判把不该动的音素提前。"""
    if mapped_phone in (AP, SP):
        return True
    if "/" in mapped_phone:
        lang, bare = mapped_phone.split("/", 1)
    else:
        lang, bare = "", mapped_phone
    bare = bare.lower()
    if lang == "en":
        return bare in _EN_VOWEL_PHONES
    if lang == "ko":
        return bare in _KO_VOWEL_PHONES
    if lang == "zh":
        return bare in _ZH_VOWEL_LETTERS
    return True


@dataclass
class PhoneToken:
    phoneme: str
    duration_sec: float
    midi: int
    lang: str
    is_slur: bool = False


@dataclass
class NoteSegment:
    midi: float
    duration_sec: float
    is_rest: bool


@dataclass
class VoicebankSettings:
    voicebank_path: str
    speaker: str = "Standard"
    language: str = "en"
    acoustic_steps: int = 10
    pitch_steps: int = 10
    variance_steps: int = 10
    seed: int = 42
    velocity: float = 1.0


class NishirenOnnxEngine:
    def __init__(self, settings: VoicebankSettings):
        self.settings = settings
        self.root = os.path.abspath(settings.voicebank_path)
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._phoneme_ids_by_module: dict[str, dict[str, int]] = {}
        self._lang_ids_by_module: dict[str, dict[str, int]] = {}
        self._ds_main: dict = {}
        self._speaker_embed: np.ndarray | None = None
        self.hop_size = 512
        self.sample_rate = 44100
        self.hidden_size = 384
        self._load()

    def _load(self) -> None:
        main_cfg_path = os.path.join(self.root, "dsconfig.yaml")
        with open(main_cfg_path, "r", encoding="utf-8") as f:
            self._ds_main = yaml.safe_load(f)

        self.hop_size = int(self._ds_main.get("hop_size", 512))
        self.sample_rate = int(self._ds_main.get("sample_rate", 44100))
        self.hidden_size = int(self._ds_main.get("hidden_size", 384))

        ph_path = os.path.join(self.root, self._ds_main["phonemes"])
        with open(ph_path, "r", encoding="utf-8") as f:
            self._phoneme_ids_by_module["dsmain"] = json.load(f)

        lang_path = os.path.join(self.root, self._ds_main["languages"])
        with open(lang_path, "r", encoding="utf-8") as f:
            main_langs = json.load(f)
        self._lang_ids_by_module["dsmain"] = main_langs

        for module in ("dspitch", "dsvariance", "dsdur"):
            cfg_path = os.path.join(self.root, module, "dsconfig.yaml")
            if not os.path.isfile(cfg_path):
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                module_cfg = yaml.safe_load(f)
            module_ph = os.path.join(self.root, module, module_cfg["phonemes"])
            with open(module_ph, "r", encoding="utf-8") as f:
                self._phoneme_ids_by_module[module] = json.load(f)
            module_lang = os.path.join(self.root, module, module_cfg["languages"])
            with open(module_lang, "r", encoding="utf-8") as f:
                self._lang_ids_by_module[module] = json.load(f)

        self._speaker_embed = self._try_load_speaker_embed(self.settings.speaker)

    @property
    def frame_ms(self) -> float:
        return self.hop_size / self.sample_rate * 1000.0

    def _model_path(self, rel_path: str) -> str:
        rel_path = rel_path.replace("\\", "/")
        return os.path.join(self.root, *rel_path.split("/"))

    def _session(self, rel_path: str) -> ort.InferenceSession:
        if rel_path not in self._sessions:
            path = self._model_path(rel_path)
            model_bytes = _seeded_model_bytes(path, self.settings.seed)
            self._sessions[rel_path] = ort.InferenceSession(
                model_bytes, providers=["CPUExecutionProvider"]
            )
        return self._sessions[rel_path]

    def _try_load_speaker_embed(self, speaker: str) -> np.ndarray | None:
        """并非所有声库都是多说话人模型（比如单一音色的声库根本没有
        embedding 文件），找不到时返回 None，而不是报错——是否真的需要
        spk_embed 由各 ONNX 模型自己的输入签名决定（见 _filter_session_
        inputs），找不到 embedding 但模型也不需要它时完全不影响合成。"""
        candidates = [
            os.path.join(self.root, "dsmain", f"{speaker}.emb"),
            os.path.join(self.root, "dsmain", speaker, "embed.emb"),
            os.path.join(self.root, speaker + ".emb"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    values = struct.unpack(f"{self.hidden_size}f", f.read())
                return np.asarray(values, dtype=np.float32)
        return None

    def _vocoder_rel_path(self) -> str:
        cfg_path = os.path.join(self.root, "dsvocoder", "vocoder.yaml")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                vocoder_cfg = yaml.safe_load(f) or {}
            model = vocoder_cfg.get("model")
            if model:
                return os.path.join("dsvocoder", model)
        return os.path.join("dsvocoder", "gda_pc-hifigan.onnx")

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.root or not os.path.isdir(self.root):
            errors.append(f"声库目录不存在: {self.root}")
            return errors

        main_cfg_path = os.path.join(self.root, "dsconfig.yaml")
        if not os.path.isfile(main_cfg_path):
            errors.append("缺少文件: dsconfig.yaml")
            return errors

        with open(main_cfg_path, "r", encoding="utf-8") as f:
            main_cfg = yaml.safe_load(f) or {}
        acoustic_rel = main_cfg.get("acoustic")
        if not acoustic_rel or not os.path.isfile(os.path.join(self.root, acoustic_rel)):
            errors.append(f"缺少 acoustic 模型: {acoustic_rel!r}")

        if not os.path.isfile(os.path.join(self.root, self._vocoder_rel_path())):
            errors.append(f"缺少 vocoder 模型: {self._vocoder_rel_path()}")

        for module, key in (("dspitch", "pitch"), ("dsvariance", "variance")):
            cfg_path = os.path.join(self.root, module, "dsconfig.yaml")
            if not os.path.isfile(cfg_path):
                errors.append(f"缺少文件: {module}/dsconfig.yaml")
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                module_cfg = yaml.safe_load(f) or {}
            rel = module_cfg.get(key)
            if not rel or not os.path.isfile(os.path.join(self.root, module, rel)):
                errors.append(f"缺少 {module} 模型: {rel!r}")

        return errors

    def tokenize(self, phoneme: str, module: str = "dsmain") -> int:
        phoneme_map = self._phoneme_ids_by_module[module]
        if phoneme not in phoneme_map:
            raise KeyError(f"声库 {module} 不支持的音素: {phoneme}")
        return int(phoneme_map[phoneme])

    # 部分声库的音素表覆盖不全（比如共享单一字母表、不区分语言前缀的
    # 声库），缺失某个音素时退化到发音最接近的替代音素，而不是直接报错
    # 中断整首歌的合成。目前只收录实际遇到过的缺口，遇到新的再补充。
    _PHONEME_FALLBACK_SUBSTITUTES = {
        "v": "f",  # 没有浊唇齿擦音 v 时，退化为最接近的清音 f
    }

    def to_voicebank_phoneme(self, phone: str, lang: str) -> str:
        if phone in (AP, SP):
            return phone
        candidates = [
            f"{lang}/{phone}",
            f"{lang}/{phone.lower()}",
            phone,
            phone.lower(),
        ]
        modules = list(self._phoneme_ids_by_module.values())
        # 优先选择在所有模块（dsmain/dspitch/dsvariance/dsdur）中都存在的写法，
        # 避免同一候选字符串在不同语言/模块间撞车导致映射到错误音素。
        for item in candidates:
            if all(item in module_map for module_map in modules):
                return item
        main_map = self._phoneme_ids_by_module.get("dsmain", {})
        for item in candidates:
            if item in main_map:
                return item

        fallback = self._PHONEME_FALLBACK_SUBSTITUTES.get(phone.lower())
        if fallback:
            try:
                return self.to_voicebank_phoneme(fallback, lang)
            except KeyError:
                pass
        raise KeyError(f"无法映射音素 {phone!r} (lang={lang})")

    def _lang_tensor(self, langs: list[str], module: str) -> np.ndarray:
        lang_map = self._lang_ids_by_module.get(module, {})
        ids = [int(lang_map.get(lang, 0)) for lang in langs]
        return np.asarray([ids], dtype=np.int64)

    @staticmethod
    def _midi_to_hz_array(midi: np.ndarray) -> np.ndarray:
        return (440.0 * (2.0 ** ((midi.astype(np.float64) - 69.0) / 12.0))).astype(
            np.float32
        )

    @staticmethod
    def _sec_list_to_frames(seconds: list[float], frame_ms: float) -> list[int]:
        """DiffSinger 风格累积取整，将秒序列转为帧数序列。"""
        if not seconds:
            return []
        frames: list[int] = []
        acc = 0
        elapsed_ms = 0.0
        for sec in seconds:
            elapsed_ms += max(sec, 0.0) * 1000.0
            target = int(round(elapsed_ms / frame_ms))
            delta = target - acc
            if sec > 0:
                delta = max(1, delta)
            else:
                delta = max(0, delta)
            frames.append(delta)
            acc += delta
        return frames

    @staticmethod
    def _match_frame_total(frames: list[int], target_total: int) -> list[int]:
        if not frames:
            return frames
        result = list(frames)
        delta = target_total - sum(result)
        if delta:
            result[-1] = max(1, result[-1] + delta)
        return result

    def _token_ids(self, phonemes: list[str], module: str) -> np.ndarray:
        return np.asarray(
            [[self.tokenize(p, module) for p in phonemes]], dtype=np.int64
        )

    @staticmethod
    def _sec_to_frames(seconds: float, frame_ms: float) -> int:
        return max(1, int(round(seconds * 1000.0 / frame_ms)))

    @staticmethod
    def _midi_to_hz(midi: int) -> float:
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    def _build_timeline(
        self, parsed: ParsedMidi
    ) -> tuple[list[PhoneToken], list[NoteSegment], list[int], list[int]]:
        """构建与 MIDI 时间轴对齐的音素序列与音符序列（含间隙休止）。

        额外返回：
        - word_div：与 phone_tokens 对应的分组大小列表（每个静音间隙或每个
          音符的音素组各算一"词"），供时长预测模型按词级目标时长重新分配
          音素内部的时长比例。
        - word_note_span：与 word_div 一一对应的 note_segments 分组大小
          列表。转音（melisma）延音符不产生新的音素/word_div 分组，却会
          各自追加一个 note_segments 条目（用于携带各自的音高），因此
          word_div 与 note_segments 并非一一对应——每个 word_div 分组实际
          对应 word_note_span 个连续的 note_segments 条目，下游需要按此
          累加定位，不能假定两者索引一致。
        """
        default_lang = self.settings.language
        phone_tokens: list[PhoneToken] = []
        note_segments: list[NoteSegment] = []
        word_div: list[int] = []
        word_note_span: list[int] = []

        notes = sorted(parsed.notes, key=lambda n: (n.start_tick, n.end_tick))
        if not notes:
            dur = max(parsed.duration_sec, 0.2)
            return (
                [PhoneToken(SP, dur, 60, default_lang)],
                [NoteSegment(60.0, dur, True)],
                [1],
                [1],
            )

        prev_end = 0.0
        last_pitch = float(notes[0].pitch)

        for note in notes:
            lang = default_lang
            if note.lyric and not note.lyric.is_sustain:
                lang = _detect_language(note.lyric.text, default_lang)

            gap = note.start_sec - prev_end
            dur = max(note.end_sec - note.start_sec, 0.05)

            if note.is_slur and phone_tokens:
                # 延音符：前面的间隙（若有）也算作延续演唱，不能转成静音，
                # 否则该间隙会被误并入刚插入的 SP token，导致这段本该发声
                # 的时间变成静音（延音符自身的时长也随之丢失）。这段延长量
                # 计入所属词的目标总时长，具体音素内部如何分配交给时长
                # 预测模型（真正的转音通常绝大部分时间落在元音上）。
                #
                # 转音（melisma）本身就是同一元音在不同音符间移动音高。
                # 这里必须追加一个新的 PhoneToken（而不是拉长上一个音素的
                # duration），否则这个被拉长的音素从头到尾都只带着延音*前*
                # 那个音符的音高——喂给 pitch 模型的逐帧音高基准和 note_midi
                # 的音符级音高对不上，会让这段转音的音高/时长听起来含糊、
                # 过渡不干脆。追加同名音素（不重新起音，仅延续），并把它计入
                # 所属词的音素数，保持 word_div 与 phone_tokens 长度一致。
                extend = dur + max(gap, 0.0)
                phone_tokens.append(
                    PhoneToken(
                        phone_tokens[-1].phoneme,
                        extend,
                        note.pitch,
                        phone_tokens[-1].lang,
                        True,
                    )
                )
                if word_div:
                    word_div[-1] += 1
                note_segments.append(NoteSegment(float(note.pitch), extend, False))
                if word_note_span:
                    word_note_span[-1] += 1
                last_pitch = float(note.pitch)
                prev_end = note.end_sec
                continue

            if gap > 0.001:
                phone_tokens.append(
                    PhoneToken(SP, gap, int(round(last_pitch)), default_lang)
                )
                note_segments.append(NoteSegment(last_pitch, gap, True))
                word_div.append(1)
                word_note_span.append(1)

            last_pitch = float(note.pitch)

            raw_phones = phonemes_for_note(note, default_lang, self.root)
            if not raw_phones or raw_phones == [SP]:
                phone_tokens.append(
                    PhoneToken(SP, dur, note.pitch, lang, note.is_slur)
                )
                word_div.append(1)
            else:
                mapped = [self.to_voicebank_phoneme(p, lang) for p in raw_phones]
                per = dur / len(mapped)
                for ph in mapped:
                    phone_tokens.append(
                        PhoneToken(ph, per, note.pitch, lang, note.is_slur)
                    )
                word_div.append(len(mapped))

            note_segments.append(NoteSegment(last_pitch, dur, False))
            word_note_span.append(1)
            prev_end = note.end_sec

        tail = parsed.duration_sec - prev_end
        if tail > 0.001:
            phone_tokens.append(
                PhoneToken(SP, tail, int(round(last_pitch)), default_lang)
            )
            note_segments.append(NoteSegment(last_pitch, tail, True))
            word_div.append(1)
            word_note_span.append(1)

        return phone_tokens, note_segments, word_div, word_note_span

    def _build_frame_plan(
        self,
        phone_tokens: list[PhoneToken],
        note_segments: list[NoteSegment],
        word_div: list[int],
        word_note_span: list[int],
    ) -> tuple[
        list[int],
        list[str],
        list[str],
        list[float],
        list[int],
        list[int],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        frame_ms = self.frame_ms

        phone_secs = [t.duration_sec for t in phone_tokens]
        ph_core_frames = self._sec_list_to_frames(phone_secs, frame_ms)
        ph_dur = [HEAD_FRAMES] + ph_core_frames + [TAIL_FRAMES]
        total_frames = sum(ph_dur)
        full_word_div = [1] + word_div + [1]
        full_word_note_span = [1] + word_note_span + [1]

        note_secs = [s.duration_sec for s in note_segments]
        note_core_frames = self._sec_list_to_frames(note_secs, frame_ms)
        note_dur = self._match_frame_total(
            [HEAD_FRAMES] + note_core_frames + [TAIL_FRAMES],
            total_frames,
        )

        phonemes = [SP] + [t.phoneme for t in phone_tokens] + [SP]
        langs = [self.settings.language] + [t.lang for t in phone_tokens] + [
            self.settings.language
        ]

        head_midi = float(phone_tokens[0].midi) if phone_tokens else 60.0
        tail_midi = float(phone_tokens[-1].midi) if phone_tokens else 60.0
        note_midi = (
            [head_midi]
            + [s.midi for s in note_segments]
            + [tail_midi]
        )
        note_rest = [True] + [s.is_rest for s in note_segments] + [True]

        padded_midis = [head_midi] + [float(t.midi) for t in phone_tokens] + [
            tail_midi
        ]

        return (
            ph_dur,
            phonemes,
            langs,
            padded_midis,
            full_word_div,
            full_word_note_span,
            np.asarray(note_midi, dtype=np.float32),
            np.asarray(note_dur, dtype=np.int64),
            np.asarray(note_rest, dtype=bool),
        )

    @staticmethod
    def _expand_pitch_midi(ph_dur: list[int], padded_midis: list[float]) -> np.ndarray:
        total_frames = sum(ph_dur)
        pitch_midi = np.zeros(total_frames, dtype=np.float32)
        idx = 0
        for dur, midi in zip(ph_dur, padded_midis):
            pitch_midi[idx : idx + dur] = midi
            idx += dur
        return pitch_midi

    def _silence_long_rests(
        self,
        waveform: np.ndarray,
        phonemes: list[str],
        durations: list[int],
    ) -> np.ndarray:
        """把连续时长超过阈值的 SP（静音/间奏）段落在波形上强制静音。"""
        frame_ms = self.frame_ms
        threshold_frames = int(round(_LONG_REST_SILENCE_THRESHOLD_SEC * 1000.0 / frame_ms))
        fade_samples = max(1, int(_SILENCE_FADE_SEC * self.sample_rate))

        runs: list[tuple[int, int]] = []
        frame_idx = 0
        run_start = None
        run_len = 0
        for ph, dur in zip(phonemes, durations):
            if ph == SP:
                if run_start is None:
                    run_start = frame_idx
                run_len += dur
            else:
                if run_start is not None and run_len >= threshold_frames:
                    runs.append((run_start, run_len))
                run_start = None
                run_len = 0
            frame_idx += dur
        if run_start is not None and run_len >= threshold_frames:
            runs.append((run_start, run_len))

        if not runs:
            return waveform

        waveform = waveform.copy()
        for start_frame, length_frames in runs:
            s0 = max(0, start_frame * self.hop_size)
            s1 = min(len(waveform), (start_frame + length_frames) * self.hop_size)
            if s1 <= s0:
                continue
            seg_len = s1 - s0
            fade = min(fade_samples, seg_len // 2)
            window = np.zeros(seg_len, dtype=np.float32)
            if fade > 0:
                window[:fade] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
                window[-fade:] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            waveform[s0:s1] *= window
        return waveform

    def _predict_phone_durations(
        self,
        phonemes: list[str],
        langs: list[str],
        naive_ph_dur: list[int],
        word_div: list[int],
        padded_midis: list[float],
        log: Callable[[str], None],
    ) -> list[int]:
        """用声库自带的 dsdur 时长预测模型，在每个"词"（音符音素组/静音
        间隙）固定的目标总帧数内，重新分配组内各音素的时长比例，替代
        简单的平均切分。平均切分会让辅音占用与元音同样长的时间，导致
        辅音被拖长、元音起唱延迟，听感含糊且与节拍对不齐。

        若声库未提供 dsdur 模型或推理异常，回退到平均切分（naive_ph_dur）。
        """
        cfg_path = os.path.join(self.root, "dsdur", "dsconfig.yaml")
        if not os.path.isfile(cfg_path):
            return naive_ph_dur
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                dur_cfg = yaml.safe_load(f)

            word_dur: list[int] = []
            idx = 0
            for w in word_div:
                word_dur.append(sum(naive_ph_dur[idx : idx + w]))
                idx += w

            tokens = self._token_ids(phonemes, "dsdur")
            langs_arr = self._lang_tensor(langs, "dsdur")
            word_div_arr = np.asarray([word_div], dtype=np.int64)
            word_dur_arr = np.asarray([word_dur], dtype=np.int64)

            ling = self._session(os.path.join("dsdur", dur_cfg["linguistic"]))
            encoder_out, x_masks = ling.run(
                None,
                self._filter_session_inputs(
                    ling,
                    {
                        "tokens": tokens,
                        "languages": langs_arr,
                        "word_div": word_div_arr,
                        "word_dur": word_dur_arr,
                    },
                ),
            )

            ph_midi = np.asarray([[int(round(m)) for m in padded_midis]], dtype=np.int64)
            dur_model = self._session(os.path.join("dsdur", dur_cfg["dur"]))
            pred = dur_model.run(
                None,
                self._filter_session_inputs(
                    dur_model,
                    {
                        "encoder_out": encoder_out,
                        "x_masks": x_masks,
                        "ph_midi": ph_midi,
                        "spk_embed": self._spk_by_phone(len(phonemes)),
                    },
                ),
            )[0][0]
            pred = np.maximum(pred, 0.0)

            result: list[int] = []
            idx = 0
            for w, target in zip(word_div, word_dur):
                if w <= 0:
                    continue
                seg = pred[idx : idx + w]
                seg_sum = float(seg.sum())
                if seg_sum <= 1e-6:
                    alloc = self._match_frame_total(
                        [max(1, target // w)] * w, target
                    )
                else:
                    alpha = target / seg_sum
                    scaled = np.round(seg * alpha).astype(np.int64)
                    diff = target - int(scaled.sum())
                    if diff != 0:
                        scaled[-1] += diff
                    # 单个音素时长不能压到 0（会造成该音素完全丢音）
                    for i in range(len(scaled)):
                        if scaled[i] <= 0:
                            donor = int(np.argmax(scaled))
                            if scaled[donor] > 1:
                                scaled[donor] -= 1
                                scaled[i] = 1
                    alloc = scaled.tolist()
                result.extend(alloc)
                idx += w

            if len(result) != len(naive_ph_dur) or sum(result) != sum(naive_ph_dur):
                log("时长模型输出异常，回退到平均切分")
                return naive_ph_dur
            return result
        except Exception as exc:
            log(f"时长模型推理失败（{exc}），回退到平均切分")
            return naive_ph_dur

    def _cap_consonant_durations(
        self,
        durations: list[int],
        phonemes: list[str],
        word_div: list[int],
    ) -> list[int]:
        """辅音时长设上限，超出部分退还给本词内最近的元音（优先其后的
        元音，其次前面的元音）。长音符（如乐句结尾的延长音）里，时长
        预测模型有时会给辅音分配远超自然发声范围的时长，这里做兜底
        修正，不改变整词总时长。"""
        frame_ms = self.frame_ms
        cap_frames = max(1, int(round(_MAX_CONSONANT_DURATION_SEC * 1000.0 / frame_ms)))
        durations = list(durations)

        idx = 0
        for w in word_div:
            if w <= 1:
                idx += w
                continue
            group = list(range(idx, idx + w))
            for pos, gi in enumerate(group):
                ph = phonemes[gi]
                if ph in (AP, SP) or _is_vowel_phone(ph):
                    continue
                if durations[gi] <= cap_frames:
                    continue
                excess = durations[gi] - cap_frames

                target = None
                for gj in group[pos + 1 :]:
                    if _is_vowel_phone(phonemes[gj]):
                        target = gj
                        break
                if target is None:
                    for gj in reversed(group[:pos]):
                        if _is_vowel_phone(phonemes[gj]):
                            target = gj
                            break
                if target is None:
                    continue

                durations[gi] = cap_frames
                durations[target] += excess
            idx += w

        return durations

    def _apply_consonant_anticipation(
        self,
        durations: list[int],
        phonemes: list[str],
        word_div: list[int],
        word_note_span: list[int],
    ) -> tuple[list[int], list[int]]:
        """辅音提前：音节开头的辅音向前"借用"上一个音素的时间，让元音
        准确落在音符起点上（专业歌声合成器的标准做法）。否则辅音天然
        需要的发声准备时间会把可听见的元音起点往后拖，且每个音节都会
        自带一小段这样的延迟，逐字累积后整句人声会明显慢于伴奏。

        对每个音节都做提前（不仅仅是"句首"）——因此借用来源分两种：
        - 上一音素是 SP（乐句/换气间的静音间隙）：可借走高达 85%
          的时长，因为间隙本就无声，损失听不见。
        - 上一音素是有声内容（前一音节的元音收尾、浊辅音等）：只借
          走其尾部 30% 且不超过约 80 ms —— 元音尾一小截被截掉几乎
          察觉不到，却能让本音节的元音准确落在下一个音符起点上。

        借来的时长在本音节最后一个音素上补回去（延长其收尾），因此
        本词总时长和后续所有内容的绝对时间完全不受影响 —— 只是把
        辅音在时间轴上的位置向左挪，不产生累积漂移。

        word_div 与 note_segments/note_dur 并非一一对应（转音延音符
        只追加 note_segments，不追加 word_div，见 _build_timeline），
        因此这里用 word_note_span 做累加，换算出每个 word_div 分组
        对应的 note_dur 起止下标，不能直接假定两者索引一致。上一个
        word 若跨多个音符（比如带转音的音节），要从其"最后一个音符"
        扣除时长；若从其"第一个音符"扣除会把这段偷来的时间标错音符，
        与 pitch/note_midi 轨对不齐。

        返回 (调整后的 durations, 长度与 note_dur 一致的增减量列表)，
        后者用于同步调整 note 级的 note_dur，保持音素级/音符级时间
        轴帧数一致（增减量之和恒为 0）。
        """
        durations = list(durations)
        total_note_segments = sum(word_note_span)
        note_dur_delta = [0] * total_note_segments

        idx = 0
        group_starts: list[int] = []
        for w in word_div:
            group_starts.append(idx)
            idx += w

        note_idx = 0
        note_starts: list[int] = []
        for span in word_note_span:
            note_starts.append(note_idx)
            note_idx += span

        voiced_borrow_cap_frames = max(
            1, int(round(_ANTICIPATION_VOICED_BORROW_MAX_SEC * 1000.0 / self.frame_ms))
        )

        for gi, w in enumerate(word_div):
            if gi == 0 or w <= 0:
                continue
            start = group_starts[gi]
            group_phonemes = phonemes[start : start + w]
            is_gap_word = w == 1 and group_phonemes[0] == SP
            if is_gap_word:
                continue

            lead = 0
            for p in group_phonemes[: min(w - 1, _ANTICIPATION_MAX_LEAD_PHONES)]:
                if _is_vowel_phone(p):
                    break
                lead += 1
            if lead <= 0:
                continue

            prev_w = word_div[gi - 1]
            prev_start = group_starts[gi - 1]
            prev_last_idx = prev_start + prev_w - 1
            prev_last_ph = phonemes[prev_last_idx]

            lead_frames = sum(durations[start : start + lead])
            available = durations[prev_last_idx]

            if prev_last_ph == SP:
                max_borrow = int(available * _ANTICIPATION_GAP_BORROW_RATIO)
            else:
                max_borrow = min(
                    int(available * _ANTICIPATION_VOICED_BORROW_RATIO),
                    voiced_borrow_cap_frames,
                )
            borrow = min(lead_frames, max_borrow)
            if borrow <= 0:
                continue

            durations[prev_last_idx] -= borrow
            durations[start + w - 1] += borrow

            prev_last_note_idx = note_starts[gi - 1] + word_note_span[gi - 1] - 1
            curr_last_note_idx = note_starts[gi] + word_note_span[gi] - 1
            note_dur_delta[prev_last_note_idx] -= borrow
            note_dur_delta[curr_last_note_idx] += borrow

        return durations, note_dur_delta

    def _spk_by_frame(self, total_frames: int) -> np.ndarray | None:
        if self._speaker_embed is None:
            return None
        embed = self._speaker_embed.astype(np.float32)
        return np.tile(embed, (total_frames, 1))[None, :, :]

    def _spk_by_phone(self, count: int) -> np.ndarray | None:
        if self._speaker_embed is None:
            return None
        embed = self._speaker_embed.astype(np.float32)
        return np.tile(embed, (count, 1))[None, :, :]

    @staticmethod
    def _filter_session_inputs(
        session: ort.InferenceSession, candidates: dict[str, np.ndarray | None]
    ) -> dict[str, np.ndarray]:
        """并非所有声库的 ONNX 模型都接受同一套输入（比如是否需要
        languages/spk_embed 因声库而异）。这里只保留该 session 实际声明
        的输入名，多余的（如 None 值，或模型压根没有这个输入）一律丢弃，
        而不是硬编码假定某个声库的固定输入集合。"""
        names = {i.name for i in session.get_inputs()}
        return {k: v for k, v in candidates.items() if k in names and v is not None}

    def synthesize(
        self,
        parsed: ParsedMidi,
        output_wav: str,
        log: Callable[[str], None] | None = None,
    ) -> str:
        log = log or (lambda _m: None)
        errors = self.validate()
        if errors:
            raise RuntimeError("\n".join(errors))

        phone_tokens, note_segments, word_div, word_note_span = self._build_timeline(parsed)
        (
            naive_ph_dur,
            phonemes,
            langs,
            padded_midis,
            full_word_div,
            full_word_note_span,
            note_midi,
            note_dur,
            note_rest,
        ) = self._build_frame_plan(phone_tokens, note_segments, word_div, word_note_span)

        log("运行时长预测模型...")
        durations = self._predict_phone_durations(
            phonemes, langs, naive_ph_dur, full_word_div, padded_midis, log
        )
        durations = self._cap_consonant_durations(durations, phonemes, full_word_div)

        durations, note_dur_delta = self._apply_consonant_anticipation(
            durations, phonemes, full_word_div, full_word_note_span,
        )
        if any(note_dur_delta):
            note_dur = np.asarray(
                [max(0, d + delta) for d, delta in zip(note_dur.tolist(), note_dur_delta)],
                dtype=np.int64,
            )

        pitch_midi = self._expand_pitch_midi(durations, padded_midis)
        total_frames = sum(durations)
        synth_sec = total_frames * self.frame_ms / 1000.0
        log(
            f"音素数: {len(phonemes)}, 总帧数: {total_frames} "
            f"(约 {synth_sec:.2f}s, MIDI {parsed.duration_sec:.2f}s)"
        )

        pitch_token_ids = self._token_ids(phonemes, "dspitch")
        pitch_langs = self._lang_tensor(langs, "dspitch")
        ph_dur = np.asarray([durations], dtype=np.int64)

        # --- pitch ---
        log("运行 pitch 模型...")
        pitch_dir = os.path.join(self.root, "dspitch")
        with open(os.path.join(pitch_dir, "dsconfig.yaml"), "r", encoding="utf-8") as f:
            pitch_cfg = yaml.safe_load(f)

        ling = self._session(os.path.join("dspitch", pitch_cfg["linguistic"]))
        ling_out = ling.run(
            None,
            self._filter_session_inputs(
                ling,
                {
                    "tokens": pitch_token_ids,
                    "languages": pitch_langs,
                    "ph_dur": ph_dur,
                },
            ),
        )
        encoder_out = ling_out[0]

        pitch_model = self._session(os.path.join("dspitch", pitch_cfg["pitch"]))
        pitch_inputs = self._filter_session_inputs(
            pitch_model,
            {
                "encoder_out": encoder_out,
                "ph_dur": ph_dur,
                "note_midi": note_midi[None, :],
                "note_dur": note_dur[None, :],
                "note_rest": note_rest[None, :],
                "pitch": pitch_midi[None, :].astype(np.float32),
                "retake": np.ones((1, total_frames), dtype=bool),
                "expr": np.ones((1, total_frames), dtype=np.float32),
                "spk_embed": self._spk_by_frame(total_frames),
                "steps": np.asarray(self.settings.pitch_steps, dtype=np.int64),
            },
        )
        pitch_pred_midi = pitch_model.run(None, pitch_inputs)[0][0].astype(np.float32)
        f0_hz = self._midi_to_hz_array(pitch_pred_midi)

        # --- variance ---
        log("运行 variance 模型...")
        var_dir = os.path.join(self.root, "dsvariance")
        with open(os.path.join(var_dir, "dsconfig.yaml"), "r", encoding="utf-8") as f:
            var_cfg = yaml.safe_load(f)

        var_token_ids = self._token_ids(phonemes, "dsvariance")
        var_langs = self._lang_tensor(langs, "dsvariance")
        var_ling = self._session(os.path.join("dsvariance", var_cfg["linguistic"]))
        var_enc = var_ling.run(
            None,
            self._filter_session_inputs(
                var_ling,
                {
                    "tokens": var_token_ids,
                    "languages": var_langs,
                    "ph_dur": ph_dur,
                },
            ),
        )[0]

        var_model = self._session(os.path.join("dsvariance", var_cfg["variance"]))
        var_input_names = {i.name for i in var_model.get_inputs()}
        # 不同声库预测的方差通道数量不同（Nishiren 只有 breathiness/voicing/
        # tension 三路，LIEE 多了 energy 共四路），retake 的通道数必须与该
        # 模型实际声明的方差输入个数一致，不能写死 3。
        variance_channel_names = [
            n for n in ("energy", "breathiness", "voicing", "tension") if n in var_input_names
        ]
        var_inputs = self._filter_session_inputs(
            var_model,
            {
                "encoder_out": var_enc,
                "ph_dur": ph_dur,
                "pitch": pitch_pred_midi[None, :].astype(np.float32),
                "energy": np.zeros((1, total_frames), dtype=np.float32),
                "breathiness": np.zeros((1, total_frames), dtype=np.float32),
                "voicing": np.zeros((1, total_frames), dtype=np.float32),
                "tension": np.zeros((1, total_frames), dtype=np.float32),
                "retake": np.ones((1, total_frames, max(1, len(variance_channel_names))), dtype=bool),
                "spk_embed": self._spk_by_frame(total_frames),
                "steps": np.asarray(self.settings.variance_steps, dtype=np.int64),
            },
        )
        var_out = var_model.run(None, var_inputs)
        var_out_by_name = {o.name: arr[0] for o, arr in zip(var_model.get_outputs(), var_out)}
        zeros_frame = np.zeros(total_frames, dtype=np.float32)
        breathiness = var_out_by_name.get("breathiness_pred", zeros_frame)
        voicing = var_out_by_name.get("voicing_pred", zeros_frame)
        tension = var_out_by_name.get("tension_pred", zeros_frame)
        energy = var_out_by_name.get("energy_pred", zeros_frame)

        # --- acoustic ---
        log("运行 acoustic 模型...")
        ac_token_ids = self._token_ids(phonemes, "dsmain")
        ac_langs = self._lang_tensor(langs, "dsmain")
        acoustic = self._session(self._ds_main["acoustic"])
        ac_inputs = self._filter_session_inputs(
            acoustic,
            {
                "tokens": ac_token_ids,
                "languages": ac_langs,
                "durations": ph_dur,
                "f0": f0_hz[None, :],
                "energy": energy[None, :],
                "breathiness": breathiness[None, :],
                "voicing": voicing[None, :],
                "tension": tension[None, :],
                "gender": np.zeros((1, total_frames), dtype=np.float32),
                "velocity": np.full((1, total_frames), self.settings.velocity, dtype=np.float32),
                "spk_embed": self._spk_by_frame(total_frames),
                "steps": np.asarray(self.settings.acoustic_steps, dtype=np.int64),
            },
        )
        mel = acoustic.run(None, ac_inputs)[0]

        # --- vocoder ---
        log("运行 vocoder...")
        vocoder = self._session(self._vocoder_rel_path())
        vocoder_inputs = self._filter_session_inputs(
            vocoder,
            {"mel": mel, "f0": f0_hz[None, :].astype(np.float32)},
        )
        waveform = vocoder.run(None, vocoder_inputs)[0][0]

        waveform = self._silence_long_rests(waveform, phonemes, durations)

        content_start = HEAD_FRAMES * self.hop_size
        content_end = len(waveform) - TAIL_FRAMES * self.hop_size
        if content_end > content_start:
            waveform = waveform[content_start:content_end]

        import soundfile as sf

        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        sf.write(output_wav, waveform, self.sample_rate, subtype="PCM_16")
        log(f"已写入: {output_wav}")
        return output_wav
