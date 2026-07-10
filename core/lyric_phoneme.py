"""歌词转 DiffSinger 音素序列。"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from core.chinese_g2p import chinese_phonemes_from_dict
from core.korean_g2p import korean_phonemes_from_dict
from core.midi_parser import (
    LyricToken,
    NoteEvent,
    ParsedMidi,
    _CJK_CHAR_RE,
    _LATIN_WORD_RE,
)

_G2P = None
_PYPINYIN = None

AP = "AP"
SP = "SP"

# Hangul jamo（与 Nishiren dsdict-ko 终声标记一致：K/N/T/L/M/P/NG）
_KO_CHO = [
    "g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s", "ss", None,
    "j", "jj", "ch", "k", "t", "p", "h",
]
_KO_JUNG: tuple[tuple[str, ...], ...] = (
    ("a",),
    ("e",),
    ("y", "a"),
    ("y", "e"),
    ("eo",),
    ("e",),
    ("y", "eo"),
    ("y", "e"),
    ("o",),
    ("w", "a"),
    ("w", "e"),
    ("w", "e"),
    ("y", "o"),
    ("u",),
    ("w", "eo"),
    ("w", "e"),
    ("w", "i"),
    ("y", "u"),
    ("eu",),
    ("eu", "i"),
    ("i",),
)
_KO_JONG = [
    None,
    "K", "K", "K", "N", "N", "N", "T", "L",
    "K", "M", "L", "L", "L", "P", "L",
    "M", "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T",
]
_HANGUL_SYLLABLE_RE = re.compile(r"[가-힣]")


def _lazy_g2p():
    global _G2P
    if _G2P is None:
        from g2p_en import G2p

        _G2P = G2p()
    return _G2P


def _lazy_pypinyin():
    global _PYPINYIN
    if _PYPINYIN is None:
        from pypinyin import Style, pinyin

        _PYPINYIN = pinyin
    return _PYPINYIN


def _strip_stress(ph: str) -> str:
    return re.sub(r"\d+$", "", ph)


def _detect_language(text: str, default: str = "en") -> str:
    if _CJK_CHAR_RE.search(text):
        if re.search(r"[가-힣]", text):
            return "ko"
        if re.search(r"[ぁ-んァ-ン]", text):
            return "ja"
        return "zh"
    if _LATIN_WORD_RE.search(text):
        return "en"
    return default


_ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}


def _syllabify_phones(phones: list[str]) -> list[list[str]]:
    """按 ARPAbet 元音音核切分音节（最大起首原则），比按字母猜音节数更贴合
    g2p_en 实际输出的音素序列，避免 #1/#2 分音节时切错发音。"""
    vowel_idx = [i for i, p in enumerate(phones) if p in _ARPABET_VOWELS]
    if len(vowel_idx) <= 1:
        return [list(phones)]

    syllables: list[list[str]] = []
    start = 0
    for k in range(len(vowel_idx) - 1):
        v_i = vowel_idx[k]
        v_next = vowel_idx[k + 1]
        cluster_len = v_next - v_i - 1
        if cluster_len <= 1:
            split_at = v_i + 1
        else:
            split_at = v_i + 1 + cluster_len // 2
        syllables.append(phones[start:split_at])
        start = split_at
    syllables.append(phones[start:])
    return syllables


@lru_cache(maxsize=4096)
def english_word_phonemes(word: str) -> tuple[str, ...]:
    g2p = _lazy_g2p()
    phones = [_strip_stress(p) for p in g2p(word) if p not in {" ", ""}]
    return tuple(p for p in phones if p.isalpha())


@lru_cache(maxsize=4096)
def english_syllable_phonemes(word: str) -> tuple[tuple[str, ...], ...]:
    all_phones = list(english_word_phonemes(word))
    if not all_phones:
        return tuple()
    return tuple(tuple(s) for s in _syllabify_phones(all_phones))


def _english_note_phonemes(token: LyricToken, text: str) -> list[str]:
    """英文音符：若未标 #1/#2，则返回整词音素；标了则返回指定音节。"""
    word = token.syllable_base or text
    syl_groups = english_syllable_phonemes(word.lower())
    if not syl_groups:
        return ["SP"]

    if token.syllable_index and token.syllable_index > 0:
        idx = min(token.syllable_index - 1, len(syl_groups) - 1)
        return list(syl_groups[idx]) or ["SP"]

    full = english_word_phonemes(word.lower())
    if full:
        return list(full)
    return list(syl_groups[0]) or ["SP"]


def resolve_melisma_syllables(notes: list[NoteEvent]) -> None:
    """按 ACE Studio 转音约定重新解释延音符（"-"）。

    一个歌词若对应多个音符，第一个音符演唱歌词内容，其余延音符默认视为
    转音（melisma，同一元音移动音高）。但若歌词是未显式标注 #1/#2 的
    多音节英文单词，应先用延音符的音符位置演唱词内剩余音节；只有音节
    全部唱完后，多出的延音符才是真正的转音。

    该函数原地修改 note.lyric / note.is_slur：把用于承载剩余音节的延音符
    改写为对应音节的真实歌词 token（is_sustain=False、is_slur=False），
    真正的转音延音符保持原样，交由下游按 slur 延续处理。
    """
    total_syllables = 0
    word_base = ""
    next_index = 0

    for note in notes:
        token = note.lyric
        if token is None:
            total_syllables = 0
            continue

        if not token.is_sustain:
            total_syllables = 0
            word_base = ""
            text = (token.syllable_base or token.text).strip()
            if token.syllable_index == 0 and text and _LATIN_WORD_RE.fullmatch(text):
                groups = english_syllable_phonemes(text.lower())
                if len(groups) > 1:
                    total_syllables = len(groups)
                    word_base = text
                    next_index = 2
            continue

        if word_base and next_index <= total_syllables:
            note.lyric = LyricToken(
                raw=token.raw,
                text=word_base,
                is_sustain=False,
                syllable_index=next_index,
                syllable_base=word_base,
            )
            note.is_slur = False
            next_index += 1
        # 否则维持原始延音符语义（真正的转音）


def hangul_syllable_phonemes(char: str) -> tuple[str, ...]:
    if len(char) != 1 or not ("가" <= char <= "힣"):
        return ()
    code = ord(char) - 0xAC00
    jong = code % 28
    jung = (code // 28) % 21
    cho = code // 28 // 21
    phones: list[str] = []
    initial = _KO_CHO[cho]
    if initial:
        phones.append(initial)
    phones.append(_KO_JUNG[jung][0])
    for extra in _KO_JUNG[jung][1:]:
        phones.append(extra)
    final = _KO_JONG[jong]
    if final:
        phones.append(final)
    return tuple(phones)


@lru_cache(maxsize=4096)
def korean_text_phonemes(text: str) -> tuple[str, ...]:
    phones: list[str] = []
    for char in text:
        if "가" <= char <= "힣":
            phones.extend(hangul_syllable_phonemes(char))
    return tuple(phones)


@lru_cache(maxsize=4096)
def chinese_char_phonemes(char: str) -> tuple[str, ...]:
    pinyin_fn = _lazy_pypinyin()
    from pypinyin import Style

    items = pinyin_fn(char, style=Style.TONE3, strict=False, errors="ignore")
    if not items or not items[0] or not items[0][0]:
        return (char,)
    py = items[0][0].lower()
    py = re.sub(r"\d", "", py)
    if len(py) <= 1:
        return (py,)
    return tuple(list(py))


def _note_language(note: NoteEvent, default_lang: str) -> str:
    if note.lyric is None:
        return default_lang
    return _detect_language(note.lyric.text, default_lang)


def phonemes_for_note(
    note: NoteEvent,
    default_lang: str = "en",
    voicebank_path: str = "",
) -> list[str]:
    if note.lyric is None:
        return ["SP"]

    token = note.lyric
    text = token.text.strip()
    if not text or text == "-":
        return ["SP"]

    lang = _detect_language(text, default_lang)

    if lang == "en":
        return _english_note_phonemes(token, text)

    if lang == "zh":
        chars = "".join(c for c in text if c.strip() and _CJK_CHAR_RE.match(c))
        if not chars:
            return ["SP"]
        if voicebank_path:
            phones = chinese_phonemes_from_dict(chars, voicebank_path)
            if phones:
                return phones
        phones = []
        for char in chars:
            phones.extend(chinese_char_phonemes(char))
        return phones or ["SP"]

    if lang == "ko":
        if voicebank_path:
            phones = korean_phonemes_from_dict(text, voicebank_path)
            if phones:
                return phones
        phones = list(korean_text_phonemes(text))
        if phones:
            return phones
        if _LATIN_WORD_RE.search(text):
            return _english_note_phonemes(token, text)
        return ["SP"]

    return ["SP"]


def build_phoneme_entries(
    parsed: ParsedMidi,
    default_lang: str = "en",
    voicebank_path: str = "",
) -> list[dict]:
    """为每个音符生成音素、时长、音高与 slur 信息。"""
    entries: list[dict] = []
    prev_word = ""

    for note in parsed.notes:
        dur = max(note.end_sec - note.start_sec, 0.05)
        phones = phonemes_for_note(note, default_lang, voicebank_path)

        is_slur = note.is_slur
        if note.lyric and note.lyric.is_sustain and prev_word:
            is_slur = True

        word_text = ""
        if note.lyric and not note.lyric.is_sustain:
            word_text = note.lyric.syllable_base or note.lyric.text
            prev_word = word_text
        elif note.lyric and note.lyric.is_sustain:
            word_text = prev_word

        per_phone_dur = dur / max(len(phones), 1)
        for index, ph in enumerate(phones):
            entries.append(
                {
                    "ph": ph,
                    "dur": per_phone_dur,
                    "pitch": note.pitch,
                    "is_slur": 1 if (is_slur and index == 0) else 0,
                    "word": word_text,
                    "note_dur": dur,
                    "is_rest": False,
                }
            )

    return entries


def entries_to_text(entries: Iterable[dict]) -> str:
    words = []
    for item in entries:
        word = item.get("word") or ""
        if word and (not words or words[-1] != word):
            words.append(word)
    return " ".join(words)


def detect_dominant_language(parsed: ParsedMidi, fallback: str = "en") -> str:
    """统计 MIDI 中各音符歌词的语种分布，返回出现次数最多的语种。

    用于"自动"模式下选一个合理的默认/兜底语言——每个音符自身的音素
    仍然按该音符文本自行检测出的语种处理（见 phonemes_for_note），
    混合中/英/韩歌词的歌曲不会被强制统一成一种语言；这里只是给
    通配符 SP 包装帧、无法从文本判断语种的边界情况一个更合理的默认值。
    """
    counts: dict[str, int] = {}
    for note in parsed.notes:
        if note.lyric is None or note.lyric.is_sustain:
            continue
        text = note.lyric.text.strip()
        if not text:
            continue
        lang = _detect_language(text, fallback)
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return fallback
    return max(counts.items(), key=lambda kv: kv[1])[0]
