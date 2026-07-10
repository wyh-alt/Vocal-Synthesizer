"""Nishiren / OpenUTAU 中文 G2P（dsdict-zh.yaml 拼音查表）。"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import yaml

_ZH_PHONEME_RE = re.compile(r"^zh/(.+)$")


def _dict_path(voicebank_path: str) -> str | None:
    root = os.path.abspath(voicebank_path)
    candidates = [
        os.path.join(root, "dsdur", "dsdict-zh.yaml"),
        os.path.join(root, "dsmain", "dsdict-zh.yaml"),
        os.path.join(root, "dsdict-zh.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _strip_zh_phoneme(token: str) -> str:
    match = _ZH_PHONEME_RE.match(token)
    if match:
        return match.group(1)
    return token


@lru_cache(maxsize=4)
def load_chinese_dictionary(voicebank_path: str) -> dict[str, tuple[str, ...]]:
    path = _dict_path(voicebank_path)
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entries = data.get("entries") or []
    result: dict[str, tuple[str, ...]] = {}
    for item in entries:
        grapheme = item.get("grapheme")
        phonemes = item.get("phonemes") or []
        if not grapheme or not phonemes:
            continue
        result[grapheme] = tuple(_strip_zh_phoneme(str(p)) for p in phonemes)
    return result


@lru_cache(maxsize=4096)
def _pinyin_syllable(char: str) -> str:
    from pypinyin import Style, pinyin as pinyin_fn

    items = pinyin_fn(char, style=Style.TONE3, strict=False, errors="ignore")
    if not items or not items[0] or not items[0][0]:
        return ""
    return re.sub(r"\d", "", items[0][0].lower())


def chinese_phonemes_from_dict(text: str, voicebank_path: str) -> list[str]:
    """按标准汉语拼音音节（不含声调）查表得到声母+韵母音素序列。

    比按拼音字母逐字拆分（如 "hao" -> h,a,o）更准确——声库的中文音素表
    使用声母/韵母整体单元（如 "zh/zh"、"zh/ir"、"zh/ang" 等），而非单个
    罗马字母，逐字母拆分会产生声库根本不认识的音素。
    """
    if not text or not voicebank_path:
        return []

    dictionary = load_chinese_dictionary(voicebank_path)
    if not dictionary:
        return []

    phones: list[str] = []
    for char in text:
        if not char.strip():
            continue
        syllable = _pinyin_syllable(char)
        if syllable and syllable in dictionary:
            phones.extend(dictionary[syllable])
    return phones
