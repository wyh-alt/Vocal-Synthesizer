"""Nishiren / OpenUTAU 韩文 G2P（dsdict-ko.yaml 查表）。"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import yaml

_KO_PHONEME_RE = re.compile(r"^ko/(.+)$")


def _dict_path(voicebank_path: str) -> str | None:
    root = os.path.abspath(voicebank_path)
    candidates = [
        os.path.join(root, "dsdur", "dsdict-ko.yaml"),
        os.path.join(root, "dsmain", "dsdict-ko.yaml"),
        os.path.join(root, "dsdict-ko.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _strip_ko_phoneme(token: str) -> str:
    match = _KO_PHONEME_RE.match(token)
    if match:
        return match.group(1)
    return token


@lru_cache(maxsize=4)
def load_korean_dictionary(voicebank_path: str) -> dict[str, tuple[str, ...]]:
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
        result[grapheme] = tuple(_strip_ko_phoneme(str(p)) for p in phonemes)
    return result


def korean_phonemes_from_dict(text: str, voicebank_path: str) -> list[str]:
    if not text or not voicebank_path:
        return []

    dictionary = load_korean_dictionary(voicebank_path)
    if not dictionary:
        return []

    phones: list[str] = []
    for char in text:
        if char in dictionary:
            phones.extend(dictionary[char])
    return phones
