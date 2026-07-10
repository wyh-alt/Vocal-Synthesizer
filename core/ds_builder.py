"""将解析后的 MIDI 转为 DiffSinger .ds 片段。"""

from __future__ import annotations

import json
from typing import Any

from core.lyric_phoneme import AP, SP, build_phoneme_entries, entries_to_text
from core.midi_parser import ParsedMidi

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_pitch_to_name(pitch: int) -> str:
    if pitch <= 0:
        return "rest"
    octave = pitch // 12 - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def _group_entries_by_note(entries: list[dict]) -> list[dict]:
    groups: list[dict] = []
    if not entries:
        return groups

    current: dict | None = None
    for entry in entries:
        key = (entry["pitch"], entry["note_dur"], entry.get("word", ""), entry["is_slur"])
        if current is None or (
            current["pitch"] != entry["pitch"]
            or abs(current["note_dur"] - entry["note_dur"]) > 1e-6
            or current.get("word") != entry.get("word")
        ):
            current = {
                "phones": [entry["ph"]],
                "pitch": entry["pitch"],
                "note_dur": entry["note_dur"],
                "is_slur": entry["is_slur"],
                "word": entry.get("word", ""),
            }
            groups.append(current)
        else:
            current["phones"].append(entry["ph"])
            if entry["is_slur"]:
                current["is_slur"] = 1
    return groups


def build_ds_segment(
    parsed: ParsedMidi,
    default_lang: str = "en",
    voicebank_path: str = "",
) -> dict[str, Any]:
    entries = build_phoneme_entries(parsed, default_lang, voicebank_path)
    if not entries:
        raise ValueError("MIDI 中没有可合成的音符")

    groups = _group_entries_by_note(entries)
    ph_seq: list[str] = [AP]
    note_seq: list[str] = ["rest"]
    note_dur: list[float] = [0.05]
    note_slur: list[int] = [0]
    ph_num: list[int] = [1]

    word_boundaries: list[int] = []
    current_word = ""
    word_phone_count = 0

    for group in groups:
        pitch_name = midi_pitch_to_name(group["pitch"])
        dur = round(float(group["note_dur"]), 6)
        phones = group["phones"] or [SP]
        word = group.get("word") or ""

        if word and word != current_word:
            if current_word and word_phone_count > 0:
                ph_num.append(word_phone_count)
                word_boundaries.append(len(ph_seq))
            current_word = word
            word_phone_count = 0

        for ph in phones:
            ph_seq.append(ph)
            word_phone_count += 1

        note_seq.append(pitch_name)
        note_dur.append(dur)
        note_slur.append(int(group.get("is_slur", 0)))

    if word_phone_count > 0:
        ph_num.append(word_phone_count)
    ph_num.append(1)
    ph_seq.append(SP)

    text = entries_to_text(entries)

    return {
        "offset": 0.0,
        "text": text.strip() or "guide vocal",
        "ph_seq": " ".join(ph_seq),
        "note_seq": " ".join(note_seq),
        "note_dur": " ".join(str(v) for v in note_dur),
        "note_slur": " ".join(str(v) for v in note_slur),
        "ph_num": " ".join(str(v) for v in ph_num),
        "lang": default_lang,
    }


def write_ds_file(path: str, segment: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([segment], f, ensure_ascii=False, indent=2)


def format_ds_preview(segment: dict[str, Any]) -> str:
    lines = [
        f"text: {segment.get('text', '')}",
        f"ph_seq ({len(segment.get('ph_seq', '').split())}): {segment.get('ph_seq', '')[:120]}...",
        f"note_seq ({len(segment.get('note_seq', '').split())}): {segment.get('note_seq', '')[:120]}...",
        f"note_dur: {segment.get('note_dur', '')[:120]}...",
        f"note_slur: {segment.get('note_slur', '')}",
    ]
    return "\n".join(lines)
