"""解析 ACE Studio 风格 MIDI：音符 + 歌词。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import mido

SUSTAIN_MARKERS = frozenset({"-", "－", "—", "–", "_", "~", "～", "…", "・・・", "ー", "ㅡ", "."})
_SYLLABLE_INDEX_RE = re.compile(r"^(.*?)[#＃](\d+)\s*$")
_CJK_CHAR_RE = re.compile(r"[가-힣一-龥ぁ-んァ-ン]")
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


@dataclass
class TempoMap:
    ticks_per_beat: int
    tempo_events: list[tuple[int, int]]

    @classmethod
    def from_midi(cls, midi_file: mido.MidiFile) -> "TempoMap":
        tempo_events: list[tuple[int, int]] = []
        for track in midi_file.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "set_tempo":
                    tempo_events.append((abs_tick, msg.tempo))
        return cls(midi_file.ticks_per_beat, tempo_events)

    def tick_to_seconds(self, tick: int) -> float:
        tick = int(tick)
        if not self.tempo_events:
            return mido.tick2second(tick, self.ticks_per_beat, 500000)

        merged: dict[int, int] = {0: 500000}
        for t, tempo in self.tempo_events:
            merged[int(t)] = int(tempo)
        events = sorted(merged.items())

        seconds = 0.0
        prev_tick = 0
        prev_tempo = events[0][1]
        for current_tick, tempo in events[1:]:
            if tick < current_tick:
                delta = tick - prev_tick
                return seconds + (delta * prev_tempo) / (self.ticks_per_beat * 1_000_000.0)
            delta = current_tick - prev_tick
            seconds += (delta * prev_tempo) / (self.ticks_per_beat * 1_000_000.0)
            prev_tick = current_tick
            prev_tempo = tempo

        delta = tick - prev_tick
        return seconds + (delta * prev_tempo) / (self.ticks_per_beat * 1_000_000.0)


@dataclass
class LyricToken:
    raw: str
    text: str
    is_sustain: bool = False
    syllable_index: int = 0
    syllable_base: str = ""


@dataclass
class NoteEvent:
    track: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    lyric: Optional[LyricToken] = None
    is_slur: bool = False


@dataclass
class ParsedMidi:
    path: str
    ticks_per_beat: int
    tempo_map: TempoMap
    notes: list[NoteEvent] = field(default_factory=list)
    duration_sec: float = 0.0
    melody_track: int = 0

    @property
    def lyric_note_count(self) -> int:
        return sum(1 for n in self.notes if n.lyric is not None)


def decode_midi_text(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        try:
            return text.encode("latin1").decode("gbk")
        except UnicodeDecodeError:
            return text


def parse_lyric_token(raw_text: str) -> Optional[LyricToken]:
    """解析单个歌词 token，支持 baby#1 / baby#2 与延音 -。"""
    raw = (raw_text or "").strip()
    if not raw:
        return None

    text = raw
    if text.startswith("[") and text.endswith("]") and text.count("[") == 1:
        text = text[1:-1].strip()
        if not text:
            return None

    if text in SUSTAIN_MARKERS:
        return LyricToken(raw=raw, text="-", is_sustain=True)

    syllable_index = 0
    syllable_base = text
    match = _SYLLABLE_INDEX_RE.match(text)
    if match:
        syllable_base = match.group(1).strip()
        if not syllable_base:
            return None
        try:
            syllable_index = int(match.group(2))
        except ValueError:
            syllable_index = 1
        text = syllable_base

    if text in SUSTAIN_MARKERS:
        return LyricToken(raw=raw, text="-", is_sustain=True)

    return LyricToken(
        raw=raw,
        text=text,
        is_sustain=False,
        syllable_index=syllable_index,
        syllable_base=syllable_base,
    )


def _is_note_off(msg) -> bool:
    return msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)


def _group_by_tick(track: mido.MidiTrack) -> list[tuple[int, list]]:
    """按绝对 tick 分组，同一 tick 内消息保持原始相对顺序。"""
    groups: list[tuple[int, list]] = []
    abs_tick = 0
    current_tick = 0
    current_group: list = []
    for msg in track:
        abs_tick += msg.time
        if msg.time > 0 and current_group:
            groups.append((current_tick, current_group))
            current_group = []
        if not current_group:
            current_tick = abs_tick
        current_group.append(msg)
    if current_group:
        groups.append((current_tick, current_group))
    return groups


def _collect_track_notes(
    track_index: int,
    track: mido.MidiTrack,
    tempo_map: TempoMap,
) -> tuple[list[NoteEvent], list[tuple[int, LyricToken]], int]:
    """收集音符与歌词。

    同一 tick 内若存在"上一个音符的 note_off"与"下一个同音高音符的
    note_on"，两者在原始文件中的先后顺序并不保证 note_off 在前——若
    note_on 先出现，会覆盖仍处于 active 状态的旧音符，导致旧音符丢失、
    新音符因 end_tick==start_tick 被当作零长度丢弃。因此同一 tick 内
    需要先处理所有 note_off，再处理 note_on，才能正确还原相邻同音高
    音符（这种写法在人声 MIDI 里非常常见）。
    """
    notes: list[NoteEvent] = []
    lyrics: list[tuple[int, LyricToken]] = []
    active: dict[tuple[int, int], dict] = {}
    note_count = 0

    for tick, group in _group_by_tick(track):
        ordered = sorted(group, key=lambda m: 0 if _is_note_off(m) else 1)
        for msg in ordered:
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active[key] = {
                    "pitch": msg.note,
                    "velocity": msg.velocity,
                    "start_tick": tick,
                }
            elif _is_note_off(msg):
                key = (msg.channel, msg.note)
                if key not in active:
                    continue
                start = active.pop(key)
                end_tick = tick
                if end_tick <= start["start_tick"]:
                    continue
                start_sec = tempo_map.tick_to_seconds(start["start_tick"])
                end_sec = tempo_map.tick_to_seconds(end_tick)
                notes.append(
                    NoteEvent(
                        track=track_index,
                        pitch=start["pitch"],
                        velocity=start["velocity"],
                        start_tick=start["start_tick"],
                        end_tick=end_tick,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
                )
                note_count += 1
            elif msg.type in ("lyrics", "text"):
                token = parse_lyric_token(decode_midi_text(msg.text))
                if token is not None:
                    lyrics.append((tick, token))

    return notes, lyrics, note_count


def _pick_melody_track(midi_file: mido.MidiFile, track_hint: str | int) -> int:
    if isinstance(track_hint, int):
        return track_hint
    if track_hint != "auto":
        try:
            return int(track_hint)
        except ValueError:
            pass

    best_index = 0
    best_score = -1
    for index, track in enumerate(midi_file.tracks):
        note_count = sum(1 for msg in track if msg.type == "note_on" and msg.velocity > 0)
        lyric_count = sum(1 for msg in track if msg.type in ("lyrics", "text"))
        score = note_count * 2 + lyric_count
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _align_lyrics_to_notes(notes: list[NoteEvent], lyrics: list[tuple[int, LyricToken]]) -> None:
    if not notes:
        return

    notes_sorted = sorted(notes, key=lambda n: (n.start_tick, n.end_tick))
    lyric_index = 0
    last_word_base = ""

    for note in notes_sorted:
        while lyric_index < len(lyrics) and lyrics[lyric_index][0] < note.start_tick:
            lyric_index += 1

        candidates = []
        for offset in (0, 1, -1, 2, -2):
            idx = lyric_index + offset
            if 0 <= idx < len(lyrics):
                tick, token = lyrics[idx]
                if abs(tick - note.start_tick) <= 8:
                    candidates.append((abs(tick - note.start_tick), idx, token))

        if not candidates and lyric_index < len(lyrics):
            tick, token = lyrics[lyric_index]
            if abs(tick - note.start_tick) <= 48:
                candidates.append((abs(tick - note.start_tick), lyric_index, token))

        if not candidates:
            continue

        _, best_idx, token = min(candidates, key=lambda item: item[0])
        lyric_index = max(lyric_index, best_idx + 1)
        note.lyric = token

        if token.is_sustain:
            note.is_slur = True
            if last_word_base:
                note.lyric = LyricToken(
                    raw=token.raw,
                    text=last_word_base,
                    is_sustain=True,
                    syllable_index=0,
                    syllable_base=last_word_base,
                )
        else:
            last_word_base = token.syllable_base or token.text


def parse_midi_file(path: str, melody_track: str | int = "auto") -> ParsedMidi:
    midi_file = mido.MidiFile(path)
    tempo_map = TempoMap.from_midi(midi_file)
    track_index = _pick_melody_track(midi_file, melody_track)

    all_notes: list[NoteEvent] = []
    all_lyrics: list[tuple[int, LyricToken]] = []

    for index, track in enumerate(midi_file.tracks):
        notes, lyrics, _ = _collect_track_notes(index, track, tempo_map)
        if index == track_index:
            all_notes.extend(notes)
            all_lyrics.extend(lyrics)
        elif melody_track == "all":
            all_notes.extend(notes)
            all_lyrics.extend(lyrics)

    if not all_notes:
        for index, track in enumerate(midi_file.tracks):
            notes, lyrics, _ = _collect_track_notes(index, track, tempo_map)
            all_notes.extend(notes)
            all_lyrics.extend(lyrics)

    _align_lyrics_to_notes(all_notes, sorted(all_lyrics, key=lambda item: item[0]))

    all_notes.sort(key=lambda n: (n.start_tick, n.end_tick))

    from core.lyric_phoneme import resolve_melisma_syllables

    resolve_melisma_syllables(all_notes)

    duration_sec = max((n.end_sec for n in all_notes), default=0.0)

    return ParsedMidi(
        path=path,
        ticks_per_beat=midi_file.ticks_per_beat,
        tempo_map=tempo_map,
        notes=all_notes,
        duration_sec=duration_sec,
        melody_track=track_index if isinstance(track_index, int) else 0,
    )


def summarize_parsed_midi(parsed: ParsedMidi) -> str:
    lines = [
        f"音符数: {len(parsed.notes)}",
        f"带歌词音符: {parsed.lyric_note_count}",
        f"时长: {parsed.duration_sec:.2f}s",
        f"旋律轨: {parsed.melody_track}",
    ]
    preview = []
    for note in parsed.notes[:12]:
        lyric = note.lyric.text if note.lyric else "(无)"
        slur = " [延音]" if note.is_slur else ""
        preview.append(
            f"  {note.start_sec:6.2f}s  MIDI{note.pitch:3d}  {lyric}{slur}"
        )
    if len(parsed.notes) > 12:
        preview.append(f"  ... 共 {len(parsed.notes)} 个音符")
    lines.extend(preview)
    return "\n".join(lines)
