"""生成测试 MIDI 并验证解析流水线。"""

from __future__ import annotations

import os
import sys
import tempfile

import mido

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.ds_builder import build_ds_segment, format_ds_preview
from core.midi_parser import parse_midi_file, summarize_parsed_midi


def make_test_midi(path: str) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    notes = [
        (64, 480, "baby#1"),
        (64, 480, "baby#2"),
        (65, 480, "girl"),
        (65, 960, "-"),
    ]

    for pitch, duration, lyric in notes:
        track.append(mido.MetaMessage("lyrics", text=lyric, time=0))
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=duration))

    mid.save(path)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        midi_path = os.path.join(tmp, "test_baby.mid")
        make_test_midi(midi_path)

        parsed = parse_midi_file(midi_path)
        print("=== 解析摘要 ===")
        print(summarize_parsed_midi(parsed))

        segment = build_ds_segment(parsed, "en")
        print("\n=== DS 预览 ===")
        print(format_ds_preview(segment))

        assert len(parsed.notes) == 4
        assert parsed.lyric_note_count == 4
        assert parsed.notes[3].is_slur
        assert "baby" in segment["text"].lower() or "B" in segment["ph_seq"]
        print("\n测试通过。")


if __name__ == "__main__":
    main()
