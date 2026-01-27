#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import partitura as pt
from parangonar import AutomaticNoteMatcher


def iter_midi_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if path.suffix.lower() in {".mid", ".midi"}:
            yield path


def load_score_note_array(score_path: Path):
    score = pt.load_musicxml(str(score_path))
    # Ensure the score has stable, unique note ids for alignment + analysis.
    pt.score.assign_note_ids(score.parts)
    score_na = score.note_array()
    return score, score_na


def align_one(score_na, perf_path: Path, matcher: AutomaticNoteMatcher):
    performance = pt.load_performance_midi(str(perf_path))
    perf_na = performance.note_array()
    alignment = matcher(score_na, perf_na)
    return performance, perf_na, alignment


def main():
    parser = argparse.ArgumentParser(description="Align MIDI performances to a MusicXML score.")
    parser.add_argument("--score", required=True, type=Path, help="Path to MusicXML score")
    parser.add_argument("--input", required=True, type=Path, help="MIDI file or directory to search")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for alignments")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing alignment files")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    score, score_na = load_score_note_array(args.score)
    matcher = AutomaticNoteMatcher()

    midi_files = sorted(iter_midi_files(args.input))
    if not midi_files:
        raise SystemExit(f"No MIDI files found under {args.input}")

    for midi_path in midi_files:
        rel = midi_path.relative_to(args.input) if args.input.is_dir() else midi_path.name
        safe_rel = str(rel).replace(os.sep, "__")
        out_path = args.output / f"{safe_rel}.alignment.json"
        if out_path.exists() and not args.overwrite:
            print(f"skip {midi_path} (exists)")
            continue

        print(f"align {midi_path}")
        try:
            performance, perf_na, alignment = align_one(score_na, midi_path, matcher)
        except Exception as exc:
            print(f"error {midi_path}: {exc}")
            continue

        payload = {
            "score_path": str(args.score),
            "performance_path": str(midi_path),
            "score_note_count": int(len(score_na)),
            "performance_note_count": int(len(perf_na)),
            "alignment_count": int(len(alignment)),
            "alignment": alignment,
        }
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
