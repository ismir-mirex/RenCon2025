#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import partitura as pt
from partitura.musicanalysis.performance_codec import encode_performance
from partitura.musicanalysis.performance_features import compute_matched_score


def load_score_with_ids(score_path: Path):
    score = pt.load_musicxml(str(score_path))
    pt.score.assign_note_ids(score.parts)
    score_na = score.note_array()
    return score, score_na


def safe_stem_from_alignment(alignment_path: Path):
    name = alignment_path.name
    if name.endswith(".alignment.json"):
        return name[: -len(".alignment.json")]
    return alignment_path.stem


def alignment_match_map(alignment):
    mapping = {}
    for item in alignment:
        if item.get("label") == "match":
            sid = item.get("score_id")
            pid = item.get("performance_id")
            if sid is not None and pid is not None:
                mapping[sid] = pid
    return mapping


def write_note_params_csv(
    out_path: Path,
    score_na,
    snote_ids,
    params,
    perf_id_map,
):
    # Index parameters by score id.
    param_by_id = {sid: params[i] for i, sid in enumerate(snote_ids)}

    fieldnames = [
        "score_id",
        "performance_id",
        "onset_beat",
        "duration_beat",
        "pitch",
        "voice",
        "beat_period",
        "tempo_bpm",
        "timing",
        "articulation_log",
        "velocity_norm",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in score_na:
            sid = row["id"]
            if sid not in param_by_id:
                continue
            p = param_by_id[sid]
            beat_period = float(p["beat_period"]) if "beat_period" in p.dtype.names else math.nan
            tempo_bpm = 60.0 / beat_period if beat_period and beat_period > 0 else math.nan
            writer.writerow(
                {
                    "score_id": sid,
                    "performance_id": perf_id_map.get(sid, ""),
                    "onset_beat": float(row["onset_beat"]),
                    "duration_beat": float(row["duration_beat"]),
                    "pitch": int(row["pitch"]),
                    "voice": int(row["voice"]),
                    "beat_period": beat_period,
                    "tempo_bpm": tempo_bpm,
                    "timing": float(p["timing"]),
                    "articulation_log": float(p["articulation_log"]),
                    "velocity_norm": float(p["velocity"]),
                }
            )


def write_tempo_curve_csv(out_path: Path, m_score, params, unique_onset_idxs):
    fieldnames = ["onset_beat", "beat_period", "tempo_bpm", "timing"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in unique_onset_idxs:
            idx0 = int(group[0])
            onset_beat = float(m_score["onset"][idx0])
            beat_period = float(params["beat_period"][idx0])
            tempo_bpm = 60.0 / beat_period if beat_period > 0 else math.nan
            timing = float(params["timing"][idx0])
            writer.writerow(
                {
                    "onset_beat": onset_beat,
                    "beat_period": beat_period,
                    "tempo_bpm": tempo_bpm,
                    "timing": timing,
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compute performance parameters from alignments."
    )
    parser.add_argument("--score", required=True, type=Path, help="Path to MusicXML score")
    parser.add_argument(
        "--alignments", required=True, type=Path, help="Directory of alignment JSON files"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    score, score_na = load_score_with_ids(args.score)

    alignment_files = sorted(args.alignments.glob("*.alignment.json"))
    if not alignment_files:
        raise SystemExit(f"No alignment JSON files under {args.alignments}")

    for align_path in alignment_files:
        payload = json.loads(align_path.read_text())
        perf_path = Path(payload["performance_path"])
        alignment = payload["alignment"]
        match_items = [a for a in alignment if a.get("label") == "match"]
        if not match_items:
            print(f"skip {perf_path} (no matches in alignment)")
            continue

        performance = pt.load_performance_midi(str(perf_path))
        try:
            params, snote_ids, unique_onset_idxs = encode_performance(
                score, performance, alignment, return_u_onset_idx=True
            )
            m_score, _, _ = compute_matched_score(score, performance, alignment)
        except Exception as exc:
            print(f"error {perf_path}: {exc}")
            continue

        perf_id_map = alignment_match_map(alignment)
        stem = safe_stem_from_alignment(align_path)
        note_out = args.output / f"{stem}.note_params.csv"
        tempo_out = args.output / f"{stem}.tempo_curve.csv"

        write_note_params_csv(note_out, score_na, snote_ids, params, perf_id_map)
        write_tempo_curve_csv(tempo_out, m_score, params, unique_onset_idxs)


if __name__ == "__main__":
    main()
