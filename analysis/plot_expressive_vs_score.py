#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCORE_BY_TEAM = {
    "VirtuosoNet": 3.62,
    "DirectorMusices": 3.06,
    "Midihum": 2.90,
    "Contin-U": 2.90,
    "ScorePerLockNAR": 2.52,
    "RenConnoisseur": 2.40,
    "ElegantAIPianist": 2.08,
    "YQX+": 1.79,
    "Human": 4.40,
}


def team_from_filename(name: str):
    if "DirectorMuscies" in name:
        return "DirectorMusices"
    if "Midihum" in name:
        return "Midihum"
    if "RenConnoseuir" in name:
        return "RenConnoisseur"
    if "Contin-U" in name:
        return "Contin-U"
    if "YQX+" in name:
        return "YQX+"
    if "ScorePerLockNAR" in name:
        return "ScorePerLockNAR"
    if "VirtuosoNet" in name:
        return "VirtuosoNet"
    if "ElegantAIPianist" in name:
        return "ElegantAIPianist"
    if "Human" in name:
        return "Human"
    return None


def robust_range(x):
    return np.nanpercentile(x, 95) - np.nanpercentile(x, 5)


def compute_metrics(params_dir: Path):
    rows = []
    for note_path in sorted(params_dir.glob("*.note_params.csv")):
        team = team_from_filename(note_path.name)
        if team is None:
            continue
        tempo_path = params_dir / note_path.name.replace(".note_params.csv", ".tempo_curve.csv")
        if not tempo_path.exists():
            continue

        note_df = pd.read_csv(note_path)
        tempo_df = pd.read_csv(tempo_path)

        tempo_bpm = tempo_df["tempo_bpm"].to_numpy()
        vel = note_df["velocity_norm"].to_numpy()

        # Basic variability stats
        tempo_std = np.nanstd(tempo_bpm)
        tempo_range = robust_range(tempo_bpm)
        vel_std = np.nanstd(vel)
        vel_range = robust_range(vel)

        # Tempo volatility: std of first differences (captures curve roughness)
        tempo_diff = np.diff(tempo_bpm)
        tempo_volatility = np.nanstd(tempo_diff)

        rows.append(
            {
                "team": team,
                "score": SCORE_BY_TEAM.get(team, np.nan),
                "tempo_std": tempo_std,
                "tempo_range": tempo_range,
                "tempo_volatility": tempo_volatility,
                "velocity_std": vel_std,
                "velocity_range": vel_range,
                "note_path": note_path.name,
            }
        )
    return pd.DataFrame(rows)


def plot(metrics: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by score for consistent labeling
    metrics = metrics.sort_values("score", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200)

    # Panel A: score vs tempo variability (range)
    ax = axes[0]
    x = metrics["tempo_range"].to_numpy()
    y = metrics["score"].to_numpy()
    ax.scatter(x, y, s=60, c="#2a4b8d")
    for i, row in metrics.iterrows():
        label = row["team"]
        if label == "Contin-U" and "transkun" in row["note_path"]:
            label = "Contin-U (transcribed)"
        ax.text(row["tempo_range"] + 0.2, row["score"], label, fontsize=8, va="center")
    # Trend line
    if np.isfinite(x).sum() > 1:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, m * xs + b, color="#7a1f1f", linewidth=1)
    ax.set_xlabel("Tempo range (P95–P5, BPM)")
    ax.set_ylabel("Audience score (0–5)")
    ax.set_title("Tempo expressiveness vs audience score")
    ax.grid(alpha=0.2, linewidth=0.5)

    # Panel B: score vs velocity variability (range)
    ax = axes[1]
    x = metrics["velocity_range"].to_numpy()
    y = metrics["score"].to_numpy()
    ax.scatter(x, y, s=60, c="#1f6f3a")
    for i, row in metrics.iterrows():
        label = row["team"]
        if label == "Contin-U" and "transkun" in row["note_path"]:
            label = "Contin-U (transcribed)"
        ax.text(row["velocity_range"] + 0.01, row["score"], label, fontsize=8, va="center")
    if np.isfinite(x).sum() > 1:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, m * xs + b, color="#7a1f1f", linewidth=1)
    ax.set_xlabel("Velocity range (P95–P5, normalized)")
    ax.set_ylabel("Audience score (0–5)")
    ax.set_title("Dynamics expressiveness vs audience score")
    ax.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot expressive parameter summaries vs scores.")
    parser.add_argument("--params", required=True, type=Path, help="Directory with *.note_params.csv")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output image path (e.g., analysis/figures/expressive_vs_score.png)",
    )
    parser.add_argument(
        "--summary",
        required=False,
        type=Path,
        help="Optional CSV to write summary metrics",
    )
    args = parser.parse_args()

    metrics = compute_metrics(args.params)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.summary, index=False)
    plot(metrics, args.out)


if __name__ == "__main__":
    main()
