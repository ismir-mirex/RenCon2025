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


def is_transcribed(name: str):
    return "transkun" in name.lower()


def robust_clip(x, p_low, p_high):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    lo = np.nanpercentile(x, p_low)
    hi = np.nanpercentile(x, p_high)
    return np.clip(x, lo, hi)


def remove_outliers_iqr(x, k=3.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return x, (x >= lo) & (x <= hi)


def filter_tempo(tempo_raw, clip_p, k_iqr=3.0, bpm_max=150.0):
    t = np.asarray(tempo_raw, dtype=float)
    mask = np.isfinite(t)
    if bpm_max is not None:
        mask = mask & (t <= bpm_max)
    t = t[mask]
    if t.size == 0:
        return t, mask
    if clip_p > 0:
        t = robust_clip(t, clip_p, 100 - clip_p)
    t, keep = remove_outliers_iqr(t, k=k_iqr)
    # rebuild mask on original length using finite mask then keep
    finite_idx = np.where(mask)[0]
    final_mask = np.zeros_like(mask, dtype=bool)
    final_mask[finite_idx[keep]] = True
    return t, final_mask


def filter_timing(timing_raw, sec_max=0.25):
    t = np.asarray(timing_raw, dtype=float)
    mask = np.isfinite(t)
    if sec_max is not None:
        mask = mask & (np.abs(t) <= sec_max)
    return t[mask], mask


def load_note_params(params_dir: Path):
    rows = []
    for note_path in sorted(params_dir.glob("*.note_params.csv")):
        team = team_from_filename(note_path.name)
        if team is None:
            continue
        if team == "Contin-U":
            # Exclude transcribed Contin-U from analysis
            continue
        note_df = pd.read_csv(note_path)
        rows.append(
            {
                "team": team,
                "note_path": note_path,
                "transcribed": is_transcribed(note_path.name),
                "note_df": note_df,
            }
        )
    return rows


def load_tempo_curves(params_dir: Path):
    rows = []
    for tempo_path in sorted(params_dir.glob("*.tempo_curve.csv")):
        team = team_from_filename(tempo_path.name)
        if team is None:
            continue
        if team == "Contin-U":
            # Exclude transcribed Contin-U from analysis
            continue
        tempo_df = pd.read_csv(tempo_path)
        rows.append(
            {
                "team": team,
                "tempo_path": tempo_path,
                "transcribed": is_transcribed(tempo_path.name),
                "tempo_df": tempo_df,
            }
        )
    return rows


def make_label(team, transcribed):
    return f"{team} (transcribed)" if transcribed else team


def plot_curves(tempo_rows, note_rows, out_dir: Path, clip_p):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tempo curve overlay
    plt.figure(figsize=(10, 4.2), dpi=200)
    for row in tempo_rows:
        df = row["tempo_df"]
        tempo_raw = df["tempo_bpm"].to_numpy()
        tempo, keep_mask = filter_tempo(tempo_raw, clip_p, bpm_max=150.0)
        onset = df["onset_beat"].to_numpy()
        onset = onset[keep_mask]
        # Keep raw (no smoothing), just sort by onset
        order = np.argsort(onset)
        onset = onset[order]
        tempo = tempo[order]
        label = make_label(row["team"], row["transcribed"])
        plt.plot(onset, tempo, linewidth=0.8, alpha=0.9, label=label)
    plt.xlabel("Onset (beats)")
    plt.ylabel("Tempo (BPM)")
    plt.title("Tempo curves (filtered, no smoothing)")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "tempo_curves.png", bbox_inches="tight")
    plt.close()

    # Velocity curve overlay (median velocity per onset_beat)
    plt.figure(figsize=(10, 4.2), dpi=200)
    for row in note_rows:
        df = row["note_df"]
        grouped = df.groupby("onset_beat")["velocity_norm"].median().reset_index()
        onset = grouped["onset_beat"].to_numpy()
        vel = grouped["velocity_norm"].to_numpy()
        order = np.argsort(onset)
        onset = onset[order]
        vel = vel[order]
        label = make_label(row["team"], row["transcribed"])
        plt.plot(onset, vel, linewidth=0.8, alpha=0.9, label=label)
    plt.xlabel("Onset (beats)")
    plt.ylabel("Velocity (normalized)")
    plt.title("Velocity curves (median per onset)")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "velocity_curves.png", bbox_inches="tight")
    plt.close()

    # Timing curve overlay (median timing per onset_beat)
    plt.figure(figsize=(10, 4.2), dpi=200)
    for row in note_rows:
        df = row["note_df"]
        timing_vals, timing_mask = filter_timing(df["timing"].to_numpy(), sec_max=0.25)
        df_t = df[timing_mask]
        grouped = df_t.groupby("onset_beat")["timing"].median().reset_index()
        onset = grouped["onset_beat"].to_numpy()
        timing = grouped["timing"].to_numpy()
        order = np.argsort(onset)
        onset = onset[order]
        timing = timing[order]
        label = make_label(row["team"], row["transcribed"])
        plt.plot(onset, timing, linewidth=0.8, alpha=0.9, label=label)
    plt.xlabel("Onset (beats)")
    plt.ylabel("Timing deviation (sec)")
    plt.title("Timing curves (median per onset)")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "timing_curves.png", bbox_inches="tight")
    plt.close()

    # Articulation curve overlay (median articulation_log per onset_beat)
    plt.figure(figsize=(10, 4.2), dpi=200)
    for row in note_rows:
        df = row["note_df"]
        grouped = df.groupby("onset_beat")["articulation_log"].median().reset_index()
        onset = grouped["onset_beat"].to_numpy()
        art = grouped["articulation_log"].to_numpy()
        order = np.argsort(onset)
        onset = onset[order]
        art = art[order]
        label = make_label(row["team"], row["transcribed"])
        plt.plot(onset, art, linewidth=0.8, alpha=0.9, label=label)
    plt.xlabel("Onset (beats)")
    plt.ylabel("Articulation (log)")
    plt.title("Articulation curves (median per onset)")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "articulation_curves.png", bbox_inches="tight")
    plt.close()


def plot_distributions(tempo_rows, note_rows, out_dir: Path, clip_p):
    out_dir.mkdir(parents=True, exist_ok=True)
    teams = []
    tempo_vals = []
    vel_vals = []
    timing_vals = []
    art_vals = []

    for row in tempo_rows:
        label = make_label(row["team"], row["transcribed"])
        tempo_raw = row["tempo_df"]["tempo_bpm"].to_numpy()
        t, _ = filter_tempo(tempo_raw, clip_p, bpm_max=150.0)
        teams.append(label)
        tempo_vals.append(t)

    for row in note_rows:
        label = make_label(row["team"], row["transcribed"])
        df = row["note_df"]
        vel_vals.append(df["velocity_norm"].to_numpy())
        timing_vals.append(filter_timing(df["timing"].to_numpy(), sec_max=0.25)[0])
        art_vals.append(df["articulation_log"].to_numpy())

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=200)
    ax = axes[0, 0]
    ax.boxplot(tempo_vals, labels=teams, vert=True, showfliers=False)
    ax.set_title("Tempo distribution (BPM)")
    ax.tick_params(axis="x", labelrotation=45)

    ax = axes[0, 1]
    ax.boxplot(vel_vals, labels=teams, vert=True, showfliers=False)
    ax.set_title("Velocity distribution (normalized)")
    ax.tick_params(axis="x", labelrotation=45)

    ax = axes[1, 0]
    ax.boxplot(timing_vals, labels=teams, vert=True, showfliers=False)
    ax.set_title("Timing deviation distribution (sec)")
    ax.tick_params(axis="x", labelrotation=45)

    ax = axes[1, 1]
    ax.boxplot(art_vals, labels=teams, vert=True, showfliers=False)
    ax.set_title("Articulation log distribution")
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", bbox_inches="tight")
    plt.close(fig)


def plot_worms(tempo_rows, note_rows, out_dir: Path, clip_p, worm_mid_pct, worm_max_points):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build tempo/dynamics trajectories at onset level per team
    tempo_map = {r["team"]: r for r in tempo_rows}
    # sort by score descending so best appears first
    note_rows = sorted(
        note_rows,
        key=lambda r: SCORE_BY_TEAM.get(r["team"], float("-inf")),
        reverse=True,
    )
    n = len(note_rows)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.3 * rows), dpi=200, sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for i, row in enumerate(note_rows):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        df = row["note_df"]
        team = row["team"]
        if team not in tempo_map:
            continue
        tempo_df = tempo_map[team]["tempo_df"]
        # tempo trajectory
        tempo_onset = tempo_df["onset_beat"].to_numpy()
        tempo_raw = tempo_df["tempo_bpm"].to_numpy()
        tempo_bpm, keep_mask = filter_tempo(tempo_raw, clip_p, bpm_max=150.0)
        tempo_onset = tempo_onset[keep_mask]
        # dynamics trajectory (median velocity per onset)
        # also drop notes at outlier tempo onsets
        df = df[df["onset_beat"].isin(tempo_onset)]
        dyn = df.groupby("onset_beat")["velocity_norm"].median().reset_index()
        dyn_onset = dyn["onset_beat"].to_numpy()
        dyn_val = dyn["velocity_norm"].to_numpy()

        # align by nearest onset (simple merge)
        if tempo_onset.size == 0 or dyn_onset.size == 0:
            continue
        # interpolate dynamics onto tempo onsets
        order = np.argsort(dyn_onset)
        dyn_onset = dyn_onset[order]
        dyn_val = dyn_val[order]
        dyn_interp = np.interp(tempo_onset, dyn_onset, dyn_val, left=np.nan, right=np.nan)

        # time ordering for fading trail
        order_t = np.argsort(tempo_onset)
        tempo_onset = tempo_onset[order_t]
        tempo_bpm = tempo_bpm[order_t]
        dyn_interp = dyn_interp[order_t]

        # take a middle segment to avoid messy global trajectory
        if worm_mid_pct > 0 and tempo_onset.size > 0:
            p_lo = 50 - worm_mid_pct / 2
            p_hi = 50 + worm_mid_pct / 2
            lo_on = np.nanpercentile(tempo_onset, p_lo)
            hi_on = np.nanpercentile(tempo_onset, p_hi)
            seg_mask = (tempo_onset >= lo_on) & (tempo_onset <= hi_on)
            tempo_onset = tempo_onset[seg_mask]
            tempo_bpm = tempo_bpm[seg_mask]
            dyn_interp = dyn_interp[seg_mask]

        # downsample if needed
        if tempo_bpm.size > worm_max_points:
            idx = np.linspace(0, tempo_bpm.size - 1, worm_max_points).astype(int)
            tempo_onset = tempo_onset[idx]
            tempo_bpm = tempo_bpm[idx]
            dyn_interp = dyn_interp[idx]

        # performance worm: tempo (x) vs dynamics (y), color by time
        npts = len(tempo_bpm)
        if npts < 2:
            continue
        colors = np.linspace(0.0, 1.0, npts)
        sc = ax.scatter(
            tempo_bpm,
            dyn_interp,
            s=18,
            c=colors,
            cmap="rainbow",
            alpha=0.9,
            edgecolors="none",
        )

        score = SCORE_BY_TEAM.get(team, float("nan"))
        label = make_label(team, row["transcribed"])
        if np.isfinite(score):
            title = f"{label} (score={score:.2f})"
        else:
            title = f"{label} (score=NA)"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Tempo (BPM)")
        ax.set_ylabel("Dynamics (velocity)")

    # Remove empty subplots
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        fig.delaxes(axes[r, c])

    # place colorbar bottom-right
    cax = fig.add_axes([0.72, 0.03, 0.25, 0.02])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Time (start → end)")
    fig.suptitle("Performance worm (tempo vs dynamics, colored by time)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_worms.png", bbox_inches="tight")
    plt.close(fig)


def plot_score_scatter(tempo_rows, note_rows, out_dir: Path, clip_p):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for row in tempo_rows:
        team = row["team"]
        score = SCORE_BY_TEAM.get(team, np.nan)
        tempo_raw = row["tempo_df"]["tempo_bpm"].to_numpy()
        tempo, _ = filter_tempo(tempo_raw, clip_p, bpm_max=150.0)
        tempo_range = np.nanpercentile(tempo, 95) - np.nanpercentile(tempo, 5)
        tempo_iqr = np.nanpercentile(tempo, 75) - np.nanpercentile(tempo, 25)
        tempo_vol = np.nanstd(np.diff(tempo)) if tempo.size > 1 else np.nan
        metrics.append(
            {
                "team": team,
                "label": make_label(team, row["transcribed"]),
                "score": score,
                "tempo_range": tempo_range,
                "tempo_iqr": tempo_iqr,
                "tempo_volatility": tempo_vol,
            }
        )

    for row in note_rows:
        team = row["team"]
        score = SCORE_BY_TEAM.get(team, np.nan)
        vel = row["note_df"]["velocity_norm"].to_numpy()
        timing = filter_timing(row["note_df"]["timing"].to_numpy(), sec_max=0.25)[0]
        art = row["note_df"]["articulation_log"].to_numpy()
        for m in metrics:
            if m["team"] == team:
                m["velocity_range"] = np.nanpercentile(vel, 95) - np.nanpercentile(vel, 5)
                m["velocity_std"] = np.nanstd(vel)
                m["timing_std"] = np.nanstd(timing)
                m["timing_median"] = np.nanmedian(timing)
                m["articulation_std"] = np.nanstd(art)
                m["articulation_mean"] = np.nanmean(art)
                break

    df = pd.DataFrame(metrics)

    # helper for interpolation line
    def add_interp(ax, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        # linear fit
        m, b = np.polyfit(x[mask], y[mask], 1)
        ys = m * xs + b
        ax.plot(xs, ys, color="#7a1f1f", linewidth=1.0)
        # Pearson correlation
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        return r

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5), dpi=200)
    axes = axes.ravel()

    plots = [
        ("tempo_range", "Tempo range (P95–P5, BPM)"),
        ("tempo_iqr", "Tempo IQR (P75–P25, BPM)"),
        ("tempo_volatility", "Tempo volatility (std of Δ tempo)"),
        ("velocity_range", "Velocity range (P95–P5)"),
        ("velocity_std", "Velocity std"),
        ("timing_std", "Timing std (sec)"),
    ]

    for ax, (col, xlabel) in zip(axes, plots):
        ax.scatter(df[col], df["score"], s=60)
        for _, r in df.iterrows():
            ax.text(r[col] + (0.02 if "tempo" in col else 0.005), r["score"], r["label"], fontsize=9, va="center")
        r = add_interp(ax, df[col], df["score"])
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Audience score", fontsize=12)
        if r is not None and np.isfinite(r):
            ax.set_title(f"{xlabel} vs score (r={r:.3f})", fontsize=12)
        else:
            ax.set_title(f"{xlabel} vs score", fontsize=12)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=11)
        # half border: keep left/bottom spines only
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "score_scatter_grid.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Expressive parameter overview plots.")
    parser.add_argument("--params", required=True, type=Path, help="Directory with params CSVs")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for figures")
    parser.add_argument(
        "--clip-pct",
        type=float,
        default=1.0,
        help="Percentile clip for tempo; 0 means no clipping",
    )
    parser.add_argument(
        "--worm-mid-pct",
        type=float,
        default=30.0,
        help="Percentile width of middle segment for worm (e.g., 30 keeps middle 30%)",
    )
    parser.add_argument(
        "--worm-max-points",
        type=int,
        default=250,
        help="Max points to plot in worm trajectory",
    )
    args = parser.parse_args()

    tempo_rows = load_tempo_curves(args.params)
    note_rows = load_note_params(args.params)

    plot_curves(tempo_rows, note_rows, args.out, args.clip_pct)
    plot_distributions(tempo_rows, note_rows, args.out, args.clip_pct)
    plot_worms(tempo_rows, note_rows, args.out, args.clip_pct, args.worm_mid_pct, args.worm_max_points)
    plot_score_scatter(tempo_rows, note_rows, args.out, args.clip_pct)


if __name__ == "__main__":
    main()
