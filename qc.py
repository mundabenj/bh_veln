"""Study 1, Phase 3: quality control of the harmonised E4 data (Protocol section 5, step 2).

Usage (from the bh_veln folder, after `study_one.py harmonise`):
    uv run python qc.py                      # all datasets
    uv run python qc.py wesad physionet      # a subset
    uv run python qc.py --overwrite          # recompute existing QC files

For every participant this writes Datasets/harmonised/<dataset>/<participant>/qc.parquet on the
4 Hz label grid, with one boolean column per artifact flag and, for each rule set, whether each
signal (EDA, BVP, TEMP) and the sample as a whole pass. reports/qc_report.md compares the rule
sets: share of data discarded per dataset, shared class and participant, and who fails the
inclusion rule (Protocol section 4: at least INCLUSION_MIN of the recording passes QC).

High accelerometer activity is flagged and reported but never discarded: physical activity is a
class of interest (Experiment C), not an artifact.
"""
import argparse
import os

import neurokit2 as nk
import numpy as np
import pandas as pd

import study_one as so

QC_FS = so.LABEL_FS          # flags live on the 4 Hz EDA/label grid
BVP_FS = 64
INCLUSION_MIN = 0.70         # Protocol section 4 placeholder [70]%
REPORT_PATH = os.path.join(so.PROJECT_DIR, "reports", "qc_report.md")

# Rule sets compared in Phase 3. Thresholds in the units of the harmonised data.
RULE_SETS = {
    # Physical sensor limits and obvious non-wear only.
    "lenient": dict(eda_min=0.01, eda_max=100.0, eda_slope=None, temp_min=25.0, temp_max=45.0,
                    ibi_min=0.25, ibi_max=2.5, ibi_change=None, buffer_s=0),
    # Kleckner et al. (2018) EDA rules, plus a 30-200 bpm plausibility range for inter-beat intervals.
    "kleckner": dict(eda_min=0.05, eda_max=60.0, eda_slope=10.0, temp_min=30.0, temp_max=40.0,
                     ibi_min=0.3, ibi_max=2.0, ibi_change=None, buffer_s=5),
    # Kleckner plus a successive-IBI change limit (20%, Malik-style ectopic/artifact rule).
    "strict": dict(eda_min=0.05, eda_max=60.0, eda_slope=10.0, temp_min=30.0, temp_max=40.0,
                   ibi_min=0.3, ibi_max=2.0, ibi_change=0.20, buffer_s=5),
}
# Wrist ENMO above 100 mg ~ moderate-to-vigorous activity (Hildebrand et al., 2014), averaged over 5 s.
ACC_ENMO_HIGH = 0.1
ACC_WINDOW_S = 5


# ---------------------------------------------------------------------------
# Signal-level helpers
# ---------------------------------------------------------------------------
def _on_grid(series, grid, tolerance_s):
    """Value of a time series at each grid time (last value at or before it, within tolerance)."""
    df = pd.merge_asof(pd.DataFrame(index=grid), series.rename("v").to_frame(), left_index=True,
                       right_index=True, direction="backward", tolerance=pd.Timedelta(seconds=tolerance_s))
    return df["v"].to_numpy()


def _dilate(flag, n):
    """Extend True runs by n samples on each side (transition buffer)."""
    if n <= 0 or not flag.any():
        return flag
    k = np.ones(2 * n + 1)
    return np.convolve(flag.astype(float), k, mode="same") > 0


def detect_beats(bvp):
    """BVP DataFrame (one session) -> Series of IBI (s) indexed by beat time, via NeuroKit2 (Elgendi)."""
    x = bvp["bvp"].to_numpy(dtype=float)
    if len(x) < BVP_FS * 10:
        return pd.Series(dtype=float)
    clean = nk.ppg_clean(x, sampling_rate=BVP_FS, method="elgendi")
    peaks = nk.ppg_findpeaks(clean, sampling_rate=BVP_FS, method="elgendi")["PPG_Peaks"]
    if len(peaks) < 2:
        return pd.Series(dtype=float)
    times = bvp.index[peaks]
    ibi = np.diff(times.asi8) / 1e9
    return pd.Series(ibi, index=times[1:])


def _ibi_flags_on_grid(ibi, grid, rules):
    """Flag grid samples whose enclosing beat interval is implausible (or has no beats around it)."""
    if ibi.empty:
        return np.ones(len(grid), bool)
    bad = (ibi < rules["ibi_min"]) | (ibi > rules["ibi_max"])
    if rules["ibi_change"]:
        prev = ibi.shift(1)
        bad |= ((ibi - prev).abs() / prev > rules["ibi_change"]).fillna(False)
    # Beat k closes the interval (t_k - ibi_k, t_k]; a grid time belongs to the first beat at or after
    # it, if it falls inside that beat's interval. Times not covered by any interval (before the first
    # beat, after the last, or in gaps between sessions) are invalid.
    k = ibi.index.searchsorted(grid, side="left")
    out = np.ones(len(grid), bool)
    has = k < len(ibi)
    starts = ibi.index.asi8 - (ibi.to_numpy() * 1e9).astype(np.int64)
    inside = np.zeros(len(grid), bool)
    inside[has] = grid.asi8[has] > starts[k[has]]
    out[inside] = bad.to_numpy()[k[inside]]
    return out


# ---------------------------------------------------------------------------
# Per-participant QC
# ---------------------------------------------------------------------------
def qc_participant(dataset, pid):
    rec = so.load_harmonised(dataset, pid)
    lab = rec["labels"]
    grid = lab.index
    n = len(grid)
    out = pd.DataFrame(index=grid)
    out["session"] = lab["session"]

    eda = rec["eda"]["eda"].to_numpy(dtype=float)             # EDA defines the grid (same order)
    assert len(eda) == n and rec["eda"].index.equals(grid), f"{dataset}/{pid}: EDA and label grids differ"
    temp = (_on_grid(rec["temp"]["temp"], grid, 0.5) if "temp" in rec else np.full(n, np.nan))
    sess = lab["session"].to_numpy()
    new_sess = np.r_[True, sess[1:] != sess[:-1]]
    slope = np.abs(np.r_[0.0, np.diff(eda)]) * QC_FS          # uS/s between consecutive samples
    slope[new_sess] = 0.0

    # Accelerometer: ENMO = max(|a| - 1 g, 0), averaged over ACC_WINDOW_S (flag only).
    if "acc" in rec:
        a = rec["acc"]
        enmo = np.clip(np.sqrt((a[["acc_x", "acc_y", "acc_z"]] ** 2).sum(axis=1)) - 1.0, 0, None)
        enmo = enmo.groupby(a["session"], observed=True).transform(
            lambda s: s.rolling(f"{ACC_WINDOW_S}s").mean())
        out["acc_enmo"] = _on_grid(enmo, grid, 0.5).astype("float32")
        out["acc_high"] = out["acc_enmo"].to_numpy() > ACC_ENMO_HIGH
    else:
        out["acc_enmo"] = np.nan
        out["acc_high"] = False

    # Beats are detected once per session; rule sets differ only in which IBIs they reject.
    ibis = []
    if "bvp" in rec:
        for _, b in rec["bvp"].groupby("session", observed=True):
            ibis.append(detect_beats(b))
    ibi = pd.concat(ibis).sort_index() if ibis else pd.Series(dtype=float)
    ibi = ibi[~ibi.index.duplicated()]   # overlapping sessions (nurse BG) can repeat beats
    if "bvp" in rec:
        out["has_bvp"] = ~np.isnan(_on_grid(rec["bvp"]["bvp"], grid, 2 / BVP_FS))
    else:
        out["has_bvp"] = False
    out["has_temp"] = ~np.isnan(temp)

    # Non-wear: EDA near zero and skin temperature low (informational, reported separately).
    out["non_wear"] = (eda < 0.05) & (np.nan_to_num(temp, nan=99) < 30)

    for name, r in RULE_SETS.items():
        buf = int(r["buffer_s"] * QC_FS)
        eda_bad = (eda < r["eda_min"]) | (eda > r["eda_max"]) | np.isnan(eda)
        if r["eda_slope"]:
            eda_bad |= slope > r["eda_slope"]
        temp_bad = ((temp < r["temp_min"]) | (temp > r["temp_max"])) & ~np.isnan(temp)
        # Kleckner: an out-of-range temperature means the sensor is off the wrist, so EDA is invalid too.
        eda_bad = _dilate(eda_bad | temp_bad, buf)
        temp_bad = _dilate(temp_bad | np.isnan(temp), buf)
        bvp_bad = _ibi_flags_on_grid(ibi, grid, r) | ~out["has_bvp"].to_numpy()

        out[f"{name}_eda_ok"] = ~eda_bad
        out[f"{name}_bvp_ok"] = ~bvp_bad
        out[f"{name}_temp_ok"] = ~temp_bad
        out[f"{name}_ok"] = ~(eda_bad | bvp_bad | temp_bad)

    out["n_beats"] = len(ibi)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _pct(x):
    return f"{100 * x:.1f}%"


def summarise(dataset, pid, qc, lab):
    rows = []
    cls = lab["label"].astype(object).fillna("excluded")
    for name in RULE_SETS:
        ok = qc[f"{name}_ok"]
        row = {"dataset": dataset, "participant": pid, "rule_set": name,
               "hours": len(qc) / QC_FS / 3600, "pass": ok.mean(),
               "eda_pass": qc[f"{name}_eda_ok"].mean(), "bvp_pass": qc[f"{name}_bvp_ok"].mean(),
               "temp_pass": qc[f"{name}_temp_ok"].mean(),
               "acc_high": qc["acc_high"].mean(), "non_wear": qc["non_wear"].mean()}
        for c in so.SHARED_CLASSES + ["excluded"]:
            m = (cls == c).to_numpy()
            row[f"n_{c}"] = int(m.sum())
            row[f"pass_{c}"] = ok[m].mean() if m.any() else np.nan
        rows.append(row)
    return rows


def write_report(summary):
    s = summary.copy()
    s["included"] = s["pass"] >= INCLUSION_MIN
    md = ["# Phase 3 quality-control report", "",
          "_Generated by `qc.py`; do not edit by hand._", "",
          "Share of data that **passes** each rule set (all of EDA, BVP and TEMP valid). "
          f"Inclusion rule: a participant is included if at least {_pct(INCLUSION_MIN)} of their "
          "recording passes. High accelerometer activity is reported but not discarded.", "",
          "## Rule sets", "",
          "| rule set | EDA range (µS) | EDA slope (µS/s) | TEMP range (°C) | IBI range (s) | IBI change | buffer (s) |",
          "|---|---|---|---|---|---|---|"]
    for name, r in RULE_SETS.items():
        md.append(f"| {name} | {r['eda_min']}–{r['eda_max']} | {r['eda_slope'] or '–'} | "
                  f"{r['temp_min']}–{r['temp_max']} | {r['ibi_min']}–{r['ibi_max']} | "
                  f"{_pct(r['ibi_change']) if r['ibi_change'] else '–'} | {r['buffer_s']} |")
    md += ["", f"Accelerometer flag: 5-s mean wrist ENMO > {ACC_ENMO_HIGH * 1000:.0f} mg.", ""]

    # Hours-weighted pass rates per dataset x rule set, overall and per class.
    md += ["## Pass rate by dataset and shared class", ""]
    for ds, g in s.groupby("dataset", sort=False):
        md += [f"### {ds}", "",
               "| rule set | overall | EDA | BVP | TEMP | " + " | ".join(so.SHARED_CLASSES + ["excluded"])
               + " | participants included | high ACC | non-wear |",
               "|---" * (8 + len(so.SHARED_CLASSES)) + "|"]
        for name, r in g.groupby("rule_set", sort=False):
            w = r["hours"]
            cells = [_pct(np.average(r[c], weights=w)) for c in ["pass", "eda_pass", "bvp_pass", "temp_pass"]]
            for c in so.SHARED_CLASSES + ["excluded"]:
                n = r[f"n_{c}"]
                cells.append(_pct(np.nansum(r[f"pass_{c}"] * n) / n.sum()) if n.sum() else "–")
            cells += [f"{int(r['included'].sum())}/{len(r)}",
                      _pct(np.average(r["acc_high"], weights=w)), _pct(np.average(r["non_wear"], weights=w))]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")

    md += ["## Participants failing the inclusion rule", ""]
    failing = s[~s["included"]]
    if failing.empty:
        md.append("None under any rule set.")
    else:
        md += ["| dataset | participant | rule set | pass | EDA | BVP | TEMP |", "|---|---|---|---|---|---|---|"]
        for _, r in failing.iterrows():
            md.append(f"| {r['dataset']} | {r['participant']} | {r['rule_set']} | {_pct(r['pass'])} | "
                      f"{_pct(r['eda_pass'])} | {_pct(r['bvp_pass'])} | {_pct(r['temp_pass'])} |")
    md.append("")

    md += ["## Per participant (pass rate)", "",
           "| dataset | participant | hours | " + " | ".join(RULE_SETS) + " | high ACC | non-wear |",
           "|---" * (5 + len(RULE_SETS)) + "|"]
    wide = s.pivot_table(index=["dataset", "participant"], columns="rule_set", values="pass", sort=False)
    first = s.drop_duplicates(["dataset", "participant"]).set_index(["dataset", "participant"])
    for (ds, pid), r in wide.iterrows():
        f = first.loc[(ds, pid)]
        md.append(f"| {ds} | {pid} | {f['hours']:.1f} | " + " | ".join(_pct(r[n]) for n in RULE_SETS)
                  + f" | {_pct(f['acc_high'])} | {_pct(f['non_wear'])} |")
    md.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        fh.write("\n".join(md))


def run(datasets, overwrite=False):
    rows = []
    for ds in datasets:
        print(f"=== {ds}")
        for pid in so.list_harmonised(ds):
            path = os.path.join(so.participant_dir(ds, pid), "qc.parquet")
            if os.path.exists(path) and not overwrite:
                qc = pd.read_parquet(path)
            else:
                qc = qc_participant(ds, pid)
                qc.to_parquet(path)
            lab = so.load_harmonised(ds, pid, ["labels"])["labels"]
            r = summarise(ds, pid, qc, lab)
            rows += r
            print(f"  {ds}/{pid}: " + ", ".join(f"{x['rule_set']} {_pct(x['pass'])}" for x in r))
    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    summary.to_csv(os.path.join(os.path.dirname(REPORT_PATH), "qc_summary.csv"), index=False)
    write_report(summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"any of {so.DATASETS} (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing qc.parquet files")
    args = ap.parse_args()
    unknown = set(args.datasets) - set(so.DATASETS)
    if unknown:
        ap.error(f"unknown dataset(s): {sorted(unknown)}; choose from {so.DATASETS}")
    run(args.datasets or so.DATASETS, args.overwrite)
