"""Study 1, Phase 2: load the three public E4 datasets and harmonise them into one format.

Usage (from the bh_veln folder):
    uv run python study_one.py harmonise                  # all datasets
    uv run python study_one.py harmonise wesad physionet  # a subset
    uv run python study_one.py harmonise --overwrite      # rebuild existing participants

Harmonised layout (Protocol section 4/5, step 1):
    Datasets/harmonised/<dataset>/<participant>/{eda,temp,bvp,acc,labels}.parquet
    Datasets/harmonised/<dataset>/participants.csv
    bh_veln/data_cards/<dataset>.md

Every table has a UTC DatetimeIndex named "time" and a "session" column. Signals keep the
E4's native rates (EDA/TEMP 4 Hz, BVP 64 Hz, ACC 32 Hz). labels.parquet is on the 4 Hz EDA
grid with a dataset-specific "phase" and a shared "label" (SHARED_CLASSES, or NA = excluded).
"""
import argparse
import io
import os
import pickle
import warnings
import zipfile

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_DIR, "..", "Datasets")
HARMONISED_DIR = os.path.join(DATASETS_DIR, "harmonised")
DATA_CARDS_DIR = os.path.join(PROJECT_DIR, "data_cards")

# Root of the local Nurses Stress dataset copy
file_path_nurses_stress = os.path.join(DATASETS_DIR, "Nurses_Stress")

# Root of the local WESAD copy; each subject has a folder S2..S17 (S1 and S12 don't exist).
file_path_wesad = os.path.join(DATASETS_DIR, "WESAD")

# PhysioNet: Wearable Device Dataset from Induced Stress and Structured Exercise Sessions
file_path_physionet = os.path.join(DATASETS_DIR, "PhysioNet_Wearable_Device")

for _p in (file_path_nurses_stress, file_path_wesad, file_path_physionet):
    if not os.path.exists(_p):
        raise FileNotFoundError(f"Path '{_p}' does not exist. Please check the path.")

E4_SIGNALS = ["ACC", "BVP", "EDA", "TEMP", "HR", "IBI"]
ACC_COLS = ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Empatica E4 CSV parsing (shared by all three datasets)
# ---------------------------------------------------------------------------
def _parse_start(value):
    """E4 start time: unix seconds (Nurses/WESAD) or a 'YYYY-MM-DD HH:MM:SS' string in UTC (PhysioNet)."""
    try:
        return float(value)
    except ValueError:
        return pd.Timestamp(str(value).strip(), tz="UTC").timestamp()


def _read_bytes(f):
    if isinstance(f, (str, os.PathLike)):
        with open(f, "rb") as fh:
            return fh.read()
    return f.read()


def parse_e4_csv(f, signal):
    """Parse one Empatica E4 CSV into a DataFrame with a UTC DatetimeIndex.

    `f` is a path or a file-like object.
    - ACC/BVP/EDA/TEMP/HR: row 0 = start time, row 1 = sample rate (Hz), then samples.
    - IBI: row 0 = start time, then (seconds since start, IBI in seconds).
    Returns an empty DataFrame if the file is empty or header-only.
    """
    content = _read_bytes(f)
    if signal == "IBI":
        try:
            raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        if len(raw) < 2:
            return pd.DataFrame()
        # Some IBI files concatenate several segments, each with its own "<start>, IBI" header.
        is_header = pd.to_numeric(raw.iloc[:, 1], errors="coerce").isna()
        seg_start = raw.iloc[:, 0].where(is_header).ffill().map(_parse_start)
        data = raw.loc[~is_header].astype(float)
        data.columns = ["t_offset", "IBI"]
        data.index = pd.to_datetime(seg_start[~is_header] + data["t_offset"], unit="s", utc=True)
        return data[["IBI"]].sort_index()

    lines = content.split(b"\n", 2)
    if len(lines) < 3 or not lines[2].strip():  # zero-byte or header-only file
        return pd.DataFrame()
    start = _parse_start(lines[0].decode().split(",")[0])
    fs = float(lines[1].decode().split(",")[0])
    data = pd.read_csv(io.BytesIO(lines[2]), header=None, dtype=np.float64)
    data.columns = ACC_COLS if signal == "ACC" else [signal]
    data.index = pd.to_datetime(start + np.arange(len(data)) / fs, unit="s", utc=True)
    data.attrs["fs"] = fs
    return data


def parse_e4_tags(f):
    """tags.csv: one timestamp per line (event button presses)."""
    try:
        tags = pd.read_csv(f, header=None)
    except pd.errors.EmptyDataError:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(tags.iloc[:, 0].map(_parse_start), unit="s", utc=True)


def load_e4_folder(folder, signals=E4_SIGNALS):
    """Load an unzipped E4 session folder -> {signal: DataFrame, 'tags': Series}."""
    out = {s: parse_e4_csv(os.path.join(folder, f"{s}.csv"), s) for s in signals}
    tags_path = os.path.join(folder, "tags.csv")
    if os.path.exists(tags_path):
        out["tags"] = parse_e4_tags(tags_path)
    return out


def load_e4_zip(zip_path, signals=E4_SIGNALS):
    """Load a zipped E4 session (as in the Nurses dataset) without extracting to disk."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for s in signals:
            if f"{s}.csv" in names:
                out[s] = parse_e4_csv(io.BytesIO(z.read(f"{s}.csv")), s)
        if "tags.csv" in names:
            out["tags"] = parse_e4_tags(io.BytesIO(z.read("tags.csv")))
    return out


# ---------------------------------------------------------------------------
# Raw loaders, one per dataset
# ---------------------------------------------------------------------------
# Nurses Stress: <subject>/<subject>_<unix start>.zip, one zip per session
def list_nurses_subjects(root=file_path_nurses_stress):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def list_nurses_sessions(subject, root=file_path_nurses_stress):
    folder = os.path.join(root, subject)
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".zip"))


# WESAD: <S#>/<S#>.pkl with chest (RespiBAN, 700 Hz) + wrist (E4) signals and labels.
# Per the WESAD readme, labels 5/6/7 should be ignored.
WESAD_LABELS = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
                5: "ignore", 6: "ignore", 7: "ignore"}
WESAD_WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
WESAD_LABEL_FS = 700


def list_wesad_subjects(root=file_path_wesad):
    subs = [d for d in os.listdir(root) if d.startswith("S") and os.path.isdir(os.path.join(root, d))]
    return sorted(subs, key=lambda s: int(s[1:]))


def load_wesad_subject(subject, root=file_path_wesad):
    """Return the raw WESAD dict: {'signal': {'chest': {...}, 'wrist': {...}}, 'label', 'subject'}.

    Each pickle is ~1 GB, so load one subject at a time.
    """
    with open(os.path.join(root, subject, f"{subject}.pkl"), "rb") as f:
        return pickle.load(f, encoding="latin1")


# PhysioNet: Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}/<folder>/<signal>.csv; metadata CSVs at the root.
# Folders are S01..S18 (protocol version 1) and f01..f18 (version 2); a few sessions are split
# into <id>_a / <id>_b folders (see data_constraints.txt).
PHYSIONET_PROTOCOLS = ["STRESS", "AEROBIC", "ANAEROBIC"]
physionet_signals_root = os.path.join(file_path_physionet, "Wearable_Dataset")


def list_physionet_folders(protocol, root=physionet_signals_root):
    folder = os.path.join(root, protocol)
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


def list_physionet_participants(root=physionet_signals_root):
    """Participant IDs with split-session suffixes (_a/_b) merged."""
    return sorted({f.split("_")[0] for p in PHYSIONET_PROTOCOLS for f in list_physionet_folders(p, root)})


def load_physionet_metadata(root=file_path_physionet):
    return {
        "subject_info": pd.read_csv(os.path.join(root, "subject-info.csv")),
        "stress_level_v1": pd.read_csv(os.path.join(root, "Stress_Level_v1.csv"), index_col=0),
        "stress_level_v2": pd.read_csv(os.path.join(root, "Stress_Level_v2.csv"), index_col=0),
    }


# ---------------------------------------------------------------------------
# Harmonised format (Protocol section 4)
# ---------------------------------------------------------------------------
# shared name: (E4 name, output columns, scale factor, unit)
SIGNAL_SPEC = {
    "eda": ("EDA", ["eda"], 1.0, "uS"),
    "temp": ("TEMP", ["temp"], 1.0, "degC"),
    "bvp": ("BVP", ["bvp"], 1.0, "a.u."),
    "acc": ("ACC", ["acc_x", "acc_y", "acc_z"], 1 / 64, "g"),  # E4 ACC is stored in 1/64 g
}
SHARED_CLASSES = ["baseline", "stress", "physical_activity", "other_arousal"]
LABEL_FS = 4.0  # labels live on the EDA grid
DATASETS = ["wesad", "physionet", "nurses"]

# Phase -> shared class. Phases not listed are excluded (label NA).
WESAD_PHASE_TO_CLASS = {"baseline": "baseline", "stress": "stress", "amusement": "other_arousal"}

PHYSIONET_PHASE_TO_CLASS = {
    "baseline": "baseline",
    "stroop": "stress", "tmct": "stress", "real_opinion": "stress",
    "opposite_opinion": "stress", "subtract": "stress",
    "warm_up": "physical_activity", "cycling": "physical_activity", "sprint": "physical_activity",
    # Excluded on purpose: first_rest/second_rest follow stress tasks (self-reported stress stays
    # high), and cool_down/rest follow exercise, so neither is a clean baseline.
}

NURSES_PHASE_TO_CLASS = {
    "no_event": "baseline",
    "event_stress_1": "stress", "event_stress_2": "stress",
    # Excluded: events the nurse rated 0 (no stress) or never rated ('na').
}
NURSES_TZ = "America/Chicago"  # survey times are local; this zone fits 305/358 events vs 58 for UTC


def _harmonise_signals(e4, session):
    """{E4 name: DataFrame} -> {shared name: DataFrame} with shared names, units and a session column."""
    out = {}
    for name, (src, cols, scale, _) in SIGNAL_SPEC.items():
        df = e4.get(src)
        if df is None or df.empty:
            continue
        df = (df * scale).astype("float32")
        df.columns = cols
        df.index.name = "time"
        df["session"] = session
        out[name] = df
    return out


def _label_frame(index, intervals, phase_to_class, default, session):
    """Assign a phase to every sample of a sorted index from (t0, t1, phase) intervals; later ones win."""
    phase = np.full(len(index), default, dtype=object)
    for t0, t1, ph in intervals:
        phase[index.searchsorted(t0):index.searchsorted(t1)] = ph
    df = pd.DataFrame({"phase": phase}, index=index)
    df["label"] = df["phase"].map(phase_to_class)
    df["session"] = session
    return df


def _concat_sessions(parts):
    """[{name: DataFrame}] per session -> {name: DataFrame} with categorical columns."""
    out = {}
    for name in list(SIGNAL_SPEC) + ["labels"]:
        dfs = [p[name] for p in parts if name in p]
        if not dfs:
            continue
        df = pd.concat(dfs).sort_index(kind="stable")
        df.index.name = "time"
        df["session"] = df["session"].astype("category")
        if name == "labels":
            df["phase"] = df["phase"].astype("category")
            df["label"] = pd.Categorical(df["label"], categories=SHARED_CLASSES)
        out[name] = df
    return out


# ---------------------------------------------------------------------------
# WESAD
# ---------------------------------------------------------------------------
def _locate(raw, segment, k=400):
    """Index where `segment` starts inside `raw` (both 1-D), or None if not found exactly once."""
    k = min(k, len(segment))
    win = np.lib.stride_tricks.sliding_window_view(raw, k)
    hits = np.flatnonzero(np.all(np.isclose(win, segment[:k], atol=1e-4), axis=1))
    return int(hits[0]) if len(hits) == 1 else None


def harmonise_wesad(subject, root=file_path_wesad):
    """The pickle holds a cropped, synchronised segment of the raw E4 recording without timestamps.
    Its UTC start is recovered by locating the pickle's EDA inside the raw E4 EDA.csv."""
    notes = []
    d = load_wesad_subject(subject, root)
    wrist = d["signal"]["wrist"]
    raw_dir = os.path.join(root, subject, f"{subject}_E4_Data")

    raw_eda = parse_e4_csv(os.path.join(raw_dir, "EDA.csv"), "EDA")
    i = _locate(raw_eda["EDA"].to_numpy(), wrist["EDA"][:, 0])
    if i is None:
        raise ValueError(f"WESAD {subject}: could not locate the pickle segment in the raw E4 EDA")
    start = raw_eda.index[i]

    # Cross-check with ACC, which has 8x the EDA rate.
    raw_acc = parse_e4_csv(os.path.join(raw_dir, "ACC.csv"), "ACC")
    j = _locate(raw_acc["x"].to_numpy(), wrist["ACC"][:, 0])
    if j is None or abs((raw_acc.index[j] - start).total_seconds()) > 0.25:
        notes.append("ACC offset does not match EDA offset; timestamps may be off by <1 s")

    e4 = {}
    for sig, fs in WESAD_WRIST_FS.items():
        arr = wrist[sig]
        cols = ACC_COLS if sig == "ACC" else [sig]
        idx = start + pd.to_timedelta(np.arange(len(arr)) / fs, unit="s")
        e4[sig] = pd.DataFrame(arr, index=idx, columns=cols)
    sig = _harmonise_signals(e4, session="lab")

    step = int(WESAD_LABEL_FS / LABEL_FS)  # 700 Hz -> 4 Hz
    codes = d["label"][::step][: len(sig["eda"])]
    labels = pd.DataFrame({"phase": pd.Series(codes).map(WESAD_LABELS).to_numpy()}, index=sig["eda"].index)
    labels["label"] = labels["phase"].map(WESAD_PHASE_TO_CLASS)
    labels["session"] = "lab"
    sig["labels"] = labels
    return _concat_sessions([sig]), notes


# ---------------------------------------------------------------------------
# PhysioNet
# ---------------------------------------------------------------------------
# S02 STRESS: E4 Connect duplicated the data; keep only the rows before the repeat
# (data_constraints.txt gives file rows 49,545 / 99,091 / 6,195, i.e. 3 more than these data rows).
PHYSIONET_TRUNCATE = {("STRESS", "S02"): {"ACC": 49542, "BVP": 99088, "EDA": 6192, "TEMP": 6192}}
# f07 STRESS: the protection dock covered the PPG and temperature sensors.
PHYSIONET_DROP = {("STRESS", "f07"): ["BVP", "TEMP"]}
# Untagged phase durations (s) from the protocol diagrams in Wearable_Dataset.ipynb.
V1_WARM_UP = 180
V2_STRESS_BASELINE = 180  # not given in the diagram; matches the 3:00 V1 baseline (assumption)
V2_AEROBIC_BASELINE, V2_AEROBIC_WARM_UP = 270, 135
V2_ANAEROBIC_BASELINE = 270


def _physionet_intervals(protocol, version, part, tags, start, end):
    """Phase intervals from tags.csv, following the protocol diagrams in Wearable_Dataset.ipynb.

    `part` is None for a whole session, or 'a'/'b' for the halves of a split session.
    Tags are numbered from 1 as in the diagrams. Gaps between tagged phases stay 'transition'.
    """
    s = pd.Timedelta(seconds=1)
    t = list(tags)
    n = len(t)

    def T(k):
        return t[k - 1]

    iv = []
    if protocol == "STRESS":
        if part == "a":  # f14_a: only the baseline was recorded, no tags
            return [(end - V2_STRESS_BASELINE * s, end, "baseline")]
        if version == 1:  # 13 tags; the 13th marks the end of the session
            spans = [(1, 2, "baseline"), (3, 4, "stroop"), (4, 5, "first_rest"), (5, 6, "tmct"),
                     (6, 7, "second_rest"), (7, 8, "real_opinion"), (9, 10, "opposite_opinion"),
                     (11, 12, "subtract")]
        else:  # 9 tags; the baseline ends at tag 1
            spans = [(2, 3, "tmct"), (3, 4, "first_rest"), (4, 5, "real_opinion"),
                     (6, 7, "opposite_opinion"), (7, 8, "second_rest"), (8, 9, "subtract")]
            if part is None:
                iv.append((T(1) - V2_STRESS_BASELINE * s, T(1), "baseline"))
        iv += [(T(a), T(b), ph) for a, b, ph in spans if b <= n]

    elif protocol == "AEROBIC":
        if version == 1:  # warm-up, cycling stages t1..t(n-1), cool-down, rest (some ended early)
            if part == "b":  # S11_b: the last 110 rpm block, then recovery
                iv += [(start, T(1), "cycling"), (T(1), end, "cool_down")]
            else:
                iv.append((T(1) - V1_WARM_UP * s, T(1), "warm_up"))
                if part == "a":  # S11_a: cycling until the connection dropped
                    iv.append((T(1), end, "cycling"))
                else:
                    iv += [(T(k), T(k + 1), "cycling") for k in range(1, n - 1)]
                    iv += [(T(n - 1), T(n), "cool_down"), (T(n), end, "rest")]
        else:  # 8 tags; baseline and warm-up both precede tag 1
            w0 = T(1) - V2_AEROBIC_WARM_UP * s
            iv += [(w0 - V2_AEROBIC_BASELINE * s, w0, "baseline"), (w0, T(1), "warm_up"),
                   (T(1), T(6), "cycling"), (T(6), T(7), "cool_down"), (T(7), T(8), "rest")]

    elif protocol == "ANAEROBIC":
        if version == 1:  # warm-up, then sprint / cool-down alternating from tag 1, rest
            iv.append((start, T(1), "cool_down") if part == "b" else (T(1) - V1_WARM_UP * s, T(1), "warm_up"))
            iv += [(T(k), T(k + 1), "sprint" if k % 2 == 1 else "cool_down") for k in range(1, n)]
            iv.append((T(n), end, "cool_down" if part == "a" else "rest"))
        else:  # 11 tags
            iv += [(T(1) - V2_ANAEROBIC_BASELINE * s, T(1), "baseline"), (T(1), T(2), "warm_up")]
            iv += [(T(k), T(k + 1), "sprint" if k % 2 == 0 else "cool_down") for k in range(2, 10)]
            iv.append((T(10), T(11), "rest"))
    return iv


# Expected tag counts per (protocol, version) for unsplit sessions; v1 exercise may end early.
PHYSIONET_EXPECTED_TAGS = {("STRESS", 1): {13}, ("STRESS", 2): {9}, ("AEROBIC", 1): set(range(7, 13)),
                           ("AEROBIC", 2): {8}, ("ANAEROBIC", 1): {5, 7}, ("ANAEROBIC", 2): {11}}


def harmonise_physionet(pid, root=physionet_signals_root):
    notes = []
    version = 1 if pid.startswith("S") else 2
    parts = []
    for protocol in PHYSIONET_PROTOCOLS:
        folders = [f for f in list_physionet_folders(protocol, root) if f.split("_")[0] == pid]
        for folder in folders:
            part = folder.split("_")[1] if "_" in folder else None
            e4 = load_e4_folder(os.path.join(root, protocol, folder), ["ACC", "BVP", "EDA", "TEMP"])
            for sig, n in PHYSIONET_TRUNCATE.get((protocol, pid), {}).items():
                e4[sig] = e4[sig].iloc[:n]
                notes.append(f"{protocol}: duplicated {sig} rows removed")
            for sig in PHYSIONET_DROP.get((protocol, pid), []):
                e4[sig] = pd.DataFrame()
                notes.append(f"{protocol}: {sig} dropped (sensor covered)")
            session = protocol if part is None else f"{protocol}_{part}"
            sig = _harmonise_signals(e4, session)
            if "eda" not in sig:
                notes.append(f"{session}: no EDA, session skipped")
                continue

            tags = e4.get("tags", pd.Series(dtype="datetime64[ns, UTC]")).sort_values().tolist()
            idx = sig["eda"].index
            if part is None and len(tags) not in PHYSIONET_EXPECTED_TAGS[(protocol, version)]:
                notes.append(f"{session}: unexpected tag count {len(tags)}, left unlabelled")
                iv = []
            else:
                iv = _physionet_intervals(protocol, version, part, tags, idx[0], idx[-1] + pd.Timedelta(seconds=1))
            sig["labels"] = _label_frame(idx, iv, PHYSIONET_PHASE_TO_CLASS, "transition", session)
            parts.append(sig)
    return _concat_sessions(parts), notes


# ---------------------------------------------------------------------------
# Nurses Stress
# ---------------------------------------------------------------------------
def load_nurses_events(root=file_path_nurses_stress):
    """SurveyResults.xlsx -> one row per detected event with UTC start/end and a phase name."""
    sv = pd.read_excel(os.path.join(root, "SurveyResults.xlsx"))
    sv["ID"] = sv["ID"].astype(str)

    def local(times):
        return pd.DatetimeIndex([pd.Timestamp.combine(d.date(), t) for d, t in zip(sv["date"], times)])

    t0, t1 = local(sv["Start time"]), local(sv["End time"])
    t1 = t1.where(t1 > t0, t1 + pd.Timedelta(days=1))  # events crossing midnight
    loc = dict(ambiguous="NaT", nonexistent="NaT")
    level = sv["Stress level"].astype(str)
    return pd.DataFrame({
        "participant": sv["ID"],
        "start": t0.tz_localize(NURSES_TZ, **loc).tz_convert("UTC"),
        "end": t1.tz_localize(NURSES_TZ, **loc).tz_convert("UTC"),
        "phase": level.map({"0": "event_no_stress", "1": "event_stress_1", "2": "event_stress_2"})
                      .fillna("event_unrated"),
    }).dropna(subset=["start", "end"])


def harmonise_nurses(subject, events, root=file_path_nurses_stress):
    """Each zip is one session. Survey events are labelled by their rating; all other time is 'no_event'.
    Unrated / no-stress events are applied first, so a stress event wins where events overlap."""
    notes = []
    ev = events[events["participant"] == subject]
    ev = ev.assign(order=ev["phase"].str.startswith("event_stress")).sort_values(["order", "start"])
    intervals = list(ev[["start", "end", "phase"]].itertuples(index=False, name=None))
    parts, matched = [], set()
    for zp in list_nurses_sessions(subject, root):
        session = os.path.splitext(os.path.basename(zp))[0]
        sig = _harmonise_signals(load_e4_zip(zp, ["ACC", "BVP", "EDA", "TEMP"]), session)
        if "eda" not in sig:
            notes.append(f"{session}: no EDA, session skipped")
            continue
        idx = sig["eda"].index
        inside = [k for k, (a, b, _) in enumerate(intervals) if a < idx[-1] and b > idx[0]]
        matched.update(inside)
        sig["labels"] = _label_frame(idx, [intervals[k] for k in inside], NURSES_PHASE_TO_CLASS, "no_event", session)
        parts.append(sig)
    notes.append(f"{len(matched)}/{len(intervals)} survey events overlap a recording")
    out = _concat_sessions(parts)
    n_dup = int(out["eda"].index.duplicated().sum()) if "eda" in out else 0
    if n_dup:
        notes.append(f"{n_dup} EDA samples in overlapping sessions")
    return out, notes


# ---------------------------------------------------------------------------
# Build, save, load (the single interface for later phases)
# ---------------------------------------------------------------------------
def participant_dir(dataset, pid):
    return os.path.join(HARMONISED_DIR, dataset, pid)


def list_harmonised(dataset):
    path = os.path.join(HARMONISED_DIR, dataset, "participants.csv")
    return pd.read_csv(path, dtype={"participant": str})["participant"].tolist()


def load_harmonised(dataset, pid, tables=("eda", "temp", "bvp", "acc", "labels")):
    """Load one harmonised participant -> {table: DataFrame}. Missing signals are skipped."""
    out = {}
    for t in tables:
        path = os.path.join(participant_dir(dataset, pid), f"{t}.parquet")
        if os.path.exists(path):
            out[t] = pd.read_parquet(path)
    return out


def _summarise(dataset, pid, rec, notes):
    lab = rec["labels"]
    h = 1 / LABEL_FS / 3600
    row = {"dataset": dataset, "participant": pid, "sessions": lab["session"].nunique(),
           "hours": round(len(lab) * h, 2)}
    for name in SIGNAL_SPEC:
        row[f"has_{name}"] = name in rec
    for c in SHARED_CLASSES:
        row[f"h_{c}"] = round(int((lab["label"] == c).sum()) * h, 3)
    row["h_excluded"] = round(int(lab["label"].isna().sum()) * h, 3)
    row["notes"] = "; ".join(notes)
    phases = (lab.groupby("phase", observed=True).size() * h).rename(pid)
    return row, phases


def harmonise(dataset, overwrite=False):
    if dataset == "wesad":
        pids, build = list_wesad_subjects(), harmonise_wesad
    elif dataset == "physionet":
        pids, build = list_physionet_participants(), harmonise_physionet
    elif dataset == "nurses":
        events = load_nurses_events()
        pids, build = list_nurses_subjects(), lambda p: harmonise_nurses(p, events)
    else:
        raise ValueError(dataset)

    table_path = os.path.join(HARMONISED_DIR, dataset, "participants.csv")
    old_notes = {}
    if os.path.exists(table_path):  # cached participants keep the notes from their build
        old = pd.read_csv(table_path, dtype={"participant": str}, keep_default_na=False)
        old_notes = dict(zip(old["participant"], old["notes"]))

    rows, phase_tables = [], []
    for pid in pids:
        out_dir = participant_dir(dataset, pid)
        cached = os.path.exists(os.path.join(out_dir, "labels.parquet")) and not overwrite
        if cached:  # only labels are needed for the summary; signals are just checked for existence
            rec = load_harmonised(dataset, pid, ["labels"])
            rec.update({s: None for s in SIGNAL_SPEC if os.path.exists(os.path.join(out_dir, f"{s}.parquet"))})
            notes = [old_notes[pid]] if old_notes.get(pid) else []
        else:
            rec, notes = build(pid)
            os.makedirs(out_dir, exist_ok=True)
            for name, df in rec.items():
                df.to_parquet(os.path.join(out_dir, f"{name}.parquet"))
        row, phases = _summarise(dataset, pid, rec, notes)
        print(f"  {dataset}/{pid}: {row['hours']} h, {row['sessions']} session(s){' [cached]' if cached else ''}")
        rows.append(row)
        phase_tables.append(phases)

    table = pd.DataFrame(rows)
    phase_hours = pd.concat(phase_tables, axis=1).fillna(0).T
    table.to_csv(table_path, index=False)
    write_data_card(dataset, table, phase_hours)
    return table


# ---------------------------------------------------------------------------
# Data cards
# ---------------------------------------------------------------------------
DATA_CARD_INFO = {
    "wesad": {
        "title": "WESAD (Schmidt et al., 2018)",
        "expected": "15 participants (S2-S17; S1 and S12 excluded by the authors)",
        "source": "Wrist (E4) signals and 700 Hz labels from each S#.pkl; UTC start recovered by matching "
                  "the pickle's EDA to the raw S#_E4_Data/EDA.csv.",
        "phase_map": WESAD_PHASE_TO_CLASS,
        "issues": ["Labels downsampled from 700 Hz to 4 Hz by taking every 175th sample.",
                   "Meditation, transient (0) and ignore (5-7) phases are excluded."],
    },
    "physionet": {
        "title": "PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions",
        "expected": "36 participants: stress 36, aerobic 30, anaerobic 31 (folder counts are higher "
                    "because f14, S11 and S16 have split _a/_b sessions)",
        "source": "Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}; phases from tags.csv using the protocol "
                  "diagrams in Wearable_Dataset.ipynb. S## = protocol version 1, f## = version 2.",
        "phase_map": PHYSIONET_PHASE_TO_CLASS,
        "issues": ["S02 STRESS: duplicated rows removed (data_constraints.txt).",
                   "f07 STRESS: BVP and TEMP dropped (protection dock covered the sensors).",
                   "Split sessions (f14 STRESS, S11 AEROBIC, S16 ANAEROBIC) labelled per half.",
                   f"Untagged phases use diagram durations: v1 warm-up {V1_WARM_UP}s; v2 aerobic baseline "
                   f"{V2_AEROBIC_BASELINE}s + warm-up {V2_AEROBIC_WARM_UP}s; v2 anaerobic baseline "
                   f"{V2_ANAEROBIC_BASELINE}s; v2 stress baseline {V2_STRESS_BASELINE}s (assumed, not in the diagram).",
                   "Rest periods after stress tasks and cool-down/rest after exercise are excluded, not baseline."],
    },
    "nurses": {
        "title": "Nurse stress dataset (Hosseini et al., 2022)",
        "expected": "15 nurses, ~1,250 h in total",
        "source": "One zip per E4 session; stress events from SurveyResults.xlsx (local times, "
                  f"converted from {NURSES_TZ} to UTC).",
        "phase_map": NURSES_PHASE_TO_CLASS,
        "issues": ["Events rated 0 (no stress) or never rated are excluded.",
                   "All time outside a survey event is labelled baseline ('no_event').",
                   "Where events overlap, a stress-rated event takes precedence."],
    },
}


def _md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write_data_card(dataset, table, phase_hours):
    info = DATA_CARD_INFO[dataset]
    total = phase_hours.sum().round(2)
    phase_df = pd.DataFrame({"phase": total.index, "hours": total.values,
                             "shared class": [info["phase_map"].get(p, "excluded") for p in total.index]})
    class_hours = {c: round(table[f"h_{c}"].sum(), 2) for c in SHARED_CLASSES}
    class_hours["excluded"] = round(table["h_excluded"].sum(), 2)
    units = ", ".join(f"{k} ({v[3]})" for k, v in SIGNAL_SPEC.items())

    md = [f"# Data card: {info['title']}", "",
          "_Generated by `study_one.py harmonise`; do not edit by hand._", "",
          "## Summary", "",
          f"- **Participants harmonised:** {len(table)} (source paper: {info['expected']})",
          f"- **Recorded hours (EDA):** {table['hours'].sum():.1f}",
          f"- **Signals:** {units}; native E4 rates, UTC timestamps",
          f"- **Source:** {info['source']}", "",
          "## Hours per shared class", "",
          _md_table(pd.DataFrame([class_hours])), "",
          "## Hours per phase", "",
          _md_table(phase_df.sort_values("hours", ascending=False)), "",
          "## Processing notes", ""]
    md += [f"- {i}" for i in info["issues"]]
    md += ["", "## Participants", "", _md_table(table.drop(columns="dataset")), ""]

    os.makedirs(DATA_CARDS_DIR, exist_ok=True)
    with open(os.path.join(DATA_CARDS_DIR, f"{dataset}.md"), "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")  # no subcommand (e.g. VS Code's Run button) -> harmonise all
    h = sub.add_parser("harmonise", help="build harmonised Parquet files and data cards")
    h.add_argument("datasets", nargs="*", help=f"any of {DATASETS} (default: all)")
    h.add_argument("--overwrite", action="store_true", help="rebuild participants that already exist")
    args = ap.parse_args()
    if args.cmd is None:
        args = ap.parse_args(["harmonise"])
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        ap.error(f"unknown dataset(s): {sorted(unknown)}; choose from {DATASETS}")

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    for ds in args.datasets or DATASETS:
        print(f"=== {ds}")
        t = harmonise(ds, overwrite=args.overwrite)
        cols = ["hours"] + [f"h_{c}" for c in SHARED_CLASSES] + ["h_excluded"]
        print(f"{len(t)} participants; totals (h):", t[cols].sum().round(2).to_dict())
