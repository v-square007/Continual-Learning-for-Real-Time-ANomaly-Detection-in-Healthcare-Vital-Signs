#v6
"""
=============================================================================
Continual Learning — Real-Time Anomaly Detection on Healthcare Vital Signals
Version 6 — Unified Scoring + Clear Training Strategy + Simple Drift + Clean Eval
=============================================================================

KEY CHANGES FROM V5:
─────────────────────────────────────────────────────────────────────────────
1. UNIFIED SCORING MECHANISM (WindowScore class):
   - Combines anomaly_score, data quality, and uncertainty into one framework
   - weighted_score = anomaly_score × trust_score
   - trust_score = quality × (1 - uncertainty)
   - Key insight: uncertainty REDUCES score (need both "unusual" AND "trustworthy")

2. CLEAR TRAINING STRATEGY (MAD filter):
   - No vague "stability" filtering
   - Use MAD (Median Absolute Deviation) on HR mean values
   - 1.4826 × MAD ≈ SD for Gaussian data
   - Cutoff: 2.5 MAD (robust to outliers, median doesn't get corrupted)

3. SIMPLE DRIFT DETECTION (rolling ratio):
   - Push weighted_score into buffer (size 50)
   - drift_ratio = current_mean / warmup_mean
   - ratio > 1.3 → scores drifting up (possible distribution shift)
   - No complex modeling, just one division and comparison

4. CLEAN EVALUATION (4 proxy checks):
   - CV (coefficient of variation) → noise sensitivity
   - Separation ratio → threshold calibration
   - Perturbation sensitivity → response to real changes
   - Temporal consistency → artefact detection
=============================================================================
"""
import os, time, warnings
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import vitaldb
    VITALDB_AVAILABLE = True
    print("[OK]  vitaldb imported successfully.")
except ImportError:
    VITALDB_AVAILABLE = False
    print("[WARN] vitaldb not found — running with physiological simulation.\n")


# =============================================================================
# CONFIG
# =============================================================================

NUMERIC_TRACKS = {
    "HR":   "Solar8000/HR",
    "SpO2": "Solar8000/PLETH_SPO2",
    "RR":   "Solar8000/RR_CO2",
    "SBP":  "Solar8000/NIBP_SBP",
    "DBP":  "Solar8000/NIBP_DBP",
    "BT":   "Solar8000/BT",
}

WAVEFORM_TRACKS = {
    "ECG":   "SNUADC/ECG_II",
    "PLETH": "SNUADC/PLETH",
}

PHYS_BOUNDS = {
    "HR":   (20,  300),
    "SpO2": (50,  100),
    "RR":   (2,   80),
    "SBP":  (40,  260),
    "DBP":  (20,  180),
    "BT":   (32,  43),
}

SAMPLE_RATE_HZ  = 1
WINDOW_SIZE_S   = 10
OVERLAP         = 0.50
STEP_SIZE_S     = int(WINDOW_SIZE_S * (1 - OVERLAP))

N_ESTIMATORS        = 100
CONTAMINATION       = 0.05
THRESHOLD_PCTILE    = 95
WARMUP_WINDOWS      = 60

# Missing value handling
MAX_MISSING_FRACTION = 0.95
MAX_FFILL_GAP_S      = 60

# Quality thresholds
MAX_NAN_SIGNAL_FRACTION = 0.50  # Skip window if >50% signals are NaN
QUALITY_DEGRADATION_THRESHOLD = 0.30  # quality < 0.3 → high uncertainty

# MAD filter for warm-up selection
MAD_CUTOFF = 2.5  # Median Absolute Deviation units (≈ 2.5 SD)

# Drift detection
DRIFT_BUFFER_SIZE = 50
DRIFT_RATIO_THRESHOLD = 1.3  # Current/warmup mean > 1.3 → drift warning

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 — WindowScore: Unified Scoring Mechanism
# =============================================================================

@dataclass
class WindowScore:
    """
    Unified scoring framework combining anomaly detection, data quality,
    and uncertainty into a single decision score.

    Fields:
    -------
    anomaly_score : float
        Raw score from Isolation Forest (higher = more anomalous)

    quality : float in [0, 1]
        Data completeness: 1.0 = all signals present, 0.0 = all missing
        Computed as: 1 - (fraction of NaN signals in window)

    uncertainty : float in [0, 1]
        Confidence in the anomaly score. Higher uncertainty when:
          - Data quality is poor (many missing signals)
          - Feature values are outside training distribution range
        Computed as: max(0, 1 - quality / QUALITY_DEGRADATION_THRESHOLD)

    trust_score : float in [0, 1]
        Derived: quality × (1 - uncertainty)
        Measures how much we should trust this window's anomaly score

    weighted_score : float
        Final decision score: anomaly_score × trust_score
        Key insight: need BOTH unusual AND trustworthy data for an alert
        Low-quality windows get downweighted even if anomaly_score is high
    """
    anomaly_score: float
    quality: float
    uncertainty: float

    @property
    def trust_score(self) -> float:
        """Trust = quality × (1 - uncertainty)"""
        return self.quality * (1.0 - self.uncertainty)

    @property
    def weighted_score(self) -> float:
        """Final score = anomaly × trust"""
        return self.anomaly_score * self.trust_score

    def __repr__(self):
        return (f"WindowScore(anomaly={self.anomaly_score:.3f}, "
                f"quality={self.quality:.2f}, uncertainty={self.uncertainty:.2f}, "
                f"trust={self.trust_score:.2f}, weighted={self.weighted_score:.3f})")


def compute_window_quality(nan_sig_fraction: float) -> float:
    """
    Quality metric: 1.0 if all signals present, 0.0 if all missing.

    quality = 1 - nan_sig_fraction

    Examples:
      0% NaN → quality = 1.0
      30% NaN → quality = 0.7
      80% NaN → quality = 0.2
    """
    return 1.0 - nan_sig_fraction


def compute_window_uncertainty(quality: float) -> float:
    """
    Uncertainty increases as quality degrades.

    uncertainty = max(0, 1 - quality / QUALITY_DEGRADATION_THRESHOLD)

    With QUALITY_DEGRADATION_THRESHOLD = 0.3:
      quality ≥ 0.3 → uncertainty = 0 (confident)
      quality = 0.15 → uncertainty = 0.5 (moderate doubt)
      quality = 0.0 → uncertainty = 1.0 (complete uncertainty)

    This means: if quality drops below 30%, we start losing confidence.
    """
    if quality >= QUALITY_DEGRADATION_THRESHOLD:
        return 0.0
    return max(0.0, 1.0 - quality / QUALITY_DEGRADATION_THRESHOLD)


# =============================================================================
# SECTION 2 — VitalDB Data Fetcher (unchanged)
# =============================================================================

def fetch_case(caseid: int, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            frames = {}

            for sig, tname in NUMERIC_TRACKS.items():
                data = vitaldb.vital_recs(
                    caseid,
                    track_names=[tname],
                    interval=SAMPLE_RATE_HZ,
                    return_timestamp=True,
                )
                if data is None or len(data) == 0:
                    print(f"    [SKIP] case {caseid}: {tname} → no data")
                    continue

                df_sig = pd.DataFrame(data, columns=["time", sig])
                df_sig["time"] = df_sig["time"].astype(float).round().astype(int)
                df_sig = df_sig.drop_duplicates(subset="time").set_index("time")[sig]
                frames[sig] = df_sig
                print(f"    [OK] case {caseid}: {sig} → {len(df_sig)} rows")

            for sig, tname in WAVEFORM_TRACKS.items():
                data = vitaldb.vital_recs(
                    caseid,
                    track_names=[tname],
                    return_timestamp=True,
                )
                if data is None or len(data) == 0:
                    print(f"    [SKIP] case {caseid}: {tname} → no data")
                    continue

                df_wave = pd.DataFrame(data, columns=["time", sig])
                df_wave["time"] = df_wave["time"].astype(float)
                df_feat = _summarise_waveform(df_wave, sig)
                for col in df_feat.columns:
                    frames[col] = df_feat[col]
                print(f"    [OK] case {caseid}: {sig} waveform → "
                      f"{len(df_feat)} 1-Hz summary rows")

            if not frames:
                print(f"  [WARN] case {caseid}: no tracks loaded")
                return pd.DataFrame()

            df = pd.concat(frames.values(), axis=1, join="outer")
            df.index.name = "time"

            if len(df) > 0:
                full_range = pd.RangeIndex(
                    start=int(df.index.min()),
                    stop=int(df.index.max()) + 1,
                    step=1
                )
                df = df.reindex(full_range)

            df = df.reset_index()
            df.rename(columns={"index": "time"}, inplace=True)
            df["caseid"] = caseid

            print(f"  [DONE] case {caseid}: {len(df)} rows on unified timeline")
            return df

        except Exception as exc:
            print(f"  [ERROR] case {caseid} attempt {attempt}: {exc}")
            time.sleep(3)

    return pd.DataFrame()


def _summarise_waveform(df_wave: pd.DataFrame, sig: str) -> pd.DataFrame:
    df_wave = df_wave.copy()
    df_wave["second"] = np.floor(df_wave["time"]).astype(int)
    grp = df_wave.groupby("second")[sig]
    feat = pd.DataFrame({
        f"{sig}_mean": grp.mean(),
        f"{sig}_std":  grp.std().fillna(0),
        f"{sig}_p2p":  grp.max() - grp.min(),
        f"{sig}_rms":  grp.apply(lambda x: float(np.sqrt(np.mean(x**2)))),
    })
    feat.index.name = "time"
    return feat


def load_real_data(case_ids=(1, 2, 3), min_rows=200) -> pd.DataFrame:
    frames = []
    for cid in case_ids:
        df = fetch_case(cid)
        if len(df) >= min_rows:
            frames.append(df)
        else:
            print(f"  [SKIP] case {cid}: only {len(df)} rows after cleaning")
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        print(f"\n[INFO] Total rows loaded: {len(combined)} "
              f"from {len(frames)} cases")
        return combined
    return pd.DataFrame()


# =============================================================================
# SECTION 3 — Simulator
# =============================================================================

def simulate_vitaldb_like(n_seconds=1800, seed=42) -> pd.DataFrame:
    """Simulates realistic surgical vital signs with induction phase and drift."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n_seconds, dtype=float)

    hr   = 72  + 5  * np.sin(2*np.pi*t/120)  + rng.normal(0, 2,    n_seconds)
    spo2 = 98  + 0.5* np.sin(2*np.pi*t/90)   + rng.normal(0, 0.3,  n_seconds)
    rr   = 16  + 2  * np.sin(2*np.pi*t/60)   + rng.normal(0, 1,    n_seconds)
    sbp  = 120 + 8  * np.sin(2*np.pi*t/180)  + rng.normal(0, 3,    n_seconds)
    dbp  = 80  + 5  * np.sin(2*np.pi*t/180)  + rng.normal(0, 2,    n_seconds)
    bt   = 36.6+ 0.2* np.sin(2*np.pi*t/300)  + rng.normal(0, 0.05, n_seconds)

    ecg_mean  = rng.normal(0,    1,    n_seconds)
    ecg_std   = np.abs(rng.normal(0.3, 0.05, n_seconds))
    ecg_p2p   = np.abs(rng.normal(1.5, 0.2,  n_seconds))
    ecg_rms   = np.abs(rng.normal(0.5, 0.05, n_seconds))
    pleth_mean= np.abs(rng.normal(50,  5,    n_seconds))
    pleth_std = np.abs(rng.normal(5,   1,    n_seconds))
    pleth_p2p = np.abs(rng.normal(30,  3,    n_seconds))
    pleth_rms = np.abs(rng.normal(52,  5,    n_seconds))

    # Induction phase instability (first 10%)
    ind_end = int(n_seconds * 0.10)
    hr  [:ind_end] += rng.normal(30, 12, ind_end)
    sbp [:ind_end] -= rng.normal(20, 10, ind_end)
    spo2[:ind_end] -= rng.uniform(2, 8,  ind_end)

    # Drift in last 1/3
    ds = int(n_seconds * 2/3)
    hr  [ds:] += rng.normal(20, 8,  n_seconds-ds)
    spo2[ds:] -= rng.uniform(3,  6, n_seconds-ds)
    sbp [ds:] += rng.normal(25, 10, n_seconds-ds)
    rr  [ds:] += rng.normal(8,  3,  n_seconds-ds)
    ecg_p2p[ds:] *= rng.uniform(1.5, 2.5, n_seconds-ds)
    pleth_std[ds:] *= rng.uniform(2, 4,   n_seconds-ds)

    hr   = np.clip(hr,   20,  300)
    spo2 = np.clip(spo2, 50,  100)
    rr   = np.clip(rr,   2,   80)
    sbp  = np.clip(sbp,  40,  260)
    dbp  = np.clip(dbp,  20,  180)
    bt   = np.clip(bt,   32,  43)

    df = pd.DataFrame({
        "time": t.astype(int), "caseid": 0,
        "HR": hr, "SpO2": spo2, "RR": rr,
        "SBP": sbp, "DBP": dbp, "BT": bt,
        "ECG_mean": ecg_mean, "ECG_std": ecg_std,
        "ECG_p2p": ecg_p2p,   "ECG_rms": ecg_rms,
        "PLETH_mean": pleth_mean, "PLETH_std": pleth_std,
        "PLETH_p2p": pleth_p2p,   "PLETH_rms": pleth_rms,
    })
    print(f"[SIM] {n_seconds}s simulated — induction instability + drift injected.")
    return df


# =============================================================================
# SECTION 4 — Missing Value Handling
# =============================================================================

def handle_missing(df: pd.DataFrame) -> tuple:
    """Handle missing values with physiological bounds clipping and imputation."""
    df = df.copy()
    non_meta = [c for c in df.columns if c not in ("time", "caseid")]

    print("\n── Missing Value Report (BEFORE) ──────────────────────────")
    for col in non_meta:
        n = df[col].isna().sum()
        pct = n / len(df) * 100
        print(f"  {col:<20s} {pct:>5.1f}%  ({n:,} NaN)")

    # Physiological bounds clipping
    for col in non_meta:
        base = col.split("_")[0]
        if base in PHYS_BOUNDS:
            lo, hi = PHYS_BOUNDS[base]
            before = df[col].notna().sum()
            df.loc[~df[col].between(lo, hi, inclusive="both") &
                   df[col].notna(), col] = np.nan
            clipped = before - df[col].notna().sum()
            if clipped > 0:
                print(f"  [CLIP] {col}: {clipped} values outside [{lo}, {hi}] → NaN")

    # Imputation: forward fill → backward fill → interpolation
    df[non_meta] = df[non_meta].ffill(limit=MAX_FFILL_GAP_S)
    df[non_meta] = df[non_meta].bfill(limit=MAX_FFILL_GAP_S)
    df[non_meta] = df[non_meta].interpolate(
        method="linear", limit=MAX_FFILL_GAP_S, limit_direction="both"
    )

    print("\n── Missing Value Report (AFTER) ───────────────────────────")
    active_sigs = []
    dropped = []
    for col in non_meta:
        n = df[col].isna().sum()
        pct = n / len(df) * 100
        if pct > MAX_MISSING_FRACTION * 100:
            print(f"  [DROP] {col:<20s} {pct:.1f}% still missing → removed")
            dropped.append(col)
        else:
            status = "✓ clean" if n == 0 else f"{n} NaN remaining"
            print(f"  [KEEP] {col:<20s} {status}")
            active_sigs.append(col)

    if dropped:
        df.drop(columns=dropped, inplace=True)

    df.dropna(subset=active_sigs, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n  Rows after cleaning : {len(df):,}")
    print(f"  Active signals      : {len(active_sigs)}")
    print("─" * 65)

    return df, active_sigs


# =============================================================================
# SECTION 5 — Feature Extraction with Quality Tracking
# =============================================================================

class FeatureMedianBuffer:
    """Maintains rolling median for masked feature imputation."""
    def __init__(self, maxlen=200):
        self._buf = deque(maxlen=maxlen)
        self._medians = None

    def update(self, feat_vec: np.ndarray):
        """Add clean feature vector."""
        if not np.any(np.isnan(feat_vec)):
            self._buf.append(feat_vec.copy())
            if len(self._buf) >= 5:
                self._medians = np.median(np.vstack(self._buf), axis=0)

    def fill(self, feat_vec: np.ndarray) -> np.ndarray:
        """Replace NaN with medians. Returns None if buffer not warm."""
        nan_mask = np.isnan(feat_vec)
        if not np.any(nan_mask):
            return feat_vec
        if self._medians is None:
            return None
        filled = feat_vec.copy()
        filled[nan_mask] = self._medians[nan_mask]
        return filled

    @property
    def is_warm(self) -> bool:
        return self._medians is not None


def extract_window_features(chunk: pd.DataFrame, active_sigs: list) -> tuple:
    """
    Extract 5 features per signal: mean, std, min, max, mean absolute difference.

    Returns:
        feat_vec (np.ndarray): Feature vector
        nan_sig_fraction (float): Fraction of signals that were entirely NaN
    """
    feats = []
    n_nan_sigs = 0

    for col in active_sigs:
        x_raw = chunk[col].values.astype(float)
        x_valid = x_raw[~np.isnan(x_raw)]

        if len(x_valid) == 0:
            feats.extend([np.nan] * 5)
            n_nan_sigs += 1
        else:
            feats.extend([
                np.mean(x_valid),
                np.std(x_valid),
                np.min(x_valid),
                np.max(x_valid),
                np.mean(np.abs(np.diff(x_valid))) if len(x_valid) > 1 else 0.0,
            ])

    feat_vec = np.array(feats, dtype=float)
    nan_sig_fraction = n_nan_sigs / max(len(active_sigs), 1)
    return feat_vec, nan_sig_fraction


def sliding_windows(df, active_sigs,
                    window_size=WINDOW_SIZE_S, step=STEP_SIZE_S):
    """Generate sliding windows with features and quality metrics."""
    n = len(df)
    wid = 0
    for start in range(0, n - window_size + 1, step):
        chunk = df.iloc[start:start + window_size]
        feat_vec, nan_frac = extract_window_features(chunk, active_sigs)
        yield (
            wid,
            float(chunk["time"].iloc[0]),
            float(chunk["time"].iloc[-1]),
            feat_vec,
            nan_frac,
        )
        wid += 1


# =============================================================================
# SECTION 6 — Clear Training Strategy: MAD-Based Warm-Up Selection
# =============================================================================

def select_warmup_windows_mad(
    df: pd.DataFrame,
    active_sigs: list,
    n_required: int = WARMUP_WINDOWS,
    mad_cutoff: float = MAD_CUTOFF,
) -> np.ndarray:
    """
    Select stable warm-up windows using MAD (Median Absolute Deviation) filter.

    Strategy:
    ---------
    1. Extract all candidate windows from first 50% of data
    2. For each window, compute mean HR (primary stability indicator)
    3. Find median of all HR means
    4. Compute MAD: median(|HR_mean - median_HR|)
    5. Convert MAD to SD-equivalent: MAD × 1.4826
    6. Keep windows where |HR_mean - median_HR| ≤ mad_cutoff × MAD_scaled
    7. Take first n_required stable windows

    Why MAD instead of SD?
    ----------------------
    - Median is robust to outliers (doesn't get corrupted by the very
      windows we're trying to exclude)
    - MAD scaled by 1.4826 ≈ SD for Gaussian data
    - mad_cutoff = 2.5 means "within 2.5 SD of median" for normal data
    - but the median itself is immune to the 5% contamination

    Why HR?
    -------
    - HR is continuously monitored, rarely missing
    - HR instability (tachycardia, bradycardia) is the clearest signal
      of induction/emergence phases we want to avoid
    - Other signals may be intermittent (NIBP every few minutes)
    """
    print(f"\n[WARMUP] MAD-based stable window selection")
    print(f"  Cutoff: {mad_cutoff} MAD (≈ {mad_cutoff} SD for Gaussian data)")

    # Step 1: Collect candidates from first 50%
    max_row = int(len(df) * 0.50)
    df_slice = df.iloc[:max_row]

    candidates = []
    hr_means = []

    for wid, t0, t1, fv, nan_frac in sliding_windows(df_slice, active_sigs):
        if nan_frac > MAX_NAN_SIGNAL_FRACTION:
            continue

        # Find HR mean feature (first signal's first feature if HR is active_sigs[0])
        hr_idx = None
        for i, sig in enumerate(active_sigs):
            if "HR" in sig:
                hr_idx = i * 5  # Each signal has 5 features, mean is first
                break

        if hr_idx is not None and not np.isnan(fv[hr_idx]):
            candidates.append(fv)
            hr_means.append(fv[hr_idx])

    if len(candidates) < n_required:
        print(f"  [WARN] Only {len(candidates)} candidates — using all")
        return np.vstack(candidates[:n_required]) if candidates else np.array([])

    # Step 2-4: Compute MAD
    hr_means = np.array(hr_means)
    median_hr = np.median(hr_means)
    mad = np.median(np.abs(hr_means - median_hr))
    mad_scaled = mad * 1.4826  # Convert to SD-equivalent

    print(f"  Candidates collected: {len(candidates)}")
    print(f"  HR median: {median_hr:.1f} bpm")
    print(f"  MAD: {mad:.2f} → SD-equivalent: {mad_scaled:.2f}")

    # Step 5-6: Filter
    deviations = np.abs(hr_means - median_hr)
    stable_mask = deviations <= (mad_cutoff * mad_scaled)

    stable_candidates = [candidates[i] for i in range(len(candidates))
                        if stable_mask[i]]

    print(f"  Stable windows (within {mad_cutoff} MAD): {len(stable_candidates)}")

    if len(stable_candidates) < n_required:
        print(f"  [WARN] Only {len(stable_candidates)} stable — need {n_required}")
        print(f"  Using all stable + filling with closest to median")
        # Fill remaining with windows closest to median
        sorted_indices = np.argsort(deviations)
        selected_indices = sorted_indices[:n_required]
        stable_candidates = [candidates[i] for i in selected_indices]

    # Step 7: Take first n_required (maintains temporal order)
    selected = stable_candidates[:n_required]
    print(f"  Selected: {len(selected)} windows for training")

    return np.vstack(selected)


# =============================================================================
# SECTION 7 — StreamingIF with Unified Scoring
# =============================================================================

class StreamingIF:
    """Isolation Forest with unified WindowScore framework."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self._fitted = False
        self.warmup_scores = None
        self.warmup_mean_score = None

    def fit(self, X: np.ndarray):
        """Train on warm-up windows and compute threshold."""
        Xs = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(Xs)
        scores = -self.model.score_samples(Xs)
        self.warmup_scores = scores
        self.warmup_mean_score = scores.mean()
        self.threshold = np.percentile(scores, THRESHOLD_PCTILE)
        self._fitted = True

        print(f"\n[MODEL] Isolation Forest trained")
        print(f"  Warm-up windows: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Score mean: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Threshold (95th pctile): {self.threshold:.4f}")
        print(f"  Gap (threshold - mean): {self.threshold - scores.mean():.4f}")

    def score(self, x: np.ndarray) -> float:
        """Get raw anomaly score."""
        assert self._fitted
        return float(-self.model.score_samples(
            self.scaler.transform(x.reshape(1, -1)))[0])

    def score_window(self, x: np.ndarray, nan_frac: float) -> WindowScore:
        """
        Score a window with unified framework.

        Returns WindowScore with:
          - anomaly_score: raw IF score
          - quality: data completeness
          - uncertainty: confidence measure
          - trust_score: quality × (1 - uncertainty)
          - weighted_score: anomaly × trust
        """
        anomaly_score = self.score(x)
        quality = compute_window_quality(nan_frac)
        uncertainty = compute_window_uncertainty(quality)

        return WindowScore(
            anomaly_score=anomaly_score,
            quality=quality,
            uncertainty=uncertainty,
        )


# =============================================================================
# SECTION 8 — Simple Drift Detection: Rolling Ratio
# =============================================================================

class DriftDetector:
    """
    Simple drift detection using rolling score ratio.

    Concept:
    --------
    - Maintain buffer of last N weighted_scores
    - drift_ratio = mean(buffer) / warmup_mean_score
    - ratio > 1.3 → scores drifting upward (distribution shift)
    - ratio < 0.7 → scores drifting downward (over-adaptation)

    No complex modeling, just one division and comparison.
    """

    def __init__(self, buffer_size: int = DRIFT_BUFFER_SIZE,
                 threshold: float = DRIFT_RATIO_THRESHOLD,
                 warmup_mean: float = None):
        self.buffer = deque(maxlen=buffer_size)
        self.threshold = threshold
        self.warmup_mean = warmup_mean
        self.drift_detected = False
        self.drift_ratio = 1.0

    def update(self, weighted_score: float):
        """Add new weighted score to buffer."""
        self.buffer.append(weighted_score)

        if len(self.buffer) >= self.buffer.maxlen:
            current_mean = np.mean(self.buffer)
            self.drift_ratio = current_mean / (self.warmup_mean + 1e-9)

            if self.drift_ratio > self.threshold:
                if not self.drift_detected:
                    print(f"\n  [DRIFT] Detected at ratio {self.drift_ratio:.2f} "
                          f"(threshold: {self.threshold})")
                    self.drift_detected = True

    def get_status(self) -> dict:
        """Return current drift status."""
        return {
            "drift_detected": self.drift_detected,
            "drift_ratio": self.drift_ratio,
            "buffer_size": len(self.buffer),
        }


# =============================================================================
# SECTION 9 — Streaming Pipeline with Unified Scoring
# =============================================================================

class StreamingPipeline:
    """Main pipeline with unified scoring and drift detection."""

    def __init__(self, active_sigs):
        self.active_sigs = active_sigs
        self.ifm = StreamingIF()
        self.feat_buffer = FeatureMedianBuffer(maxlen=200)
        self.drift_detector = None
        self.results = []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("STREAMING PIPELINE v6 — START")
        print(f"  Signals: {self.active_sigs}")
        print(f"  Window: {WINDOW_SIZE_S}s | Step: {STEP_SIZE_S}s")
        print("=" * 70)

        # Step 1: Select stable warm-up windows using MAD filter
        warmup_X = select_warmup_windows_mad(df, self.active_sigs)

        if len(warmup_X) < WARMUP_WINDOWS:
            print(f"[ERROR] Insufficient warm-up windows ({len(warmup_X)})")
            return pd.DataFrame()

        # Warm up median buffer
        for fv in warmup_X:
            self.feat_buffer.update(fv)

        # Step 2: Train model
        self.ifm.fit(warmup_X)

        # Step 3: Initialize drift detector
        self.drift_detector = DriftDetector(
            warmup_mean=self.ifm.warmup_mean_score
        )

        # Step 4: Score windows
        inf_times_ms = []
        skipped_nan = 0
        skipped_early = 0
        warmup_end_idx = WARMUP_WINDOWS * STEP_SIZE_S + WINDOW_SIZE_S

        print(f"\n{'WinID':>6}  {'t_start':>9}  {'Score':>8}  {'Quality':>7}  "
              f"{'Trust':>6}  {'Weighted':>9}  {'Status':>8}")
        print("-" * 75)

        for wid, t0, t1, feat_raw, nan_frac in sliding_windows(df, self.active_sigs):
            # Skip warm-up windows
            row_start = wid * STEP_SIZE_S
            if row_start < warmup_end_idx - WINDOW_SIZE_S:
                skipped_early += 1
                continue

            # Skip if too many NaN signals
            if nan_frac > MAX_NAN_SIGNAL_FRACTION:
                skipped_nan += 1
                continue

            # Fill NaN with median
            feat = self.feat_buffer.fill(feat_raw)
            if feat is None:
                self.feat_buffer.update(feat_raw)
                skipped_early += 1
                continue

            self.feat_buffer.update(feat)

            # Score with unified framework
            t_inf0 = time.perf_counter()
            ws = self.ifm.score_window(feat, nan_frac)
            t_inf1 = time.perf_counter()
            inf_ms = (t_inf1 - t_inf0) * 1000
            inf_times_ms.append(inf_ms)

            # Check anomaly
            is_anomaly = ws.weighted_score > self.ifm.threshold
            status = "⚠ ANOM" if is_anomaly else "  ok  "

            # Update drift detector
            self.drift_detector.update(ws.weighted_score)

            print(f"{wid:>6}  {t0:>9.1f}  {ws.anomaly_score:>8.4f}  "
                  f"{ws.quality:>7.2f}  {ws.trust_score:>6.2f}  "
                  f"{ws.weighted_score:>9.4f}  {status:>8}")

            self.results.append({
                "window_id": wid,
                "t_start": t0,
                "t_end": t1,
                "anomaly_score": ws.anomaly_score,
                "quality": ws.quality,
                "uncertainty": ws.uncertainty,
                "trust_score": ws.trust_score,
                "weighted_score": ws.weighted_score,
                "is_anomaly": is_anomaly,
                "inf_ms": inf_ms,
                "nan_sig_frac": nan_frac,
            })

        print(f"\n  Skipped: early/warmup={skipped_early}, high-NaN={skipped_nan}")

        # Latency report
        if inf_times_ms:
            arr = np.array(inf_times_ms)
            print(f"\n  Inference latency: mean={arr.mean():.2f}ms, "
                  f"p95={np.percentile(arr, 95):.2f}ms")

        # Drift report
        drift_status = self.drift_detector.get_status()
        print(f"\n  Drift ratio: {drift_status['drift_ratio']:.2f} "
              f"({'DRIFT DETECTED' if drift_status['drift_detected'] else 'stable'})")

        return pd.DataFrame(self.results)


# =============================================================================
# SECTION 10 — Clean Evaluation: Four Proxy Checks
# =============================================================================

def evaluate_model(results: pd.DataFrame, ifm: StreamingIF,
                   df_clean: pd.DataFrame, active_sigs: list) -> dict:
    """
    Four proxy checks for model reliability without ground truth labels.

    Check 1 — CV (Coefficient of Variation):
        Train model 10 times with different seeds, measure anomaly rate variance
        Low CV (<20%) → stable, not noise-sensitive

    Check 2 — Separation Ratio:
        Ratio of mean anomaly score to threshold
        Good separation: ratio > 1.5 (scores well above threshold)

    Check 3 — Perturbation Sensitivity:
        Add small noise to normal windows, see if scores increase
        Should increase by 10-30% (responds to changes but not too sensitive)

    Check 4 — Temporal Consistency:
        Fraction of anomalies that are isolated (cluster length = 1)
        Low isolation (<30%) → detecting real events, not artefacts
    """
    print(f"\n{'='*70}")
    print("CLEAN EVALUATION — 4 Proxy Checks")
    print(f"{'='*70}")

    eval_results = {}

    # ── Check 1: CV (noise sensitivity) ───────────────────────────────────
    print("\n[Check 1] Coefficient of Variation (noise sensitivity)")

    # Get warmup data for retraining
    warmup_X = ifm.scaler.inverse_transform(
        np.random.randn(WARMUP_WINDOWS, ifm.scaler.mean_.shape[0])
        * np.sqrt(ifm.scaler.var_) + ifm.scaler.mean_
    )

    anom_rates = []
    for seed in range(10):
        model_boot = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=seed,
            n_jobs=-1,
        )
        Xs = ifm.scaler.transform(warmup_X)
        model_boot.fit(Xs)
        boot_scores = -model_boot.score_samples(Xs)
        boot_thresh = np.percentile(boot_scores, THRESHOLD_PCTILE)

        # Score all result windows
        rate = 0
        for _, row in results.iterrows():
            if row["weighted_score"] > boot_thresh:
                rate += 1
        anom_rates.append(100.0 * rate / len(results))

    cv = np.std(anom_rates) / (np.mean(anom_rates) + 1e-9) * 100
    eval_results["cv"] = cv
    eval_results["cv_rates"] = anom_rates

    print(f"  Anomaly rates across 10 seeds: "
          f"[{', '.join(f'{r:.1f}%' for r in anom_rates)}]")
    print(f"  CV: {cv:.1f}%")
    if cv < 20:
        print(f"  ✔ CV < 20% → stable (not noise-sensitive)")
    elif cv < 40:
        print(f"  ⚠ CV = {cv:.0f}% → moderate sensitivity")
    else:
        print(f"  ✘ CV > 40% → high noise sensitivity")

    # ── Check 2: Separation Ratio ─────────────────────────────────────────
    print("\n[Check 2] Separation Ratio (threshold calibration)")

    anom_scores = results[results["is_anomaly"]]["weighted_score"].values
    if len(anom_scores) > 0:
        mean_anom = anom_scores.mean()
        sep_ratio = mean_anom / ifm.threshold
        eval_results["separation_ratio"] = sep_ratio

        print(f"  Mean anomaly score: {mean_anom:.4f}")
        print(f"  Threshold: {ifm.threshold:.4f}")
        print(f"  Separation ratio: {sep_ratio:.2f}")
        if sep_ratio > 1.5:
            print(f"  ✔ Ratio > 1.5 → good separation")
        elif sep_ratio > 1.2:
            print(f"  ⚠ Ratio = {sep_ratio:.2f} → acceptable separation")
        else:
            print(f"  ✘ Ratio < 1.2 → poor separation (threshold too loose)")
    else:
        print("  No anomalies detected")
        eval_results["separation_ratio"] = None

    # ── Check 3: Perturbation Sensitivity ─────────────────────────────────
    print("\n[Check 3] Perturbation Sensitivity (response to changes)")

    normal_windows = results[~results["is_anomaly"]].head(20)
    if len(normal_windows) > 0:
        rng = np.random.default_rng(42)
        score_increases = []

        for _, row in normal_windows.iterrows():
            t0 = row["t_start"]
            t1 = row["t_end"]
            chunk = df_clean[(df_clean["time"] >= t0) &
                            (df_clean["time"] <= t1)]
            if chunk.empty:
                continue

            fv, _ = extract_window_features(chunk, active_sigs)
            if np.any(np.isnan(fv)):
                continue

            # Original score
            orig_score = ifm.score(fv)

            # Add 5% noise
            noise = rng.normal(0, 0.05 * np.abs(fv), size=fv.shape)
            fv_pert = fv + noise
            pert_score = ifm.score(fv_pert)

            increase = (pert_score - orig_score) / (orig_score + 1e-9) * 100
            score_increases.append(increase)

        if score_increases:
            mean_increase = np.mean(score_increases)
            eval_results["perturbation_sensitivity"] = mean_increase

            print(f"  Mean score increase with 5% noise: {mean_increase:.1f}%")
            if 10 <= mean_increase <= 30:
                print(f"  ✔ 10-30% increase → responds appropriately")
            elif mean_increase < 10:
                print(f"  ⚠ <10% increase → may be under-sensitive")
            else:
                print(f"  ⚠ >{mean_increase:.0f}% increase → may be over-sensitive")
        else:
            print("  Not enough normal windows for test")
            eval_results["perturbation_sensitivity"] = None

    # ── Check 4: Temporal Consistency ─────────────────────────────────────
    print("\n[Check 4] Temporal Consistency (artefact detection)")

    anom_flags = results["is_anomaly"].values.astype(int)
    clusters = []
    run = 0
    for flag in anom_flags:
        if flag:
            run += 1
        elif run > 0:
            clusters.append(run)
            run = 0
    if run > 0:
        clusters.append(run)

    if clusters:
        clusters = np.array(clusters)
        isolated_frac = (clusters == 1).sum() / len(clusters)
        eval_results["temporal_consistency"] = 1 - isolated_frac
        eval_results["isolated_fraction"] = isolated_frac
        eval_results["clusters"] = clusters

        print(f"  Anomaly clusters: {len(clusters)}")
        print(f"  Mean cluster length: {clusters.mean():.1f} windows")
        print(f"  Isolated (length=1): {isolated_frac*100:.1f}%")
        if isolated_frac < 0.3:
            print(f"  ✔ <30% isolated → temporally consistent")
        elif isolated_frac < 0.5:
            print(f"  ⚠ {isolated_frac*100:.0f}% isolated → mixed")
        else:
            print(f"  ✘ >{isolated_frac*100:.0f}% isolated → likely artefacts")
    else:
        print("  No anomaly clusters")
        eval_results["temporal_consistency"] = None

    print(f"\n{'='*70}")

    return eval_results


# =============================================================================
# SECTION 11 — Plotting
# =============================================================================

def plot_unified_results(results: pd.DataFrame, df_clean: pd.DataFrame,
                        active_sigs: list, threshold: float,
                        eval_results: dict, prefix: str = "vitaldb"):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Real-Time Anomaly Detection v6 — Unified Scoring Framework\n"
        "MAD-based Training | Rolling Drift Detection | Clean Proxy Evaluation",
        fontsize=11, fontweight="bold"
    )

    wt = results["t_start"].values

    # (1,1) Anomaly scores vs weighted scores
    ax = axes[0, 0]
    ax.plot(wt, results["anomaly_score"].values, lw=0.8, alpha=0.6,
            color="steelblue", label="Raw anomaly score")
    ax.plot(wt, results["weighted_score"].values, lw=1.2,
            color="darkblue", label="Weighted score (anomaly × trust)")
    ax.axhline(threshold, color="orange", lw=1.5, ls="--",
               label=f"Threshold = {threshold:.3f}")
    anom_mask = results["is_anomaly"].values
    ax.scatter(wt[anom_mask], results.loc[anom_mask, "weighted_score"].values,
               color="red", s=20, zorder=5, label="Anomaly")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title("(1) Unified Scoring: Raw vs Weighted")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (1,2) Quality and trust scores
    ax = axes[0, 1]
    ax.plot(wt, results["quality"].values, lw=1, color="green",
            label="Quality (data completeness)")
    ax.plot(wt, results["trust_score"].values, lw=1, color="purple",
            label="Trust = quality × (1 - uncertainty)")
    ax.axhline(QUALITY_DEGRADATION_THRESHOLD, color="red", lw=1, ls="--",
               label=f"Quality threshold = {QUALITY_DEGRADATION_THRESHOLD}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title("(2) Quality and Trust Metrics")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # (2,1) Vital signs
    ax = axes[1, 0]
    numeric = [c for c in active_sigs
               if not c.endswith(("_mean","_std","_p2p","_rms"))]
    colors = plt.cm.tab10.colors
    for i, col in enumerate(numeric[:6]):  # Max 6 signals
        if col not in df_clean.columns:
            continue
        v = df_clean[col].values.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        vn = (v - lo) / (hi - lo + 1e-9)
        ax.plot(df_clean["time"].values, vn, lw=0.6, alpha=0.7,
                color=colors[i % len(colors)], label=col)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value")
    ax.set_title("(3) Vital Signs (normalized)")
    ax.legend(ncol=3, fontsize=6)
    ax.grid(alpha=0.3)

    # (2,2) Evaluation metrics
    ax = axes[1, 1]
    ax.axis('off')

    eval_text = "EVALUATION SUMMARY\n" + "="*35 + "\n\n"

    cv = eval_results.get("cv")
    if cv is not None:
        status = "✔" if cv < 20 else "⚠" if cv < 40 else "✘"
        eval_text += f"[1] CV: {cv:.1f}%  {status}\n"
        eval_text += f"    (noise sensitivity)\n\n"

    sep = eval_results.get("separation_ratio")
    if sep is not None:
        status = "✔" if sep > 1.5 else "⚠" if sep > 1.2 else "✘"
        eval_text += f"[2] Separation: {sep:.2f}  {status}\n"
        eval_text += f"    (threshold calibration)\n\n"

    pert = eval_results.get("perturbation_sensitivity")
    if pert is not None:
        status = "✔" if 10 <= pert <= 30 else "⚠"
        eval_text += f"[3] Perturbation: {pert:.1f}%  {status}\n"
        eval_text += f"    (response to changes)\n\n"

    temp = eval_results.get("isolated_fraction")
    if temp is not None:
        status = "✔" if temp < 0.3 else "⚠" if temp < 0.5 else "✘"
        eval_text += f"[4] Isolated: {temp*100:.0f}%  {status}\n"
        eval_text += f"    (temporal consistency)\n"

    ax.text(0.1, 0.9, eval_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # (3,1) Cluster lengths
    ax = axes[2, 0]
    clusters = eval_results.get("clusters")
    if clusters is not None and len(clusters) > 0:
        max_cl = min(clusters.max(), 20)
        bins = np.arange(0.5, max_cl + 1.5, 1)
        ax.hist(clusters, bins=bins, color="coral", alpha=0.8, edgecolor="white")
        ax.axvline(1.5, color="red", lw=1.5, ls="--", label="Isolated (len=1)")
        ax.set_xlabel("Cluster length (windows)")
        ax.set_ylabel("Count")
        ax.set_title(f"(4) Anomaly Cluster Lengths "
                    f"(isolated: {eval_results.get('isolated_fraction', 0)*100:.0f}%)")
        ax.legend()
        ax.grid(alpha=0.3)

    # (3,2) CV across seeds
    ax = axes[2, 1]
    cv_rates = eval_results.get("cv_rates")
    if cv_rates is not None:
        ax.bar(range(len(cv_rates)), cv_rates, color="steelblue",
               alpha=0.8, edgecolor="white")
        mean_rate = np.mean(cv_rates)
        ax.axhline(mean_rate, color="navy", lw=2, ls="--",
                   label=f"Mean: {mean_rate:.1f}%")
        ax.fill_between(range(len(cv_rates)),
                        mean_rate - np.std(cv_rates),
                        mean_rate + np.std(cv_rates),
                        alpha=0.15, color="navy", label="±1 SD")
        ax.set_xticks(range(len(cv_rates)))
        ax.set_xticklabels([f"s{i}" for i in range(len(cv_rates))], fontsize=7)
        ax.set_ylabel("Anomaly rate (%)")
        ax.set_title(f"(5) Bootstrap Stability (CV = {cv:.1f}%)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{prefix}_v6_unified.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Real-Time Anomaly & Drift Detection ")
    print("  Unified Scoring + MAD Filter + Rolling Drift + Clean Eval")
    print("=" * 70)

    # Load data
    if VITALDB_AVAILABLE:
        print("\n[STEP 1] Fetching VitalDB data…")
        df_raw = load_real_data(case_ids=[1, 2, 3], min_rows=200)
        if len(df_raw) < 200:
            print("[WARN] Insufficient data — using simulation.")
            df_raw = simulate_vitaldb_like()
    else:
        print("\n[STEP 1] Simulating data…")
        df_raw = simulate_vitaldb_like()

    # Handle missing values
    print("\n[STEP 2] Handling missing values…")
    df_clean, active_sigs = handle_missing(df_raw)

    print(f"\n  Active signals: {active_sigs}")
    print(f"  Features per window: {len(active_sigs) * 5}")

    if len(df_clean) < (WARMUP_WINDOWS * STEP_SIZE_S + WINDOW_SIZE_S):
        print("[ERROR] Not enough data. Exiting.")
        return

    # Run pipeline
    print("\n[STEP 3] Running streaming pipeline…")
    pipeline = StreamingPipeline(active_sigs)
    results = pipeline.run(df_clean)

    if results.empty:
        print("[ERROR] No results generated.")
        return

    # Evaluate
    print("\n[STEP 4] Evaluating model…")
    eval_results = evaluate_model(results, pipeline.ifm, df_clean, active_sigs)

    # Plot
    print("\n[STEP 5] Generating plots…")
    prefix = "vitaldb_real" if VITALDB_AVAILABLE else "vitaldb_sim"
    plot_unified_results(results, df_clean, active_sigs,
                        pipeline.ifm.threshold, eval_results, prefix=prefix)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Windows scored: {len(results)}")
    print(f"  Anomalies: {results['is_anomaly'].sum()} "
          f"({results['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Mean quality: {results['quality'].mean():.2f}")
    print(f"  Mean trust: {results['trust_score'].mean():.2f}")
    print(f"  Drift detected: {pipeline.drift_detector.drift_detected}")
    print(f"  Drift ratio: {pipeline.drift_detector.drift_ratio:.2f}")
    print(f"\n[DONE]")

    return results, df_clean


if __name__ == "__main__":
    main()
