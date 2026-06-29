"""Every tunable parameter for the cascade lives here — this is the UI knob surface.

A future Streamlit UI binds sliders/inputs to ExoConfig fields, calls `to_dict()` to
log what was used, and `from_dict()` to apply user overrides. Nothing tunable should be
hardcoded anywhere else in the package.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib, json

# Feature groups (fixed schema, not knobs). Pixel-channel features (centroid_shift,
# rho_consistency, contratio) are deliberately ABSENT — they need TPFs this dataset
# doesn't have, and they're the only real false-positive-blend discriminators.
# That's a dataset task, not a median-filled column. See features.py.
DIP_FEATURES  = ["depth_ppm", "duration_hr", "odd_even_diff", "odd_even_sig",
                 "secondary_depth", "secondary_snr", "trap_shape", "ingress_frac",
                 "snr", "sde"]
SPOT_FEATURES = ["ls_power", "acf_peak", "rot_amplitude"]
# pixel-channel features (built by pixels.py from TPFs). Only used when cfg.use_pixels.
PIXEL_FEATURES = ["centroid_shift", "rho_consistency", "contratio"]
ALL_FEATURES  = DIP_FEATURES + SPOT_FEATURES   # light-curve-only default (use *_features(cfg) below)
# heavy-tailed features get log1p before the OOD distance so a deep-but-real transit
# isn't a geometric outlier (handoff §2).
HEAVY_FEATURES = ["depth_ppm", "secondary_depth", "secondary_snr", "snr", "sde"]


@dataclass
class ExoConfig:
    # ---- data ----
    dataset_root: str = r"D:\Codes\ISRO_Hackathon\dataset"
    feature_cache: str = r"D:\Codes\ISRO_Hackathon\exo\features_real.csv"
    model_dir: str = r"D:\Codes\ISRO_Hackathon\exo\models"
    log_path: str = r"D:\Codes\ISRO_Hackathon\exo\predictions.jsonl"

    # ---- detrend / peel (Stage 0) ----
    flatten_window: int = 401          # long-cadence points (~8 d) — removes rotation, keeps transits

    # ---- Lomb-Scargle rotation search ----
    rot_period_min: float = 0.1
    rot_period_max: float = 30.0
    rot_n_freqs: int = 12000

    # ---- BLS dip search ----
    bls_period_min: float = 0.5
    bls_period_max: float = 15.0
    bls_n_periods: int = 5000
    bls_durations: tuple = (0.05, 0.1, 0.15, 0.2, 0.3)   # days

    # ---- pixel channel (Stage 0, TPF) ----
    max_quarters: int = 4              # difference-image centroid averaged over up to N quarters (1b)
                                       # ponytail: 4 halves the ~0.4px noise floor; more = diminishing, more download

    # ---- heads (Stage 1) ----
    n_estimators: int = 400            # HistGBT max_iter / LightGBM n_estimators
    learning_rate: float = 0.1
    calibrate: bool = True             # isotonic calibration of each head's probability
    balanced_weights: bool = True
    use_pixels: bool = False           # add TPF features (centroid_shift, rho, contratio) to dip heads
    hp_search: bool = False            # tune LightGBM params per head (RandomizedSearchCV) before fitting (3a)

    # ---- decision thresholds (Stage 4) — recall-first on transit ----
    thr_transit: float = 0.30          # lower than 0.5: missing a planet costs more than a review
    thr_eb: float = 0.50
    thr_rotating: float = 0.50

    # ---- OOD gate (Stage 3) ----
    inlier_retention: float = 0.95     # keep 95% of training inliers below the abstain threshold

    # ---- split / repro ----
    test_size: float = 0.25
    seed: int = 42
    n_jobs: int = -1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bls_durations"] = list(self.bls_durations)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExoConfig":
        d = dict(d)
        if "bls_durations" in d:
            d["bls_durations"] = tuple(d["bls_durations"])
        known = {f for f in cls.__dataclass_fields__}          # ignore stray keys from old logs
        return cls(**{k: v for k, v in d.items() if k in known})

    def fingerprint(self) -> str:
        """Stable short hash of the params that affect the model. Excludes paths so the
        same recipe over moved data still matches; data identity is hashed separately."""
        relevant = {k: v for k, v in self.to_dict().items()
                    if k not in ("dataset_root", "feature_cache", "model_dir", "log_path", "n_jobs")}
        blob = json.dumps(relevant, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


def dip_features(cfg: "ExoConfig") -> list:
    """Transit/EB head features — light-curve dip features, plus pixel features if enabled."""
    return DIP_FEATURES + (PIXEL_FEATURES if cfg.use_pixels else [])

def all_features(cfg: "ExoConfig") -> list:
    """Full feature vector the OOD gate scores over."""
    return dip_features(cfg) + SPOT_FEATURES


def label_bits(folder: str) -> dict:
    """Multi-label ground truth from a folder name. The three phenomena co-occur,
    so a folder contributes to every head whose phenomenon it names."""
    return {
        "is_transit":  int("Transit"  in folder),
        "is_EB":       int("EB"        in folder),
        "is_rotating": int("Starspot"  in folder),
    }
