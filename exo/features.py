"""Stage 0 — turn one light curve into the feature row the cascade consumes.

Adapted from model_lab.ipynb `extract_real`, but takes a pre-loaded light curve
(arrays / DataFrame) instead of fetching from MAST. Same physics: Lomb-Scargle models
rotation, flatten peels it off, BLS searches the residual for dips.

NOT computed here: centroid_shift, rho_consistency, contratio. Those need Target Pixel
Files (not in this dataset) and are the only real discriminators of a background-EB
false-positive blend. Adding the pixel channel is the next dataset task; faking these with
medians would just add constant columns, so they're left out entirely.
"""
from __future__ import annotations
import glob, os, warnings
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle

from .config import ExoConfig, DIP_FEATURES, SPOT_FEATURES, ALL_FEATURES, label_bits

warnings.filterwarnings("ignore")


def _acf_peak(time, flux, cfg: ExoConfig) -> float:
    """Height of the strongest autocorrelation peak in the rotation-lag window.
    This is the feature most aligned with the labels: McQuillan+2014 (a main rotation
    catalog) measured rotation by ACF, so the rotation folders are ~ACF-selected."""
    f = flux - np.mean(flux)
    acf = np.correlate(f, f, mode="full")[len(f) - 1:]
    if acf[0] == 0:
        return 0.0
    acf = acf / acf[0]
    dt = np.median(np.diff(time))                      # days per sample
    lo = max(1, int(cfg.rot_period_min / dt))
    hi = min(len(acf) - 1, int(cfg.rot_period_max / dt))
    return float(np.max(acf[lo:hi])) if hi > lo else 0.0


def _bls_dip(time, flux, cfg: ExoConfig) -> dict:
    """BLS on a (already detrended) curve -> dip features + the geometry needed to peel it."""
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(cfg.bls_period_min, cfg.bls_period_max, cfg.bls_n_periods)
    res = bls.power(periods, np.array(cfg.bls_durations))
    i = int(np.argmax(res.power))
    P, t0, dur, depth = (float(res.period[i]), float(res.transit_time[i]),
                         float(res.duration[i]), float(res.depth[i]))
    sde = float((res.power[i] - res.power.mean()) / (res.power.std() + 1e-12))

    ph = ((time - t0) / P) % 1.0
    half = dur / P / 2
    in_tr = (ph < half) | (ph > 1 - half)
    oot = flux[~in_tr]
    sigma = np.std(oot) if len(oot) else np.std(flux)
    snr = float(depth / (sigma / np.sqrt(max(1, in_tr.sum())))) if sigma > 0 else 0.0

    cyc = np.floor((time - t0) / P).astype(int)
    odd, even = flux[in_tr & (cyc % 2 == 1)], flux[in_tr & (cyc % 2 == 0)]
    odd_even = float(abs(np.median(odd) - np.median(even))) if len(odd) and len(even) else 0.0
    # odd_even_sig: that depth difference in units of its own error (EB tell). A raw diff of X is
    # meaningless without the scatter; significance is what actually separates EBs from noise.
    oe_sig = 0.0
    if len(odd) > 1 and len(even) > 1:
        se = np.sqrt(np.var(odd) / len(odd) + np.var(even) / len(even))
        oe_sig = float(abs(np.median(odd) - np.median(even)) / se) if se > 0 else 0.0

    sec = flux[(ph > 0.5 - half) & (ph < 0.5 + half)]
    sec_depth = float(max(0.0, (1 - np.median(sec)) * 1e6)) if len(sec) else 0.0
    # secondary_snr: secondary-eclipse depth in units of the out-of-eclipse noise. A real secondary
    # (EB) stands above the noise; a depth value alone can't tell a true secondary from scatter.
    sec_snr = float((1 - np.median(sec)) / (sigma / np.sqrt(max(1, len(sec))))) \
        if len(sec) and sigma > 0 else 0.0

    # trap_shape: 0 = flat-bottomed box (planet/U), ->1 = pointy (grazing EB/V).
    # ponytail: proxy = how much shallower the median in-transit point is than the deepest.
    intr = flux[in_tr]
    trap = 0.0
    if len(intr):
        dmin = 1 - float(np.min(intr))
        dmid = 1 - float(np.median(intr))
        if dmin > 0:
            trap = float(np.clip(1 - dmid / dmin, 0, 1))
    # ingress_frac: fraction of the dip spent in ingress+egress (the sloped walls) vs the flat floor.
    # Planet (sharp walls) -> small; grazing EB (V-shape, all slope) -> large. Complements trap_shape
    # with a phase-resolved view. ponytail: floor = points within 25% of full depth of the deepest.
    ingress_frac = 0.0
    if len(intr) and dmin > 0:
        on_floor = (1 - intr) >= 0.75 * dmin
        ingress_frac = float(1 - on_floor.sum() / len(intr))

    return dict(depth_ppm=depth * 1e6, duration_hr=dur * 24, odd_even_diff=odd_even,
                odd_even_sig=oe_sig, secondary_depth=sec_depth, secondary_snr=sec_snr,
                trap_shape=trap, ingress_frac=ingress_frac, snr=snr, sde=sde,
                period=P, t0=t0, duration_d=dur)


def extract_features(time, flux, cfg: ExoConfig = None, eclipse_mask=None) -> dict:
    """Full Stage 0 on one curve. Returns ALL_FEATURES + metadata (period/t0/P_rot/rotating).
    `eclipse_mask` (bool, True=drop) lets the peel test re-extract on the residual."""
    cfg = cfg or ExoConfig()
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    if eclipse_mask is not None:
        good &= ~np.asarray(eclipse_mask, bool)
    time, flux = time[good], flux[good]
    if len(flux) < 50:
        return None
    flux = flux / np.median(flux)                       # normalize (parquet is ~normalized already)

    # (1) rotation on the un-flattened curve
    freq = np.linspace(1 / cfg.rot_period_max, 1 / cfg.rot_period_min, cfg.rot_n_freqs)
    power = LombScargle(time, flux).power(freq)
    j = int(np.argmax(power))
    P_rot, ls_power = float(1 / freq[j]), float(power[j])
    rot_amplitude = float(np.std(flux))
    acf_peak = _acf_peak(time, flux, cfg)

    # (2) peel: flatten removes rotation/systematics before the dip search
    fl = _flatten(flux, cfg.flatten_window)
    dip = _bls_dip(time, fl, cfg)

    feat = {**dip, "ls_power": ls_power, "acf_peak": acf_peak,
            "rot_amplitude": rot_amplitude, "P_rot": P_rot}
    return feat


def eclipse_points(time, period, t0, duration_d) -> np.ndarray:
    """Boolean mask of in-eclipse cadences — used by the Stage 2 peel test."""
    time = np.asarray(time, float)
    ph = ((time - t0) / period) % 1.0
    half = duration_d / period / 2
    return (ph < half) | (ph > 1 - half)


def _flatten(flux, window):
    """Median-filter detrend. ponytail: stdlib-ish rolling median instead of pulling in
    lightkurve just to flatten a parquet array."""
    w = int(window) | 1                                 # force odd
    if w >= len(flux):
        return flux / np.median(flux)
    s = pd.Series(flux)
    trend = s.rolling(w, center=True, min_periods=1).median().values
    return flux / trend


# ---------------- build the training feature table ----------------
def _row(path: str, cfg: ExoConfig):
    try:
        df = pd.read_parquet(path, columns=["time_bjd", "flux", "kic", "label"])
        feat = extract_features(df["time_bjd"].values, df["flux"].values, cfg)
        if feat is None:
            return None
        folder = str(df["label"].iloc[0])
        feat.update(label_bits(folder))
        feat["folder"] = folder
        feat["kic"] = int(df["kic"].iloc[0])
        return feat
    except Exception as e:
        return {"_error": f"{os.path.basename(path)}: {type(e).__name__}: {e}"}


def _dedup_kic(table: pd.DataFrame) -> pd.DataFrame:
    """Some KICs sit in two folders with conflicting labels (catalog drift across the original
    build's topup rounds). Keep the most-specific folder (overlap folders were added later from
    more complete catalogs, handoff §5) — one row per star, no contradictory labels."""
    if "kic" not in table or not table["kic"].duplicated().any():
        return table
    spec = table["folder"].str.count(r"\+")
    return (table.assign(_spec=spec).sort_values("_spec", ascending=False)
            .drop_duplicates("kic", keep="first").drop(columns="_spec")
            .sort_index().reset_index(drop=True))


def build_feature_table(cfg: ExoConfig = None, use_cache: bool = True) -> pd.DataFrame:
    """Glob every parquet under dataset_root, extract features, return one row per curve.
    Cached to cfg.feature_cache. No hardcoded counts — drop in more parquets and rebuild."""
    cfg = cfg or ExoConfig()
    if use_cache and os.path.exists(cfg.feature_cache):
        return _dedup_kic(pd.read_csv(cfg.feature_cache))

    paths = sorted(glob.glob(os.path.join(cfg.dataset_root, "**", "*.parquet"), recursive=True))
    paths = [p for p in paths if not os.path.basename(p).startswith("_")]
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=cfg.n_jobs, verbose=5)(delayed(_row)(p, cfg) for p in paths)

    errs = [r["_error"] for r in out if r and "_error" in r]
    rows = [r for r in out if r and "_error" not in r]
    if errs:
        print(f"[build_feature_table] {len(errs)} failed (kept {len(rows)}). e.g. {errs[:3]}")
    table = _dedup_kic(pd.DataFrame(rows))
    os.makedirs(os.path.dirname(cfg.feature_cache), exist_ok=True)
    table.to_csv(cfg.feature_cache, index=False)
    print(f"[build_feature_table] {len(table)} rows -> {cfg.feature_cache}")
    return table


if __name__ == "__main__":
    # self-check: extract on one real Transit curve, assert physically sane outputs
    cfg = ExoConfig()
    p = sorted(glob.glob(os.path.join(cfg.dataset_root, "Transit", "*.parquet")))[0]
    df = pd.read_parquet(p)
    f = extract_features(df["time_bjd"].values, df["flux"].values, cfg)
    print(p)
    for k in ALL_FEATURES + ["period", "P_rot"]:
        print(f"  {k:16s} {f[k]:.4g}")
    assert f is not None and f["depth_ppm"] > 0, "no dip found"
    assert cfg.bls_period_min <= f["period"] <= cfg.bls_period_max, "period out of grid"
    assert set(ALL_FEATURES) <= set(f), "missing features"
    print("features self-check OK")
