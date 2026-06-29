"""Pixel channel — the only real discriminator of a background-EB false-positive blend.

A light curve is summed pixels; that sum hides *where* the dip came from. This module goes
back to the Target Pixel Files (TPF — the postage-stamp image cube the light curve was made
from) and measures the three features the flux curve cannot:

  centroid_shift   — difference-image photocenter motion during the dip. Real transit dims the
                     target -> centroid stays put; a blended background EB dims an offset star
                     -> centroid shifts. THE blend tell. In pixels (Kepler pixel = 3.98").
  contratio        — 1 - CROWDSAP: fraction of aperture flux NOT from the target (contamination).
  rho_consistency  — log10(transit-implied stellar density / catalogued density). Off-by-a-lot
                     means the dip can't be a planet on this star (blend / wrong host).

Download is resumable + teardown-proof (same lessons as build_dataset.py): socket timeout,
open-write-close logging, incremental index, skip-already-done.
"""
from __future__ import annotations
import os, sys, time, socket, warnings
warnings.filterwarnings("ignore")
socket.setdefaulttimeout(90)
import numpy as np, pandas as pd

from .config import ExoConfig

PIX_DIR = r"D:\Codes\ISRO_Hackathon\exo\pixels"
PIX_INDEX = os.path.join(PIX_DIR, "_index.csv")
PIX_OUT   = os.path.join(PIX_DIR, "pixel_features.csv")
STELLAR_CACHE = os.path.join(PIX_DIR, "_stellar.csv")
WORKERS = 5

G_SI, MSUN, RSUN, RHO_SUN = 6.674e-11, 1.989e30, 6.957e8, 1408.0


def _say(msg):
    try:
        with open(os.path.join(PIX_DIR, "_run.log"), "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass


def _centroid(img):
    """Flux-weighted centroid (row, col) of the positive part of an image."""
    img = np.where(np.isfinite(img), img, 0.0)
    img = np.clip(img, 0, None)
    tot = img.sum()
    if tot <= 0:
        return np.nan, np.nan
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float((img * ys).sum() / tot), float((img * xs).sum() / tot)


def centroid_shift_from_tpf(tpf, period, t0, dur_d):
    """Difference-image centroid offset (pixels) for the dominant dip ephemeris.
    ponytail: difference imaging, NOT ground-based PSF/seeing math — Kepler is space-based and
    undersampled (handoff §5.2). Uses absolute BJD t0+period so any quarter folds correctly."""
    t = tpf.time.value
    flux = tpf.flux.value                                    # (ncad, ny, nx)
    ph = ((t - t0) / period) % 1.0
    half = dur_d / period / 2
    intr = (ph < half) | (ph > 1 - half)
    oot = ~intr
    if intr.sum() < 3 or oot.sum() < 3:
        return np.nan
    img_in = np.nanmean(flux[intr], axis=0)
    img_oot = np.nanmean(flux[oot], axis=0)
    diff = img_oot - img_in                                  # light that disappeared during the dip
    cy_d, cx_d = _centroid(diff)                             # where the dimming source sits
    cy_o, cx_o = _centroid(img_oot)                          # where the target sits
    if not np.isfinite([cy_d, cx_d, cy_o, cx_o]).all():
        return np.nan
    return float(np.hypot(cy_d - cy_o, cx_d - cx_o))


def rho_consistency(period_d, dur_d, M, R):
    """log10(transit-implied density / catalogued density). ~0 = planet-on-this-star consistent."""
    if not (np.isfinite([period_d, dur_d, M, R]).all() and dur_d > 0 and M > 0 and R > 0):
        return np.nan
    P_s = period_d * 86400.0
    aR = period_d / (np.pi * dur_d)                          # a/Rstar, central small-planet approx
    rho_tr = 3 * np.pi / (G_SI * P_s ** 2) * aR ** 3         # SI
    rho_cat = (M * MSUN) / ((4 / 3) * np.pi * (R * RSUN) ** 3)
    if rho_tr <= 0 or rho_cat <= 0:
        return np.nan
    return float(np.clip(np.log10(rho_tr / rho_cat), -3, 3))


def fetch_stellar():
    """Bulk Kepler stellar params (Mathur+2017 DR25, Vizier J/ApJS/229/30) -> KIC: (M, R)."""
    if os.path.exists(STELLAR_CACHE):
        return pd.read_csv(STELLAR_CACHE).set_index("kic")
    from astroquery.vizier import Vizier
    v = Vizier(columns=["KIC", "Mass", "Rad"]); v.ROW_LIMIT = -1
    t = v.get_catalogs("J/ApJS/229/30")[0].to_pandas()
    t = t.rename(columns={"KIC": "kic", "Mass": "M", "Rad": "R"})[["kic", "M", "R"]]
    t["kic"] = t["kic"].astype(int)
    os.makedirs(PIX_DIR, exist_ok=True)
    t.to_csv(STELLAR_CACHE, index=False)
    return t.set_index("kic")


def _clear_cache(kic):
    """Evict any corrupt cached files for this KIC. The original 5-worker build raced the
    shared lightkurve cache and left partial files that fail type-detection on every re-read;
    deleting them lets the re-download succeed."""
    import glob as _g
    cache = os.path.expanduser("~/.cache/lightkurve")
    for p in _g.glob(cache + f"/**/*kplr{kic:09d}*", recursive=True):
        try:
            os.remove(p)
        except Exception:
            pass


def _one(kic, period, t0, dur_d, retries=2, clear_cache=False):
    import lightkurve as lk
    last = "err"
    for attempt in range(retries):
        try:
            if clear_cache and attempt == 0:
                _clear_cache(kic)               # drop poisoned cache before the first re-download
            sr = lk.search_targetpixelfile(f"KIC {kic}", mission="Kepler", cadence="long")
            if len(sr) == 0:
                return {"kic": kic, "status": "no data"}
            tpf = sr[0].download()
            cs = centroid_shift_from_tpf(tpf, period, t0, dur_d)
            crowd = tpf.hdu[1].header.get("CROWDSAP", np.nan)
            contratio = float(1 - crowd) if crowd is not None and np.isfinite(crowd) else np.nan
            return {"kic": kic, "centroid_shift": cs, "contratio": contratio, "status": "ok"}
        except Exception as e:
            last = f"err {type(e).__name__}"
            time.sleep(2 * (attempt + 1))                   # backoff: most failures are transient MAST throttling
    return {"kic": kic, "status": last}


def topup_pixels(cfg: ExoConfig = None, workers: int = 3):
    """Re-attempt the err-status rows (transient MAST failures — the KICs do have TPFs).
    Fewer workers + more retries to dodge throttling. Updates _index in place, then re-merges."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cfg = cfg or ExoConfig()
    idx = pd.read_csv(PIX_INDEX)
    feats = pd.read_csv(cfg.feature_cache).set_index("kic")
    failed = idx[idx["status"].str.startswith("err", na=False)]["kic"].astype(int).tolist()
    _say(f"\npixel topup: retrying {len(failed)} failed KICs, {workers} workers")
    rows = {int(r["kic"]): r.to_dict() for _, r in idx.iterrows()}   # kic -> latest result
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for k in failed:
            if k in feats.index:
                r = feats.loc[k]
                rr = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                futs[ex.submit(_one, k, float(rr["period"]), float(rr["t0"]),
                               float(rr["duration_d"]), 4, True)] = k   # clear_cache=True
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["status"] == "ok":
                rows[res["kic"]] = res
            if i % 20 == 0:
                _say(f"  topup [{i}/{len(futs)}] ({time.time()-t0:.0f}s)")
                pd.DataFrame(list(rows.values())).to_csv(PIX_INDEX, index=False)
    pd.DataFrame(list(rows.values())).to_csv(PIX_INDEX, index=False)
    n_ok = sum(1 for r in rows.values() if r.get("status") == "ok")
    _say(f"pixel topup done: {n_ok} ok total")
    return _rebuild_output(cfg)


def build_pixel_table(cfg: ExoConfig = None):
    """Download a TPF per star, compute centroid_shift + contratio; merge stellar -> rho_consistency.
    Reads period/t0/duration_d from the cached feature table. Resumable: re-run to continue."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cfg = cfg or ExoConfig()
    os.makedirs(PIX_DIR, exist_ok=True)
    feats = pd.read_csv(cfg.feature_cache)
    need = feats[["kic", "period", "t0", "duration_d"]].dropna()

    done = set(pd.read_csv(PIX_INDEX)["kic"].astype(int)) if os.path.exists(PIX_INDEX) else set()
    jobs = [r for r in need.itertuples(index=False) if int(r.kic) not in done]
    _say(f"\npixel build: {len(jobs)} to do ({len(done)} cached), {WORKERS} workers")

    rows = pd.read_csv(PIX_INDEX).to_dict("records") if os.path.exists(PIX_INDEX) else []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one, int(r.kic), float(r.period), float(r.t0), float(r.duration_d)): r
                for r in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 20 == 0:
                _say(f"  [{i}/{len(jobs)}] ({time.time()-t0:.0f}s)")
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(PIX_INDEX, index=False)
    pd.DataFrame(rows).to_csv(PIX_INDEX, index=False)
    return _rebuild_output(cfg)


def _rebuild_output(cfg: ExoConfig):
    """Join the per-KIC index (centroid_shift, contratio) with the feature table's transit
    geometry and the stellar catalog -> the 3-feature pixel table. Idempotent; run after
    build or topup."""
    idx = pd.read_csv(PIX_INDEX)
    need = pd.read_csv(cfg.feature_cache)[["kic", "period", "duration_d"]].dropna()
    try:
        stel = fetch_stellar()
    except Exception as e:
        _say(f"stellar fetch failed ({e}); rho_consistency left NaN"); stel = None
    px = (idx[idx["status"] == "ok"][["kic", "centroid_shift", "contratio"]]
          .drop_duplicates("kic", keep="last"))      # build appended + topup rewrote -> dedup
    merged = need.merge(px, on="kic", how="inner")
    if stel is not None:
        merged = merged.join(stel, on="kic")
        merged["rho_consistency"] = [rho_consistency(p, d, m, r) for p, d, m, r in
                                     zip(merged["period"], merged["duration_d"],
                                         merged.get("M", np.nan), merged.get("R", np.nan))]
    else:
        merged["rho_consistency"] = np.nan
    out = merged[["kic", "centroid_shift", "rho_consistency", "contratio"]]
    out.to_csv(PIX_OUT, index=False)
    _say(f"pixel output: {len(out)} rows -> {PIX_OUT}")
    return out


def merge_into_features(cfg: ExoConfig = None) -> str:
    """Join pixel features into the light-curve feature table -> a new augmented CSV.
    Original features_real.csv is left untouched."""
    cfg = cfg or ExoConfig()
    feats = pd.read_csv(cfg.feature_cache)
    px = pd.read_csv(PIX_OUT)
    merged = feats.merge(px, on="kic", how="left")
    out = cfg.feature_cache.replace(".csv", "_px.csv")
    merged.to_csv(out, index=False)
    n_px = merged["centroid_shift"].notna().sum()
    _say(f"merged: {n_px}/{len(merged)} rows have pixel features -> {out}")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_pixel_table()
    elif cmd == "topup":
        topup_pixels()
    elif cmd == "merge":
        merge_into_features()
    elif cmd == "test":
        # self-check: centroid + rho on one real star, assert finite & sane
        import lightkurve as lk
        cfg = ExoConfig(); f = pd.read_csv(cfg.feature_cache)
        r = f.dropna(subset=["period", "t0", "duration_d"]).iloc[0]
        tpf = lk.search_targetpixelfile(f"KIC {int(r.kic)}", mission="Kepler", cadence="long")[0].download()
        cs = centroid_shift_from_tpf(tpf, r.period, r.t0, r.duration_d)
        rho = rho_consistency(r.period, r.duration_d, 1.0, 1.0)
        print(f"KIC{int(r.kic)}  centroid_shift={cs:.4f}px  rho_consistency(M=R=1)={rho:.3f}")
        assert np.isfinite(cs) and 0 <= cs < 5, f"centroid_shift insane: {cs}"
        print("pixels self-check OK")
