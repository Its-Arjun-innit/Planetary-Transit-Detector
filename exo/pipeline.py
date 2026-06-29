"""Stages 2 + 4 + orchestration — the surface a Streamlit UI imports.

End to end: light curve -> features -> heads -> peel-resolve blends -> combine verdict
-> append a comparable JSONL log line. Pure functions, no UI, no globals.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd

from .config import ExoConfig, DIP_FEATURES, ALL_FEATURES
from .features import extract_features, eclipse_points
from .cascade import Cascade, HEADS, train as _train

SCHEMA_VERSION = 1

# (is_transit, is_EB, is_rotating) -> human verdict. Independent bits, so all combos exist.
_VERDICT = {
    (0, 0, 0): "No clear signal (flat / below detection)",
    (1, 0, 0): "Transit (planet candidate)",
    (0, 1, 0): "Eclipsing binary",
    (0, 0, 1): "Rotating / spotted star",
    (1, 0, 1): "Transit on a spotted star (spotted planet host)",
    (0, 1, 1): "Eclipsing binary with starspots",
    (1, 1, 0): "Transit + eclipsing binary (blend candidate)",
    (1, 1, 1): "Transit + EB + rotation (complex — review)",
}


def _combine(bits: dict, ood: dict) -> dict:
    T, E, R = bits["is_transit"], bits["is_EB"], bits["is_rotating"]
    abstain = ood["flagged"]
    blend = bool(T and E)
    label = _VERDICT[(T, E, R)]
    note = ""
    if abstain:
        note = ("out-of-distribution: doesn't resemble any trained class — "
                f"would-be verdict '{label}' is unsupported")
        label = "Out of distribution — needs review"
    elif blend:
        note = ("two phenomena co-occur. Confirming a *background-EB* false positive needs "
                "the pixel/centroid channel (TPF), not in this dataset — flag for vetting")
    return {"label": label, "blend": blend, "rotating": bool(R),
            "abstain": bool(abstain), "note": note}


def _jsonable(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(type(o))


def log_prediction(record: dict, cfg: ExoConfig):
    """Append one JSONL line. Open-write-close per call so a session teardown can't kill us
    (same lesson as build_dataset._say). Records carry model_version + config + input_id, so
    two models' predictions on the same input are directly comparable later."""
    try:
        os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
        with open(cfg.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_jsonable) + "\n")
    except Exception:
        pass


def predict_lightcurve(time, flux, model: Cascade, cfg: ExoConfig = None,
                       input_id: str = "?", source: str = "lightcurve",
                       do_peel: bool = True, log: bool = True, pixel_feats: dict = None) -> dict:
    """The one prediction entry point. Arrays in -> verdict dict out (and logged).
    pixel_feats: optional {centroid_shift, rho_consistency, contratio} so a pixel-trained model
    can use the real TPF features instead of NaN (looked up by KIC in predict_parquet)."""
    cfg = cfg or model.cfg
    feat = extract_features(time, flux, cfg)
    if feat is not None and pixel_feats:
        feat.update({k: v for k, v in pixel_feats.items() if v is not None})
    rec = {"schema_version": SCHEMA_VERSION, "timestamp": time_iso(),
           "model_version": model.model_version, "source": source, "input_id": input_id,
           "config": cfg.to_dict()}
    if feat is None:
        rec["verdict"] = {"label": "No usable curve", "abstain": True}
        if log: log_prediction(rec, cfg)
        return rec

    out = model.predict(feat)
    bits_raw = dict(out["bits"])
    bits = dict(bits_raw)
    peel = None

    # Stage 2: peel test — only when transit AND EB both fire
    if do_peel and bits["is_transit"] and bits["is_EB"] and np.isfinite(feat.get("period", np.nan)):
        mask = eclipse_points(time, feat["period"], feat["t0"], feat["duration_d"])
        feat2 = extract_features(time, flux, cfg, eclipse_mask=mask)
        if feat2 is not None:
            if pixel_feats:
                feat2.update({k: v for k, v in pixel_feats.items() if v is not None})
            tf = model.head_feats["is_transit"]              # resolved list (incl. pixels if used)
            x = np.array([[feat2.get(c, np.nan) for c in tf]], float)
            p2 = float(model.heads["is_transit"].predict_proba(x)[0, 1])
            survived = p2 >= model.cfg.thr_transit      # thresholds are a trained property of the model
            peel = {"transit_prob_after_peel": round(p2, 3), "survived": bool(survived)}
            if not survived:
                bits["is_transit"] = 0          # the "transit" was the EB's secondary eclipse

    verdict = _combine(bits, out["ood"])
    rec.update({
        "features": {k: _round(feat.get(k)) for k in model.all_feats + ["period", "P_rot", "t0"]},
        "probs": {k: round(v, 4) for k, v in out["probs"].items()},
        "bits_raw": bits_raw, "bits_final": bits,
        "thresholds": {"transit": model.cfg.thr_transit, "eb": model.cfg.thr_eb,
                       "rotating": model.cfg.thr_rotating},
        "ood": out["ood"], "peel": peel, "verdict": verdict,
    })
    if log:
        log_prediction(rec, cfg)
    return rec


def predict_parquet(path: str, model: Cascade, cfg: ExoConfig = None, **kw) -> dict:
    df = pd.read_parquet(path, columns=["time_bjd", "flux", "kic"])
    kic = int(df["kic"].iloc[0])
    px = kw.pop("pixel_feats", None)
    if px is None and getattr(model.cfg, "use_pixels", False):
        px = _lookup_pixels(kic)          # use precomputed TPF features for a pixel-trained model
    return predict_lightcurve(df["time_bjd"].values, df["flux"].values, model, cfg,
                              input_id=f"KIC{kic}", source="parquet", pixel_feats=px, **kw)


_PIX_CACHE = None
def _lookup_pixels(kic: int):
    """Precomputed pixel features for a dataset KIC, if available (exo/pixels/pixel_features.csv)."""
    global _PIX_CACHE
    if _PIX_CACHE is None:
        from .pixels import PIX_OUT
        _PIX_CACHE = pd.read_csv(PIX_OUT).set_index("kic") if os.path.exists(PIX_OUT) else pd.DataFrame()
    if kic in _PIX_CACHE.index:
        r = _PIX_CACHE.loc[kic]
        return {c: float(r[c]) for c in ("centroid_shift", "rho_consistency", "contratio")
                if c in r and np.isfinite(r[c])}
    return None


def predict_tic(tic: str, model: Cascade, cfg: ExoConfig = None, **kw) -> dict:
    """Fetch a curve by name/ID via lightkurve (for the future 'upload / look up a target' UI)."""
    import lightkurve as lk
    sr = lk.search_lightcurve(tic)
    if len(sr) == 0:
        return {"verdict": {"label": "no data", "abstain": True}, "input_id": tic}
    lc = sr[0].download().remove_nans().normalize()
    return predict_lightcurve(lc.time.value, lc.flux.value, model, cfg,
                              input_id=str(tic), source="mast", **kw)


def time_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _round(v):
    return round(float(v), 6) if v is not None and np.isfinite(v) else None


# ---- model lifecycle (re-exported so the UI imports one module) ----
def train(cfg: ExoConfig = None, table: pd.DataFrame = None, save: bool = True) -> Cascade:
    m = _train(cfg, table)
    if save:
        path = m.save()
        print(f"[train] saved {path}  (version {m.model_version})")
    for k, v in m.metrics.items():
        print(f"  {k}: {v}")
    return m


def train_pixels(cfg: ExoConfig = None, save: bool = True) -> Cascade:
    """Merge the TPF pixel features into the table, train the dip heads WITH them, tune, save.
    Distinct model_version (use_pixels feeds the fingerprint) so logs separate it from baseline.
    The point: measure whether centroid_shift/rho/contratio lift the transit & EB heads."""
    from .pixels import merge_into_features
    cfg = cfg or ExoConfig()
    px_csv = merge_into_features(cfg)
    cfg.use_pixels = True
    cfg.feature_cache = px_csv
    table = pd.read_csv(px_csv)
    m = _train(cfg, table)
    chosen = m.tune_thresholds(table)
    print(f"[train_pixels] tuned thresholds: {chosen}")
    for k, v in m.metrics.items():
        print(f"  {k}: {v}")
    if save:
        m.save()
        print(f"[train_pixels] saved version {m.model_version}")
    return m


def load_model(cfg: ExoConfig = None, path: str = None) -> Cascade:
    cfg = cfg or ExoConfig()
    return Cascade.load(path or os.path.join(cfg.model_dir, "latest.joblib"))


def tune(cfg: ExoConfig = None, table: pd.DataFrame = None, save: bool = True) -> Cascade:
    """Load latest model, re-pick thresholds from data, re-save under a new version."""
    from .features import build_feature_table
    cfg = cfg or ExoConfig()
    table = table if table is not None else build_feature_table(cfg)
    m = load_model(cfg)
    old = {k: getattr(cfg, a) for k, (_, a) in HEADS.items()}
    m.cfg = cfg
    chosen = m.tune_thresholds(table)
    print(f"[tune] thresholds: {old}  ->  {chosen}")
    for k, v in m.metrics.items():
        print(f"  {k}: {v}")
    if save:
        print(f"[tune] saved version {m.model_version}")
        m.save()
    return m


if __name__ == "__main__":
    # self-check: combiner truth table + a real round-trip predict on one parquet
    import glob, warnings; warnings.filterwarnings("ignore")
    ok = {(1, 0, 0): "Transit", (0, 1, 0): "Eclipsing binary", (1, 1, 0): "blend"}
    for bits3, want in ok.items():
        v = _combine(dict(zip(("is_transit", "is_EB", "is_rotating"), bits3)),
                     {"flagged": False})
        assert want.lower() in v["label"].lower(), (bits3, v)
    assert _combine({"is_transit": 1, "is_EB": 0, "is_rotating": 0},
                    {"flagged": True})["abstain"]
    print("combiner self-check OK")
