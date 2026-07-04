"""Build a labeled Kepler light-curve dataset.

Phase 1: cross-match 3 KIC-keyed catalogs -> assign each star to one of 6 folders.
Phase 2: download one Kepler quarter per star, write parquet into its class folder.

Run `python build_dataset.py plan`     -> phase 1 only (counts, no downloads)
Run `python build_dataset.py download` -> phase 2 (parallel, resumable)
"""
import os, sys, io, warnings, time, socket
warnings.filterwarnings("ignore")
socket.setdefaulttimeout(60)   # catch-all: break a stalled MAST socket regardless of which lib owns it
import numpy as np, pandas as pd

ROOT          = r"D:\Codes\ISRO_Hackathon\dataset"
MAX_PER_CLASS = 400            # ponytail: balanced cap; McQuillan has ~34k rotators
WORKERS       = 5              # parallel MAST downloads
PLAN_CSV      = os.path.join(ROOT, "_plan.csv")

FOLDERS = ["Transit", "EB", "Starspot", "Transit+EB", "EB+Starspot", "Starspot+Transit"]


# ---------------- Phase 1: labels via catalog cross-match ----------------
def fetch_catalogs():
    from astroquery.vizier import Vizier
    v = Vizier(columns=["**"]); v.ROW_LIMIT = -1

    # Transit: confirmed KOI hosts (kepid == KIC)
    print("  KOI confirmed (NASA Exoplanet Archive TAP)...")
    koi_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               "select+distinct+kepid+from+cumulative+where+koi_disposition='CONFIRMED'&format=csv")
    C = set(pd.read_csv(koi_url)["kepid"].astype(int))

    # EB: Villanova Kepler EB catalog (Kirk+2016)
    print("  Villanova Kepler EB (Vizier J/AJ/151/68)...")
    eb = v.get_catalogs("J/AJ/151/68")[0].to_pandas()
    eb_kic = next(c for c in eb.columns if c.upper() in ("KIC", "KIC ID", "KICID"))
    E = set(eb[eb_kic].astype(int))

    # Starspot: McQuillan+2014 (clean field rotators, excludes KOIs/EBs)
    #   U Santos+2021 (includes KOIs/EBs -> populates the rotation-blend folders)
    print("  McQuillan+2014 rotation (Vizier J/ApJS/211/24)...")
    mcq = v.get_catalogs("J/ApJS/211/24")[0].to_pandas()
    print("  Santos+2021 rotation (Vizier J/ApJS/255/17 table1)...")
    san = v.get_catalogs("J/ApJS/255/17")[0].to_pandas()   # table1 = detected rotators
    print("  McQuillan+2013 KOI rotation (Vizier J/ApJ/775/L11)...")
    koi_rot = v.get_catalogs("J/ApJ/775/L11")[0].to_pandas()  # 2010 KOIs -> boosts Starspot+Transit
    R = set(mcq["KIC"].astype(int)) | set(san["KIC"].astype(int)) | set(koi_rot["KIC"].astype(int))
    return C, E, R


def _regions():
    C, E, R = fetch_catalogs()
    print(f"  raw sizes:  Transit(C)={len(C)}  EB(E)={len(E)}  Starspot(R)={len(R)}")
    region = {                                    # 6 disjoint Venn regions
        "Transit":          C - E - R,
        "EB":               E - C - R,
        "Starspot":         R - C - E,
        "Transit+EB":       (C & E) - R,
        "EB+Starspot":      (E & R) - C,
        "Starspot+Transit": (C & R) - E,
    }
    return region, C & E & R                      # + triple


def plan():
    os.makedirs(ROOT, exist_ok=True)
    print("Phase 1: fetching catalogs + cross-matching by KIC...")
    region, triple = _regions()

    rows = []
    rng = np.random.default_rng(42)
    for folder in FOLDERS:
        kics = sorted(region[folder])
        if len(kics) > MAX_PER_CLASS:                        # balanced subsample
            kics = list(rng.choice(kics, MAX_PER_CLASS, replace=False))
        for k in kics:
            rows.append({"kic": int(k), "folder": folder})

    plan_df = pd.DataFrame(rows)
    plan_df.to_csv(PLAN_CSV, index=False)

    print("\n  folder counts (after cap):")
    for f in FOLDERS:
        print(f"    {f:18s} {len(region[f]):6d} available -> {(plan_df.folder==f).sum():4d} planned")
    if triple:
        print(f"    [triple C&E&R: {len(triple)} stars logged, not downloaded]")
    print(f"\n  total to download: {len(plan_df)}  ->  plan saved {PLAN_CSV}")
    return plan_df


# ---------------- Phase 2: download one quarter, write parquet ----------------
def _set_mast_timeout(sec=60):
    """Short MAST timeout so a hung socket fails fast instead of blocking a worker ~10 min."""
    try:
        from astroquery.mast import Observations, Mast
        Observations.TIMEOUT = Mast.TIMEOUT = sec
    except Exception:
        pass

def _one(kic, folder):
    import lightkurve as lk
    _set_mast_timeout(60)
    out_dir = os.path.join(ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"KIC{kic}.parquet")
    if os.path.exists(out):
        return kic, folder, "cached"
    last = "err"
    for attempt in range(2):                              # 1 retry: shrug off a transient MAST hang
        try:
            sr = lk.search_lightcurve(f"KIC {kic}", mission="Kepler", cadence="long")
            if len(sr) == 0:
                sr = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")  # fallback: any cadence
            if len(sr) == 0:
                return kic, folder, "no data"
            lc = sr[0].download().remove_nans().normalize()
            fe = getattr(lc, "flux_err", None)
            df = pd.DataFrame({
                "time_bjd": lc.time.value,
                "flux": lc.flux.value,
                "flux_err": fe.value if fe is not None else np.nan,
                "kic": kic, "label": folder,
            })
            df.to_parquet(out, compression="snappy", index=False)
            return kic, folder, f"ok ({len(df)})"
        except Exception as e:
            last = f"err {type(e).__name__}"
    return kic, folder, last


def _say(msg):
    """Bulletproof log: open-write-close per line so session teardown can't close a held handle and kill us."""
    try:
        with open(os.path.join(ROOT, "_run.log"), "a") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass

def download():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pd.DataFrame({"x": [1]}).to_parquet(os.path.join(ROOT, "_engine_check.parquet"))  # fail loud if no parquet engine
    if not os.path.exists(PLAN_CSV):
        plan_df = plan()
    else:
        plan_df = pd.read_csv(PLAN_CSV)
    jobs = list(plan_df.itertuples(index=False))
    _say(f"\nPhase 2: downloading {len(jobs)} curves, {WORKERS} workers, resumable...")
    idx_path = os.path.join(ROOT, "_index.csv")
    log, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one, int(r.kic), r.folder): r for r in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            kic, folder, status = fut.result()
            log.append({"kic": kic, "folder": folder, "status": status})
            if i % 20 == 0 or "ok" in status:
                _say(f"  [{i:4d}/{len(jobs)}] KIC{kic} {folder:18s} {status}  ({time.time()-t0:.0f}s)")
            if i % 25 == 0:                                   # incremental index: survives a teardown-kill
                pd.DataFrame(log).to_csv(idx_path, index=False)
    pd.DataFrame(log).to_csv(idx_path, index=False)
    n_ok = sum("ok" in l["status"] or l["status"] == "cached" for l in log)
    _say(f"\ndone. {n_ok}/{len(jobs)} curves saved under {ROOT}")


# ---------------- Phase 3: top-up failed/no-data slots from spare KICs ----------------
def _saved(folder):
    d = os.path.join(ROOT, folder)
    return {int(f[3:-8]) for f in os.listdir(d)} if os.path.isdir(d) else set()  # KIC<n>.parquet

def topup():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _say("\nPhase 3: top-up — refill each folder to its planned count from spare region KICs...")
    region, _ = _regions()
    idx = os.path.join(ROOT, "_index.csv")
    attempted = set(pd.read_csv(idx)["kic"].astype(int)) if os.path.exists(idx) else set()

    for folder in FOLDERS:
        pool = region[folder]
        target = min(MAX_PER_CLASS, len(pool))
        saved = _saved(folder)
        need = target - len(saved)
        candidates = sorted(pool - attempted - saved)        # fresh, never-tried KICs
        if need <= 0:
            _say(f"  {folder:18s} {len(saved)}/{target}  complete"); continue
        if not candidates:
            _say(f"  {folder:18s} {len(saved)}/{target}  SHORT by {need} — region exhausted, no spares"); continue
        _say(f"  {folder:18s} {len(saved)}/{target}  need {need}, {len(candidates)} spares available...")
        got = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(_one, k, folder) for k in candidates]
            for fut in as_completed(futs):
                _, _, status = fut.result()
                if status.startswith("ok") or status == "cached":
                    got += 1
                if got >= need:
                    for f in futs: f.cancel()
                    break
        # trim any overshoot from in-flight threads so the count is exact
        extra = sorted(_saved(folder) - saved)
        while len(saved) + len(extra) > target:
            os.remove(os.path.join(ROOT, folder, f"KIC{extra.pop()}.parquet"))
        final = len(_saved(folder))
        tag = "complete" if final >= target else f"SHORT by {target-final} (spares ran out)"
        _say(f"    -> {folder}: {final}/{target}  {tag}")
    _say("top-up done.")


# ---------------- Phase 4: false-positive / blend class (centroid-offset KOIs) ----------------
FP_CAP = 700   # FalsePositive hard-negatives lift precision on EVERY head; own cap, > MAX_PER_CLASS


def falsepos():
    """Download the 'FalsePositive' folder: KOIs dispositioned FALSE POSITIVE with the
    centroid-offset flag (koi_fpflag_co=1) — the signal comes from a *different* star in the
    aperture (background-EB blend). These are transit-LOOKING hard negatives whose only real
    tell is centroid motion, so they're what makes the pixel channel decisive (handoff §8.1).
    Folder name has no Transit/EB/Starspot substring -> label_bits() = all-zeros automatically.

    Expandable + resumable: each run skips KICs already saved or already attempted (recorded in
    _index_fp.csv) and pulls FRESH ones from the ~1.8k available pool, up to FP_CAP saved. Rerun
    (ideally in a plain terminal for a long pass) until it reports the target reached."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    folder = "FalsePositive"
    idx_path = os.path.join(ROOT, "_index_fp.csv")
    print("Phase 4: fetching FALSE POSITIVE KOIs with centroid offset (koi_fpflag_co=1)...")
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
           "select+distinct+kepid+from+cumulative+where+koi_disposition='FALSE+POSITIVE'"
           "+and+koi_fpflag_co=1&format=csv")
    avail = sorted(set(pd.read_csv(url)["kepid"].astype(int)))

    saved = _saved(folder)                                    # parquets already on disk
    prev = pd.read_csv(idx_path) if os.path.exists(idx_path) else \
        pd.DataFrame(columns=["kic", "folder", "status"])
    tried = set(prev["kic"].astype(int))                     # attempted before (landed OR permanent no-data)
    need = FP_CAP - len(saved)
    if need <= 0:
        print(f"  {len(saved)} FP already saved (cap {FP_CAP}) — nothing to do."); return

    candidates = [k for k in avail if k not in saved and k not in tried]
    rng = np.random.default_rng(42)
    if len(candidates) > need * 2:                            # ~60% land -> attempt ~2x the shortfall
        candidates = sorted(rng.choice(candidates, need * 2, replace=False).tolist())
    print(f"  {len(saved)} saved, target {FP_CAP}; attempting {len(candidates)} fresh KICs "
          f"({len(avail)} available, {len(tried)} previously tried)")

    _say(f"\nPhase 4: {len(candidates)} fresh {folder} attempts, target {FP_CAP}, resumable...")
    log, t0, got = [], time.time(), 0

    def _flush():
        pd.concat([prev, pd.DataFrame(log)]).drop_duplicates("kic", keep="last") \
          .to_csv(idx_path, index=False)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one, int(k), folder): k for k in candidates}
        for i, fut in enumerate(as_completed(futs), 1):
            kic, _, status = fut.result()
            log.append({"kic": kic, "folder": folder, "status": status})
            if status.startswith("ok") or status == "cached":
                got += 1
            if i % 20 == 0 or "ok" in status:
                _say(f"  [{i:4d}/{len(candidates)}] KIC{kic} {status}  ({time.time()-t0:.0f}s)")
            if i % 25 == 0:
                _flush()
            if len(saved) + got >= FP_CAP:                    # target reached — stop consuming results
                break
    _flush()
    total = len(_saved(folder))
    _say(f"\ndone. {total} {folder} curves saved (target {FP_CAP})")   # _say is teardown-proof;
    # no trailing print() — a bare print on a closed stdout (backgrounded run) raises and would
    # crash us after the work is done (handoff §2 / §6.5). The log line above is the record.


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    os.makedirs(ROOT, exist_ok=True)
    {"plan": plan, "download": download, "topup": topup, "falsepos": falsepos}[cmd]()
