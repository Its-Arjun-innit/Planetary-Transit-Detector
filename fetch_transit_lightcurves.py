"""Fetch confirmed exoplanet host light curves from MAST via lightkurve,
save each star's light curve (time, flux, flux_err) to its own sheet in one xlsx,
labeled 'transit'.

ponytail: capped at MAX_STARS — full 5800-host run takes hours and ~tens of GB;
raise the cap when you actually want it. Excel sheet-name limit is 31 chars and
xlsx supports very many sheets but UI gets unusable past a few hundred.
"""
import re, sys, warnings, time
warnings.filterwarnings("ignore")
import pandas as pd
import lightkurve as lk

MAX_STARS   = 30                                   # ponytail: bump for a longer run
OUT_XLSX    = r"D:\Codes\ISRO_Hackathon\transit_lightcurves.xlsx"
MAX_ROWS    = 200_000                              # per sheet; downsample if longer

def sheet_name(s, used):
    n = re.sub(r"[\\/*?:\[\]]", "_", s)[:31] or "star"
    base, i = n, 1
    while n in used:
        suf = f"_{i}"; n = base[:31-len(suf)] + suf; i += 1
    used.add(n); return n

print("fetching confirmed planet host list from NASA Exoplanet Archive...")
url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
       "query=select+distinct(hostname)+from+ps&format=csv")
hosts = pd.read_csv(url)["hostname"].dropna().unique().tolist()
print(f"  {len(hosts)} hosts total; processing first {MAX_STARS}")

writer = pd.ExcelWriter(OUT_XLSX, engine="openpyxl")
manifest, used, ok = [], set(), 0

for i, star in enumerate(hosts[:MAX_STARS], 1):
    t0 = time.time()
    try:
        sr = lk.search_lightcurve(star)
        if len(sr) == 0:
            print(f"[{i:3d}/{MAX_STARS}] {star:25s} -> no data");
            manifest.append({"host": star, "label": "transit", "sheet": "", "n_points": 0, "mission": "", "status": "no data"})
            continue
        lc = sr[0].download().remove_nans().normalize()
        t, f = lc.time.value, lc.flux.value
        fe = getattr(lc, "flux_err", None)
        fe = fe.value if fe is not None else [None]*len(t)
        if len(t) > MAX_ROWS:                       # ponytail: stride-downsample big curves
            step = len(t)//MAX_ROWS + 1
            t, f, fe = t[::step], f[::step], fe[::step]
        df = pd.DataFrame({"time_bjd": t, "flux": f, "flux_err": fe,
                           "label": "transit", "host": star,
                           "mission": getattr(lc, "mission", [""])[0] if hasattr(lc, "mission") else ""})
        sn = sheet_name(star, used)
        df.to_excel(writer, sheet_name=sn, index=False)
        manifest.append({"host": star, "label": "transit", "sheet": sn,
                         "n_points": len(df), "mission": str(df["mission"].iloc[0]),
                         "status": "ok"})
        ok += 1
        print(f"[{i:3d}/{MAX_STARS}] {star:25s} -> {len(df):6d} pts  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[{i:3d}/{MAX_STARS}] {star:25s} -> ERROR {type(e).__name__}: {e}")
        manifest.append({"host": star, "label": "transit", "sheet": "", "n_points": 0, "mission": "", "status": f"error: {type(e).__name__}"})

pd.DataFrame(manifest).to_excel(writer, sheet_name="_index", index=False)
writer.close()
print(f"\ndone. {ok}/{MAX_STARS} stars saved -> {OUT_XLSX}")
