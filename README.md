# Planetary Transit Detector

A multi-label cascade that reads a star's brightness-over-time (a **light curve**) and
answers three independent yes/no questions with calibrated confidences:

- Is there a **planet transit**?
- Is there an **eclipsing binary** companion?
- Is the star a **spotted rotator**?

It also detects **blends** (a companion star disguised as a planet), flags **novelties**
(anything unlike its training data) for human review, and — via Kepler Target Pixel Files —
performs the centroid check that decides whether a dip came from the target or from a
neighbour in the aperture.

Built for the ISRO Bharatiya Antariksh Hackathon on the *AI-Enabled Detection of Exoplanets*
problem statement.

---

## What's in the box

```
exo/
  config.py        every tunable parameter (ExoConfig dataclass)
  features.py      Stage 0: Lomb-Scargle rotation + BLS dip search
  cascade.py       Stages 1+3: three LightGBM heads + Mahalanobis OOD gate
  pixels.py        Target Pixel File channel: centroid_shift, contratio, rho_consistency
  pipeline.py      Stages 2+4: peel test, verdict combiner, JSONL logging
  __main__.py      CLI: train / tune / features / train-px / predict
build_dataset.py   Dataset builder (Transit / EB / Starspot / FalsePositive folders)
app.py             Streamlit UI (this file)
```

See `App Workflow & Architecture.md` in the design notes for the plain-language walkthrough of
what each stage does physically; see the `Claude method -*.md` handoffs for the technical
build history.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt         # or the deps listed at the bottom of this file

# 2. Build the dataset (one-time; downloads ~2k Kepler light curves)
python build_dataset.py fetch

# 3. Train the cascade
python -m exo train                     # LC-only baseline
python -m exo train --rebuild           # force feature re-extraction

# 4. (Optional) add the pixel channel — needs a separate TPF download
python -m exo.pixels build              # downloads Target Pixel Files, computes centroids
python -m exo train-px                  # trains a pixel-augmented model

# 5. Launch the UI
streamlit run app.py
```

The trained model lives at `exo/models/latest.joblib`. Every prediction the system makes —
CLI or UI — is appended to `exo/predictions.jsonl` with the exact model version, config
fingerprint, and inputs, so any two runs can be compared fairly.

### Shipped checkpoints

Four curated models ship with the repo so the UI dropdown offers a real before/after
comparison. `latest.joblib` is a copy of the newest (v4).

| file | channel | features / data | notes |
| --- | --- | --- | --- |
| `cascade_v1_lightcurve-only.joblib` | flux only | 10 dip + 3 spot | no TPF download needed; runs anywhere |
| `cascade_v2_pixel-hp-baseline.joblib` | + pixel centroid | + HP search | the method-5 deployed baseline |
| `cascade_v3_shape-vectors.joblib` | + pixel centroid | + folded shape vectors, eccentric secondary, rotation harmonics | EB PR-AUC 0.814→0.835, rotating 0.642→0.653 (8-seed) |
| `cascade_v4_more-falsepos.joblib` | + pixel centroid | 670 false positives (was 221) | **current** — transit precision@0.95-recall 0.43→0.55 on a fixed test set |

Other historical checkpoints are regenerable and left gitignored.

---

## The Streamlit UI — user guide

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Dark theme by default (`.streamlit/config.toml`).

### Layout

**Left sidebar — inputs and knobs**

- **Model picker** — defaults to `latest.joblib`. The dropdown lists every trained cascade in
  `exo/models/`. The caption underneath shows the version fingerprint and whether the model
  uses the pixel channel. Expand *Training metrics* to see PR-AUC / recall / Brier for each
  head plus the leave-one-folder-out novelty scores.
- **Input source** — pick one:
  - *Dataset star*: browse the local `dataset/` folders (Transit, EB, Starspot,
    FalsePositive, plus the combined co-occurrence folders).
  - *Upload file*: drop a Parquet or CSV with time + flux columns; the app auto-detects
    columns whose names start with `time` and `flux`.
  - *MAST lookup*: type a target name (e.g. `KIC 8462852`, `TIC 261136679`) and click
    *Fetch*. The result is cached so slider changes don't re-download.
- **Decision thresholds** — one slider per head, seeded from the loaded model's tuned values.
  Move them to explore the precision/recall trade-off live; the head-probability panel and
  verdict update on the same values.
- **Peel test toggle** — enabled by default. When both the transit and EB heads fire, the
  peel test masks the eclipse and re-scores transit on the residual. If it dies, the "planet"
  was the EB's secondary eclipse.

**Center — the graph** (the point of the whole UI)

Three tabs, all rendered in Plotly:

- **Light curve** — raw normalized flux vs time (BJD). Points inside the strongest BLS-found
  dip are highlighted red. A *Detrended* toggle divides out the rotation trend, so a shallow
  planet dip becomes visible under a large starspot wave.
- **Folded on dip period** — every point re-plotted by its phase within the BLS period, with
  the dip parked at phase 0.5. A red overlay is the binned median — a real transit shows a
  flat-bottomed dip, a grazing EB shows a V.
- **Folded on rotation** — same idea folded on the Lomb-Scargle period, in purple.

**Below the graph — verdict internals**

- **Head probabilities** — horizontal bars for `is_transit`, `is_EB`, `is_rotating`. Bar
  color turns green when it clears its threshold; a red tick marks the threshold itself.
- **OOD gate** — the star's Mahalanobis distance to its closest training class, versus the
  95%-inlier-retention threshold. Anything above the threshold gets an *abstain* verdict.
- **Peel test** — only populated when both transit and EB fired. Shows the transit
  probability after the eclipse was peeled off, plus whether it survived the threshold.

**At the bottom — history and internals**

- *Extracted features* — the full feature dict the model saw for this prediction.
- *Prediction history* — the last 200 rows from `predictions.jsonl`, newest first.

### A typical session

1. Pick *Dataset star* → *Transit* → any KIC.
2. Verdict banner reads *"Transit (planet candidate)"* in green. The three probability bars
   show high transit, low EB, low rotation.
3. Switch the tab to *Folded on dip period*. You should see a clean U-shaped dip.
4. Switch model to a non-pixel one (e.g. `cascade_5cf9bf6089-*.joblib`), and the verdict may
   change — that model has never seen centroid features.
5. Drag the *Transit* threshold from 0.35 up to 0.9: transit precision goes up, recall drops,
   the head-probability panel and the verdict update live. Push the *EB* threshold way down
   on a clean Transit and watch the peel test kick in.
6. Try MAST lookup with a known variable (e.g. `KIC 8462852` — Tabby's Star): the OOD gate
   should flag it as out-of-distribution.

---

## CLI reference

```bash
python -m exo features               # build / refresh the feature table only
python -m exo features --rebuild     # force re-extraction (needed after schema changes)
python -m exo train                  # train + save; prints per-head metrics
python -m exo train --rebuild        # rebuild features then train
python -m exo train-px               # train with pixel features (run exo.pixels build first)
python -m exo tune                   # re-pick thresholds from data, re-save
python -m exo predict path.parquet   # one-off prediction with latest.joblib

python -m exo.pixels build           # download TPFs; compute centroid_shift + contratio + rho
python -m exo.pixels topup           # retry MAST failures with fewer workers + cache clear
python -m exo.pixels merge           # join pixel features into the light-curve feature table

python build_dataset.py fetch        # download Transit / EB / Starspot light curves
python build_dataset.py falsepos     # download koi_fpflag_co=1 centroid-offset false positives
```

Every module also has a `python -m exo.<module>` self-check with an `assert`-based sanity
test — run e.g. `python -m exo.cascade` to verify the multi-label plumbing on synthetic data.

---

## How it decides — the pipeline in one page

```
light curve in
    │
    ▼
Stage 0 (features.py)
    Lomb-Scargle rotation search → P_rot, ls_power, acf_peak, rot_amplitude
    Median-filter detrend (flatten the rotation trend)
    Box Least Squares dip search → period, t0, duration, depth, snr, sde
    Odd-even significance, secondary-eclipse SNR, trap shape, ingress fraction
    (optional) Target Pixel File → centroid_shift, contratio, rho_consistency
    │
    ▼
Stage 1 (cascade.py)
    Three calibrated LightGBM heads, independent (multi-label, not softmax):
      is_transit   ← dip features (+ pixel features if enabled)
      is_EB        ← dip features (+ pixel features if enabled)
      is_rotating  ← spot features only
    │
    ▼
Stage 2 (pipeline.py) — peel test, only if BOTH transit and EB fire
    Mask the eclipse cadences, re-extract features on the residual,
    re-score transit head. If it drops below threshold, the "transit"
    was actually the EB's secondary eclipse.
    │
    ▼
Stage 3 (cascade.py) — Mahalanobis OOD gate
    Per-folder Ledoit-Wolf covariance; log-transform heavy-tailed features
    first so a deep-but-real transit isn't a geometric outlier.
    If the minimum distance to any known class > 95%-inlier threshold, abstain.
    │
    ▼
Stage 4 (pipeline.py) — combine and log
    (is_transit, is_EB, is_rotating) → human-readable verdict
    Append the full record (config, features, probs, bits, OOD, peel) to
    predictions.jsonl for reproducibility and post-hoc comparison.
```

---

## Current benchmarks

Deployed model **v4** (`2d2734ed58-cb0b94bb51`) — pixels + HP search + shape vectors, trained
on the expanded false-positive set (670 FPs, up from 221).

**The v4 change — more false positives.** Adding 449 flux-identical hard negatives directly
attacks the transit head's known weakness. Measured on a **fixed test set** (varying only
whether the new FPs are in *training*, so the base rate is identical and PR-AUC is comparable):

| head           | trained without new FPs | trained with them (v4) | delta   |
| -------------- | ----------------------- | ---------------------- | ------- |
| `is_transit`   | 0.547 PR-AUC            | **0.641 PR-AUC**       | +0.094  |
| `is_EB`        | 0.811                   | 0.838                  | +0.027  |
| `is_rotating`  | 0.546                   | 0.541                  | −0.005  |

Transit **precision at 0.95 recall rose 0.43 → 0.55 (+0.12)** on that shared test set — the
biggest single-pass gain on the weakest head. (Note: because v4's own held-out set now
contains many more negatives, its *raw* single-split PR-AUC reads lower than v3's — a base-rate
artifact, not a regression. The fixed-test-set numbers above are the honest comparison.)

**Prior pass (v3) — feature work, robust 8-seed PR-AUC** (group-split by star):

| head           | v2 → v3 PR-AUC        | lever                    |
| -------------- | --------------------- | ------------------------ |
| `is_transit`   | 0.628 (variance halved) | folded shape vectors   |
| `is_EB`        | 0.814 → **0.835**     | shape vectors (U vs V)   |
| `is_rotating`  | 0.642 → 0.653         | rotation harmonics       |

Transit stays recall-first (~0.95 — never miss a planet). The remaining precision ceiling is
centroid-bound: the new FPs still need their TPF centroid features computed
(`python -m exo.pixels build`, resumable) to unlock the pixel channel's full lift on them.
See `Claude method -5.md` for the earlier history.

Dataset: 2,273 stars (one row per KIC, deduped) across 7 folders, including 670
centroid-offset false positives.

---

## Tech stack

**Language:** Python 3.11+
**Numerics:** NumPy, pandas, SciPy
**Astronomy:** `astropy` (Box Least Squares, Lomb-Scargle), `lightkurve` + `astroquery` (MAST
downloads), Kepler / K2 / TESS support via `predict_tic`
**ML:** LightGBM (heads), scikit-learn (isotonic calibration, Ledoit-Wolf Mahalanobis OOD,
group-aware split + CV, RandomizedSearchCV), joblib (serialization)
**Storage:** Parquet (light curves), CSV (feature tables), JSONL (prediction log)
**UI:** Streamlit + Plotly (dark theme, native `base="dark"`)
**No** deep-learning framework, **no** GPU required, **no** database — everything is flat
files and one pip install.

### Install

```bash
pip install numpy pandas scipy astropy lightkurve astroquery \
            scikit-learn lightgbm joblib pyarrow \
            streamlit plotly
```

Or drop this into `requirements.txt` and `pip install -r requirements.txt`.

---

## Reproducing the current model

```bash
# fresh checkout, everything gitignored data-side needs to be regenerated:
python build_dataset.py fetch                     # ~2k Transit/EB/Starspot light curves
python build_dataset.py falsepos                  # centroid-offset false positives
python -m exo features --rebuild                  # build feature table
python -m exo.pixels build                        # TPF centroid channel (long — resumable)
python -m exo train-px                            # trains the pixel-augmented cascade
```

The build is deliberately resumable and teardown-proof — the pixel download in particular
writes an incremental index every 25 stars, so a killed process picks up where it left off
on the next `build`. See `Claude method -3.md` and `Claude method -5.md` for the story of
how those safety nets got there.

---

## Layout

```
.
├── app.py                    Streamlit UI
├── build_dataset.py          dataset builder (Kepler downloads)
├── fetch_transit_lightcurves.py
├── model_lab.ipynb           the original synthetic-model exploration notebook
├── exo/                      the package (see top of README)
├── dataset/                  gitignored — regenerable via build_dataset.py
├── .streamlit/config.toml    dark theme
└── README.md                 you are here
```
