"""Streamlit UI over exo.pipeline — the product surface the package was designed for.

    streamlit run app.py

Sidebar: model pick + input source + decision knobs. Center: the light curve.
Everything predictive goes through pipeline.predict_lightcurve (logged, comparable).
"""
import glob, json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from exo.config import ExoConfig
from exo import pipeline
from exo.features import _flatten, eclipse_points

st.set_page_config(page_title="Exo Cascade", layout="wide", initial_sidebar_state="expanded")
CFG = ExoConfig()
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=30, b=40), height=430)


# ---------------- cached backends ----------------
@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str):
    return pipeline.load_model(CFG, path)


@st.cache_data(show_spinner=False)
def load_parquet(path: str):
    df = pd.read_parquet(path)
    return df["time_bjd"].values, df["flux"].values, f"KIC{int(df['kic'].iloc[0])}"


@st.cache_data(show_spinner=False)
def load_upload(name: str, raw: bytes):
    import io
    try:
        df = pd.read_parquet(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw))
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in cols if c.startswith("time")), df.columns[0])
    fcol = next((cols[c] for c in cols if c.startswith("flux") and "err" not in c), df.columns[1])
    return df[tcol].values.astype(float), df[fcol].values.astype(float), name


@st.cache_data(show_spinner="Fetching from MAST (can take a minute)…")
def fetch_mast(target: str):
    import lightkurve as lk
    sr = lk.search_lightcurve(target)
    if len(sr) == 0:
        return None
    lc = sr[0].download().remove_nans().normalize()
    return lc.time.value, np.asarray(lc.flux.value, float), str(target)


@st.cache_data(show_spinner="Running cascade…")
def run_predict(model_version, input_id, time, flux, thr_t, thr_e, thr_r, do_peel, _model):
    _model.cfg.thr_transit, _model.cfg.thr_eb, _model.cfg.thr_rotating = thr_t, thr_e, thr_r
    px = None
    if _model.cfg.use_pixels and input_id.startswith("KIC"):
        px = pipeline._lookup_pixels(int(input_id[3:]))   # pixel-trained model: real TPF features
    return pipeline.predict_lightcurve(time, flux, _model, input_id=input_id,
                                       source="streamlit", do_peel=do_peel, pixel_feats=px)


def binned_median(phase, flux, nbins=80):
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    med = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            med[b] = np.median(flux[m])
    return (edges[:-1] + edges[1:]) / 2, med


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("Exo Cascade")

    latest = os.path.join(CFG.model_dir, "latest.joblib")
    if not os.path.exists(latest):
        st.error(f"No trained model at `{latest}`. Run `python -m exo train` first, "
                 "or commit `exo/models/latest.joblib` to the deploy repo.")
        st.stop()
    models = sorted(glob.glob(os.path.join(CFG.model_dir, "cascade_*.joblib")))
    labels = [os.path.basename(m) for m in models]
    pick = st.selectbox("Model", ["latest.joblib"] + labels)
    model = get_model(latest if pick == "latest.joblib" else os.path.join(CFG.model_dir, pick))
    st.caption(f"version `{model.model_version}` · pixels {'on' if model.cfg.use_pixels else 'off'}")
    with st.expander("Training metrics"):
        st.json(model.metrics)

    st.divider()
    has_dataset = os.path.isdir(CFG.dataset_root)
    sources = (["Dataset star"] if has_dataset else []) + ["Upload file", "MAST lookup"]
    src = st.radio("Input", sources)
    curve = None
    if src == "Dataset star":
        folders = sorted(d for d in os.listdir(CFG.dataset_root)
                         if os.path.isdir(os.path.join(CFG.dataset_root, d)))
        folder = st.selectbox("Class folder", folders)
        files = sorted(glob.glob(os.path.join(CFG.dataset_root, folder, "*.parquet")))
        f = st.selectbox("Star", files, format_func=os.path.basename)
        if f:
            curve = load_parquet(f)
    elif src == "Upload file":
        up = st.file_uploader("Parquet or CSV with time + flux columns", type=["parquet", "csv"])
        if up:
            curve = load_upload(up.name, up.getvalue())
    else:
        target = st.text_input("Target (e.g. `KIC 8462852`, `TIC 261136679`)")
        if st.button("Fetch") and target.strip():
            st.session_state["mast_target"] = target.strip()
        if st.session_state.get("mast_target"):
            curve = fetch_mast(st.session_state["mast_target"])
            if curve is None:
                st.error("No light curve found on MAST.")
    st.divider()

    st.subheader("Decision thresholds")
    thr_t = st.slider("Transit", 0.0, 1.0, float(model.cfg.thr_transit), 0.01)
    thr_e = st.slider("Eclipsing binary", 0.0, 1.0, float(model.cfg.thr_eb), 0.01)
    thr_r = st.slider("Rotating", 0.0, 1.0, float(model.cfg.thr_rotating), 0.01)
    do_peel = st.checkbox("Peel test on transit+EB blends", True)

# ---------------- main ----------------
if curve is None:
    st.info("Pick a star, upload a curve, or fetch one from MAST — the verdict and plots appear here.")
    st.stop()

time_arr, flux_arr, input_id = curve
rec = run_predict(model.model_version, input_id, time_arr, flux_arr,
                  thr_t, thr_e, thr_r, do_peel, model)
v = rec["verdict"]

if v.get("abstain") and "usable" in v["label"]:
    st.error(f"**{input_id}** — {v['label']}")
    st.stop()

color = ("orange" if v["abstain"] else "gold" if v["blend"] else
         "#4da6ff" if "binary" in v["label"] else
         "#c792ea" if "Rotating" in v["label"] else
         "limegreen" if "Transit" in v["label"] else "gray")
st.markdown(f"### {input_id} &nbsp;→&nbsp; <span style='color:{color}'>**{v['label']}**</span>",
            unsafe_allow_html=True)
if v.get("note"):
    st.caption(v["note"])

feats = rec["features"]
period, t0, P_rot = feats.get("period"), feats.get("t0"), feats.get("P_rot")
dur_d = (feats.get("duration_hr") or 0) / 24
good = np.isfinite(time_arr) & np.isfinite(flux_arr)
t, f = time_arr[good], flux_arr[good]
f = f / np.median(f)

tab_lc, tab_dip, tab_rot = st.tabs(["Light curve", "Folded on dip period", "Folded on rotation"])

with tab_lc:
    show_flat = st.toggle("Detrended (rotation removed)", False)
    y = _flatten(f, CFG.flatten_window) if show_flat else f
    fig = go.Figure()
    fig.add_scattergl(x=t, y=y, mode="markers", marker=dict(size=3, color="#7fb3d5"),
                      name="flux")
    if period and t0 and dur_d > 0:
        m = eclipse_points(t, period, t0, dur_d)
        fig.add_scattergl(x=t[m], y=y[m], mode="markers",
                          marker=dict(size=4, color="crimson"), name="in dip")
    fig.update_layout(**PLOT, xaxis_title="time (BJD)", yaxis_title="normalized flux",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

with tab_dip:
    if period and t0:
        fl = _flatten(f, CFG.flatten_window)
        ph = ((t - t0) / period + 0.5) % 1.0
        fig = go.Figure()
        fig.add_scattergl(x=ph, y=fl, mode="markers",
                          marker=dict(size=3, color="#566573"), name="flux")
        bx, by = binned_median(ph, fl)
        fig.add_scatter(x=bx, y=by, mode="lines", line=dict(color="crimson", width=2),
                        name="binned median")
        fig.update_layout(**PLOT, xaxis_title=f"phase (P = {period:.4f} d, dip at 0.5)",
                          yaxis_title="detrended flux", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dip period found.")

with tab_rot:
    if P_rot:
        ph = (t / P_rot) % 1.0
        fig = go.Figure()
        fig.add_scattergl(x=ph, y=f, mode="markers",
                          marker=dict(size=3, color="#566573"), name="flux")
        bx, by = binned_median(ph, f)
        fig.add_scatter(x=bx, y=by, mode="lines", line=dict(color="#c792ea", width=2),
                        name="binned median")
        fig.update_layout(**PLOT, xaxis_title=f"phase (P_rot = {P_rot:.3f} d)",
                          yaxis_title="normalized flux", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rotation period found.")

# ---------------- verdict internals ----------------
c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    st.subheader("Head probabilities")
    names = ["is_transit", "is_EB", "is_rotating"]
    probs = [rec["probs"][n] for n in names]
    thrs = [thr_t, thr_e, thr_r]
    bits = [rec["bits_final"][n] for n in names]
    fig = go.Figure()
    fig.add_bar(y=names, x=probs, orientation="h",
                marker_color=["limegreen" if b else "#5d6d7e" for b in bits])
    fig.add_scatter(y=names, x=thrs, mode="markers", name="threshold",
                    marker=dict(symbol="line-ns", size=22, color="crimson",
                                line=dict(width=3, color="crimson")))
    fig.update_layout(**{**PLOT, "height": 220}, xaxis=dict(range=[0, 1]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("OOD gate")
    ood = rec["ood"]
    st.metric("Mahalanobis score", ood["score"],
              delta=f"threshold {ood['threshold']}", delta_color="off")
    st.markdown(":red[**FLAGGED — abstain**]" if ood["flagged"] else ":green[in distribution]")

with c3:
    st.subheader("Peel test")
    if rec.get("peel"):
        p = rec["peel"]
        st.metric("Transit prob after removing EB", p["transit_prob_after_peel"])
        st.markdown(":green[transit survived]" if p["survived"]
                    else ":red[dip was the EB's secondary]")
    else:
        st.caption("Not triggered (needs transit + EB both above threshold).")

with st.expander("Extracted features"):
    st.dataframe(pd.DataFrame([{k: v for k, v in feats.items() if v is not None}]).T
                 .rename(columns={0: "value"}), use_container_width=True)

with st.expander("Prediction history (predictions.jsonl)"):
    if os.path.exists(CFG.log_path):
        lines = open(CFG.log_path, encoding="utf-8").read().splitlines()[-200:]
        hist = []
        for ln in lines:
            try:
                r = json.loads(ln)
                hist.append({"time": r.get("timestamp"), "input": r.get("input_id"),
                             "verdict": r.get("verdict", {}).get("label"),
                             "model": r.get("model_version"),
                             **{k: r.get("probs", {}).get(k) for k in
                                ("is_transit", "is_EB", "is_rotating")}})
            except Exception:
                pass
        st.dataframe(pd.DataFrame(hist[::-1]), use_container_width=True)
    else:
        st.caption("No predictions logged yet.")
