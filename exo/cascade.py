"""Stages 1 + 3 — the trained model.

Three independent calibrated binary heads (multi-label, because the phenomena co-occur):
    is_transit, is_EB   -> on dip features
    is_rotating         -> on spot features
plus a per-class (per-folder) Mahalanobis OOD gate that can abstain.

No 6-way softmax: that would force mutual exclusivity onto independent phenomena and
starve the 48/50-sample overlap folders. As multi-label they're just dual-positive rows.
"""
from __future__ import annotations
import hashlib, json, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, recall_score, brier_score_loss
import joblib

from .config import (ExoConfig, DIP_FEATURES, SPOT_FEATURES, HEAVY_FEATURES,
                     dip_features, all_features)

HEADS = {                                   # head -> (feature group, threshold attr)
    "is_transit":  ("dip",  "thr_transit"),
    "is_EB":       ("dip",  "thr_eb"),
    "is_rotating": ("spot", "thr_rotating"),
}


def _base_estimator(cfg: ExoConfig, params: dict = None):
    cw = "balanced" if cfg.balanced_weights else None
    params = params or {}
    try:
        from lightgbm import LGBMClassifier
        kw = dict(n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
                  class_weight=cw, random_state=cfg.seed, verbose=-1)
        kw.update(params)                                  # tuned params override defaults (3a)
        return LGBMClassifier(**kw)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=cfg.n_estimators,
                                              learning_rate=cfg.learning_rate,
                                              class_weight=cw, random_state=cfg.seed)


# LightGBM search space — small, tabular-physics-appropriate. Regularization + capacity knobs that
# matter at ~1.8k rows; learning_rate/n_estimators left to cfg so calibration stays comparable.
HP_SPACE = {
    "num_leaves":        [15, 31, 63, 127],
    "max_depth":         [-1, 4, 6, 8, 12],
    "min_child_samples": [5, 10, 20, 40],
    "subsample":         [0.7, 0.85, 1.0],
    "colsample_bytree":  [0.7, 0.85, 1.0],
    "reg_lambda":        [0.0, 1.0, 5.0],
}


def _logtf(X, heavy_idx):
    Z = np.asarray(X, float).copy()
    Z[:, heavy_idx] = np.log1p(np.clip(Z[:, heavy_idx], 0, None))
    return Z


class Cascade:
    def __init__(self, cfg: ExoConfig):
        self.cfg = cfg
        self.heads = {}            # name -> fitted (calibrated) classifier
        self.head_feats = {}       # name -> resolved feature columns (cfg-dependent)
        self.all_feats = None      # full vector the OOD gate scores over
        self.heavy_idx = None
        self.ood_medians = None    # impute NaN (e.g. missing pixel features) before OOD scaler
        self.scaler = None
        self.cov = {}              # folder -> LedoitWolf
        self.ood_thr = None
        self.metrics = {}
        self.hp = {}               # head -> tuned LightGBM params (3a); empty = library defaults
        self.model_version = None

    def _group_feats(self, group):
        return dip_features(self.cfg) if group == "dip" else SPOT_FEATURES

    # ---------------- train ----------------
    def fit(self, table: pd.DataFrame):
        cfg = self.cfg
        self.all_feats = all_features(cfg)
        self.heavy_idx = [self.all_feats.index(c) for c in HEAVY_FEATURES]
        self.head_feats = {name: self._group_feats(g) for name, (g, _) in HEADS.items()}
        self.ood_medians = table[self.all_feats].median()
        if cfg.hp_search and not self.hp:
            self.tune_hyperparams(table)        # tune on the (improved) features before fitting heads

        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
        tr_i, te_i = next(gss.split(table, groups=table["kic"].values))
        tr, te = table.iloc[tr_i], table.iloc[te_i]
        assert not (set(tr["kic"]) & set(te["kic"])), "star leak across split"

        for name, (group, thr_attr) in HEADS.items():
            feats = self.head_feats[name]
            thr = getattr(cfg, thr_attr)
            clf = self._fit_head(tr, feats, name)
            p = clf.predict_proba(te[feats].values)[:, 1]
            y = te[name].values
            self.metrics.setdefault(name, {}).update({       # update, not replace: keep hp_cv_ap (3a)
                "n_pos": int(table[name].sum()),
                "pr_auc": round(float(average_precision_score(y, p)), 3),
                "recall_at_thr": round(float(recall_score(y, (p >= thr).astype(int),
                                                          zero_division=0)), 3),
                "brier": round(float(brier_score_loss(y, p)), 3),
                "threshold": thr,
            })
            self.heads[name] = self._fit_head(table, feats, name)   # deploy: refit on all data

        self._fit_ood(table)
        self.metrics["ood"] = self._loo_ood_eval(table)        # honest novelty proxy
        self.model_version = self._version(table)
        return self

    def _fit_head(self, frame, feats, name):
        est = _base_estimator(self.cfg, self.hp.get(name))     # gradient-boosted trees handle NaN natively
        if self.cfg.calibrate:
            est = CalibratedClassifierCV(est, method="isotonic", cv=3)
        est.fit(frame[feats].values, frame[name].values)
        return est

    def tune_hyperparams(self, table: pd.DataFrame, n_iter: int = 25):
        """Per-head group-aware RandomizedSearchCV over HP_SPACE, scoring average-precision (PR-AUC) —
        the metric that matters under class imbalance. Tunes the RAW estimator (calibration wraps the
        winner afterward). Stores best params in self.hp so _fit_head picks them up. Run AFTER the
        feature/centroid improvements so the search optimizes on the richer inputs (3a synergy)."""
        from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score
        groups = table["kic"].values
        cv = GroupKFold(n_splits=3)
        for name, (group, _) in HEADS.items():
            feats = self.head_feats[name]
            X, y = table[feats].values, table[name].values
            # default-params CV score is the bar the search must clear
            base_ap = float(np.mean(cross_val_score(_base_estimator(self.cfg), X, y, groups=groups,
                                                     cv=cv, scoring="average_precision")))
            search = RandomizedSearchCV(
                _base_estimator(self.cfg), HP_SPACE, n_iter=n_iter, scoring="average_precision",
                cv=cv, random_state=self.cfg.seed, n_jobs=self.cfg.n_jobs, error_score=0.0)
            search.fit(X, y, groups=groups)
            # only adopt tuned params if they actually beat the default — HP can help or no-op, never hurt
            if search.best_score_ > base_ap:
                self.hp[name] = {k: search.best_params_[k] for k in HP_SPACE}
                self.metrics.setdefault(name, {})["hp_cv_ap"] = round(float(search.best_score_), 3)
            else:
                self.metrics.setdefault(name, {})["hp_cv_ap"] = f"default ({base_ap:.3f} >= {search.best_score_:.3f})"
        return self.hp

    def _impute(self, X, medians):
        X = np.atleast_2d(np.asarray(X, float)).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(np.asarray(medians, float), np.where(bad)[1])
        return X

    def _fit_ood(self, table):
        X = _logtf(self._impute(table[self.all_feats].values, self.ood_medians), self.heavy_idx)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.cov = {f: LedoitWolf().fit(Z[table["folder"].values == f])
                    for f in table["folder"].unique()}
        self.ood_thr = float(np.quantile(self._ood_raw(table[self.all_feats].values),
                                         self.cfg.inlier_retention))

    def _ood_raw(self, X):
        Z = self.scaler.transform(_logtf(self._impute(X, self.ood_medians), self.heavy_idx))
        return np.min([c.mahalanobis(Z) for c in self.cov.values()], axis=0)

    def _loo_ood_eval(self, table):
        """Leave-one-folder-out: hold each folder out as 'novel', refit OOD on the rest,
        measure how well the score separates it. Honest novelty check without needing
        externally-labeled novelties."""
        from sklearn.metrics import roc_auc_score
        feats, hidx = self.all_feats, self.heavy_idx
        aucs = {}
        for held in table["folder"].unique():
            ins = table[table["folder"] != held]
            med = ins[feats].median()
            X = _logtf(self._impute(ins[feats].values, med), hidx)
            sc = StandardScaler().fit(X)
            cov = {f: LedoitWolf().fit(sc.transform(X)[ins["folder"].values == f])
                   for f in ins["folder"].unique()}
            def score(M):
                Z = sc.transform(_logtf(self._impute(M, med), hidx))
                return np.min([c.mahalanobis(Z) for c in cov.values()], axis=0)
            s_in = score(ins[feats].values)
            s_out = score(table[table["folder"] == held][feats].values)
            y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
            try:
                aucs[held] = round(float(roc_auc_score(y, np.r_[s_in, s_out])), 3)
            except Exception:
                aucs[held] = None
        return {"loo_novelty_auc": aucs}

    def tune_thresholds(self, table: pd.DataFrame, transit_min_recall: float = 0.95) -> dict:
        """Pick each head's decision threshold from its held-out Precision-Recall curve
        instead of hand-setting it. Transit: max precision at recall >= target (keeps the
        false-negative guarantee). EB / rotating: max F1. Updates cfg + version, heads unchanged."""
        from sklearn.metrics import precision_recall_curve
        cfg = self.cfg
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
        tr_i, te_i = next(gss.split(table, groups=table["kic"].values))
        tr, te = table.iloc[tr_i], table.iloc[te_i]
        chosen = {}
        for name, (group, thr_attr) in HEADS.items():
            feats = self.head_feats[name]
            clf = self._fit_head(tr, feats, name)
            p = clf.predict_proba(te[feats].values)[:, 1]
            y = te[name].values
            prec, rec, thsh = precision_recall_curve(y, p)
            prec, rec = prec[:-1], rec[:-1]                 # align with thresholds
            if name == "is_transit":
                ok = rec >= transit_min_recall
                t = float(thsh[int(np.argmax(np.where(ok, prec, -1)))]) if ok.any() \
                    else getattr(cfg, thr_attr)
            else:
                f1 = 2 * prec * rec / (prec + rec + 1e-12)
                t = float(thsh[int(np.argmax(f1))])
            chosen[name] = round(t, 3)
            setattr(cfg, thr_attr, round(t, 3))
            self.metrics[name]["threshold"] = round(t, 3)
            self.metrics[name]["recall_at_thr"] = round(float(recall_score(
                y, (p >= t).astype(int), zero_division=0)), 3)
            self.metrics[name]["precision_at_thr"] = round(float(
                (p >= t)[y == 1].sum() / max(1, (p >= t).sum())), 3)
        self.model_version = self._version(table)           # thresholds feed the fingerprint
        return chosen

    def _version(self, table):
        counts = table["folder"].value_counts().sort_index().to_dict()
        data_fp = hashlib.sha1(json.dumps({"n": len(table), **counts},
                                          sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.cfg.fingerprint()}-{data_fp}"

    # ---------------- predict ----------------
    def predict(self, feat: dict) -> dict:
        """One feature dict -> per-head calibrated probs, thresholded bits, OOD verdict.
        Combiner/peel/logging live in pipeline.py — this stays pure."""
        probs, bits = {}, {}
        for name, (group, thr_attr) in HEADS.items():
            feats = self.head_feats[name]
            x = np.array([[feat.get(c, np.nan) for c in feats]], float)   # missing pixel cols -> NaN, handled
            p = float(self.heads[name].predict_proba(x)[0, 1])
            probs[name] = p
            bits[name] = int(p >= getattr(self.cfg, thr_attr))
        score = float(self._ood_raw([feat.get(c, np.nan) for c in self.all_feats])[0])
        return {
            "probs": probs, "bits": bits,
            "ood": {"score": round(score, 2), "threshold": round(self.ood_thr, 2),
                    "flagged": bool(score > self.ood_thr)},
            "model_version": self.model_version,
        }

    # ---------------- persist ----------------
    def save(self, path=None) -> str:
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        path = path or os.path.join(self.cfg.model_dir, f"cascade_{self.model_version}.joblib")
        joblib.dump(self, path)
        joblib.dump(self, os.path.join(self.cfg.model_dir, "latest.joblib"))
        return path

    @staticmethod
    def load(path) -> "Cascade":
        return joblib.load(path)


def train(cfg: ExoConfig = None, table: pd.DataFrame = None) -> Cascade:
    from .features import build_feature_table
    cfg = cfg or ExoConfig()
    table = table if table is not None else build_feature_table(cfg)
    return Cascade(cfg).fit(table)


if __name__ == "__main__":
    # self-check: each bit drives ONE dedicated, non-conflicting feature, so any label
    # combination is separable -> proves the multi-label plumbing learns each head.
    import warnings as _w; _w.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    rows = []
    folders = ["Transit", "EB", "Starspot", "Transit+EB", "EB+Starspot", "Starspot+Transit"]
    for i in range(600):
        folder = folders[i % 6]
        T, E, R = ("Transit" in folder), ("EB" in folder), ("Starspot" in folder)
        rows.append(dict(
            depth_ppm=rng.normal(3000, 300),
            duration_hr=rng.normal(8 if T else 4, 0.5),               # transit -> longer
            odd_even_diff=abs(rng.normal(0, .02)),
            odd_even_sig=abs(rng.normal(8 if E else 0.5, 1)),         # EB -> significant odd-even
            secondary_depth=abs(rng.normal(800 if E else 0, 50)),     # EB -> secondary eclipse
            secondary_snr=rng.normal(10 if E else 0, 1.5),           # EB -> significant secondary
            trap_shape=rng.beta(2, 6), ingress_frac=rng.beta(2, 6),
            snr=rng.normal(30, 3), sde=rng.normal(15, 2),
            ls_power=rng.normal(0.9 if R else 0.1, .03),              # rotation -> LS power
            acf_peak=rng.normal(0.9 if R else 0.1, .03),
            rot_amplitude=rng.normal(0.02 if R else 0.002, .001),
            folder=folder, kic=i,
            **{"is_transit": int(T), "is_EB": int(E), "is_rotating": int(R)}))
    t = pd.DataFrame(rows)
    m = Cascade(ExoConfig(n_estimators=80)).fit(t)
    print("model_version", m.model_version)
    for k, v in m.metrics.items():
        print(f"  {k}: {v}")
    for name, mt in m.metrics.items():
        if name != "ood":
            assert mt["pr_auc"] > 0.8, f"{name} head failed to separate ({mt['pr_auc']})"
    print("cascade self-check OK")
