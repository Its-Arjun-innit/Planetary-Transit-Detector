"""CLI: build features + train, or predict one curve.

    python -m exo train                 # build feature table (cached) + train + save
    python -m exo train --rebuild       # force feature re-extraction
    python -m exo tune                  # re-pick thresholds from data, re-save
    python -m exo features              # build feature table only
    python -m exo train-px              # train WITH pixel features (run exo.pixels build first)
    python -m exo predict <parquet>     # predict one curve with the latest model

  Pixel channel (separate, TPF downloads):
    python -m exo.pixels build          # download TPFs -> centroid_shift, contratio, rho
    python -m exo.pixels merge          # join pixel features into the feature table
"""
import sys, warnings
warnings.filterwarnings("ignore")
from .config import ExoConfig
from .features import build_feature_table
from . import pipeline


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    cfg = ExoConfig()

    if cmd == "features":
        build_feature_table(cfg, use_cache="--rebuild" not in sys.argv)

    elif cmd == "train":
        table = build_feature_table(cfg, use_cache="--rebuild" not in sys.argv)
        pipeline.train(cfg, table)

    elif cmd == "tune":
        table = build_feature_table(cfg, use_cache="--rebuild" not in sys.argv)
        pipeline.tune(cfg, table)

    elif cmd == "train-px":
        pipeline.train_pixels(cfg)        # needs exo.pixels build to have run first

    elif cmd == "predict":
        model = pipeline.load_model(cfg)
        rec = pipeline.predict_parquet(sys.argv[2], model, cfg)
        v = rec["verdict"]
        print(f"{rec['input_id']}: {v['label']}")
        print(f"  probs    {rec.get('probs')}")
        print(f"  ood      {rec.get('ood')}")
        if rec.get("peel"):
            print(f"  peel     {rec['peel']}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
