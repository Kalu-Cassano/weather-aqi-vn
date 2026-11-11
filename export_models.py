"""Copy the trained models into cloud_models/ in a compact binary format (.ubj),
so GitHub Actions can use them to make forecasts every hour.

Run after every python train.py:

    python export_models.py
"""
import json
import shutil

import config

SRC = config.ROOT / "models"
DST = config.ROOT / "cloud_models"
LIMIT_MB = 50  # GitHub warns above 50 MB and blocks files above 100 MB

meta_path = SRC / "meta.json"
if not meta_path.exists():
    raise SystemExit("models/meta.json not found. Run python train.py first.")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
if meta.get("model") != "XGBoost":
    raise SystemExit("Only XGBoost models can be exported.")

import xgboost as xgb  # noqa: E402

DST.mkdir(exist_ok=True)
total = 0.0
for t in meta["targets"]:
    for h in meta["horizons"]:
        m = xgb.XGBRegressor()
        m.load_model(SRC / f"{t}_h{h}.json")
        out = DST / f"{t}_h{h}.ubj"
        m.save_model(out)
        mb = out.stat().st_size / 1e6
        total += mb
        flag = "  <-- TOO BIG" if mb > LIMIT_MB else ""
        print(f"{out.name:<24} {mb:6.1f} MB{flag}")
shutil.copy2(meta_path, DST / "meta.json")
print(f"Total {total:.1f} MB. Commit the folder cloud_models/ to GitHub.")
