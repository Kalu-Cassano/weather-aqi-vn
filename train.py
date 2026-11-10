"""BƯỚC TRAIN MÔ HÌNH.

Với mỗi mục tiêu (nhiệt độ, PM2.5) và mỗi tầm dự báo (1, 3, 6 giờ tới), so sánh 3 cách:
  1. Baseline "giữ nguyên": giờ sau = giờ này (mốc tối thiểu phải vượt qua)
     Baseline "cùng giờ hôm qua": lấy giá trị đúng giờ đó của ngày hôm trước
  2. Ridge Regression (mô hình tuyến tính, đơn giản)
  3. XGBoost (mô hình chính, chạy GPU nếu có)

Chia dữ liệu THEO THỜI GIAN (không chia ngẫu nhiên, vì như vậy là "nhìn trộm tương lai"):
  train  : toàn bộ trước giai đoạn validation
  val    : 60 ngày trước tập test (để dừng train sớm, chống overfit)
  test   : 180 ngày cuối cùng (chỉ dùng để chấm điểm)

Kết quả:
  models/<mục tiêu>_h<giờ>.json   mô hình XGBoost
  models/meta.json                danh sách đặc trưng, ngày train...
  reports/metrics.csv             bảng điểm MAE / RMSE / R² của mọi mô hình
  reports/feature_importance.csv  đặc trưng quan trọng nhất
  reports/test_predictions.csv    dự báo vs thực tế trên tập test (để vẽ biểu đồ)

Cách chạy:
    python train.py
    python train.py --quick     # train nhanh để thử (ít cây hơn)
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from features import HORIZONS, TARGETS, feature_columns, load_table, make_features
from om_client import log

MODEL_DIR = config.ROOT / "models"
REPORT_DIR = config.ROOT / "reports"
TEST_DAYS = 180
VAL_DAYS = 60
# Tỉnh dùng để lưu dự báo mẫu cho biểu đồ (giữ file nhỏ)
SAMPLE_LOCATIONS = ["tp_hcm", "ha_noi", "da_nang"]


def get_model(quick: bool):
    """XGBoost nếu đã cài (ưu tiên GPU), nếu không thì dùng mô hình tương tự của scikit-learn."""
    n = 300 if quick else 1500
    try:
        import xgboost as xgb
        params = dict(n_estimators=n, learning_rate=0.05, max_depth=8, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                      early_stopping_rounds=50, eval_metric="mae")
        # Thử GPU trên một tập nhỏ; lỗi thì quay về CPU
        try:
            probe = xgb.XGBRegressor(n_estimators=2, device="cuda")
            probe.fit(np.random.rand(50, 3), np.random.rand(50), verbose=False)
            params["device"] = "cuda"
            device = "GPU (CUDA)"
        except Exception:
            params["n_jobs"] = -1
            device = "CPU"
        return "XGBoost", lambda: xgb.XGBRegressor(**params), device
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return ("HistGradientBoosting",
                lambda: HistGradientBoostingRegressor(max_iter=n // 3, learning_rate=0.1,
                                                      early_stopping=True, random_state=0),
                "CPU")


def scores(y_true, y_pred):
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    log("Đọc dữ liệu đã xử lý...")
    raw = load_table(config.DATA_DIR / "processed" / "merged")
    log(f"{len(raw):,} dòng, {raw['location'].nunique()} tỉnh. Tạo đặc trưng...")
    df = make_features(raw)
    del raw

    model_name, new_model, device = get_model(args.quick)
    log(f"Mô hình chính: {model_name} trên {device}")

    metrics_rows, importance_rows, sample_preds = [], [], []
    meta = {"trained_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name, "device": device, "horizons": HORIZONS,
            "targets": TARGETS, "features": {}, "data_rows": int(len(df))}

    g = df.groupby("location", sort=False)
    for target in TARGETS:
        feats = feature_columns(target, df.columns)
        meta["features"][target] = feats

        for h in HORIZONS:
            tag = f"{target}_h{h}"
            y = g[target].shift(-h)  # giá trị thật ở h giờ sau
            # Baseline mạnh hơn: giá trị cùng giờ của hôm trước (tức t + h - 24)
            yday = g[target].shift(24 - h)
            data = df[["location", "time", target]].assign(y=y, yday=yday)
            X_all = df[feats]
            mask = data["y"].notna() & data["yday"].notna() & X_all.notna().all(axis=1)
            data, X_all = data[mask], X_all[mask]

            t_end = data["time"].max()
            test_start = t_end - pd.Timedelta(days=TEST_DAYS)
            val_start = test_start - pd.Timedelta(days=VAL_DAYS)
            tr = data["time"] < val_start
            va = (data["time"] >= val_start) & (data["time"] < test_start)
            te = data["time"] >= test_start

            Xtr, ytr = X_all[tr], data.loc[tr, "y"]
            Xva, yva = X_all[va], data.loc[va, "y"]
            Xte, yte = X_all[te], data.loc[te, "y"]
            log(f"[{tag}] train {len(Xtr):,} | val {len(Xva):,} | test {len(Xte):,} dòng, "
                f"{len(feats)} đặc trưng")

            preds = {}
            # 1) Baseline: giữ nguyên giá trị hiện tại
            preds["Baseline (persistence)"] = data.loc[te, target].to_numpy()
            # 1b) Baseline: lấy giá trị cùng giờ hôm qua (bắt được chu kỳ ngày đêm)
            preds["Baseline (same hour yesterday)"] = data.loc[te, "yday"].to_numpy()

            # 2) Ridge
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            ridge.fit(Xtr, ytr)
            preds["Ridge Regression"] = ridge.predict(Xte)

            # 3) XGBoost
            t1 = time.time()
            model = new_model()
            if model_name == "XGBoost":
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                model.save_model(MODEL_DIR / f"{tag}.json")
                imp = model.feature_importances_
            else:
                model.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
                import joblib
                joblib.dump(model, MODEL_DIR / f"{tag}.joblib")
                imp = np.zeros(len(feats))
            preds[model_name] = model.predict(Xte)
            log(f"[{tag}] train {model_name} mất {time.time() - t1:.0f} giây")

            base_mae = None
            for name, p in preds.items():
                s = scores(yte, p)
                if base_mae is None:
                    base_mae = s["MAE"]
                s["improve_vs_baseline_%"] = round(100 * (base_mae - s["MAE"]) / base_mae, 1)
                metrics_rows.append({"target": target, "horizon_h": h, "model": name, **s,
                                     "test_rows": int(len(yte))})
                log(f"[{tag}] {name:<24} MAE={s['MAE']:.3f}  RMSE={s['RMSE']:.3f}  "
                    f"R²={s['R2']:.3f}  (tốt hơn baseline {s['improve_vs_baseline_%']}%)")

            for f_name, v in sorted(zip(feats, imp), key=lambda x: -x[1])[:20]:
                importance_rows.append({"target": target, "horizon_h": h,
                                        "feature": f_name, "importance": round(float(v), 5)})

            # Lưu dự báo mẫu 30 ngày cuối của vài tỉnh để vẽ biểu đồ
            sm = data.loc[te, ["location", "time"]].copy()
            sm["target"], sm["horizon_h"] = target, h
            sm["target_time"] = sm["time"] + pd.Timedelta(hours=h)
            sm["actual"] = yte.to_numpy()
            sm["baseline"] = preds["Baseline (persistence)"]
            sm["baseline_yesterday"] = preds["Baseline (same hour yesterday)"]
            sm["model"] = preds[model_name]
            keep = sm["location"].isin(SAMPLE_LOCATIONS) & (sm["time"] >= t_end - pd.Timedelta(days=30))
            sample_preds.append(sm[keep])

    pd.DataFrame(metrics_rows).to_csv(REPORT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).to_csv(REPORT_DIR / "feature_importance.csv", index=False,
                                         encoding="utf-8-sig")
    pd.concat(sample_preds).to_csv(REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    meta["train_seconds"] = round(time.time() - t0, 1)
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"XONG sau {meta['train_seconds']:.0f} giây. Bảng điểm: reports/metrics.csv")


if __name__ == "__main__":
    main()
