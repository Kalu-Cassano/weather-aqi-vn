"""Vẽ biểu đồ PNG cho báo cáo đồ án.

Đọc kết quả ETL (data/processed/) và kết quả train (reports/), lưu 6 hình vào
reports/figures/.

Cách chạy:
    python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

# Bảng màu phân loại (3 màu đầu phân biệt tốt cả với người mù màu)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"

CITIES = {"ha_noi": ("Hà Nội", BLUE), "da_nang": ("Đà Nẵng", ORANGE), "tp_hcm": ("TP.HCM", AQUA)}
MODELS = [("Baseline", GRAY), ("Ridge", ORANGE), ("XGBoost", BLUE)]
TARGETS = {"temperature_2m": ("Nhiệt độ", "°C"), "pm2_5": ("PM2.5", "µg/m³")}


def setup_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Segoe UI", "Arial") if f in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  Đã lưu {path.relative_to(ROOT)}")
    return path


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def model_label(name: str) -> str:
    n = name.lower()
    if "baseline" in n:
        return "Baseline"
    if "ridge" in n:
        return "Ridge"
    if "xgb" in n:
        return "XGBoost"
    return name


# ---------------------------------------------------------------------------
def fig_monthly(monthly, col, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for loc, (label, color) in CITIES.items():
        d = monthly[(monthly["location"] == loc)].dropna(subset=[col]).sort_values("month")
        if d.empty:
            continue
        x = pd.to_datetime(d["month"] + "-01")
        ax.plot(x, d[col], color=color, linewidth=2, marker="o", markersize=3.5, label=label)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Tháng")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.autofmt_xdate()
    return save(fig, name)


def fig_pm25_ranking(summary):
    d = summary.dropna(subset=["avg_pm2_5"]).sort_values("avg_pm2_5")
    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(d["province"], d["avg_pm2_5"], color=BLUE, height=0.7)
    for b, v in zip(bars, d["avg_pm2_5"]):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=8, color=INK2)
    ax.set_title(f"Xếp hạng PM2.5 trung bình của {len(d)} tỉnh/thành", loc="left")
    ax.set_xlabel("PM2.5 trung bình (µg/m³)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.margins(x=0.08)
    return save(fig, "03_xep_hang_pm25.png")


def fig_model_compare(metrics):
    m = metrics.copy()
    m["model_short"] = m["model"].map(model_label)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.26
    for ax, (target, (tname, unit)) in zip(axes, TARGETS.items()):
        d = m[m["target"] == target]
        horizons = sorted(d["horizon_h"].unique())
        for i, (mname, color) in enumerate(MODELS):
            vals = [d[(d["horizon_h"] == h) & (d["model_short"] == mname)]["MAE"].mean()
                    for h in horizons]
            xs = [j + (i - 1) * width for j in range(len(horizons))]
            bars = ax.bar(xs, vals, width=width - 0.03, color=color, label=mname)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"{h} giờ" for h in horizons])
        ax.set_xlabel("Dự báo trước")
        ax.set_ylabel(f"MAE ({unit})")
        ax.set_title(f"{tname}: sai số MAE (càng thấp càng tốt)", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "04_so_sanh_mo_hinh.png")


def fig_feature_importance(fi):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (target, (tname, _)) in zip(axes, TARGETS.items()):
        d = (fi[(fi["target"] == target) & (fi["horizon_h"] == 1)]
             .nlargest(15, "importance").iloc[::-1])
        ax.barh(d["feature"], d["importance"], color=BLUE, height=0.7)
        ax.set_title(f"{tname} (dự báo 1 giờ): top 15 đặc trưng", loc="left")
        ax.set_xlabel("Mức độ quan trọng (XGBoost)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return save(fig, "05_feature_importance.png")


def fig_forecast_vs_actual(pred, location="tp_hcm"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for ax, (target, (tname, unit)) in zip(axes, TARGETS.items()):
        d = pred[(pred["location"] == location) & (pred["target"] == target)
                 & (pred["horizon_h"] == 1)].copy()
        d["target_time"] = pd.to_datetime(d["target_time"])
        d = d.sort_values("target_time")
        d = d[d["target_time"] > d["target_time"].max() - pd.Timedelta(days=7)]
        ax.plot(d["target_time"], d["actual"], color=INK, linewidth=2, label="Thực tế")
        ax.plot(d["target_time"], d["model"], color=BLUE, linewidth=2, label="XGBoost")
        ax.plot(d["target_time"], d["baseline"], color=ORANGE, linewidth=1.5,
                linestyle="--", label="Baseline (giữ nguyên)")
        ax.set_title(f"{tname} tại TP.HCM: dự báo 1 giờ so với thực tế (7 ngày cuối)", loc="left")
        ax.set_ylabel(f"{tname} ({unit})")
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    axes[-1].set_xlabel("Thời gian")
    fig.autofmt_xdate()
    fig.tight_layout()
    return save(fig, "06_du_bao_vs_thuc_te.png")


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    monthly = read_csv(PROCESSED / "monthly_means.csv")
    summary = read_csv(PROCESSED / "summary_by_province.csv")
    metrics = read_csv(REPORTS / "metrics.csv")
    fi = read_csv(REPORTS / "feature_importance.csv")
    pred = read_csv(REPORTS / "test_predictions.csv")

    print("Vẽ biểu đồ...")
    fig_monthly(monthly, "avg_temperature", "Nhiệt độ trung bình (°C)",
                "Nhiệt độ trung bình theo tháng", "01_nhiet_do_theo_thang.png")
    fig_monthly(monthly, "avg_pm2_5", "PM2.5 trung bình (µg/m³)",
                "PM2.5 trung bình theo tháng", "02_pm25_theo_thang.png")
    fig_pm25_ranking(summary)
    fig_model_compare(metrics)
    fig_feature_importance(fi)
    fig_forecast_vs_actual(pred)
    print("Xong.")


if __name__ == "__main__":
    main()
