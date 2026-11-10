"""BƯỚC XỬ LÝ DỮ LIỆU LỚN (ETL) bằng PySpark.

Đọc toàn bộ file CSV lịch sử (hàng triệu dòng) -> làm sạch -> ghép thời tiết với
chất lượng không khí -> thống kê -> lưu bảng sạch cho bước train.

Kết quả trong data/processed/:
  merged.parquet             bảng sạch đã ghép, dùng để train
  summary_by_province.csv    thống kê theo tỉnh (số dòng, thiếu dữ liệu, trung bình)
  monthly_means.csv          trung bình theo tháng của từng tỉnh (để vẽ biểu đồ)
  etl_report.json            số liệu quá trình xử lý (để đưa vào báo cáo)

Cách chạy:
    python prepare_data.py                  # dùng PySpark
    python prepare_data.py --engine pandas  # dự phòng nếu PySpark lỗi trên máy
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

import config
from features import save_table
from om_client import log

OUT_DIR = config.DATA_DIR / "processed"

# Ngưỡng hợp lý để loại dữ liệu lỗi
VALID_RANGES = {
    "temperature_2m": (-10, 50),
    "relative_humidity_2m": (0, 100),
    "pressure_msl": (900, 1100),
    "pm2_5": (0, 1000),
    "pm10": (0, 2000),
}


# ---------------------------------------------------------------------------
# Chuẩn bị môi trường cho Spark trên Windows
# ---------------------------------------------------------------------------
def setup_spark_env():
    # Spark cần biết dùng đúng trình thông dịch Python đang chạy
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    if os.environ.get("JAVA_HOME") or shutil.which("java"):
        return
    # Java vừa cài nhưng terminal cũ chưa nhận PATH: tự tìm ở thư mục cài mặc định
    patterns = [
        r"C:\Program Files\Eclipse Adoptium\jdk-*",
        r"C:\Program Files\Java\jdk-*",
        r"C:\Program Files\Microsoft\jdk-*",
    ]
    for pat in patterns:
        found = sorted(glob.glob(pat), reverse=True)
        if found:
            java_home = found[0]
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ["PATH"]
            log(f"Tự tìm thấy Java tại: {java_home}")
            return
    log("CẢNH BÁO: không tìm thấy Java. PySpark cần Java 17 hoặc 21.")


def read_csv_dir(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Không có file CSV trong {d}. Hãy chạy fetch_history.py trước.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def pandas_to_spark(spark, pdf: pd.DataFrame, schema):
    """Đưa bảng vào Spark qua Apache Arrow.

    Trên Windows, Spark đọc file trực tiếp cần bộ thư viện Hadoop riêng (winutils.exe,
    hadoop.dll) nên dễ bị treo. Vì vậy ở đây đọc CSV bằng pandas rồi chuyển vào Spark
    trong bộ nhớ; mọi bước làm sạch, ghép, thống kê phía sau vẫn do Spark thực hiện."""
    import pyarrow as pa
    names = [f.name for f in schema.fields]
    pa_schema = pa.schema([(f.name, pa.string() if f.dataType.typeName() == "string"
                            else pa.float64()) for f in schema.fields])
    pdf = pdf[names].copy()
    for f in schema.fields:
        if f.dataType.typeName() == "string":
            pdf[f.name] = pdf[f.name].astype(object)
        else:
            pdf[f.name] = pd.to_numeric(pdf[f.name], errors="coerce").astype("float64")
    table = pa.Table.from_pandas(pdf, schema=pa_schema, preserve_index=False)
    try:
        return spark.createDataFrame(table)  # PySpark >= 4.0 nhận thẳng bảng Arrow
    except Exception:
        return spark.createDataFrame(pdf, schema=schema)


def spark_to_pandas(sdf) -> pd.DataFrame:
    try:
        return sdf.toArrow().to_pandas()  # PySpark >= 4.0, nhanh và ổn định
    except Exception:
        return sdf.toPandas()


# ---------------------------------------------------------------------------
# ETL bằng PySpark
# ---------------------------------------------------------------------------
def run_spark():
    setup_spark_env()
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    spark = (SparkSession.builder
             .master("local[*]")
             .appName("VN-Weather-AQI-ETL")
             .config("spark.driver.memory", "8g")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Đã khởi động Spark {spark.version} với {spark.sparkContext.defaultParallelism} luồng CPU")

    def schema(vars_):
        fields = [StructField("location", StringType()), StructField("province", StringType()),
                  StructField("lat", DoubleType()), StructField("lon", DoubleType()),
                  StructField("time", StringType())]
        fields += [StructField(v, DoubleType()) for v in vars_]
        return StructType(fields)

    log("Nạp CSV vào Spark...")
    weather = pandas_to_spark(spark, read_csv_dir(config.HISTORY_WEATHER_DIR),
                              schema(config.WEATHER_VARS)).repartition(16, "location")
    aq = pandas_to_spark(spark, read_csv_dir(config.HISTORY_AQ_DIR),
                         schema(config.AQ_VARS)).repartition(16, "location")

    stats = {"engine": "pyspark"}
    stats["weather_rows_raw"] = weather.count()
    stats["aq_rows_raw"] = aq.count()
    log(f"Đọc {stats['weather_rows_raw']:,} dòng thời tiết, {stats['aq_rows_raw']:,} dòng không khí")

    # Làm sạch: bỏ trùng (location, time), bỏ dòng thiếu thời gian, bỏ giá trị vô lý
    weather = weather.dropna(subset=["time"]).dropDuplicates(["location", "time"])
    aq = aq.dropna(subset=["time"]).dropDuplicates(["location", "time"])
    for col, (lo, hi) in VALID_RANGES.items():
        if col in config.WEATHER_VARS:
            weather = weather.withColumn(col, F.when(F.col(col).between(lo, hi), F.col(col)))
        if col in config.AQ_VARS:
            aq = aq.withColumn(col, F.when(F.col(col).between(lo, hi), F.col(col)))

    # Ghép: mỗi giờ của mỗi tỉnh có cả thời tiết và (nếu có) chất lượng không khí
    merged = weather.join(aq.select("location", "time", *config.AQ_VARS),
                          on=["location", "time"], how="left").cache()
    stats["merged_rows"] = merged.count()
    stats["duplicates_removed"] = stats["weather_rows_raw"] - stats["merged_rows"]

    # Thống kê theo tỉnh
    agg = [F.count("*").alias("rows"), F.min("time").alias("first_time"),
           F.max("time").alias("last_time"),
           F.round(F.avg("temperature_2m"), 2).alias("avg_temperature"),
           F.round(F.max("temperature_2m"), 2).alias("max_temperature"),
           F.round(F.avg("pm2_5"), 2).alias("avg_pm2_5"),
           F.round(F.max("pm2_5"), 2).alias("max_pm2_5"),
           F.round(F.sum("precipitation"), 1).alias("total_precipitation"),
           F.sum(F.col("temperature_2m").isNull().cast("int")).alias("missing_temperature"),
           F.sum(F.col("pm2_5").isNotNull().cast("int")).alias("rows_with_pm2_5")]
    summary = spark_to_pandas(merged.groupBy("location", "province").agg(*agg).orderBy("location"))

    monthly = spark_to_pandas(merged.withColumn("month", F.substring("time", 1, 7))
               .groupBy("location", "province", "month")
               .agg(F.round(F.avg("temperature_2m"), 2).alias("avg_temperature"),
                    F.round(F.avg("pm2_5"), 2).alias("avg_pm2_5"),
                    F.round(F.avg("us_aqi"), 1).alias("avg_us_aqi"),
                    F.round(F.sum("precipitation"), 1).alias("total_precipitation"))
               .orderBy("location", "month"))

    log("Chuyển dữ liệu từ Spark sang pandas để train...")
    df = spark_to_pandas(merged)
    spark.stop()
    return df, summary, monthly, stats


# ---------------------------------------------------------------------------
# ETL bằng pandas (dự phòng)
# ---------------------------------------------------------------------------
def run_pandas():
    def read_dir(d: Path):
        files = sorted(d.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"Không có file CSV trong {d}. Hãy chạy fetch_history.py trước.")
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    weather = read_dir(config.HISTORY_WEATHER_DIR)
    aq = read_dir(config.HISTORY_AQ_DIR)
    stats = {"engine": "pandas", "weather_rows_raw": len(weather), "aq_rows_raw": len(aq)}
    log(f"Đọc {len(weather):,} dòng thời tiết, {len(aq):,} dòng không khí")

    weather = weather.dropna(subset=["time"]).drop_duplicates(["location", "time"])
    aq = aq.dropna(subset=["time"]).drop_duplicates(["location", "time"])
    for col, (lo, hi) in VALID_RANGES.items():
        for frame in (weather, aq):
            if col in frame:
                frame[col] = frame[col].where(frame[col].between(lo, hi))

    df = weather.merge(aq[["location", "time", *config.AQ_VARS]], on=["location", "time"], how="left")
    stats["merged_rows"] = len(df)
    stats["duplicates_removed"] = stats["weather_rows_raw"] - len(df)

    g = df.groupby(["location", "province"])
    summary = pd.DataFrame({
        "rows": g.size(),
        "first_time": g["time"].min(), "last_time": g["time"].max(),
        "avg_temperature": g["temperature_2m"].mean().round(2),
        "max_temperature": g["temperature_2m"].max().round(2),
        "avg_pm2_5": g["pm2_5"].mean().round(2), "max_pm2_5": g["pm2_5"].max().round(2),
        "total_precipitation": g["precipitation"].sum().round(1),
        "missing_temperature": g["temperature_2m"].apply(lambda s: s.isna().sum()),
        "rows_with_pm2_5": g["pm2_5"].count(),
    }).reset_index()

    df["month"] = df["time"].str[:7]
    monthly = (df.groupby(["location", "province", "month"])
               .agg(avg_temperature=("temperature_2m", "mean"), avg_pm2_5=("pm2_5", "mean"),
                    avg_us_aqi=("us_aqi", "mean"), total_precipitation=("precipitation", "sum"))
               .round(2).reset_index())
    df = df.drop(columns="month")
    return df, summary, monthly, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["spark", "pandas"], default="spark")
    args = ap.parse_args()

    t0 = time.time()
    if args.engine == "spark":
        try:
            df, summary, monthly, stats = run_spark()
        except Exception as e:
            log(f"PySpark gặp lỗi: {type(e).__name__}: {str(e)[:300]}")
            log("Chuyển sang xử lý bằng pandas để không bị kẹt. "
                "Gửi lỗi trên cho trợ lý AI nếu muốn sửa để dùng Spark.")
            df, summary, monthly, stats = run_pandas()
    else:
        df, summary, monthly, stats = run_pandas()

    df = df.sort_values(["location", "time"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = save_table(df, OUT_DIR / "merged")
    summary.to_csv(OUT_DIR / "summary_by_province.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_means.csv", index=False, encoding="utf-8-sig")

    stats.update({
        "provinces": int(df["location"].nunique()),
        "first_time": str(df["time"].min()), "last_time": str(df["time"].max()),
        "rows_with_pm2_5": int(df["pm2_5"].notna().sum()),
        "seconds": round(time.time() - t0, 1),
    })
    with open(OUT_DIR / "etl_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log(f"XONG ({stats['engine']}, {stats['seconds']} giây). {len(df):,} dòng sạch -> {path.name}")
    log(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
