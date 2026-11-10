"""Streamlit interface. Start with: streamlit run app.py."""

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import assistant
from config import LOCATIONS


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "realtime"
REPORT_DIR = ROOT / "reports"
TARGETS = {
    "temperature_2m": ("Temperature", "°C"),
    "pm2_5": ("PM2.5", "µg/m³"),
}
HORIZONS = (1, 3, 6)
PROVINCES = pd.DataFrame(
    LOCATIONS, columns=["location", "province", "representative", "latitude", "longitude"]
)
# Ngưỡng và màu US AQI của EPA/AirNow:
# https://www.airnow.gov/aqi/aqi-basics/
AQI_LEVELS = (
    (50, "Good", "#00e400", "0–50"),
    (100, "Moderate", "#ffff00", "51–100"),
    (150, "Unhealthy for sensitive groups", "#ff7e00", "101–150"),
    (200, "Unhealthy", "#ff0000", "151–200"),
    (300, "Very unhealthy", "#8f3f97", "201–300"),
    (float("inf"), "Hazardous", "#7e0023", "301+"),
)
NO_AQI = ("No data", "#9ca3af")
NUMERIC_COLUMNS = (
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pm2_5", "us_aqi",
    "weather_code", "precipitation",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1480px; padding-top: 2rem; padding-bottom: 2rem;}
        h1 {letter-spacing: -.035em;}
        h2, h3 {letter-spacing: -.02em;}
        [data-testid="stSidebar"] {border-right: 1px solid #e5eaf2;}
        [data-testid="stMetric"], .kpi-card {
            background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
            padding: 18px 20px; box-shadow: 0 4px 18px rgba(15, 23, 42, .045);
        }
        [data-testid="stMetricLabel"], .kpi-label {color: #64748b; font-size: .87rem;}
        [data-testid="stMetricValue"] {font-size: clamp(1.3rem, 2.1vw, 2rem); font-weight: 650;}
        .kpi-card {min-height: 152px; border-top: 3px solid #1f6feb;}
        .kpi-value {font-size: clamp(1.4rem, 2.4vw, 2.1rem); font-weight: 700;
                    letter-spacing: -.04em; color: #0f172a; margin: 10px 0 4px;}
        .kpi-detail {color: #64748b; font-size: .85rem;}
        .weather-banner {display: flex; align-items: center; gap: 16px; padding: 18px 22px;
            margin-bottom: 20px; border: 1px solid #dbeafe; border-radius: 16px;
            background: linear-gradient(120deg, #eff6ff, #fff);}
        .weather-icon {font-size: 2.8rem; line-height: 1;}
        .weather-title {font-size: 1.15rem; font-weight: 650; color: #0f172a;}
        .weather-subtitle {font-size: .85rem; color: #64748b; margin-top: 4px;}
        .hourly-strip {display: grid; grid-template-columns: repeat(6, minmax(146px, 1fr));
            gap: 12px; overflow-x: auto; padding: 4px 2px 16px; margin-bottom: 12px;}
        .hour-card {padding: 18px 14px; border-radius: 14px; border: 1px solid #e2e8f0;
            background: #fff; box-shadow: 0 3px 12px rgba(15, 23, 42, .04);}
        .hour-time {font-weight: 650; color: #334155;}
        .hour-date {font-size: .75rem; color: #64748b; margin: 2px 0 12px;}
        .hour-temp {font-size: 1.65rem; font-weight: 700; color: #0f172a;}
        .hour-rain {color: #64748b; font-size: .8rem; margin-top: 8px;}
        .hour-model {font-size: .8rem; font-weight: 600; color: #1f6feb;
            border-top: 1px solid #edf2f7; margin-top: 12px; padding-top: 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_csv(
    path: str,
    required_columns: tuple[str, ...],
    date_columns: tuple[str, ...] = ("time", "fetched_at"),
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS,
    valid_columns: tuple[str, ...] = ("location", "time"),
) -> tuple[pd.DataFrame, str]:
    """Read CSV data and return a clear message when data is not ready."""
    filename = Path(path).name
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(), f"File not found: {filename}. Data will appear after collection or report creation."
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), f"File {filename} is empty. Please wait for the next update."
    except (OSError, UnicodeError, pd.errors.ParserError):
        return pd.DataFrame(), f"Could not read {filename}. Please check the file or try again later."

    missing = set(required_columns) - set(data.columns)
    if missing:
        return pd.DataFrame(), f"File {filename} is missing columns: {', '.join(sorted(missing))}."
    if data.empty:
        return data, f"File {filename} has no rows yet. Please wait for the next update."

    for column in date_columns:
        if column in data.columns:
            parsed = pd.to_datetime(data[column], errors="coerce", format="mixed")
            if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                parsed = parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
            data[column] = parsed
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce").replace(
                [float("inf"), -float("inf")], float("nan")
            )
    data = data.dropna(subset=list(valid_columns))
    sort_columns = [c for c in ("time", "target_time", "made_at", "fetched_at") if c in data.columns]
    if sort_columns:
        data = data.sort_values(sort_columns, na_position="first", kind="stable")
    if data.empty:
        return data, f"File {filename} has no valid rows to show."
    return data, ""


def read_table(path: Path, required_columns: tuple[str, ...], **options) -> pd.DataFrame:
    data, message = load_csv(str(path), required_columns, **options)
    if message:
        st.info(message)
    return data


@st.cache_data(ttl=300)
def load_etl_report(path: str) -> tuple[dict, str]:
    try:
        with Path(path).open(encoding="utf-8-sig") as source:
            report = json.load(source)
    except FileNotFoundError:
        return {}, "No ETL report yet. Results will appear after data processing."
    except (OSError, UnicodeError, ValueError):
        return {}, "Could not read etl_report.json. Please check the ETL report."
    if not isinstance(report, dict):
        return {}, "The ETL report has an invalid format."
    return report, ""


@st.cache_data(ttl=300)
def load_report_images(directory: str) -> tuple[list[tuple[str, bytes]], str]:
    folder = Path(directory)
    try:
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    except FileNotFoundError:
        return [], "No report images in reports/figures/ yet."
    except OSError:
        return [], "Could not read the report image folder."
    images, unreadable = [], []
    for path in paths:
        try:
            images.append((path.name, path.read_bytes()))
        except OSError:
            unreadable.append(path.name)
    message = f"Could not read images: {', '.join(unreadable)}." if unreadable else ""
    return images, message


def aqi_level(value: float) -> tuple[str, str]:
    if pd.isna(value) or value < 0:
        return NO_AQI
    for upper, label, color, _ in AQI_LEVELS:
        if value <= upper:
            return label, color
    return NO_AQI


def latest_by_location(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest observation and the latest fetch for the same time."""
    if data.empty:
        return data
    return data.sort_values(
        ["time", "fetched_at"], na_position="first", kind="stable"
    ).drop_duplicates("location", keep="last")


def selected_history(data: pd.DataFrame, location: str) -> pd.DataFrame:
    if data.empty:
        return data
    # Một hours quan trắc có thể được thu thập nhiều lần: giữ lần thu mới nhất.
    return data.loc[data["location"].eq(location)].drop_duplicates("time", keep="last")


def metric_value(row: pd.Series, column: str, unit: str, decimals: int = 1) -> str:
    value = row.get(column)
    return "—" if pd.isna(value) else f"{value:.{decimals}f}{unit}"


def national_snapshot(weather: pd.DataFrame, air: pd.DataFrame) -> pd.DataFrame:
    """Keep all configured provinces, including those without data."""
    snapshot = PROVINCES.copy()
    for data, fields, time_name in (
        (weather, ("temperature_2m",), "weather_time"),
        (air, ("us_aqi", "pm2_5"), "aq_time"),
    ):
        latest = latest_by_location(data)
        if latest.empty:
            for field in fields:
                snapshot[field] = float("nan")
            snapshot[time_name] = pd.NaT
        else:
            snapshot = snapshot.merge(
                latest[["location", "time", *fields]].rename(columns={"time": time_name}),
                on="location", how="left", validate="one_to_one",
            )
    snapshot.loc[snapshot["us_aqi"].lt(0), "us_aqi"] = float("nan")
    snapshot["aqi_label"] = snapshot["us_aqi"].map(lambda value: aqi_level(value)[0])
    return snapshot


def kpi_card(column, label: str, value: str, detail: str) -> None:
    column.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-detail">{escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def national_map(snapshot: pd.DataFrame, metric: str):
    data = snapshot.copy()
    data["Temperature"] = data["temperature_2m"].map(
        lambda value: "No data" if pd.isna(value) else f"{value:.1f} °C"
    )
    data["US AQI"] = data["us_aqi"].map(
        lambda value: "No data" if pd.isna(value) else f"{value:.0f}"
    )
    for field, label in (("weather_time", "Weather at"), ("aq_time", "Air quality at")):
        data[label] = data[field].dt.strftime("%H:%M · %d/%m/%Y").fillna("No data")
    field = "temperature_2m" if metric == "Temperature" else "us_aqi"
    available, missing = data.loc[data[field].notna()], data.loc[data[field].isna()]
    options = dict(
        lat="latitude", lon="longitude", hover_name="province",
        hover_data={
            "latitude": False, "longitude": False, "temperature_2m": False, "us_aqi": False,
            "Temperature": True, "US AQI": True, "Weather at": True, "Air quality at": True,
        },
        map_style="open-street-map", center={"lat": 16.0, "lon": 106.0}, zoom=4.4, height=590,
    )
    if available.empty:
        figure = px.scatter_map(data, **options)
        figure.update_traces(marker={"color": NO_AQI[1], "size": 20, "opacity": 0.9})
    else:
        if field == "temperature_2m":
            color_options = {"color_continuous_scale": ["#2563eb", "#7dd3fc", "#fbbf24", "#f97316", "#dc2626"]}
        else:
            # Thang màu bậc thang theo EPA, không nội suy giữa các mức AQI.
            color_options = {
                "range_color": (0, 500),
                "color_continuous_scale": [
                    (0, "#00e400"), (.1, "#00e400"), (.1, "#ffff00"), (.2, "#ffff00"),
                    (.2, "#ff7e00"), (.3, "#ff7e00"), (.3, "#ff0000"), (.4, "#ff0000"),
                    (.4, "#8f3f97"), (.6, "#8f3f97"), (.6, "#7e0023"), (1, "#7e0023"),
                ],
            }
        figure = px.scatter_map(available, color=field, **color_options, **options)
        figure.update_traces(marker={"size": 20, "opacity": 0.9})
        figure.update_coloraxes(colorbar_title="Temperature (°C)" if field == "temperature_2m" else "US AQI")
        if not missing.empty:
            gray = px.scatter_map(missing, **options)
            gray.update_traces(
                marker={"color": NO_AQI[1], "size": 20, "opacity": 0.9},
                name="No data", showlegend=True,
            )
            figure.add_traces(gray.data)
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"orientation": "h", "y": -.05})
    return figure


def show_overview(weather: pd.DataFrame, air: pd.DataFrame) -> None:
    snapshot = national_snapshot(weather, air)
    columns = st.columns(4)
    for column, field, highest, label, unit in (
        (columns[0], "temperature_2m", True, "🔥 Highest temperature", " °C"),
        (columns[1], "temperature_2m", False, "❄️ Lowest temperature", " °C"),
        (columns[2], "us_aqi", True, "🌫️ Highest US AQI", ""),
    ):
        valid = snapshot.dropna(subset=[field])
        if valid.empty:
            kpi_card(column, label, "—", "No data")
            continue
        row = valid.loc[valid[field].idxmax() if highest else valid[field].idxmin()]
        value = metric_value(row, field, unit, 1 if field == "temperature_2m" else 0)
        kpi_card(column, label, value, row["province"])
    aqi_count = int(snapshot["us_aqi"].notna().sum())
    kpi_card(
        columns[3], "⚠️ Provinces with US AQI > 100",
        str(int(snapshot["us_aqi"].gt(100).sum())) if aqi_count else "—",
        f"Across {aqi_count}/{len(snapshot)} provinces with AQI data" if aqi_count else "No AQI data",
    )
    st.caption(
        f"Latest data for each province · Temperature: {snapshot['temperature_2m'].notna().sum()}/{len(snapshot)} provinces"
        f" · US AQI: {aqi_count}/{len(snapshot)} provinces"
    )
    st.subheader("🗺️ National map")
    metric = st.segmented_control("Color by", ["Temperature", "US AQI"], default="US AQI", key="national_map_metric")
    st.plotly_chart(national_map(snapshot, metric or "US AQI"), use_container_width=True)
    st.caption("Each point represents one province. Gray means no data for the selected value.")

    st.subheader("🏆 Ranking of 34 provinces")
    ranking = snapshot.sort_values(["us_aqi", "temperature_2m", "province"], ascending=[False, False, True], na_position="last")
    ranking = ranking[["province", "temperature_2m", "us_aqi", "aqi_label", "pm2_5", "weather_time", "aq_time"]].reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    maximum = max(500.0, float(snapshot["us_aqi"].max())) if aqi_count else 500.0
    st.caption("Highest AQI first. Provinces without AQI data are last. Click a column header to change the order.")
    st.dataframe(
        ranking, hide_index=True, use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("No.", width="small", format="%d"),
            "province": st.column_config.TextColumn("Province", width="medium"),
            "temperature_2m": st.column_config.NumberColumn("Temperature", format="%.1f °C"),
            "us_aqi": st.column_config.ProgressColumn("US AQI", min_value=0, max_value=maximum, format="%.0f"),
            "aqi_label": st.column_config.TextColumn("Air quality level", width="medium"),
            "pm2_5": st.column_config.NumberColumn("PM2.5", format="%.1f µg/m³"),
            "weather_time": st.column_config.DatetimeColumn("Weather time", format="DD/MM HH:mm"),
            "aq_time": st.column_config.DatetimeColumn("Air quality time", format="DD/MM HH:mm"),
        },
    )


def weather_description(value) -> tuple[str, str]:
    try:
        code = float(value)
    except (TypeError, ValueError):
        return "🌡️", "No weather details"
    descriptions = {
        0: ("☀️", "Clear sky"), 1: ("🌤️", "Mostly clear"),
        2: ("⛅", "Partly cloudy"), 3: ("☁️", "Cloudy"),
        45: ("🌫️", "Fog"), 48: ("🌫️", "Freezing fog"),
    }
    if code in descriptions:
        return descriptions[code]
    for lower, upper, icon, label in (
        (51, 57, "🌦️", "Drizzle"), (61, 67, "🌧️", "Rain"),
        (71, 77, "🌨️", "Snow"), (80, 82, "🌧️", "Rain showers"),
        (85, 86, "🌨️", "Snow showers"), (95, 99, "⛈️", "Thunderstorm"),
    ):
        if code.is_integer() and lower <= code <= upper:
            return icon, label
    return "🌡️", "No weather details"


def next_six_hours(
    forecasts: pd.DataFrame, predictions: pd.DataFrame, location: str, now: pd.Timestamp,
) -> pd.DataFrame:
    hours = pd.DataFrame({"target_time": pd.date_range(now.floor("h") + pd.Timedelta(hours=1), periods=6, freq="h")})
    if not forecasts.empty:
        rows = forecasts.loc[forecasts["location"].eq(location)]
        if not rows.empty:
            rows = rows.loc[rows["fetched_at"].eq(rows["fetched_at"].max())].drop_duplicates("target_time", keep="last")
            hours = hours.merge(rows[["target_time", "temperature_2m", "precipitation"]], on="target_time", how="left", validate="one_to_one")
    for field in ("temperature_2m", "precipitation"):
        if field not in hours:
            hours[field] = float("nan")
    if not predictions.empty:
        rows = predictions.loc[
            predictions["location"].eq(location) & predictions["target"].eq("temperature_2m")
            & predictions["made_at"].le(now)
        ].sort_values("made_at", kind="stable").drop_duplicates("target_time", keep="last")
        hours = hours.merge(
            rows[["target_time", "prediction"]].rename(columns={"prediction": "group_prediction"}),
            on="target_time", how="left", validate="one_to_one",
        )
    if "group_prediction" not in hours:
        hours["group_prediction"] = float("nan")
    return hours


def show_hourly_forecast(location: str) -> None:
    st.subheader("🕒 Forecast for the next 6 hours")
    forecasts = read_table(
        DATA_DIR / "om_forecast.csv", ("fetched_at", "location", "target_time", "temperature_2m", "precipitation"),
        date_columns=("fetched_at", "target_time"), numeric_columns=("temperature_2m", "precipitation"),
        valid_columns=("location", "target_time", "fetched_at"),
    )
    predictions = read_table(
        DATA_DIR / "predictions.csv", ("made_at", "location", "target", "target_time", "prediction"),
        date_columns=("made_at", "target_time"), numeric_columns=("prediction",),
        valid_columns=("location", "target", "made_at", "target_time"),
    )
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    hours = next_six_hours(forecasts, predictions, location, now)
    if hours["temperature_2m"].isna().all():
        st.info("No Open-Meteo forecast for the next 6 hours. Please wait for new data.")
    st.caption("Vietnam time (UTC+7) · Temperature and rain from the latest Open-Meteo update.")
    cards = []
    for _, row in hours.iterrows():
        temperature = metric_value(row, "temperature_2m", "°C")
        precipitation = metric_value(row, "precipitation", " mm")
        model_line = ""
        if pd.notna(row["group_prediction"]):
            model_line = f'<div class="hour-model">Group model: {row["group_prediction"]:.1f}°C</div>'
        cards.append(
            f'<div class="hour-card"><div class="hour-time">{row["target_time"]:%H:%M}</div>'
            f'<div class="hour-date">{row["target_time"]:%d/%m}</div>'
            f'<div class="hour-temp">{temperature}</div><div class="hour-rain">💧 Rain: {precipitation}</div>'
            f'{model_line}</div>'
        )
    st.markdown('<div class="hourly-strip">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def show_current(weather: pd.DataFrame, air: pd.DataFrame, location: str) -> None:
    weather_rows = selected_history(weather, location)
    air_rows = selected_history(air, location)
    current_weather = weather_rows.iloc[-1] if not weather_rows.empty else pd.Series(dtype=object)
    current_air = air_rows.iloc[-1] if not air_rows.empty else pd.Series(dtype=object)

    icon, description = weather_description(current_weather.get("weather_code"))
    st.markdown(
        f'<div class="weather-banner"><div class="weather-icon">{icon}</div>'
        f'<div><div class="weather-title">{description}</div>'
        '<div class="weather-subtitle">Weather at the selected province's sample location</div></div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Temperature", metric_value(current_weather, "temperature_2m", " °C"))
    columns[1].metric("Humidity", metric_value(current_weather, "relative_humidity_2m", " %", 0))
    columns[2].metric("Wind speed", metric_value(current_weather, "wind_speed_10m", " km/h"))
    aqi = current_air.get("us_aqi", float("nan"))
    label, color = aqi_level(aqi)
    with columns[3]:
        st.metric("US AQI", "—" if label == NO_AQI[0] else f"{aqi:.0f}")
        text_color = "#111827" if color in ("#00e400", "#ffff00", "#ff7e00", "#9ca3af") else "#ffffff"
        st.markdown(
            f'<span style="background:{color};color:{text_color};padding:5px 10px;'
            f'border-radius:6px;display:inline-block">{label}</span>',
            unsafe_allow_html=True,
        )

    if weather_rows.empty:
        st.info("No weather data for the selected province yet.")
    else:
        st.caption(f"Weather at {current_weather['time']:%H:%M · %d/%m/%Y}")
    if air_rows.empty:
        st.info("No air quality data for the selected province yet.")
    else:
        st.caption(f"Air quality at {current_air['time']:%H:%M · %d/%m/%Y}")

    show_hourly_forecast(location)
    st.subheader("🌏 Air quality across 34 provinces")
    latest_air = latest_by_location(air)
    map_data = PROVINCES.copy()
    if latest_air.empty:
        map_data["us_aqi"] = float("nan")
        map_data["time"] = pd.NaT
    else:
        map_data = map_data.merge(
            latest_air[["location", "us_aqi", "time"]], on="location", how="left"
        )
    map_data["Level"] = map_data["us_aqi"].map(lambda value: aqi_level(value)[0])
    map_data["US AQI"] = map_data["us_aqi"].map(
        lambda value: "No data" if aqi_level(value) == NO_AQI else f"{value:.0f}"
    )
    map_data["Time"] = map_data["time"].dt.strftime("%H:%M · %d/%m/%Y").fillna("—")
    colors = {label: color for _, label, color, _ in AQI_LEVELS}
    colors[NO_AQI[0]] = NO_AQI[1]
    figure = px.scatter_map(
        map_data,
        lat="latitude",
        lon="longitude",
        color="Level",
        color_discrete_map=colors,
        category_orders={"Level": list(colors)},
        hover_name="province",
        hover_data={
            "latitude": False, "longitude": False, "representative": True,
            "US AQI": True, "Time": True,
        },
        labels={"representative": "Sample location"},
        map_style="open-street-map",
        center={"lat": 16.0, "lon": 106.0},
        zoom=4.4,
        height=620,
    )
    figure.update_traces(marker={"size": 13, "opacity": 0.9})
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False)
    st.plotly_chart(figure, use_container_width=True)
    legend = []
    for _, level, level_color, interval in AQI_LEVELS:
        legend.append(
            f'<span style="display:inline-block;margin:3px 14px 3px 0">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{level_color};border:1px solid #777;border-radius:50%"></span> '
            f'{interval}: {level}</span>'
        )
    st.markdown("".join(legend), unsafe_allow_html=True)
    st.caption(
        "Gray means no data. Each point represents one province. "
        "[US AQI levels from EPA/AirNow](https://www.airnow.gov/aqi/aqi-basics/)."
    )


def show_history(weather: pd.DataFrame, air: pd.DataFrame, location: str) -> None:
    for data, column, title, unit, color in (
        (weather, "temperature_2m", "Temperature", "°C", "#ea580c"),
        (air, "pm2_5", "PM2.5", "µg/m³", "#2563eb"),
    ):
        st.subheader(title)
        history = selected_history(data, location)
        if not history.empty:
            history = history.dropna(subset=[column])
        if history.empty:
            st.info(f"No data {title.lower()} for the selected province.")
            continue
        figure = px.line(
            history, x="time", y=column, markers=True,
            labels={"time": "Time", column: f"{title} ({unit})"},
        )
        figure.update_traces(line_color=color, connectgaps=False)
        figure.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=320)
        st.plotly_chart(figure, use_container_width=True)


def latest_prediction_batch(predictions: pd.DataFrame, location: str) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    rows = predictions.loc[predictions["location"].eq(location)]
    if rows.empty:
        return rows
    return rows.loc[rows["made_at"].eq(rows["made_at"].max())].drop_duplicates(
        ["target", "horizon_h"], keep="last"
    )


def latest_om_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts
    return forecasts.sort_values("fetched_at", kind="stable").drop_duplicates(
        ["location", "target_time"], keep="last"
    )


def verified_realtime_predictions(
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    air: pd.DataFrame,
    om_forecasts: pd.DataFrame,
    now: pd.Timestamp,
    location: str | None = None,
) -> pd.DataFrame:
    """Keep one checked forecast per province, target, horizon and time."""
    columns = ["location", "target", "horizon_h", "target_time", "prediction", "actual", "om_prediction"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    past = predictions.loc[
        predictions["target_time"].lt(now) & predictions["horizon_h"].isin(HORIZONS)
        & predictions["target"].isin(TARGETS)
    ].dropna(subset=["location", "target_time", "prediction"]).copy()
    if location is not None:
        past = past.loc[past["location"].eq(location)]
    # Các cột thời điểm tạo/giờ gốc là tùy chọn trong file dự báo.
    for field in ("made_at", "base_time"):
        if field in past.columns:
            past = past.loc[past[field].lt(past["target_time"])]
    revision_columns = [field for field in ("made_at", "base_time") if field in past.columns]
    if revision_columns:
        past = past.sort_values(revision_columns, kind="stable")
    past = past.drop_duplicates(
        ["location", "target", "horizon_h", "target_time"], keep="last"
    )
    parts = []
    for target, observations in (("temperature_2m", weather), ("pm2_5", air)):
        target_predictions = past.loc[past["target"].eq(target)]
        if target_predictions.empty or observations.empty:
            continue
        actual = observations.dropna(subset=["location", "time", target]).copy()
        if location is not None:
            actual = actual.loc[actual["location"].eq(location)]
        # Quan trắc có thể ở phút 15/30/45: giữ bản cuối trong cùng hours.
        actual["target_time"] = actual["time"].dt.floor("h")
        actual_sort = [field for field in ("time", "fetched_at") if field in actual.columns]
        actual = actual.sort_values(actual_sort, kind="stable").drop_duplicates(
            ["location", "target_time"], keep="last"
        )
        matched = target_predictions[["location", "target", "horizon_h", "target_time", "prediction"]].merge(
            actual[["location", "target_time", target]].rename(columns={target: "actual"}),
            on=["location", "target_time"], how="inner", validate="many_to_one",
        )
        parts.append(matched)
    if not parts:
        return pd.DataFrame(columns=columns)
    verified = pd.concat(parts, ignore_index=True)
    if not om_forecasts.empty:
        # Lần thu gần nhất TRƯỚC target_time, không dùng bản thu bằng/sau hours đích.
        eligible = om_forecasts.loc[
            om_forecasts["fetched_at"].lt(om_forecasts["target_time"])
        ].dropna(subset=["location", "target_time", "temperature_2m"])
        eligible = latest_om_forecasts(eligible)
        verified = verified.merge(
            eligible[["location", "target_time", "temperature_2m"]].rename(
                columns={"temperature_2m": "om_prediction"}
            ),
            on=["location", "target_time"], how="left", validate="many_to_one",
        )
        verified.loc[verified["target"].ne("temperature_2m"), "om_prediction"] = float("nan")
    else:
        verified["om_prediction"] = float("nan")
    return verified[columns].sort_values(["target_time", "location", "target", "horizon_h"])


def realtime_accuracy_metrics(verified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for horizon in HORIZONS:
            samples = verified.loc[verified["target"].eq(target) & verified["horizon_h"].eq(horizon)]
            paired = samples.dropna(subset=["om_prediction"])
            rows.append({
                "target": target, "horizon_h": horizon,
                "group_mae": (samples["prediction"] - samples["actual"]).abs().mean(),
                "group_count": len(samples),
                "om_mae": (paired["om_prediction"] - paired["actual"]).abs().mean(),
                "om_count": len(paired),
                "group_mae_on_om": (paired["prediction"] - paired["actual"]).abs().mean(),
            })
    return pd.DataFrame(rows)


def realtime_score_display(scores: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Format table values while keeping the original numeric scores."""
    display = scores.copy()
    for field in ("group_mae", "group_mae_on_om", "om_mae"):
        if field in display.columns:
            display[field] = display[field].map(
                lambda value: "Not enough data" if pd.isna(value) else f"{value:.3f} {unit}"
            )
    return display


def hourly_realtime_mae(verified: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Find hourly MAE across provinces; keep missing hours empty."""
    columns = ["hour", "source", "mae", "samples"]
    if verified.empty:
        return pd.DataFrame(columns=columns)
    rows = verified.loc[
        verified["target"].eq(target) & verified["horizon_h"].eq(horizon)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["hour"] = rows["target_time"].dt.floor("h")
    hours = pd.date_range(rows["hour"].min(), rows["hour"].max(), freq="h")
    sources = [("prediction", "Group model")]
    if target == "temperature_2m":
        sources.append(("om_prediction", "Open-Meteo"))
    parts = []
    for field, label in sources:
        samples = rows.dropna(subset=[field, "actual"]).copy()
        samples["absolute_error"] = (samples[field] - samples["actual"]).abs()
        hourly = samples.groupby("hour").agg(
            mae=("absolute_error", "mean"), samples=("absolute_error", "count")
        ).reindex(hours)
        hourly["samples"] = hourly["samples"].fillna(0).astype(int)
        hourly["source"] = label
        parts.append(hourly.rename_axis("hour").reset_index())
    return pd.concat(parts, ignore_index=True)[columns]


def hourly_mae_figure(hourly: pd.DataFrame, title: str, unit: str):
    # Giữ khoảng trống giữa các hours, bỏ nguồn hoàn toàn chưa có mẫu khỏi chú giải.
    plotted = hourly.groupby("source", group_keys=False).filter(lambda rows: rows["mae"].notna().any())
    figure = px.line(
        plotted, x="hour", y="mae", color="source", markers=True,
        hover_data={"samples": True},
        labels={"hour": "Forecast target time (UTC+7)", "mae": f"MAE {title.lower()} ({unit})", "source": "Source", "samples": "Samples"},
        color_discrete_map={"Group model": "#2563eb", "Open-Meteo": "#ea580c"},
    )
    figure.update_traces(connectgaps=False)
    figure.update_layout(height=340, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title_text="", yaxis_rangemode="tozero")
    return figure


def past_forecast_comparison(
    predictions: pd.DataFrame, observations: pd.DataFrame, om_forecasts: pd.DataFrame,
    location: str, target: str, horizon: int, now: pd.Timestamp,
) -> pd.DataFrame:
    verified = verified_realtime_predictions(
        predictions, observations if target == "temperature_2m" else pd.DataFrame(),
        observations if target == "pm2_5" else pd.DataFrame(), om_forecasts, now, location,
    )
    comparison = verified.loc[verified["target"].eq(target) & verified["horizon_h"].eq(horizon)]
    comparison = comparison.rename(columns={"prediction": "Group model", "actual": "Actual", "om_prediction": "Open-Meteo"})
    if target != "temperature_2m":
        comparison = comparison.drop(columns="Open-Meteo")
    return comparison.sort_values("target_time")


def comparison_chart(data: pd.DataFrame, series: list[str], title: str, unit: str) -> None:
    figure = px.line(
        data, x="target_time", y=series, markers=True,
        labels={"target_time": "Time", "value": f"{title} ({unit})", "variable": "Source"},
        color_discrete_map={
            "Actual": "#111827", "Group model": "#2563eb", "XGBoost": "#2563eb",
            "Open-Meteo": "#ea580c", "Baseline": "#9ca3af",
        },
    )
    figure.update_traces(connectgaps=False)
    figure.update_layout(height=340, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)


def show_forecast(location: str) -> None:
    predictions = read_table(
        DATA_DIR / "predictions.csv",
        ("made_at", "location", "province", "base_time", "target", "horizon_h", "target_time", "prediction", "value_at_base"),
        date_columns=("made_at", "base_time", "target_time"),
        numeric_columns=("horizon_h", "prediction", "value_at_base"),
        valid_columns=("location", "target", "horizon_h", "made_at", "base_time", "target_time"),
    )
    om_forecasts = read_table(
        DATA_DIR / "om_forecast.csv",
        ("fetched_at", "location", "target_time", "temperature_2m"),
        date_columns=("fetched_at", "target_time"),
        numeric_columns=("temperature_2m",),
        valid_columns=("location", "target_time", "fetched_at"),
    )
    batch = latest_prediction_batch(predictions, location)
    st.subheader("Latest forecast · Group model vs Open-Meteo")
    if batch.empty:
        st.info("No group model forecast for the selected province yet.")
    else:
        st.caption(f"Latest forecast made at: {batch['made_at'].max():%H:%M · %d/%m/%Y}")
        om_latest = latest_om_forecasts(om_forecasts)
        for target, (title, unit) in TARGETS.items():
            st.markdown(f"**{title} ({unit})**")
            for column, horizon in zip(st.columns(3), HORIZONS):
                with column:
                    st.markdown(f"**{horizon} hours ahead**")
                    rows = batch.loc[batch["target"].eq(target) & batch["horizon_h"].eq(horizon)]
                    if rows.empty:
                        st.info("The latest forecast does not include this time yet.")
                        continue
                    row = rows.iloc[-1]
                    st.caption(f"Forecast time: {row['target_time']:%H:%M · %d/%m/%Y}")
                    if target == "temperature_2m":
                        match = pd.DataFrame() if om_latest.empty else om_latest.loc[
                            om_latest["location"].eq(location)
                            & om_latest["target_time"].eq(row["target_time"])
                        ]
                        om_row = match.iloc[-1] if not match.empty else pd.Series(dtype=object)
                        model_column, om_column = st.columns(2)
                        model_column.metric("Group model", metric_value(row, "prediction", f" {unit}"))
                        om_column.metric("Open-Meteo", metric_value(om_row, "temperature_2m", " °C"))
                        if not match.empty:
                            st.caption(f"Fetched at {om_row['fetched_at']:%H:%M · %d/%m/%Y}")
                        else:
                            st.caption("No Open-Meteo forecast for the same time yet.")
                    else:
                        st.metric("Group model", metric_value(row, "prediction", f" {unit}"))
        now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
        if batch["target_time"].max() < now:
            st.caption("These forecast times have passed. Please wait for the next forecast.")

    st.subheader("Past forecasts vs actual values")
    horizon = st.selectbox("Forecast horizon to compare", HORIZONS, format_func=lambda h: f"{h} hours", key="forecast_horizon")
    st.caption("Match the province and hour. Use the last observation in the hour and only forecasts made before the target time.")
    weather = read_table(DATA_DIR / "weather_current.csv", ("location", "time", "fetched_at", "temperature_2m"))
    air = read_table(DATA_DIR / "aq_current.csv", ("location", "time", "fetched_at", "pm2_5"))
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    for target, (title, unit) in TARGETS.items():
        st.markdown(f"**{title} · forecast horizon: {horizon} hours**")
        comparison = past_forecast_comparison(
            predictions, weather if target == "temperature_2m" else air,
            om_forecasts, location, target, horizon, now,
        )
        if len(comparison) < 2:
            st.info("We need a few more hours of data to compare.")
            continue
        series = ["Actual", "Group model"]
        if target == "temperature_2m":
            if "Open-Meteo" in comparison and comparison["Open-Meteo"].notna().any():
                series.append("Open-Meteo")
            else:
                st.caption("No valid Open-Meteo forecasts for the same hours yet.")
        comparison_chart(comparison, series, title, unit)


def last_test_week(data: pd.DataFrame, location: str, target: str, horizon: int) -> pd.DataFrame:
    if data.empty:
        return data
    rows = data.loc[
        data["location"].eq(location) & data["target"].eq(target) & data["horizon_h"].eq(horizon)
    ].dropna(subset=["actual"])
    if rows.empty:
        return rows
    end = rows["target_time"].max()
    return rows.loc[rows["target_time"].gt(end - pd.Timedelta(days=7))].sort_values(
        ["target_time", "time"], kind="stable"
    ).drop_duplicates("target_time", keep="last")


def show_etl_report() -> None:
    st.subheader("Big Data processing with PySpark")
    report, message = load_etl_report(str(ROOT / "data" / "processed" / "etl_report.json"))
    if message:
        st.info(message)
    if not report:
        return
    for fields in (
        (("engine", "Engine"), ("weather_rows_raw", "Raw weather rows"), ("aq_rows_raw", "Raw air quality rows"), ("merged_rows", "Merged rows")),
        (("provinces", "Provinces"), ("first_time", "Start time"), ("last_time", "End time"), ("seconds", "Processing time (seconds)")),
    ):
        for column, (key, label) in zip(st.columns(4), fields):
            value = report.get(key)
            if value is None or isinstance(value, (dict, list)):
                display = "—"
            elif key in ("first_time", "last_time"):
                timestamp = pd.to_datetime(str(value), errors="coerce")
                display = "—" if pd.isna(timestamp) else timestamp.strftime("%d/%m/%Y %H:%M")
            elif isinstance(value, (int, float)):
                display = f"{value:,.1f}" if key == "seconds" else f"{value:,.0f}"
            else:
                display = str(value)
            column.metric(label, display)


def show_realtime_accuracy(location: str) -> None:
    st.subheader("🎯 Real-time accuracy")
    names = dict(zip(PROVINCES["location"], PROVINCES["province"]))
    scope = st.segmented_control(
        "Real-time evaluation area", ["National", "Selected province"],
        default="National", key="realtime_accuracy_scope",
    ) or "National"
    selected_location = location if scope == "Selected province" else None
    scope_label = names.get(location, location) if selected_location else "National · all 34 provinces"
    st.caption(f"{scope_label} · Check past forecasts against actual values for the same province and hour.")
    st.caption("Updated every hour when new data is available.")
    predictions = read_table(
        DATA_DIR / "predictions.csv", ("location", "target", "horizon_h", "target_time", "prediction"),
        date_columns=("made_at", "base_time", "target_time"), numeric_columns=("horizon_h", "prediction"),
        valid_columns=("location", "target", "horizon_h", "target_time"),
    )
    weather = read_table(DATA_DIR / "weather_current.csv", ("location", "time", "temperature_2m"))
    air = read_table(DATA_DIR / "aq_current.csv", ("location", "time", "pm2_5"))
    om_forecasts = read_table(
        DATA_DIR / "om_forecast.csv", ("location", "fetched_at", "target_time", "temperature_2m"),
        date_columns=("fetched_at", "target_time"), numeric_columns=("temperature_2m",),
        valid_columns=("location", "fetched_at", "target_time"),
    )
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    if not predictions.empty:
        predictions = predictions.loc[predictions["location"].isin(PROVINCES["location"])]
    verified = verified_realtime_predictions(predictions, weather, air, om_forecasts, now, selected_location)
    scores = realtime_accuracy_metrics(verified)
    if selected_location is None:
        st.caption(f"Currently {verified['location'].nunique()}/{len(PROVINCES)} provinces have checked forecasts.")
    st.caption(
        "Mỗi provinces, mục tiêu, hạn và hours dự báo chỉ tính một lần, giữ lần tạo mới nhất. "
        "Actual là quan trắc cuối cùng trong hours. Open-Meteo dùng lần thu gần nhất trước target_time."
    )
    chart_horizon = st.selectbox(
        "Forecast horizon for the MAE chart", HORIZONS, format_func=lambda h: f"{h} hours", key="realtime_mae_horizon",
    )
    for target, (title, unit) in TARGETS.items():
        st.markdown(f"**{title} · MAE real-time ({unit})**")
        rows = scores.loc[scores["target"].eq(target)]
        if not rows["group_count"].sum():
            st.info("No past forecasts have matching actual values yet. We need a few more hours of data.")
        fields = ["horizon_h", "group_mae", "group_count"]
        if target == "temperature_2m":
            fields += ["group_mae_on_om", "om_mae", "om_count"]
        st.dataframe(
            realtime_score_display(rows[fields], unit), hide_index=True, use_container_width=True,
            column_config={
                "horizon_h": st.column_config.NumberColumn("Hạn dự báo", format="%d hours"),
                "group_mae": st.column_config.TextColumn("Group model MAE"),
                "group_count": st.column_config.NumberColumn("Checked group forecasts", format="%d"),
                "group_mae_on_om": st.column_config.TextColumn("Group MAE · same samples as OM"),
                "om_mae": st.column_config.TextColumn("Open-Meteo MAE"),
                "om_count": st.column_config.NumberColumn("Open-Meteo samples", format="%d"),
            },
        )
        if target == "temperature_2m":
            st.caption(
                "Group model MAE dùng toàn bộ dự báo đã kiểm chứng. "
                "Để so sánh trên cùng tập mẫu, xem “Group MAE · same samples as OM” và “Open-Meteo MAE”. "
                "“Not enough data” nghĩa là chưa có mẫu phù hợp."
            )
        st.markdown(f"**MAE {title.lower()} over time · forecast horizon: {chart_horizon} hours**")
        hourly = hourly_realtime_mae(verified, target, chart_horizon)
        if hourly.empty or not hourly["mae"].notna().any():
            st.info("Not enough data to show hourly real-time MAE.")
            continue
        st.plotly_chart(hourly_mae_figure(hourly, title, unit), use_container_width=True)
        st.caption(
            "Mỗi hours một điểm: MAE của các dự báo có cùng hours đích và hạn đang chọn, "
            "gộp các provinces trong phạm vi đánh giá. Rê chuột để xem số mẫu; hours thiếu dữ liệu được để trống."
        )
        if target == "temperature_2m":
            st.caption("The two sources may have different sample counts. Without Open-Meteo samples, the chart shows only the group model.")


def show_evaluation(location: str) -> None:
    show_realtime_accuracy(location)
    st.divider()
    metrics = read_table(
        REPORT_DIR / "metrics.csv",
        ("target", "horizon_h", "model", "MAE", "RMSE", "R2", "improve_vs_baseline_%", "test_rows"),
        date_columns=(), numeric_columns=("horizon_h", "MAE", "RMSE", "R2", "improve_vs_baseline_%", "test_rows"),
        valid_columns=("target", "horizon_h", "model"),
    )
    st.subheader("Test set results")
    st.caption("Lower MAE and RMSE are better. R² closer to 1 is better. Scores cover all provinces in the test set.")
    for target, (title, unit) in TARGETS.items():
        st.markdown(f"**{title} ({unit})**")
        rows = pd.DataFrame() if metrics.empty else metrics.loc[metrics["target"].eq(target)].sort_values(["horizon_h", "model"])
        if rows.empty:
            st.info(f"No scores for {title.lower()}.")
            continue
        st.dataframe(
            rows.drop(columns="target").rename(columns={
                "horizon_h": "Forecast horizon (hours)", "model": "Model", "R2": "R²",
                "improve_vs_baseline_%": "Improvement over Baseline (persistence) (%)", "test_rows": "Test rows",
            }), hide_index=True, use_container_width=True,
        )
        chart_data = rows.dropna(subset=["MAE"]).copy()
        if chart_data.empty:
            st.info("No valid MAE values to show yet.")
        else:
            chart_data["Hạn dự báo"] = chart_data["horizon_h"].map(lambda h: f"{h:g} hours")
            figure = px.bar(
                chart_data, x="Hạn dự báo", y="MAE", color="model", barmode="group",
                labels={"model": "Model", "MAE": f"MAE ({unit})"},
            )
            figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0})
            st.plotly_chart(figure, use_container_width=True)

    horizon = st.selectbox("Forecast horizon for test and feature charts", HORIZONS, format_func=lambda h: f"{h} hours", key="evaluation_horizon")
    st.subheader("Actual vs model vs baseline · last 7 days")
    test_data = read_table(
        REPORT_DIR / "test_predictions.csv",
        ("location", "time", "target", "horizon_h", "target_time", "actual", "baseline", "model"),
        date_columns=("time", "target_time"), numeric_columns=("horizon_h", "actual", "baseline", "model"),
        valid_columns=("location", "time", "target", "horizon_h", "target_time"),
    )
    test_location = location
    if not test_data.empty and not test_data["location"].eq(location).any():
        test_location = "tp_hcm"
        st.info("The selected province is not in the test file. Showing TP. Hồ Chí Minh.")
    for target, (title, unit) in TARGETS.items():
        st.markdown(f"**{title} · forecast horizon: {horizon} hours**")
        week = last_test_week(test_data, test_location, target, horizon)
        if week.empty:
            st.info(f"No test data for {title.lower()} for this province and forecast horizon.")
            continue
        province_names = dict(zip(PROVINCES["location"], PROVINCES["province"]))
        st.caption(
            f"{province_names.get(test_location, test_location)} · "
            f"{week['target_time'].min():%d/%m/%Y %H:%M} – {week['target_time'].max():%d/%m/%Y %H:%M}"
        )
        comparison_chart(
            week.rename(columns={"actual": "Actual", "model": "XGBoost", "baseline": "Baseline"}),
            ["Actual", "XGBoost", "Baseline"], title, unit,
        )

    st.subheader("Top 10 important features")
    importance = read_table(
        REPORT_DIR / "feature_importance.csv", ("target", "horizon_h", "feature", "importance"),
        date_columns=(), numeric_columns=("horizon_h", "importance"),
        valid_columns=("target", "horizon_h", "feature", "importance"),
    )
    for target, (title, _) in TARGETS.items():
        st.markdown(f"**{title} · forecast horizon: {horizon} hours**")
        top = pd.DataFrame() if importance.empty else importance.loc[
            importance["target"].eq(target) & importance["horizon_h"].eq(horizon)
        ].sort_values("importance", ascending=False).drop_duplicates("feature").head(10)
        if top.empty:
            st.info("No feature importance data for this target and forecast horizon.")
            continue
        figure = px.bar(
            top.sort_values("importance"), x="importance", y="feature", orientation="h",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        figure.update_layout(height=380, margin={"l": 0, "r": 0, "t": 10, "b": 0}, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, use_container_width=True)

    show_etl_report()
    images, image_message = load_report_images(str(REPORT_DIR / "figures"))
    if image_message:
        st.caption(image_message)
    if images:
        st.subheader("Report images")
        for name, content in images:
            try:
                st.image(content, caption=name, use_container_width=True)
            except (OSError, ValueError):
                st.info(f"Could not show image {name}. Please check the PNG file.")


def show_system_metrics() -> None:
    st.subheader("📊 Big Data processing with PySpark")
    report, message = load_etl_report(str(ROOT / "data" / "processed" / "etl_report.json"))
    if message:
        st.info(message)
    if not report:
        return

    def number(key: str, decimals: int = 0) -> str:
        value = report.get(key)
        return f"{value:,.{decimals}f}" if isinstance(value, (int, float)) and pd.notna(value) else "—"

    columns = st.columns(4)
    engine = report.get("engine")
    for column, label, value, detail in (
        (columns[0], "⚙️ Engine", engine if isinstance(engine, str) else "—", "ETL engine"),
        (columns[1], "🌤️ Raw weather rows", number("weather_rows_raw"), "Input rows"),
        (columns[2], "🌫️ Raw air quality rows", number("aq_rows_raw"), "Input rows"),
        (columns[3], "🔗 Merged rows", number("merged_rows"), "Rows in merged.parquet"),
    ):
        kpi_card(column, label, value, detail)

    first = pd.to_datetime(str(report.get("first_time", "")), errors="coerce")
    last = pd.to_datetime(str(report.get("last_time", "")), errors="coerce")
    period = "—" if pd.isna(first) or pd.isna(last) else f"{first:%d/%m/%Y} → {last:%d/%m/%Y}"
    period_detail = "No valid date range" if pd.isna(first) or pd.isna(last) else f"{first:%d/%m/%Y %H:%M} → {last:%d/%m/%Y %H:%M}"
    st.write("")
    columns = st.columns(3)
    kpi_card(columns[0], "📍 Provinces", number("provinces"), "Area covered by the data")
    columns[1].markdown(
        '<div class="kpi-card"><div class="kpi-label">🗓️ Date range</div>'
        f'<div class="kpi-value" style="font-size:1.15rem;line-height:1.7">{escape(period)}</div>'
        f'<div class="kpi-detail">{escape(period_detail)}</div></div>',
        unsafe_allow_html=True,
    )
    kpi_card(columns[2], "⏱️ Time xử lý", f"{number('seconds', 1)} seconds", "ETL time from the report")


def monthly_heatmap_data(monthly: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Sort months, keep all 34 provinces and remove months with no data."""
    province_names = PROVINCES["province"].tolist()
    if monthly.empty or value_column not in monthly.columns:
        return pd.DataFrame(index=pd.Index(province_names, name="province"))
    data = monthly.copy()
    names = dict(zip(PROVINCES["location"], PROVINCES["province"]))
    data["province"] = data["province"].replace(names)
    if "location" in data.columns:
        data["province"] = data["location"].map(names).combine_first(data["province"])
    data["month"] = pd.to_datetime(data["month"], format="%Y-%m", errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce").replace(
        [float("inf"), -float("inf")], float("nan")
    )
    data = data.dropna(subset=["province", "month", value_column])
    if data.empty:
        return pd.DataFrame(index=pd.Index(province_names, name="province"))
    data["month"] = data["month"].dt.strftime("%Y-%m")
    matrix = data.pivot_table(index="province", columns="month", values=value_column, aggfunc="mean")
    matrix = matrix.reindex(province_names).dropna(axis=1, how="all")
    return matrix.reindex(sorted(matrix.columns), axis=1)


def monthly_heatmap_figure(matrix: pd.DataFrame, title: str, unit: str, colors: str):
    figure = px.imshow(
        matrix, aspect="auto", origin="upper", color_continuous_scale=colors,
        labels={"x": "Month", "y": "Province", "color": f"{title} ({unit})"},
        height=1040,
    )
    figure.update_traces(
        hoverongaps=False,
        hovertemplate=f"Province: %{{y}}<br>Month: %{{x}}<br>{title}: %{{z:.2f}} {unit}<extra></extra>",
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"tickangle": -45, "nticks": 18, "title": "Month"},
        yaxis={"dtick": 1, "title": "Province", "tickfont": {"size": 11}},
    )
    return figure


def show_province_summary() -> None:
    st.subheader("📋 Data summary by province")
    summary = read_table(
        ROOT / "data" / "processed" / "summary_by_province.csv", ("province",),
        date_columns=("first_time", "last_time"),
        numeric_columns=(
            "rows", "avg_temperature", "max_temperature", "avg_pm2_5", "max_pm2_5",
            "total_precipitation", "missing_temperature", "rows_with_pm2_5",
        ),
        valid_columns=("province",),
    )
    if summary.empty:
        return
    summary = summary.copy()
    names = dict(zip(PROVINCES["location"], PROVINCES["province"]))
    summary["province"] = summary["province"].replace(names)
    if "location" in summary.columns:
        summary["province"] = summary["location"].map(names).combine_first(summary["province"])
    if "rows" in summary and "rows_with_pm2_5" in summary:
        summary["pm25_coverage_pct"] = summary["rows_with_pm2_5"].div(summary["rows"].where(summary["rows"].gt(0))).mul(100)
    summary = summary.sort_values("province", kind="stable").reset_index(drop=True)
    preferred = [
        "province", "rows", "first_time", "last_time", "avg_temperature", "max_temperature",
        "avg_pm2_5", "max_pm2_5", "pm25_coverage_pct", "rows_with_pm2_5",
        "missing_temperature", "total_precipitation",
    ]
    st.caption("One row per province. PM2.5 coverage is the share of rows with a PM2.5 value. Empty cells mean no data.")
    st.dataframe(
        summary, hide_index=True, use_container_width=True,
        column_order=[field for field in preferred if field in summary.columns],
        column_config={
            "location": None,
            "province": st.column_config.TextColumn("Province", width="medium"),
            "rows": st.column_config.NumberColumn("Total rows", format="%d"),
            "first_time": st.column_config.DatetimeColumn("Start time", format="DD/MM/YYYY HH:mm"),
            "last_time": st.column_config.DatetimeColumn("End time", format="DD/MM/YYYY HH:mm"),
            "avg_temperature": st.column_config.NumberColumn("Average temperature", format="%.2f °C"),
            "max_temperature": st.column_config.NumberColumn("Highest temperature", format="%.1f °C"),
            "avg_pm2_5": st.column_config.NumberColumn("Average PM2.5", format="%.2f µg/m³"),
            "max_pm2_5": st.column_config.NumberColumn("Highest PM2.5", format="%.1f µg/m³"),
            "pm25_coverage_pct": st.column_config.ProgressColumn("PM2.5 coverage", min_value=0, max_value=100, format="%.1f%%"),
            "rows_with_pm2_5": st.column_config.NumberColumn("Rows with PM2.5", format="%d"),
            "missing_temperature": st.column_config.NumberColumn("Rows missing temperature", format="%d"),
            "total_precipitation": st.column_config.NumberColumn("Total rain", format="%.1f mm"),
        },
    )


def show_big_data_system() -> None:
    st.caption("From data collection to Big Data processing, model training and forecasts.")
    st.subheader("🔄 Data and forecast pipeline")
    st.graphviz_chart(
        r'''
        digraph WeatherAQI {
            graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];
            node [shape=box, style="rounded,filled", fillcolor="#f4f6fa", color="#cbd5e1",
                  fontname="Arial", fontsize=11, margin="0.14,0.1"];
            edge [color="#64748b", arrowsize=0.7];
            api [label="Open-Meteo API", fillcolor="#dbeafe"];
            collect [label="collect_realtime.py\n(every hour)"];
            realtime [label="Real-time CSV\nweather_current.csv\naq_current.csv", shape=folder];
            archive [label="Open-Meteo Archive", fillcolor="#dbeafe"];
            history_fetch [label="fetch_history.py"];
            history [label="Historical CSV", shape=folder];
            etl [label="PySpark ETL\nprepare_data.py", fillcolor="#dbeafe"];
            parquet [label="merged.parquet", shape=folder];
            train [label="train.py\nXGBoost · GPU", fillcolor="#dbeafe"];
            models [label="models/", shape=folder];
            predict [label="predictor.py\n(every hour)"];
            predictions [label="predictions.csv", shape=folder];
            app [label="Streamlit", fillcolor="#1f6feb", fontcolor="white", color="#1f6feb"];
            api -> collect -> realtime;
            archive -> history_fetch -> history -> etl -> parquet -> train -> models -> predict -> predictions -> app;
            realtime -> app;
        }
        ''',
        use_container_width=True,
    )
    show_system_metrics()

    st.subheader("🧩 Monthly averages across 34 provinces")
    monthly = read_table(
        ROOT / "data" / "processed" / "monthly_means.csv",
        ("province", "month", "avg_pm2_5", "avg_temperature"), date_columns=(),
        numeric_columns=("avg_pm2_5", "avg_temperature"), valid_columns=("province", "month"),
    )
    if not monthly.empty:
        tabs = st.tabs(["🌫️ Average PM2.5", "🌡️ Average temperature"])
        for tab, field, title, unit, colors in (
            (tabs[0], "avg_pm2_5", "Average PM2.5", "µg/m³", "YlOrRd"),
            (tabs[1], "avg_temperature", "Average temperature", "°C", "RdYlBu_r"),
        ):
            with tab:
                matrix = monthly_heatmap_data(monthly, field)
                if matrix.empty:
                    st.info(f"No data {title.lower()} monthly data to show.")
                    continue
                st.caption(
                    f"{len(matrix)} provinces · {matrix.columns[0]} → {matrix.columns[-1]} · "
                    "Chỉ hiện tháng có dữ liệu; ô trống là chưa có số liệu, không phải giá trị 0."
                )
                st.plotly_chart(monthly_heatmap_figure(matrix, title, unit, colors), use_container_width=True)
    show_province_summary()


def clear_ai_chat() -> None:
    st.session_state["ai_chat_messages"] = []


def show_ai_context(context: str) -> None:
    with st.expander("📊 Data used by AI"):
        st.code(context or "No data context is available for this answer.", language="text", wrap_lines=True)


def show_ai_chat(location: str) -> None:
    """Show chat without an auto-refresh fragment."""
    st.subheader("🤖 AI chat")
    st.write("This AI assistant runs locally on a GPU. It uses real-time data and forecasts from the group model.")
    messages = st.session_state.setdefault("ai_chat_messages", [])
    with st.sidebar:
        st.divider()
        st.button("🗑️ Clear chat", key="clear_ai_chat", on_click=clear_ai_chat, use_container_width=True)
        st.caption("AI chat does not refresh itself, so you can keep typing.")

    try:
        ok, message = assistant.check_ollama()
    except Exception:
        ok, message = False, "Could not check Ollama. Open Ollama on the server and make sure the model is downloaded."
    color, background = ("#15803d", "#dcfce7") if ok else ("#b91c1c", "#fee2e2")
    status = "🟢 Ollama is ready" if ok else "🔴 Ollama is not ready"
    st.markdown(
        f'<span style="display:inline-block;padding:6px 12px;border-radius:999px;'
        f'background:{background};color:{color};font-weight:600">{status}</span>',
        unsafe_allow_html=True,
    )
    if ok:
        st.caption(message)
    else:
        st.error(message)
        st.caption("After opening Ollama or downloading the model, click the button below to check again.")
        st.button("🔄 Check connection again", key="ai_check_connection")

    for entry in messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                show_ai_context(entry.get("context", ""))

    suggestions = st.empty()
    suggested_question = None
    if not messages:
        with suggestions.container():
            st.markdown("**Try a question**")
            questions = (
                "How is the weather in Hanoi in the next 3 hours?",
                "Which province has the worst air right now?",
                "Is it a good time to go running in Saigon?",
                "Compare the weather in Da Nang and Da Lat",
            )
            columns = st.columns(2)
            for index, question in enumerate(questions):
                if columns[index % 2].button(
                    question, key=f"ai_suggestion_{index}", disabled=not ok, use_container_width=True,
                ):
                    suggested_question = question

    submitted_question = st.chat_input(
        "Ask about weather, air quality or forecasts…", key="ai_chat_input", disabled=not ok,
    )
    question = submitted_question or suggested_question
    if not ok or not question or not question.strip():
        return
    question = question.strip()
    suggestions.empty()
    # ask() tự thêm câu hỏi hiện tại: lịch sử chỉ gồm các lượt trước, không chứa metadata.
    history = [{"role": entry["role"], "content": entry["content"]} for entry in messages]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    context = ""
    with st.chat_message("assistant"):
        try:
            with st.spinner("AI is reading the data and preparing an answer…"):
                # Chụp ngữ cảnh trước khi sinh câu trả lời, không đọc lại khi xem lịch sử.
                context = assistant.build_context(question, location)
                response = st.write_stream(assistant.ask(question, history, location))
            if not response:
                response = "The AI did not return an answer. Please try your question again."
                st.info(response)
        except Exception:
            response = "Could not finish the answer. Please check Ollama and try again."
            st.error(response)
        show_ai_context(context)
    messages.append({"role": "assistant", "content": str(response), "context": context, "location": location})


@st.fragment(run_every=300)
def render_page(page: str, location: str) -> None:
    """Read data and update the open data page every five minutes."""
    names = dict(zip(PROVINCES["location"], PROVINCES["province"]))
    titles = {
        "overview": "🗺️ National overview", "current": "🌤️ Current",
        "history": "📈 Real-time history", "forecast": "🔮 Forecast", "evaluation": "🎯 Model evaluation",
        "system": "⚙️ Big Data system",
    }
    st.subheader(titles[page] if page in ("overview", "system") else f"{titles[page]} · {names[location]}")
    refreshed_at = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    st.caption(f"Refreshes every 5 minutes · Last shown: {refreshed_at:%H:%M:%S · %d/%m/%Y} (UTC+7)")
    if page == "system":
        show_big_data_system()
    elif page == "forecast":
        show_forecast(location)
    elif page == "evaluation":
        show_evaluation(location)
    else:
        weather, weather_message = load_csv(
            str(DATA_DIR / "weather_current.csv"),
            ("location", "time", "fetched_at", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"),
        )
        air, air_message = load_csv(
            str(DATA_DIR / "aq_current.csv"),
            ("location", "time", "fetched_at", "pm2_5", "us_aqi"),
        )
        for message in (weather_message, air_message):
            if message:
                st.info(message)
        if page == "overview":
            show_overview(weather, air)
        elif page == "current":
            show_current(weather, air, location)
        else:
            show_history(weather, air, location)


def main() -> None:
    st.set_page_config(
        page_title="Weather & AQI VN", page_icon="🌤️", layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()
    st.title("Weather & AQI VN")
    st.caption("Track 34 provinces · Observation times are in Vietnam time (UTC+7)")

    names = dict(zip(PROVINCES["location"], PROVINCES["province"]))
    codes = list(names)
    with st.sidebar:
        st.header("🌤️ Weather & AQI VN")
        st.caption("Weather and air quality at a glance")
        location = st.selectbox(
            "📍 Province", codes, index=codes.index("tp_hcm"), format_func=names.get,
            key="selected_province",
        )
        st.caption("Used by Current, History, Forecast, Model evaluation and AI chat.")
        st.divider()
        st.caption("Data pages refresh every 5 minutes. AI chat does not refresh itself.")

    navigation = st.navigation([
        st.Page(lambda: render_page("overview", location), title="National overview", icon="🗺️", default=True, url_path="tong-quan"),
        st.Page(lambda: render_page("current", location), title="Current", icon="🌤️", url_path="hien-tai"),
        st.Page(lambda: render_page("history", location), title="Real-time history", icon="📈", url_path="lich-su"),
        st.Page(lambda: render_page("forecast", location), title="Forecast", icon="🔮", url_path="du-bao"),
        st.Page(lambda: render_page("evaluation", location), title="Model evaluation", icon="🎯", url_path="danh-gia"),
        st.Page(lambda: render_page("system", location), title="Big Data system", icon="⚙️", url_path="he-thong-big-data"),
        st.Page(lambda: show_ai_chat(location), title="AI chat", icon="🤖", url_path="hoi-dap-ai"),
    ])
    navigation.run()
    st.divider()
    st.caption("Data: Open-Meteo, Copernicus Atmosphere Monitoring Service (CAMS). Group 8.")


if __name__ == "__main__":
    main()
