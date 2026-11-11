# Weather & AQI VN

Hourly temperature and PM2.5 forecasts for all 34 provinces of Vietnam.
Final project for the Big Data Analytics course, **Group 8**.

- **Project page:** https://kalu-cassano.github.io/weather-aqi-vn/
- **Live demo:** https://group8-bigdata-weather.streamlit.app

## What it does

- Collects new weather and air quality data for 34 provinces **every hour** (Open-Meteo API).
- **Scrapes weather and disaster news** from 8 Vietnamese news feeds every hour (`news_scraper.py`).
- Processes about **2.77 million rows** of history with **PySpark**.
- Trains **6 XGBoost models** (temperature and PM2.5, 1, 3 and 6 hours ahead) and compares them with baselines.
- Saves **204 new forecasts every hour** and checks old forecasts against real values.
- Has an **AI assistant** that answers questions with the project's data and cites news sources. On the PC it runs Qwen3 8B on a local GPU (Ollama); the online demo uses the Groq API.

## Pipeline

| Step | Script | Runs |
|---|---|---|
| 1. Collect real-time data and scrape news | `collect_realtime.py`, `news_scraper.py` | every hour (GitHub Actions + PC) |
| 2. Download history | `fetch_history.py` | once |
| 3. Clean and join data with PySpark | `prepare_data.py` | before training |
| 4. Train and evaluate models | `train.py` | when retraining |
| 5. Forecast in real time | `predictor.py` | every hour (called by step 1) |
| 6. Web app | `app.py` | always on |
| AI assistant | `assistant.py` | used by the web app |

Shared code: `config.py` (34 provinces, variables, API links), `om_client.py` (API calls with retry), `features.py` (feature engineering for training and real-time).

## Results (test set: last 180 days, 146,914 samples per model)

Mean absolute error (lower is better):

| Variable | Horizon | Baseline (persistence) | Ridge | **XGBoost** |
|---|---|---|---|---|
| Temperature (°C) | 1 h | 0.65 | 0.44 | **0.37** |
| | 3 h | 1.60 | 0.81 | **0.65** |
| | 6 h | 2.71 | 1.00 | **0.80** |
| PM2.5 (µg/m³) | 1 h | 2.59 | 2.13 | **1.74** |
| | 3 h | 6.43 | 5.12 | **4.22** |
| | 6 h | 10.43 | 7.25 | **6.38** |

Full results: `reports/metrics.csv`. Charts: `reports/figures/`.

## How to run

Requirements: Python 3.10+, Java 17 or 21 (for PySpark), an NVIDIA GPU is optional (XGBoost and Ollama use it if available).

```bash
pip install -r requirements.txt

python collect_realtime.py      # keep running: collects data every hour
python fetch_history.py         # download history (takes some time because of API limits)
python prepare_data.py          # PySpark ETL (use --engine pandas if Spark fails)
python train.py                 # train and evaluate the models
python -m streamlit run app.py  # open the web app
```

AI assistant (optional): install [Ollama](https://ollama.com), then run `ollama pull qwen3:8b`.

## Known limitations

- The data comes from models, not weather stations: the weather history is reanalysis data, and PM2.5 comes from the CAMS model (about 45 km, one value every 3 hours). This data is smoother than real sensor data.
- We train on reanalysis data but run on Open-Meteo's current data, so real accuracy can be different. The web app tracks it on the Evaluation page.
- Forecasts are short-term only (1 to 6 hours).
- The online demo sends the question and the selected data to the Groq API.
- Data is stored as CSV/Parquet files in the repository, not in a database.
- Spark runs in local mode on one PC. On Windows, data is loaded into Spark through Apache Arrow because Hadoop cannot read local files without extra tools.

## Data sources

- Weather: [Open-Meteo](https://open-meteo.com/) (ERA5, ECMWF IFS)
- Air quality: Copernicus Atmosphere Monitoring Service (CAMS), via Open-Meteo
- News: RSS feeds and article pages of VnExpress, Tuổi Trẻ, Thanh Niên, VietnamPlus, VnExpress International, Tuoi Tre News and Viet Nam News (title, short summary and link only; robots.txt respected)
- Provinces: the 34 provincial-level units of Vietnam in effect from 1 July 2025

## Team (Group 8)

- Ngô Khải Luân
- Nguyễn Thanh Sơn
