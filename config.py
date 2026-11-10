"""Cấu hình chung cho dự án Dự báo Thời tiết & Chất lượng Không khí Việt Nam.

Mỗi tỉnh/thành (34 đơn vị hành chính cấp tỉnh sau sáp nhập 01/07/2025) được
đại diện bằng một điểm tọa độ đặt tại trung tâm hành chính của tỉnh.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Thư mục dữ liệu
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HISTORY_WEATHER_DIR = DATA_DIR / "history" / "weather"
HISTORY_AQ_DIR = DATA_DIR / "history" / "air_quality"
REALTIME_DIR = DATA_DIR / "realtime"
LOG_DIR = ROOT / "logs"

TIMEZONE = "Asia/Ho_Chi_Minh"

# ---------------------------------------------------------------------------
# Địa điểm: (mã ngắn, tên tỉnh/thành, điểm đại diện, vĩ độ, kinh độ)
# ---------------------------------------------------------------------------
LOCATIONS = [
    # 11 tỉnh/thành giữ nguyên
    ("ha_noi",      "Hà Nội",       "Hà Nội",          21.0285, 105.8542),
    ("hue",         "Huế",          "Huế",             16.4637, 107.5909),
    ("lai_chau",    "Lai Châu",     "Lai Châu",        22.3964, 103.4582),
    ("dien_bien",   "Điện Biên",    "Điện Biên Phủ",   21.3860, 103.0230),
    ("son_la",      "Sơn La",       "Sơn La",          21.3270, 103.9141),
    ("lang_son",    "Lạng Sơn",     "Lạng Sơn",        21.8537, 106.7615),
    ("quang_ninh",  "Quảng Ninh",   "Hạ Long",         20.9599, 107.0425),
    ("thanh_hoa",   "Thanh Hóa",    "Thanh Hóa",       19.8067, 105.7852),
    ("nghe_an",     "Nghệ An",      "Vinh",            18.6796, 105.6813),
    ("ha_tinh",     "Hà Tĩnh",      "Hà Tĩnh",         18.3428, 105.9057),
    ("cao_bang",    "Cao Bằng",     "Cao Bằng",        22.6657, 106.2577),
    # 23 tỉnh/thành mới sau sáp nhập
    ("tuyen_quang", "Tuyên Quang",  "Tuyên Quang",     21.8236, 105.2140),
    ("lao_cai",     "Lào Cai",      "Yên Bái",         21.7229, 104.9113),
    ("thai_nguyen", "Thái Nguyên",  "Thái Nguyên",     21.5942, 105.8482),
    ("phu_tho",     "Phú Thọ",      "Việt Trì",        21.3227, 105.4019),
    ("bac_ninh",    "Bắc Ninh",     "Bắc Giang",       21.2731, 106.1946),
    ("hung_yen",    "Hưng Yên",     "Hưng Yên",        20.6464, 106.0511),
    ("hai_phong",   "Hải Phòng",    "Hải Phòng",       20.8449, 106.6881),
    ("ninh_binh",   "Ninh Bình",    "Ninh Bình",       20.2506, 105.9745),
    ("quang_tri",   "Quảng Trị",    "Đồng Hới",        17.4833, 106.6000),
    ("da_nang",     "Đà Nẵng",      "Đà Nẵng",         16.0544, 108.2022),
    ("quang_ngai",  "Quảng Ngãi",   "Quảng Ngãi",      15.1214, 108.8044),
    ("gia_lai",     "Gia Lai",      "Quy Nhơn",        13.7820, 109.2196),
    ("khanh_hoa",   "Khánh Hòa",    "Nha Trang",       12.2388, 109.1967),
    ("lam_dong",    "Lâm Đồng",     "Đà Lạt",          11.9404, 108.4583),
    ("dak_lak",     "Đắk Lắk",      "Buôn Ma Thuột",   12.6667, 108.0500),
    ("tp_hcm",      "TP. Hồ Chí Minh", "TP. Hồ Chí Minh", 10.7769, 106.7009),
    ("dong_nai",    "Đồng Nai",     "Biên Hòa",        10.9574, 106.8427),
    ("tay_ninh",    "Tây Ninh",     "Tân An",          10.5360, 106.4130),
    ("can_tho",     "Cần Thơ",      "Cần Thơ",         10.0452, 105.7469),
    ("vinh_long",   "Vĩnh Long",    "Vĩnh Long",       10.2537, 105.9722),
    ("dong_thap",   "Đồng Tháp",    "Mỹ Tho",          10.3600, 106.3600),
    ("ca_mau",      "Cà Mau",       "Cà Mau",           9.1769, 105.1524),
    ("an_giang",    "An Giang",     "Rạch Giá",        10.0125, 105.0809),
]

# ---------------------------------------------------------------------------
# Biến dữ liệu (dùng chung cho dữ liệu lịch sử và real-time để khớp cột)
# ---------------------------------------------------------------------------
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

AQ_VARS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

# Dữ liệu chất lượng không khí toàn cầu (CAMS Global) chỉ có từ tháng 8/2022
AQ_EARLIEST_DATE = "2022-08-01"

# ---------------------------------------------------------------------------
# Endpoint Open-Meteo (miễn phí, không cần API key)
# ---------------------------------------------------------------------------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
