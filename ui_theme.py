"""Visual identity for the Streamlit app, matching the project landing page (docs/index.html).

Calm and bright: off-white page, deep slate text, one teal accent, few borders, generous space.

Usage in app.py:
    import ui_theme as ui
    ui.apply_theme()                     # call once, right after st.set_page_config(...)
    ui.page_header("National overview", "Latest data for all 34 provinces")
    ui.kpi_row([ui.kpi("Highest temperature", "25.5 °C", "TP. Hồ Chí Minh"), ...])
    fig.update_layout(template="weather")   # or rely on the default set below
    ui.style_map(fig)                    # for px.scatter_map figures
"""
import base64
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Palette (same values as the landing page)
# ---------------------------------------------------------------------------
PAPER = "#fbfcfd"
PANEL = "#ffffff"
INK = "#15242e"
INK_SOFT = "#5d6d77"
RULE = "#e4eaee"
ACCENT = "#0d8a80"        # teal: our model, primary actions
ACCENT_SOFT = "#e1f4f1"
SKY = "#e6f3f6"
BAR_1 = "#c9d3d9"         # baseline
BAR_2 = "#9db4be"         # second baseline / Ridge
BAR_3 = "#6f8f9c"
OPEN_METEO = "#e08a3c"    # Open-Meteo forecast lines (warm, clearly "other")
FONT = "Be Vietnam Pro, Segoe UI, Roboto, sans-serif"

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = str(ASSETS / "logo.svg")
LOGO_ICON = str(ASSETS / "logo_icon.svg")


def _data_uri(name):
    p = ASSETS / name
    if not p.exists():
        return ""
    return "data:image/svg+xml;base64," + base64.b64encode(p.read_bytes()).decode()


HERO_ART = _data_uri("halong.svg")   # original illustration of Ha Long Bay

# Material icons for st.Page(..., icon=ui.ICONS["overview"])
ICONS = {
    "overview": ":material/map:",
    "current": ":material/partly_cloudy_day:",
    "history": ":material/timeline:",
    "forecast": ":material/query_stats:",
    "evaluation": ":material/fact_check:",
    "bigdata": ":material/hub:",
    "chat": ":material/forum:",
}

# Colors for model names (any name not listed gets a neutral grey)
MODEL_COLORS = {
    "XGBoost": ACCENT,
    "Ridge Regression": BAR_2,
    "Baseline (persistence)": BAR_1,
    "Baseline (same hour yesterday)": BAR_3,
    "Open-Meteo": OPEN_METEO,
    "Actual": INK,
}

# US EPA AQI bands: (upper limit, label, color)
AQI_BANDS = [
    (50, "Good", "#22a35a"),
    (100, "Moderate", "#e0c12c"),
    (150, "Unhealthy for sensitive groups", "#f08a24"),
    (200, "Unhealthy", "#d93b3b"),
    (300, "Very unhealthy", "#8a4c9a"),
    (10_000, "Hazardous", "#7a2236"),
]


def aqi_label(value):
    if value is None or value != value:  # None or NaN
        return "No data"
    for limit, label, _ in AQI_BANDS:
        if value <= limit:
            return label
    return "Hazardous"


def aqi_color(value):
    if value is None or value != value:
        return BAR_1
    for limit, _, color in AQI_BANDS:
        if value <= limit:
            return color
    return AQI_BANDS[-1][2]


# Continuous scales for maps
AQI_SCALE = [[0.0, "#22a35a"], [0.1, "#22a35a"], [0.2, "#e0c12c"], [0.3, "#f08a24"],
             [0.4, "#d93b3b"], [0.6, "#8a4c9a"], [1.0, "#7a2236"]]   # use with range_color=(0, 500)
TEMP_SCALE = [[0.0, "#3b7dd8"], [0.35, "#7fc4d6"], [0.55, "#f3e3a1"], [0.75, "#f2a65a"], [1.0, "#d9483b"]]

# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
pio.templates["weather"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[ACCENT, OPEN_METEO, BAR_2, "#3b7dd8", BAR_3, "#8a4c9a"],
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left"),
        xaxis=dict(showgrid=False, linecolor=RULE, ticks="outside", tickcolor=RULE,
                   tickfont=dict(color=INK_SOFT), title=dict(font=dict(color=INK_SOFT))),
        yaxis=dict(gridcolor=RULE, zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=INK_SOFT), title=dict(font=dict(color=INK_SOFT))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SOFT)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(family=FONT, color=INK)),
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.25, bargroupgap=0.08,
    )
)
pio.templates.default = "weather"
px.defaults.template = "weather"

MAP_STYLE = "carto-positron"          # light, quiet basemap
MAP_CENTER = {"lat": 16.0, "lon": 106.3}
MAP_ZOOM = 4.6


def style_map(fig, height=640):
    """Use on px.scatter_map figures: light basemap centered on Vietnam."""
    fig.update_layout(map=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM),
                      height=height, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(marker=dict(size=16, opacity=0.9))
    return fig


def style_chart(fig, height=380):
    fig.update_layout(template="weather", height=height)
    return fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: {FONT} !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] .stDeployButton, .stDeployButton {{ display: none; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}

h1, h2, h3 {{ color: {INK}; letter-spacing: -0.02em; font-weight: 600 !important; }}
h1 {{ font-size: 2.3rem !important; }}
h2 {{ font-size: 1.55rem !important; margin-top: 1.6rem !important; }}
h3 {{ font-size: 1.15rem !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {INK_SOFT} !important; }}
hr {{ border-color: {RULE} !important; }}

/* Sidebar: white, quiet, logo on top, pill for the current page */
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: .4rem; }}
[data-testid="stSidebarHeader"] {{ padding: 1.2rem 1.2rem .6rem; }}
[data-testid="stLogo"] {{ height: 2.6rem !important; max-width: 100%; }}
[data-testid="stSidebarNav"] {{ padding-top: .2rem; }}
[data-testid="stSidebarNav"] ul {{ gap: 2px; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; padding: .45rem .75rem; transition: background .15s ease; }}
[data-testid="stSidebarNav"] a:hover {{ background: {PAPER}; }}
[data-testid="stSidebarNav"] a span {{ color: {INK_SOFT} !important; font-size: .95rem; }}
[data-testid="stSidebarNav"] a [data-testid="stIconMaterial"] {{ color: {INK_SOFT} !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {ACCENT_SOFT}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNav"] a[aria-current="page"] [data-testid="stIconMaterial"] {{ color: {ACCENT} !important; font-weight: 600; }}
[data-testid="stSidebarNavSeparator"] {{ display: none; }}
[data-testid="stSidebarUserContent"] {{ padding-top: .4rem; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] label p {{ font-size: .82rem; color: {INK_SOFT} !important; font-weight: 500; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ border-radius: 12px; border-color: {RULE}; background: {PAPER}; }}
[data-testid="stSidebar"] hr {{ margin: .9rem 0; }}
.wq-side-note {{ color: {INK_SOFT}; font-size: .8rem; line-height: 1.5; }}
.wq-side-note b {{ color: {INK}; font-weight: 600; }}

/* Page header band */
.wq-header {{ position: relative; overflow: hidden; background: linear-gradient(180deg, {SKY} 0%, {PAPER} 100%);
  border-radius: 18px; padding: 30px 32px 28px; margin-bottom: 26px; min-height: 150px; }}
.wq-header .art {{ position: absolute; right: 0; bottom: 0; height: 100%; width: 58%; object-fit: cover;
  object-position: right bottom; pointer-events: none;
  -webkit-mask-image: linear-gradient(to right, transparent 0%, #000 35%); mask-image: linear-gradient(to right, transparent 0%, #000 35%); }}
.wq-header .txt {{ position: relative; max-width: 60%; }}
@media (max-width: 900px) {{ .wq-header .art {{ opacity: .35; width: 100%; }} .wq-header .txt {{ max-width: 100%; }} }}
.wq-header .eyebrow {{ color: {ACCENT}; font-weight: 600; font-size: .9rem; margin-bottom: 6px; }}
.wq-header h1 {{ margin: 0 !important; padding: 0 !important; font-size: 2.1rem !important; font-weight: 500 !important; }}
.wq-header p {{ color: {INK_SOFT}; margin: 8px 0 0; font-size: 1rem; max-width: 70ch; }}

/* KPI cards */
.wq-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 6px 0 22px; }}
.wq-kpi {{ background: {PANEL}; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(15,30,40,.05), 0 8px 24px rgba(15,30,40,.05); }}
.wq-kpi .lab {{ color: {INK_SOFT}; font-size: .88rem; }}
.wq-kpi .val {{ font-size: 1.9rem; font-weight: 500; letter-spacing: -.02em; line-height: 1.25; margin-top: 6px;
  font-variant-numeric: tabular-nums; color: {INK}; }}
.wq-kpi .sub {{ color: {INK_SOFT}; font-size: .86rem; margin-top: 2px; }}
.wq-kpi.accent .val {{ color: {ACCENT}; }}
.wq-pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-top: 8px; }}

/* Native st.metric, in case the app still uses it */
[data-testid="stMetric"] {{ background: {PANEL}; border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(15,30,40,.05), 0 8px 24px rgba(15,30,40,.05); }}
[data-testid="stMetricValue"] {{ font-weight: 500; letter-spacing: -.02em; }}
[data-testid="stMetricLabel"] p {{ color: {INK_SOFT} !important; }}

/* Buttons, segmented controls, tabs */
.stButton > button, .stDownloadButton > button {{ border-radius: 999px; border: 1px solid {RULE};
  background: {PANEL}; color: {INK}; font-weight: 500; padding: .45rem 1.1rem; }}
.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.stButton > button[kind="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
[data-baseweb="tab-list"] {{ gap: 6px; }}
[data-baseweb="tab"] {{ border-radius: 999px !important; padding: 6px 16px !important; }}
[data-baseweb="tab-highlight"] {{ background: {ACCENT} !important; }}

/* Tables and charts */
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 14px; overflow: hidden; border: 1px solid {RULE}; }}
[data-testid="stPlotlyChart"] {{ background: {PANEL}; border-radius: 16px; padding: 10px 8px;
  box-shadow: 0 1px 2px rgba(15,30,40,.04); }}
[data-testid="stExpander"] {{ border: 1px solid {RULE}; border-radius: 14px; background: {PANEL}; }}
[data-testid="stAlert"] {{ border-radius: 14px; }}

/* Chat */
[data-testid="stChatMessage"] {{ background: {PANEL}; border-radius: 16px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(15,30,40,.04); }}
[data-testid="stChatInput"] textarea {{ border-radius: 14px; }}
</style>
"""


def apply_theme():
    # Markdown ends an HTML block at a blank line, so remove every blank line and indentation
    css = "\n".join(line.strip() for line in _CSS.splitlines() if line.strip())
    st.markdown(css, unsafe_allow_html=True)


def page_header(title, subtitle="", eyebrow="Weather & AQI VN", art=True):
    """Header band on top of each page, with the Ha Long Bay illustration on the right."""
    img = f'<img class="art" src="{HERO_ART}" alt="">' if (art and HERO_ART) else ""
    st.markdown(
        f'<div class="wq-header">{img}<div class="txt"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
        + (f"<p>{subtitle}</p>" if subtitle else "") + "</div></div>",
        unsafe_allow_html=True,
    )


def setup_logo():
    """Logo at the top of the sidebar (call once, after apply_theme)."""
    try:
        st.logo(LOGO, icon_image=LOGO_ICON, size="large")
    except TypeError:  # older Streamlit without size=
        st.logo(LOGO, icon_image=LOGO_ICON)


def sidebar_note(updated_text=""):
    """Small quiet note at the bottom of the sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown(
        '<div class="wq-side-note"><b>Group 8</b> · Big Data Analytics<br>'
        "Data: Open-Meteo, CAMS"
        + (f"<br>{updated_text}" if updated_text else "") + "</div>",
        unsafe_allow_html=True,
    )


def kpi(label, value, sub="", accent=False, pill=None, pill_color=None):
    """Return HTML for one KPI card. pill: optional small colored label (e.g. an AQI level)."""
    pill_html = ""
    if pill:
        bg = pill_color or ACCENT_SOFT
        fg = "#ffffff" if pill_color else ACCENT
        pill_html = f'<div><span class="wq-pill" style="background:{bg};color:{fg}">{pill}</span></div>'
    return (f'<div class="wq-kpi{" accent" if accent else ""}"><div class="lab">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div>{pill_html}</div>')


def kpi_row(cards):
    st.markdown('<div class="wq-kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)
