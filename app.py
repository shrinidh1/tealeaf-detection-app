import base64
import csv
import html
import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

from disease_info import get_info

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "yolo11_S4_best.pt"
ASSETS_DIR = BASE_DIR / "assets"          # logo.png and the four disease pictures live here
LOGO_PATH = ASSETS_DIR / "logo.png"      # your leaf logo (put logo.png inside an "assets" folder)
GITHUB_URL = ""      # paste your repo link to show a "view the source code" link in the intro
DATA_CREDIT = ""     # optional dataset credit shown in the footer
GUIDE_ORDER = ["Anthracnose", "Brown Blight", "White Spot", "Algal Leaf Spot"]
CONF_MIN, CONF_MAX = 0.10, 0.90   # range of the confidence slider
APP_NAME = "TeaLeaf-AI"
APP_TITLE = "Tea Plant Disease Detection Tool"
PROJECT_TITLE = "Development of Tea Plant Disease Detection Tool"
DEPARTMENT = "Department of Information Technology"
TEAM = [  # (icon, name, role) - shown in the project card at the bottom of the home page
    ("teacher", "Prof. Jaidhar C. D.", "Principal Investigator"),
    ("student", "Shrinidhi Angadi", "Student Research Intern"),
]

# ---------- logo (read from assets/logo.png; falls back to a built-in SVG if the file is missing) ----------
FALLBACK_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="#485E21">'
    '<path d="M16 2C21.5 8.5 21.5 19.5 16 30 10.5 19.5 10.5 8.5 16 2Z"/>'
    '<path d="M2.5 9C12 9 14.5 16.5 14 25 6 23 2.5 17 2.5 9Z"/>'
    '<path d="M29.5 9C20 9 17.5 16.5 18 25 26 23 29.5 17 29.5 9Z"/></svg>'
)


def _logo_uri():
    if LOGO_PATH.exists():
        return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return "data:image/svg+xml;base64," + base64.b64encode(FALLBACK_LOGO_SVG.encode()).decode()


def _page_icon():
    try:
        return Image.open(LOGO_PATH)
    except Exception:
        return "🍃"


LOGO_URI = _logo_uri()

SHIELD_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="#485E21" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>'
)

# ---------- About page content (edit here) ----------
MODEL_NAME = "YOLO11 (small)"
DATASET_IMAGES = "1481 photographs of tea leaves"
DATA_SPLIT = "80% training / 20% test"
MODEL_ROWS = [
    ("Model", MODEL_NAME),
    ("Task", "Object detection (draws boxes around diseased areas)"),
    ("Classes", "4 (Anthracnose, Brown Blight, White Spot, Algal Leaf Spot)"),
    ("Input size", "640 \u00d7 640 pixels"),
    ("Dataset", DATASET_IMAGES),
    ("Data split", DATA_SPLIT),
]
# (disease, precision, recall, mAP50, mAP50-95)
METRICS = [
    ("Anthracnose", 0.729, 0.851, 0.865, 0.520),
    ("Brown Blight", 0.594, 0.571, 0.575, 0.287),
    ("White Spot", 0.984, 1.000, 0.995, 0.675),
    ("Algal Leaf Spot", 0.565, 0.741, 0.674, 0.533),
]
METRICS_ALL = ("All classes", 0.718, 0.791, 0.777, 0.504)

ICONS = {
    "teacher": ('<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="#7a7f70">'
                '<path d="M9 3h12a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-8v-2h7V5H10v1H8V4a1 1 0 0 1 1-1z"/>'
                '<circle cx="6.5" cy="10.5" r="3"/><path d="M1 21c0-3.6 2.4-6 5.5-6s5.5 2.4 5.5 6z"/></svg>'),
    "student": ('<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="#7a7f70">'
                '<path d="M12 2 1.5 7 12 12l8.5-4V13h1.5V7z"/>'
                '<path d="M6 11.5v1.6c0 1.2 2.7 2.4 6 2.4s6-1.2 6-2.4v-1.6L12 14z"/>'
                '<path d="M3 22c0-3.3 4-5 9-5s9 1.7 9 5z"/></svg>'),
}

st.set_page_config(page_title=APP_TITLE, page_icon=_page_icon(), layout="wide",
                   initial_sidebar_state="collapsed")

# Palette from the Figma design: #485E21 (dark green), #C5D378 (light green), #D2D3CB (grey)
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
:root { --dark:#485E21; --dark-hover:#3a4c1a; --light:#C5D378; --grey:#D2D3CB;
        --text:#1f2417; --muted:#7a7f70; --line:#dfe2d6; --soft:#f4f6ea; }

html, body, .stApp, [data-testid="stMarkdownContainer"], label, button, input, p {
    font-family: 'Poppins', sans-serif;
}
html, body, .stApp, [data-testid="stMain"] { overflow-x: hidden; }

/* page background: light green with soft green glows down both sides.
   The glows tile every 900px and use attachment: local, so they scroll WITH the page (never fixed behind results) */
.stApp { background: #F4FFDF; }
[data-testid="stMain"], section.main {
    background-image: radial-gradient(ellipse 430px 340px at 0% 420px, rgba(197,211,120,.85), rgba(197,211,120,0)),
                      radial-gradient(ellipse 430px 340px at 100% 480px, rgba(197,211,120,.85), rgba(197,211,120,0));
    background-size: 100% 900px; background-repeat: repeat-y; background-attachment: local;
    scroll-behavior: smooth;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

/* wider main column: navbar, hero card, input card and footer all share the same left/right edges */
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1090px !important; width: 100%;
    margin-left: auto !important; margin-right: auto !important;
    padding: 3rem 2rem 0 2rem !important;
}

/* ---------- navbar ---------- */
.brand-wrap { margin: 0; padding: 0; }
.brand-link { display: inline-flex; align-items: center; gap: .7rem; line-height: 1; }
.brand-link img { width: 44px; height: 44px; object-fit: contain; display: block; }
.brand-link span { display: block; font-weight: 500; font-size: 1.5rem; line-height: 1; color: #1b1b1b; }
.brand-link, .brand-link:hover { text-decoration: none !important; color: inherit; }

[data-testid="stElementContainer"]:has([data-testid="stPageLink"]) { width: 100%; }
[data-testid="stPageLink"] { display: flex; justify-content: flex-end; width: 100%; }
[data-testid="stPageLink"] a { margin-left: auto; background: var(--dark) !important; border-radius: 999px;
                               padding: .5rem 1.5rem; text-decoration: none; width: auto; }
[data-testid="stPageLink"] a:hover { background: var(--dark-hover) !important; }
[data-testid="stPageLink"] a, [data-testid="stPageLink"] a:link, [data-testid="stPageLink"] a:visited,
[data-testid="stPageLink"] a:hover, [data-testid="stPageLink"] a:active, [data-testid="stPageLink"] a:focus,
[data-testid="stPageLink"] a * { color: #ffffff !important; font-weight: 600; font-size: .9rem; }

/* "About" (middle navbar column) is a plain text link, not a pill */
[data-testid="stColumn"]:nth-child(2) [data-testid="stPageLink"] a,
[data-testid="column"]:nth-child(2) [data-testid="stPageLink"] a { background: transparent !important; padding: .4rem .2rem; }
[data-testid="stColumn"]:nth-child(2) [data-testid="stPageLink"] a:hover,
[data-testid="column"]:nth-child(2) [data-testid="stPageLink"] a:hover { background: transparent !important; }
[data-testid="stColumn"]:nth-child(2) [data-testid="stPageLink"] a *,
[data-testid="column"]:nth-child(2) [data-testid="stPageLink"] a * { color: #6b7062 !important; font-weight: 500; font-size: 1.05rem; }
[data-testid="stColumn"]:nth-child(2) [data-testid="stPageLink"] a:hover *,
[data-testid="column"]:nth-child(2) [data-testid="stPageLink"] a:hover * { color: var(--dark) !important; }

/* ---------- hero (no card, everything centred) ---------- */
.hero { text-align: center; padding: 0; margin: 4.5rem 0 4.5rem 0; }
.hero-title { font-weight: 700; font-size: 3.2rem; line-height: 1.2; color: var(--dark);
              letter-spacing: -0.01em; text-align: center; width: 100%; }
.hero-sub { font-size: 1.2rem; line-height: 1.6; color: #111111; text-align: center !important;
            max-width: 680px; width: 100%; margin: 1.1rem auto 1.7rem auto !important; }
.hero-sub:last-child { margin-bottom: 0 !important; }
.hero.compact { margin: 2.6rem 0 3.4rem 0; }
.hero.compact .hero-sub { max-width: 560px; }
.hero-sub a { color: var(--dark); font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; justify-content: center; gap: .8rem; }
.chip { border: 1px solid #7b8070; color: #5f6456; border-radius: 999px; padding: .35rem 1.05rem; font-size: .95rem; }

/* ---------- input card ---------- */
[data-testid="stForm"] { background: #ffffff; border: 1.5px solid var(--grey); border-radius: 8px;
                         padding: 1.5rem 1.9rem 1.5rem 1.9rem; }
.field-label { font-weight: 500; font-size: 1.3rem; color: var(--text); border-left: 4px solid var(--light);
               padding-left: .7rem; line-height: 1.2; margin: 0 0 1rem 0; border-left-width: 5px; }
.field-label.gap { margin-top: 1.4rem; }
.upload-help { font-size: .95rem; color: var(--muted); margin: .7rem 0 0 0; }

/* upload box: dashed outline, ONE "Upload" button pinned on the left, file name / limits to its right */
[data-testid="stFileUploaderDropzone"] { background: #ffffff; border: 1.2px dashed #5f6456; border-radius: 4px;
                                         padding: 1.15rem 1.2rem; display: flex; flex-direction: row;
                                         justify-content: flex-start; align-items: center; gap: 1rem; }
/* the default "Limit 200MB per file" text is replaced by the caption under the box */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

/* only the real browse button gets restyled (the small "x" that removes a file keeps its own look) */
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    order: -1; background: #ffffff; border: 1px solid #6b7062; border-radius: 4px;
    padding: .45rem 1rem; min-height: 0; font-size: 0 !important; line-height: 1;
    display: inline-flex; align-items: center; gap: .4rem; flex: 0 0 auto; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] * { display: none !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::before {
    content: ""; width: 16px; height: 16px; display: inline-block;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235f6456' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 16V4M7 9l5-5 5 5M4 20h16'/%3E%3C/svg%3E") no-repeat center / contain;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::after {
    content: "Upload"; font-size: .95rem; color: #5f6456; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--dark); }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover::after { color: var(--dark); }

/* slider: dark green fill comes from primaryColor in .streamlit/config.toml; ring around the thumb here */
[data-testid="stSlider"] [role="slider"] { background: var(--dark) !important;
                                           box-shadow: 0 0 0 5px var(--light) !important; border: none !important; }
[data-testid="stSliderThumbValue"] { color: var(--dark) !important; font-weight: 500; }
/* Streamlit only shows its own 0.10 / 0.90 labels on hover, so they are hidden and replaced by an always-visible row */
[data-testid="stSliderTickBar"] { display: none !important; }
.slider-range { display: flex; justify-content: space-between; font-size: .8rem; color: var(--muted); padding: 0 .15rem; margin-top: -.2rem; }
.slider-help { font-size: .95rem; color: var(--muted); margin-top: .3rem; }

[data-testid="stFormSubmitButton"] button { background: var(--dark); border: 1px solid var(--dark); border-radius: 8px;
                                            font-weight: 600; min-height: 3.3rem; }
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button p {
    color: #ffffff !important; font-size: 1.3rem;
}
[data-testid="stFormSubmitButton"] button:hover { background: var(--dark-hover); border-color: var(--dark-hover); }

/* ---------- results ---------- */
/* captions under the "Your image" / "Detected areas" pictures: big, dark and readable */
[data-testid="stImageCaption"], [data-testid="stImageCaption"] *, [data-testid="stImage"] figcaption {
    font-size: 1.5rem !important; font-weight: 600 !important; color: var(--text) !important;
    text-align: center !important; line-height: 1.3 !important; margin-top: .5rem; }
.h2 { font-weight: 600; font-size: 1.6rem; color: var(--dark); margin: 2.2rem 0 .9rem 0; }
.notice { background: #ffffff; color: var(--text); border: 1px solid var(--light);
          border-radius: 8px; padding: .9rem 1.1rem; font-size: 1rem; margin: 1.2rem 0 1rem 0; }
.stats { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1rem 0; }
.stat { flex: 1 1 170px; background: #ffffff; color: var(--text); border: 1px solid var(--light);
        border-radius: 10px; padding: .9rem 1.1rem; }
.stat .n { font-size: 2.2rem; font-weight: 700; line-height: 1; color: var(--dark); }
.stat .l { font-size: 1.05rem; font-weight: 600; margin-top: .3rem; }
.stat .c { font-size: .85rem; color: var(--muted); }
table.det { width: 100%; border-collapse: collapse; background: #ffffff; color: var(--text); font-size: 1rem; margin-bottom: .5rem; }
table.det th { text-align: left; color: var(--muted); font-size: .85rem; padding: .5rem .7rem; border-bottom: 2px solid var(--grey); }
table.det td { padding: .5rem .7rem; border-bottom: 1px solid var(--line); }

/* download buttons: PDF = solid green, CSV = white with green outline */
.dl-note { font-size: .95rem; color: var(--muted); margin: -.3rem 0 1rem 0; }
[data-testid="stDownloadButton"] button { border-radius: 8px; min-height: 3.1rem; font-weight: 600; width: 100%; }
[data-testid="stDownloadButton"] button p { font-size: 1.05rem; font-weight: 600; }
[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"],
[data-testid="stDownloadButton"] button[kind="primary"] { background: var(--dark); border: 1.5px solid var(--dark); }
[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"] *,
[data-testid="stDownloadButton"] button[kind="primary"] * { color: #ffffff !important; }
[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stDownloadButton"] button[kind="primary"]:hover { background: var(--dark-hover); border-color: var(--dark-hover); }
[data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button[kind="secondary"] { background: #ffffff; border: 1.5px solid var(--dark); }
[data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"] *,
[data-testid="stDownloadButton"] button[kind="secondary"] * { color: var(--dark) !important; }
[data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stDownloadButton"] button[kind="secondary"]:hover { background: var(--soft); }

/* ---------- disease cards (guide page + detection results) ---------- */
.dcard { background: #ffffff; border-radius: 14px; padding: 1.8rem 1.9rem 2rem 1.9rem; margin: 0 0 4.5rem 0;
         box-shadow: 0 2px 14px rgba(72,94,33,.06); color: var(--text); }
.d-top { display: flex; gap: 1.9rem; align-items: flex-start; flex-wrap: wrap; }
.d-img { flex: 0 0 190px; width: 190px; height: 190px; border-radius: 6px; overflow: hidden; background: #eef3d6;
         display: flex; align-items: center; justify-content: center; }
.d-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.d-img.ph img { width: 46%; height: auto; object-fit: contain; opacity: .45; }
.d-intro { flex: 1 1 320px; min-width: 0; }
.d-title { font-weight: 700; font-size: 1.6rem; line-height: 1.2; color: var(--dark); }
.d-cause { font-style: italic; color: var(--muted); font-size: .85rem; margin: .3rem 0 .9rem 0; }
.d-summary { font-size: 1.05rem; line-height: 1.6; color: #33362d; margin: 0 0 .7rem 0; }
.d-detail { font-size: .92rem; line-height: 1.55; color: #111111; }
.dcard ul { margin: 0; padding-left: 1.3rem; }
.dcard ul li { font-size: .95rem !important; line-height: 1.55; margin-bottom: .2rem; color: #111111; }
.dcard ul.dl li { font-size: .92rem !important; }
.d-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 0 2.4rem; margin-top: 1.6rem; }
.d-head { font-weight: 700; font-size: 1.25rem; color: var(--dark); margin: 0 0 .6rem 0; }
.d-col .d-head:not(:first-child) { margin-top: 1.5rem; }
@media (max-width: 760px) {
    .d-cols { grid-template-columns: 1fr; }
    .d-img { flex-basis: 100%; width: 100%; height: 220px; }
}

/* ---------- About page ---------- */
.page-title { font-weight: 700; font-size: 2.2rem; line-height: 1.2; color: var(--dark); text-align: center;
              margin: 2.6rem 0 2.6rem 0; }
.about { color: #111111; }
.about .a-h { font-weight: 500; font-size: 1.5rem; line-height: 1.3; color: #111111; margin: 2.4rem 0 .9rem 0; }
.about p { font-size: 1.05rem !important; line-height: 1.55; text-align: justify; margin: 0 0 .9rem 0; }
.about ul { margin: .2rem 0 1rem 0; padding-left: 1.5rem; }
.about ul li { font-size: 1.05rem !important; line-height: 1.55; text-align: justify; margin-bottom: .15rem; }
table.tbl { width: 100%; border-collapse: collapse; background: #ffffff; color: var(--text);
            font-size: 1.05rem; margin: .8rem 0 0 0; }
table.tbl th, table.tbl td { text-align: left; padding: .7rem .9rem; border-bottom: 1px solid var(--line); }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl th { font-weight: 600; color: #111111; }
table.tbl.kv td:first-child { width: 20%; }
table.tbl.metrics th:first-child, table.tbl.metrics td:first-child { width: 28%; }
table.tbl.metrics th:not(:first-child), table.tbl.metrics td:not(:first-child) { width: 18%; }
table.tbl tr.sum td { font-weight: 600; border-top: 1.5px solid var(--grey); }

/* ---------- project card ---------- */
.project { background: #ffffff; border: 1.5px solid var(--grey); border-radius: 8px;
           padding: 1.4rem 1.7rem 1.5rem 1.7rem; margin: 3.2rem 0 0 0; }
.p-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .8rem; }
.p-title { border-left: 5px solid var(--light); padding-left: .7rem; font-size: 1.1rem; line-height: 1.3;
           color: var(--muted); font-weight: 500; }
.p-title b { color: #111111; font-weight: 500; }
.p-dept { border: 1px solid #7b8070; color: #5f6456; border-radius: 999px; padding: .25rem 1rem; font-size: .9rem; }
.p-team { display: flex; flex-wrap: wrap; gap: 1.2rem 4.5rem; margin-top: 1.4rem; }
.p-member { display: flex; align-items: center; gap: .9rem; }
.p-name { font-size: 1.1rem; font-weight: 500; color: var(--text); line-height: 1.3; }
.p-role { font-size: .95rem; color: var(--muted); line-height: 1.3; }

/* ---------- footer (full-width band, content aligned with the main column) ---------- */
.site-footer { width: 100vw; position: relative; left: 50%; margin-left: -50vw; margin-top: 5rem;
               background: #485E21; border-top: none; }
.f-inner { max-width: 1090px; margin: 0 auto; padding: 1.7rem 2rem 1.5rem 2rem; box-sizing: border-box; }
.f-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
.f-brand { display: flex; align-items: center; gap: .8rem; }
.f-logo { width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; }
.f-logo img { width: 38px; height: 38px; object-fit: contain; filter: brightness(0) invert(1); }
.f-name { font-weight: 600; font-size: 1.2rem; color: #ffffff; line-height: 1.1; }
.f-tag { font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: #ffffff;
         font-weight: 500; margin-top: .25rem; }
.f-pill { display: inline-flex; align-items: center; gap: .5rem; background: #ffffff;
          border: 1px solid var(--line); border-radius: 999px; padding: .5rem 1.05rem;
          font-size: .82rem; color: var(--text); box-shadow: 0 2px 8px rgba(72,94,33,.08); }
.f-note { border-top: 1px solid rgba(255,255,255,.25); margin-top: 1.3rem; padding-top: 1rem;
          font-size: .82rem; line-height: 1.6; color: #ffffff; text-align: center; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- helpers ----------
def notice(msg):
    st.markdown(f'<div class="notice">{html.escape(msg)}</div>', unsafe_allow_html=True)


def ul(items):
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def find_image(name):
    """Find a picture in assets/ whose file name starts with the disease name,
    ignoring case, spaces, dashes and underscores (anthracnose.jpg, Brown-Blight.PNG, white_spot_1.jpeg ...)."""
    if not ASSETS_DIR.exists():
        return None
    target = _key(name)
    for f in sorted(ASSETS_DIR.iterdir()):
        if f.suffix.lower() in IMG_EXTS and _key(f.stem).startswith(target):
            return f
    return None


@st.cache_data(show_spinner=False)
def image_uri(path, mtime):
    """Shrink a picture and return it as an inline data URI (mtime only makes the cache refresh on changes)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((520, 520))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def disease_card(name):
    info = get_info(name)
    if not info:
        return (f'<div class="dcard"><div class="d-title">{html.escape(name)}</div>'
                f'<div class="d-summary">No description has been added for this disease yet.</div></div>')

    pic = find_image(info.get("image") or name)
    if pic:
        img = (f'<div class="d-img"><img src="{image_uri(str(pic), pic.stat().st_mtime)}" '
               f'alt="{html.escape(info["title"])}"></div>')
    else:   # no picture found: soft green box with the logo, so the layout stays the same
        img = f'<div class="d-img ph"><img src="{LOGO_URI}" alt=""></div>'

    details = info.get("details") or []
    if len(details) == 1:
        detail = f'<div class="d-detail">{html.escape(details[0])}</div>'
    elif details:
        detail = '<ul class="dl">' + "".join(f"<li>{html.escape(d)}</li>" for d in details) + "</ul>"
    else:
        detail = ""

    def section(title, key):
        return f'<div class="d-head">{title}</div>{ul(info[key])}' if info.get(key) else ""

    left = section("Symptoms", "symptoms") + section("How it spreads", "spread")
    right = section("Management", "management")
    return (
        f'<div class="dcard"><div class="d-top">{img}'
        f'<div class="d-intro"><div class="d-title">{html.escape(info["title"])}</div>'
        f'<div class="d-cause">{html.escape(info["cause"])}</div>'
        f'<div class="d-summary">{html.escape(info["summary"])}</div>{detail}</div></div>'
        f'<div class="d-cols"><div class="d-col">{left}</div><div class="d-col">{right}</div></div></div>'
    )


def navbar(link_label, link_target, about_active=False):
    left, mid, right = st.columns([5.6, 1.1, 1.5], vertical_alignment="center")
    active_css = ""
    if about_active:   # highlight "About" while on the About page
        active_css = ('<style>[data-testid="stColumn"]:nth-child(2) [data-testid="stPageLink"] a *, '
                      '[data-testid="column"]:nth-child(2) [data-testid="stPageLink"] a * '
                      '{ color: #111111 !important; font-weight: 600 !important; }</style>')
    left.markdown(f'<div class="brand-wrap"><a class="brand-link" href="/" target="_self"><img src="{LOGO_URI}" alt="">'
                  f'<span>{APP_NAME}</span></a></div>{active_css}', unsafe_allow_html=True)
    with mid:
        st.page_link(about_page, label="About")
    with right:
        st.page_link(link_target, label=link_label)


def hero(title, subtitle_html, chips=(), compact=False):
    chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    chips_block = f'<div class="chips">{chips_html}</div>' if chips_html else ""
    # the subtitle is a <div> (not <p>) so Streamlit's own paragraph styles can't push it off-centre
    st.markdown(
        f'<div class="hero{" compact" if compact else ""}"><div class="hero-title">{html.escape(title)}</div>'
        f'<div class="hero-sub">{subtitle_html}</div>{chips_block}</div>',
        unsafe_allow_html=True,
    )


def about_card():
    members = "".join(
        f'<div class="p-member">{ICONS.get(icon, "")}<div><div class="p-name">{html.escape(name)}</div>'
        f'<div class="p-role">{html.escape(role)}</div></div></div>'
        for icon, name, role in TEAM
    )
    st.markdown(
        '<div class="project" id="about"><div class="p-top">'
        f'<div class="p-title">Project Title: <b>{html.escape(PROJECT_TITLE)}</b></div>'
        f'<div class="p-dept">{html.escape(DEPARTMENT)}</div></div>'
        f'<div class="p-team">{members}</div></div>',
        unsafe_allow_html=True,
    )


def finish():
    about_card()
    footer()


def footer():
    year = datetime.now().year
    credit = f"<br>{html.escape(DATA_CREDIT)}" if DATA_CREDIT else ""
    st.markdown(
        '<div class="site-footer"><div class="f-inner">'
        '<div class="f-top">'
        f'<div class="f-brand"><div class="f-logo"><img src="{LOGO_URI}" alt=""></div>'
        f'<div><div class="f-name">{APP_NAME}</div><div class="f-tag">AI-powered tea plant health</div></div></div>'
        f'<div class="f-pill">{SHIELD_SVG}<span>&copy; {year} {APP_NAME}. All rights reserved.</span></div>'
        '</div>'
        '<div class="f-note">AI predictions are a guide, not a diagnosis. Confirm with an agricultural expert '
        f'before treating.{credit}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading the detection model...")
def load_model():
    return YOLO(str(MODEL_PATH))


# ---------- results: export (CSV + PDF) ----------
def make_csv(dets):
    """The detection table as CSV bytes (every detection, not only the 30 shown on screen)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["No.", "Disease", "Confidence"])
    for i, (name, c) in enumerate(dets, 1):
        w.writerow([i, name, f"{c:.2f}"])
    return buf.getvalue().encode("utf-8-sig")      # BOM so Excel opens it correctly


def make_pdf(img, annotated, dets, conf, filename, stamp):
    """A one-file report: both images, a summary, the full detection table and the disease information."""
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.platypus import (HRFlowable, Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    green, soft, grey, line = (colors.HexColor(c) for c in ("#485E21", "#F4F6EA", "#7a7f70", "#dfe2d6"))
    page_w = A4[0] - 36 * mm                      # usable width (18 mm margins)

    def t(x):                                     # text safe for the built-in PDF fonts
        return escape(str(x).encode("cp1252", "replace").decode("cp1252"))

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#1f2417"))
    small = ParagraphStyle("small", parent=base, fontSize=8.5, leading=12, textColor=grey)
    brand = ParagraphStyle("brand", parent=base, fontName="Helvetica-Bold", fontSize=15, textColor=green)
    title = ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=green)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=green,
                        spaceBefore=14, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=10.5, textColor=green,
                        spaceBefore=6, spaceAfter=2)
    cap = ParagraphStyle("cap", parent=base, fontName="Helvetica-Bold", fontSize=12, alignment=TA_CENTER, spaceBefore=4)
    bullet = ParagraphStyle("bullet", parent=base, fontSize=9.5, leading=13, leftIndent=11, firstLineIndent=-9)
    cell = ParagraphStyle("cell", parent=base, fontSize=9.5, leading=12)
    cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Helvetica-Bold")
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    def fit(pil, max_w, max_h):
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, "JPEG", quality=88)
        buf.seek(0)
        w, h = pil.size
        k = min(max_w / w, max_h / h)
        return RLImage(buf, width=w * k, height=h * k)

    def grid(rows, widths, header=True):
        tb = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.5, line)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), green)]
            style += [("BACKGROUND", (0, i), (-1, i), soft) for i in range(2, len(rows), 2)]
        tb.setStyle(TableStyle(style))
        return tb

    story = []
    # header: logo + name on the left, date on the right
    logo = RLImage(str(LOGO_PATH), 11 * mm, 11 * mm, mask="auto") if LOGO_PATH.exists() else ""
    story.append(Table([[logo, Paragraph(APP_NAME, brand),
                         Paragraph(t(stamp.strftime("%d %b %Y, %H:%M")), ParagraphStyle("r", parent=small, alignment=TA_RIGHT))]],
                       colWidths=[15 * mm, page_w - 15 * mm - 50 * mm, 50 * mm],
                       style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    story += [Spacer(1, 3 * mm), HRFlowable(width="100%", color=line, thickness=1), Spacer(1, 4 * mm)]
    story.append(Paragraph("Tea Plant Disease Detection Report", title))
    story.append(Spacer(1, 3 * mm))

    # image + settings
    meta = [[Paragraph("Image", cell_b), Paragraph(t(filename), cell)],
            [Paragraph("Confidence threshold", cell_b), Paragraph(f"{conf:.2f}", cell)],
            [Paragraph("Model", cell_b), Paragraph("YOLO11 (custom-trained)", cell)]]
    story.append(grid(meta, [42 * mm, page_w - 42 * mm], header=False))
    story.append(Spacer(1, 4 * mm))
    if dets:
        kinds = len({n for n, _ in dets})
        verdict = f"{len(dets)} affected area{'s' if len(dets) != 1 else ''} found, across {kinds} disease type{'s' if kinds != 1 else ''}."
    else:
        verdict = "No disease was detected at this confidence."
    story.append(Paragraph(f"<b>Result:</b> {t(verdict)}", base))
    story.append(Spacer(1, 4 * mm))

    # both pictures side by side
    half = (page_w - 6 * mm) / 2
    story.append(Table([[fit(img, half, 95 * mm), fit(annotated, half, 95 * mm)],
                        [Paragraph("Your image", cap), Paragraph("Detected areas", cap)]],
                       colWidths=[half + 3 * mm, half + 3 * mm],
                       style=[("VALIGN", (0, 0), (-1, 0), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))

    if dets:
        counts = Counter(n for n, _ in dets)
        best = {}
        for n, c in dets:
            best[n] = max(best.get(n, 0.0), c)
        order = [n for n, _ in counts.most_common()]

        story.append(Paragraph("Summary", h2))
        rows = [[Paragraph("Disease", head), Paragraph("Areas found", head), Paragraph("Highest confidence", head)]]
        rows += [[Paragraph(t(n), cell), Paragraph(str(counts[n]), cell), Paragraph(f"{best[n]:.2f}", cell)] for n in order]
        story.append(grid(rows, [page_w * 0.5, page_w * 0.2, page_w * 0.3]))

        story.append(Paragraph("All detections", h2))
        rows = [[Paragraph("No.", head), Paragraph("Disease", head), Paragraph("Confidence", head)]]
        rows += [[Paragraph(str(i), cell), Paragraph(t(n), cell), Paragraph(f"{c:.2f}", cell)]
                 for i, (n, c) in enumerate(dets, 1)]
        story.append(grid(rows, [page_w * 0.12, page_w * 0.58, page_w * 0.30]))

        story.append(Paragraph("Disease information", h2))
        for n in order:
            info = get_info(n)
            if not info:
                continue
            block = [Paragraph(t(info["title"]), ParagraphStyle("dt", parent=h3, fontSize=12.5, spaceBefore=8)),
                     Paragraph(f"<i>{t(info['cause'])}</i>", small), Spacer(1, 1.5 * mm),
                     Paragraph(t(info["summary"]), base)]
            for head_txt, key in (("Symptoms", "symptoms"), ("How it spreads", "spread"), ("Management", "management")):
                if info.get(key):
                    block.append(Paragraph(head_txt, h3))
                    block += [Paragraph("&bull; " + t(x), bullet) for x in info[key]]
            story.append(KeepTogether(block))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(grey)
        canvas.drawString(18 * mm, 10 * mm, "AI predictions are a guide, not a diagnosis. Confirm with an agricultural expert before treating.")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                      bottomMargin=18 * mm, title="Tea Plant Disease Detection Report",
                      author=APP_NAME).build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out.getvalue()


def build_result(img, annotated, dets, conf, filename):
    """Everything about one analysis, kept in session_state so the page (and its download buttons) survive reruns."""
    stamp = datetime.now()
    try:
        pdf = make_pdf(img, annotated, dets, conf, filename, stamp)
    except Exception:          # e.g. reportlab not installed: the rest of the app keeps working
        pdf = None
    return {"img": img, "annotated": annotated, "dets": dets, "conf": conf, "stamp": stamp,
            "pdf": pdf, "csv": make_csv(dets)}


def show_results(r):
    st.markdown('<div class="h2">Result</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.image(r["img"], caption="Your image", width="stretch")
    col_b.image(r["annotated"], caption="Detected areas", width="stretch")

    dets = r["dets"]
    if not dets:
        notice("No disease was detected at this confidence. Try a closer, well-lit photo, "
               "or lower the confidence slider.")
    else:
        counts = Counter(n for n, _ in dets)
        best = {}
        for n, c in dets:
            best[n] = max(best.get(n, 0.0), c)
        order = [n for n, _ in counts.most_common()]

        stats = "".join(
            f'<div class="stat"><div class="n">{counts[n]}</div><div class="l">{html.escape(n)}</div>'
            f'<div class="c">highest confidence {best[n]:.2f}</div></div>'
            for n in order
        )
        st.markdown(f'<div class="stats">{stats}</div>', unsafe_allow_html=True)

        rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{c:.2f}</td></tr>" for n, c in dets[:30])
        st.markdown(f'<table class="det"><tr><th>Disease</th><th>Confidence</th></tr>{rows}</table>',
                    unsafe_allow_html=True)
        if len(dets) > 30:
            st.markdown(f'<div class="dl-note">Showing the 30 most confident of {len(dets)} detections. '
                        'The downloads contain all of them.</div>', unsafe_allow_html=True)

    # ----- download results -----
    stem = "tealeaf_results_" + r["stamp"].strftime("%Y%m%d_%H%M")
    st.markdown('<div class="h2">Download results</div>', unsafe_allow_html=True)
    st.markdown('<div class="dl-note">Save this analysis as a PDF report (with both images) '
                'or as a CSV file of the detection table.</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        if r["pdf"]:
            st.download_button("Download PDF report", data=r["pdf"], file_name=stem + ".pdf",
                               mime="application/pdf", icon=":material/picture_as_pdf:", type="primary",
                               key="dl_pdf", width="stretch")
        else:
            notice("The PDF report is not available on this server.")
    with d2:
        st.download_button("Download CSV", data=r["csv"], file_name=stem + ".csv", mime="text/csv",
                           icon=":material/table_view:", key="dl_csv", disabled=not dets, width="stretch")

    if dets:
        st.markdown('<div class="h2">Disease information</div>', unsafe_allow_html=True)
        for n in [n for n, _ in Counter(n for n, _ in dets).most_common()]:
            st.markdown(disease_card(n), unsafe_allow_html=True)


# ---------- page 1: detector ----------
def home():
    navbar("Disease Guide", guide_page)

    try:
        model = load_model()
        detectable = sorted(model.names.values())
    except Exception:
        model, detectable = None, sorted(GUIDE_ORDER)

    source_link = (f' View the source code on <a href="{html.escape(GITHUB_URL)}">GitHub</a>.'
                   if GITHUB_URL else "")
    hero(APP_TITLE,
         "Upload a tea leaf image and receive instant disease detection powered by a custom-trained "
         f"YOLO11 model.{source_link}",
         detectable)

    if model is None:
        notice("The detection model could not be loaded. Please try again later.")
        finish()
        return

    with st.form("analyse"):
        st.markdown('<div class="field-label">Input Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown('<div class="upload-help">Please upload a JPG or PNG file up to 200 MB.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="field-label gap">Detection confidence</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1.15], vertical_alignment="center")
        with c1:
            conf = st.slider("Detection confidence", CONF_MIN, CONF_MAX, 0.35, 0.05, label_visibility="collapsed")
            st.markdown(f'<div class="slider-range"><span>{CONF_MIN:.2f}</span><span>{CONF_MAX:.2f}</span></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="slider-help">Lower = more detections (may include noise) · '
                        'Higher = only high-confidence boxes</div>', unsafe_allow_html=True)
        with c2:
            submitted = st.form_submit_button("Analyze", width="stretch")

    if submitted:
        if uploaded is None:
            st.session_state.pop("result", None)
            notice("Please upload a leaf image first.")
        else:
            try:
                img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")   # fixes rotated phone photos
                img.thumbnail((1280, 1280))                                          # keeps CPU time low
            except Exception:
                st.session_state.pop("result", None)
                notice("That file could not be read as an image. Please upload a JPG or PNG.")
                finish()
                return

            with st.spinner("Analysing the leaf..."):
                res = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
                annotated = Image.fromarray(res.plot()[:, :, ::-1])                  # BGR to RGB
                dets = sorted(((model.names[int(b.cls)], float(b.conf)) for b in res.boxes), key=lambda d: -d[1])
                st.session_state["result"] = build_result(img, annotated, dets, conf, uploaded.name)

    if st.session_state.get("result"):
        show_results(st.session_state["result"])

    finish()


# ---------- page 2: disease guide ----------
def guide():
    navbar("Back to detector", home_page)
    hero("Disease Guide",
         "How to recognize the tea plant diseases the detector looks for, "
         "and how each one is usually managed.",
         compact=True)
    for name in GUIDE_ORDER:
        st.markdown(disease_card(name), unsafe_allow_html=True)
    footer()


# ---------- page 3: about ----------
def kv_table(rows):
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    return f'<table class="tbl kv">{body}</table>'


def metrics_table():
    head = "".join(f"<th>{h}</th>" for h in ("Disease", "Precision", "Recall", "mAP50", "mAP50-95"))

    def row(r, cls=""):
        cells = f"<td>{html.escape(r[0])}</td>" + "".join(f"<td>{v:.3f}</td>" for v in r[1:])
        return f'<tr class="{cls}">{cells}</tr>'

    rows = "".join(row(r) for r in METRICS) + row(METRICS_ALL, "sum")
    return f'<table class="tbl metrics"><tr>{head}</tr>{rows}</table>'


def about():
    navbar("Disease Guide", guide_page, about_active=True)
    st.markdown(
        '<div class="page-title">About this tool</div><div class="about">'
        f'<p>{html.escape(APP_TITLE)} is a free web tool that looks at a photo of a tea leaf and marks areas that '
        'resemble one of four common tea leaf diseases. It was developed as a student research project in the '
        f'{html.escape(DEPARTMENT)}. It is a quick screening aid. It is not a replacement for an agronomist, a '
        'plant pathologist or a lab test.</p>'
        '<div class="a-h">How it works</div>'
        + ul([
            "Upload a photo (JPG or PNG) of a tea leaf. The app corrects phone-camera rotation and resizes large images.",
            "The model scans the image. A YOLO11 object-detection model draws a box around each area that looks like "
            "a trained disease and gives it a label and a confidence score between 0 and 1.",
            "You see the results: the annotated image, how many areas were found per disease, a table of detections, "
            "and general information on each disease detected.",
        ])
        + '<p>The confidence slider sets the minimum score a detection needs to be shown. Lower values find more '
        'affected areas but also produce more false alarms. Higher values show only the detections the model is '
        'most sure of, but may miss real lesions. The default is 0.35, a balanced starting point.</p>'
        '<div class="a-h">About the model</div>'
        + kv_table(MODEL_ROWS)
        + '<div class="a-h">How well does it work?</div>'
        '<p>The model was evaluated on test images it never saw during training. These are the standard measures:</p>'
        + ul([
            "Precision: of the areas the model flagged, the share that were correct. Low precision means false alarms.",
            "Recall: of the real diseased areas, the share the model found. Low recall means missed disease.",
            "mAP50: an overall accuracy score that counts a detection as correct if its box overlaps the true box "
            "by at least 50%.",
            "mAP50-95: a stricter version that also requires tight, precise boxes. It is lower than mAP50.",
        ])
        + metrics_table()
        + '</div>',
        unsafe_allow_html=True,
    )
    footer()


# ---------- navigation ----------
home_page = st.Page(home, title=APP_TITLE, default=True)
guide_page = st.Page(guide, title="Disease Guide", url_path="disease-guide")
about_page = st.Page(about, title="About", url_path="about")
st.navigation([home_page, guide_page, about_page], position="hidden").run()