from __future__ import annotations

import io
import os
from typing import Optional

import streamlit as st
from PIL import Image

from src.vision_engine import VisionEngine
from src.energy_math import estimate_energy
from src.weather_api import fetch_daily_rain

MODELS_DIR = os.path.join("models")
DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "best.pt")
DATA_DIR = os.path.join("Trash Detection.v1i.yolov8")


def _ensure_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)


def sidebar_controls() -> dict:
    st.sidebar.header("Model & Inference Settings")

    weights_path = st.sidebar.text_input(
        "Weights path",
        value=DEFAULT_WEIGHTS,
        help="Path to YOLOv8 .pt weights file.",
    )
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.01)
    iou = st.sidebar.slider("IOU threshold", 0.1, 0.9, 0.45, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.header("Optional Weather Input")
    use_weather = st.sidebar.checkbox("Factor daily rainfall (Open-Meteo)", value=False)
    lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")

    return {
        "weights_path": weights_path,
        "conf": float(conf),
        "iou": float(iou),
        "use_weather": bool(use_weather),
        "lat": float(lat),
        "lon": float(lon),
    }


def render_topbar():
    st.title("WastePredictor: Trash Detection + Energy Potential")
    st.caption("YOLOv8-powered waste detection with Biogas/RDF estimates")


def run_detection_ui(params: dict):
    st.header("1) Upload an Image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        st.header("2) Run Detection")
        engine = VisionEngine(weights_path=params["weights_path"])  # lazy load
        try:
            output = engine.predict(image, conf=params["conf"], iou=params["iou"])  # loads model first call
        except FileNotFoundError:
            st.error(f"Weights not found at {params['weights_path']}. Train a model or update the path.")
            return
        except Exception as e:
            st.error(f"Detection error: {e}")
            return

        st.subheader("Detections")
        det_count = sum(output["by_class"].values())
        st.write(f"Objects detected: {det_count}")
        st.json(output["by_class"]) if output["by_class"] else st.info("No detections above threshold.")

        st.image(output["plotted"], caption="Detections", use_column_width=True)

        st.header("3) Energy Estimates")
        energy = estimate_energy(output["by_class"]) if output["by_class"] else {"rdf_kwh": 0.0, "biogas_m3": 0.0, "biogas_mj": 0.0}
        col1, col2, col3 = st.columns(3)
        col1.metric("RDF potential (kWh)", f"{energy['rdf_kwh']:.2f}")
        col2.metric("Biogas potential (m³)", f"{energy['biogas_m3']:.2f}")
        col3.metric("Biogas energy (MJ)", f"{energy['biogas_mj']:.2f}")

        if params["use_weather"]:
            st.subheader("Weather Adjustment (optional)")
            rain = fetch_daily_rain(params["lat"], params["lon"])  # may return None
            if rain is None:
                st.warning("Could not fetch rain data. Check internet connection or try later.")
            else:
                st.write(f"Daily rainfall (mm): {rain.daily_rain_mm:.2f}")
                # Example: damp conditions reduce RDF effectiveness by 5% per 10mm rain up to 30%
                reduction = min(0.30, 0.05 * (rain.daily_rain_mm / 10.0))
                adj_rdf = energy["rdf_kwh"] * (1 - reduction)
                st.write(f"Adjusted RDF potential (kWh): {adj_rdf:.2f}  (reduction {reduction*100:.0f}%)")


def main():
    _ensure_dirs()
    params = sidebar_controls()
    render_topbar()
    run_detection_ui(params)


if __name__ == "__main__":
    st.set_page_config(page_title="WastePredictor", page_icon="♻️", layout="wide")
    main()
