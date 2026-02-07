# AI-004 WastePredictor

End-to-end waste detection and energy potential estimator using YOLOv8 and Streamlit.

## Project Structure

```
AI-004-WastePredictor/
├── Trash Detection.v1i.yolov8/        # Your downloaded dataset (provide locally)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── models/                            # Trained weights will be saved here
│   └── (best.pt)                      # Created after training or copy your own
├── src/
│   ├── __init__.py
│   ├── vision_engine.py               # Detection logic
│   ├── energy_math.py                 # Biogas/RDF calculations
│   └── weather_api.py                 # Optional weather data helper
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Setup

1) Python 3.9–3.11 recommended.

2) Create a virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
```

3) Ensure dataset is placed at:

```
AI-004-WastePredictor/Trash Detection.v1i.yolov8/
```
with a data.yaml describing your splits.

4) (Optional) If you already have trained weights, place them at:

```
AI-004-WastePredictor/models/best.pt
```

## Training

Train YOLOv8n on your dataset:

```
python -m ultralytics cfg                                              # optional: verify install
yolo task=detect mode=train model=yolov8n.pt data="Trash Detection.v1i.yolov8/data.yaml" imgsz=640 epochs=50 project=models name=run1
```

Artifacts appear under `models/run1/` with `weights/best.pt`. Copy or symlink to `models/best.pt`:

```
cp models/run1/weights/best.pt models/best.pt
```

## Running the App

```
streamlit run app.py
```

The app lets you:
- Upload an image and run trash detection
- See detected classes and counts
- Estimate energy potential (Biogas/RDF) from detected categories
- (Optional) Factor local rainfall from Open-Meteo

## Environment Variables (optional)

No keys are required for the default weather source (Open-Meteo). If you switch providers, add your variables to a `.env` and load via `python-dotenv`.

## Notes
- If GPU is available (CUDA), ultralytics will use it automatically. Otherwise inference runs on CPU.
- Adjust energy factors in `src/energy_math.py` per your local baseline.
