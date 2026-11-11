# TeaLeaf-AI: Tea Plant Disease Detection Tool

A web app that looks at a photo of a tea leaf and marks areas that resemble one of four common tea leaf diseases. It uses a custom-trained **YOLO11** object-detection model and is built with **Streamlit**.

> **Live demo:** <https://tealeaf-disease-detection.streamlit.app/> 

This is a student research project from the Department of Information Technology. It is a quick screening aid, **not** a replacement for an agronomist, a plant pathologist or a lab test.

---

## Features

- **Disease detection.** Upload a JPG or PNG leaf photo and get boxes, labels and confidence scores. 
- **Adjustable confidence.** A slider (0.10 to 0.90) sets the minimum score a detection needs to be shown.
- **Sample images.** Try the app without a photo: pick any image from the `sample_images/` folder.
- **Results at a glance.** Annotated image, count and highest confidence per disease, and a table of all detections.
- **Download results.** Save the analysis as a **PDF report** (both images, summary, full detection table and disease information) or as a **CSV** of the detection table.
- **Disease Guide.** A page with a photo, symptoms, how each disease spreads and how it is usually managed.
- **About page.** How the tool works, model details and evaluation metrics.

## Detectable diseases

| Disease | Cause |
|---|---|
| Anthracnose | Fungi of the genus *Colletotrichum* |
| Brown Blight | *Colletotrichum camelliae* (also given as *Glomerella cingulata*) |
| White Spot (White Scab) | *Elsinoe leucospila* |
| Algal Leaf Spot | Alga *Cephaleuros virescens* |

## Model

| | |
|---|---|
| Model | YOLO11 (small) |
| Task | Object detection |
| Classes | 4 |
| Input size | 640 × 640 pixels |
| Dataset | 1481 photographs of tea leaves |
| Data split | 80% training / 20% test |

## Acknowledgements

- Detection model: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- App framework: [Streamlit](https://streamlit.io)
- PDF reports: [ReportLab](https://www.reportlab.com)
- Dataset: kaggle
