# TeaLeaf-AI: Tea Plant Disease Detection Tool

A web app that looks at a photo of a tea leaf and marks areas that resemble one of four common tea leaf diseases. It uses a custom-trained **YOLO11** object-detection model and is built with **Streamlit**.

> **Live demo:** <https://tealeaf-detection-app-acqjucvhnpbd3qmswcik2h.streamlit.app> *(replace with your current app URL)*

This is a student research project from the Department of Information Technology. It is a quick screening aid, **not** a replacement for an agronomist, a plant pathologist or a lab test.

---

## Features

- **Disease detection.** Upload a JPG or PNG leaf photo and get boxes, labels and confidence scores. Phone-camera rotation is corrected and large images are resized automatically.
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

Evaluation on the held-out test images:

| Disease | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Anthracnose | 0.729 | 0.851 | 0.865 | 0.520 |
| Brown Blight | 0.594 | 0.571 | 0.575 | 0.287 |
| White Spot | 0.984 | 1.000 | 0.995 | 0.675 |
| Algal Leaf Spot | 0.565 | 0.741 | 0.674 | 0.533 |
| **All classes** | **0.718** | **0.791** | **0.777** | **0.504** |

Brown Blight is the weakest class, so treat its results with extra care.

## Project structure

```
tealeaf-detection-app/
├── app.py                  # the Streamlit app (all pages and styling)
├── disease_info.py         # text shown in the Disease Guide and under the results
├── requirements.txt        # Python packages
├── packages.txt            # system packages for Streamlit Cloud
├── .streamlit/
│   └── config.toml         # theme colour
├── models/
│   └── yolo11_S4_best.pt   # trained model weights
├── assets/
│   ├── logo.png            # logo (navbar, footer, browser tab, PDF)
│   └── anthracnose.jpg, brown_blight.jpg, white_spot.jpg, algal_leaf_spot.jpg
└── sample_images/          # example leaf photos for the "try a sample" buttons
```

## Run it locally

You need Python 3.10 or newer.

```bash
git clone <your-repo-url>
cd tealeaf-detection-app

python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Deploy on Streamlit Community Cloud

1. Push the whole project to a GitHub repository. Make sure the model file, the `assets/` folder and the `sample_images/` folder are committed.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app** and choose your repository, branch `main` and main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**.
4. Deploy. Streamlit installs `packages.txt` and `requirements.txt` for you.

To change the app's web address, open the app's **Settings** in your Streamlit workspace and edit the **App URL**.

## Customising

Most things you may want to edit are plain settings near the top of `app.py`:

| Setting | What it changes |
|---|---|
| `APP_NAME`, `APP_TITLE` | Name in the navbar and footer, and the main title |
| `PROJECT_TITLE`, `DEPARTMENT`, `TEAM` | The project card at the bottom of the home page |
| `MODEL_ROWS`, `METRICS`, `DATA_SPLIT` | The tables on the About page |
| `GITHUB_URL` | Adds a "View the source code on GitHub" link to the home page |
| `DATA_CREDIT` | Adds a dataset credit line to the footer |
| `CONF_MIN`, `CONF_MAX` | Range of the confidence slider |

- **Disease text** lives in `disease_info.py`.
- **Disease pictures:** put one picture per disease in `assets/`, named `anthracnose`, `brown_blight`, `white_spot` and `algal_leaf_spot` (.jpg, .jpeg, .png or .webp).
- **Sample images:** drop pictures into `sample_images/`. File names become the button labels, so `brown_blight_1.jpg` shows as "Brown Blight 1".

## Limitations

- The model only knows four diseases and has no "healthy leaf" class, so a healthy leaf, or a photo of something else, can still produce false detections.
- Blurry, dark or distant photos give weaker results. Use a close, well-lit photo of a single leaf where possible.
- Lower confidence values find more affected areas but also raise false alarms. Higher values miss real lesions.
- AI predictions are a guide, not a diagnosis. Confirm with an agricultural expert before treating, and use only products and doses recommended by your local tea research institute.

## Team

- **Prof. Jaidhar C. D.**, Principal Investigator
- **Shrinidhi Angadi**, Student Research Intern

Department of Information Technology

## Acknowledgements

- Detection model: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- App framework: [Streamlit](https://streamlit.io)
- PDF reports: [ReportLab](https://www.reportlab.com)
- Dataset: *add your dataset source and credit here*

## License

*Add your licence here (for example MIT), or remove this section.*
