# Portfolio demos — Fatima Zahrae Ahannuk

Live, testable demos of my projects, deployed for free on Streamlit Community Cloud.
🌐 Portfolio with the full case studies: **https://fatima-zahrae-ahannuk.vercel.app**
Each folder is a standalone Streamlit app (`streamlit_app.py` + `requirements.txt`).

| Demo | Live app | What you can test | Project code |
|---|---|---|---|
| `autoplus/` | [▶ fahan-autoplus.streamlit.app](https://fahan-autoplus.streamlit.app) | Used-car price estimation for Morocco (XGBoost, R² 0.94) with per-feature explanation | [AutoPlus-Maroc](https://github.com/fahan860/AutoPlus-Maroc) |
| `deepfake/` | [▶ fahan-face-detection.streamlit.app](https://fahan-face-detection.streamlit.app) | Fake vs real face detection (DenseNet-121 and AlexNet) | [DL1](https://github.com/merouane01/DL1) |
| `fakenews/` | [▶ fahan-fakenews.streamlit.app](https://fahan-fakenews.streamlit.app) | Fake news headline classifier (TF-IDF + Logistic Regression) with word-level explanation | [fake_news_detection](https://github.com/fahan860/fake_news_detection) |
| `diabetes/` | [▶ fahan-diabetes.streamlit.app](https://fahan-diabetes.streamlit.app) | Diabetes risk screening, 4 models compared, recall-first | [diabetes-prediction-ml](https://github.com/fahan860/diabetes-prediction-ml) |
| `smartlearn/` | [▶ fahan-smartlearn.streamlit.app](https://fahan-smartlearn.streamlit.app) | Course recommender (sentence-transformers embeddings, cold-start fallback) | [smartlearn-platform](https://github.com/fahan860/smartlearn-platform) |
| `excel/` | [▶ fahan-excel.streamlit.app](https://fahan-excel.streamlit.app) | Excel sales automation: upload files, download the generated report | [excel-automation-project](https://github.com/fahan860/excel-automation-project) |

Models are either downloaded from the project repositories at startup or retrained in a few seconds,
so each demo runs the same logic as the original project.

Run one locally: `pip install -r autoplus/requirements.txt streamlit && streamlit run autoplus/streamlit_app.py`
