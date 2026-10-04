# Portfolio demos — Fatima Zahrae Ahannuk

Live, testable demos of my projects, deployed for free on Streamlit Community Cloud.
Each folder is a standalone Streamlit app (`streamlit_app.py` + `requirements.txt`).

| Demo | What you can test | Project code |
|---|---|---|
| `autoplus/` | Used-car price estimation for Morocco (XGBoost, R² 0.94) with per-feature explanation | [AutoPlus-Maroc](https://github.com/fahan860/AutoPlus-Maroc) |
| `deepfake/` | Fake vs real face detection (DenseNet-121, AlexNet, custom CNN) | [DL1](https://github.com/merouane01/DL1) |
| `fakenews/` | Fake news headline classifier (TF-IDF + Logistic Regression) with word-level explanation | [fake_news_detection](https://github.com/fahan860/fake_news_detection) |
| `diabetes/` | Diabetes risk screening, 4 models compared, recall-first | [diabetes-prediction-ml](https://github.com/fahan860/diabetes-prediction-ml) |
| `smartlearn/` | Course recommender (sentence-transformers embeddings, cold-start fallback) | [smartlearn-platform](https://github.com/fahan860/smartlearn-platform) |
| `excel/` | Excel sales automation: upload files, download the generated report | [excel-automation-project](https://github.com/fahan860/excel-automation-project) |

Models are either downloaded from the project repositories at startup or retrained in a few seconds,
so each demo runs the same logic as the original project.

Run one locally: `pip install -r autoplus/requirements.txt streamlit && streamlit run autoplus/streamlit_app.py`
