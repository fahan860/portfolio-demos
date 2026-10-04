"""Diabetes risk screening demo — Fatima Zahrae Ahannuk.
Models are (re)trained at startup on the Pima Indians dataset (768 rows, < 2 s),
so the demo never depends on a pickled model or a library version."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="wide")
DATA = "https://raw.githubusercontent.com/fahan860/diabetes-prediction-ml/main/data/raw/diabetes.csv"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class Clean(BaseEstimator, TransformerMixin):
    """0 = missing value in the Pima dataset -> training median; log1p on Insulin."""
    def fit(self, X, y=None):
        self.medians_ = X[ZERO_INVALID].replace(0, np.nan).median()
        return self

    def transform(self, X):
        X = X.copy()
        X[ZERO_INVALID] = X[ZERO_INVALID].replace(0, np.nan).fillna(self.medians_)
        X["Insulin"] = np.log1p(X["Insulin"])
        return X


@st.cache_resource
def train():
    df = pd.read_csv(DATA)
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["Outcome"], test_size=0.2, random_state=2)
    models = {
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", random_state=42, probability=True),
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    pipes, rows = {}, []
    for name, m in models.items():
        p = Pipeline([("clean", Clean()), ("scale", StandardScaler()), ("model", m)]).fit(Xtr, ytr)
        pred = p.predict(Xte)
        pipes[name] = p
        rows.append([name, accuracy_score(yte, pred), precision_score(yte, pred), recall_score(yte, pred), f1_score(yte, pred)])
    board = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]).sort_values("F1", ascending=False)
    return pipes, board


pipes, board = train()
best = board.iloc[0]["Model"]

st.title("🩺 Diabetes Risk Screening")
st.markdown(
    "Binary classification on the **Pima Indians Diabetes** dataset (768 patients). 4 models compared; preprocessing "
    "(missing-value imputation, log-insulin, scaling) lives inside a scikit-learn `Pipeline`, so training and inference "
    "use exactly the same transforms.  \n⚠️ *Educational demo — not a medical tool.* · "
    "[Code](https://github.com/fahan860/diabetes-prediction-ml) · by **Fatima Zahrae Ahannuk**"
)
left, right = st.columns(2)
with left:
    model = st.selectbox("Model", list(pipes), index=list(pipes).index(best))
    vals = [
        st.slider("Pregnancies", 0, 17, 2),
        st.slider("Glucose (mg/dL)", 40, 200, 120),
        st.slider("Blood pressure (mm Hg)", 30, 122, 72),
        st.slider("Skin thickness (mm)", 0, 99, 29),
        st.slider("Insulin (µU/mL)", 0, 846, 125),
        st.slider("BMI", 15.0, 67.0, 32.0, 0.1),
        st.slider("Diabetes pedigree function", 0.05, 2.5, 0.47, 0.01),
        st.slider("Age", 21, 81, 33),
    ]
with right:
    proba = float(pipes[model].predict_proba(pd.DataFrame([vals], columns=FEATURES))[0][1])
    st.metric("Estimated diabetes risk", f"{proba:.0%}", "high" if proba >= 0.5 else "low", delta_color="inverse" if proba >= 0.5 else "normal")
    st.progress(proba)
    st.subheader(f"Test-set leaderboard (best F1: {best})")
    show = board.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1"]:
        show[c] = (show[c] * 100).round(1).astype(str) + " %"
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Recall is prioritised for screening: missing a diabetic patient costs more than a false alarm.")
