"""Fake news headline classifier — Fatima Zahrae Ahannuk.
TF-IDF (1-2 grams, 20k features) + Logistic Regression trained at startup on FakeNewsNet
headlines (PolitiFact + GossipCop), so the demo never depends on a pickled model."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
RAW = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/{}_{}.csv"


@st.cache_resource
def train():
    parts = []
    for site in ("politifact", "gossipcop"):
        for label, y in (("fake", 1), ("real", 0)):
            parts.append(pd.read_csv(RAW.format(site, label), usecols=["title"]).dropna().assign(y=y))
    data = pd.concat(parts).drop_duplicates("title")
    Xtr, Xte, ytr, yte = train_test_split(data["title"], data["y"], test_size=0.2, random_state=42, stratify=data["y"])
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
    model = LogisticRegression(class_weight="balanced", max_iter=200).fit(vec.fit_transform(Xtr), ytr)
    pred = model.predict(vec.transform(Xte))
    return vec, model, accuracy_score(yte, pred), f1_score(yte, pred), len(yte)


with st.spinner("Training the model on ~22k headlines (first visit only)…"):
    vec, model, acc, f1, n = train()
vocab, coef = np.array(vec.get_feature_names_out()), model.coef_[0]

st.title("📰 Fake News Headline Detector")
st.markdown(
    "NLP classifier trained on **FakeNewsNet** headlines (PolitiFact + GossipCop): **TF-IDF** (1–2 grams) + "
    f"**Logistic Regression** — test accuracy **{acc:.1%}**, F1 (fake) {f1:.2f} on {n:,} held-out headlines.  \n"
    "⚠️ *Educational demo: headline-only model, best on US politics / celebrity news.* · "
    "[Code](https://github.com/fahan860/fake_news_detection) · by **Fatima Zahrae Ahannuk**"
)
examples = ["Brad Pitt and Jennifer Aniston secretly remarried in Mexico, insider claims",
            "Senate passes bipartisan infrastructure bill after months of negotiations",
            "Pope Francis endorses Donald Trump for president",
            "Taylor Swift announces new album release date at Grammys"]
choice = st.selectbox("Try an example…", ["(write your own)"] + examples)
text = st.text_area("News headline (English)", "" if choice.startswith("(") else choice, height=90)
if text.strip():
    x = vec.transform([text])
    p_fake = float(model.predict_proba(x)[0][1])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Prediction", "FAKE" if p_fake >= 0.5 else "REAL", f"{max(p_fake, 1 - p_fake):.0%} confidence", delta_color="off")
        st.progress(p_fake, text=f"P(fake) = {p_fake:.0%}")
    with c2:
        idx = x.nonzero()[1]
        contrib = sorted(((vocab[i], x[0, i] * coef[i]) for i in idx), key=lambda t: abs(t[1]), reverse=True)[:8]
        if contrib:
            st.markdown("**Words that drove the decision** (tf-idf × model weight)")
            st.dataframe(pd.DataFrame({"term": [w for w, _ in contrib],
                                       "pushes towards": ["🔴 fake" if c > 0 else "🟢 real" for _, c in contrib],
                                       "weight": [round(abs(c), 3) for _, c in contrib]}), hide_index=True, width="stretch")
        else:
            st.info("No known vocabulary in this text.")
