"""AUTO+ Maroc — used-car price estimator (Model A) — Fatima Zahrae Ahannuk.

The real model of the AUTO+ project (XGBoost on log(price), trained on 59,812 Moroccan
listings, test R² 0.94) is downloaded from the public GitHub repository, so this app runs
exactly the artefact produced by the training pipeline."""
import gzip
import json
import urllib.request
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import xgboost

st.set_page_config(page_title="AUTO+ — car price estimator", page_icon="🚗", layout="wide")
RAW = "https://raw.githubusercontent.com/fahan860/AutoPlus-Maroc/main/ml/models/model_a/"
CATEGORICAL = ["marque", "modele", "boite", "carburant", "etat", "origine", "premiere_main", "ville"]
EQUIP = {
    "ABS": "abs", "Airbags": "airbags", "Bluetooth": "bluetooth", "Rear camera": "camera_recul",
    "Air conditioning": "climatisation", "ESP": "esp", "GPS": "gps", "Alloy wheels": "jantes_alu",
    "Speed limiter": "limiteur_vitesse", "On-board computer": "ordinateur_bord", "Parking sensors": "radar_recul",
    "Cruise control": "regulateur_vitesse", "Leather seats": "sieges_cuir", "Sunroof": "toit_ouvrant",
    "Central locking": "verrouillage_central", "Power windows": "vitres_electriques",
}
FUEL = {"Diesel": "diesel", "Petrol": "essence", "Hybrid": "hybride", "Electric": "electrique", "LPG": "gpl"}
GEAR = {"Manual": "manuelle", "Automatic": "automatique"}
COND = {"Very good": "tres_bon", "Excellent": "excellent", "Good": "bon", "Fair": "correct", "New": "neuf", "Unknown": "inconnu"}
ORIG = {"Bought new in Morocco (WW)": "ww_maroc", "Cleared through customs": "dedouanee", "Imported new": "importee_neuve",
        "Not cleared through customs": "non_dedouanee", "Unknown": "inconnu"}
FIRST = {"Unknown": "inconnu", "Yes": "oui", "No": "non"}
NAMES = {"marque": "Make", "modele": "Model", "age": "Age", "kilometrage": "Mileage", "boite": "Gearbox",
         "carburant": "Fuel", "puissance_fiscale": "Fiscal HP", "etat": "Condition", "origine": "Origin",
         "premiere_main": "First owner", "ville": "City", "nb_portes": "Doors", "nb_equipements": "Nb of equipment",
         "km_inconnu": "Mileage unknown"}


def fetch(name):
    with urllib.request.urlopen(RAW + name, timeout=120) as r:
        return r.read()


@st.cache_resource
def load():
    booster = xgboost.Booster()
    booster.load_model(bytearray(gzip.decompress(fetch("xgboost.ubj.gz"))))
    return booster, json.loads(fetch("preparateur.json")), json.loads(fetch("fiche_modele.json"))


with st.spinner("Loading the XGBoost model (first visit only)…"):
    booster, prep, card = load()
VARIABLES, CATS = prep["variables"], prep["categories"]
TEST, TRANCHES = card["mesures_test"], card["fourchettes"]["tranches"]
MODELS_BY_MAKE = {}
for v in CATS["modele"]:
    mk, md = v.split(" | ", 1)
    MODELS_BY_MAKE.setdefault(mk, []).append(md)


def transform(row):
    X = pd.DataFrame([row])[VARIABLES].copy()
    for col in CATEGORICAL:
        known = CATS[col]
        X[col] = pd.Categorical(X[col].where(X[col].isin(known), "autre"), categories=known + ["autre"])
    for col in [c for c in VARIABLES if c.startswith("eq_")] + ["km_inconnu"]:
        X[col] = X[col].astype(int)
    return X


def mad(x):
    return f"{int(round(x, -3)):,}".replace(",", " ") + " MAD"


st.title("🚗 AUTO+ Maroc — used-car price estimator")
st.markdown(
    "Model A of **AUTO+**, a Data & AI platform for car services in Morocco (year-end engineering project, team of two, "
    "ENSA Tétouan). XGBoost on log(price), 30 features, trained on 59,812 cleaned 2024 listings (MUCars-2024, CC BY 4.0). "
    f"Test set: **R² {TEST['R2']:.3f}**, MAE {mad(TEST['MAE_DH'])}, median error {mad(TEST['erreur_mediane_DH'])}.  \n"
    "[Code on GitHub](https://github.com/fahan860/AutoPlus-Maroc) · Author: **Fatima Zahrae Ahannuk**"
)
left, right = st.columns([1, 1])
with left:
    makes = sorted(MODELS_BY_MAKE)
    c1, c2 = st.columns(2)
    make = c1.selectbox("Make", makes, index=makes.index("dacia"))
    models = sorted(MODELS_BY_MAKE[make])
    model = c2.selectbox("Model", models, index=models.index("logan") if "logan" in models else 0)
    year = st.slider("Year", 1995, date.today().year, 2019)
    c1, c2 = st.columns([2, 1])
    km_unknown = c2.checkbox("Mileage unknown")
    km = c1.number_input("Mileage (km)", 0, 1_000_000, 90_000, 5_000, disabled=km_unknown)
    c1, c2, c3 = st.columns(3)
    gearbox = c1.radio("Gearbox", list(GEAR))
    fuel = c2.selectbox("Fuel", list(FUEL))
    doors = c3.radio("Doors", [5, 3])
    cv = st.slider("Fiscal horsepower (CV)", 3, 40, 6)
    c1, c2 = st.columns(2)
    condition = c1.selectbox("Condition", list(COND))
    first_hand = c2.selectbox("First owner", list(FIRST))
    c1, c2 = st.columns(2)
    origin = c1.selectbox("Origin", list(ORIG))
    cities = sorted(CATS["ville"])
    city = c2.selectbox("City", cities, index=cities.index("Casablanca") if "Casablanca" in cities else 0)
    equipment = st.multiselect("Equipment", list(EQUIP), ["Air conditioning", "ABS", "Airbags", "Power windows"])

row = {
    "marque": make, "modele": f"{make} | {model}", "age": max(0, date.today().year - year),
    "kilometrage": np.nan if km_unknown else float(km), "km_inconnu": km_unknown, "boite": GEAR[gearbox],
    "puissance_fiscale": float(cv), "carburant": FUEL[fuel], "etat": COND[condition], "nb_portes": float(doors),
    "origine": ORIG[origin], "premiere_main": FIRST[first_hand], "ville": city, "nb_equipements": len(equipment),
}
for label, code in EQUIP.items():
    row[f"eq_{code}"] = label in equipment
dm = xgboost.DMatrix(transform(row), enable_categorical=True)
price = float(np.exp(booster.predict(dm)[0]))
t = next(t for t in reversed(TRANCHES) if price >= t["prix_predit_min"])
contrib = booster.predict(dm, pred_contribs=True)[0][:-1]

with right:
    st.metric("Estimated price", mad(price))
    st.markdown(f"80% price range: **{mad(price * t['facteur_bas'])} – {mad(price * t['facteur_haut'])}** (calibrated on the test set)")
    if price < 50_000:
        st.warning("Entry-level car: the real price depends a lot on its actual condition (test error ~22% in this range).")
    if km_unknown:
        st.info("Mileage unknown: less precise estimate.")
    items = []
    for v, c in zip(VARIABLES, contrib):
        label = NAMES.get(v) or next((k for k, code in EQUIP.items() if v == f"eq_{code}"), v)
        items.append((label, round(float((np.exp(c) - 1) * 100), 1)))
    items = sorted(items, key=lambda x: -abs(x[1]))[:8]
    st.markdown("**Why this price?** Top factors vs. an average listing (XGBoost contributions, % effect on price)")
    st.bar_chart(pd.DataFrame({"Effect on price (%)": [i[1] for i in items]}, index=[i[0] for i in items]), horizontal=True)
    st.caption("Limits: asking prices from 2024 listings (not transaction prices); makes / models / cities seen fewer than "
               "20 times in training are treated as 'other'.")
