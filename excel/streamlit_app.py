"""Excel sales automation — Fatima Zahrae Ahannuk.
Runs the real pipeline of github.com/fahan860/excel-automation-project (downloaded at startup):
merge monthly Excel files → clean (mixed date formats, inconsistent product names, missing
values, duplicates) → formatted multi-sheet executive report."""
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Excel sales automation", page_icon="📊", layout="wide")
ZIP = "https://codeload.github.com/fahan860/excel-automation-project/zip/refs/heads/main"


@st.cache_resource
def get_repo():
    root = Path(tempfile.gettempdir()) / "excel_project"
    if not root.exists():
        with urllib.request.urlopen(ZIP, timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    return next(root.iterdir())


repo = get_repo()
sys.path.insert(0, str(repo / "src"))
from excel_automation.data_processing.cleaning import clean_dataframe  # noqa: E402
from excel_automation.io.excel_io import read_excel_file  # noqa: E402
from excel_automation.reporting.report_generator import generate_report  # noqa: E402

st.title("📊 Excel sales automation")
st.markdown(
    "Upload monthly sales spreadsheets (columns: date, product, category, quantity, price, country) — or keep the "
    "3 sample files. The pipeline merges them, fixes mixed date formats, standardises product names, imputes missing "
    "values, removes duplicates and builds a formatted multi-sheet report (KPIs, top products, revenue by country and "
    "category).  \n[Code](https://github.com/fahan860/excel-automation-project) · Author: **Fatima Zahrae Ahannuk**"
)
uploads = st.file_uploader("Monthly Excel files (optional)", type=["xlsx", "xls"], accept_multiple_files=True)
sources = [(u.name, u) for u in uploads] if uploads else [(p.name, p) for p in sorted((repo / "data").glob("*.xlsx"))]
if not uploads:
    st.caption("Using the 3 sample files from the repository: " + ", ".join(n for n, _ in sources))

frames = []
for name, src in sources:
    try:
        if hasattr(src, "read"):
            tmp = Path(tempfile.mkdtemp()) / name
            tmp.write_bytes(src.getvalue())
            src = tmp
        frames.append(read_excel_file(Path(src)))
    except Exception as err:
        st.error(f"{name}: could not be read ({err})")
if frames:
    merged = pd.concat(frames, ignore_index=True)
    clean = clean_dataframe(merged)
    out = Path(tempfile.mkdtemp()) / "sales_report.xlsx"
    generate_report(clean, out)
    c = st.columns(4)
    c[0].metric("Rows merged", f"{len(merged):,}")
    c[1].metric("Rows after cleaning", f"{len(clean):,}", f"-{len(merged) - len(clean)} invalid / duplicates", delta_color="off")
    c[2].metric("Total revenue", f"{clean['total_price'].sum():,.0f}")
    c[3].metric("Products · countries", f"{clean['product'].nunique()} · {clean['country'].nunique()}")
    st.download_button("⬇️ Download the generated report (sales_report.xlsx)", out.read_bytes(), "sales_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    g1, g2 = st.columns(2)
    g1.markdown("**Top products by revenue**")
    g1.bar_chart(clean.groupby("product")["total_price"].sum().sort_values(ascending=False).head(8))
    g2.markdown("**Revenue by category**")
    g2.bar_chart(clean.groupby("category")["total_price"].sum().sort_values(ascending=False))
    b1, b2 = st.columns(2)
    b1.markdown("**Before** — raw merged rows (note the mixed dates and product names)")
    b1.dataframe(merged.head(12), width="stretch")
    b2.markdown("**After** — cleaned rows")
    b2.dataframe(clean.head(12), width="stretch")
