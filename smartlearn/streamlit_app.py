"""SmartLearn — content-based course recommender — Fatima Zahrae Ahannuk.

Same logic as SmartLearn's FastAPI recommendation microservice: each course is embedded with
sentence-transformers (all-MiniLM-L6-v2) from its title, description, level and tags; the
learner profile is the mean embedding of the courses they followed; candidates are ranked by
cosine similarity; with no history (cold start) the service falls back to popularity.
Here it runs on a small demo catalogue instead of the platform's MongoDB."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartLearn recommender", page_icon="🎓", layout="wide")

C = [  # title, category, level, tags, description, popularity (enrolments in the demo data)
    ("Python Fundamentals", "Programming", "beginner", "python basics syntax", "Variables, loops, functions and files: your first Python programs.", 980),
    ("Object-Oriented Python", "Programming", "intermediate", "python oop classes", "Classes, inheritance and design patterns to structure larger Python projects.", 410),
    ("JavaScript Essentials", "Programming", "beginner", "javascript web basics", "The language of the web: types, functions, DOM and events.", 870),
    ("TypeScript for Large Codebases", "Programming", "intermediate", "typescript javascript types", "Static typing, generics and tooling for maintainable front-end and Node code.", 300),
    ("Java & Spring Boot", "Programming", "intermediate", "java spring backend", "Build REST back-ends with Java, Spring Boot and JPA.", 350),
    ("Go for Backend Services", "Programming", "intermediate", "go concurrency backend", "Goroutines, channels and building fast HTTP services in Go.", 180),
    ("Rust Systems Programming", "Programming", "advanced", "rust memory safety systems", "Ownership, borrowing and zero-cost abstractions for safe systems code.", 140),
    ("Data Structures & Algorithms", "Programming", "intermediate", "algorithms complexity interview", "Arrays, trees, graphs, sorting and Big-O analysis with exercises.", 620),
    ("SQL for Data Analysis", "Data Science", "beginner", "sql databases queries", "SELECT, joins, aggregations and window functions on real datasets.", 760),
    ("Statistics for Data Science", "Data Science", "beginner", "statistics probability", "Distributions, hypothesis tests, confidence intervals and A/B testing.", 520),
    ("Data Analysis with Pandas", "Data Science", "beginner", "python pandas data cleaning", "Load, clean, reshape and summarise tabular data with pandas.", 690),
    ("Data Visualization with Matplotlib & Plotly", "Data Science", "intermediate", "visualization charts dashboards", "Tell stories with data: chart choice, interactive plots and dashboards.", 330),
    ("Machine Learning with scikit-learn", "Data Science", "intermediate", "machine learning classification regression", "Supervised learning, pipelines, cross-validation and model evaluation.", 640),
    ("Feature Engineering & Model Tuning", "Data Science", "advanced", "machine learning xgboost optuna", "Encoding, leakage, gradient boosting and hyper-parameter search.", 210),
    ("Deep Learning with PyTorch", "Data Science", "intermediate", "deep learning neural networks pytorch", "Tensors, autograd, CNNs and training loops in PyTorch.", 450),
    ("Computer Vision with CNNs", "Data Science", "advanced", "deep learning computer vision images", "Convolutional networks, transfer learning and image classification.", 260),
    ("Natural Language Processing", "Data Science", "intermediate", "nlp text transformers", "Tokenisation, TF-IDF, embeddings and transformer models for text.", 380),
    ("LLMs & Retrieval-Augmented Generation", "Data Science", "advanced", "llm rag embeddings vector database", "Prompting, embeddings, vector search and building RAG assistants.", 400),
    ("Big Data with Apache Spark", "Data Science", "intermediate", "spark big data distributed", "DataFrames, transformations and distributed ETL with PySpark.", 290),
    ("Data Engineering Pipelines", "Data Science", "advanced", "etl airflow data warehouse", "Batch pipelines, orchestration, data lakes and warehouse modelling.", 230),
    ("HTML & CSS from Scratch", "Web Development", "beginner", "html css web design", "Semantic HTML, flexbox, grid and responsive layouts.", 810),
    ("React: Modern Front-End", "Web Development", "intermediate", "react javascript frontend", "Components, hooks, state management and routing in React.", 590),
    ("Vue.js Essentials", "Web Development", "beginner", "vue javascript frontend", "Reactive components and single-page apps with Vue 3.", 220),
    ("Angular in Practice", "Web Development", "intermediate", "angular typescript frontend", "Modules, services, RxJS and forms in Angular.", 200),
    ("Node.js & Express APIs", "Web Development", "intermediate", "nodejs express rest api", "Build and secure REST APIs with Node.js, Express and JWT.", 480),
    ("REST API Design", "Web Development", "intermediate", "rest api http design", "Resources, status codes, versioning, pagination and documentation.", 260),
    ("GraphQL APIs", "Web Development", "advanced", "graphql api schema", "Schemas, resolvers and performance for GraphQL servers.", 150),
    ("Docker for Developers", "Cloud & DevOps", "beginner", "docker containers", "Images, containers, volumes and Docker Compose for local stacks.", 530),
    ("Kubernetes Fundamentals", "Cloud & DevOps", "intermediate", "kubernetes orchestration containers", "Pods, deployments, services and scaling on Kubernetes.", 310),
    ("CI/CD with GitHub Actions", "Cloud & DevOps", "intermediate", "ci cd automation git", "Automate tests, builds and deployments with GitHub Actions.", 270),
    ("AWS Cloud Practitioner", "Cloud & DevOps", "beginner", "aws cloud services", "Core AWS services: compute, storage, networking and pricing.", 440),
    ("Azure Fundamentals", "Cloud & DevOps", "beginner", "azure cloud microsoft", "Azure services, identity and resource management basics.", 230),
    ("MLOps: Deploying ML Models", "Cloud & DevOps", "advanced", "mlops deployment fastapi monitoring", "Serve models with FastAPI and Docker, track experiments and monitor drift.", 190),
    ("Git & GitHub Workflow", "Cloud & DevOps", "beginner", "git version control collaboration", "Branches, pull requests, merges and team workflows.", 720),
    ("Cybersecurity Basics", "Security", "beginner", "security networks threats", "Threats, encryption, authentication and secure habits.", 360),
    ("Web Application Security", "Security", "intermediate", "owasp web security", "OWASP Top 10: injection, XSS, CSRF and how to prevent them.", 210),
]
TITLES = [c[0] for c in C]
TEXTS = [f"{t}\n{d}\nlevel: {lv}\ntags: {tg}" for t, cat, lv, tg, d, _ in C]


@st.cache_resource
def load():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return model, model.encode(TEXTS, normalize_embeddings=True)


with st.spinner("Loading the sentence-transformers model (first visit only)…"):
    model, EMB = load()

st.title("🎓 SmartLearn — personalised course recommendations")
st.markdown(
    "Content-based recommender from my **SmartLearn** platform (React/Express front & back end, FastAPI ML microservice, "
    "Spark ETL, 5 Docker services). Courses are embedded with **sentence-transformers (all-MiniLM-L6-v2)**; your profile "
    "is the mean embedding of what you followed; ranking = cosine similarity; cold start = popularity.  \n"
    "Running here on a demo catalogue of 36 courses · [Code](https://github.com/fahan860/smartlearn-platform) · "
    "Author: **Fatima Zahrae Ahannuk**"
)
left, right = st.columns([1, 1.3])
with left:
    history = st.multiselect("Courses you already followed", TITLES, ["Python Fundamentals", "Data Analysis with Pandas"])
    goal = st.text_input("Optional: what do you want to learn? (free text)",
                         placeholder="e.g. build a chatbot that answers questions about my documents")
    level = st.radio("Level", ["any", "beginner", "intermediate", "advanced"], horizontal=True)
    k = st.slider("How many recommendations", 3, 10, 5)

taken = [TITLES.index(h) for h in history]
vecs = [EMB[i] for i in taken]
if goal.strip():
    vecs.append(model.encode([goal.strip()], normalize_embeddings=True)[0])
allowed = [i for i in range(len(C)) if i not in taken and (level == "any" or C[i][2] == level)]
with right:
    if not vecs:
        idx = sorted(allowed, key=lambda i: -C[i][5])[:k]
        st.info("**Cold start** (no history, no goal): popularity ranking, like SmartLearn's fallback.")
        last = ("Enrolments", [C[i][5] for i in idx])
    else:
        profile = np.mean(vecs, axis=0)
        profile /= np.linalg.norm(profile) + 1e-12
        scores = EMB @ profile
        idx = sorted(allowed, key=lambda i: -scores[i])[:k]
        src = f"{len(taken)} followed course(s)" + (" + your goal" if goal.strip() else "")
        st.success(f"**Personalised** — profile = mean embedding of {src}; ranked by cosine similarity.")
        last = ("Similarity", [round(float(scores[i]), 2) for i in idx])
    st.dataframe(pd.DataFrame({"Course": [C[i][0] for i in idx], "Category": [C[i][1] for i in idx],
                               "Level": [C[i][2] for i in idx], last[0]: last[1]}),
                 hide_index=True, width="stretch")
