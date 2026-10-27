import os
import random
import sqlite3
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

random.seed(42)

TEMPLATES = {
    "Network": [
        "VPN not connecting from home",
        "WiFi keeps dropping in office",
        "Cannot reach internal portal, network timeout",
    ],
    "Access": [
        "Forgot password for email account",
        "Need access to shared drive",
        "Account locked after failed login",
    ],
    "Hardware": [
        "Laptop screen flickering",
        "Keyboard keys not working",
        "Printer not printing, paper jam",
    ],
    "Software": [
        "Excel crashes on opening file",
        "Need Python installed on my laptop",
        "Outlook not syncing emails",
    ],
}

PRIORITY = {"Network": "High", "Access": "Medium", "Hardware": "Low", "Software": "Medium"}

PREFIX = ["", "Urgent: ", "Hi team, ", "Please help - ", "Issue: "]
SUFFIX = ["", " since morning", " again today", " please resolve", " affecting my work"]

DB_PATH = "tickets.db"

@st.cache_data
def load_data():
    """Use data/tickets.csv (columns: text, category) if present, else synthetic data."""
    if os.path.exists("data/tickets.csv"):
        df = pd.read_csv("data/tickets.csv")[["text", "category"]].dropna()
        df["base"] = df["text"]
        return df
    rows = []
    for category, texts in TEMPLATES.items():
        for t in texts:
            for _ in range(40):
                text = random.choice(PREFIX) + t + random.choice(SUFFIX)
                rows.append((text, category, t))
    return pd.DataFrame(rows, columns=["text", "category", "base"])

@st.cache_resource
def train_model(df):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.category, test_size=0.2, random_state=42, stratify=df.category
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    accuracy = accuracy_score(y_te, pred)
    print(classification_report(y_te, pred))
    all_vecs = vec.transform(df.text)
    return vec, clf, all_vecs, accuracy

def triage(ticket, df, vec, clf, all_vecs):
    v = vec.transform([ticket])
    probs = clf.predict_proba(v)[0]
    category = clf.classes_[probs.argmax()]
    confidence = float(probs.max())
    sims = cosine_similarity(v, all_vecs)[0]
    similar, seen = [], set()
    for i in sims.argsort()[::-1]:
        base = df.base.iloc[i]
        if base not in seen:
            seen.add(base)
            similar.append((base, round(float(sims[i]), 2)))
        if len(similar) == 3:
            break
    return category, PRIORITY.get(category, "Medium"), confidence, similar

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS log("
            "ts TEXT DEFAULT CURRENT_TIMESTAMP, text TEXT, category TEXT, priority TEXT)"
        )

def log_ticket(text, category, priority):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO log(text, category, priority) VALUES (?, ?, ?)",
            (text, category, priority),
        )

def recent_tickets():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT ts, text, category, priority FROM log ORDER BY rowid DESC LIMIT 10",
            conn,
        )

# ---------- UI ----------
st.set_page_config(page_title="AI IT Service Desk Assistant", layout="wide")
init_db()
df = load_data()
vec, clf, all_vecs, accuracy = train_model(df)

st.title("AI IT Service Desk Assistant")
st.caption("Automatic ticket triage: category, priority, and similar past tickets")
st.sidebar.metric("Model accuracy (test set)", f"{accuracy:.0%}")
st.sidebar.write(f"Training tickets: {len(df)}")

ticket = st.text_area("Describe the issue", placeholder="e.g. VPN not connecting from home")
if st.button("Triage") and ticket.strip():
    category, priority, confidence, similar = triage(ticket, df, vec, clf, all_vecs)
    log_ticket(ticket, category, priority)
    st.success(f"Category: {category} | Priority: {priority} | Confidence: {confidence:.0%}")
    st.subheader("Similar past tickets")
    for text, score in similar:
        st.write(f"- {text} (similarity {score})")

st.subheader("Recent tickets")
st.dataframe(recent_tickets(), use_container_width=True)
