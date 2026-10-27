# 🛠️ AI IT Service Desk Assistant

> **Autonomous Incident Triage, Priority Mapping, and Similar Historical Solution Retrieval System**  
> *Prepared for Lasya | Interview Preparation for UST (GD + Technical Interview)*

---

## 📌 Project Overview
- **What it does:** A web application where an employee or IT agent types a support issue (e.g., *"VPN is not working since morning"*), and the system automatically:
  1. Predicts the ticket category (`Network`, `Access`, `Hardware`, `Software`) with confidence probability.
  2. Assigns an operational priority (`High`, `Medium`, `Low`).
  3. Recommends the specific assignment routing group.
  4. Retrieves the **top 3 most similar past tickets** using Cosine Similarity along with their **verified historical resolutions**.
  5. Generates the ServiceNow Incident REST API payload (`incident` table).
  6. Persists every triage transaction into a local SQLite database (`tickets.db`).
- **Why it matters:** Enterprise IT operations (like UST support teams) handle thousands of incoming tickets daily. Manual triage is slow and error-prone. This automated assistant reduces Mean Time to Resolution (MTTR), avoids misrouting, and empowers agents to reuse past resolutions instantly.
- **Relation to ServiceNow:** ServiceNow is the complete ITSM platform managing incident lifecycles (create, assign, SLA escalations, resolve, close). This project is a functional architectural prototype illustrating how ServiceNow **Predictive Intelligence** works under the hood.

---

## ✨ Features
1. **Intelligent Ticket Classification:** TF-IDF n-gram vectorization + Logistic Regression classifier with probability confidence score.
2. **Operational Priority Mapping:** Rule-based urgency mapping (`Network` $\rightarrow$ High, `Access`/`Software` $\rightarrow$ Medium, `Hardware` $\rightarrow$ Low).
3. **Historical Ticket & Solution Retrieval:** Top 3 distinct historical tickets matched via Cosine Similarity, showing match percentage and recommended resolutions.
4. **Interactive Streamlit Web UI:**
   - ⚡ 5 Quick-Fill Scenario Buttons for seamless live demos (`VPN Failure`, `Password Reset`, `Screen Flickering`, `Outlook Sync`, `Printer Jam`).
   - Category probability distribution visual chart.
   - Live metrics bar (Test Accuracy, Knowledge Base Records, Total Triaged, Architecture).
5. **ServiceNow Incident Integration Simulator:**
   - Previews the JSON payload for `POST /api/now/table/incident`.
   - Displays sample ServiceNow Async Business Rule script calling this NLP microservice.
6. **SQLite Triage Persistence:**
   - Automatic table creation and schema migration.
   - Interactive table of recent tickets.
   - 📥 One-click CSV export of logged tickets.
   - 🗑️ Clear log option.
7. **Model Performance Analytics:**
   - Per-category Precision, Recall, and F1-Score breakdown in sidebar.
   - Evaluated on a 20% stratified holdout test set.
8. **Dataset Flexibility:** Automatically detects and loads `data/tickets.csv` if present (100 realistic IT tickets with resolutions); seamlessly falls back to synthetic generation if removed.

---

## 🏗️ System Architecture

```text
User / ServiceNow Ticket Text
             │
             ▼
   Streamlit Web Interface (app.py)
             │
             ├──► TF-IDF Vectorizer (Unigrams + Bigrams)
             │          │
             │          ▼
             │    Logistic Regression Classifier
             │          │
             │          ├──► Predicted Category + Confidence Score
             │          └──► Priority Assignment Rule (Network->High, etc.)
             │
             ├──► Cosine Similarity against Historical Knowledge Base
             │          │
             │          ▼
             │    Top 3 Distinct Past Tickets + Verified Resolutions
             │
             ├──► SQLite Database (tickets.db) Logging & CSV Export
             │
             └──► ServiceNow REST API Incident Payload Generator
```

---

## 💻 Tech Stack
- **Programming Language:** Python 3.10+ (tested on Python 3.14)
- **Data Manipulation:** `pandas`
- **Machine Learning & NLP:** `scikit-learn` (TF-IDF Vectorizer, Logistic Regression, Cosine Similarity, train_test_split, classification_report)
- **Web Interface:** `Streamlit`
- **Database:** `SQLite` (built-in Python `sqlite3`)
- **Version Control:** `Git`

---

## 📂 Project Structure
```text
Lasya-Ai/
├── app.py                 # Complete Streamlit UI + ML pipeline + SQLite persistence
├── requirements.txt       # Dependencies (scikit-learn, pandas, streamlit)
├── README.md              # Complete project documentation & interview guide
├── .gitignore             # Git ignore file for cache, db, and virtual environments
├── data/
│   └── tickets.csv        # 100 realistic IT support tickets with verified resolutions
├── tickets.db             # Auto-generated SQLite database
├── test_pipeline.py       # Automated test suite for Section 11 test cases & edge cases
└── generate_dataset.py    # Script used to generate real-world IT dataset
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python test_pipeline.py
```

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 Verified Test Cases

| Input Issue | Predicted Category | Priority | Recommended Action / Verified Resolution |
| :--- | :---: | :---: | :--- |
| `VPN is not working since morning` | **Network** | **High** | Restart Cisco AnyConnect client & verify MFA credentials. |
| `I forgot my password` | **Access** | **Medium** | Trigger automated SSPR link & verify identity via backup email. |
| `My laptop screen is flickering` | **Hardware** | **Low** | Adjust refresh rate to 60Hz & update OEM display drivers. |
| `Outlook is not syncing` | **Software** | **Medium** | Remove cached Exchange .OST file & let Outlook rebuild cache. |
| `Printer is jammed` | **Hardware** | **Low** | Open rear maintenance door, clear paper path & reset roller latch. |

---

## 🎙️ Two-Minute Interview Elevator Pitch

> *"I built an AI IT Service Desk Assistant prototype designed to solve ticket routing bottlenecks in enterprise IT operations.*
> 
> *When an incident description is submitted, the system vectorizes the text using TF-IDF with bigrams and predicts the ticket category using a multinomial logistic regression classifier with confidence probabilities. It automatically applies operational priority mapping and uses cosine similarity to retrieve the top 3 similar past tickets along with verified resolutions so support agents can reuse solutions.*
> 
> *All triage events are logged to an SQLite database with full CSV export capabilities. Having earned ServiceNow CSA and CAD certifications, I designed this to illustrate the underlying mechanics of ServiceNow Predictive Intelligence. The application includes a ServiceNow REST API preview showing how a Business Rule or Flow Designer action calls this microservice to auto-populate incident category and priority fields."*

---

## ❓ Common Interview Q&A

1. **Why TF-IDF?**  
   It is lightweight, fast to compute, runs without GPU requirements, and effectively elevates discriminative domain keywords (e.g. *"VPN"*, *"SSO"*, *"flickering"*) over common stopwords.
2. **Why Logistic Regression?**  
   It is a proven, highly interpretable baseline for text classification. It trains rapidly and outputs calibrated probability distributions for confidence scoring.
3. **What is Cosine Similarity?**  
   It evaluates the cosine of the angle between two multi-dimensional text vectors, comparing text meaning independently of ticket length.
4. **Why not deep learning / LLMs initially?**  
   For closed-domain classification with well-defined categories, TF-IDF + Logistic Regression delivers sub-millisecond latency, zero inference cost, and high determinism before scaling to LLMs or Sentence-Transformers.
5. **How does this connect to ServiceNow?**  
   In ServiceNow, this model can be hosted as an internal REST microservice. A ServiceNow Scripted REST API or Outbound REST Message triggered by an Incident `before insert` Business Rule sends the `short_description` and automatically populates `category`, `urgency`, `impact`, and `work_notes`.
#   A I - I T - S e r v i c e - D e s k - A s s i s t a n t - P r o j e c t - G u i d e  
 