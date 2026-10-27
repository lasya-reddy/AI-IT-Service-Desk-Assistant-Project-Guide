# AI IT Service Desk Assistant

Automatic IT ticket triage: category prediction, priority, and similar-ticket search.

## Features
- **Ticket classification:** Predicts issue category (Network, Access, Hardware, Software) with confidence score
- **Priority assignment:** Automatic mapping based on category urgency (e.g., Network -> High)
- **Similar ticket search:** Top 3 similar past tickets using Cosine Similarity
- **Streamlit UI:** Clean interactive interface with triage button and real-time statistics
- **Ticket log:** SQLite database (`tickets.db`) persisting all triaged tickets
- **Recent tickets table:** Last 10 triaged tickets displayed directly in the UI
- **Model metrics:** Test set accuracy and classification report

## Tech Stack
- **Language:** Python 3.10+
- **Data handling:** pandas
- **Text vectorization:** TF-IDF (unigram + bigram, scikit-learn)
- **Classifier:** Logistic Regression (scikit-learn)
- **Similarity metric:** Cosine similarity
- **Web UI:** Streamlit
- **Database:** SQLite (`tickets.db`)

## Project Structure
```
Lasya-Ai/
├── app.py              # Streamlit app + ML pipeline + SQLite logging
├── requirements.txt    # Project dependencies
├── README.md           # Documentation & interview guide
├── data/               # Optional real dataset
│   └── tickets.csv     # (columns: text, category)
└── tickets.db          # Auto-generated SQLite database
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.

## Model Details
- **Architecture:** TF-IDF Vectorizer (`ngram_range=(1, 2)`, English stop words) + Logistic Regression (`max_iter=1000`).
- **Data split:** 80% Train, 20% Test (stratified by category).
- **Synthetic baseline accuracy:** 100% on structured synthetic data; lower/realistic on noisy real-world data.

## Relation to ServiceNow
ServiceNow is the enterprise IT Service Management (ITSM) platform covering full incident lifecycles (create, assign, SLA tracking, escalate, resolve, close). This application is a prototype demonstrating how AI-driven triage (similar to ServiceNow Predictive Intelligence) operates under the hood.

## Limitations
- Synthetic dataset gives high accuracy due to controlled phrasing.
- Priority is rule-mapped from category rather than trained from business impact.
- TF-IDF relies on exact word overlap rather than semantic embedding models.
- Standalone prototype without live REST integration or multi-user authentication.

## Future Scope
1. Semantic sentence embeddings (e.g., `sentence-transformers` / MiniLM).
2. Machine-learned priority model based on urgency and user impact.
3. Exposing endpoints via FastAPI for REST-based ServiceNow Flow/Business Rule integration.
4. Containerization with Docker.
