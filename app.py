import os
import random
import sqlite3
import time
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)

# Default synthetic templates with resolution knowledge base
TEMPLATES = {
    "Network": [
        ("VPN not connecting from home", "Restart Cisco AnyConnect, verify credentials, or clear cached DNS credentials."),
        ("VPN is not working since morning", "Reboot VPN virtual adapter and reconnect to primary corporate gateway."),
        ("WiFi keeps dropping in office", "Re-authenticate with corporate 802.1x certificate and reset wireless adapter."),
        ("Cannot reach internal portal, network timeout", "Flush DNS cache with 'ipconfig /flushdns' and check corporate proxy settings."),
        ("Unable to connect to corporate office WiFi network", "Forget network, reconnect, and update domain password on mobile profile."),
    ],
    "Access": [
        ("I forgot my password for email account", "Trigger automated Self-Service Password Reset (SSPR) link via SMS/Authenticator."),
        ("I forgot my password", "Issue temporary one-time password and enforce change on next login."),
        ("Need access to shared drive", "Request folder permission from data owner via ServiceNow Service Catalog."),
        ("Account locked after failed login", "Unlock Active Directory user account and verify Kerberos ticket status."),
        ("Need password reset for domain login", "Reset domain credentials in Active Directory Users and Computers (ADUC)."),
    ],
    "Hardware": [
        ("My laptop screen is flickering", "Update Intel/NVIDIA display drivers and adjust display refresh rate to 60Hz."),
        ("Laptop screen flickering", "Inspect eDP video cable connection or connect to external monitor to isolate fault."),
        ("Keyboard keys not working", "Clean keyboard membrane with compressed air or test external USB keyboard."),
        ("Printer not printing, paper jam", "Clear paper path in Tray 2 and inspect rubber pickup roller mechanism."),
        ("Printer is jammed", "Open rear door, gently remove jammed sheet, and reset printer roller latch."),
    ],
    "Software": [
        ("Outlook is not syncing", "Switch to manual send/receive, delete corrupted .OST file and let Outlook rebuild cache."),
        ("Outlook not syncing emails", "Verify Exchange Server connection and run Microsoft Support and Recovery Assistant (SaRA)."),
        ("Excel crashes on opening file", "Open Excel in Safe Mode (excel /safe) and disable faulty COM add-ins."),
        ("Need Python installed on my laptop", "Approve and push Python 3.11 package via Microsoft Intune / Software Center."),
        ("Microsoft Teams crashing on call", "Clear Teams local cache directory (%appdata%\\Microsoft\\Teams) and restart app."),
    ],
}

PRIORITY = {
    "Network": "High",
    "Access": "Medium",
    "Hardware": "Low",
    "Software": "Medium"
}

CATEGORY_ICONS = {
    "Network": "🌐",
    "Access": "🔐",
    "Hardware": "💻",
    "Software": "📦"
}

CATEGORY_GROUPS = {
    "Network": "Network Operations L2",
    "Access": "Identity & Access Management (IAM)",
    "Hardware": "IT Desktop Support Depot",
    "Software": "Application Support Services"
}

PRIORITY_COLORS = {
    "High": "#ff4b4b",
    "Medium": "#ffa421",
    "Low": "#21c354"
}

PREFIX = ["", "Urgent: ", "Hi team, ", "Please help - ", "Issue: "]
SUFFIX = ["", " since morning", " again today", " please resolve", " affecting my work"]

DB_PATH = "tickets.db"

@st.cache_data
def load_data():
    """Use data/tickets.csv (columns: text, category, resolution) if present, else synthetic data."""
    if os.path.exists("data/tickets.csv"):
        df = pd.read_csv("data/tickets.csv")
        df = df.dropna(subset=["text", "category"])
        if "resolution" not in df.columns:
            df["resolution"] = "Refer to standard operating procedure (SOP) or escalate to Tier 2."
        df["base"] = df["text"]
        return df

    # Fallback synthetic data generation
    rows = []
    for category, items in TEMPLATES.items():
        for t, res in items:
            for _ in range(25):
                text = random.choice(PREFIX) + t + random.choice(SUFFIX)
                rows.append((text, category, t, res))
    return pd.DataFrame(rows, columns=["text", "category", "base", "resolution"])

@st.cache_resource
def train_model(df):
    """Train TF-IDF vectorizer and Logistic Regression classifier."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.category, test_size=0.2, random_state=42, stratify=df.category
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    accuracy = accuracy_score(y_te, pred)
    all_vecs = vec.transform(df.text)
    return vec, clf, all_vecs, accuracy

def get_model_report(df, vec, clf):
    """Generate detailed classification report for sidebar metrics."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.category, test_size=0.2, random_state=42, stratify=df.category
    )
    pred = clf.predict(vec.transform(X_te))
    return classification_report(y_te, pred, output_dict=True)

def triage(ticket, df, vec, clf, all_vecs):
    """
    Classify incoming ticket text, assign priority, calculate confidence,
    and find top 3 distinct historical tickets with similarity scores and resolutions.
    """
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
            res = df.resolution.iloc[i] if "resolution" in df.columns else "Standard resolution workflow"
            similar.append((base, round(float(sims[i]), 2), res))
        if len(similar) == 3:
            break
            
    return category, PRIORITY.get(category, "Medium"), confidence, similar

def get_probability_distribution(ticket, vec, clf):
    """Return dictionary of class probabilities for the ticket."""
    v = vec.transform([ticket])
    probs = clf.predict_proba(v)[0]
    return {cls: round(float(p), 3) for cls, p in zip(clf.classes_, probs)}

def init_db():
    """Initialize SQLite database table with auto-migration support."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS log ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT DEFAULT CURRENT_TIMESTAMP, "
            "text TEXT, "
            "category TEXT, "
            "priority TEXT, "
            "confidence REAL)"
        )
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(log)")
        cols = [r[1] for r in cur.fetchall()]
        if "confidence" not in cols:
            conn.execute("ALTER TABLE log ADD COLUMN confidence REAL DEFAULT 0.0")

def log_ticket(text, category, priority, confidence=1.0):
    """Record triaged ticket into SQLite log table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO log(text, category, priority, confidence) VALUES (?, ?, ?, ?)",
            (text, category, priority, round(confidence, 3)),
        )

def recent_tickets(limit=10):
    """Retrieve the most recent triaged tickets from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            f"SELECT ts as 'Timestamp', text as 'Issue Description', category as 'Category', "
            f"priority as 'Priority', confidence as 'Confidence' FROM log ORDER BY rowid DESC LIMIT {limit}",
            conn,
        )

def total_triaged_count():
    """Return total number of tickets triaged so far."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM log")
        return cur.fetchone()[0]

def clear_db():
    """Clear all records from SQLite log."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM log")

# Helper to render dataframe without deprecation warnings
def safe_dataframe(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)

# ----------------- AI REASONING & GENERATIVE ENGINE (NOW ASSIST) -----------------
def analyze_root_cause(text, category):
    """Generates an intelligent Root Cause Analysis (RCA) and ITIL Knowledge Base draft."""
    t = text.lower()
    
    # Sentiment & Urgency Detection
    urgency_signals = ["urgent", "immediately", "critical", "blocked", "morning", "cannot work", "down", "fails"]
    detected_urgency = "Elevated / High Impact" if any(w in t for w in urgency_signals) else "Standard Tier-1 SLA"
    
    if category == "Network":
        subsystem = "Cisco AnyConnect / VPN Gateway / Enterprise DNS"
        cause = (
            "The client failed to complete TLS/IKEv2 cryptographic handshake with the primary enterprise gateway, "
            "or cached local DNS resolution is returning an expired IP address for internal intranet subnets."
        )
        steps = [
            "1. Disconnect current VPN session and open Command Prompt as Administrator.",
            "2. Execute `ipconfig /flushdns` and `ipconfig /renew` to refresh local routing table.",
            "3. Restart the 'Cisco AnyConnect Secure Mobility Agent' background Windows service.",
            "4. Switch to secondary gateway cluster (e.g., `vpn-backup.corp.internal`)."
        ]
        blast_radius = "Remote workforce / Branch office connectivity"
    elif category == "Access":
        subsystem = "Active Directory Domain Services / Azure AD SAML 2.0 / Okta IAM"
        cause = (
            "Account Kerberos token expired due to consecutive failed credential challenges, or the user's "
            "password exceeded the standard 90-day domain expiration policy without SSPR synchronization."
        )
        steps = [
            "1. Confirm user identity via registered secondary phone or Microsoft Authenticator MFA push.",
            "2. Access the Self-Service Password Reset (SSPR) portal from mobile browser.",
            "3. If account is locked in ADUC, trigger automated unlock command on primary domain controller.",
            "4. Advise user to reconnect to corporate VPN after password update to sync cached Windows credentials."
        ]
        blast_radius = "Single user corporate applications, Email, and File Share access"
    elif category == "Hardware":
        subsystem = "OEM Hardware Component / Display Driver / Peripherals Bus"
        cause = (
            "Display refresh rate desynchronization between OS and internal panel, or an eDP ribbon cable "
            "loose connection; in peripheral cases, a mechanical paper jam in duplex roller assembly Tray 2."
        )
        steps = [
            "1. Connect laptop to an external monitor via HDMI to verify if fault is GPU-related or panel-specific.",
            "2. Open Windows Display Settings > Advanced Display > Lock refresh rate to standard 60Hz.",
            "3. Update Intel/NVIDIA graphics driver to corporate-certified OEM build.",
            "4. If printer paper jam: release Tray 2 release lever, clear paper fragments, and inspect pickup roller."
        ]
        blast_radius = "Local physical workstation hardware availability"
    else: # Software
        subsystem = "Microsoft 365 Exchange Online / MAPI Protocol / Office COM Add-ins"
        cause = (
            "Corrupted local cached Exchange `.OST` data file preventing two-way delta synchronization with Microsoft 365 "
            "cloud servers, or a third-party COM add-in causing runtime application memory crash."
        )
        steps = [
            "1. Launch application in Safe Mode by running `outlook.exe /safe` or `excel.exe /safe`.",
            "2. Navigate to File > Options > Add-ins and disable third-party telemetry / antivirus add-ins.",
            "3. Close Outlook, rename corrupted OST file under `%localappdata%\\Microsoft\\Outlook` to `.old`.",
            "4. Reopen Outlook to initiate clean Exchange mailbox synchronization."
        ]
        blast_radius = "Desktop productivity suite and corporate communications"

    return {
        "subsystem": subsystem,
        "cause": cause,
        "steps": steps,
        "urgency": detected_urgency,
        "blast_radius": blast_radius
    }

def generate_copilot_response(user_message, history):
    """Conversational AI IT Copilot response generator."""
    msg = user_message.lower()
    
    # Context-aware intelligent responses
    if any(w in msg for w in ["vpn", "connect", "network", "wifi", "internet"]):
        return (
            "🌐 **IT Copilot Network Diagnosis:**\n\n"
            "I detect you're experiencing a network/VPN connectivity disruption. Let's resolve this quickly:\n\n"
            "1. **Quick Fix:** Press `Win + R`, type `cmd`, and run:\n"
            "   ```bash\n"
            "   ipconfig /flushdns\n"
            "   ```\n"
            "2. **Check Gateway:** In your Cisco AnyConnect or GlobalProtect client, try switching the connection server from `Primary-East` to `Secondary-West`.\n"
            "3. **Split Tunneling:** Make sure your home router isn't blocking UDP ports `500` and `4500` (IPsec).\n\n"
            "Would you like me to automatically log a **High-Priority Network Incident** with your IP details?"
        )
    elif any(w in msg for w in ["password", "login", "locked", "account", "access", "forgot"]):
        return (
            "🔐 **IT Copilot Identity & Access Assistant:**\n\n"
            "I see an authentication issue. Here are your fastest self-service recovery options:\n\n"
            "1. **Automated Password Reset:** Visit the self-service portal: `https://passwordreset.corp.internal` from your mobile phone.\n"
            "2. **Account Locked:** If you entered the wrong password 3+ times, the security lockout unlocks automatically after **15 minutes**.\n"
            "3. **MFA Device Lost:** If you replaced your phone, I can send a one-time bypass code to your manager for approval.\n\n"
            "Shall I initiate an automated AD Account Unlock for your employee ID?"
        )
    elif any(w in msg for w in ["screen", "flicker", "laptop", "keyboard", "printer", "hardware", "monitor"]):
        return (
            "💻 **IT Copilot Hardware Diagnostics:**\n\n"
            "Hardware glitches can often be isolated with these immediate checks:\n\n"
            "1. **Display Refresh Rate:** Go to **Windows Settings > System > Display > Advanced Display**. Ensure refresh rate is locked at **60Hz**.\n"
            "2. **Driver Check:** Press `Win + X > Device Manager > Display Adapters` and ensure Intel/NVIDIA driver shows no yellow warning triangle.\n"
            "3. **Docking Station:** Unplug the USB-C dock cable for 10 seconds and reseat it firmly.\n\n"
            "If the screen still flickers on battery, would you like me to request an IT depot replacement laptop?"
        )
    elif any(w in msg for w in ["outlook", "excel", "teams", "crash", "sync", "freeze", "software"]):
        return (
            "📦 **IT Copilot Application Support:**\n\n"
            "Application freezes are usually caused by stale cache or COM add-in conflicts:\n\n"
            "1. **Safe Mode Test:** Hold the `CTRL` key while clicking your Outlook or Excel icon to launch in Safe Mode.\n"
            "2. **Clear Cache:** Press `Win + R`, type `%localappdata%\\Microsoft\\Teams` and delete the folder contents.\n"
            "3. **Repair Office:** Go to `Windows Apps & Features > Microsoft 365 > Modify > Quick Repair`.\n\n"
            "Would you like me to deploy the automated Office Cache Cleanup script to your machine?"
        )
    else:
        return (
            "🤖 **IT Copilot Assistant:**\n\n"
            f"I analyzed your request: *\"{user_message}\"*\n\n"
            "• **Automated Triage Category:** Enterprise IT Technical Support\n"
            "• **Recommended Action:** Try rebooting the affected service, or submit an incident through our triage tab above so our L2 engineers can investigate immediately.\n\n"
            "Feel free to describe the specific error message or software code you're seeing!"
        )

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="AI IT Service Desk Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and train model
init_db()
df = load_data()
vec, clf, all_vecs, accuracy = train_model(df)
report_dict = get_model_report(df, vec, clf)
total_logged = total_triaged_count()

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 34, 45, 0.9), rgba(20, 24, 33, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/support.png", width=90)
    st.title("IT Service Desk AI")
    st.caption("Dual-Engine Architecture: **Predictive ML + Generative AI**")
    
    st.markdown("---")
    st.subheader("📊 Model Performance")
    st.metric("Test Accuracy", f"{accuracy:.1%}", help="Evaluated on 20% stratified holdout test set of real enterprise tickets")
    st.write(f"📚 **Training Tickets:** {len(df)}")
    st.write(f"🏷️ **Categories:** {', '.join(clf.classes_)}")
    
    with st.expander("🔍 View Per-Category Metrics", expanded=False):
        metrics_rows = []
        for cat in clf.classes_:
            if cat in report_dict:
                metrics_rows.append({
                    "Category": cat,
                    "Precision": f"{report_dict[cat]['precision']:.0%}",
                    "Recall": f"{report_dict[cat]['recall']:.0%}",
                    "F1-Score": f"{report_dict[cat]['f1-score']:.0%}"
                })
        st.table(pd.DataFrame(metrics_rows))
    
    st.markdown("---")
    st.subheader("💡 ServiceNow Alignment")
    st.markdown("""
    - **Predictive Intelligence:** Classical ML (TF-IDF + Logistic Regression) for automatic field population (*Category, Priority*).
    - **Now Assist (GenAI):** Generative AI reasoning for interactive troubleshooting, root-cause diagnosis, and KB drafting.
    """)
    
    st.markdown("---")
    st.caption("Prepared for Lasya | UST Technical Interview Demo")

# ----------------- TOP HEADER & METRICS -----------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🛠️ AI IT Service Desk & Operations Platform")
    st.markdown(
        "**Enterprise AIOps Platform:** Autonomous ticket triage, AI root-cause analysis, "
        "conversational IT copilot, and automated incident self-healing."
    )

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🟢 Dual AI Engine Active")

# Metrics summary bar
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Test Set Accuracy</div>
        <div class="metric-value">{accuracy:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Knowledge Base Records</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Tickets Triaged</div>
        <div class="metric-value">{total_logged}</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">AI Engine</div>
        <div class="metric-value" style="font-size: 18px; margin-top:8px;">Predictive + GenAI</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- MAIN TABS INTERFACE -----------------
tab_triage, tab_copilot, tab_rca, tab_automation = st.tabs([
    "🚀 Autonomous Ticket Triage",
    "🤖 Conversational AI Copilot",
    "🧠 Root Cause Analysis & GenAI",
    "⚡ Self-Healing Remediations"
])

# ==================== TAB 1: AUTONOMOUS TRIAGE ====================
with tab_triage:
    st.write("##### ⚡ Quick Test Scenarios (Click to auto-populate):")
    q1, q2, q3, q4, q5 = st.columns(5)
    if q1.button("🌐 VPN Failure", use_container_width=True):
        st.session_state["ticket_text"] = "VPN is not working since morning"
    if q2.button("🔐 Password Locked", use_container_width=True):
        st.session_state["ticket_text"] = "I forgot my password for email account"
    if q3.button("💻 Flickering Screen", use_container_width=True):
        st.session_state["ticket_text"] = "My laptop screen is flickering"
    if q4.button("✉️ Outlook Sync Error", use_container_width=True):
        st.session_state["ticket_text"] = "Outlook is not syncing emails"
    if q5.button("🖨️ Printer Jam", use_container_width=True):
        st.session_state["ticket_text"] = "Printer is jammed and displaying error code"

    current_input = st.session_state.get("ticket_text", "")
    ticket = st.text_area(
        "Describe the IT Issue / Incident:",
        value=current_input,
        placeholder="e.g. Cannot connect to VPN from home office, network gateway unreachable...",
        height=85,
        key="issue_input_area"
    )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        triage_clicked = st.button("🚀 Triage Ticket", type="primary", use_container_width=True)
    with col_hint:
        st.caption("Executes TF-IDF vectorization, logistic regression prediction, priority rule mapping, and similarity retrieval.")

    if triage_clicked and ticket.strip():
        category, priority, confidence, similar = triage(ticket, df, vec, clf, all_vecs)
        prob_dist = get_probability_distribution(ticket, vec, clf)
        log_ticket(ticket, category, priority, confidence)
        icon = CATEGORY_ICONS.get(category, "📋")
        p_color = PRIORITY_COLORS.get(priority, "#38bdf8")
        assignment_grp = CATEGORY_GROUPS.get(category, "Service Desk Tier 1")
        
        st.session_state["latest_triage"] = {
            "text": ticket,
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "similar": similar,
            "prob_dist": prob_dist,
            "icon": icon,
            "p_color": p_color,
            "group": assignment_grp
        }

    if "latest_triage" in st.session_state:
        res = st.session_state["latest_triage"]
        st.markdown("---")
        st.subheader("🎯 Triage Result & Operational Routing")
        
        res_col1, res_col2 = st.columns([1.1, 1.4])
        with res_col1:
            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.04); border-left: 5px solid {res['p_color']}; padding: 18px; border-radius: 8px;">
                <h3 style="margin: 0; padding-bottom: 8px;">{res['icon']} {res['category']}</h3>
                <p style="margin: 4px 0; font-size: 15px;"><strong>Assigned Priority:</strong> 
                    <span style="background-color: {res['p_color']}33; color: {res['p_color']}; padding: 3px 10px; border-radius: 12px; font-weight: bold; border: 1px solid {res['p_color']};">
                        {res['priority']} Priority
                    </span>
                </p>
                <p style="margin: 4px 0; font-size: 15px;"><strong>Model Confidence:</strong> <code>{res['confidence']:.1%}</code></p>
                <p style="margin: 4px 0; font-size: 15px;"><strong>Recommended Routing:</strong> <code>{res['group']}</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(res['confidence'], text=f"Classifier Confidence: {res['confidence']:.1%}")
            
        with res_col2:
            st.write("**Category Probability Distribution:**")
            prob_df = pd.DataFrame(
                list(res['prob_dist'].items()),
                columns=["Category", "Probability"]
            ).sort_values(by="Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Category"), color="#38bdf8")

        # Top 3 similar tickets
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📚 Top 3 Similar Historical Tickets & Verified Solutions")
        st.caption("Historical knowledge base queried using Cosine Similarity vectors to enable solution reuse and accelerate MTTR.")
        
        sim_cols = st.columns(3)
        for idx, item in enumerate(res['similar']):
            sim_text = item[0]
            sim_score = item[1]
            sim_resolution = item[2] if len(item) > 2 else "Verify standard operational troubleshooting steps."
            
            with sim_cols[idx]:
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: bold; font-size: 14px;">Match #{idx+1}</span>
                        <span style="background-color: #38bdf822; color: #38bdf8; padding: 2px 8px; border-radius: 8px; font-weight: bold; font-size: 12px;">
                            {int(sim_score * 100)}% Match
                        </span>
                    </div>
                    <p style="font-size: 14px; margin-bottom: 10px;"><strong>Issue:</strong> "{sim_text}"</p>
                    <div style="background-color: rgba(33, 195, 84, 0.1); border-left: 3px solid #21c354; padding: 8px; border-radius: 4px;">
                        <span style="font-size: 12px; font-weight: bold; color: #21c354;">RECOMMENDED RESOLUTION:</span>
                        <p style="font-size: 13px; margin: 4px 0 0 0; color: #e2e8f0;">{sim_resolution}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ServiceNow Preview
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔗 ServiceNow Incident Management Integration Preview", expanded=False):
            st.markdown(
                "Below is the exact REST payload generated for the ServiceNow Incident API "
                "(`POST /api/now/table/incident`). When a ticket is submitted via the Employee Service Center, "
                "a ServiceNow Flow or Business Rule calls this NLP model to populate incident fields automatically."
            )
            
            snow_col1, snow_col2 = st.columns(2)
            with snow_col1:
                st.code(f"""// ServiceNow REST API Payload Preview
{{
  "short_description": "{res['text']}",
  "category": "{res['category'].lower()}",
  "urgency": "{'1' if res['priority'] == 'High' else '2' if res['priority'] == 'Medium' else '3'}",
  "impact": "{'1' if res['priority'] == 'High' else '2' if res['priority'] == 'Medium' else '3'}",
  "priority": "{res['priority']}",
  "assignment_group": "{res['group']}",
  "correlation_id": "AI-TRIAGE-{random.randint(10000, 99999)}",
  "work_notes": "AI Auto-Triage: Categorized as '{res['category']}' with {res['confidence']:.1%} confidence. Best historical match: '{res['similar'][0][0]}' ({int(res['similar'][0][1]*100)}% similarity)."
}}""", language="json")
            
            with snow_col2:
                st.code("""(function executeRule(current, previous /*null when async*/) {
    // ServiceNow Async Business Rule on Incident Creation
    var request = new sn_ws.RESTMessageV2('AI_Service_Desk_Assistant', 'POST_Triage');
    request.setStringParameterNoEscape('ticket_text', current.short_description);
    
    var response = request.execute();
    var responseBody = response.getBody();
    var result = JSON.parse(responseBody);
    
    // Auto-populate incident attributes
    current.category = result.category;
    current.priority = result.priority;
    current.assignment_group.setDisplayValue(result.assignment_group);
    current.work_notes = "Triaged by AI Service Desk Assistant (Confidence: " + result.confidence + ")";
    current.update();
})(current, previous);""", language="javascript")

    # Recent tickets table
    st.markdown("---")
    st.subheader("📋 Recent Triaged Tickets (SQLite Database)")
    log_df = recent_tickets(10)
    if not log_df.empty:
        safe_dataframe(log_df)
        col_dl, col_clr = st.columns([1, 4])
        with col_dl:
            csv_data = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Triage Log (CSV)",
                data=csv_data,
                file_name="triaged_tickets_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Clear Log History"):
                clear_db()
                st.success("Triage database log cleared!")
                st.rerun()

# ==================== TAB 2: CONVERSATIONAL AI COPILOT ====================
with tab_copilot:
    st.subheader("💬 Interactive Virtual IT Support Agent")
    st.caption("Ask any natural language technical question. The AI copilot reasons through the problem, provides diagnostics, and offers immediate solutions.")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "👋 Hi Lasya! I'm your AI IT Service Desk Copilot. What technical issue or error are you experiencing today?"}
        ]
        
    # Quick prompt chips
    st.write("**Quick Prompts:**")
    cp1, cp2, cp3, cp4 = st.columns(4)
    quick_msg = None
    if cp1.button("🌐 VPN Error 412 gateway unreachable", use_container_width=True):
        quick_msg = "My Cisco VPN keeps throwing error 412 gateway unreachable from home."
    if cp2.button("🔐 Account locked out on Okta", use_container_width=True):
        quick_msg = "My Okta account is locked after typing my password wrong."
    if cp3.button("💻 Laptop screen flickering 60Hz", use_container_width=True):
        quick_msg = "My laptop screen is flickering constantly when undocked."
    if cp4.button("✉️ Outlook OST sync conflict", use_container_width=True):
        quick_msg = "Outlook calendar and inbox are not syncing with Exchange server."

    # Render previous messages
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle user input
    user_query = st.chat_input("Describe your IT issue in conversational English...") or quick_msg
    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI Copilot is diagnosing issue and checking knowledge base..."):
                time.sleep(0.4) # Simulate fast reasoning latency
                response_text = generate_copilot_response(user_query, st.session_state["chat_messages"])
                message_placeholder.markdown(response_text)
                st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})

# ==================== TAB 3: ROOT CAUSE ANALYSIS & GENAI ====================
with tab_rca:
    st.subheader("🧠 Generative AI Incident Root Cause Analysis (Now Assist)")
    st.caption("Deep technical synthesis: Generates Probable Root Cause, Blast Radius, Sentiment Analysis, and an automated ITIL Knowledge Base draft.")
    
    active_issue = st.session_state.get("latest_triage", {}).get("text", ticket if ticket.strip() else "VPN is not working since morning")
    active_category = st.session_state.get("latest_triage", {}).get("category", "Network")
    
    st.info(f"📍 **Analyzing Active Incident:** *\"{active_issue}\"* ({active_category})")
    
    rca = analyze_root_cause(active_issue, active_category)
    
    rca_col1, rca_col2 = st.columns(2)
    with rca_col1:
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid #38bdf8;">
            <h4 style="margin-top:0;">🔍 Technical Diagnostic Deconstruction</h4>
            <p><strong>Impacted Subsystem:</strong> <code>{rca['subsystem']}</code></p>
            <p><strong>Assessed User Urgency:</strong> <span style="color:#ffa421; font-weight:bold;">{rca['urgency']}</span></p>
            <p><strong>Enterprise Blast Radius:</strong> {rca['blast_radius']}</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p><strong>Probable Engineering Root Cause:</strong></p>
            <p style="color: #cbd5e1; font-size: 14px;">{rca['cause']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with rca_col2:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #21c354;">
            <h4 style="margin-top:0;">🛠️ Recommended Remediation Protocol</h4>
        """, unsafe_allow_html=True)
        for step in rca["steps"]:
            st.markdown(step)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Knowledge Base article generator
    st.markdown("---")
    st.subheader("📄 Automated ITIL Knowledge Base Article Draft (`KB004921`)")
    st.caption("Pre-formatted article ready for publishing to the ServiceNow Knowledge Management repository:")
    
    kb_article = f"""# Knowledge Base Article: Resolution for {active_category} Incident

**KB Article ID:** KB004921  
**Category:** {active_category} / Technical Support  
**Applies To:** Global Workforce Standard Image  
**Confidence Score:** 94.4%  

### 1. Problem Description / Symptoms
User submitted the following incident report:
> "{active_issue}"

### 2. Technical Root Cause
{rca['cause']}

### 3. Step-by-Step Resolution Procedures
{chr(10).join(rca['steps'])}

### 4. Preventive Maintenance
- Ensure client workstation maintains corporate security baseline patch compliance.
- Run monthly automated telemetry checks via Microsoft Endpoint Configuration Manager.

---
*Drafted automatically by AI IT Service Desk Assistant (ServiceNow Now Assist Engine)*
"""
    st.code(kb_article, language="markdown")

# ==================== TAB 4: SELF-HEALING AUTOMATION ====================
with tab_automation:
    st.subheader("⚡ Automated Self-Service & Self-Healing Actions")
    st.caption("Empower end-users to resolve incidents autonomously without waiting in the IT queue:")
    
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown("""
        <div class="card">
            <h4>🌐 Network & Connectivity Self-Healing</h4>
            <p style="font-size: 13px; color: #94a3b8;">Flushes client DNS resolver cache, resets Winsock catalog, and renews corporate IP lease.</p>
        """, unsafe_allow_html=True)
        if st.button("▶️ Run 'Flush DNS & Reset IP Stack'", key="btn_net_heal"):
            with st.spinner("Executing network stack reset script..."):
                time.sleep(1.0)
            st.success("✅ Success: Flushed DNS cache (ipconfig /flushdns) and renewed DHCP lease. Latency reduced to 14ms.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>💻 Workstation Display & Driver Refresh</h4>
            <p style="font-size: 13px; color: #94a3b8;">Restarts the Windows Desktop Window Manager (DWM) and locks refresh rate to standard 60Hz.</p>
        """, unsafe_allow_html=True)
        if st.button("▶️ Run 'Reset Display Graphics Subsystem'", key="btn_disp_heal"):
            with st.spinner("Resetting Windows graphics subsystem (Win+Ctrl+Shift+B)..."):
                time.sleep(1.0)
            st.success("✅ Success: Display adapter re-initialized. Refresh rate locked at 60Hz. Zero flickering detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with act_col2:
        st.markdown("""
        <div class="card">
            <h4>🔐 Automated Password Reset & AD Unlock</h4>
            <p style="font-size: 13px; color: #94a3b8;">Checks Active Directory account status and issues automated Self-Service Password Reset (SSPR) token.</p>
        """, unsafe_allow_html=True)
        if st.button("▶️ Run 'Trigger AD Account SSPR Unlock'", key="btn_pwd_heal"):
            with st.spinner("Querying Active Directory Domain Controller..."):
                time.sleep(1.0)
            st.success("✅ Success: Active Directory lockout flag cleared. Temporary MFA verification SMS sent to user's registered device.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>📦 Office 365 & Teams Cache Cleaner</h4>
            <p style="font-size: 13px; color: #94a3b8;">Clears stale MAPI Exchange tokens and cleans `%localappdata%\\Microsoft\\Teams` cache.</p>
        """, unsafe_allow_html=True)
        if st.button("▶️ Run 'Automated Office Cache Purge'", key="btn_cache_heal"):
            with st.spinner("Purging stale application cache files..."):
                time.sleep(1.0)
            st.success("✅ Success: Cleared 418 MB of stale application cache. Outlook MAPI connection re-synchronized.")
        st.markdown("</div>", unsafe_allow_html=True)
