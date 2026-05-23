import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  FULL CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

:root {
  --bg0:#02080d; --bg1:#060f18; --bg2:#0a1a26; --bg3:#0e2233;
  --card:rgba(10,26,40,0.85);
  --red:#ff2d55; --red2:#c0102a;
  --teal:#00d4ff; --teal2:#0092b3;
  --gold:#ffd60a; --green:#00ff88;
  --text:#d4eef7; --dim:#4a7a90;
  --border:rgba(0,212,255,0.15); --border2:rgba(0,212,255,0.35);
  --gR:0 0 20px rgba(255,45,85,0.5);
  --gT:0 0 20px rgba(0,212,255,0.45);
}

/* ── KILL TOP WHITESPACE ── */
#root > div:first-child { padding-top:0 !important; }
.block-container { padding-top:0.4rem !important; padding-bottom:1rem !important; }
[data-testid="stAppViewContainer"] > .main { padding-top:0 !important; }
header[data-testid="stHeader"] { display:none !important; }
.stMainBlockContainer { padding-top:0.4rem !important; }
div[data-testid="stDecoration"] { display:none !important; }

/* ── GLOBAL ── */
html, body, [class*="css"] {
  font-family:'Inter',sans-serif !important;
  background:var(--bg0) !important;
  color:var(--text) !important;
}
.stApp {
  background:var(--bg0) !important;
  background-image:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,0.012) 2px,rgba(0,212,255,0.012) 4px) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#040c14 0%,#06111c 100%) !important;
  border-right:1px solid var(--border2) !important;
  box-shadow:4px 0 30px rgba(0,212,255,0.08) !important;
}
[data-testid="stSidebar"] * { color:var(--text) !important; }
[data-testid="stSidebar"] > div:first-child { background:transparent !important; padding-top:0.5rem !important; }
[data-testid="stSidebar"] .stFileUploader {
  border:1px dashed var(--border2) !important;
  border-radius:8px !important;
  background:var(--bg2) !important;
  padding:8px !important;
}
[data-testid="stSidebar"] .stFileUploader > div { background:transparent !important; }
[data-testid="stSidebar"] section { background:transparent !important; }

/* ── HEADINGS ── */
h1 {
  font-family:'Orbitron',monospace !important;
  color:var(--teal) !important;
  font-size:2rem !important;
  font-weight:900 !important;
  letter-spacing:0.12em !important;
  text-shadow:0 0 30px rgba(0,212,255,0.6),0 0 60px rgba(0,212,255,0.2) !important;
  animation:neon-flicker 8s infinite !important;
}
h2 {
  font-family:'Rajdhani',sans-serif !important;
  color:var(--teal) !important;
  font-size:1.35rem !important;
  font-weight:700 !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  border-bottom:1px solid var(--border2) !important;
  padding-bottom:6px !important;
  margin-top:1.5rem !important;
}
h3 { font-family:'Rajdhani',sans-serif !important; color:var(--text) !important; font-weight:600 !important; }

/* ── CUSTOM KPI CARDS (replaces st.metric) ── */
.kpi-grid {
  display:grid;
  grid-template-columns:repeat(5,1fr);
  gap:14px;
  margin:16px 0 8px;
}
.kpi-card {
  background:var(--card);
  border:1px solid var(--border2);
  border-top:2px solid var(--teal);
  border-radius:12px;
  padding:16px 14px 14px;
  backdrop-filter:blur(12px);
  box-shadow:var(--gT),inset 0 1px 0 rgba(255,255,255,0.05);
  transition:transform .2s,box-shadow .2s;
  animation:card-appear 0.6s ease both;
  text-align:left;
}
.kpi-card:hover {
  transform:translateY(-4px);
  box-shadow:0 0 36px rgba(0,212,255,0.55),inset 0 1px 0 rgba(255,255,255,0.07);
}
.kpi-label {
  font-family:'Share Tech Mono',monospace;
  font-size:0.68rem;
  color:var(--dim);
  letter-spacing:0.1em;
  text-transform:uppercase;
  line-height:1.4;
  margin-bottom:8px;
  word-break:break-word;
  white-space:normal;
}
.kpi-value {
  font-family:'Orbitron',monospace;
  font-size:1.55rem;
  font-weight:700;
  color:var(--teal);
  line-height:1;
  text-shadow:0 0 12px rgba(0,212,255,0.4);
  animation:value-pop 0.5s ease both;
}

/* ── MODEL METRIC CARDS (5-col in tab 2) ── */
.metric-grid {
  display:grid;
  grid-template-columns:repeat(5,1fr);
  gap:12px;
  margin:12px 0 20px;
}
.metric-card {
  background:var(--card);
  border:1px solid var(--border2);
  border-top:2px solid var(--red);
  border-radius:10px;
  padding:14px 12px;
  text-align:left;
  animation:card-appear 0.5s ease both;
  box-shadow:var(--gR);
}
.metric-card:hover { transform:translateY(-3px); box-shadow:0 0 28px rgba(255,45,85,0.5); }
.metric-card .kpi-label { color:var(--dim); }
.metric-card .kpi-value { font-size:1.3rem; color:var(--red); text-shadow:0 0 10px rgba(255,45,85,0.4); }

/* ── PREDICT MINI CARDS (3-col) ── */
.pred-mini-grid {
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:14px;
  margin:16px 0;
}
.pred-mini-card {
  background:var(--card);
  border:1px solid var(--border2);
  border-top:2px solid var(--gold);
  border-radius:10px;
  padding:14px 12px;
  animation:card-appear 0.5s ease both;
}
.pred-mini-card .kpi-label { color:var(--dim); }
.pred-mini-card .kpi-value { font-size:1.25rem; color:var(--gold); }

/* ── TABS ── */
[data-baseweb="tab-list"] {
  background:var(--bg1) !important;
  border-bottom:1px solid var(--border2) !important;
  gap:2px !important; padding:0 8px !important;
}
[data-baseweb="tab"] {
  font-family:'Rajdhani',sans-serif !important;
  font-weight:700 !important; font-size:0.88rem !important;
  letter-spacing:0.07em !important; text-transform:uppercase !important;
  color:var(--dim) !important;
  border-radius:6px 6px 0 0 !important;
  padding:10px 22px !important; transition:all .2s !important;
}
[data-baseweb="tab"]:hover { color:var(--text) !important; background:var(--bg3) !important; }
[data-baseweb="tab"][aria-selected="true"] {
  color:var(--teal) !important; background:var(--bg2) !important;
  border-top:2px solid var(--teal) !important;
  box-shadow:0 -4px 20px rgba(0,212,255,0.2) !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background:linear-gradient(135deg,var(--red2),var(--red)) !important;
  color:#fff !important; border:none !important; border-radius:6px !important;
  font-family:'Rajdhani',sans-serif !important;
  font-weight:700 !important; font-size:0.95rem !important;
  letter-spacing:0.1em !important; text-transform:uppercase !important;
  padding:10px 28px !important; box-shadow:var(--gR) !important;
  transition:all .25s !important;
}
.stButton > button:hover { transform:translateY(-2px) scale(1.02) !important; box-shadow:0 0 36px rgba(255,45,85,0.75) !important; }

/* ── INPUTS ── */
[data-baseweb="select"] > div {
  background:var(--bg3) !important; border-color:var(--border2) !important;
  color:var(--text) !important; border-radius:6px !important;
}
.stNumberInput input, .stTextInput input {
  background:var(--bg3) !important; border:1px solid var(--border2) !important;
  color:var(--text) !important; border-radius:6px !important;
  font-family:'Share Tech Mono',monospace !important;
}
label { color:var(--dim) !important; font-size:0.8rem !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div { background:var(--teal) !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  background:var(--bg2) !important; border:1px solid var(--border2) !important;
  border-radius:8px !important; overflow:hidden !important;
}

/* ── CODE BLOCK ── */
.stCodeBlock { background:var(--bg2) !important; border:1px solid var(--border) !important; border-radius:8px !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] {
  background:var(--bg3) !important; border-left:3px solid var(--gold) !important;
  border-radius:6px !important; color:var(--text) !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color:var(--teal) !important; }

/* ── ANIMATIONS ── */
@keyframes pulse-red {
  0%,100% { box-shadow:0 0 16px rgba(255,45,85,0.4); }
  50%      { box-shadow:0 0 40px rgba(255,45,85,0.9); }
}
@keyframes pulse-teal {
  0%,100% { box-shadow:0 0 16px rgba(0,212,255,0.4); }
  50%      { box-shadow:0 0 40px rgba(0,212,255,0.8); }
}
@keyframes slide-in {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes scan {
  0%   { top:-2px; }
  100% { top:100%; }
}
@keyframes rotate-ring {
  from { transform:rotate(0deg); }
  to   { transform:rotate(360deg); }
}
@keyframes card-appear {
  from { opacity:0; transform:scale(0.93) translateY(10px); }
  to   { opacity:1; transform:scale(1) translateY(0); }
}
@keyframes neon-flicker {
  0%,19%,21%,23%,25%,54%,56%,100% {
    text-shadow:0 0 30px rgba(0,212,255,0.6),0 0 60px rgba(0,212,255,0.3);
  }
  20%,24%,55% { text-shadow:none; }
}
@keyframes float-particle {
  0%   { transform:translateY(0px) translateX(0px); opacity:0.4; }
  33%  { transform:translateY(-18px) translateX(8px); opacity:0.8; }
  66%  { transform:translateY(-8px) translateX(-6px); opacity:0.5; }
  100% { transform:translateY(0px) translateX(0px); opacity:0.4; }
}
@keyframes bar-grow {
  from { width:0% !important; }
}
@keyframes value-pop {
  0%   { transform:scale(0.8); opacity:0; }
  70%  { transform:scale(1.08); }
  100% { transform:scale(1); opacity:1; }
}
@keyframes hbmove {
  0%   { background-position:100% 0; }
  100% { background-position:-100% 0; }
}

/* ── CUSTOM COMPONENTS ── */
.pred-danger {
  position:relative; overflow:hidden;
  background:linear-gradient(135deg,#140008,#260010);
  border:1.5px solid var(--red); border-radius:12px;
  padding:28px 24px; text-align:center;
  animation:pulse-red 2s ease-in-out infinite; margin:16px 0;
}
.pred-safe {
  position:relative; overflow:hidden;
  background:linear-gradient(135deg,#001520,#002030);
  border:1.5px solid var(--teal); border-radius:12px;
  padding:28px 24px; text-align:center;
  animation:pulse-teal 2s ease-in-out infinite; margin:16px 0;
}
.pred-scan-line {
  position:absolute; left:0; width:100%; height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,0.6),transparent);
  animation:scan 2.5s linear infinite;
}
.pred-label { font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:900; letter-spacing:0.1em; margin-bottom:8px; }
.pred-prob  { font-family:'Share Tech Mono',monospace; font-size:1rem; color:#88aabb; margin-top:6px; }
.pred-advice { font-size:0.85rem; color:#7a9aaa; margin-top:10px; letter-spacing:0.04em; }

.hb-line {
  width:100%; height:2px;
  background:linear-gradient(90deg,var(--bg0) 0%,var(--red) 20%,var(--bg0) 22%,var(--bg0) 30%,var(--red) 36%,var(--bg0) 37%,#fff 39%,var(--red) 43%,var(--bg0) 46%,var(--red) 52%,var(--bg0) 55%,var(--bg0) 100%);
  margin:12px 0 20px; border-radius:1px;
  animation:hbmove 3s linear infinite; background-size:300% 100%;
}
.glass-card {
  background:rgba(10,26,40,0.7); border:1px solid var(--border2);
  border-radius:12px; padding:20px 22px; backdrop-filter:blur(16px);
  box-shadow:0 4px 30px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.04);
  transition:transform .25s,box-shadow .25s;
}
.glass-card:hover { transform:translateY(-4px); box-shadow:0 8px 40px rgba(0,0,0,0.6),var(--gT); }
.badge { display:inline-block; font-family:'Share Tech Mono',monospace; font-size:0.7rem; letter-spacing:0.1em; padding:3px 10px; border-radius:20px; margin:2px; }
.badge-red  { background:rgba(255,45,85,0.15);  color:var(--red);  border:1px solid rgba(255,45,85,0.35); }
.badge-teal { background:rgba(0,212,255,0.12);  color:var(--teal); border:1px solid rgba(0,212,255,0.3); }
.badge-gold { background:rgba(255,214,10,0.12); color:var(--gold); border:1px solid rgba(255,214,10,0.3); }
.section-enter { animation:slide-in .5s ease both; }
.ranking-row {
  display:flex; align-items:center; gap:12px;
  padding:10px 14px; margin:6px 0;
  background:var(--bg2); border:1px solid var(--border);
  border-radius:8px; transition:all .2s;
}
.ranking-row:hover { border-color:var(--border2); background:var(--bg3); }
.rank-num { font-family:'Orbitron',monospace; font-size:0.75rem; font-weight:700; color:var(--dim); min-width:22px; }
.rank-bar-bg  { flex:1; height:6px; background:var(--bg3); border-radius:3px; }
.rank-bar-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,var(--teal2),var(--teal)); animation:bar-grow 0.8s ease both; }
.rank-val { font-family:'Share Tech Mono',monospace; font-size:0.82rem; color:var(--teal); min-width:48px; text-align:right; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#060f18', 'axes.facecolor':  '#0a1a26',
    'axes.edgecolor':   '#0e2233', 'axes.labelcolor': '#d4eef7',
    'xtick.color':      '#4a7a90', 'ytick.color':     '#4a7a90',
    'text.color':       '#d4eef7', 'grid.color':      '#0e2233',
    'grid.linestyle':   '--',      'axes.grid':       True,
    'axes.spines.top':  False,     'axes.spines.right':False,
    'figure.dpi':       130,
})
RED  = '#ff2d55'
TEAL = '#00d4ff'
GOLD = '#ffd60a'


# ─────────────────────────────────────────────
#  HELPER: custom KPI card
# ─────────────────────────────────────────────
def kpi_card(label, value, card_class="kpi-card", delay=0):
    return f"""
    <div class="{card_class}" style="animation-delay:{delay:.1f}s;">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="animation-delay:{delay+0.1:.1f}s;">{value}</div>
    </div>"""


# ─────────────────────────────────────────────
#  DATA & MODEL HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(values_file, labels_file):
    values = pd.read_csv(values_file)
    labels = pd.read_csv(labels_file)
    df = values.merge(labels, on="patient_id")
    df.drop_duplicates(inplace=True)
    num_f = df.select_dtypes(include=np.number).columns
    cat_f = df.select_dtypes(include='object').columns
    for c in num_f: df[c] = df[c].fillna(df[c].median())
    for c in cat_f: df[c] = df[c].fillna(df[c].mode()[0])
    df['age_group'] = pd.cut(df['age'], bins=[0,40,50,60,70,120],
                             labels=['Below 40','40-50','51-60','61-70','71+'])
    df['chol_age_ratio'] = df['serum_cholesterol_mg_per_dl'] / df['age']
    return df


@st.cache_resource(show_spinner=False)
def train_pipeline(_df, _id):
    X = _df.drop(['patient_id','heart_disease_present'], axis=1, errors='ignore')
    y = _df['heart_disease_present']
    X = pd.get_dummies(X, drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr)
    Xte_sc = scaler.transform(X_te)
    try:
        from imblearn.over_sampling import SMOTE
        ratio = y_tr.value_counts(normalize=True).min()
        if ratio < 0.4:
            sm = SMOTE(random_state=42)
            Xtr_sc, y_tr = sm.fit_resample(Xtr_sc, y_tr)
    except ImportError:
        pass
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(random_state=42),
        'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42),
    }
    results = []
    for name, m in models.items():
        m.fit(Xtr_sc, y_tr)
        yp  = m.predict(Xte_sc)
        ypr = m.predict_proba(Xte_sc)[:,1]
        results.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_te, yp), 4),
            'Precision': round(precision_score(y_te, yp), 4),
            'Recall':    round(recall_score(y_te, yp), 4),
            'F1':        round(f1_score(y_te, yp), 4),
            'ROC AUC':   round(roc_auc_score(y_te, ypr), 4),
        })
    rf = RandomForestClassifier(random_state=42)
    rs = RandomizedSearchCV(rf, {
        'n_estimators':      [100,200,300],
        'max_depth':         [5,10,15,None],
        'min_samples_split': [2,5,10],
        'min_samples_leaf':  [1,2,4],
    }, n_iter=10, scoring='f1', cv=5, random_state=42, n_jobs=-1)
    rs.fit(Xtr_sc, y_tr)
    best = rs.best_estimator_
    yp  = best.predict(Xte_sc)
    ypr = best.predict_proba(Xte_sc)[:,1]
    fi  = pd.DataFrame({'Feature': X.columns, 'Importance': best.feature_importances_})\
            .sort_values('Importance', ascending=False).reset_index(drop=True)
    return (best, scaler, X.columns.tolist(),
            X_te, Xte_sc, y_te, yp, ypr,
            pd.DataFrame(results), fi)


def single_predict(model, scaler, fcols, inp):
    row = pd.DataFrame([inp])
    row = pd.get_dummies(row, drop_first=True)
    row = row.reindex(columns=fcols, fill_value=0)
    prob = model.predict_proba(scaler.transform(row))[0][1]
    return int(prob >= 0.4), prob


# ─────────────────────────────────────────────
#  ANIMATED HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-enter" style="text-align:center;padding:18px 0 8px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;">
    <div style="position:absolute;top:10%;left:8%;width:4px;height:4px;background:#00d4ff;border-radius:50%;opacity:0.5;animation:float-particle 4s ease-in-out infinite;"></div>
    <div style="position:absolute;top:30%;left:15%;width:3px;height:3px;background:#ff2d55;border-radius:50%;opacity:0.4;animation:float-particle 5.5s ease-in-out infinite 1s;"></div>
    <div style="position:absolute;top:60%;left:5%;width:5px;height:5px;background:#00d4ff;border-radius:50%;opacity:0.3;animation:float-particle 3.8s ease-in-out infinite 0.5s;"></div>
    <div style="position:absolute;top:20%;right:10%;width:4px;height:4px;background:#ffd60a;border-radius:50%;opacity:0.45;animation:float-particle 4.5s ease-in-out infinite 2s;"></div>
    <div style="position:absolute;top:50%;right:8%;width:3px;height:3px;background:#00d4ff;border-radius:50%;opacity:0.4;animation:float-particle 6s ease-in-out infinite 0.3s;"></div>
    <div style="position:absolute;top:75%;right:18%;width:4px;height:4px;background:#ff2d55;border-radius:50%;opacity:0.35;animation:float-particle 5s ease-in-out infinite 1.5s;"></div>
  </div>
  <div style="position:relative;z-index:1;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;letter-spacing:0.25em;color:#4a7a90;margin-bottom:8px;animation:slide-in 0.5s ease both;">
      HEART DISEASE PREDICTION · AI DIAGNOSTIC SYSTEM · v2.0
    </div>
    <h1 style="margin:0;">🫀 HEART DISEASE PREDICTION</h1>
    <div style="font-family:'Rajdhani',sans-serif;color:#4a7a90;letter-spacing:0.15em;font-size:0.9rem;margin-top:6px;animation:slide-in 0.7s ease both;">
      CLINICAL MACHINE LEARNING DASHBOARD
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hb-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.88rem;color:#00d4ff;letter-spacing:0.1em;">
        Heart Disease Prediction
      </div>
      <div style="font-size:0.72rem;color:#4a7a90;margin-top:4px;letter-spacing:0.1em;font-family:'Share Tech Mono',monospace;">
        DIAGNOSTIC MODULE
      </div>
    </div>
    <hr style="border-color:rgba(0,212,255,0.2);margin:8px 0 16px;">
    """, unsafe_allow_html=True)
    st.markdown("**Upload Patient Data**")
    values_file = st.file_uploader("VALUES.CSV", type="csv", key="val")
    labels_file = st.file_uploader("LABELS.CSV", type="csv", key="lab")
    st.markdown("<hr style='border-color:rgba(0,212,255,0.2);margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#4a7a90;line-height:1.9;">
      <div><span style="color:#00d4ff;">MODEL</span> &nbsp; Random Forest (tuned)</div>
      <div><span style="color:#00d4ff;">THRESHOLD</span> &nbsp; 0.40 (recall-optimised)</div>
      <div><span style="color:#00d4ff;">VALIDATION</span> &nbsp; 5-fold CV</div>
      <div><span style="color:#00d4ff;">IMBALANCE</span> &nbsp; SMOTE (auto)</div>
    </div>
    """, unsafe_allow_html=True)

if not values_file or not labels_file:
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:48px 24px;margin-top:40px;">
      <div style="font-size:3rem;margin-bottom:16px;">📂</div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.2rem;color:#00d4ff;letter-spacing:0.08em;margin-bottom:8px;">
        AWAITING DATA UPLOAD
      </div>
      <div style="color:#4a7a90;font-size:0.88rem;">
        Upload <code>values.csv</code> and <code>labels.csv</code> in the sidebar to initialise the system.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  LOAD + TRAIN
# ─────────────────────────────────────────────
with st.spinner("Initialising diagnostic system..."):
    df = load_data(values_file, labels_file)

with st.spinner("Training models — please wait..."):
    (best_rf, scaler, fcols,
     X_te, Xte_sc, y_te, y_pred, y_prob,
     results_df, fi_df) = train_pipeline(df, id(df))

# ─────────────────────────────────────────────
#  TOP KPIs — fully custom HTML cards (no st.metric)
# ─────────────────────────────────────────────
total    = len(df)
alive    = int((df['heart_disease_present']==0).sum())
disease  = int((df['heart_disease_present']==1).sum())
pct      = round(disease/total*100, 1)
model_f1 = round(f1_score(y_te, y_pred)*100, 1)

kpi_items = [
    ("Total Patients",   f"{total:,}"),
    ("No Heart Disease", f"{alive:,}"),
    ("Heart Disease",    f"{disease:,}"),
    ("Prevalence",       f"{pct}%"),
    ("Model F1 Score",   f"{model_f1}%"),
]
cards_html = '<div class="kpi-grid">'
for i, (lbl, val) in enumerate(kpi_items):
    cards_html += kpi_card(lbl, val, "kpi-card", delay=i*0.1)
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("<hr style='border-color:rgba(0,212,255,0.12);margin:8px 0 16px;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["📊  Dashboard", "🤖  Model Results", "🔬  Predict Patient", "📈  Feature Analysis"])


# ══════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ══════════════════════════════════════════════
with tabs[0]:
    st.header("Patient Analytics Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cases by Age Group")
        age_g = df.groupby(['age_group','heart_disease_present']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6,3.8))
        x = np.arange(len(age_g)); w = 0.38
        b1 = ax.bar(x-w/2, age_g.get(0, pd.Series([0]*len(age_g))), width=w, color=TEAL, alpha=0.85, label='No Disease')
        b2 = ax.bar(x+w/2, age_g.get(1, pd.Series([0]*len(age_g))), width=w, color=RED,  alpha=0.85, label='Disease')
        ax.bar_label(b1, fontsize=8, color='#d4eef7', padding=2)
        ax.bar_label(b2, fontsize=8, color='#d4eef7', padding=2)
        ax.set_xticks(x); ax.set_xticklabels(age_g.index.astype(str), fontsize=9)
        ax.legend(facecolor='#0a1a26', edgecolor='#0e2233', labelcolor='#d4eef7', fontsize=9)
        ax.set_xlabel('Age Group'); ax.set_ylabel('Count')
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Avg Serum Cholesterol by Age Group")
        chol = df.groupby('age_group')['serum_cholesterol_mg_per_dl'].mean()
        fig, ax = plt.subplots(figsize=(6,3.8))
        colors = [TEAL if v < chol.mean() else RED for v in chol.values]
        bars = ax.bar(chol.index.astype(str), chol.values, color=colors, alpha=0.88)
        ax.bar_label(bars, fmt='%.0f', color='#d4eef7', fontsize=9, padding=3)
        ax.axhline(chol.mean(), color=GOLD, linestyle='--', lw=1.2, label=f'Avg: {chol.mean():.0f}')
        ax.legend(facecolor='#0a1a26', edgecolor='#0e2233', labelcolor='#d4eef7', fontsize=9)
        ax.set_xlabel('Age Group'); ax.set_ylabel('Avg Cholesterol (mg/dL)')
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(5,3.8))
        vals = df['heart_disease_present'].value_counts()
        wedges, texts, autotexts = ax.pie(vals.values, labels=['No Disease','Disease'],
            colors=[TEAL,RED], autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='#060f18', linewidth=2.5, width=0.65),
            textprops={'color':'#d4eef7','fontsize':10})
        for at in autotexts: at.set_fontweight('bold')
        ax.set_facecolor('#060f18')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(5,3.8))
        n, bins, patches = ax.hist(df['age'], bins=22, edgecolor='#060f18', lw=0.8)
        norm = plt.Normalize(n.min(), n.max())
        for patch, val in zip(patches, n):
            r = norm(val)
            patch.set_facecolor((r*1.0, (1-r)*0.83, (1-r)*0.33+r*1.0, 0.88))
        ax.set_xlabel('Age'); ax.set_ylabel('Count')
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Correlation Heatmap")
    num_c = df.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(figsize=(13,7))
    mask = np.triu(np.ones_like(df[num_c].corr(), dtype=bool))
    sns.heatmap(df[num_c].corr(), mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', ax=ax, linecolor='#060f18', linewidths=0.5,
                annot_kws={'size':8}, vmin=-1, vmax=1, cbar_kws={'shrink':.7})
    ax.set_title('Feature Correlations (lower triangle)', color='#d4eef7', pad=12, fontsize=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
#  TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════
with tabs[1]:
    st.header("Model Evaluation — Tuned Random Forest")

    acc = accuracy_score(y_te, y_pred)
    pre = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1v = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    # Custom HTML metric cards — labels never truncate
    m_items = [
        ("Accuracy",  f"{acc:.2%}"),
        ("Precision", f"{pre:.2%}"),
        ("Recall",    f"{rec:.2%}"),
        ("F1 Score",  f"{f1v:.2%}"),
        ("ROC AUC",   f"{auc:.2%}"),
    ]
    m_html = '<div class="metric-grid">'
    for i, (lbl, val) in enumerate(m_items):
        m_html += kpi_card(lbl, val, "metric-card", delay=i*0.08)
    m_html += '</div>'
    st.markdown(m_html, unsafe_allow_html=True)

    # Model comparison bars
    st.subheader("All 5 Models — F1 Comparison")
    rd = results_df.sort_values('F1', ascending=False).reset_index(drop=True)
    bar_html = ""
    for i, row in rd.iterrows():
        pct_bar = int(row['F1'] * 100)
        color   = TEAL if i == 0 else '#4a7a90'
        badge   = '<span class="badge badge-teal">BEST</span>' if i == 0 else ''
        bar_html += f"""
        <div class="ranking-row" style="animation:slide-in .4s ease {i*0.08:.2f}s both;">
          <div class="rank-num">#{i+1}</div>
          <div style="flex:2;font-family:'Rajdhani',sans-serif;font-size:0.92rem;color:#d4eef7;">
            {row['Model']} {badge}
          </div>
          <div class="rank-bar-bg" style="flex:3;">
            <div class="rank-bar-fill" style="width:{pct_bar}%;background:linear-gradient(90deg,#0092b3,{color});"></div>
          </div>
          <div class="rank-val">{row['F1']:.4f}</div>
        </div>"""
    st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Dark HTML table for model results
    cols_show = ['Model','Accuracy','Precision','Recall','F1','ROC AUC']
    rd_cols = [c for c in cols_show if c in rd.columns]
    rows_tbl2 = ""
    for i, row in rd.iterrows():
        badge_b = '<span style="background:rgba(0,212,255,0.15);color:#00d4ff;border:1px solid rgba(0,212,255,0.3);border-radius:20px;font-size:0.65rem;padding:2px 8px;margin-left:6px;font-family:\'Share Tech Mono\',monospace;">BEST</span>' if i == 0 else ''
        rows_tbl2 += '<tr>'
        for j, c in enumerate(rd_cols):
            val = row[c]
            style_td = "color:#00d4ff;font-family:'Orbitron',monospace;font-size:0.78rem;" if j > 0 else "color:#d4eef7;font-family:'Rajdhani',sans-serif;font-weight:600;"
            display = f"{val:.4f}" if isinstance(val, float) else str(val)
            extra = badge_b if c == 'Model' else ''
            rows_tbl2 += f'<td style="padding:9px 12px;border-bottom:1px solid rgba(0,212,255,0.08);{style_td}">{display}{extra}</td>'
        rows_tbl2 += '</tr>'
    hdr = "".join([f'<th style="padding:10px 12px;text-align:left;font-family:\'Share Tech Mono\',monospace;color:#00d4ff;font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;border-bottom:1px solid rgba(0,212,255,0.35);background:#0e2233;">{c}</th>' for c in rd_cols])
    st.markdown(f"""
    <div style="background:#0a1a26;border:1px solid rgba(0,212,255,0.35);border-radius:10px;overflow:hidden;margin-bottom:16px;animation:slide-in 0.5s ease both;">
      <table style="width:100%;border-collapse:collapse;"><thead><tr>{hdr}</tr></thead><tbody>{rows_tbl2}</tbody></table>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease','Disease'],
                    yticklabels=['No Disease','Disease'],
                    linecolor='#060f18', linewidths=0.8,
                    annot_kws={'size':14,'weight':'bold'})
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.fill_between(fpr, tpr, alpha=0.12, color=RED)
        ax.plot(fpr, tpr, color=RED, lw=2.5, label=f'AUC = {auc:.3f}')
        ax.plot([0,1],[0,1],'--', color='#4a7a90', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(facecolor='#0a1a26', edgecolor='#0e2233', labelcolor='#d4eef7')
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Classification Report")
    rpt = classification_report(y_te, y_pred, target_names=['No Disease','Disease'])
    st.code(rpt, language='text')


# ══════════════════════════════════════════════
#  TAB 3 — PREDICT PATIENT
# ══════════════════════════════════════════════
with tabs[2]:
    st.header("Single Patient Risk Assessment")
    st.markdown("""
    <div class="badge badge-gold">THRESHOLD 0.40</div>
    <div class="badge badge-teal">RECALL-OPTIMISED</div>
    <div class="badge badge-red">CLINICAL USE ONLY</div>
    <br><br>
    """, unsafe_allow_html=True)

    num_inp = [c for c in df.select_dtypes(include=np.number).columns
               if c not in ('patient_id','heart_disease_present')]
    cat_inp = [c for c in df.select_dtypes(include='object').columns
               if c not in ('patient_id',)]

    inp = {}
    with st.form("pred_form"):
        st.markdown("#### Numerical Features")
        ncols = st.columns(3)
        for i, col in enumerate(num_inp):
            mn = float(df[col].min()); mx = float(df[col].max())
            med = float(df[col].median())
            inp[col] = ncols[i%3].number_input(
                col.replace('_',' ').title(), min_value=mn, max_value=mx, value=med)
        if cat_inp:
            st.markdown("#### Categorical Features")
            ccols = st.columns(3)
            for i, col in enumerate(cat_inp):
                opts = sorted(df[col].dropna().unique().tolist())
                inp[col] = ccols[i%3].selectbox(col.replace('_',' ').title(), opts)
        submitted = st.form_submit_button("🔍  RUN DIAGNOSTIC SCAN")

    if submitted:
        if 'age' in inp and 'serum_cholesterol_mg_per_dl' in inp:
            inp['chol_age_ratio'] = inp['serum_cholesterol_mg_per_dl'] / max(inp['age'],1)
        if 'age' in inp:
            a = inp['age']
            ag = ('Below 40' if a<40 else '40-50' if a<=50 else
                  '51-60' if a<=60 else '61-70' if a<=70 else '71+')
            for lbl in ['Below 40','40-50','51-60','61-70']:
                inp[f'age_group_{lbl}'] = int(ag==lbl)

        pred, prob = single_predict(best_rf, scaler, fcols, inp)
        pct_p = prob * 100

        if pred == 1:
            st.markdown(f"""
            <div class="pred-danger">
              <div class="pred-scan-line"></div>
              <div class="pred-label" style="color:{RED};">⚠ HIGH RISK — HEART DISEASE DETECTED</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;color:{RED};margin:12px 0;text-shadow:0 0 30px rgba(255,45,85,0.8);">
                {pct_p:.1f}%
              </div>
              <div class="pred-prob">Risk Probability &nbsp;|&nbsp; Threshold: 0.40</div>
              <div class="pred-advice">Immediate clinical review recommended. Refer to cardiologist.<br>Consider ECG, stress test, and lipid panel.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-safe">
              <div class="pred-scan-line" style="background:linear-gradient(90deg,transparent,rgba(0,212,255,0.5),transparent);"></div>
              <div class="pred-label" style="color:{TEAL};">✓ LOW RISK — NO HEART DISEASE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;color:{TEAL};margin:12px 0;text-shadow:0 0 30px rgba(0,212,255,0.8);">
                {pct_p:.1f}%
              </div>
              <div class="pred-prob">Risk Probability &nbsp;|&nbsp; Threshold: 0.40</div>
              <div class="pred-advice">Continue routine monitoring and preventive care.<br>Maintain healthy lifestyle. Schedule regular check-ups.</div>
            </div>""", unsafe_allow_html=True)

        # Mini breakdown — custom HTML cards
        p_items = [
            ("Prediction", "HIGH RISK" if pred else "LOW RISK"),
            ("Risk Score",  f"{pct_p:.1f}%"),
            ("Confidence",  f"{abs(prob-0.5)*200:.1f}%"),
        ]
        p_html = '<div class="pred-mini-grid">'
        for lbl, val in p_items:
            p_html += f'<div class="pred-mini-card"><div class="kpi-label">{lbl}</div><div class="kpi-value" style="color:{"#ff2d55" if pred else "#00d4ff"};">{val}</div></div>'
        p_html += '</div>'
        st.markdown(p_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 4 — FEATURE ANALYSIS
# ══════════════════════════════════════════════
with tabs[3]:
    st.header("Feature Importance Analysis")
    top_n = st.slider("Show top N features", 5, min(25, len(fi_df)), 15)
    top = fi_df.head(top_n)

    st.subheader("Feature Ranking")
    rows_html = ""
    max_imp = float(fi_df['Importance'].max())
    for i, row in top.iterrows():
        pct_bar = int((row['Importance']/max_imp)*100)
        hue = f"hsl({int(180 - 180*(row['Importance']/max_imp))}, 80%, 60%)"
        rows_html += f"""
        <div class="ranking-row" style="animation:slide-in .35s ease {i*0.04:.2f}s both;">
          <div class="rank-num">#{i+1}</div>
          <div style="flex:3;font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#d4eef7;word-break:break-word;">
            {row['Feature']}
          </div>
          <div class="rank-bar-bg" style="flex:4;">
            <div class="rank-bar-fill" style="width:{pct_bar}%;background:linear-gradient(90deg,#004a60,{hue});"></div>
          </div>
          <div class="rank-val">{row['Importance']:.4f}</div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Importance Chart")
    fig, ax = plt.subplots(figsize=(10, max(4, top_n*0.42)))
    cmap = plt.cm.RdYlGn_r
    colors_bar = [cmap(i/max(top_n-1,1)) for i in range(len(top))]
    bars = ax.barh(top['Feature'][::-1], top['Importance'][::-1],
                   color=colors_bar[::-1], alpha=0.88, height=0.7)
    ax.bar_label(bars, fmt='%.4f', color='#d4eef7', fontsize=9, padding=4)
    ax.set_xlabel('Importance Score')
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Full Table")
    rows_tbl = ""
    for i, row in fi_df.iterrows():
        rows_tbl += (f'<tr>'
                     f'<td style="color:#4a7a90;padding:8px 12px;border-bottom:1px solid rgba(0,212,255,0.08);">{i}</td>'
                     f'<td style="padding:8px 12px;border-bottom:1px solid rgba(0,212,255,0.08);font-family:\'Share Tech Mono\',monospace;color:#d4eef7;font-size:0.82rem;">{row["Feature"]}</td>'
                     f'<td style="padding:8px 12px;border-bottom:1px solid rgba(0,212,255,0.08);text-align:right;font-family:\'Orbitron\',monospace;color:#00d4ff;font-size:0.8rem;">{row["Importance"]:.4f}</td>'
                     f'</tr>')
    st.markdown(f"""
    <div style="background:#0a1a26;border:1px solid rgba(0,212,255,0.35);border-radius:10px;overflow:hidden;animation:slide-in 0.5s ease both;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#0e2233;">
            <th style="padding:10px 12px;text-align:left;font-family:'Share Tech Mono',monospace;color:#4a7a90;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid rgba(0,212,255,0.35);"></th>
            <th style="padding:10px 12px;text-align:left;font-family:'Share Tech Mono',monospace;color:#00d4ff;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid rgba(0,212,255,0.35);">Feature</th>
            <th style="padding:10px 12px;text-align:right;font-family:'Share Tech Mono',monospace;color:#00d4ff;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid rgba(0,212,255,0.35);">Importance</th>
          </tr>
        </thead>
        <tbody>{rows_tbl}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)