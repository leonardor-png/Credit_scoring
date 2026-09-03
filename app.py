import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

# --- 1. SIMULAZIONE DATASET DA TERMINALE FINANZIARIO ---
@st.cache_data
def load_and_prep_data():
    """
    Simula il caricamento di un dataset cross-section/panel estratto da 
    Refinitiv Workspace o Bloomberg, contenente metriche finanziarie ed ESG.
    """
    np.random.seed(42)
    n = 2500
    
    df = pd.DataFrame({
        'Reddito': np.random.normal(45000, 20000, n),
        'DTI': np.random.uniform(0.1, 0.6, n),
        'Ritardi': np.random.poisson(0.3, n),
        'ESG_Pillar_Score': np.random.normal(55, 18, n), # Score da 0 a 100
        'Contratto': np.random.choice(['Indeterminato', 'Determinato', 'Partita IVA'], n, p=[0.65, 0.20, 0.15])
    })
    
    # Rigore econometrico: Winsorizzazione per gestire gli outlier senza distorcere i coefficienti
    df['Reddito'] = winsorize(df['Reddito'], limits=[0.02, 0.05])
    df['ESG_Pillar_Score'] = np.clip(df['ESG_Pillar_Score'], 0, 100)
    
    # Encoding variabili dummy
    df = pd.get_dummies(df, columns=['Contratto'], drop_first=True)
    
    # Generazione variabile target (Probability of Default)
    score_latente = (df['DTI']*4.5) + (df['Ritardi']*2.2) - (df['Reddito']/40000) - (df['ESG_Pillar_Score']/120) + np.random.normal(0, 1, n)
    df['Default'] = (1 / (1 + np.exp(-score_latente)) > 0.5).astype(int)
    
    return df

# --- 2. ADDESTRAMENTO MODELLO ---
@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Default'])
    y = df['Default']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

df_storico = load_and_prep_data()
modello, scaler, features = train_model(df_storico)

# --- 3. PRICING ENGINE (Calcolo Spread e Greenium) ---
def calcola_pricing(prob_default, esg_score):
    """
    Trasforma il rischio in prezzo e applica il Greenium per asset sostenibili.
    """
    tasso_base = 3.50 # Es. Euribor o costo della provvista
    
    # Calcolo dello Spread di rischio basato sulla PD
    if prob_default < 0.05: spread_rischio = 1.00
    elif prob_default < 0.15: spread_rischio = 2.50
    elif prob_default < 0.30: spread_rischio = 4.50
    else: spread_rischio = 7.00
    
    # Calcolo del Greenium (Sconto applicato per alto ESG Score)
    green_discount = 0.0
    if esg_score >= 75:
        green_discount = 0.30 # -30 basis points
    elif esg_score >= 60:
        green_discount = 0.15 # -15 basis points
        
    tasso_finito = tasso_base + spread_rischio - green_discount
    return tasso_finito, spread_rischio, green_discount

# --- 4. INTERFACCIA WEB ---
st.set_page_config(page_title="Risk & Pricing Model", layout="wide")
st.title("🏦 Modello di Credit Scoring & Sostenibilità")
st.markdown("Algoritmo di valutazione del rischio con integrazione delle dinamiche di **Greenium** nel pricing del debito.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dati Finanziari Richiedente")
    reddito = st.number_input("Reddito Annuo Netto (€)", 10000, 200000, 35000, step=1000)
    dti = st.slider("Debt-to-Income (DTI)", 0.0, 1.0, 0.30, 0.01)
    ritardi = st.number_input("Ritardi Storici", 0, 10, 0)
    contratto = st.selectbox("Tipologia di Contratto", ['Indeterminato', 'Determinato', 'Partita IVA'])

with col2:
    st.subheader("Metriche di Sostenibilità")
    esg_score = st.slider("ESG Pillar Score (0-100)", 0, 100, 50, 1, help="Score estratto da database provider")
    st.info("💡 Un punteggio ESG > 60 innesca il fenomeno del Greenium, riducendo lo spread richiesto.")

if st.button("Valuta Merito Creditizio e Pricing", type="primary"):
    
    # Preparazione input
    is_det = 1 if contratto == 'Determinato' else 0
    is_piva = 1 if contratto == 'Partita IVA' else 0
    
    input_data = pd.DataFrame([[reddito, dti, ritardi, esg_score, is_det, is_piva]], columns=features)
    prob_default = modello.predict_proba(scaler.transform(input_data))[0][1]
    
    tasso, spread, greenium = calcola_pricing(prob_default, esg_score)
    
    st.divider()
    st.subheader("Esito Analisi e Struttura del Pricing")
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Probability of Default", f"{prob_default:.2%}")
    r2.metric("Tasso Base (Mercato)", "3.50%")
    r3.metric("Spread Rischio", f"+{spread:.2f}%")
    r4.metric("Effetto Greenium", f"-{greenium:.2f}%" if greenium > 0 else "0.00%")
    
    st.success(f"### TASSO FINITO PROPOSTO: {tasso:.2f}%")
