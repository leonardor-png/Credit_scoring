import streamlit as st
import pandas as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats.mstats import winsorize

# --- 1. PREPARAZIONE DATI E MODELLO ---
@st.cache_resource # Evita di riaddestrare il modello ad ogni click
def train_model():
    # Generiamo un dataset storico fittizio di 1000 clienti
    np.random.seed(42)
    reddito = np.random.normal(30000, 15000, 1000)
    debito_pregresso = np.random.normal(10000, 8000, 1000)
    
    # Pulizia econometrica: Winsorizzazione al 5% per coda per gestire gli outlier
    reddito_win = winsorize(reddito, limits=[0.05, 0.05])
    debito_win = winsorize(debito_pregresso, limits=[0.05, 0.05])
    
    # Creiamo una variabile target fittizia (1 = Default, 0 = Solvibile)
    # Logica di base: alto debito e basso reddito aumentano le probabilità di default
    score_lineare = (debito_win / 1000) - (reddito_win / 5000) + np.random.normal(0, 2, 1000)
    default = (score_lineare > 0).astype(int)
    
    df = pd.DataFrame({
        'Reddito': reddito_win,
        'Debito': debito_win,
        'Default': default
    })
    
    X = df[['Reddito', 'Debito']]
    y = df['Default']
    
    # Inizializziamo e addestriamo la regressione logistica
    model = LogisticRegression()
    model.fit(X, y)
    
    return model

modello = train_model()

# --- 2. INTERFACCIA UTENTE (STREAMLIT) ---
st.title("🏦 App di Credit Scoring")
st.write("Inserisci i dati del richiedente per stimare la Probability of Default (PD).")

# Creiamo due colonne per l'input utente
col1, col2 = st.columns(2)

with col1:
    input_reddito = st.number_input("Reddito Annuo (€)", min_value=0, max_value=200000, value=30000, step=1000)

with col2:
    input_debito = st.number_input("Debito Pregresso (€)", min_value=0, max_value=200000, value=5000, step=1000)

# Bottone per lanciare la previsione
if st.button("Calcola Score"):
    # Creiamo un DataFrame con il nuovo input
    nuovo_cliente = pd.DataFrame({
        'Reddito': [input_reddito],
        'Debito': [input_debito]
    })
    
    # Il modello restituisce due probabilità: [Prob Non-Default, Prob Default]
    prob_default = modello.predict_proba(nuovo_cliente)[0][1]
    
    # Trasformiamo la probabilità in uno score in centesimi per facilitare la lettura
    score = (1 - prob_default) * 100 
    
    # --- 3. OUTPUT E RISULTATI ---
    st.markdown("---")
    st.subheader("Risultati della Valutazione")
    
    st.write(f"**Probability of Default (PD):** {prob_default:.2%}")
    
    # Assegnazione di un rating visivo
    if prob_default < 0.10:
        st.success(f"✅ Rischio Basso. Credit Score: {score:.0f}/100")
    elif prob_default < 0.40:
        st.warning(f"⚠️ Rischio Medio. Credit Score: {score:.0f}/100")
    else:
        st.error(f"❌ Rischio Alto. Credit Score: {score:.0f}/100")
