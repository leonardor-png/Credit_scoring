import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats.mstats import winsorize

# --- 1. PREPARAZIONE DATI E MODELLO ---
@st.cache_resource
def train_complex_model():
    np.random.seed(42)
    n_samples = 2000
    
    # Generazione di variabili indipendenti più complesse
    eta = np.random.randint(18, 75, n_samples)
    reddito = np.random.normal(35000, 15000, n_samples)
    dti = np.random.uniform(0.1, 0.7, n_samples) # Debt-to-Income (Rapporto Rata/Reddito)
    utilizzo_credito = np.random.uniform(0, 1, n_samples) # % di carte di credito e fidi utilizzati
    ritardi_passati = np.random.poisson(0.5, n_samples) # Numero di rate pagate in ritardo
    
    # Variabile qualitativa: Tipo di impiego
    contratto = np.random.choice(
        ['Indeterminato', 'Determinato', 'Partita IVA'], 
        n_samples, 
        p=[0.60, 0.25, 0.15]
    )
    
    df = pd.DataFrame({
        'Eta': eta,
        'Reddito': reddito,
        'DTI': dti,
        'Utilizzo_Credito': utilizzo_credito,
        'Ritardi': ritardi_passati,
        'Contratto': contratto
    })
    
    # Pulizia: Winsorizzazione del reddito per limitare l'effetto degli outlier estremi
    df['Reddito'] = winsorize(df['Reddito'], limits=[0.01, 0.05])
    
    # Encoding: Creazione delle variabili Dummy per il tipo di contratto (Drop first per evitare multicollinearità)
    df = pd.get_dummies(df, columns=['Contratto'], drop_first=True)
    
    # Creazione della variabile target (Default) basata su una combinazione lineare con rumore stocastico
    score_latente = (
        (df['DTI'] * 3.5) + 
        (df['Utilizzo_Credito'] * 2.0) + 
        (df['Ritardi'] * 1.8) + 
        (df.get('Contratto_Determinato', 0) * 0.8) + 
        (df.get('Contratto_Partita IVA', 0) * 1.2) - 
        (df['Reddito'] / 40000) -
        (df['Eta'] / 100) +
        np.random.normal(0, 1, n_samples) # Termine di errore
    )
    
    # Trasformiamo lo score in una probabilità (Sigmoide) e generiamo la classificazione binaria
    prob_reale = 1 / (1 + np.exp(-score_latente))
    df['Default'] = (prob_reale > 0.5).astype(int)
    
    X = df.drop(columns=['Default'])
    y = df['Default']
    
    # Addestramento della regressione logistica
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return model, X.columns

modello, colonne_features = train_complex_model()


# --- 2. INTERFACCIA UTENTE (STREAMLIT) ---
st.set_page_config(page_title="Advanced Credit Scoring", page_icon="📊", layout="centered")

st.title("📊 Advanced Credit Scoring App")
st.markdown("""
Questo modello stima la **Probability of Default (PD)** analizzando indicatori finanziari complessi, 
stabilità lavorativa e storico creditizio tramite un modello logit.
""")

st.divider()

# Layout a griglia per un inserimento dati più pulito
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dati Anagrafici e Reddituali")
    input_eta = st.number_input("Età", min_value=18, max_value=90, value=29, step=1)
    input_reddito = st.number_input("Reddito Annuo Netto (€)", min_value=5000, max_value=250000, value=30000, step=1000)
    input_contratto = st.selectbox("Tipologia di Contratto", ['Indeterminato', 'Determinato', 'Partita IVA'])

with col2:
    st.subheader("Indicatori di Rischio (Credit Bureau)")
    input_dti = st.slider("Debt-to-Income (DTI)", min_value=0.0, max_value=1.0, value=0.30, step=0.01, 
                          help="Percentuale del reddito mensile già assorbita da rate o mutui.")
    input_utilizzo = st.slider("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.01,
                               help="Percentuale di plafond delle carte di credito/fidi attualmente in uso.")
    input_ritardi = st.number_input("Ritardi Storici (Ultime rate)", min_value=0, max_value=10, value=0, step=1)

# --- 3. ELABORAZIONE E PREVISIONE ---
if st.button("Calcola Credit Score", type="primary"):
    
    # Dobbiamo replicare la stessa struttura (dummy variables) usata in fase di addestramento
    is_determinato = 1 if input_contratto == 'Determinato' else 0
    is_piva = 1 if input_contratto == 'Partita IVA' else 0
    
    # Creazione del vettore di input con lo stesso ordine delle colonne addestrate
    input_dati = pd.DataFrame([[
        input_eta, 
        input_reddito, 
        input_dti, 
        input_utilizzo, 
        input_ritardi, 
        is_determinato, 
        is_piva
    ]], columns=colonne_features)
    
    # Previsione
    prob_default = modello.predict_proba(input_dati)[0][1]
    score_finale = (1 - prob_default) * 1000 # Score in millesimi (stile FICO)
    
    st.divider()
    st.subheader("Risultato dell'Analisi")
    
    # Metriche in evidenza
    met_col1, met_col2 = st.columns(2)
    met_col1.metric(label="Credit Score (Base 1000)", value=f"{score_finale:.0f}")
    met_col2.metric(label="Probability of Default (PD)", value=f"{prob_default:.2%}")
    
    # Barre di progresso e alert visivi basati sul rating
    st.progress(1 - prob_default)
    
    if prob_default < 0.15:
        st.success("✅ **RATING: A (Rischio Basso)** - Il profilo presenta ottime probabilità di solvibilità. Finanziamento pre-approvabile con spread minimi.")
    elif prob_default < 0.40:
        st.warning("⚠️ **RATING: B (Rischio Moderato)** - Profilo in area di attenzione. Richiesta ulteriore revisione manuale o garanzie aggiuntive.")
    else:
        st.error("❌ **RATING: C (Rischio Alto)** - La probabilità di insolvenza eccede le soglie di tolleranza standard. Finanziamento sconsigliato.")
