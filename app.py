import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

# --- 1. PREPARAZIONE DATI E MODELLO ---
@st.cache_resource
def train_enterprise_model():
    np.random.seed(42)
    n_samples = 3000
    
    # Generazione Variabili
    eta = np.random.randint(18, 75, n_samples)
    reddito = np.random.normal(35000, 15000, n_samples)
    dti = np.random.uniform(0.1, 0.7, n_samples)
    utilizzo_credito = np.random.uniform(0, 1, n_samples)
    ritardi = np.random.poisson(0.5, n_samples)
    
    # Integrazione Parametri ESG (Es. Refinitiv Pillar Scores o rating interni)
    esg_score = np.random.normal(50, 15, n_samples) 
    
    contratto = np.random.choice(['Indeterminato', 'Determinato', 'Partita IVA'], n_samples, p=[0.60, 0.25, 0.15])
    
    df = pd.DataFrame({
        'Eta': eta,
        'Reddito': reddito,
        'DTI': dti,
        'Utilizzo_Credito': utilizzo_credito,
        'Ritardi': ritardi,
        'ESG_Score': esg_score,
        'Contratto': contratto
    })
    
    # Winsorizzazione per code distributive (gestione outlier econometrici)
    df['Reddito'] = winsorize(df['Reddito'], limits=[0.01, 0.05])
    df['ESG_Score'] = np.clip(df['ESG_Score'], 0, 100) # Tronchiamo tra 0 e 100
    
    # Encoding Variabili Categoriche
    df = pd.get_dummies(df, columns=['Contratto'], drop_first=True)
    
    # Equazione Latente (Un alto punteggio ESG riduce marginalmente il rischio, simulando l'effetto greenium)
    score_latente = (
        (df['DTI'] * 4.0) + 
        (df['Utilizzo_Credito'] * 2.5) + 
        (df['Ritardi'] * 2.0) + 
        (df.get('Contratto_Determinato', 0) * 1.0) + 
        (df.get('Contratto_Partita IVA', 0) * 1.5) - 
        (df['Reddito'] / 35000) -
        (df['Eta'] / 80) -
        (df['ESG_Score'] / 150) + 
        np.random.normal(0, 1, n_samples)
    )
    
    prob_reale = 1 / (1 + np.exp(-score_latente))
    df['Default'] = (prob_reale > 0.5).astype(int)
    
    X = df.drop(columns=['Default'])
    y = df['Default']
    
    # Standardizzazione delle variabili (Z-score) per rendere i coefficienti beta comparabili
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Addestramento
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    
    # Estrazione dell'importanza delle variabili (valore assoluto dei coefficienti beta)
    feature_importanza = pd.DataFrame({
        'Variabile': X.columns,
        'Importanza': np.abs(model.coef_[0])
    }).sort_values(by='Importanza', ascending=True)
    
    return model, scaler, X.columns, feature_importanza

modello, scaler, colonne_features, feature_importanza = train_enterprise_model()


# --- 2. INTERFACCIA UTENTE ---
st.set_page_config(page_title="Enterprise Credit Scoring", page_icon="🏦", layout="wide")

st.title("🏦 Enterprise Credit & ESG Scoring Model")
st.markdown("Valutazione del rischio di credito integrata con metriche di sostenibilità finanziaria.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Dati Finanziari Base**")
    input_eta = st.number_input("Età Richiedente", min_value=18, max_value=90, value=29, step=1)
    input_reddito = st.number_input("Reddito Netto (€)", min_value=5000, value=35000, step=1000)
    input_contratto = st.selectbox("Contratto", ['Indeterminato', 'Determinato', 'Partita IVA'])

with col2:
    st.markdown("**Metriche di Rischio (Credit Bureau)**")
    input_dti = st.slider("Debt-to-Income (DTI)", 0.0, 1.0, 0.35, 0.01)
    input_utilizzo = st.slider("Utilizzo Plafond Credito", 0.0, 1.0, 0.20, 0.01)
    input_ritardi = st.number_input("Eventi di Insolvenza Passati", min_value=0, max_value=10, value=0, step=1)

with col3:
    st.markdown("**Metriche di Sostenibilità**")
    input_esg = st.slider("Punteggio Sostenibilità (ESG Score)", 0, 100, 50, 1, 
                          help="Un rating ESG elevato può sbloccare condizioni di tasso agevolate (Green Premium).")

# --- 3. PREVISIONE E ANALISI ---
if st.button("Esegui Analisi del Rischio", type="primary", use_container_width=True):
    
    is_determinato = 1 if input_contratto == 'Determinato' else 0
    is_piva = 1 if input_contratto == 'Partita IVA' else 0
    
    # Vettore di input
    input_raw = pd.DataFrame([[
        input_eta, input_reddito, input_dti, input_utilizzo, 
        input_ritardi, input_esg, is_determinato, is_piva
    ]], columns=colonne_features)
    
    # Applichiamo lo stesso scaling usato in fase di addestramento
    input_scaled = scaler.transform(input_raw)
    
    prob_default = modello.predict_proba(input_scaled)[0][1]
    score_finale = (1 - prob_default) * 1000 
    
    st.divider()
    
    # Layout risultati a due colonne
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("Esito Delibera")
        st.metric(label="Credit Score (Scala 0-1000)", value=f"{score_finale:.0f}")
        st.metric(label="Probability of Default (PD)", value=f"{prob_default:.2%}")
        
        if prob_default < 0.15:
            st.success("✅ **APPROVATO (Rischio Basso)** - Applicabile spread standard o agevolato se ESG > 75.")
        elif prob_default < 0.35:
            st.warning("⚠️ **IN REVISIONE (Rischio Moderato)** - Richiesto intervento analista umano.")
        else:
            st.error("❌ **DECLINATO (Rischio Alto)** - Parametri fuori policy creditizia.")

    with res_col2:
        st.subheader("Interpretabilità del Modello (AI Act Compliance)")
        st.markdown("Peso delle variabili nella decisione algoritmica:")
        # Grafico a barre orizzontale nativo di Streamlit
        st.bar_chart(feature_importanza.set_index('Variabile'))
