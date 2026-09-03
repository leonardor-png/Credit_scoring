"""
Credit Scoring & Sustainable Pricing Engine
--------------------------------------------------------------------
Modello dimostrativo di scoring del credito per PMI con integrazione
di un fattore ESG nel pricing del debito.

ATTENZIONE METODOLOGICA
Il dataset e' interamente simulato. La relazione tra ESG e default e'
imposta nel processo generatore dei dati (funzione simula_portafoglio):
il modello la ritrova, non la scopre. Lo strumento serve a rendere
visibile e manipolabile un meccanismo, non a stimarlo.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# CONFIGURAZIONE
# --------------------------------------------------------------------

SEED = 42
N_IMPRESE = 4000

# Esposizione alla transizione climatica per settore.
# Valore alto = maggiore sensibilita' del merito creditizio al profilo ESG.
SETTORI = {
    "Manifattura":  {"quota": 0.25, "rischio_base":  0.10, "esposizione_transizione": 1.0},
    "Servizi":      {"quota": 0.30, "rischio_base": -0.20, "esposizione_transizione": 0.3},
    "Costruzioni":  {"quota": 0.15, "rischio_base":  0.45, "esposizione_transizione": 1.2},
    "Energia":      {"quota": 0.10, "rischio_base":  0.00, "esposizione_transizione": 1.8},
    "Commercio":    {"quota": 0.20, "rischio_base":  0.15, "esposizione_transizione": 0.5},
}

# Coefficienti del processo generatore dei dati (DGP).
# Sono ASSUNZIONI, non stime. Esplicitarli e' il punto.
DGP = {
    "intercetta":       -4.80,  # calibrata su un tasso di default di portafoglio ~5%
    "leva":              0.55,   # Debito / EBITDA
    "ritardi":           0.85,   # eventi di scaduto negli ultimi 24 mesi
    "log_fatturato":    -0.45,   # dimensione come proxy di resilienza
    "anni_attivita":    -0.030,
    "esg_centrato":     -0.022,  # effetto ESG, modulato per settore
    "rumore":            0.60,
}

TASSO_BASE = 3.50          # costo della raccolta + margine minimo
SOGLIA_ESG_PIENA = 75      # sconto pieno
SOGLIA_ESG_PARZIALE = 60   # sconto ridotto
SCONTO_PIENO = 0.30
SCONTO_PARZIALE = 0.15

FEATURES = [
    "Anni_Attivita",
    "Log_Fatturato",
    "Leva",
    "Ritardi",
    "ESG_Score",
    "Esposizione_Transizione",
]


# --------------------------------------------------------------------
# 1. PROCESSO GENERATORE DEI DATI
# --------------------------------------------------------------------

@st.cache_data
def simula_portafoglio(n: int = N_IMPRESE, seed: int = SEED) -> pd.DataFrame:
    """Genera un portafoglio sintetico di PMI affidate.

    Il default e' estratto da una Bernoulli sulla probabilita' latente,
    non ottenuto tagliando la sigmoide a 0.5: le imprese con lo stesso
    profilo possono avere esiti diversi, come nella realta'.
    """
    rng = np.random.default_rng(seed)

    nomi_settori = list(SETTORI.keys())
    quote = np.array([SETTORI[s]["quota"] for s in nomi_settori])
    settore = rng.choice(nomi_settori, size=n, p=quote / quote.sum())

    rischio_base = np.array([SETTORI[s]["rischio_base"] for s in settore])
    esposizione = np.array([SETTORI[s]["esposizione_transizione"] for s in settore])

    anni_attivita = rng.integers(1, 41, size=n)
    fatturato = rng.lognormal(mean=14.0, sigma=0.85, size=n)          # ~1.2 mln mediano
    fatturato = np.clip(fatturato, 150_000, 80_000_000)
    leva = np.clip(rng.gamma(shape=4.0, scale=0.65, size=n), 0.2, 8.0)  # Debito/EBITDA
    ritardi = rng.poisson(0.35, size=n)

    # Imprese piu' grandi e strutturate tendono ad avere ESG migliore:
    # correlazione debole ma realistica, non ortogonalita' artificiale.
    esg = 52 + 4.5 * (np.log(fatturato) - 14.0) + rng.normal(0, 16, size=n)
    esg = np.clip(esg, 0, 100)

    esg_centrato = esg - 50.0

    # L'effetto ESG e' modulato dall'esposizione settoriale alla transizione:
    # in un settore energivoro un profilo ambientale debole pesa di piu'.
    logit = (
        DGP["intercetta"]
        + DGP["leva"] * leva
        + DGP["ritardi"] * ritardi
        + DGP["log_fatturato"] * (np.log(fatturato) - 14.0)
        + DGP["anni_attivita"] * anni_attivita
        + DGP["esg_centrato"] * esg_centrato * esposizione
        + rischio_base
        + rng.normal(0, DGP["rumore"], size=n)
    )

    prob = 1.0 / (1.0 + np.exp(-logit))
    default = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "Anni_Attivita": anni_attivita,
            "Fatturato": fatturato,
            "Log_Fatturato": np.log(fatturato) - 14.0,
            "Leva": leva,
            "Ritardi": ritardi,
            "ESG_Score": esg,
            "Settore": settore,
            "Esposizione_Transizione": esposizione,
            "Default": default,
        }
    )


# --------------------------------------------------------------------
# 2. STIMA E VALIDAZIONE
# --------------------------------------------------------------------

@st.cache_resource
def stima_modello(df: pd.DataFrame):
    """Stima una logistica su train e la valuta su un test set separato."""
    X = df[FEATURES]
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X_train, y_train)

    auc_train = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    coef = pd.DataFrame(
        {
            "Variabile": FEATURES,
            "Coefficiente (std.)": pipeline.named_steps["logit"].coef_[0],
        }
    ).sort_values("Coefficiente (std.)", key=abs, ascending=False)

    metriche = {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "gini_test": 2 * auc_test - 1,
        "tasso_default": y.mean(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, coef, metriche


# --------------------------------------------------------------------
# 3. MOTORE DI PRICING
# --------------------------------------------------------------------

def calcola_spread(pd_stimata: float) -> float:
    """Spread creditizio a fasce di rating interno."""
    if pd_stimata < 0.02:
        return 0.80
    if pd_stimata < 0.05:
        return 1.60
    if pd_stimata < 0.10:
        return 2.80
    if pd_stimata < 0.20:
        return 4.50
    return 7.00


def calcola_sconto_esg(esg: float, esposizione: float) -> float:
    """Sconto ESG a soglie, scalato per esposizione settoriale.

    NOTA: e' una regola commerciale parametrica, non un coefficiente
    stimato. Lo sconto vale di piu' dove la transizione morde di piu'.
    """
    if esg >= SOGLIA_ESG_PIENA:
        base = SCONTO_PIENO
    elif esg >= SOGLIA_ESG_PARZIALE:
        base = SCONTO_PARZIALE
    else:
        base = 0.0
    return round(base * esposizione, 2)


def costruisci_pricing(pd_stimata: float, esg: float, esposizione: float) -> dict:
    spread = calcola_spread(pd_stimata)
    sconto = calcola_sconto_esg(esg, esposizione)
    return {
        "tasso_base": TASSO_BASE,
        "spread": spread,
        "sconto": sconto,
        "tasso_finito": TASSO_BASE + spread - sconto,
    }


# --------------------------------------------------------------------
# 4. INTERFACCIA
# --------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Credit Scoring & Sustainable Pricing", layout="wide")

    st.title("Credit Scoring & Sustainable Pricing Engine")
    st.caption(
        "Modello dimostrativo di scoring del credito PMI con integrazione "
        "di un fattore ESG nel pricing del debito."
    )

    st.warning(
        "**Dati simulati.** Il portafoglio e' generato artificialmente e la relazione "
        "tra ESG e default e' imposta nel processo generatore: il modello la ritrova, "
        "non la dimostra. Vedi la nota metodologica in fondo alla pagina.",
        icon=":material/science:",
    )

    df = simula_portafoglio()
    modello, coefficienti, metriche = stima_modello(df)

    # ---------------- input ----------------
    st.subheader("Profilo dell'impresa richiedente")
    col1, col2 = st.columns(2)

    with col1:
        settore = st.selectbox(
            "Settore di attività",
            list(SETTORI.keys()),
            index=0,
            help="Determina il rischio di base e l'esposizione al rischio di transizione.",
        )
        anni = st.number_input(
            "Anni di attività", min_value=1, max_value=60, value=12,
            help="Proxy della stabilità dell'impresa e della profondità dello storico creditizio.",
        )
        fatturato = st.number_input(
            "Fatturato annuo (€)", min_value=150_000, max_value=80_000_000,
            value=1_500_000, step=50_000,
            help="Proxy dimensionale. Entra nel modello in forma logaritmica.",
        )

    with col2:
        leva = st.slider(
            "Leva finanziaria (Debito / EBITDA)", 0.2, 8.0, 2.5, 0.1,
            help="Sopra 4x la sostenibilità del debito è generalmente considerata critica.",
        )
        ritardi = st.number_input(
            "Eventi di scaduto (ultimi 24 mesi)", min_value=0, max_value=12, value=0,
            help="Segnalazioni di ritardo nei pagamenti. È il predittore più forte del modello.",
        )
        esg = st.slider(
            "ESG Score (0-100)", 0, 100, 55, 1,
            help=(
                "Punteggio di sostenibilità dell'impresa, ispirato nella scala ai pillar score "
                "dei terminali di mercato. ASSUNZIONE DEL MODELLO: sopra 60 attiva uno sconto "
                "sul tasso, scalato per l'esposizione settoriale alla transizione."
            ),
        )

    esposizione = SETTORI[settore]["esposizione_transizione"]
    st.caption(
        f"Esposizione alla transizione per il settore **{settore}**: "
        f"`{esposizione:.1f}x` — modula sia il peso dell'ESG nella PD sia lo sconto sul tasso."
    )

    # ---------------- output ----------------
    if st.button("Elabora merito creditizio", type="primary"):
        riga = pd.DataFrame(
            [[anni, np.log(fatturato) - 14.0, leva, ritardi, esg, esposizione]],
            columns=FEATURES,
        )
        pd_stimata = float(modello.predict_proba(riga)[0][1])
        pricing = costruisci_pricing(pd_stimata, esg, esposizione)

        st.divider()
        st.subheader("Esito e struttura del pricing")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Probability of Default", f"{pd_stimata:.2%}")
        c2.metric("Tasso base", f"{pricing['tasso_base']:.2f}%")
        c3.metric("Spread di rischio", f"+{pricing['spread']:.2f}%")
        c4.metric(
            "Effetto ESG",
            f"-{pricing['sconto']:.2f}%" if pricing["sconto"] > 0 else "0.00%",
        )

        st.success(f"### Tasso finito proposto: {pricing['tasso_finito']:.2f}%")

        # controfattuale: quanto vale il profilo ESG su questa pratica
        sconto_max = calcola_sconto_esg(100, esposizione)
        if pricing["sconto"] < sconto_max:
            st.info(
                f"Portando l'ESG Score sopra {SOGLIA_ESG_PIENA}, il tasso scenderebbe a "
                f"**{TASSO_BASE + pricing['spread'] - sconto_max:.2f}%** "
                f"(-{sconto_max - pricing['sconto']:.2f} punti percentuali)."
            )

    # ---------------- diagnostica ----------------
    st.divider()
    st.subheader("Diagnostica del modello")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("AUC (test)", f"{metriche['auc_test']:.3f}")
    d2.metric("Gini (test)", f"{metriche['gini_test']:.3f}")
    d3.metric("AUC (train)", f"{metriche['auc_train']:.3f}")
    d4.metric("Tasso di default", f"{metriche['tasso_default']:.1%}")

    st.caption(
        f"Stima su {metriche['n_train']:,} imprese, validazione su {metriche['n_test']:,} "
        "osservazioni non utilizzate in stima. Lo scarto contenuto fra AUC train e test "
        "indica assenza di overfitting; il potere discriminante elevato riflette anche il "
        "fatto che i dati sono generati dallo stesso tipo di funzione che il modello stima."
        .replace(",", ".")
    )

    with st.expander("Coefficienti stimati (variabili standardizzate)"):
        st.dataframe(coefficienti, hide_index=True, width="stretch")
        st.caption(
            "Segno positivo = aumenta la probabilità di default. I valori sono confrontabili "
            "fra loro perché le variabili sono standardizzate."
        )

    with st.expander("Nota metodologica — leggere prima di trarre conclusioni"):
        st.markdown(
            """
**Cosa fa questo strumento.** Simula un portafoglio di PMI affidate, stima su quei dati
una regressione logistica per la probabilità di default, e traduce la PD in un tasso
attraverso uno spread a fasce con uno sconto legato al profilo ESG.

**Cosa non fa.**

1. **I dati sono sintetici.** Nessuna osservazione reale. Le distribuzioni sono scelte
   per essere plausibili, non perché stimate su un portafoglio esistente.

2. **La relazione ESG-default è imposta, non scoperta.** Nel processo generatore l'ESG
   entra con un coefficiente negativo esplicito, modulato per esposizione settoriale.
   La logistica lo ritrova: è una verifica di coerenza interna, non evidenza empirica.
   L'AUC elevato va letto in questa luce.

3. **Lo sconto sul tasso è una regola parametrica**, non un coefficiente stimato: soglie
   a 60 e 75 punti, scalate per esposizione settoriale. È il modo in cui una banca
   potrebbe *tradurre operativamente* un greenium, non una misura del greenium stesso.

**Perché puo' comunque essere utile.** Rende esplicito e manipolabile il meccanismo con
cui un fattore ambientale può entrare nel pricing del credito — che è esattamente ciò che
le linee guida di vigilanza chiedono alle banche di formalizzare. La stima empirica del
differenziale di prezzo è oggetto della tesi, su dati di mercato.
            """
        )


if __name__ == "__main__":
    main()
