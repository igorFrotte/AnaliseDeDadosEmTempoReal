import os
import glob

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAÇÕES BÁSICAS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIEW_PATH = os.path.join(BASE_DIR, "view", "predicoes")

st.set_page_config(
    page_title="Monitor 3W - Classificação em Tempo Real",
    layout="wide",
)

st.title("📈 Monitor 3W – Classificação de Anomalias em Tempo Real (3W + Kafka + Spark)")

st.markdown(
    """
    Este painel mostra, em tempo quase real, as leituras do dataset 3W da Petrobras
    sendo classificadas por um modelo de Machine Learning treinado previamente.
    
    **Pipeline:** Kafka (producer) → Spark Streaming (consumer) → Modelo RandomForest → Dashboard.
    """
)

# Intervalo de atualização automática
refresh_sec = st.sidebar.slider("Intervalo de atualização da tela (segundos)", 1, 10, 3)

# Dispara auto-refresh da página a cada X segundos
st_autorefresh(interval=refresh_sec * 1000, key="data_refresh")


# ==========================
# FUNÇÃO PARA CARREGAR DADOS ACUMULADOS
# ==========================

def carregar_dados_acumulados(max_arquivos: int = 50) -> pd.DataFrame | None:
    """
    Carrega e concatena até `max_arquivos` CSVs de predição
    a partir de VIEW_PATH (recursivamente nas subpastas, ex.: batch_0, batch_1, ...),
    ignorando arquivos temporários criados pelo Spark.
    """
    # Procura todos os CSVs abaixo de view/predicoes (incluindo subpastas)
    pattern = os.path.join(VIEW_PATH, "batch_*", "part-*.csv")
    arquivos = glob.glob(pattern, recursive=True)

    if not arquivos:
        return None

    # Ignora arquivos em pastas _temporary (artefatos de escrita do Spark)
    arquivos = [
        arq for arq in arquivos
        if "_temporary" not in arq
    ]

    if not arquivos:
        return None

    # Ordena por data de modificação (do mais antigo para o mais recente)
    arquivos_ordenados = sorted(arquivos, key=os.path.getmtime)

    # Pega só os últimos N arquivos (para evitar excesso de dados)
    arquivos_uso = arquivos_ordenados[-max_arquivos:]

    dfs = []
    for arq in arquivos_uso:
        try:
            df_parc = pd.read_csv(arq)
            dfs.append(df_parc)
        except Exception:
            # Se algum arquivo der problema (corrompido, vazio, etc.), apenas ignora silenciosamente
            continue

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    return df


# ==========================
# CARREGA DADOS (ACUMULADOS)
# ==========================

# Você pode ajustar aqui quantos arquivos (batches) quer considerar no histórico
df = carregar_dados_acumulados(max_arquivos=50)

if df is None or df.empty:
    st.warning(
        "Ainda não há dados disponíveis em `view/predicoes`.\n\n"
        "Verifique se o *consumer* Spark está em execução e aguarde os próximos batches."
    )
    st.stop()

# Conversões e ordenação
if "timestamp_envio" in df.columns:
    df["timestamp_envio"] = pd.to_datetime(df["timestamp_envio"], errors="coerce")
    df = df.sort_values("timestamp_envio")

# Garante tipos numéricos nas classes
if "prediction" in df.columns:
    df["prediction"] = df["prediction"].astype(float).astype(int)
if "classe_real" in df.columns:
    df["classe_real"] = df["classe_real"].astype(float).astype(int)

# ==========================
# PAINEL DE KPIs (SALA DE CONTROLE)
# ==========================

ultima_linha = df.tail(1).iloc[0]

ultima_pred = int(ultima_linha["prediction"]) if "prediction" in df.columns else None
ultima_real = int(ultima_linha["classe_real"]) if "classe_real" in df.columns else None

total_eventos = len(df)

if "prediction" in df.columns and "classe_real" in df.columns:
    acuracia_global = (df["prediction"] == df["classe_real"]).mean()
else:
    acuracia_global = None

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric("Última classe prevista", value=str(ultima_pred) if ultima_pred is not None else "-")

with col_kpi2:
    st.metric("Última classe real", value=str(ultima_real) if ultima_real is not None else "-")

with col_kpi3:
    st.metric("Total de eventos considerados", value=str(total_eventos))

with col_kpi4:
    if acuracia_global is not None:
        st.metric("Acurácia no conjunto acumulado", value=f"{acuracia_global*100:.1f}%")
    else:
        st.metric("Acurácia no conjunto acumulado", value="N/A")


st.markdown("---")

# ==========================
# LINHA 1: TABELA + DISTRIBUIÇÃO POR CLASSE
# ==========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Últimas medições e predições (dados acumulados)")
    colunas_existentes = [
        c for c in [
            "timestamp_envio",
            "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL",
            "prediction", "classe_real"
        ]
        if c in df.columns
    ]

    st.dataframe(
        df[colunas_existentes].tail(50),
        use_container_width=True,
        height=400,
    )

with col2:
    st.subheader("Distribuição das classes previstas (dados acumulados)")
    if "prediction" in df.columns:
        counts_pred = df["prediction"].value_counts().sort_index()
        st.bar_chart(counts_pred)
    else:
        st.info("Coluna 'prediction' não encontrada no dataframe.")


st.markdown("---")

# ==========================
# LINHA 2: MATRIZ DE CONFUSÃO + HEATMAP
# ==========================

col3, col4 = st.columns(2)

if "classe_real" in df.columns and "prediction" in df.columns:
    # Matriz de confusão (tabela)
    cm = pd.crosstab(
        df["classe_real"],
        df["prediction"],
        rownames=["Classe real"],
        colnames=["Classe prevista"]
    )

    with col3:
        st.subheader("Matriz de confusão (dados acumulados)")
        st.dataframe(cm, use_container_width=True)

    with col4:
        st.subheader("Heatmap da matriz de confusão")
        fig, ax = plt.subplots()
        im = ax.imshow(cm.values, cmap="OrRd")

        # Eixos
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns)
        ax.set_yticklabels(cm.index)

        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")

        # Rótulos de contagem em cada célula
        max_val = cm.values.max() if cm.values.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm.values[i, j]
                ax.text(
                    j, i,
                    val,
                    ha="center", va="center",
                    color="black" if max_val == 0 or val < max_val / 2 else "white",
                    fontsize=8,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
else:
    st.info("Não foi possível gerar a matriz de confusão: é necessário ter 'classe_real' e 'prediction' no dataframe.")


st.markdown("---")

# ==========================
# LINHA 3: SÉRIE TEMPORAL DE UMA VARIÁVEL (EX.: P-TPT)
# ==========================

st.subheader("Série temporal recente de T-JUS-CKP (dados acumulados)")

if "timestamp_envio" in df.columns and "T-JUS-CKP" in df.columns:
    serie = (
        df[["timestamp_envio", "T-JUS-CKP"]]
        .dropna()
        .set_index("timestamp_envio")
        .tail(300)
    )
    st.line_chart(serie)
else:
    st.info("Colunas 'timestamp_envio' e/ou 'P-TPT' não encontradas para gerar a série temporal.")
