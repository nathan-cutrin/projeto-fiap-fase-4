import streamlit as st
import pandas as pd
import numpy as np
import requests  # Para comunicar com a API
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Petrobras Predictor - Frontend", layout="wide")

# URL da sua API FastAPI (Local ou Deployada)
API_URL = "http://127.0.0.1:8000/predict"

def fetch_latest_data(ticker="PETR4.SA", window=20):
    try:
        data = yf.download(ticker, period="60d", interval="1d")
        if data.empty: return None
        df = data[['Close', 'Open']].tail(window).copy()
        df.columns = ['close', 'open']
        df.index = df.index.date
        return df.sort_index(ascending=False)
    except Exception:
        return None

# --- INICIALIZAÇÃO DO ESTADO ---
if 'df_final' not in st.session_state:
    data = fetch_latest_data()
    st.session_state['df_final'] = data

# --- INTERFACE ---
st.title("📈 PETR4 Predictor (Streamlit + API)")

if st.sidebar.button("🔄 Puxar Dados Reais"):
    new_data = fetch_latest_data()
    if new_data is not None:
        st.session_state['df_final'] = new_data
        st.rerun()

st.markdown("### Histórico de Preços")
edited_df = st.data_editor(st.session_state['df_final'], use_container_width=True, num_rows="fixed")
st.session_state['df_final'] = edited_df

if st.button("🚀 Solicitar Previsão à API"):
    try:
        # 1. Preparar o JSON para a API (ordem cronológica: antigo -> novo)
        df_ordered = edited_df.sort_index(ascending=True)
        
        # IMPORTANTE: Pegamos apenas as colunas que a API espera (close e open)
        # Isso remove a coluna 'Data' que está causando o erro de serialização
        history_list = df_ordered[['close', 'open']].to_dict(orient='records')
        
        payload = {"history": history_list}

        # 2. Chamada à API
        with st.spinner("Comunicando com a API FastAPI..."):
            response = requests.post(API_URL, json=payload)

        
        if response.status_code == 200:
            result = response.json()
            final_pred = result["prediction_next_close"]

            # 3. Exibição dos resultados
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Previsão da API (Próximo Fechamento)", value=f"R$ {final_pred:.2f}")
                st.success("Conexão com API: OK")
            
            with c2:
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_df = edited_df.sort_index(ascending=True)
                ax.plot(plot_df.index, plot_df['close'], marker='o', label="Histórico")
                
                # Ponto futuro
                last_date = pd.to_datetime(edited_df.index.max())
                future_date = last_date + pd.offsets.BDay(1)
                ax.scatter(future_date.date(), final_pred, color='red', s=100, label="Previsão API")
                
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)
        else:
            st.error(f"Erro na API: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Erro de conexão: Verifique se a API no app.py está rodando em {API_URL}, {e}")