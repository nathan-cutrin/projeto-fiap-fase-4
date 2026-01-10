import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import joblib
from datetime import datetime
from model.model_class import LSTMModel # Importante para o fallback local

st.set_page_config(page_title="Petrobras Predictor - Híbrido", layout="wide")

# URL da sua API no Render ou Local
API_URL = "https://projeto-fiap-fase-4.onrender.com/predict"

# --- FUNÇÃO DE CARREGAMENTO LOCAL (FALLBACK) ---
@st.cache_resource
def load_model_local():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = joblib.load("model/scaler_final.pkl")
        model = LSTMModel(input_size=2, hidden_size=119, num_layers=1, dropout=0.0)
        model.load_state_dict(torch.load("model/modelo_lstm_final.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, scaler, device
    except Exception as e:
        st.error(f"Erro ao carregar modelo local: {e}")
        return None, None, None

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

# --- INICIALIZAÇÃO ---
if 'df_final' not in st.session_state:
    st.session_state['df_final'] = fetch_latest_data()

# --- INTERFACE ---
st.title("📈 PETR4 Predictor - Sistema Híbrido")
st.info("O sistema prioriza a **API FastAPI**. Caso a API esteja offline, o modelo será processado **localmente**.")

if st.sidebar.button("🔄 Atualizar Cotações"):
    new_data = fetch_latest_data()
    if new_data is not None:
        st.session_state['df_final'] = new_data
        st.rerun()

st.markdown("### Dados para Predição")
edited_df = st.data_editor(st.session_state['df_final'], use_container_width=True, num_rows="fixed")
st.session_state['df_final'] = edited_df

if st.button("🚀 Calcular Previsão"):
    final_pred = None
    metodo_utilizado = ""

    # 1. TENTATIVA VIA API
    try:
        df_ordered = edited_df.sort_index(ascending=True)
        history_list = df_ordered[['close', 'open']].to_dict(orient='records')
        payload = {"history": history_list}

        with st.spinner("Tentando conexão com a API..."):
            response = requests.post(API_URL, json=payload, timeout=5)
            
        if response.status_code == 200:
            final_pred = response.json()["prediction_next_close"]
            metodo_utilizado = "📡 Via API REST (Nuvem/Local)"
        else:
            st.warning("API retornou erro. Tentando processamento local...")

    except Exception:
        st.warning("API offline ou inacessível. Acionando modelo de contingência local...")

    # 2. TENTATIVA LOCAL (FALLBACK)
    if final_pred is None:
        model, scaler, device = load_model_local()
        if model:
            with st.spinner("Processando localmente (Inference Mode)..."):
                df_ordered = edited_df.sort_index(ascending=True)
                values = df_ordered[['close', 'open']].values.astype(np.float32)
                scaled_data = scaler.transform(values)
                input_tensor = torch.tensor(scaled_data).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor).cpu().item()
                
                dummy = np.zeros((1, 2))
                dummy[0, 0] = output
                final_pred = scaler.inverse_transform(dummy)[0, 0]
                metodo_utilizado = "💻 Via Modelo Local (Fallback)"

    # --- EXIBIÇÃO DO RESULTADO ---
    if final_pred:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Resultado")
            st.metric(label="Previsão Próximo Fechamento", value=f"R$ {final_pred:.2f}")
            st.write(f"**Método:** {metodo_utilizado}")
            
            if "Local" in metodo_utilizado:
                st.caption("Nota: O processamento local foi utilizado porque a API não respondeu.")
            else:
                st.success("Conexão com API realizada com sucesso!")

        with c2:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = edited_df.sort_index(ascending=True)
            ax.plot(plot_df.index, plot_df['close'], marker='o', label="Histórico")
            
            last_date = pd.to_datetime(edited_df.index.max())
            future_date = last_date + pd.offsets.BDay(1)
            ax.scatter(future_date.date(), final_pred, color='red', s=100, label="Previsão")
            
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("Não foi possível calcular a previsão nem via API nem localmente. Verifique os arquivos na pasta /model.")