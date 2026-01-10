import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import joblib
import time
from datetime import datetime
from model.model_class import LSTMModel

# Configuração da Página
st.set_page_config(page_title="Petrobras Predictor - Tech Challenge", layout="wide")

# --- INICIALIZAÇÃO DO MONITORAMENTO ---
if 'monitoring_data' not in st.session_state:
    st.session_state['monitoring_data'] = pd.DataFrame(columns=['Timestamp', 'Latency', 'Status', 'Method'])

# URL da sua API no Render
API_URL = "https://projeto-fiap-fase-4.onrender.com/predict"
HEALTH_URL = "https://projeto-fiap-fase-4.onrender.com/monitoramento"

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

# --- INICIALIZAÇÃO DOS DADOS ---
if 'df_final' not in st.session_state:
    st.session_state['df_final'] = fetch_latest_data()

# --- INTERFACE PRINCIPAL ---
st.title("📈 PETR4 Predictor - Sistema Híbrido")
st.markdown("""
Esta aplicação realiza a predição do preço de fechamento da Petrobras (PETR4). 
O sistema tenta primeiro a comunicação com a **API FastAPI (Render)** e aciona o **Modelo Local** automaticamente em caso de falha.
""")

if st.sidebar.button("🔄 Atualizar Dados Reais"):
    with st.spinner("Buscando cotações atuais..."):
        new_data = fetch_latest_data()
        if new_data is not None:
            st.session_state['df_final'] = new_data
            st.rerun()

st.markdown("### Dados de Entrada (Últimos 20 dias)")
edited_df = st.data_editor(st.session_state['df_final'], use_container_width=True, num_rows="fixed")
st.session_state['df_final'] = edited_df

if st.button("🚀 Calcular Previsão"):
    final_pred = None
    metodo_utilizado = ""
    status_request = "Pending"
    start_time = time.time()
    
    # 1. TENTATIVA VIA API (REQUISITO 4)
    try:
        df_ordered = edited_df.sort_index(ascending=True)
        history_list = df_ordered[['close', 'open']].to_dict(orient='records')
        payload = {"history": history_list}

        with st.spinner("Enviando requisição para API..."):
            response = requests.post(API_URL, json=payload, timeout=12)
            
        if response.status_code == 200:
            final_pred = response.json()["prediction_next_close"]
            metodo_utilizado = "API REST (Cloud)"
            status_request = "Success"
        else:
            status_request = "API Error"
    except Exception:
        status_request = "Connection Failed"

    # 2. TENTATIVA LOCAL (FALLBACK)
    if final_pred is None:
        st.warning("⚠️ API Indisponível. Acionando contingência local...")
        model, scaler, device = load_model_local()
        if model:
            df_ordered = edited_df.sort_index(ascending=True)
            values = df_ordered[['close', 'open']].values.astype(np.float32)
            scaled_data = scaler.transform(values)
            input_tensor = torch.tensor(scaled_data).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor).cpu().item()
            dummy = np.zeros((1, 2))
            dummy[0, 0] = output
            final_pred = scaler.inverse_transform(dummy)[0, 0]
            metodo_utilizado = "Modelo Local (Fallback)"
            if status_request == "Pending": status_request = "Local Only"

    latency = time.time() - start_time
    
    # SALVAR MÉTRICAS (REQUISITO 5)
    new_metric = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%H:%M:%S")],
        'Latency': [latency],
        'Status': [status_request],
        'Method': [metodo_utilizado]
    })
    st.session_state['monitoring_data'] = pd.concat([st.session_state['monitoring_data'], new_metric], ignore_index=True)

    if final_pred:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎯 Resultado")
            st.metric(label="Preço Previsto", value=f"R$ {final_pred:.2f}")
            st.write(f"**Método:** {metodo_utilizado} | **Tempo:** {latency:.3f}s")
        with c2:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = edited_df.sort_index(ascending=True)
            ax.plot(plot_df.index, plot_df['close'], marker='o', label="Real")
            last_date = pd.to_datetime(edited_df.index.max())
            future_date = last_date + pd.offsets.BDay(1)
            ax.scatter(future_date.date(), final_pred, color='red', s=150, label="Previsão")
            ax.legend()
            st.pyplot(fig)

# --- SEÇÃO DE MONITORAMENTO (REQUISITO 5) ---
st.markdown("---")
st.header("📊 Monitoramento de Performance e Recursos")

if not st.session_state['monitoring_data'].empty:
    m1, m2, m3 = st.columns(3)
    m1.metric("Latência Média", f"{st.session_state['monitoring_data']['Latency'].mean():.3f}s")
    m2.metric("Disponibilidade", f"{(st.session_state['monitoring_data']['Status'].isin(['Success', 'Local Only'])).mean()*100:.1f}%")
    m3.metric("Total Requisições", len(st.session_state['monitoring_data']))

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Latência (s)")
        st.line_chart(st.session_state['monitoring_data'].set_index('Timestamp')['Latency'])
    with g2:
        st.subheader("Métodos Utilizados")
        st.bar_chart(st.session_state['monitoring_data']['Method'].value_counts())

# BOTÃO DE SAÚDE DO SERVIDOR (REQUISITO 5 - UTILIZAÇÃO DE RECURSOS)
st.write("### Verificação de Recursos do Servidor")
if st.button("🔍 Consultar CPU/RAM da API no Render"):
    try:
        health_res = requests.get(HEALTH_URL, timeout=5)
        if health_res.status_code == 200:
            health_data = health_res.json()
            h1, h2 = st.columns(2)
            h1.metric("Uso de CPU (Servidor)", f"{health_data['cpu_usage_percent']}%")
            h2.metric("Uso de RAM (API)", f"{health_data['memory_usage_mb']:.1f} MB")
        else:
            st.error("Servidor respondeu, mas não enviou métricas.")
    except:
        st.error("Não foi possível conectar ao endpoint de monitoramento da API.")