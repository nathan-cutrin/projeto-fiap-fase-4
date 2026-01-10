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

# URL da sua API no Render (Certifique-se de que o deploy foi feito)
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

# --- INICIALIZAÇÃO DOS DADOS ---
if 'df_final' not in st.session_state:
    st.session_state['df_final'] = fetch_latest_data()

# --- INTERFACE PRINCIPAL ---
st.title("📈 PETR4 Predictor - Sistema Híbrido")
st.markdown("""
Esta aplicação realiza a predição do preço de fechamento da Petrobras (PETR4). 
O sistema utiliza uma arquitetura de microsserviços: tenta primeiro a comunicação com a **API FastAPI (Render)** e aciona o **Modelo Local** automaticamente em caso de falha.
""")

st.info("💡 **Dica:** Você pode editar os valores na tabela abaixo para simular cenários.")

if st.sidebar.button("🔄 Atualizar Dados Reais (YFinance)"):
    with st.spinner("Buscando cotações atuais..."):
        new_data = fetch_latest_data()
        if new_data is not None:
            st.session_state['df_final'] = new_data
            st.sidebar.success("Dados atualizados!")
            st.rerun()

st.markdown("### Dados de Entrada (Últimos 20 dias)")
edited_df = st.data_editor(st.session_state['df_final'], use_container_width=True, num_rows="fixed")
st.session_state['df_final'] = edited_df

if st.button("🚀 Calcular Previsão"):
    final_pred = None
    metodo_utilizado = ""
    start_time = time.time()
    
    # 1. TENTATIVA VIA API (REQUISITO 4)
    try:
        df_ordered = edited_df.sort_index(ascending=True)
        # Envia apenas close e open (evita erro de serialização de data)
        history_list = df_ordered[['close', 'open']].to_dict(orient='records')
        payload = {"history": history_list}

        with st.spinner("Enviando requisição para API FastAPI no Render..."):
            response = requests.post(API_URL, json=payload, timeout=10)
            
        if response.status_code == 200:
            final_pred = response.json()["prediction_next_close"]
            metodo_utilizado = "API REST (Cloud)"
            status_request = "Success"
        else:
            status_request = "API Error"
            st.warning(f"API retornou status {response.status_code}. Acionando fallback...")

    except Exception as e:
        status_request = "Connection Failed"
        st.warning("API inacessível. Processando via Modelo Local (Contingência)...")

    # 2. TENTATIVA LOCAL (FALLBACK)
    if final_pred is None:
        model, scaler, device = load_model_local()
        if model:
            with st.spinner("Processando localmente (PyTorch Inference)..."):
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
                # Se chegou aqui é porque a API falhou mas o Local salvou
                if status_request == "Success": status_request = "Local" 
        else:
            status_request = "Total Failure"

    # Cálculo final de latência para o monitoramento
    latency = time.time() - start_time
    
    # SALVAR MÉTRICAS (REQUISITO 5)
    new_metric = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%H:%M:%S")],
        'Latency': [latency],
        'Status': [status_request],
        'Method': [metodo_utilizado]
    })
    st.session_state['monitoring_data'] = pd.concat([st.session_state['monitoring_data'], new_metric], ignore_index=True)

    # --- EXIBIÇÃO DO RESULTADO ---
    if final_pred:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎯 Resultado da Predição")
            st.metric(label="Preço Previsto (Próximo Dia Útil)", value=f"R$ {final_pred:.2f}")
            st.write(f"**Método de Processamento:** {metodo_utilizado}")
            st.write(f"**Tempo de Resposta:** {latency:.3f} segundos")
            
            if "Local" in metodo_utilizado:
                st.info("Nota: O processamento local garante a disponibilidade do serviço mesmo sem internet ou API fora do ar.")
            else:
                st.success("Dados processados via API RESTful com sucesso.")

        with c2:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = edited_df.sort_index(ascending=True)
            ax.plot(plot_df.index, plot_df['close'], marker='o', label="Histórico Real")
            
            # Cálculo da data futura para o gráfico
            last_date = pd.to_datetime(edited_df.index.max())
            future_date = last_date + pd.offsets.BDay(1)
            ax.scatter(future_date.date(), final_pred, color='red', s=150, label="Previsão", zorder=5)
            
            ax.set_title("Evolução de Preço e Projeção")
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("Falha crítica: Não foi possível realizar a predição. Verifique a pasta /model.")

# --- SEÇÃO DE MONITORAMENTO (REQUISITO 5 - ESCALABILIDADE E MONITORAMENTO) ---
st.markdown("---")
st.header("📊 Dashboard de Monitoramento de Performance")
st.write("Métricas em tempo real da pipeline de produção (API e Recursos).")

if not st.session_state['monitoring_data'].empty:
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        avg_lat = st.session_state['monitoring_data']['Latency'].mean()
        st.metric("Latência Média", f"{avg_lat:.3f}s")
        
    with m2:
        success_rate = (st.session_state['monitoring_data']['Status'].isin(['Success', 'Local'])).mean() * 100
        st.metric("Taxa de Disponibilidade", f"{success_rate:.1f}%")

    with m3:
        api_usage = (st.session_state['monitoring_data']['Method'] == 'API REST (Cloud)').sum()
        st.metric("Requisições API", api_usage)
        
    with m4:
        st.metric("Total de Predições", len(st.session_state['monitoring_data']))

    # Gráficos de Monitoramento
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Histórico de Latência (Segundos)")
        st.line_chart(st.session_state['monitoring_data'].set_index('Timestamp')['Latency'])
    
    with g2:
        st.subheader("Distribuição por Método")
        method_counts = st.session_state['monitoring_data']['Method'].value_counts()
        st.bar_chart(method_counts)
        
else:
    st.info("Realize uma predição para visualizar os gráficos de monitoramento de performance.")