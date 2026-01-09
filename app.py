import streamlit as st
import torch
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from model.model_class import LSTMModel

st.set_page_config(page_title="Petrobras Predictor", layout="wide")

@st.cache_resource
def load_model_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("model/scaler_final.pkl")
    model = LSTMModel(input_size=2, hidden_size=119, num_layers=1, dropout=0.0)
    model.load_state_dict(torch.load("model/modelo_lstm_final.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, scaler, device

def fetch_latest_data(ticker="PETR4.SA", window=20):
    try:
        data = yf.download(ticker, period="60d", interval="1d")
        if data.empty: 
            return None
        df = data[['Close', 'Open']].tail(window).copy()
        df.columns = ['close', 'open']
        df.index = df.index.date
        return df.sort_index(ascending=False)
    except Exception:
        return None

# --- INICIALIZAÇÃO DO ESTADO ---
model, scaler, device = load_model_assets()
window_size = 20

# Se não existe dados no estado, busca a primeira vez
if 'df_final' not in st.session_state:
    data = fetch_latest_data()
    if data is None:
        dates = pd.bdate_range(end=datetime.now(), periods=window_size)
        data = pd.DataFrame({'close': 35.0, 'open': 35.0}, index=dates.date).sort_index(ascending=False)
    st.session_state['df_final'] = data

# --- INTERFACE ---
st.title("📈 Predição PETR4 - Edição Manual Preservada")

# Botão de Atualização (Sobrescreve o estado apenas quando clicado)
if st.sidebar.button("🔄 Puxar Cotações do Yahoo Finance"):
    new_data = fetch_latest_data()
    if new_data is not None:
        st.session_state['df_final'] = new_data
        st.sidebar.success("Dados atualizados!")
        st.rerun()

st.markdown("### Histórico de Preços")
st.write("Qualquer alteração feita na tabela abaixo é salva automaticamente.")

# O segredo: o 'key' no data_editor faz com que ele gerencie o estado
# E salvamos o resultado de volta no session_state
edited_df = st.data_editor(
    st.session_state['df_final'],
    use_container_width=True,
    num_rows="fixed",
    key="my_editor" # Chave única para o widget
)

# Atualiza o session_state com o que foi editado para não perder ao clicar em outros botões
st.session_state['df_final'] = edited_df

if st.button("🚀 Calcular Previsão"):
    try:
        # 1. Ordem cronológica para o modelo
        df_ordered = edited_df.sort_index(ascending=True)
        values = df_ordered[['close', 'open']].values.astype(np.float32)
        
        # 2. Escalonamento e Tensor
        scaled_data = scaler.transform(values)
        input_tensor = torch.tensor(scaled_data).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().item()
        
        # 3. Desnormalização
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, 0] = pred_scaled
        final_pred = scaler.inverse_transform(dummy)[0, 0]

        last_date = pd.to_datetime(edited_df.index.max())
        next_date = (last_date + pd.offsets.BDay(1)).strftime('%d/%m/%Y')

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"Fechamento Previsto ({next_date})", value=f"R$ {final_pred:.2f}")
            st.write(f"Baseado no último fechamento editado: R$ {edited_df.iloc[0]['close']:.2f}")
        
        with c2:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = edited_df.sort_index(ascending=True)
            ax.plot(plot_df.index, plot_df['close'], marker='o', label="Dados da Tabela")
            ax.scatter(last_date + pd.offsets.BDay(1), final_pred, color='red', s=100, label="Previsão")
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Erro: {e}")