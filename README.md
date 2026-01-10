📈 PETR4 Predictor - Tech Challenge Fase 4
Este repositório contém a solução para o Tech Challenge da Fase 4 (FIAP - IA para Devs). O projeto consiste em uma pipeline completa de Deep Learning para predição de preços de fechamento das ações da Petrobras (PETR4), utilizando redes neurais LSTM (Long Short-Term Memory).

🚀 Arquitetura do Projeto
A solução foi desenhada seguindo padrões de sistemas distribuídos e resilientes:

Modelo Preditivo: Rede neural LSTM desenvolvida com PyTorch.

Backend ([API](https://projeto-fiap-fase-4.onrender.com/docs#/)): Desenvolvido com FastAPI, hospedado no Render, responsável por servir o modelo via REST.

Frontend ([Dashboard](https://projeto-fiap-fase-4-ccechfemfcdt7gjzkcral6.streamlit.app)): Desenvolvido com Streamlit, oferecendo uma interface intuitiva para usuários e monitoramento de performance.

Sistema Híbrido: O frontend possui uma lógica de Graceful Degradation, acionando um modelo local caso a API esteja indisponível.

🛠️ Tecnologias Utilizadas
Linguagem: Python 3.11+

Deep Learning: PyTorch

API Framework: FastAPI

Interface: Streamlit

Gestão de Pacotes: uv (Astral)

Dados: Yahoo Finance (yfinance)

Monitoramento: psutil e métricas internas de latência.

📋 Requisitos do Tech Challenge Atendidos
1. Coleta e Pré-processamento
Uso da biblioteca yfinance para extração de dados históricos.

Normalização de dados com MinMaxScaler para otimização do treinamento da LSTM.

2. Desenvolvimento e Avaliação do Modelo
Implementação de rede LSTM para capturar padrões temporais.

Métricas de avaliação: MAE, RMSE e MAPE (foco em erro percentual).

3. API RESTful (FastAPI)
Endpoint /predict para predição via POST.

Endpoint /monitoramento para telemetria de recursos do servidor.

Documentação automática via Swagger UI (/docs).

4. Monitoramento e Escalabilidade
Tempo de Resposta: Rastreamento de latência por requisição no dashboard.

Uso de Recursos: Monitoramento em tempo real de CPU e RAM do servidor de produção.

⚙️ Como Executar Localmente
Este projeto utiliza o uv para gerenciamento ultrarápido de dependências.

Instalação
Bash

# Instale o uv caso não tenha
pip install uv

# Sincronize as dependências
uv sync
Executar a API (Backend)
Bash

uv run uvicorn app:app --reload
Executar o Dashboard (Frontend)
Bash

uv run streamlit run streamlit_app.py

📊 Dashboard de Monitoramento
O Dashboard integrado no Streamlit permite visualizar:

Latência Média: Tempo que a API leva para responder.

Disponibilidade: Porcentagem de sucesso das requisições.

Saúde do Servidor: Consumo de hardware no ambiente de nuvem.

📄 Estrutura de Pastas
/model: Pesos do modelo (.pth), scaler (.pkl) e definição da classe LSTM.

/notebooks: Exploração de dados e prototipagem do modelo.

app.py: Código principal da API FastAPI.

streamlit_app.py: Interface do usuário e dashboard de monitoramento.

Próximo Passo Sugerido
Gostaria que eu gerasse também o guia de roteiro para o vídeo de apresentação, destacando onde cada um desses pontos aparece no código?