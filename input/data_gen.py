import yfinance as yf
import pandas as pd
import os

# Configurações
SYMBOL = "PETR4.SA"
START_DATE = "2023-12-31" 
END_DATE = "2026-01-01"
OUTPUT_FOLDER = "input"
OUTPUT_FILE = "dataset_petrobras.csv"

def fetch_and_clean_data(symbol, start, end):
    
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    
    if df.empty:
        raise ValueError(f"❌ Erro: Nenhum dado retornado para {symbol}. Verifique o ticker ou a internet.")

    # Tratamento das colunas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
        
    # Remove dias vazios caso haja
    df.dropna(inplace=True)
    
    df.index = pd.to_datetime(df.index)
    
    return df

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📂 Pasta '{OUTPUT_FOLDER}' criada.")

    # Baixa e processa
    df = fetch_and_clean_data(SYMBOL, START_DATE, END_DATE)

    file_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)


    df.to_csv(file_path)
    
if __name__ == "__main__":
    main()