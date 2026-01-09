import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib 
import os
import random
import copy  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from model.model_class import LSTMModel

def set_seeds(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Força determinismo em algoritmos de convolução/LSTM
    torch.backends.cudnn.benchmark = False 

def create_sequences(data, lookback, target_idx=0):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback, target_idx]) 
    return np.array(X), np.array(y)

def get_dataloaders(df, feature_cols, target_col='close', lookback=30, 
                    split_ratio=0.8, batch_size=32, use_scaler=True):
    
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)
    
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    scaler = None
    if use_scaler:
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
    X_train, y_train = create_sequences(train_data, lookback, target_idx)
    
    # Contexto para o teste
    last_window_train = train_data[-lookback:]
    test_data_concat = np.concatenate((last_window_train, test_data))
    X_test, y_test = create_sequences(test_data_concat, lookback, target_idx)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, target_idx

def train_and_evaluate(model, train_loader, test_loader, scaler, target_idx, epochs, patience, device, lr):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                pred = model(X_val)
                val_loss += criterion(pred, y_val).item()
        val_loss /= len(test_loader)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early Stopping na época {epoch}")
                break
    
    model.load_state_dict(best_weights)
    model.eval()
    
    preds, targets = [], []
    with torch.no_grad():
        for X_t, y_t in test_loader:
            X_t = X_t.to(device)
            p = model(X_t)
            preds.append(p.cpu().numpy())
            targets.append(y_t.numpy())
    
    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(targets).flatten()

    if scaler:
        d_pred = np.zeros((len(y_pred), scaler.n_features_in_))
        d_true = np.zeros((len(y_true), scaler.n_features_in_))
        d_pred[:, target_idx] = y_pred
        d_true[:, target_idx] = y_true
        inv_pred = scaler.inverse_transform(d_pred)[:, target_idx]
        inv_true = scaler.inverse_transform(d_true)[:, target_idx]
    else:
        inv_pred, inv_true = y_pred, y_true

    mape = mean_absolute_percentage_error(inv_true, inv_pred) * 100
    rmse = np.sqrt(mean_squared_error(inv_true, inv_pred))
    
    return {'MAPE': mape, 'RMSE': rmse, 'y_true': inv_true, 'y_pred': inv_pred}

if __name__ == "__main__":
    set_seeds(43)
    
    INPUT_FILE = "input/dataset_petrobras.csv"
    if not os.path.exists(INPUT_FILE):
        INPUT_FILE = "dataset_petrobras.csv"

    CONFIG = {
        'features': ["close", "open"],  
        'window': 20,                 
        'hidden_size': 119,
        'layers': 1,
        'dropout': 0.2181710369026575,
        'lr': 0.005295762643270361,
        'epochs': 400,
        'patience': 20,
        'batch_size': 16
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    train_loader, test_loader, scaler_obj, target_idx = get_dataloaders(
        df, CONFIG['features'], lookback=CONFIG['window'], batch_size=CONFIG['batch_size']
    )

    model = LSTMModel(
        input_size=len(CONFIG['features']), 
        hidden_size=CONFIG['hidden_size'], 
        num_layers=CONFIG['layers'], 
        dropout=CONFIG['dropout']
    ).to(device)

    metrics = train_and_evaluate(
        model, train_loader, test_loader, scaler_obj, target_idx, 
        CONFIG['epochs'], CONFIG['patience'], device, CONFIG['lr']
    )

    print("\nTREINO CONCLUÍDO")
    print(f"MAPE: {metrics['MAPE']:.4f}% | RMSE: {metrics['RMSE']:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), 'model/modelo_lstm_final.pth')
    joblib.dump(scaler_obj, 'model/scaler_final.pkl')