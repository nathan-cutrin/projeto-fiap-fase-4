import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import copy

def set_seeds(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # para multi-GPU
    
    # Garante que as operações da GPU sejam determinísticas (pode ficar um pouco mais lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_sequences(dataset, lookback, target_idx=0):
    """
    Args:
        dataset (np.array): Matriz de dados (Numpy) onde as colunas são as features.
        lookback (int): Tamanho da janela.
        target_idx (int): Índice da coluna que é o alvo (default=0 para 'close' se for a primeira).
    """
    # Se vier DataFrame, converte para Numpy para garantir
    if hasattr(dataset, 'values'):
        data = dataset.values
    else:
        data = dataset

    X, y = [], []
    
    # Loop principal (sua lógica original, mantida)
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        # Pega o valor do alvo no dia seguinte ao fim da janela
        y.append(data[i + lookback, target_idx]) 
        
    return np.array(X), np.array(y)

def get_dataloaders(df, feature_cols, target_col='close', lookback=30, 
                    split_ratio=0.8, batch_size=32, use_scaler=True):
    """
    Prepara os DataLoaders de treino e teste.
    """
    # 1. Extração dos dados brutos
    data = df[feature_cols].values
    
    # Identificar índice do target dentro da lista de features para o create_sequences
    target_idx = feature_cols.index(target_col)
    
    # 2. Split Cronológico (Antes de escalar para evitar leakage)
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    scaler = None
    
    # 3. Escalonamento Condicional
    if use_scaler:
        scaler = MinMaxScaler()
        # Fit apenas no treino
        train_data = scaler.fit_transform(train_data)
        # Transform no teste (usando estatísticas do treino)
        test_data = scaler.transform(test_data)
        
    # 4. Criar Sequências (usando sua função)
    X_train, y_train = create_sequences(train_data, lookback, target_idx)
    
    # Para o teste, precisamos do final do treino para dar contexto à primeira janela
    # Concatenamos os últimos 'lookback' dias do treino com o teste
    last_window_train = train_data[-lookback:]
    test_data_concat = np.concatenate((last_window_train, test_data))
    
    X_test, y_test = create_sequences(test_data_concat, lookback, target_idx)
    
    # 5. Converter para Tensores
    # X shape: (amostras, lookback, n_features)
    # y shape: (amostras, 1)
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
    
    # Retornamos o target_idx também para saber qual coluna desnormalizar depois
    return train_loader, test_loader, scaler, target_idx

def train_and_evaluate(model, train_loader, test_loader, scaler, target_idx,
                       epochs=100, lr=0.001, patience=15, device='cpu'):
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())
    
    # --- LOOP DE TREINO ---
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                pred = model(X_val)
                val_loss += criterion(pred, y_val).item()
        
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break # Early Stop
    
    # --- AVALIAÇÃO FINAL ---
    model.load_state_dict(best_weights)
    model.eval()
    
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            pred = model(X_test)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y_test.numpy())
    
    # Flatten para ficar 1D (N,)
    y_pred = np.concatenate(preds_list).flatten()
    y_true = np.concatenate(targets_list).flatten()
    
    # --- DESNORMALIZAÇÃO UNIVERSAL (CORREÇÃO) ---
    if scaler:
        # 1. Cria uma matriz vazia com o formato que o scaler espera (N_samples, N_features)
        n_features = scaler.n_features_in_
        dummy_pred = np.zeros((len(y_pred), n_features))
        dummy_true = np.zeros((len(y_true), n_features))
        
        # 2. Preenche apenas a coluna do Target com nossos valores
        dummy_pred[:, target_idx] = y_pred
        dummy_true[:, target_idx] = y_true
        
        # 3. Faz o inverse_transform da matriz completa
        inv_pred = scaler.inverse_transform(dummy_pred)
        inv_true = scaler.inverse_transform(dummy_true)
        
        # 4. Recupera apenas a coluna que nos interessa
        y_pred_final = inv_pred[:, target_idx]
        y_true_final = inv_true[:, target_idx]
    else:
        y_pred_final = y_pred
        y_true_final = y_true
        
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    mae = mean_absolute_error(y_true_final, y_pred_final)
    mape = mean_absolute_percentage_error(y_true_final, y_pred_final) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'y_true': y_true_final,
        'y_pred': y_pred_final
    }