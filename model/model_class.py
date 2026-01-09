import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout manual para a saída se for layer única (PyTorch bug fix)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Inicializa hidden state e cell state com zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Pega apenas o último time step
        out = self.dropout_layer(out[:, -1, :])
        out = self.fc(out)
        return out