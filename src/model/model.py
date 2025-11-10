import torch.nn as nn
import torch
from src.config.config import FEATURE_DIM

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10, dropout=0.3, bidirectional=False):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * factor, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, feat)
        out, (hn, cn) = self.rnn(x)
        last = out[:, -1, :]
        return self.classifier(last)

def build_model(num_classes, input_dim=FEATURE_DIM, **kwargs):
    return LSTMClassifier(input_dim=input_dim, num_classes=num_classes, **kwargs)
