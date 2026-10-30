# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChurnLSTM(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        return self.fc(hn[-1])

class ChurnCNN1D(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=32, num_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        out = F.relu(self.conv1(embedded))
        out = F.relu(self.conv2(out))
        out = self.pool(out).squeeze(2)
        return self.fc(out)

class ChurnTransformer(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=32, num_heads=4, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(1000, embedding_dim) # 최대 시퀀스 길이 고려
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        out = self.embedding(x) + self.pos_embedding(positions)
        out = self.transformer_encoder(out)
        
        # 시퀀스 차원에 대해 평균 풀링(Global Average Pooling) 수행
        out = out.mean(dim=1)
        out = self.fc(out)
        return out