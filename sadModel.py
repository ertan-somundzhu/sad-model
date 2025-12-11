
import torch
from torch import nn

class sadModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_layers=1, output_dim=800):
        super(sadModel, self).__init__()
        
        # GRU expects input: (seq_len, batch, input_size)
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2 * 400, output_dim)  # 2 for bidirectional
    
    def forward(self, x):
        # x: (batch, 1, 40, 400) -> remove channel dim and permute
        x = x.squeeze(1).permute(0, 2, 1)  # (batch, 400, 40)
        
        # pass through gru
        out, _ = self.gru(x)  # out: (batch, 400, hidden_dim*2)
        
        # flatten time dimension
        out = out.contiguous().view(out.size(0), -1)  # (batch, 400*hidden_dim*2)
        
        out = self.fc(out)  # (batch, 800)

        return out
