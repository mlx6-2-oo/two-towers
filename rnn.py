import torch.nn as nn

class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 256
        self.rnn = nn.RNN(input_size=312, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 64)  # Final projection to desired output size

    # Input(312) -> RNN(256 hidden) -> Linear (64 output)
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, features] -> [batch, seq_len=1, features]
        
        # Run RNN
        output, _ = self.rnn(x)  # output shape: [batch, seq_len, hidden_size]
        
        # Get final output
        last_output = output[:, -1, :]  # Take last sequence output
        return self.fc(last_output)  # Project to final dimension

class TowerOne(Tower):
    pass

class TowerTwo(Tower):
    pass
