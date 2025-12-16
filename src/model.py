import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, cell="lstm"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.cell = cell

    def forward(self, x, hx=None):
        x = self.emb(x)
        out, hx = self.rnn(x, hx)
        logits = self.fc(out)
        return logits, hx
