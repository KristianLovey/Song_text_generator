import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import CharRNN

class CharDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len
    def __len__(self):
        return len(self.ids) - self.seq_len - 1
    def __getitem__(self, i):
        x = self.ids[i:i+self.seq_len]
        y = self.ids[i+1:i+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.npz")
    ap.add_argument("--vocab", default="data/processed/vocab.json")
    ap.add_argument("--cell", choices=["lstm","gru"], default="lstm")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    z = np.load(args.data)
    ids = z["ids"]
    seq_len = int(z["seq_len"])

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    vocab_size = len(vocab["stoi"])

    ds = CharDataset(ids, seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharRNN(vocab_size, args.embed_dim, args.hidden_dim, args.num_layers, args.dropout, cell=args.cell).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(1, args.epochs+1):
        model.train()
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device).long(), y.to(device).long()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.numel()
            n += x.numel()
        avg = tot / n
        print(f"epoch {e} loss {avg:.4f} ppl {math.exp(avg):.2f}")

    Path("models").mkdir(exist_ok=True)
    save_path = args.save or f"models/{args.cell}_last.pt"
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "cfg": vars(args)}, save_path)
    print("SAVED:", save_path)

if __name__ == "__main__":
    main()
