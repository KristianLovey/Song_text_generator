import argparse, json
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    in_dir = Path(args.input)
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in in_dir.rglob("*.txt"))
    text = text.lower().strip()

    chars = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}

    ids = np.array([stoi[ch] for ch in text], dtype=np.int64)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir/"dataset.npz", ids=ids, seq_len=args.seq_len)
    (out_dir/"vocab.json").write_text(json.dumps({"stoi": stoi, "itos": itos}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK:", out_dir)

if __name__ == "__main__":
    main()
