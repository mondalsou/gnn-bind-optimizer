"""SMILES policy loader for GNNBindOptimizer RL agent."""
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

class SMILESTokenizer:
    TWO_CHAR = {"Cl","Br","Si","Se","Na","Li","Mg","Ca","Fe","Cu","Zn","Al","As","@@"}
    PAD,BOS,EOS,UNK = "<PAD>","<BOS>","<EOS>","<UNK>"
    def __init__(self, vocab):
        self.vocab = vocab
        self.c2i = {c:i for i,c in enumerate(vocab)}
        self.i2c = {i:c for i,c in enumerate(vocab)}
        self.pad_idx = self.c2i[self.PAD]; self.bos_idx = self.c2i[self.BOS]
        self.eos_idx = self.c2i[self.EOS]; self.unk_idx = self.c2i[self.UNK]
    def tokenize(self, s):
        tokens, i = [], 0
        while i < len(s):
            two = s[i:i+2]
            if two in self.TWO_CHAR: tokens.append(two); i += 2
            else: tokens.append(s[i]); i += 1
        return tokens
    def encode(self, s):
        return [self.bos_idx] + [self.c2i.get(t, self.unk_idx) for t in self.tokenize(s)] + [self.eos_idx]
    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos_idx: break
            if i in (self.pad_idx, self.bos_idx): continue
            out.append(self.i2c.get(i,"?"))
        return "".join(out)
    def __len__(self): return len(self.vocab)

class SMILESPolicy(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim; self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True,
                             dropout=0.1 if n_layers > 1 else 0.0)
        self.fc    = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(out), hidden

def load_policy(ckpt_path: Path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok  = SMILESTokenizer(ckpt["tokenizer_vocab"])
    cfg  = ckpt["config"]
    pol  = SMILESPolicy(len(tok), cfg["embed_dim"], cfg["hidden"], cfg["layers"]).to(device)
    pol.load_state_dict(ckpt["policy_state_dict"])
    pol.eval()
    return pol, tok
