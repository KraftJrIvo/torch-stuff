import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import random
import torch
import torch.nn as NN
from torch.nn import functional as F
import matplotlib.pyplot as plt

with open('data/cubexfiles.txt', 'r', encoding='utf-8') as f:
    text = f.read()

N = 1000

import bpeasy
from bpeasy.tokenizer import BPEasyTokenizer

gpt4_regex = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
vocab = bpeasy.train_bpe(iter([text]), gpt4_regex, 16, N)
special_tokens = []
tokenizer = BPEasyTokenizer(vocab, gpt4_regex, [], fill_to_nearest_multiple_of_eight=True, name="skype_msgs_tok")

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)
text_enc = encode(text)
LEN = len(text_enc)

CTX_SZ = 128 #8
xs, ys = [], []
for s in range(LEN - CTX_SZ):
    xs.append(text_enc[s:s+CTX_SZ])
    ys.append(text_enc[s+1:s+CTX_SZ+1])

#tmp = list(zip(xs, ys))
#random.shuffle(tmp)
#xs, ys = zip(*tmp)
#xs, ys = list(xs), list(ys)

n = int(0.9 * LEN)
#xs_trn, ys_trn = torch.tensor(xs[:n], dtype=torch.int64), torch.tensor(ys[:n], dtype=torch.int64)
#xs_val, ys_val = torch.tensor(xs[n:], dtype=torch.int64), torch.tensor(ys[n:], dtype=torch.int64)
xs_trn, ys_trn = xs[:n], ys[:n]
tmp = list(zip(xs_trn, ys_trn))
random.shuffle(tmp)
xs_trn, ys_trn = zip(*tmp)
xs_trn, ys_trn = list(xs_trn), list(ys_trn)
xs_val, ys_val = xs[n:], ys[n:]
tmp = list(zip(xs_val, ys_val))
random.shuffle(tmp)
xs_val, ys_val = zip(*tmp)
xs_val, ys_val = list(xs_val), list(ys_val)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#xs_trn, ys_trn = xs_trn.to(device), ys_trn.to(device)
#xs_val, ys_val = xs_val.to(device), ys_val.to(device)

class SelfAttentionHead(NN.Module):
    def __init__(self, head_sz, n_emb, ctx_sz, dropout):
        super().__init__()
        self.keys = NN.Linear(n_emb, head_sz, bias=False)
        self.queries = NN.Linear(n_emb, head_sz, bias=False)
        self.values = NN.Linear(n_emb, head_sz, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_sz, ctx_sz)))
        self.dropout = NN.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x) #    (B,T,C)
        q = self.queries(x) # (B,T,C)
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)
        v = self.values(x) # (B,T,C)
        out = weights @ v # (B,T,C)
        return out

class SelfAttentionMultiHead(NN.Module):
    def __init__(self, n_heads, head_sz, n_emb, ctx_sz, dropout):
        super().__init__()
        self.heads = NN.ModuleList([SelfAttentionHead(head_sz, n_emb, ctx_sz, dropout) for _ in range(n_heads)])
        self.proj = NN.Linear(n_emb, n_emb)
        self.dropout = NN.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(NN.Module):
    def __init__(self, n_emb, multiplier, dropout):
        super().__init__()
        self.net = NN.Sequential(
            NN.Linear(n_emb, n_emb * multiplier),
            NN.ReLU(),
            NN.Linear(multiplier * n_emb, n_emb),
            NN.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(NN.Module):
    def __init__(self, n_heads, n_emb, ctx_sz, dropout):
        super().__init__()
        head_sz = n_emb // n_heads
        self.sa = SelfAttentionMultiHead(n_heads, head_sz, n_emb, ctx_sz, dropout)
        self.ffwd = FeedForward(n_emb, 4, dropout)
        self.ln1 = NN.LayerNorm(n_emb)
        self.ln2 = NN.LayerNorm(n_emb)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(NN.Module):
    def __init__(self, n_blocks, n_heads, voc_sz, n_emb, ctx_sz, dropout):
        super().__init__()
        self.ctx_sz = ctx_sz
        self.tok_emb_table = NN.Embedding(voc_sz, n_emb)
        self.pos_emb_table = NN.Embedding(ctx_sz, n_emb)
        self.blocks = NN.Sequential(*(
            ([TransformerBlock(n_heads, n_emb, ctx_sz, dropout)] * n_blocks) +
            [NN.LayerNorm(n_emb)]
        ))
        self.lm_head = NN.Linear(n_emb, voc_sz)
    def forward(self, idx, targets=None):
        B, T = idx.shape        
        tok_emb = self.tok_emb_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(B * T))        
        return logits, loss
    def generate(self, idx, max_new_tok):
        for _ in range(max_new_tok):
            idx_crop = idx[:, -self.ctx_sz:]
            logits, _ = self(idx_crop)
            probs = F.softmax(logits[:, -1, :], dim=1)
            idx_nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_nxt), dim=1)
        return idx

print(n)

N_EMB = 512#384 #32
N_HEADS = 8
N_BLOCKS = 8
EVAL_INT = 10
torch.set_float32_matmul_precision('high')
model = TransformerLanguageModel(N_BLOCKS, N_HEADS, N, N_EMB, CTX_SZ, 0.2).to(device)
#model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

@torch.no_grad()
def est_loss(model, d):
    out = {}
    model.eval()
    for splt in ['trn', 'val']:
        lossi = torch.zeros(EVAL_INT)
        for k in range(EVAL_INT):
            bix = torch.randint(d[splt][0].shape[0] - 1, (BATCH_SZ,)).to(device)
            _, loss = model(d[splt][0][bix], d[splt][1][bix])
            lossi[k] = loss.item()
        out[splt] = lossi.mean()
    model.train()
    return out

#model.load_state_dict(torch.load('./cbx777.m'))

BATCH_SZ = 16 #32
lossi = {'trn':[], 'val':[]}
for k in range(10):
    for i in range(100000):
        xb, yb = torch.tensor(xs_trn[i*BATCH_SZ : (i+1)*BATCH_SZ], dtype=torch.int64).to(device), torch.tensor(ys_trn[i*BATCH_SZ : (i+1)*BATCH_SZ], dtype=torch.int64).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % 10000 == 0:
            xb, yb = torch.tensor(xs_trn, dtype=torch.int64).to(device), torch.tensor(ys_trn, dtype=torch.int64).to(device)
            xvb, yvb = torch.tensor(xs_val, dtype=torch.int64).to(device), torch.tensor(ys_val, dtype=torch.int64).to(device)
            est = est_loss(model, {'trn': [xb, yb], 'val': [xvb, yvb]})
            lossi['trn'].append(est['trn'])
            lossi['val'].append(est['val'])
            print(est['trn'].item(), est['val'].item(), "./" + str(k) + '_' + str(i))
            torch.save(model.state_dict(), "./" + str(k) + '_' + str(i) + '.m')
#plt.plot(lossi['trn'])
#plt.plot(lossi['val'])