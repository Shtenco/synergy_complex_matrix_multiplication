import os, math, time, glob, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=20260929
torch.manual_seed(SEED)
torch.set_num_threads(min(16,os.cpu_count() or 4))
DEVICE='cpu'; VOCAB=256; BLOCK=64; D=128; HEADS=4; LAYERS=2; FF=512; BATCH=48
STEPS=int(os.environ.get('QCNO_STEPS','600')); LR=3e-4
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'))
raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__(); self.q=nn.Linear(D,D,bias=False); self.k=nn.Linear(D,D,bias=False); self.v=nn.Linear(D,D,bias=False); self.o=nn.Linear(D,D,bias=False)
    def forward(self,x):
        B,T,C=x.shape
        q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=CausalSelfAttention(); self.ln2=nn.LayerNorm(D); self.fc1=nn.Linear(D,FF,bias=False); self.fc2=nn.Linear(FF,D,bias=False)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class TinyLM(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D)
        self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)]); self.lnf=nn.LayerNorm(D)
        self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None):
        B,T=idx.shape; x=self.tok(idx)+self.pos(torch.arange(T,device=idx.device))[None,:,:]
        for b in self.blocks: x=b(x)
        logits=self.head(self.lnf(x))
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None
        return logits,loss
model=TinyLM().to(DEVICE)
def _init(m):
    if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,mean=0.0,std=0.02)
    if isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
model.apply(_init)
resume=os.environ.get('QCNO_RESUME','')
if resume: model.load_state_dict(torch.load(resume,map_location='cpu',weights_only=False)['state_dict'])
opt=torch.optim.AdamW(model.parameters(),lr=float(os.environ.get('QCNO_LR',LR)),weight_decay=.01)
def batch(src,batch_size=BATCH):
    ix=torch.randint(0,len(src)-BLOCK-1,(batch_size,))
    return torch.stack([src[i:i+BLOCK] for i in ix]), torch.stack([src[i+1:i+BLOCK+1] for i in ix])
@torch.no_grad()
def eval_loss(src,iters=40):
    model.eval(); vals=[]
    for _ in range(iters):
        x,y=batch(src,32); _,loss=model(x,y); vals.append(loss.item())
    model.train(); return sum(vals)/len(vals)
print(json.dumps({'corpus_bytes':len(data),'train_bytes':len(train),'val_bytes':len(val),'params':sum(p.numel() for p in model.parameters()),'steps':STEPS}),flush=True)
t0=time.perf_counter()
for step in range(1,STEPS+1):
    x,y=batch(train); _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%100==0 or step==1:
        vl=eval_loss(val,10); print(f'step={step} train={loss.item():.4f} val={vl:.4f} ppl={math.exp(vl):.3f}',flush=True)
vl=eval_loss(val,80)
result={'val_loss':vl,'val_ppl':math.exp(vl),'train_seconds':time.perf_counter()-t0,'params':sum(p.numel() for p in model.parameters())}
print('FINAL',json.dumps(result),flush=True)
torch.save({'state_dict':model.state_dict(),'result':result},'/mnt/data/qcno_v2_baseline.pt')
