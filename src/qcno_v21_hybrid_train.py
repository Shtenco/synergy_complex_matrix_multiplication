import os, math, time, glob, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
SEED=20260929; torch.manual_seed(SEED); torch.set_num_threads(min(5,os.cpu_count() or 4))
VOCAB=256; BLOCK=64; D=128; HEADS=4; LAYERS=2; FF=512; BATCH=48
R=int(os.environ.get('QCNO_R','8')); K=int(os.environ.get('QCNO_K','8')); STEPS=int(os.environ.get('QCNO_STEPS','150')); LR=float(os.environ.get('QCNO_LR','0.0003'))
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt')); raw=b'

'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]
i=torch.arange(D)[:,None]; j=torch.arange(D)[None,:]; J=j.expand(D,D); S=(j-i)%D
class HybridSquare(nn.Module):
    def __init__(self):
        super().__init__(); sq=(0.02**2/max(R,1))**0.25
        self.a=nn.Parameter(torch.randn(D,R)*sq); self.b=nn.Parameter(torch.randn(R,D)*sq)
        sl=(0.02**2/max(K,1))**0.25
        self.u=nn.Parameter(torch.randn(D,K)*sl); self.v=nn.Parameter(torch.randn(K,D)*sl)
    def weight(self):
        Z=self.a@self.b; A=Z[J,S]
        A=A+self.u@self.v
        return A.T.contiguous()
    def forward(self,x): return F.linear(x,self.weight())
class Expand(nn.Module):
    def __init__(self): super().__init__(); self.parts=nn.ModuleList([HybridSquare() for _ in range(4)])
    def forward(self,x): return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
    def __init__(self): super().__init__(); self.parts=nn.ModuleList([HybridSquare() for _ in range(4)])
    def forward(self,x):
        xs=x.split(D,-1); y=self.parts[0](xs[0])
        for z,p in zip(xs[1:],self.parts[1:]): y=y+p(z)
        return y
class Attn(nn.Module):
    def __init__(self): super().__init__(); self.q=HybridSquare();self.k=HybridSquare();self.v=HybridSquare();self.o=HybridSquare()
    def forward(self,x):
        B,T,C=x.shape; q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class BlockM(nn.Module):
    def __init__(self): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=Attn(); self.ln2=nn.LayerNorm(D); self.fc1=Expand();self.fc2=Contract()
    def forward(self,x): x=x+self.attn(self.ln1(x)); x=x+self.fc2(F.gelu(self.fc1(self.ln2(x)))); return x
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([BlockM() for _ in range(LAYERS)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
        nn.init.normal_(self.tok.weight,0,.02);nn.init.normal_(self.pos.weight,0,.02)
        for m in self.modules():
            if isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        B,T=idx.shape;x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]
        for b in self.blocks:x=b(x)
        logits=self.head(self.lnf(x));loss=None
        if targets is not None:loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1))
        return logits,loss
model=Model(); resume=os.environ.get('QCNO_RESUME','')
if resume: model.load_state_dict(torch.load(resume,map_location='cpu',weights_only=False)['state_dict'])
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.01)
def batch(src,bs=BATCH,g=None):
    ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g);return torch.stack([src[q:q+BLOCK] for q in ix]),torch.stack([src[q+1:q+BLOCK+1] for q in ix])
@torch.no_grad()
def ev(iters=60):
    model.eval();g=torch.Generator().manual_seed(777);ls=[];cor=tot=0
    for _ in range(iters):
        x,y=batch(val,32,g);z,l=model(x,y);ls.append(l.item());cor+=int((z.argmax(-1)==y).sum());tot+=y.numel()
    model.train();l=sum(ls)/len(ls);return l,math.exp(l),cor/tot
params=sum(p.numel() for p in model.parameters()); target=24*2*D*(R+K)
print(json.dumps({'R':R,'K':K,'params':params,'target_params':target,'linear_compression':393216/target}),flush=True)
t=time.perf_counter()
for s in range(1,STEPS+1):
    x,y=batch(train);_,l=model(x,y);opt.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    if s%100==0 or s==1:
        vl,p,a=ev(10);print(f'step={s} train={l.item():.4f} val={vl:.4f} ppl={p:.3f} acc={a:.4f}',flush=True)
vl,p,a=ev();res={'R':R,'K':K,'steps':STEPS,'val_loss':vl,'byte_ppl':p,'next_byte_accuracy':a,'params':params,'target_params':target,'linear_compression_x':393216/target,'total_compression_vs_dense_baseline_x':435456/params,'train_seconds':time.perf_counter()-t}
print('FINAL',json.dumps(res),flush=True);Path(f'/mnt/data/qcno_v21_hybrid_R{R}_K{K}_metrics.json').write_text(json.dumps(res,indent=2));torch.save({'state_dict':model.state_dict(),'result':res},f'/mnt/data/qcno_v21_hybrid_R{R}_K{K}.pt')
