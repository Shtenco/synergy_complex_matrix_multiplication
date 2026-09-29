from pathlib import Path
import os, math, glob, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
torch.manual_seed(20260929); torch.set_num_threads(min(6,os.cpu_count() or 6))
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256
OUT=Path('/mnt/data'); ck=torch.load(OUT/'qcno_v31_adaptive_avg32_best_final.pt',map_location='cpu',weights_only=False); alloc=ck['alloc']
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt')); raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]
def batch(src,bs=5,g=None):
    ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g); return torch.stack([src[k:k+BLOCK] for k in ix]),torch.stack([src[k+1:k+BLOCK+1] for k in ix])
class Sq(nn.Module):
    def __init__(self,r): super().__init__(); self.a=nn.Parameter(torch.empty(D,int(r))); self.b=nn.Parameter(torch.empty(int(r),D))
    def forward(self,x):
        shp=x.shape; xx=x.reshape(-1,D); xf=torch.fft.rfft(xx,dim=-1); bf=torch.fft.rfft(self.b,dim=-1); c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1); return (c*self.a.T[None,:,:]).sum(1).reshape(*shp[:-1],D)
class Attn(nn.Module):
    def __init__(self,li): super().__init__(); self.q=Sq(alloc[f'L{li}.q']); self.k=Sq(alloc[f'L{li}.k']); self.v=Sq(alloc[f'L{li}.v']); self.o=Sq(alloc[f'L{li}.o'])
    def forward(self,x):
        B,T,C=x.shape; q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2); y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Expand(nn.Module):
    def __init__(self,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc1.{p}']) for p in range(4)])
    def forward(self,x): return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
    def __init__(self,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc2.{p}']) for p in range(4)])
    def forward(self,x):
        xs=x.split(D,-1); y=self.parts[0](xs[0])
        for i in range(1,4): y=y+self.parts[i](xs[i])
        return y
class Block(nn.Module):
    def __init__(self,li): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=Attn(li); self.ln2=nn.LayerNorm(D); self.fc1=Expand(li); self.fc2=Contract(li)
class Model(nn.Module):
    def __init__(self): super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([Block(i) for i in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
m=Model();m.load_state_dict(ck['state_dict']);m.eval()
def fht(x):
    y=x; h=1
    while h<D:
        shp=y.shape; z=y.reshape(*shp[:-1],-1,2,h); a=z[...,0,:]; b=z[...,1,:]; y=torch.cat((a+b,a-b),dim=-1).reshape(shp); h*=2
    return y/math.sqrt(D)
states=[[] for _ in range(4)]; g=torch.Generator().manual_seed(919191)
with torch.no_grad():
  for _ in range(8):
    idx,_=batch(train,5,g); x=m.tok(idx)+m.pos(torch.arange(BLOCK))[None,:,:];p=0
    for b in m.blocks:
      x=x+b.attn(b.ln1(x));states[p].append(x.reshape(-1,D).clone());p+=1
      x=x+b.fc2(F.gelu(b.fc1(b.ln2(x))));states[p].append(x.reshape(-1,D).clone());p+=1
means=[torch.cat(s,0).mean(0) for s in states]; perms=[]; signs=[]
for point,slist in enumerate(states):
    X=torch.cat(slist,0)[:700]-means[point]; best=None; gen=torch.Generator().manual_seed(7000+point); candidates=[(torch.arange(D),torch.ones(D))]
    for c in range(7): candidates.append((torch.randperm(D,generator=gen),(torch.randint(0,2,(D,),generator=gen)*2-1).float()))
    for ci,(perm,sign) in enumerate(candidates):
        z=fht(X[:,perm]*sign); vals,_=torch.topk(z.square(),96,dim=-1,sorted=False); score=float((vals.sum(-1)/(z.square().sum(-1)+1e-12)).mean())
        if best is None or score>best[0]: best=(score,perm,sign,ci)
    perms.append(best[1]);signs.append(best[2]);print('point',point,'best_energy96',best[0],'candidate',best[3],flush=True)
def compress_fast(x,point,k):
    if k>=D:return x
    mu=means[point];perm=perms[point];sign=signs[point]; shp=x.shape; xx=x.reshape(-1,D)-mu; z=fht(xx[:,perm]*sign)
    _,ii=torch.topk(z.abs(),k,dim=-1,sorted=False); zz=torch.zeros_like(z);zz.scatter_(-1,ii,z.gather(-1,ii)); rec=fht(zz)*sign; out=torch.empty_like(rec); out[:,perm]=rec
    return (out+mu).reshape(shp)
def forward(idx,targets,k):
    x=m.tok(idx)+m.pos(torch.arange(idx.shape[1]))[None,:,:];p=0
    for b in m.blocks:
        x=x+b.attn(b.ln1(x));x=compress_fast(x,p,k);p+=1
        x=x+b.fc2(F.gelu(b.fc1(b.ln2(x))));x=compress_fast(x,p,k);p+=1
    z=m.head(m.lnf(x));l=F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1));return z,l
rows=[]
with torch.no_grad():
  for k in [256,128,96,64,32]:
    gg=torch.Generator().manual_seed(424242);ls=[];cor=tot=0;t0=time.perf_counter()
    for _ in range(40):
      x,y=batch(val,5,gg);z,l=forward(x,y,k);ls.append(l.item());cor+=int((z.argmax(-1)==y).sum());tot+=y.numel()
    loss=sum(ls)/len(ls);rows.append({'basis':'signed_permuted_hadamard','k':k,'active_fraction':k/D,'ppl':math.exp(loss),'accuracy':cor/tot,'eval_seconds':time.perf_counter()-t0});print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'qcno_v103_fast_basis_common40.csv',index=False)
