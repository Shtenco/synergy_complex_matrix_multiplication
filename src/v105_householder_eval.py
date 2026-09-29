from pathlib import Path
import os, math, glob, time, torch
import torch.nn as nn, torch.nn.functional as F
import pandas as pd
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256; KREF=24
OUT=Path('/mnt/data');torch.manual_seed(20260929);torch.set_num_threads(min(6,os.cpu_count() or 6))
ck=torch.load(OUT/'qcno_v31_adaptive_avg32_best_final.pt',map_location='cpu',weights_only=False);alloc=ck['alloc']
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];data=torch.tensor(list(raw),dtype=torch.long);val=data[int(.95*len(data)):]
def batch(src,bs=5,g=None):
 ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g);return torch.stack([src[k:k+BLOCK] for k in ix]),torch.stack([src[k+1:k+BLOCK+1] for k in ix])
class Sq(nn.Module):
 def __init__(self,r):super().__init__();self.a=nn.Parameter(torch.empty(D,int(r)));self.b=nn.Parameter(torch.empty(int(r),D))
 def forward(self,x):
  shp=x.shape;xx=x.reshape(-1,D);xf=torch.fft.rfft(xx,dim=-1);bf=torch.fft.rfft(self.b,dim=-1);c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1);return(c*self.a.T[None,:,:]).sum(1).reshape(*shp[:-1],D)
class Attn(nn.Module):
 def __init__(self,li):super().__init__();self.q=Sq(alloc[f'L{li}.q']);self.k=Sq(alloc[f'L{li}.k']);self.v=Sq(alloc[f'L{li}.v']);self.o=Sq(alloc[f'L{li}.o'])
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Expand(nn.Module):
 def __init__(self,li):super().__init__();self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc1.{p}']) for p in range(4)])
 def forward(self,x):return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
 def __init__(self,li):super().__init__();self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc2.{p}']) for p in range(4)])
 def forward(self,x):
  xs=x.split(D,-1);y=self.parts[0](xs[0])
  for i in range(1,4):y=y+self.parts[i](xs[i])
  return y
class Block(nn.Module):
 def __init__(self,li):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=Attn(li);self.ln2=nn.LayerNorm(D);self.fc1=Expand(li);self.fc2=Contract(li)
class M(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([Block(i) for i in range(LAYERS)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
m=M();m.load_state_dict(ck['state_dict']);m.eval()
class HH(nn.Module):
 def __init__(self,state):super().__init__();self.v=nn.Parameter(torch.empty(KREF,D),requires_grad=False);self.load_state_dict(state)
 def fwd(self,x):
  y=x
  for r in range(KREF):
   v=self.v[r];vn=v/(v.norm()+1e-8);y=y-2*(y@vn)[:,None]*vn[None,:]
  return y
 def inv(self,x):
  y=x
  for r in range(KREF-1,-1,-1):
   v=self.v[r];vn=v/(v.norm()+1e-8);y=y-2*(y@vn)[:,None]*vn[None,:]
  return y
hhs=[];means=[]
for p in range(4):
 c=torch.load(OUT/f'qcno_v105_hh_point{p}.pt',map_location='cpu',weights_only=False);hhs.append(HH(c['state_dict']).eval());means.append(c['mean'])
def comp(x,p,k):
 if k>=D:return x
 shp=x.shape;xx=x.reshape(-1,D)-means[p];z=hhs[p].fwd(xx);_,ii=torch.topk(z.abs(),k,dim=-1,sorted=False);zz=torch.zeros_like(z);zz.scatter_(-1,ii,z.gather(-1,ii));return(hhs[p].inv(zz)+means[p]).reshape(shp)
def forward(idx,y,k):
 x=m.tok(idx)+m.pos(torch.arange(idx.shape[1]))[None,:,:];p=0
 for b in m.blocks:
  x=x+b.attn(b.ln1(x));x=comp(x,p,k);p+=1
  x=x+b.fc2(F.gelu(b.fc1(b.ln2(x))));x=comp(x,p,k);p+=1
 z=m.head(m.lnf(x));return z,F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1))
rows=[]
with torch.no_grad():
 for k in [256,128,96,64,32]:
  g=torch.Generator().manual_seed(424242);ls=[];cor=tot=0;t0=time.perf_counter()
  for _ in range(40):
   x,y=batch(val,5,g);z,l=forward(x,y,k);ls.append(l.item());cor+=int((z.argmax(-1)==y).sum());tot+=y.numel()
  loss=sum(ls)/len(ls);rows.append({'basis':'householder24','k':k,'active_fraction':k/D,'ppl':math.exp(loss),'accuracy':cor/tot,'eval_seconds':time.perf_counter()-t0});print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'qcno_v105_householder_common40.csv',index=False)
