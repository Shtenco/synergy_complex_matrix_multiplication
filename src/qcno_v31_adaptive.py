import os, math, time, glob, json, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ap=argparse.ArgumentParser()
ap.add_argument('--steps',type=int,default=45); ap.add_argument('--lr',type=float,default=3e-4)
ap.add_argument('--avg',type=str,default='16'); ap.add_argument('--resume',type=str,default='')
ap.add_argument('--batch',type=int,default=5); ap.add_argument('--objective',choices=['distill','ce'],default='distill')
args=ap.parse_args()
torch.manual_seed(20260929); torch.set_num_threads(min(5,os.cpu_count() or 5))
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256
alloc=json.loads(Path('/mnt/data/qcno_v31_rank_allocations.json').read_text())['allocations'][args.avg]
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'))
raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]

def batch(src,bs=None,g=None):
    bs=bs or args.batch; ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g)
    return torch.stack([src[k:k+BLOCK] for k in ix]),torch.stack([src[k+1:k+BLOCK+1] for k in ix])

class Sq(nn.Module):
    def __init__(self,r):
        super().__init__(); self.r=r; std=(0.02**2/r)**.25
        self.a=nn.Parameter(torch.randn(D,r)*std); self.b=nn.Parameter(torch.randn(r,D)*std)
    def forward(self,x):
        shp=x.shape; xx=x.reshape(-1,D)
        xf=torch.fft.rfft(xx,dim=-1); bf=torch.fft.rfft(self.b,dim=-1)
        c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1)
        return (c*self.a.T[None,:,:]).sum(1).reshape(*shp[:-1],D)

class AAttn(nn.Module):
    def __init__(self,li):
        super().__init__()
        self.q=Sq(alloc[f'L{li}.q']); self.k=Sq(alloc[f'L{li}.k']); self.v=Sq(alloc[f'L{li}.v']); self.o=Sq(alloc[f'L{li}.o'])
    def forward(self,x):
        B,T,C=x.shape
        q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Expand(nn.Module):
    def __init__(self,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc1.{p}']) for p in range(4)])
    def forward(self,x): return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
    def __init__(self,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc2.{p}']) for p in range(4)])
    def forward(self,x):
        xs=x.split(D,-1); y=self.parts[0](xs[0])
        for i in range(1,4): y=y+self.parts[i](xs[i])
        return y
class ABlock(nn.Module):
    def __init__(self,li):
        super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=AAttn(li); self.ln2=nn.LayerNorm(D); self.fc1=Expand(li); self.fc2=Contract(li)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Student(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D)
        self.blocks=nn.ModuleList([ABlock(i) for i in range(LAYERS)]); self.lnf=nn.LayerNorm(D)
        self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None):
        x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:]
        for b in self.blocks: x=b(x)
        logits=self.head(self.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None
        return logits,loss
