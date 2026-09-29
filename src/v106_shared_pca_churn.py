from pathlib import Path
import os,math,glob,time,torch
import torch.nn as nn, torch.nn.functional as F
import pandas as pd
D=512;FF=2048;HEADS=8;LAYERS=2;BLOCK=48;VOCAB=256
OUT=Path('/mnt/data');torch.manual_seed(20260929);torch.set_num_threads(min(6,os.cpu_count() or 6))
ck=torch.load(OUT/'qcno_v31_adaptive_avg32_best_final.pt',map_location='cpu',weights_only=False);alloc=ck['alloc']
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];data=torch.tensor(list(raw),dtype=torch.long);split=int(.95*len(data));train=data[:split];val=data[split:]
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
S=[];g=torch.Generator().manual_seed(919191)
with torch.no_grad():
 for _ in range(12):
  idx,_=batch(train,5,g);x=m.tok(idx)+m.pos(torch.arange(BLOCK))[None,:,:]
  for b in m.blocks:
   x=x+b.attn(b.ln1(x));S.append(x.reshape(-1,D).clone())
   x=x+b.fc2(F.gelu(b.fc1(b.ln2(x))));S.append(x.reshape(-1,D).clone())
X=torch.cat(S,0);mu=X.mean(0);Xc=X-mu;cov=(Xc.T@Xc)/(Xc.shape[0]-1);vals,U=torch.linalg.eigh(cov);order=torch.argsort(vals,descending=True);U=U[:,order];vals=vals[order]
def comp(x,k):
 if k>=D:return x
 shp=x.shape;c=(x.reshape(-1,D)-mu)@U;_,ii=torch.topk(c.abs(),k,dim=-1,sorted=False);cc=torch.zeros_like(c);cc.scatter_(-1,ii,c.gather(-1,ii));return(cc@U.T+mu).reshape(shp)
def forward(idx,y,k,return_support=False):
 x=m.tok(idx)+m.pos(torch.arange(idx.shape[1]))[None,:,:]; sups=[]
 for b in m.blocks:
  x=x+b.attn(b.ln1(x))
  if return_support:
   c=(x.reshape(-1,D)-mu)@U;sups.append(torch.topk(c.abs(),k,dim=-1,sorted=False).indices)
  x=comp(x,k);x=x+b.fc2(F.gelu(b.fc1(b.ln2(x))))
  if return_support:
   c=(x.reshape(-1,D)-mu)@U;sups.append(torch.topk(c.abs(),k,dim=-1,sorted=False).indices)
  x=comp(x,k)
 z=m.head(m.lnf(x));l=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1));return z,l,sups
rows=[]; churn=[]
with torch.no_grad():
 for k in [256,128,96,64]:
  g=torch.Generator().manual_seed(424242);ls=[];cor=tot=0;overlaps=[[],[],[]];t0=time.perf_counter()
  for bi in range(40):
   x,y=batch(val,5,g);z,l,sups=forward(x,y,k,True);ls.append(l.item());cor+=int((z.argmax(-1)==y).sum());tot+=y.numel()
   if bi<10:
    for p in range(3):
     A=sups[p];B=sups[p+1];ov=(A[:,:,None]==B[:,None,:]).any(-1).float().sum(-1)/k;overlaps[p].append(ov.mean().item())
  loss=sum(ls)/len(ls);rows.append({'k':k,'active_fraction':k/D,'ppl':math.exp(loss),'accuracy':cor/tot,'eval_seconds':time.perf_counter()-t0})
  for p in range(3):churn.append({'k':k,'transition':f'{p}->{p+1}','retained_support_fraction':sum(overlaps[p])/len(overlaps[p]),'churn_fraction':1-sum(overlaps[p])/len(overlaps[p])})
  print(rows[-1],churn[-3:],flush=True)
pd.DataFrame(rows).to_csv(OUT/'qcno_v106_shared_pca_common40.csv',index=False);pd.DataFrame(churn).to_csv(OUT/'qcno_v106_support_churn.csv',index=False)
pd.DataFrame([{'k':k,'pca_variance_fraction':float(torch.cumsum(vals,0)[k-1]/vals.sum())} for k in [64,96,128,256]]).to_csv(OUT/'qcno_v106_shared_pca_energy.csv',index=False)
