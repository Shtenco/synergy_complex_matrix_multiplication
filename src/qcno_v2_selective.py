# Selective Q-CNO post-hoc compression on the 600-step dense baseline.
import os,math,time,glob,copy,json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd

torch.set_num_threads(min(5,os.cpu_count() or 4)); VOCAB=256;BLOCK=64;D=128;HEADS=4;LAYERS=2;FF=512
class A(nn.Module):
 def __init__(self):super().__init__();self.q=nn.Linear(D,D,bias=False);self.k=nn.Linear(D,D,bias=False);self.v=nn.Linear(D,D,bias=False);self.o=nn.Linear(D,D,bias=False)
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class B(nn.Module):
 def __init__(self):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=A();self.ln2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF,bias=False);self.fc2=nn.Linear(FF,D,bias=False)
 def forward(self,x):x=x+self.attn(self.ln1(x));x=x+self.fc2(F.gelu(self.fc1(self.ln2(x))));return x
class M(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([B() for _ in range(LAYERS)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
 def forward(self,idx,targets=None):
  B,T=idx.shape;x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]
  for b in self.blocks:x=b(x)
  z=self.head(self.lnf(x));l=F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None;return z,l
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];d=torch.tensor(list(raw),dtype=torch.long);val=d[int(.95*len(d)):]
base=M();base.load_state_dict(torch.load('/mnt/data/qcno_v2_baseline_600.pt',map_location='cpu',weights_only=False)['state_dict']);base.eval()
i=torch.arange(D)[:,None];j=torch.arange(D)[None,:];J=j.expand(D,D);S=(j-i)%D
def fac(A):
 jj=torch.arange(D)[:,None];ss=torch.arange(D)[None,:];ii=(jj-ss)%D;Z=A.double()[ii,jj.expand(D,D)]
 U,sv,Vh=torch.linalg.svd(Z,full_matrices=False);return (U*sv[None,:]).float(),Vh.float()
def app(f,R):
 Z=f[0][:,:R]@f[1][:R,:];A0=torch.empty_like(Z);jj=torch.arange(D)[:,None];ss=torch.arange(D)[None,:];ii=(jj-ss)%D;A0[ii,jj.expand(D,D)]=Z;return A0
def parts(w):
 A0=w.T.contiguous();ni,no=A0.shape
 if ni==D and no==D:return [A0], 'sq'
 if ni==D:return [A0[:,k*D:(k+1)*D] for k in range(4)],'out'
 return [A0[k*D:(k+1)*D,:] for k in range(4)],'in'
def join(xs,k):return xs[0] if k=='sq' else torch.cat(xs,1 if k=='out' else 0)
def mod(m,l,n):
 b=m.blocks[l];return {'q':b.attn.q,'k':b.attn.k,'v':b.attn.v,'o':b.attn.o,'fc1':b.fc1,'fc2':b.fc2}[n]
T=[]
for l,b in enumerate(base.blocks):
 for n,mm in [('q',b.attn.q),('k',b.attn.k),('v',b.attn.v),('o',b.attn.o),('fc1',b.fc1),('fc2',b.fc2)]:
  ps,k=parts(mm.weight);T.append((l,n,k,[fac(x) for x in ps],ps))
@torch.no_grad()
def ev(m,it=40):
 g=torch.Generator().manual_seed(777);ls=[];c=t=0
 for _ in range(it):
  ix=torch.randint(0,len(val)-BLOCK-1,(32,),generator=g);x=torch.stack([val[q:q+BLOCK] for q in ix]);y=torch.stack([val[q+1:q+BLOCK+1] for q in ix]);z,l=m(x,y);ls.append(l.item());c+=int((z.argmax(-1)==y).sum());t+=y.numel()
 L=sum(ls)/len(ls);return L,math.exp(L),c/t
bl,bp,ba=ev(base,60);print('base',bp,ba)
rows=[]
for group in ['attention','mlp','both']:
 for R in [8,16,32,48]:
  m=copy.deepcopy(base);factor=0;dense_kept=0
  for l,n,k,fs,ps in T:
   chosen=(group=='both') or (group=='attention' and n in ['q','k','v','o']) or (group=='mlp' and n in ['fc1','fc2'])
   if chosen:
    Ar=join([app(f,R) for f in fs],k);mod(m,l,n).weight.data.copy_(Ar.T);factor+=len(fs)*2*D*R
   else:dense_kept+=mod(m,l,n).weight.numel()
  L,p,a=ev(m,35);stored=dense_kept+factor;rows.append({'group':group,'R':R,'ppl':p,'acc':a,'linear_storage_params':stored,'linear_compression_x':393216/stored,'ppl_ratio':p/bp});print(group,R,p,a,393216/stored,flush=True)
pd.DataFrame(rows).to_csv('/mnt/data/qcno_v2_selective_results.csv',index=False)
