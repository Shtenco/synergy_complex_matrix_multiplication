from pathlib import Path
import os, glob, sys, math, torch
import torch.nn as nn, torch.nn.functional as F
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256; KREF=24
point=int(sys.argv[1]); OUT=Path('/mnt/data')
torch.manual_seed(20260929+point); torch.set_num_threads(min(6,os.cpu_count() or 6))
ck=torch.load(OUT/'qcno_v31_adaptive_avg32_best_final.pt',map_location='cpu',weights_only=False);alloc=ck['alloc']
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];data=torch.tensor(list(raw),dtype=torch.long);train=data[:int(.95*len(data))]
def batch(src,bs=5,g=None):
 ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g);return torch.stack([src[k:k+BLOCK] for k in ix])
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
 for _ in range(10):
  idx=batch(train,5,g);x=m.tok(idx)+m.pos(torch.arange(BLOCK))[None,:,:];p=0
  for b in m.blocks:
   x=x+b.attn(b.ln1(x))
   if p==point:S.append(x.reshape(-1,D).clone())
   p+=1
   x=x+b.fc2(F.gelu(b.fc1(b.ln2(x))))
   if p==point:S.append(x.reshape(-1,D).clone())
   p+=1
X=torch.cat(S,0);mu=X.mean(0);X=X-mu;X=X[:1800]
class HH(nn.Module):
 def __init__(self):super().__init__();self.v=nn.Parameter(torch.randn(KREF,D)/math.sqrt(D))
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
hh=HH();opt=torch.optim.Adam(hh.parameters(),lr=.03);gen=torch.Generator().manual_seed(777+point)
for step in range(1,71):
 ix=torch.randint(0,X.shape[0],(160,),generator=gen);z=hh.fwd(X[ix]);total=z.square().sum(-1)+1e-12;score=0
 for k in (64,96,128):score+=torch.topk(z.square(),k,dim=-1,sorted=False).values.sum(-1)/total
 loss=-score.mean()/3;opt.zero_grad();loss.backward();opt.step()
 if step in (1,20,40,70):
  with torch.no_grad():
   zz=hh.fwd(X[:500]);en=[float((torch.topk(zz.square(),k,dim=-1,sorted=False).values.sum(-1)/(zz.square().sum(-1)+1e-12)).mean()) for k in (64,96,128)]
  print(point,step,en,flush=True)
torch.save({'mean':mu,'state_dict':hh.state_dict(),'KREF':KREF},OUT/f'qcno_v105_hh_point{point}.pt')
