import os,math,glob,json
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
torch.manual_seed(20260929);torch.set_num_threads(min(8,os.cpu_count() or 8))
D=512;M=16;N=32;FF=2048;HEADS=8;LAYERS=2;BLOCK=48;VOCAB=256;RK=7
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];data=torch.tensor(list(raw),dtype=torch.long);split=int(.95*len(data));val=data[split:]
def batch(src,bs=5,g=None):
 ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g);x=torch.stack([src[k:k+BLOCK] for k in ix]);y=torch.stack([src[k+1:k+BLOCK+1] for k in ix]);return x,y
def monarch(xx,A,B,g):
 bs=xx.shape[0];y=xx.reshape(bs,M,N).permute(1,0,2).contiguous();y=torch.bmm(y,A).permute(1,0,2).contiguous();y=y.transpose(1,2).contiguous().permute(1,0,2).contiguous();y=torch.bmm(y,B).permute(1,0,2).contiguous();y=y.transpose(1,2).contiguous().reshape(bs,D);return y*g
class H(nn.Module):
 def __init__(self):super().__init__();self.A=nn.Parameter(torch.empty(M,N,N));self.B=nn.Parameter(torch.empty(N,M,M));self.gain=nn.Parameter(torch.empty(D));self.U=nn.Parameter(torch.empty(D,RK));self.V=nn.Parameter(torch.empty(RK,D))
 def forward(self,x):
  shp=x.shape;xx=x.reshape(-1,D);return (monarch(xx,self.A,self.B,self.gain)+(xx@self.U)@self.V).reshape(*shp[:-1],D)
class Expand(nn.Module):
 def __init__(self):super().__init__();self.parts=nn.ModuleList([H() for _ in range(4)])
 def forward(self,x):return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
 def __init__(self):super().__init__();self.parts=nn.ModuleList([H() for _ in range(4)])
 def forward(self,x):
  xs=x.split(D,-1);y=self.parts[0](xs[0])
  for i in range(1,4):y=y+self.parts[i](xs[i])
  return y
class Attn(nn.Module):
 def __init__(self):super().__init__();self.q=H();self.k=H();self.v=H();self.o=H()
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
 def __init__(self):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=Attn();self.ln2=nn.LayerNorm(D);self.fc1=Expand();self.fc2=Contract()
 def forward(self,x):x=x+self.attn(self.ln1(x));return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class LM(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([Block() for _ in range(2)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
 def forward(self,idx,targets):
  x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:]
  for b in self.blocks:x=b(x)
  z=self.head(self.lnf(x));return F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1))
def named_ops(m):
 out=[]
 for li,b in enumerate(m.blocks):
  for n in ['q','k','v','o']:out.append((f'blocks.{li}.attn.{n}',getattr(b.attn,n)))
  for p,q in enumerate(b.fc1.parts):out.append((f'blocks.{li}.fc1.parts.{p}',q))
  for p,q in enumerate(b.fc2.parts):out.append((f'blocks.{li}.fc2.parts.{p}',q))
 return out
m=LM();m.load_state_dict(torch.load('/mnt/data/qcno_v53_monarch_lowrank.pt',map_location='cpu',weights_only=False)['state_dict']);m.eval()
g=torch.Generator().manual_seed(424242);batches=[batch(val,5,g) for _ in range(8)]
@torch.no_grad()
def loss():return sum(float(m(x,y)) for x,y in batches)/len(batches)
base=loss();rows=[]
for name,op in named_ops(m):
 U=op.U.detach().clone();V=op.V.detach().clone()
 with torch.no_grad(): op.U.zero_();op.V.zero_()
 l=loss()
 with torch.no_grad(): op.U.copy_(U);op.V.copy_(V)
 delta=l-base;rows.append({'op':name,'base_loss':base,'ablated_loss':l,'delta_loss':delta,'residual_norm':float((U@V).norm())})
minr,maxr,budget=3,16,168;alloc={r['op']:minr for r in rows};scores={r['op']:max(r['delta_loss'],1e-6) for r in rows}
for _ in range(budget-minr*len(rows)):
 candidates=[n for n in alloc if alloc[n]<maxr];n=max(candidates,key=lambda z:scores[z]/(alloc[z]+1));alloc[n]+=1
out={'baseline_loss':base,'budget':budget,'min_rank':min(alloc.values()),'max_rank':max(alloc.values()),'alloc':alloc,'sensitivities':rows}
Path('/mnt/data/qcno_v56_sensitivity_alloc.json').write_text(json.dumps(out,indent=2))
