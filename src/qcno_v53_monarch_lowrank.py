import os,math,time,glob,json,argparse
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=180);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--batch',type=int,default=5);ap.add_argument('--rk',type=int,default=7);ap.add_argument('--resume',default='');ap.add_argument('--objective',choices=['distill','ce'],default='distill');args=ap.parse_args()
torch.manual_seed(20260929);torch.set_num_threads(min(8,os.cpu_count() or 8))
D=512;M=16;N=32;FF=2048;HEADS=8;LAYERS=2;BLOCK=48;VOCAB=256;RK=args.rk
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];data=torch.tensor(list(raw),dtype=torch.long);split=int(.95*len(data));train=data[:split];val=data[split:]
def batch(src,bs=None,g=None):
 bs=bs or args.batch;ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g);x=torch.stack([src[k:k+BLOCK] for k in ix]);y=torch.stack([src[k+1:k+BLOCK+1] for k in ix]);return x,y
class DAttn(nn.Module):
 def __init__(self):super().__init__();self.q=nn.Linear(D,D,bias=False);self.k=nn.Linear(D,D,bias=False);self.v=nn.Linear(D,D,bias=False);self.o=nn.Linear(D,D,bias=False)
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class DBlock(nn.Module):
 def __init__(self):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=DAttn();self.ln2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF,bias=False);self.fc2=nn.Linear(FF,D,bias=False)
 def forward(self,x):x=x+self.attn(self.ln1(x));return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([DBlock() for _ in range(LAYERS)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
 def forward(self,idx,targets=None,return_hidden=False):
  x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:];hs=[]
  for b in self.blocks:x=b(x);hs.append(x)
  z=self.head(self.lnf(x));loss=F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None;return (z,loss,hs) if return_hidden else (z,loss)
class HybridSquare(nn.Module):
 def __init__(self):
  super().__init__();self.A=nn.Parameter(torch.randn(M,N,N)/math.sqrt(N));self.B=nn.Parameter(torch.randn(N,M,M)/math.sqrt(M));self.gain=nn.Parameter(torch.full((D,),0.45));self.U=nn.Parameter(torch.randn(D,RK)*0.01);self.V=nn.Parameter(torch.zeros(RK,D))
 def forward(self,x):
  shp=x.shape;xx=x.reshape(-1,D);y=xx.reshape(-1,M,N);y=torch.einsum('bmn,mnk->bmk',y,self.A);y=y.transpose(1,2);y=torch.einsum('bnm,nmk->bnk',y,self.B);y=y.transpose(1,2).contiguous().reshape(-1,D);y=y*self.gain + (xx@self.U)@self.V;return y.reshape(*shp[:-1],D)
class Expand(nn.Module):
 def __init__(self):super().__init__();self.parts=nn.ModuleList([HybridSquare() for _ in range(4)])
 def forward(self,x):return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
 def __init__(self):super().__init__();self.parts=nn.ModuleList([HybridSquare() for _ in range(4)])
 def forward(self,x):
  xs=x.split(D,-1);y=self.parts[0](xs[0])
  for i in range(1,4):y=y+self.parts[i](xs[i])
  return y
class Attn(nn.Module):
 def __init__(self):super().__init__();self.q=HybridSquare();self.k=HybridSquare();self.v=HybridSquare();self.o=HybridSquare()
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
 def __init__(self):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=Attn();self.ln2=nn.LayerNorm(D);self.fc1=Expand();self.fc2=Contract()
 def forward(self,x):x=x+self.attn(self.ln1(x));return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Student(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
 def forward(self,idx,targets=None,return_hidden=False):
  x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:];hs=[]
  for b in self.blocks:x=b(x);hs.append(x)
  z=self.head(self.lnf(x));loss=F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None;return (z,loss,hs) if return_hidden else (z,loss)
@torch.no_grad()
def evalm(m,iters=15):
 m.eval();g=torch.Generator().manual_seed(777);ls=[];c=tot=0
 for _ in range(iters):
  x,y=batch(val,5,g);z,l=m(x,y);ls.append(l.item());c+=int((z.argmax(-1)==y).sum());tot+=y.numel()
 m.train();l=sum(ls)/len(ls);return l,math.exp(l),c/tot
t=Teacher();t.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']);t.eval();[p.requires_grad_(False) for p in t.parameters()]
s=Student()
if args.resume:
 base=torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict'];s.load_state_dict(base,strict=False)
else:
 base=torch.load('/mnt/data/qcno_v52_monarch.pt',map_location='cpu',weights_only=False)['state_dict'];s.load_state_dict(base,strict=False)
opt=torch.optim.AdamW(s.parameters(),lr=args.lr,weight_decay=.01);Tt=2.;st=time.perf_counter()
for step in range(1,args.steps+1):
 x,y=batch(train)
 if args.objective=='distill':
  with torch.no_grad():tz,_,th=t(x,return_hidden=True)
  sz,ce,sh=s(x,y,return_hidden=True);kd=F.kl_div(F.log_softmax(sz/Tt,-1),F.softmax(tz/Tt,-1),reduction='batchmean')*(Tt*Tt)/BLOCK;hm=sum(F.mse_loss(F.layer_norm(a,(D,)),F.layer_norm(b,(D,))) for a,b in zip(sh,th))/len(sh);loss=.35*ce+.4*kd+.25*hm
 else:
  sz,ce=s(x,y);loss=ce
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1.0);opt.step()
l,p,a=evalm(s,15);tp=evalm(t,15)[1];dense=6291456;perop=M*N*N+N*M*M+D+2*D*RK;structured=24*perop;res={'rk':RK,'steps_this_run':args.steps,'byte_ppl':p,'teacher_ppl':tp,'ppl_ratio':p/tp,'acc':a,'structured_linear_params':structured,'dense_linear_params':dense,'linear_compression_x':dense/structured,'seconds':time.perf_counter()-st}
torch.save({'state_dict':s.state_dict(),'result':res},'/mnt/data/qcno_v53_monarch_lowrank.pt')
