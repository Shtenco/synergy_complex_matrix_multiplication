import os,math,time,glob,json,argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=60);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--resume',default='');ap.add_argument('--batch',type=int,default=5);args=ap.parse_args()
torch.manual_seed(20260929);torch.set_num_threads(min(5,os.cpu_count() or 5))
D=512;FF=2048;HEADS=8;LAYERS=2;BLOCK=48;VOCAB=256;STAGES=int(math.log2(D));assert 2**STAGES==D
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
class ButterflySquare(nn.Module):
 def __init__(self):
  super().__init__();self.ws=nn.ParameterList()
  for s in range(STAGES):
   stride=1<<s;groups=D//(2*stride);w=torch.zeros(groups,stride,2,2);w[...,0,0]=1.;w[...,1,1]=1.;w+=0.01*torch.randn_like(w);self.ws.append(nn.Parameter(w))
  self.gain=nn.Parameter(torch.full((D,),0.40))
 def forward(self,x):
  shp=x.shape;y=x.reshape(-1,D)
  for s,w in enumerate(self.ws):
   stride=1<<s;groups=D//(2*stride);z=y.reshape(-1,groups,2,stride).permute(0,1,3,2);z=torch.einsum('bgsi,gsio->bgso',z,w);y=z.permute(0,1,3,2).reshape(-1,D)
  y=y*self.gain
  return y.reshape(*shp[:-1],D)
class BExpand(nn.Module):
 def __init__(self):super().__init__();self.parts=nn.ModuleList([ButterflySquare() for _ in range(4)])
 def forward(self,x):return torch.cat([p(x) for p in self.parts],-1)
class BContract(nn.Module):
 def __init__(self):super().__init__();self.parts=nn.ModuleList([ButterflySquare() for _ in range(4)])
 def forward(self,x):
  xs=x.split(D,-1);y=self.parts[0](xs[0])
  for i in range(1,4):y=y+self.parts[i](xs[i])
  return y
class BAttn(nn.Module):
 def __init__(self):super().__init__();self.q=ButterflySquare();self.k=ButterflySquare();self.v=ButterflySquare();self.o=ButterflySquare()
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class BBlock(nn.Module):
 def __init__(self):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=BAttn();self.ln2=nn.LayerNorm(D);self.fc1=BExpand();self.fc2=BContract()
 def forward(self,x):x=x+self.attn(self.ln1(x));return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Student(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([BBlock() for _ in range(LAYERS)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
 def forward(self,idx,targets=None,return_hidden=False):
  x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:];hs=[]
  for b in self.blocks:x=b(x);hs.append(x)
  z=self.head(self.lnf(x));loss=F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None;return (z,loss,hs) if return_hidden else (z,loss)
@torch.no_grad()
def evaluate(m,iters=15):
 m.eval();g=torch.Generator().manual_seed(777);ls=[];c=t=0
 for _ in range(iters):
  x,y=batch(val,5,g);z,l=m(x,y);ls.append(l.item());c+=int((z.argmax(-1)==y).sum());t+=y.numel()
 m.train();l=sum(ls)/len(ls);return l,math.exp(l),c/t
teacher=Teacher();teacher.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']);teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
s=Student();s.tok.weight.data.copy_(teacher.tok.weight);s.pos.weight.data.copy_(teacher.pos.weight);s.lnf.load_state_dict(teacher.lnf.state_dict())
for sb,tb in zip(s.blocks,teacher.blocks):sb.ln1.load_state_dict(tb.ln1.state_dict());sb.ln2.load_state_dict(tb.ln2.state_dict())
if args.resume:s.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict'])
opt=torch.optim.AdamW(s.parameters(),lr=args.lr,weight_decay=.01);Tt=2.;st=time.perf_counter()
for step in range(1,args.steps+1):
 x,y=batch(train)
 with torch.no_grad():tz,_,th=teacher(x,return_hidden=True)
 sz,ce,sh=s(x,y,return_hidden=True);kd=F.kl_div(F.log_softmax(sz/Tt,-1),F.softmax(tz/Tt,-1),reduction='batchmean')*(Tt*Tt)/BLOCK;hm=sum(F.mse_loss(F.layer_norm(a,(D,)),F.layer_norm(b,(D,))) for a,b in zip(sh,th))/len(sh);loss=.35*ce+.4*kd+.25*hm
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1.0);opt.step()
l,p,a=evaluate(s,15);dense_lin=6291456;but_lin=24*(2*D*STAGES+D);res={'ppl':p,'teacher_ppl':evaluate(teacher,15)[1],'acc':a,'structured_linear_params':but_lin,'linear_compression_x':dense_lin/but_lin,'stages':STAGES,'seconds':time.perf_counter()-st}
torch.save({'state_dict':s.state_dict(),'result':res},'/mnt/data/qcno_v4_butterfly.pt')
