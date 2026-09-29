import os,math,time,glob,json,argparse
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=0);ap.add_argument('--lr',type=float,default=7e-5);ap.add_argument('--batch',type=int,default=5);ap.add_argument('--budget',type=int,default=168);ap.add_argument('--resume',default='');ap.add_argument('--objective',choices=['distill','ce'],default='distill');ap.add_argument('--seed-fixed',default='');args=ap.parse_args()
torch.manual_seed(20260929);torch.set_num_threads(min(8,os.cpu_count() or 8))
D=512;M=16;N=32;FF=2048;HEADS=8;LAYERS=2;BLOCK=48;VOCAB=256
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
def op_names():
 out=[]
 for li in range(2):
  for n in ['q','k','v','o']:out.append(f'blocks.{li}.attn.{n}')
  for p in range(4):out.append(f'blocks.{li}.fc1.parts.{p}')
  for p in range(4):out.append(f'blocks.{li}.fc2.parts.{p}')
 return out
names=op_names()
def teacher_A(t,name):
 parts=name.split('.');li=int(parts[1]);kind=parts[2];b=t.blocks[li]
 if kind=='attn':return getattr(b.attn,parts[3]).weight.detach().T.float().contiguous()
 p=int(parts[4])
 if kind=='fc1':return b.fc1.weight.detach().T[:,p*D:(p+1)*D].float().contiguous()
 return b.fc2.weight.detach().T[p*D:(p+1)*D,:].float().contiguous()
def monarch_apply(xx,A,B,gain):
 bs=xx.shape[0];y=xx.reshape(bs,M,N).permute(1,0,2).contiguous();y=torch.bmm(y,A).permute(1,0,2).contiguous();y=y.transpose(1,2).contiguous().permute(1,0,2).contiguous();y=torch.bmm(y,B).permute(1,0,2).contiguous();y=y.transpose(1,2).contiguous().reshape(bs,D);return y*gain
@torch.no_grad()
def monarch_dense(sd,name):
 A=sd[name+'.A'];B=sd[name+'.B'];g=sd[name+'.gain'];return monarch_apply(torch.eye(D),A,B,g)
t=Teacher();t.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']);t.eval();[p.requires_grad_(False) for p in t.parameters()]
base_sd=torch.load('/mnt/data/qcno_v52_monarch.pt',map_location='cpu',weights_only=False)['state_dict']
alloc_file=Path(f'/mnt/data/qcno_v54_alloc_budget{args.budget}.json');svd_file=Path(f'/mnt/data/qcno_v54_svd_budget{args.budget}.pt')
if not alloc_file.exists() or not svd_file.exists():
 svds={};cands=[]
 for name in names:
  R=teacher_A(t,name)-monarch_dense(base_sd,name);U,S,Vh=torch.linalg.svd(R,full_matrices=False);svds[name]=(U,S,Vh);cands += [(float(S[k]**2),name,k) for k in range(D)]
 alloc={n:1 for n in names};chosen={(n,0) for n in names};remaining=args.budget-len(names);pool=sorted((c for c in cands if (c[1],c[2]) not in chosen),reverse=True,key=lambda z:z[0])[:remaining]
 for _,n,k in pool:alloc[n]+=1
 torch.save(svds,svd_file);alloc_file.write_text(json.dumps({'budget':args.budget,'alloc':alloc},indent=2))
else:
 alloc=json.loads(alloc_file.read_text())['alloc'];svds=torch.load(svd_file,map_location='cpu',weights_only=False)
class HybridSquare(nn.Module):
 def __init__(self,r):
  super().__init__();self.r=r;self.A=nn.Parameter(torch.randn(M,N,N)/math.sqrt(N));self.B=nn.Parameter(torch.randn(N,M,M)/math.sqrt(M));self.gain=nn.Parameter(torch.full((D,),.45));self.U=nn.Parameter(torch.zeros(D,r));self.V=nn.Parameter(torch.zeros(r,D))
 def forward(self,x):
  shp=x.shape;xx=x.reshape(-1,D);y=monarch_apply(xx,self.A,self.B,self.gain);y=y+(xx@self.U)@self.V;return y.reshape(*shp[:-1],D)
class Expand(nn.Module):
 def __init__(self,li):super().__init__();self.parts=nn.ModuleList([HybridSquare(alloc[f'blocks.{li}.fc1.parts.{p}']) for p in range(4)])
 def forward(self,x):return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
 def __init__(self,li):super().__init__();self.parts=nn.ModuleList([HybridSquare(alloc[f'blocks.{li}.fc2.parts.{p}']) for p in range(4)])
 def forward(self,x):
  xs=x.split(D,-1);y=self.parts[0](xs[0])
  for i in range(1,4):y=y+self.parts[i](xs[i])
  return y
class Attn(nn.Module):
 def __init__(self,li):super().__init__();self.q=HybridSquare(alloc[f'blocks.{li}.attn.q']);self.k=HybridSquare(alloc[f'blocks.{li}.attn.k']);self.v=HybridSquare(alloc[f'blocks.{li}.attn.v']);self.o=HybridSquare(alloc[f'blocks.{li}.attn.o'])
 def forward(self,x):
  B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
 def __init__(self,li):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=Attn(li);self.ln2=nn.LayerNorm(D);self.fc1=Expand(li);self.fc2=Contract(li)
 def forward(self,x):x=x+self.attn(self.ln1(x));return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Student(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.blocks=nn.ModuleList([Block(i) for i in range(2)]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.tok.weight
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
s=Student()
if args.resume:s.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict'])
else:
 s.load_state_dict(base_sd,strict=False)
 with torch.no_grad():
  for name in names:
   mod=s
   for z in name.split('.'):mod=mod[int(z)] if z.isdigit() else getattr(mod,z)
   U,S,Vh=svds[name];r=mod.r;mod.U.copy_(U[:,:r]*S[:r]);mod.V.copy_(Vh[:r,:])
opt=torch.optim.AdamW(s.parameters(),lr=args.lr,weight_decay=.01);Tt=2.;st=time.perf_counter()
for step in range(1,args.steps+1):
 x,y=batch(train)
 if args.objective=='distill':
  with torch.no_grad():tz,_,th=t(x,return_hidden=True)
  sz,ce,sh=s(x,y,return_hidden=True);kd=F.kl_div(F.log_softmax(sz/Tt,-1),F.softmax(tz/Tt,-1),reduction='batchmean')*(Tt*Tt)/BLOCK;hm=sum(F.mse_loss(F.layer_norm(a,(D,)),F.layer_norm(b,(D,))) for a,b in zip(sh,th))/len(sh);loss=.35*ce+.4*kd+.25*hm
 else:sz,ce=s(x,y);loss=ce
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1.0);opt.step()
l,p,a=evalm(s,15);tp=evalm(t,15)[1];dense=6291456;baseper=M*N*N+N*M*M+D;structured=24*baseper+2*D*sum(alloc.values());res={'budget_rank_sum':sum(alloc.values()),'min_rank':min(alloc.values()),'max_rank':max(alloc.values()),'byte_ppl':p,'teacher_ppl':tp,'ppl_ratio':p/tp,'acc':a,'structured_linear_params':structured,'dense_linear_params':dense,'linear_compression_x':dense/structured,'seconds':time.perf_counter()-st}
torch.save({'state_dict':s.state_dict(),'alloc':alloc,'result':res},'/mnt/data/qcno_v54_adaptive_residual.pt')
