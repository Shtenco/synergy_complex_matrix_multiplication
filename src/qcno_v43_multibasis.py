import os, math, time, glob, json, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=45); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--avg',type=str,default='16'); ap.add_argument('--resume',type=str,default=''); ap.add_argument('--batch',type=int,default=5); ap.add_argument('--objective',choices=['distill','ce'],default='distill'); ap.add_argument('--seed-qcno',default=''); ap.add_argument('--corr-r',type=int,default=4); args=ap.parse_args()
torch.manual_seed(20260929); torch.set_num_threads(min(5,os.cpu_count() or 5))
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256
BITS=int(math.log2(D))
def _bitrev(i): return int(format(i,f'0{BITS}b')[::-1],2)
PERM=torch.tensor([_bitrev(i) for i in range(D)],dtype=torch.long); INV=torch.empty_like(PERM); INV[PERM]=torch.arange(D)
alloc=json.loads(Path('/mnt/data/qcno_v31_rank_allocations.json').read_text())['allocations'][args.avg]
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt')); raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]; data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]
def batch(src,bs=None,g=None):
    bs=bs or args.batch; ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g); x=torch.stack([src[k:k+BLOCK] for k in ix]); y=torch.stack([src[k+1:k+BLOCK+1] for k in ix]); return x,y
class DAttn(nn.Module):
    def __init__(self): super().__init__(); self.q=nn.Linear(D,D,bias=False); self.k=nn.Linear(D,D,bias=False); self.v=nn.Linear(D,D,bias=False); self.o=nn.Linear(D,D,bias=False)
    def forward(self,x):
        B,T,C=x.shape; q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2); y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class DBlock(nn.Module):
    def __init__(self): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=DAttn(); self.ln2=nn.LayerNorm(D); self.fc1=nn.Linear(D,FF,bias=False); self.fc2=nn.Linear(FF,D,bias=False)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Teacher(nn.Module):
    def __init__(self): super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([DBlock() for _ in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None,return_hidden=False):
        x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:]; hs=[]
        for b in self.blocks: x=b(x); hs.append(x)
        logits=self.head(self.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None; return (logits,loss,hs) if return_hidden else (logits,loss)
class Sq(nn.Module):
    def __init__(self,r):
        super().__init__(); self.r=r; std=(0.02**2/r)**.25; self.a=nn.Parameter(torch.randn(D,r)*std); self.b=nn.Parameter(torch.randn(r,D)*std)
        cr=args.corr_r; cstd=(0.02**2/max(cr,1))**.25; self.ca=nn.Parameter(torch.randn(D,cr)*cstd); self.cb=nn.Parameter(torch.zeros(cr,D))
    def forward(self,x):
        shp=x.shape; xx=x.reshape(-1,D)
        xf=torch.fft.rfft(xx,dim=-1); bf=torch.fft.rfft(self.b,dim=-1); c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1); y=(c*self.a.T[None,:,:]).sum(1)
        xp=xx[:,PERM]; xpf=torch.fft.rfft(xp,dim=-1); cbf=torch.fft.rfft(self.cb,dim=-1); cc=torch.fft.irfft(xpf[:,None,:]*cbf[None,:,:],n=D,dim=-1); yp=(cc*self.ca.T[None,:,:]).sum(1); y=y+yp[:,INV]
        return y.reshape(*shp[:-1],D)
class AAttn(nn.Module):
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
class ABlock(nn.Module):
    def __init__(self,li): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=AAttn(li); self.ln2=nn.LayerNorm(D); self.fc1=Expand(li); self.fc2=Contract(li)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Student(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([ABlock(i) for i in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None,return_hidden=False):
        x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:]; hs=[]
        for b in self.blocks: x=b(x); hs.append(x)
        logits=self.head(self.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None; return (logits,loss,hs) if return_hidden else (logits,loss)
@torch.no_grad()
def evaluate(m,iters=15):
    m.eval(); g=torch.Generator().manual_seed(777); ls=[]; cor=tot=0
    for _ in range(iters):
        x,y=batch(val,5,g); z,l=m(x,y); ls.append(l.item()); cor+=int((z.argmax(-1)==y).sum()); tot+=y.numel()
    m.train(); l=sum(ls)/len(ls); return l,math.exp(l),cor/tot
t=Teacher(); t.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']); t.eval(); [p.requires_grad_(False) for p in t.parameters()]
s=Student(); ckpt=Path(f'/mnt/data/qcno_v43_multibasis_avg{args.avg}_C{args.corr_r}.pt')
if args.resume: s.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict'])
elif args.seed_qcno: s.load_state_dict(torch.load(args.seed_qcno,map_location='cpu',weights_only=False)['state_dict'],strict=False)
opt=torch.optim.AdamW(s.parameters(),lr=args.lr,weight_decay=.01); Tt=2.; st=time.perf_counter()
for step in range(1,args.steps+1):
    x,y=batch(train)
    if args.objective=='distill':
        with torch.no_grad(): tz,_,th=t(x,return_hidden=True)
        sz,ce,sh=s(x,y,return_hidden=True); kd=F.kl_div(F.log_softmax(sz/Tt,-1),F.softmax(tz/Tt,-1),reduction='batchmean')*(Tt*Tt)/BLOCK; hm=sum(F.mse_loss(F.layer_norm(a,(D,)),F.layer_norm(b,(D,))) for a,b in zip(sh,th))/len(sh); loss=.35*ce+.4*kd+.25*hm
    else: sz,ce=s(x,y); loss=ce
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt.step()
l,p,a=evaluate(s,15); tp=evaluate(t,15)[1]; structured=2*D*sum(alloc.values())+24*2*D*args.corr_r; dense=LAYERS*(4*D*D+2*D*FF); res={'avg_rank':float(args.avg),'rank_sum':sum(alloc.values()),'byte_ppl':p,'teacher_ppl':tp,'ppl_ratio':p/tp,'acc':a,'structured_linear_params':structured,'dense_linear_params':dense,'linear_compression_x':dense/structured,'seconds':time.perf_counter()-st}
torch.save({'state_dict':s.state_dict(),'alloc':alloc,'result':res},ckpt)
