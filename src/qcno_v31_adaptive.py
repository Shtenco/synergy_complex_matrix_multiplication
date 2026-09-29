import os, math, time, glob, json, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=45); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--avg',type=str,default='16'); ap.add_argument('--resume',type=str,default=''); ap.add_argument('--batch',type=int,default=5); ap.add_argument('--objective',choices=['distill','ce'],default='distill'); args=ap.parse_args()
torch.manual_seed(20260929); torch.set_num_threads(min(5,os.cpu_count() or 5))
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256
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
        B,T=idx.shape; x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]; hs=[]
        for b in self.blocks: x=b(x); hs.append(x)
        logits=self.head(self.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None; return (logits,loss,hs) if return_hidden else (logits,loss)
class Sq(nn.Module):
    def __init__(self,r): super().__init__(); self.r=r; std=(0.02**2/r)**.25; self.a=nn.Parameter(torch.randn(D,r)*std); self.b=nn.Parameter(torch.randn(r,D)*std)
    def forward(self,x):
        shp=x.shape; xx=x.reshape(-1,D); xf=torch.fft.rfft(xx,dim=-1); bf=torch.fft.rfft(self.b,dim=-1); c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1); y=(c*self.a.T[None,:,:]).sum(1); return y.reshape(*shp[:-1],D)
class AAttn(nn.Module):
    def __init__(self,li):
        super().__init__(); self.q=Sq(alloc[f'L{li}.q']); self.k=Sq(alloc[f'L{li}.k']); self.v=Sq(alloc[f'L{li}.v']); self.o=Sq(alloc[f'L{li}.o'])
    def forward(self,x):
        B,T,C=x.shape; q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2); y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Expand(nn.Module):
    def __init__(self,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc1.{p}']) for p in range(4)])
    def forward(self,x): return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
    def __init__(self,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc2.{p}']) for p in range(4)])
    def forward(self,x):
        xs=x.split(D,-1); y=self.parts[0](xs[0]);
        for i in range(1,4): y=y+self.parts[i](xs[i])
        return y
class ABlock(nn.Module):
    def __init__(self,li): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=AAttn(li); self.ln2=nn.LayerNorm(D); self.fc1=Expand(li); self.fc2=Contract(li)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Student(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([ABlock(i) for i in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None,return_hidden=False):
        B,T=idx.shape; x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]; hs=[]
        for b in self.blocks: x=b(x); hs.append(x)
        logits=self.head(self.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None; return (logits,loss,hs) if return_hidden else (logits,loss)

def unwrap(A):
    n=A.shape[0]; j=torch.arange(n)[:,None]; s=torch.arange(n)[None,:]; i=(j-s)%n; return A[i,j.expand(n,n)]
def init_sq(q,A):
    Z=unwrap(A.float()); r=q.r; U,S,V=torch.svd_lowrank(Z,q=min(D,max(r+8,2*r)),niter=3); q.a.data.copy_(U[:,:r]*S[:r]); q.b.data.copy_(V[:,:r].T)
def init_student(s,t):
    s.tok.weight.data.copy_(t.tok.weight); s.pos.weight.data.copy_(t.pos.weight); s.lnf.load_state_dict(t.lnf.state_dict())
    for li,(sb,tb) in enumerate(zip(s.blocks,t.blocks)):
        sb.ln1.load_state_dict(tb.ln1.state_dict()); sb.ln2.load_state_dict(tb.ln2.state_dict())
        for n in ['q','k','v','o']: init_sq(getattr(sb.attn,n),getattr(tb.attn,n).weight.detach().T)
        A=tb.fc1.weight.detach().T
        for p,q in enumerate(sb.fc1.parts): init_sq(q,A[:,p*D:(p+1)*D])
        A=tb.fc2.weight.detach().T
        for p,q in enumerate(sb.fc2.parts): init_sq(q,A[p*D:(p+1)*D,:])
@torch.no_grad()
def evaluate(m,iters=15):
    m.eval(); g=torch.Generator().manual_seed(777); ls=[]; cor=tot=0
    for _ in range(iters):
        x,y=batch(val,5,g); z,l=m(x,y); ls.append(l.item()); cor+=int((z.argmax(-1)==y).sum()); tot+=y.numel()
    m.train(); l=sum(ls)/len(ls); return l,math.exp(l),cor/tot

t=Teacher(); t.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']); t.eval(); [p.requires_grad_(False) for p in t.parameters()]
s=Student()
ckpt=Path(f'/mnt/data/qcno_v31_adaptive_avg{args.avg}.pt')
if args.resume:
    s.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict']); print('resumed',args.resume,flush=True)
else:
    st=time.perf_counter(); init_student(s,t); print('init_seconds',time.perf_counter()-st,flush=True)
opt=torch.optim.AdamW(s.parameters(),lr=args.lr,weight_decay=.01); Tt=2.0
print(json.dumps({'avg':args.avg,'rank_sum':sum(alloc.values()),'minR':min(alloc.values()),'maxR':max(alloc.values()),'params':sum(p.numel() for p in s.parameters()),'teacher_ppl':evaluate(t)[1]}),flush=True)
st=time.perf_counter()
for step in range(1,args.steps+1):
    x,y=batch(train)
    if args.objective=='distill':
        with torch.no_grad(): tz,_,th=t(x,return_hidden=True)
        sz,ce,sh=s(x,y,return_hidden=True); kd=F.kl_div(F.log_softmax(sz/Tt,-1),F.softmax(tz/Tt,-1),reduction='batchmean')*(Tt*Tt)/BLOCK; hm=sum(F.mse_loss(F.layer_norm(a,(D,)),F.layer_norm(b,(D,))) for a,b in zip(sh,th))/len(sh); loss=.35*ce+.4*kd+.25*hm
    else:
        sz,ce=s(x,y); loss=ce; kd=torch.tensor(0.0); hm=torch.tensor(0.0)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt.step()
    if step==1 or step%40==0:
        l,p,a=evaluate(s,8); print(f'step={step} ppl={p:.3f} acc={a:.4f} ce={ce.item():.3f} kd={kd.item():.3f} hid={hm.detach().item():.3f} elapsed={time.perf_counter()-st:.1f}',flush=True)
l,p,a=evaluate(s,15); tp=evaluate(t,15)[1]; structured=2*D*sum(alloc.values()); dense=LAYERS*(4*D*D+2*D*FF); res={'avg_rank':float(args.avg),'rank_sum':sum(alloc.values()),'steps_this_run':args.steps,'byte_ppl':p,'teacher_ppl':tp,'ppl_ratio':p/tp,'acc':a,'params':sum(p.numel() for p in s.parameters()),'structured_linear_params':structured,'dense_linear_params':dense,'linear_compression_x':dense/structured,'min_rank':min(alloc.values()),'max_rank':max(alloc.values()),'seconds':time.perf_counter()-st}
torch.save({'state_dict':s.state_dict(),'alloc':alloc,'result':res},ckpt); Path(str(ckpt).replace('.pt','_metrics.json')).write_text(json.dumps(res,indent=2)); print('FINAL',json.dumps(res),flush=True)
