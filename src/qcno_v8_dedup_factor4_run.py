import math, glob, os, time, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

torch.manual_seed(20260929)
torch.set_num_threads(min(6, os.cpu_count() or 6))
D=512; FF=2048; HEADS=8; LAYERS=2; BLOCK=48; VOCAB=256
CK='/mnt/data/qcno_v31_adaptive_avg32_best_final.pt'
ck=torch.load(CK,map_location='cpu',weights_only=False); base_alloc=ck['alloc']
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'))
raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]

def batch(src,bs=5,g=None):
    ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g)
    x=torch.stack([src[k:k+BLOCK] for k in ix]); y=torch.stack([src[k+1:k+BLOCK+1] for k in ix]); return x,y
class Sq(nn.Module):
    def __init__(self,r):
        super().__init__(); self.r=int(r); std=(0.02**2/max(self.r,1))**.25
        self.a=nn.Parameter(torch.randn(D,self.r)*std); self.b=nn.Parameter(torch.randn(self.r,D)*std)
    def forward(self,x):
        shp=x.shape; xx=x.reshape(-1,D); xf=torch.fft.rfft(xx,dim=-1); bf=torch.fft.rfft(self.b,dim=-1)
        c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1)
        y=(c*self.a.T[None,:,:]).sum(1); return y.reshape(*shp[:-1],D)
class Attn(nn.Module):
    def __init__(self,alloc,li):
        super().__init__(); self.q=Sq(alloc[f'L{li}.q']); self.k=Sq(alloc[f'L{li}.k']); self.v=Sq(alloc[f'L{li}.v']); self.o=Sq(alloc[f'L{li}.o'])
    def forward(self,x):
        B,T,C=x.shape; q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Expand(nn.Module):
    def __init__(self,alloc,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc1.{p}']) for p in range(4)])
    def forward(self,x): return torch.cat([p(x) for p in self.parts],-1)
class Contract(nn.Module):
    def __init__(self,alloc,li): super().__init__(); self.parts=nn.ModuleList([Sq(alloc[f'L{li}.fc2.{p}']) for p in range(4)])
    def forward(self,x):
        xs=x.split(D,-1); y=self.parts[0](xs[0])
        for i in range(1,4): y=y+self.parts[i](xs[i])
        return y
class Block(nn.Module):
    def __init__(self,alloc,li): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=Attn(alloc,li); self.ln2=nn.LayerNorm(D); self.fc1=Expand(alloc,li); self.fc2=Contract(alloc,li)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Model(nn.Module):
    def __init__(self,alloc):
        super().__init__(); self.alloc=alloc; self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([Block(alloc,i) for i in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None):
        B,T=idx.shape; x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]
        for b in self.blocks: x=b(x)
        z=self.head(self.lnf(x)); l=F.cross_entropy(z.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None; return z,l
@torch.no_grad()
def evalm(m,iters=30):
    m.eval(); g=torch.Generator().manual_seed(424242); ls=[]; cor=tot=0
    for _ in range(iters):
        x,y=batch(val,5,g); z,l=m(x,y); ls.append(l.item()); cor+=int((z.argmax(-1)==y).sum()); tot+=y.numel()
    m.train(); l=sum(ls)/len(ls); return l,math.exp(l),cor/tot
def list_sq(m):
    out=[]
    for li,b in enumerate(m.blocks):
        for n in ['q','k','v','o']: out.append((f'L{li}.{n}',getattr(b.attn,n)))
        for p,q in enumerate(b.fc1.parts): out.append((f'L{li}.fc1.{p}',q))
        for p,q in enumerate(b.fc2.parts): out.append((f'L{li}.fc2.{p}',q))
    return out
def copy_non_sq(dst,src):
    dst.tok.weight.data.copy_(src.tok.weight); dst.pos.weight.data.copy_(src.pos.weight); dst.lnf.load_state_dict(src.lnf.state_dict())
    for db,sb in zip(dst.blocks,src.blocks): db.ln1.load_state_dict(sb.ln1.state_dict()); db.ln2.load_state_dict(sb.ln2.state_dict())
def dedup_sq(src,dst):
    B=src.b.detach().cpu().numpy().astype(np.float32); A=src.a.detach().cpu().numpy().astype(np.float32)
    norms=np.linalg.norm(B,axis=1)+1e-12; U=B/norms[:,None]
    im=np.argmax(np.abs(U),axis=1); sg=np.sign(U[np.arange(len(U)),im]); sg[sg==0]=1
    U*=sg[:,None]; coeff=norms*sg; K=dst.r
    km=KMeans(n_clusters=K,n_init=8,max_iter=200,random_state=20260929)
    ids=km.fit_predict(U); C=km.cluster_centers_.astype(np.float32); C/=np.linalg.norm(C,axis=1,keepdims=True)+1e-12
    proj=np.sum(U*C[ids],axis=1)*coeff; Anew=np.zeros((D,K),np.float32)
    for r,k in enumerate(ids): Anew[:,k]+=A[:,r]*proj[r]
    dst.a.data.copy_(torch.from_numpy(Anew)); dst.b.data.copy_(torch.from_numpy(C))
    Bhat=C[ids]*(proj[:,None]); return float(np.linalg.norm(B-Bhat)/(np.linalg.norm(B)+1e-12))
def make_dedup(src,factor):
    alloc={k:max(1,int(math.ceil(v/factor))) for k,v in base_alloc.items()}
    dst=Model(alloc); copy_non_sq(dst,src); errs=[]
    for (n,sq),(n2,dq) in zip(list_sq(src),list_sq(dst)):
        assert n==n2; errs.append(dedup_sq(sq,dq))
    return dst,alloc,float(np.mean(errs))
base=Model(base_alloc); base.load_state_dict(ck['state_dict']); print('BASE',evalm(base,8),flush=True)
rows=[]
for factor in [4]:
    m,alloc,berr=make_dedup(base,factor); pre=evalm(m,8); print('PRE',factor,pre,'b_err',berr,'ranksum',sum(alloc.values()),flush=True)
    opt=torch.optim.AdamW(m.parameters(),lr=8e-5,weight_decay=.01); st=time.perf_counter()
    for step in range(1,61):
        x,y=batch(train,5); z,l=m(x,y); opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    post=evalm(m,10); print('POST',factor,post,flush=True)
    dense_linear=6291456; structured=2*D*sum(alloc.values())
    row={'dedup_factor':factor,'rank_sum':sum(alloc.values()),'linear_compression_vs_dense_x':dense_linear/structured,'mean_filter_bank_relerr':berr,'pre_ppl':pre[1],'post_ppl':post[1],'post_acc':post[2],'seconds':time.perf_counter()-st}; rows.append(row)
    torch.save({'state_dict':m.state_dict(),'alloc':alloc,'result':row},f'/mnt/data/qcno_v8_dedup_factor{factor}.pt')
Path('/mnt/data/qcno_v8_dedup_factor4_results.json').write_text(json.dumps(rows,indent=2)); print('RESULTS',json.dumps(rows),flush=True)
