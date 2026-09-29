import os, math, time, glob, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=20260929; torch.manual_seed(SEED); torch.set_num_threads(min(5, os.cpu_count() or 4))
VOCAB=256; BLOCK=64; D=128; HEADS=4; LAYERS=2; FF=512; BATCH=48
R=int(os.environ.get('QCNO_R','16')); STEPS=int(os.environ.get('QCNO_STEPS','300')); LR=3e-4
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'))
raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); train=data[:split]; val=data[split:]
i=torch.arange(D)[:,None]; j=torch.arange(D)[None,:]; J=j.expand(D,D); S=(j-i)%D
class QCNOSquare(nn.Module):
    def __init__(self):
        super().__init__(); std=(0.02**2/R)**.25
        self.a=nn.Parameter(torch.randn(D,R)*std); self.b=nn.Parameter(torch.randn(R,D)*std)
    def weight(self):
        Z=self.a@self.b; A=Z[J,S]; return A.T.contiguous()
    def forward(self,x): return F.linear(x,self.weight())
class QCNOExpand(nn.Module):
    def __init__(self): super().__init__(); self.parts=nn.ModuleList([QCNOSquare() for _ in range(4)])
    def forward(self,x): return torch.cat([p(x) for p in self.parts],dim=-1)
class QCNOContract(nn.Module):
    def __init__(self): super().__init__(); self.parts=nn.ModuleList([QCNOSquare() for _ in range(4)])
    def forward(self,x):
        xs=x.split(D,dim=-1); y=self.parts[0](xs[0])
        for k in range(1,4): y=y+self.parts[k](xs[k])
        return y
class Attn(nn.Module):
    def __init__(self): super().__init__(); self.q=QCNOSquare(); self.k=QCNOSquare(); self.v=QCNOSquare(); self.o=QCNOSquare()
    def forward(self,x):
        B,T,C=x.shape; q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2); k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2); v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class BlockM(nn.Module):
    def __init__(self): super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=Attn(); self.ln2=nn.LayerNorm(D); self.fc1=QCNOExpand(); self.fc2=QCNOContract()
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([BlockM() for _ in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
        nn.init.normal_(self.tok.weight,0,.02); nn.init.normal_(self.pos.weight,0,.02)
        for m in self.modules():
            if isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        x=self.tok(idx)+self.pos(torch.arange(idx.shape[1]))[None,:,:]
        for b in self.blocks: x=b(x)
        logits=self.head(self.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1)) if targets is not None else None
        return logits,loss
model=Model(); resume=os.environ.get('QCNO_RESUME','')
if resume: model.load_state_dict(torch.load(resume,map_location='cpu',weights_only=False)['state_dict'])
opt=torch.optim.AdamW(model.parameters(),lr=float(os.environ.get('QCNO_LR',LR)),weight_decay=.01)
def batch(src,bs=BATCH,g=None):
    ix=torch.randint(0,len(src)-BLOCK-1,(bs,),generator=g); return torch.stack([src[k:k+BLOCK] for k in ix]),torch.stack([src[k+1:k+BLOCK+1] for k in ix])
@torch.no_grad()
def evaluate(iters=50):
    model.eval(); g=torch.Generator().manual_seed(777); losses=[]; correct=total=0
    for _ in range(iters):
        x,y=batch(val,32,g); logits,loss=model(x,y); losses.append(loss.item()); correct+=int((logits.argmax(-1)==y).sum()); total+=y.numel()
    model.train(); l=sum(losses)/len(losses); return l,math.exp(l),correct/total
params=sum(p.numel() for p in model.parameters()); target_params=24*2*D*R
t0=time.perf_counter()
for step in range(1,STEPS+1):
    x,y=batch(train); _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
l,p,a=evaluate(60)
res={'R':R,'steps':STEPS,'val_loss':l,'byte_ppl':p,'next_byte_accuracy':a,'params':params,'structured_linear_params':target_params,'dense_linear_equiv':393216,'linear_compression_x':393216/target_params,'total_compression_vs_dense_baseline_x':435456/params,'train_seconds':time.perf_counter()-t0}
Path(f'/mnt/data/qcno_v2_native_R{R}_metrics.json').write_text(json.dumps(res,indent=2)); torch.save({'state_dict':model.state_dict(),'result':res},f'/mnt/data/qcno_v2_native_R{R}.pt')
