import os, math, time, glob, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(5, os.cpu_count() or 4))
SEED=20260929
torch.manual_seed(SEED)
VOCAB=256; BLOCK=64; D=128; HEADS=4; LAYERS=2; FF=512

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__(); self.q=nn.Linear(D,D,bias=False); self.k=nn.Linear(D,D,bias=False); self.v=nn.Linear(D,D,bias=False); self.o=nn.Linear(D,D,bias=False)
    def forward(self,x):
        B,T,C=x.shape
        q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=CausalSelfAttention(); self.ln2=nn.LayerNorm(D); self.fc1=nn.Linear(D,FF,bias=False); self.fc2=nn.Linear(FF,D,bias=False)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); x=x+self.fc2(F.gelu(self.fc1(self.ln2(x)))); return x
class TinyLM(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(VOCAB,D); self.pos=nn.Embedding(BLOCK,D); self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)]); self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None):
        B,T=idx.shape; x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]
        for b in self.blocks: x=b(x)
        logits=self.head(self.lnf(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1))
        return logits,loss

paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'))
raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024]
data=torch.tensor(list(raw),dtype=torch.long); split=int(.95*len(data)); val=data[split:]
ck=torch.load('/mnt/data/qcno_v2_baseline.pt',map_location='cpu',weights_only=False)
base=TinyLM(); base.load_state_dict(ck['state_dict']); base.eval()

def cyclic_unwrap(A):
    n=A.shape[0]; j=torch.arange(n)[:,None]; s=torch.arange(n)[None,:]; i=(j-s)%n
    return A[i,j.expand(n,n)]
def cyclic_wrap(Z):
    n=Z.shape[0]; A=torch.empty_like(Z); j=torch.arange(n)[:,None]; s=torch.arange(n)[None,:]; i=(j-s)%n
    A[i,j.expand(n,n)]=Z; return A
def factor_square(A):
    Z=cyclic_unwrap(A.double()); U,S,Vh=torch.linalg.svd(Z,full_matrices=False)
    return (U*S[None,:]).float(),Vh.float(),S.float()
def approx_square(factors,R):
    Aout,Bker,S=factors; return cyclic_wrap((Aout[:,:R]@Bker[:R,:]).float())
def split_linear_A(weight):
    A=weight.detach().T.contiguous(); ni,no=A.shape; blocks=[]
    if ni==D and no==D: blocks=[('square',0,A)]
    elif ni==D and no==FF:
        for k in range(FF//D): blocks.append(('out',k,A[:,k*D:(k+1)*D]))
    elif ni==FF and no==D:
        for k in range(FF//D): blocks.append(('in',k,A[k*D:(k+1)*D,:]))
    else: raise ValueError((ni,no))
    return blocks
def join_linear_A(kind_blocks,ap):
    kind=kind_blocks[0][0]
    if kind=='square': return ap[0]
    if kind=='out': return torch.cat(ap,dim=1)
    if kind=='in': return torch.cat(ap,dim=0)
def get_module(model,li,name):
    b=model.blocks[li]
    return {'q':b.attn.q,'k':b.attn.k,'v':b.attn.v,'o':b.attn.o,'fc1':b.fc1,'fc2':b.fc2}[name]

targets=[]
for li,b in enumerate(base.blocks):
    for name,module in [('q',b.attn.q),('k',b.attn.k),('v',b.attn.v),('o',b.attn.o),('fc1',b.fc1),('fc2',b.fc2)]:
        parts=split_linear_A(module.weight)
        targets.append({'layer':li,'name':name,'parts':parts,'factors':[factor_square(p[2]) for p in parts],'weight':module.weight.detach().clone()})

@torch.no_grad()
def evaluate(model,iters=40,batch_size=32):
    g=torch.Generator().manual_seed(777); losses=[]; correct=0; total=0
    for _ in range(iters):
        ix=torch.randint(0,len(val)-BLOCK-1,(batch_size,),generator=g)
        x=torch.stack([val[i:i+BLOCK] for i in ix]); y=torch.stack([val[i+1:i+BLOCK+1] for i in ix])
        logits,loss=model(x,y); losses.append(loss.item()); correct+=int((logits.argmax(-1)==y).sum()); total+=y.numel()
    l=sum(losses)/len(losses); return l,math.exp(l),correct/total

base_loss,base_ppl,base_acc=evaluate(base,60)
ranks=[2,4,8,16,24,32,48,64,96,128]; rows=[]; block_spectrum=[]
for t in targets:
    for pi,fac in enumerate(t['factors']):
        S=fac[2].double(); energy=S*S; cum=torch.cumsum(energy,0)/energy.sum()
        block_spectrum.append({'layer':t['layer'],'name':t['name'],'part':pi,'r50':int(torch.searchsorted(cum,torch.tensor(.50,dtype=cum.dtype))+1),'r90':int(torch.searchsorted(cum,torch.tensor(.90,dtype=cum.dtype))+1),'r99':int(torch.searchsorted(cum,torch.tensor(.99,dtype=cum.dtype))+1)})
for R in ranks:
    m=copy.deepcopy(base); sq_num=0.; sq_den=0.
    for t in targets:
        ap=[]
        for part,fac in zip(t['parts'],t['factors']):
            A0=part[2]; Ar=approx_square(fac,R); ap.append(Ar)
            sq_num+=float(((Ar-A0)**2).sum()); sq_den+=float((A0**2).sum())
        get_module(m,t['layer'],t['name']).weight.data.copy_(join_linear_A(t['parts'],ap).T)
    m.eval(); loss,ppl,acc=evaluate(m,35)
    dense_target_params=sum(t['weight'].numel() for t in targets); n_parts=sum(len(t['parts']) for t in targets); factor_params=n_parts*2*D*R
    rows.append({'rank_R':R,'relative_weight_Fro_error':math.sqrt(sq_num/sq_den),'val_loss':loss,'byte_ppl':ppl,'next_byte_accuracy':acc,'ppl_ratio_vs_baseline':ppl/base_ppl,'dense_target_params':dense_target_params,'qcno_factor_params':factor_params,'compression_x_dense_over_qcno':dense_target_params/factor_params})
pd.DataFrame(rows).to_csv('/mnt/data/qcno_v2_distillation_results.csv',index=False)
pd.DataFrame(block_spectrum).to_csv('/mnt/data/qcno_v2_spectrum.csv',index=False)
