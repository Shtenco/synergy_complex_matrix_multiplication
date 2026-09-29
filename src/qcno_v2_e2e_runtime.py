import os,time,math,json,copy
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(5,os.cpu_count() or 4)); torch.manual_seed(42)
VOCAB=256; BLOCK=64;D=128;HEADS=4;LAYERS=2;FF=512;R=8
i=torch.arange(D)[:,None];j=torch.arange(D)[None,:];J=j.expand(D,D);S=(j-i)%D

def dense_weight(a,b):
    Z=a@b; A=Z[J,S]; return A.T.contiguous()

class ImpSq(nn.Module):
    def __init__(self,a,b): super().__init__(); self.register_buffer('a',a.clone());self.register_buffer('b',b.clone())
    def forward(self,x):
        sh=x.shape; xf=torch.fft.rfft(x.reshape(-1,D),dim=-1); bf=torch.fft.rfft(self.b,dim=-1)
        conv=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=D,dim=-1)
        y=(conv*self.a.T[None,:,:]).sum(1); return y.reshape(*sh[:-1],D)
class DenseSq(nn.Module):
    def __init__(self,a,b): super().__init__(); self.register_buffer('w',dense_weight(a,b))
    def forward(self,x): return F.linear(x,self.w)
class Expand(nn.Module):
    def __init__(self,parts,cls): super().__init__();self.p=nn.ModuleList([cls(a,b) for a,b in parts])
    def forward(self,x):return torch.cat([m(x) for m in self.p],-1)
class Contract(nn.Module):
    def __init__(self,parts,cls): super().__init__();self.p=nn.ModuleList([cls(a,b) for a,b in parts])
    def forward(self,x):
        xs=x.split(D,-1); y=self.p[0](xs[0])
        for z,m in zip(xs[1:],self.p[1:]):y=y+m(z)
        return y
class Attn(nn.Module):
    def __init__(self,pars,cls): super().__init__();self.q=cls(*pars['q']);self.k=cls(*pars['k']);self.v=cls(*pars['v']);self.o=cls(*pars['o'])
    def forward(self,x):
        B,T,C=x.shape;q=self.q(x).view(B,T,HEADS,C//HEADS).transpose(1,2);k=self.k(x).view(B,T,HEADS,C//HEADS).transpose(1,2);v=self.v(x).view(B,T,HEADS,C//HEADS).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True);return self.o(y.transpose(1,2).contiguous().view(B,T,C))
class B(nn.Module):
    def __init__(self,pars,cls,ln1w,ln1b,ln2w,ln2b):
        super().__init__();self.ln1=nn.LayerNorm(D);self.ln2=nn.LayerNorm(D);self.ln1.weight.data.copy_(ln1w);self.ln1.bias.data.copy_(ln1b);self.ln2.weight.data.copy_(ln2w);self.ln2.bias.data.copy_(ln2b);self.attn=Attn(pars,cls);self.fc1=Expand(pars['fc1'],cls);self.fc2=Contract(pars['fc2'],cls)
    def forward(self,x):x=x+self.attn(self.ln1(x));x=x+self.fc2(F.gelu(self.fc1(self.ln2(x))));return x
class M(nn.Module):
    def __init__(self,state,cls):
        super().__init__();self.tok=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(BLOCK,D);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False)
        self.tok.weight.data.copy_(state['tok.weight']);self.pos.weight.data.copy_(state['pos.weight']);self.lnf.weight.data.copy_(state['lnf.weight']);self.lnf.bias.data.copy_(state['lnf.bias']);self.head.weight=self.tok.weight
        blocks=[]
        for l in range(LAYERS):
            pre=f'blocks.{l}.'; pars={}
            for nm in ['q','k','v','o']:
                pars[nm]=(state[pre+f'attn.{nm}.a'],state[pre+f'attn.{nm}.b'])
            for nm in ['fc1','fc2']:
                pars[nm]=[(state[pre+f'{nm}.parts.{k}.a'],state[pre+f'{nm}.parts.{k}.b']) for k in range(4)]
            blocks.append(B(pars,cls,state[pre+'ln1.weight'],state[pre+'ln1.bias'],state[pre+'ln2.weight'],state[pre+'ln2.bias']))
        self.blocks=nn.ModuleList(blocks)
    def forward(self,idx):
        B,T=idx.shape;x=self.tok(idx)+self.pos(torch.arange(T))[None,:,:]
        for b in self.blocks:x=b(x)
        return self.head(self.lnf(x))

state=torch.load('/mnt/data/qcno_v2_native_R8.pt',map_location='cpu',weights_only=False)['state_dict']
md=M(state,DenseSq).eval(); mi=M(state,ImpSq).eval(); x=torch.randint(0,256,(16,64))
with torch.no_grad(): yd=md(x); yi=mi(x)
rel=float(torch.linalg.norm(yd-yi)/torch.linalg.norm(yd)); maxerr=float((yd-yi).abs().max())
def med(fn,reps=10):
    with torch.no_grad():
        for _ in range(2):fn()
        ts=[]
        for _ in range(reps):t=time.perf_counter();fn();ts.append(time.perf_counter()-t)
    return float(np.median(ts))*1000
td=med(lambda:md(x),12);ti=med(lambda:mi(x),8)
res={'R':R,'batch':16,'seq':64,'tokens':1024,'dense_materialized_ms':td,'implicit_fft_ms':ti,'speedup_dense_over_implicit':td/ti,'logits_relative_error':rel,'logits_max_abs_error':maxerr}
print(json.dumps(res,indent=2));open('/mnt/data/qcno_v2_e2e_runtime.json','w').write(json.dumps(res,indent=2))
