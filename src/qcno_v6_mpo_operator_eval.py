import os,math,json,time
from pathlib import Path
import torch,torch.nn as nn,pandas as pd
torch.manual_seed(20260929);torch.set_num_threads(min(8,os.cpu_count() or 8))
D=512;BITS=9;FF=2048
class A(nn.Module):
 def __init__(self):super().__init__();self.q=nn.Linear(D,D,bias=False);self.k=nn.Linear(D,D,bias=False);self.v=nn.Linear(D,D,bias=False);self.o=nn.Linear(D,D,bias=False)
class B(nn.Module):
 def __init__(self):super().__init__();self.ln1=nn.LayerNorm(D);self.attn=A();self.ln2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF,bias=False);self.fc2=nn.Linear(FF,D,bias=False)
class T(nn.Module):
 def __init__(self):super().__init__();self.tok=nn.Embedding(256,D);self.pos=nn.Embedding(48,D);self.blocks=nn.ModuleList([B(),B()]);self.lnf=nn.LayerNorm(D);self.head=nn.Linear(D,256,bias=False);self.head.weight=self.tok.weight
t=T();t.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']);t.eval()
def ops():
 out=[]
 for li,b in enumerate(t.blocks):
  for n in ['q','k','v','o']:out.append((f'L{li}.{n}',getattr(b.attn,n).weight.detach().T.float()))
  X=b.fc1.weight.detach().T.float()
  for p in range(4):out.append((f'L{li}.fc1.{p}',X[:,p*D:(p+1)*D].contiguous()))
  X=b.fc2.weight.detach().T.float()
  for p in range(4):out.append((f'L{li}.fc2.{p}',X[p*D:(p+1)*D,:].contiguous()))
 return out
def matrix_to_phys(A):
 x=A.reshape(*([2]*BITS),*([2]*BITS));perm=[]
 for k in range(BITS):perm += [k,BITS+k]
 return x.permute(*perm).contiguous().reshape(*([4]*BITS))
def phys_to_matrix(x):
 y=x.reshape(*sum(([2,2] for _ in range(BITS)),[]));row_axes=list(range(0,2*BITS,2));col_axes=list(range(1,2*BITS,2));return y.permute(*(row_axes+col_axes)).contiguous().reshape(D,D)
@torch.no_grad()
def ttsvd(A,chi):
 x=matrix_to_phys(A);cores=[];r=1;rest=x
 for k in range(BITS-1):
  mat=rest.reshape(r*4,-1);U,S,Vh=torch.linalg.svd(mat,full_matrices=False);q=min(chi,S.numel());U=U[:,:q];S=S[:q];Vh=Vh[:q,:];cores.append(U.reshape(r,4,q));rest=(S[:,None]*Vh);r=q
 cores.append(rest.reshape(r,4,1));return cores
@torch.no_grad()
def reconstruct(cores):
 z=cores[0]
 for c in cores[1:]:z=torch.tensordot(z,c,dims=([-1],[0]))
 z=z.squeeze(0).squeeze(-1);return phys_to_matrix(z)
def params(cores):return sum(c.numel() for c in cores)
rows=[];st=time.perf_counter()
for oi,(name,W) in enumerate(ops()):
 for chi in [4,8,16,32]:
  c=ttsvd(W,chi);Wh=reconstruct(c);e=float((W-Wh).norm()/W.norm());p=params(c);rows.append({'op':name,'chi':chi,'params':p,'compression_x':D*D/p,'relative_error':e});print(name,chi,p,round(e,5),flush=True)
df=pd.DataFrame(rows);df.to_csv('/mnt/data/qcno_v6_mpo_operator_results.csv',index=False)
sumdf=df.groupby('chi').agg(mean_relative_error=('relative_error','mean'),median_relative_error=('relative_error','median'),mean_params=('params','mean'),mean_compression_x=('compression_x','mean')).reset_index();sumdf.to_csv('/mnt/data/qcno_v6_mpo_operator_summary.csv',index=False);print(sumdf.to_string(index=False));print('seconds',time.perf_counter()-st)
