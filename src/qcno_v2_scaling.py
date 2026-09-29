import os,time,math,gc
import numpy as np, pandas as pd, torch
from scipy.linalg import circulant

torch.set_num_threads(min(5,os.cpu_count() or 4)); torch.manual_seed(123)
T=32
cases=[(256,r) for r in (1,4,8,16)]+[(512,r) for r in (1,4,8,16)]+[(1024,r) for r in (1,4,8,16)]+[(2048,r) for r in (1,4,8,16)]+[(4096,r) for r in (1,4,8)]
rows=[]
def med(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))*1000
def implicit(x,a,b):
    xf=torch.fft.rfft(x,dim=-1); bf=torch.fft.rfft(b,dim=-1)
    conv=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=x.shape[-1],dim=-1)
    return (conv*a[None,:,:]).sum(dim=1)
for N,R in cases:
    a=torch.randn(R,N,dtype=torch.float32)/math.sqrt(R*N); b=torch.randn(R,N,dtype=torch.float32)/math.sqrt(N)
    A=np.zeros((N,N),dtype=np.float32); an=a.numpy(); bn=b.numpy()
    for r in range(R): A+=circulant(bn[r]).T.astype(np.float32,copy=False)*an[r][None,:]
    At=torch.from_numpy(A); x=torch.randn(T,N); yd=x@At; yi=implicit(x,a,b)
    rel=float(torch.linalg.norm(yd-yi)/torch.linalg.norm(yd)); reps=9 if N<=1024 else (6 if N<=2048 else 4)
    td=med(lambda:x@At,reps); ti=med(lambda:implicit(x,a,b),max(4,reps-2))
    dense_work=T*N*N; qcno_work=T*N*((R+1)*math.log2(N)+2*R)
    rows.append({'N':N,'R':R,'tokens_T':T,'dense_ms':td,'implicit_ms':ti,'speedup_dense_over_implicit':td/ti,'relative_error':rel,'weight_compression_x':N*N/(2*R*N),'theoretical_work_ratio_x':dense_work/qcno_work})
    del A,At,x,a,b,yd,yi; gc.collect()
pd.DataFrame(rows).to_csv('/mnt/data/qcno_v2_scaling_results.csv',index=False)
