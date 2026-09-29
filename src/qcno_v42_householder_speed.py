import torch,time,math,os,pandas as pd,numpy as np
torch.set_num_threads(min(5,os.cpu_count() or 5));torch.manual_seed(20260929)
def qdense(a,b):
 n=a.shape[0];Z=a@b;i=torch.arange(n)[:,None];j=torch.arange(n)[None,:];s=(j-i)%n;return Z[j.expand(n,n),s]
def apply_h(y,hv,hlogd,ha):
 for k in range(hv.shape[0]):
  y=y*torch.exp(hlogd[k]);v=hv[k];proj=(y*v).sum(-1,keepdim=True)/(v.square().sum()+1e-6);y=y-ha[k]*proj*v
 return y
def materialize_h(A,hv,hlogd,ha):
 for k in range(hv.shape[0]):
  A=A*torch.exp(hlogd[k])[None,:];v=hv[k];Av=A@v[:,None];A=A-ha[k]*(Av@v[None,:])/(v.square().sum()+1e-6)
 return A
def imp(x,a,bf,hv,hlogd,ha):
 n=x.shape[-1];xf=torch.fft.rfft(x,dim=-1);c=torch.fft.irfft(xf[:,None,:]*bf[None,:,:],n=n,dim=-1);y=(c*a.T[None,:,:]).sum(1);return apply_h(y,hv,hlogd,ha)
def bench(fn,reps=7):
 for _ in range(3):fn()
 ts=[]
 for _ in range(reps):t=time.perf_counter();fn();ts.append((time.perf_counter()-t)*1000)
 return float(np.median(ts))
rows=[]
for n in [1024,2048,4096]:
 r=8;k=4;T=32;a=torch.randn(n,r)/math.sqrt(n*r);b=torch.randn(r,n)/math.sqrt(n);bf=torch.fft.rfft(b,dim=-1);hv=torch.randn(k,n)/math.sqrt(n);hlogd=.03*torch.randn(k,n);ha=.2*torch.randn(k);x=torch.randn(T,n);A=materialize_h(qdense(a,b),hv,hlogd,ha);yd=x@A;yi=imp(x,a,bf,hv,hlogd,ha);rel=float(torch.linalg.norm(yd-yi)/torch.linalg.norm(yd));td=bench(lambda:x@A,7);ti=bench(lambda:imp(x,a,bf,hv,hlogd,ha),7);rows.append({'N':n,'R':r,'K_householder':k,'dense_ms':td,'implicit_ms':ti,'speedup_x':td/ti,'relative_error':rel,'linear_compression_x':n*n/(2*n*r+k*(2*n+1))});print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('/mnt/data/qcno_v42_householder_speed_results.csv',index=False)
