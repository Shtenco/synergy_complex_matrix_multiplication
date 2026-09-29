import time, math
import torch, pandas as pd
torch.manual_seed(20260929);torch.set_num_threads(8)
def factors(n):
 best=None
 for a in range(1,int(math.sqrt(n))+1):
  if n%a==0:
   b=n//a;score=abs(math.log2(b/a));best=(score,a,b) if best is None or score<best[0] else best
 return best[1],best[2]
def med(fn,reps=9,warm=3):
 for _ in range(warm):fn()
 ts=[]
 for _ in range(reps):
  t=time.perf_counter();fn();ts.append(time.perf_counter()-t)
 return sorted(ts)[len(ts)//2]
rows=[]
for D in [512,1024,2048,4096]:
 M,N=factors(D);RK=7;T=32;A=torch.randn(M,N,N)/math.sqrt(N);B=torch.randn(N,M,M)/math.sqrt(M);gain=torch.randn(D)*.2+.8;U=torch.randn(D,RK)/math.sqrt(D);V=torch.randn(RK,D)/math.sqrt(RK);x=torch.randn(T,D)
 def imp(z):
  shp=z.shape;xx=z.reshape(-1,D);bs=xx.shape[0]
  y=xx.reshape(bs,M,N).permute(1,0,2).contiguous()
  y=torch.bmm(y,A).permute(1,0,2).contiguous()
  y=y.transpose(1,2).contiguous().permute(1,0,2).contiguous()
  y=torch.bmm(y,B).permute(1,0,2).contiguous()
  y=y.transpose(1,2).contiguous().reshape(bs,D)
  return (y*gain+(xx@U)@V).reshape(*shp[:-1],D)
 mats=[];chunk=256
 for s in range(0,D,chunk):
  c=min(chunk,D-s);e=torch.zeros(c,D);e[torch.arange(c),torch.arange(s,s+c)]=1.;mats.append(imp(e))
 W=torch.cat(mats,0);yd=x@W;yi=imp(x);err=float((yd-yi).norm()/(yd.norm()+1e-12));td=med(lambda:x@W,7 if D<4096 else 5);ti=med(lambda:imp(x),7 if D<4096 else 5)
 per=M*N*N+N*M*M+D+2*D*RK
 row={'N':D,'M':M,'Q':N,'dense_ms':td*1000,'implicit_bmm_ms':ti*1000,'speedup_x':td/ti,'relative_error':err,'compression_x':D*D/per};rows.append(row);print(row,flush=True)
pd.DataFrame(rows).to_csv('/mnt/data/qcno_v53_speed_bmm_results.csv',index=False)
