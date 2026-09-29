import time,math,torch,pandas as pd
torch.manual_seed(20260929);torch.set_num_threads(8);D=4096;M=64;N=64;RK=7
A=torch.randn(M,N,N)/math.sqrt(N);B=torch.randn(N,M,M)/math.sqrt(M);gain=torch.randn(D)*.2+.8;U=torch.randn(D,RK)/math.sqrt(D);V=torch.randn(RK,D)/math.sqrt(RK)
def imp(z):
 shp=z.shape;xx=z.reshape(-1,D);bs=xx.shape[0];y=xx.reshape(bs,M,N).permute(1,0,2).contiguous();y=torch.bmm(y,A).permute(1,0,2).contiguous();y=y.transpose(1,2).contiguous().permute(1,0,2).contiguous();y=torch.bmm(y,B).permute(1,0,2).contiguous();y=y.transpose(1,2).contiguous().reshape(bs,D);return (y*gain+(xx@U)@V).reshape(*shp[:-1],D)
mats=[]
for s in range(0,D,256):
 c=min(256,D-s);e=torch.zeros(c,D);e[torch.arange(c),torch.arange(s,s+c)]=1.;mats.append(imp(e))
W=torch.cat(mats,0)
def med(fn,reps=9,warm=3):
 for _ in range(warm):fn()
 ts=[]
 for _ in range(reps):
  t=time.perf_counter();fn();ts.append(time.perf_counter()-t)
 return sorted(ts)[len(ts)//2]
rows=[]
for T in [1,4,8,16,32,64,128,256]:
 x=torch.randn(T,D);err=float(((x@W)-imp(x)).norm()/((x@W).norm()+1e-12));td=med(lambda:x@W,7);ti=med(lambda:imp(x),7);rows.append({'T_vectors':T,'dense_ms':td*1000,'implicit_ms':ti*1000,'speedup_x':td/ti,'relative_error':err});print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('/mnt/data/qcno_v53_tokenwidth_speed_results.csv',index=False)
