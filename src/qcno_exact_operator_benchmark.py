import time, gc, math
import numpy as np
import pandas as pd
from scipy import fft as spfft

def median_time(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    vals=[]
    for _ in range(repeats):
        t0=time.perf_counter(); fn(); vals.append(time.perf_counter()-t0)
    return float(np.median(vals))

def apply_implicit(x,dk,dx):
    z=spfft.fft(x,axis=1,norm="ortho")
    z*=dk[None,:]
    z=spfft.ifft(z,axis=1,norm="ortho")
    z*=dx[None,:]
    return z

def run_benchmark():
    rng=np.random.default_rng(20260929); T=32; L=4
    sizes=[256,512,1024,2048,4096]; rows=[]
    for n in sizes:
        theta_k=rng.uniform(-np.pi,np.pi,n).astype(np.float32)
        theta_x=rng.uniform(-np.pi,np.pi,n).astype(np.float32)
        dk=np.exp(1j*theta_k).astype(np.complex64)
        dx=np.exp(1j*theta_x).astype(np.complex64)
        x=(rng.standard_normal((T,n),dtype=np.float32)+1j*rng.standard_normal((T,n),dtype=np.float32)).astype(np.complex64)
        x/=np.sqrt(n)
        t0=time.perf_counter(); eye=np.eye(n,dtype=np.complex64); M=apply_implicit(eye,dk,dx)
        materialization_s=time.perf_counter()-t0; del eye; gc.collect()
        y_implicit=apply_implicit(x,dk,dx); y_dense=x@M
        rel_err_1=float(np.linalg.norm(y_implicit-y_dense)/np.linalg.norm(y_dense))
        def dense_1(): return x@M
        def implicit_1(): return apply_implicit(x,dk,dx)
        def dense_chain():
            y=x
            for _ in range(L): y=y@M
            return y
        def implicit_chain():
            y=x
            for _ in range(L): y=apply_implicit(y,dk,dx)
            return y
        reps=5 if n<=1024 else (3 if n<=2048 else 2)
        td1=median_time(dense_1,reps,1); ti1=median_time(implicit_1,reps,1)
        tdc=median_time(dense_chain,max(2,reps-1),1); tic=median_time(implicit_chain,max(2,reps-1),1)
        yd4=dense_chain(); yi4=implicit_chain()
        rel_err_4=float(np.linalg.norm(yi4-yd4)/np.linalg.norm(yd4))
        dense_mb=M.nbytes/(1024**2); implicit_mb=(dk.nbytes+dx.nbytes)/(1024**2)
        dense_work=T*n*n; struct_work=2*T*n*math.log2(n)+2*T*n
        rows.append({"N":n,"T_tokens":T,"layers_chain":L,"materialize_same_operator_s":materialization_s,
                     "dense_same_op_1layer_ms":td1*1000,"implicit_fft_phase_1layer_ms":ti1*1000,
                     "speedup_1layer_x":td1/ti1,"dense_same_op_4layer_ms":tdc*1000,
                     "implicit_fft_phase_4layer_ms":tic*1000,"speedup_4layer_x":tdc/tic,
                     "relative_error_1layer":rel_err_1,"relative_error_4layer":rel_err_4,
                     "dense_weight_MB_per_layer":dense_mb,"implicit_weight_MB_per_layer":implicit_mb,
                     "weight_memory_reduction_x":dense_mb/implicit_mb,"theoretical_work_ratio_x":dense_work/struct_work})
        del x,dk,dx,M,y_implicit,y_dense,yd4,yi4; gc.collect()
    return pd.DataFrame(rows)

if __name__=="__main__":
    print(run_benchmark().to_string(index=False))
