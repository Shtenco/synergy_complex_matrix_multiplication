import torch,torch.nn.functional as F,glob,time,math,json,os
from pathlib import Path
ns={}; exec(open('/mnt/data/qcno_v31_e2e.py').read().split('rows=[];')[0],ns)
Student=ns['Student']; DenseStudent=ns['DenseStudent']; alloc_all=ns['alloc_all']; D=ns['D']; BLOCK=ns['BLOCK']; HEADS=ns['HEADS']
torch.set_num_threads(min(5,os.cpu_count() or 5)); torch.manual_seed(20260929)
paths=sorted(glob.glob('/usr/share/vim/vim91/doc/*.txt'));raw=b'\n\n'.join(Path(p).read_bytes() for p in paths)[:8*1024*1024];data=torch.tensor(list(raw),dtype=torch.long);train=data[:int(.95*len(data))]
def batch(bs=4):
 ix=torch.randint(0,len(train)-BLOCK-1,(bs,)); return torch.stack([train[k:k+BLOCK] for k in ix])
teacher=DenseStudent(); teacher.load_state_dict(torch.load('/mnt/data/qcno_v3_teacher_d512.pt',map_location='cpu',weights_only=False)['state_dict']);teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
student=Student(alloc_all['32']); student.load_state_dict(torch.load('/mnt/data/qcno_v31_adaptive_avg32_best_preop.pt',map_location='cpu',weights_only=False)['state_dict'])
params=[]
for n,p in student.named_parameters():
 if n.endswith('.a') or n.endswith('.b'): params.append(p)
 else: p.requires_grad_(False)
opt=torch.optim.AdamW(params,lr=1.5e-4,weight_decay=0.0)
def nmse(yhat,y): return F.mse_loss(yhat,y)/(y.pow(2).mean().detach()+1e-6)
def step_loss(idx):
 with torch.no_grad(): x=teacher.tok(idx)+teacher.pos(torch.arange(idx.shape[1]))[None,:,:]
 losses=[]
 for tb,sb in zip(teacher.blocks,student.blocks):
  with torch.no_grad():
   u=tb.ln1(x); tq=tb.attn.q(u); tk=tb.attn.k(u); tv=tb.attn.v(u)
  losses += [nmse(sb.attn.q(u),tq),nmse(sb.attn.k(u),tk),nmse(sb.attn.v(u),tv)]
  with torch.no_grad():
   B,T,C=x.shape; q=tq.view(B,T,HEADS,C//HEADS).transpose(1,2); k=tk.view(B,T,HEADS,C//HEADS).transpose(1,2); v=tv.view(B,T,HEADS,C//HEADS).transpose(1,2); ctx=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(B,T,C); to=tb.attn.o(ctx); xa=x+to; m=tb.ln2(xa); t1=tb.fc1(m); act=F.gelu(t1); t2=tb.fc2(act)
  losses += [nmse(sb.attn.o(ctx),to),nmse(sb.fc1(m),t1),nmse(sb.fc2(act),t2)]
  with torch.no_grad(): x=xa+t2
 return sum(losses)/len(losses)
st=time.perf_counter()
for step in range(1,81):
 idx=batch(); loss=step_loss(idx); opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
 if step==1 or step%20==0: print('step',step,'operator_nmse',float(loss.detach()),'elapsed',time.perf_counter()-st,flush=True)
res={'steps':80,'seconds':time.perf_counter()-st,'final_operator_nmse':float(loss.detach())}
out='/mnt/data/qcno_v33_adaptive_avg32_opdistill.pt';torch.save({'state_dict':student.state_dict(),'alloc':alloc_all['32'],'result':res},out);Path('/mnt/data/qcno_v33_operator_distill_metrics.json').write_text(json.dumps(res,indent=2));print('FINAL',json.dumps(res),flush=True)
