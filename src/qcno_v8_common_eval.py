import torch, math
from pathlib import Path
src=Path('/mnt/data/qcno_v8_greedy_dedup_50.py').read_text()
prefix=src[:src.index("base=Model(base_alloc)")]
ns={'__name__':'defs'}
exec(compile(prefix,'defs','exec'),ns)
Model=ns['Model']; evalm=ns['evalm']; base_alloc=ns['base_alloc']; ck=ns['ck']
rows=[]
base=Model(base_alloc); base.load_state_dict(ck['state_dict']); l,p,a=evalm(base,40); rows.append(('Q-CNO base',sum(base_alloc.values()),6291456/(2*512*sum(base_alloc.values())),l,p,a))
for name,path in [('Dedup2 greedy','/mnt/data/qcno_v8_greedy_factor2.pt'),('Dedup4 kmeans','/mnt/data/qcno_v8_dedup_factor4.pt')]:
    c=torch.load(path,map_location='cpu',weights_only=False); m=Model(c['alloc']); m.load_state_dict(c['state_dict']); l,p,a=evalm(m,40); rows.append((name,sum(c['alloc'].values()),6291456/(2*512*sum(c['alloc'].values())),l,p,a))
for r in rows: print('|'.join(map(str,r)))
