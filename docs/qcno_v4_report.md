# Q-CNO V4 — Fast quantum-compatible operator search

## Current best measured frontier

| Model | PPL | Linear compression | Large-N exact-operator speed proxy |
|---|---:|---:|---:|
| Dense teacher | 6.322 | 1× | 1× |
| Adaptive Q-CNO avgR=32 | 8.558 | 8× | 0.83× |
| Adaptive Q-CNO avgR=16 | 11.327 | 16× | 3.36× |
| Adaptive Q-CNO avgR=8 | 15.522 | 32× | 9.70× |
| Q-CNO avgR=8 + Householder K=4 | **13.790** | **21.33×** | **7.08×** |

The Householder hybrid is the first tested configuration that simultaneously clears
>5× large-N exact-operator CPU speed and >8× linear-weight compression while improving
quality relative to the plain avgR=8 speed branch. It does **not** close the quality target.

## V3.3 negative result

Teacher-forced layerwise optimization reduced normalized local operator error from
about 0.624 to 0.418, yet end-to-end PPL temporarily worsened from about 8.56 to 15.63.

Conclusion: local linear parity does not guarantee global Transformer parity.

## V4.0 pure butterfly

9-stage learned butterfly at hidden 512 used about 26.95× fewer linear parameters,
but remained near PPL 29 after the tested training window.

## V4.2 Householder correction

Architecture:

    y0 = Q-CNO_R8(x)
    y_(k+1) = D_k [ y_k - alpha_k proj_v_k(y_k) ]

Representative exact same-operator benchmark:
- N=4096, R=8, K=4
- speedup 7.077×
- relative error 3.892e-07

Trained d=512 model:
- common PPL 13.790
- accuracy 26.292%
- linear compression 21.33×

## Not proved

- quantum hardware advantage;
- state-preparation/readout efficiency;
- fault-tolerant resource advantage;
- <=5% PPL gap to dense teacher.
