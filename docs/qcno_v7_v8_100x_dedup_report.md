# Q-CNO V7/V8 — 100× matrix operations + real-weight deduplication

## Result 1 — complex-amplitude execution

Exact same structured operator, N=12288:

- best Q-FFT speedup: **140.00×**
- numerical relative error: **4.141e-07**

The dense baseline is the fully materialized form of the same Fourier/circulant/diagonal transformation.

## Result 2 — normalized + deduplicated bit-amplitude dictionary

Each output vector:

    weight = scale × normalized sign-state[ID]

Unique states are evaluated once with packed XOR + popcount.

Best measured point:
- N=12288
- T=8
- dictionary K=64
- measured speedup **199.02×**
- relative error **2.914e-07**
- dense weight memory 576.0 MiB
- structured weight memory 0.1875 MiB
- memory reduction **3072×**

This is a classical qubit-inspired benchmark, not quantum hardware.

## Result 3 — real trained Q-CNO checkpoint

| Model | rank sum | compression | PPL | accuracy |
|---|---:|---:|---:|---:|
| Q-CNO base | 768 | 8.0× | 8.004599 | 0.459896 |
| Dedup2 greedy | 391 | 15.7136× | 13.760652 | 0.270729 |
| Dedup4 kmeans | 201 | 30.5672× | 27.112725 | 0.121146 |

Whole-vector dedup increases compression substantially but degrades language quality.

## Next architecture — V9

Product-quantized normalized dedup:
- split each weight vector into G normalized subspaces;
- each subspace selects one of K shared states;
- store state IDs + local scales;
- share state evaluations across Q/K/V when input activation is shared;
- distill end-to-end.

Hard gates:
- PPL <= 8.4 then <= 6.65;
- linear compression >=8×;
- large-N exact dictionary kernel >=100×;
- no hidden N² materialization.
