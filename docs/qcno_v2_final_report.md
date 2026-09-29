# Q-CNO V2 / Transformer Distillation & Native Training Benchmark

## Status

This is a classical CPU research proxy for a quantum-compatible structured neural operator.
It is **not** a quantum-hardware benchmark and does **not** claim proven quantum advantage.

## Experimental model

- Causal byte-level Transformer trained from scratch on a real 8 MiB text corpus (Vim documentation).
- Held-out split: last 5%.
- Hidden size: 128
- Attention heads: 4
- Transformer layers: 2
- FFN width: 512
- Dense baseline total parameters: 435,456
- Dense target linear parameters (Q/K/V/O/fc1/fc2): 393,216
- Metric: byte-level held-out perplexity.

Dense baseline after 600 total steps:
- PPL: 6.5967

## Q-CNO factorization

For a square row-vector operator A:

A[i,j] ≈ sum_r a_r[j] b_r[(j-i) mod N].

Each rank component is a circulant operator followed by a diagonal operator:

A_r = C(b_r) D(a_r)
    = F^-1 diag(FFT(b_r)) F D(a_r).

The cyclicly unwrapped matrix

Z[j,s] = A[(j-s) mod N, j]

turns a sum of R Q-CNO blocks into an ordinary rank-R matrix factorization:

Z ≈ U_R S_R V_R^T.

Therefore truncated SVD gives the optimal Frobenius approximation inside this
specific sum-of-diagonal-circulant class for a fixed R.

## Post-hoc distillation

- R=16, 4× linear compression: PPL ≈ 34.07
- R=32, 2× linear compression: PPL ≈ 17.38
- R=128: reconstruction relative weight error ≈ 2.55e-7 and PPL returns to ≈ 8.89

## Native Q-CNO training

At 600 total optimization steps:

| Model | Linear compression | Total model compression | Held-out PPL | Next-byte accuracy |
|---|---:|---:|---:|---:|
| Dense | 1× | 1× | 6.60 | ~50% |
| Q-CNO R=32 | 2× | 1.82× | 10.35 | 38.45% |
| Q-CNO R=16 | 4× | 3.10× | 11.34 | 34.98% |
| Q-CNO R=8 | 8× | 4.76× | 12.35 | 32.67% |

## Exact implicit-vs-dense operator runtime

Representative measured results:

- N=1024, R=8: 1.55×
- N=2048, R=8: 3.10×
- N=4096, R=8: 10.48×
- N=4096, R=4: 8.98×
- N=4096, R=1: 28.83×

Numerical relative errors were about 3e-7 to 4e-7.

## End-to-end exactness

Final R=8 Transformer, batch=16, sequence=64:

- dense-materialized Q-CNO model: 9.367 ms
- implicit FFT Q-CNO model: 50.551 ms
- relative logits error: 5.173e-07

At hidden size 128 the full implicit model is slower because FFT overhead dominates.

## What is proven / not proven

Proven:
- exact algebraic equivalence of dense-materialized and implicit FFT evaluation;
- numerical parity to ~1e-7;
- measurable CPU crossover and >10× layer speedup at N=4096, R=8;
- native structured training outperforms post-hoc compression.

Not proven:
- parity with GPT-2/Llama;
- standard BPE perplexity parity;
- fault-tolerant quantum speedup;
- efficient quantum state preparation/readout.
