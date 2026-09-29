# Q-CNO V11 — Verified continuation

## Goal

V11 targets:

- a fast learned sparse basis approaching PCA-quality;
- support-preserving complex gates with exact \`k -> k\` support;
- improved Q-CNO expressivity without hidden \`N^2\` materialization;
- eventually PPL(student)/PPL(teacher) <= 1.05, >=8x compression and >=100x end-to-end wall-clock.

The hard end-to-end gate is **not closed** in this report.

## 1. Deep fast learned basis

Architecture:

- hidden width N=512;
- one shared orthogonal butterfly-like basis;
- 4 passes x 9 stages = 36 learned stages;
- 9,216 angle parameters;
- fixed permutations between passes;
- top-k coding followed by the exact inverse transform.

Training objective maximized retained top-k energy across real hidden states from the best Q-CNO checkpoint.

Retained energy after training:

- k=64: 0.81748
- k=96: 0.86823
- k=128: 0.90387

Common deterministic 40-batch evaluation:

| basis | k | PPL |
|---|---:|---:|
| Q-CNO baseline | 512 | 8.0046 |
| PCA oracle | 128 | 8.1083 |
| Householder-24 | 128 | 9.3596 |
| old Butterfly-9 | 128 | 12.0550 |
| **V11 Deep Butterfly-36** | **128** | **9.2396** |
| V11 Deep Butterfly-36 | 96 | 9.9786 |
| V11 Deep Butterfly-36 | 64 | 11.4586 |

At k=128, V11 improves PPL relative to the old Butterfly-9 by **23.35%** and relative to Householder-24 by **1.28%**.

It is still **13.95%** worse than the PCA quality oracle.

### Runtime caveat

The current PyTorch reference implementation is not a fused kernel.

For 240 x 512 states, encode+decode measured approximately:

- dense 512x512 basis: ~2.87 ms with one PyTorch thread;
- V11 Deep Butterfly-36: ~17.58 ms with one PyTorch thread.

Therefore V11 has improved algorithmic structure and quality, but the current staged PyTorch implementation is **not yet a wall-clock fast basis** at N=512. A fused kernel is required before making a speed claim for this basis.

## 2. Exact support-preserving complex gates

The gate network operates only on the k active complex amplitudes.

Each layer pairs active slots and applies an SU(2)-like unitary two-state transform:

    a' = cos(theta) a + exp(i phi) sin(theta) b
    b' = -exp(-i phi) sin(theta) a + cos(theta) b

The active index set is unchanged:

    support_before = k
    support_after  = k

so there is no fanout or fill-in.

The exact same composite gate operator was materialized as a dense k x k complex matrix and used as the baseline.

Canonical timings use **single-thread BLAS** to avoid the large small-matrix thread-launch overhead that inflated an earlier multithread comparison.

Selected results, T=32:

| k | gate layers | speedup vs dense active operator |
|---:|---:|---:|
| 128 | 4 | 2.19x |
| 256 | 4 | 4.02x |
| 512 | 4 | 6.48x |
| 1024 | 4 | 6.83x |
| 1024 | 8 | 3.23x |

Best canonical wall-clock speedup in this sweep: **6.83x**, at k=1024, L=4.

Maximum numerical relative error: **2.294e-07**.

Operation-count reduction can be much larger (up to 512x in this sweep), but that must not be reported as wall-clock speedup.

### Correction of the multithread result

A preliminary comparison with multithread BLAS showed apparent 100x-300x gains for some small active matrices. A single-thread stress test showed that most of that number came from BLAS thread/dispatch overhead. Those 100x-300x values are **not** the canonical V11 speed result.

## 3. Two-basis Q-CNO at identical rank budget

A second V11 hypothesis used

    W ~= Q_r1 + P Q_r2 P^T
    r1 + r2 = r

with one ordinary cyclic basis and one global bit-reversal basis.

All 24 real teacher Q/K/V/O/FFN square blocks improved, but the mean Frobenius error reduction was only **0.123%**.

This is too small to justify doubling the number of FFT branches. This branch is therefore not considered the main V11 path.

## 4. Current V11 status

Verified:

- Deep fast basis quality improved substantially versus the previous fast basis.
- It slightly surpassed Householder-24 at k=128.
- PCA remains the quality oracle and is still materially better.
- Exact support-preserving complex gates maintain \`k -> k\`, eliminating V10 fill-in.
- Canonical gate wall-clock gain is currently single-digit, not 100x.
- Two-basis Q-CNO gives only a tiny reconstruction benefit at fixed rank budget.

Not verified:

- PPL(student)/PPL(teacher) <= 1.05.
- >=8x total compression after counting the new basis parameters.
- >=100x end-to-end Transformer wall-clock.
- >=1000x end-to-end wall-clock.
- hardware quantum advantage.

## 5. Next falsifiable V11.1

1. keep the 36-stage basis, but fuse it into one native kernel;
2. make the basis parameter-budget-neutral by removing exactly nine Q-CNO rank components (9 x 1024 parameters = 9,216 basis parameters);
3. jointly distill the Q-CNO factors and the sparse basis from teacher hidden states and logits;
4. keep hidden states in sparse coordinates between compatible operations rather than decode/re-encode at every residual point;
5. apply the support-preserving complex gate network directly to the active coefficients;
6. benchmark against dense BLAS, CSR/sparse-aware baselines and the current Q-CNO.

Only one-model simultaneous measurements should count toward the final V11 gate.
