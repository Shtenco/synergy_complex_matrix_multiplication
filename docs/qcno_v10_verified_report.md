# Q-CNO V10 — Verified Facts After Stress Testing

## Correction to the early 1000× claim

The early 6445× number was a valid timing for an exact k=8 sparse transition kernel
against a dense materialization, but it omitted coalescing and did not preserve the
75% active-state fraction observed in the real D=512 language-model checkpoint.

After adding exact coalescing and stress-testing multiple N, k and fanout settings:

- fixed k=384, R=4, N=8192: about 336× vs dense, ~1.30× vs CSR.
- constant active fraction k=0.75N, R=4, N=8192: **7.95× vs dense** and **0.092× vs CSR**.

Therefore **1000× end-to-end is not demonstrated** for a quality-preserving state fraction.

## Quality-preserving coordinate sparsity

| k | Active fraction | PPL | Accuracy |
|---:|---:|---:|---:|
| 512 | 1.000 | 8.004599 | 0.459896 |
| 448 | 0.875 | 8.012556 | 0.459479 |
| 384 | 0.750 | 8.060834 | 0.458646 |
| 320 | 0.625 | 8.212186 | 0.455521 |
| 256 | 0.500 | 8.533668 | 0.446042 |
| 192 | 0.375 | 9.291681 | 0.423333 |

Dense teacher PPL = 6.3223. Baseline Q-CNO PPL = 8.0046.

## Learned-basis state compression

Per-residual PCA:

| mode | k | active fraction | PPL | accuracy |
|---|---:|---:|---:|---:|
| baseline | 512 | 1.0 | 8.004599 | 0.459896 |
| pca_topk | 128 | 0.25 | 8.108348 | 0.454479 |
| pca_topk | 96 | 0.1875 | 8.194525 | 0.454375 |
| pca_topk | 64 | 0.125 | 8.472662 | 0.442396 |

PCA proves strong hidden-state compressibility, but PCA itself costs O(N^2).

## Fast basis attempts

Householder-24 k=128: PPL ≈ 9.36.
Learned 9-stage butterfly k=128: PPL ≈ 12.05.
Signed/permuted Hadamard k=128: PPL ≈ 16.44.

Current fast structured bases do not yet match PCA quality.

## Shared PCA and churn

Shared PCA k=128: PPL ≈ 8.2854.
At k=128 support retention was about 51.4%, 83.4%, 68.4% across the three measured transitions.

## Current factual frontier

Verified:
- highly sparse exact structured kernels can exceed 1000× vs dense;
- the advantage collapses at 75% active fraction;
- 25% hidden coefficients in learned PCA basis cause only ~1.3% extra PPL loss relative to Q-CNO;
- no tested fast basis yet preserves PCA-level quality;
- Q-CNO itself remains ~26.6% worse in PPL than the dense teacher.

A new 1000× claim should only be made if the same model simultaneously meets:
- PPL ratio <= 1.05 vs teacher;
- no hidden N^2 materialization;
- >=1000× measured end-to-end speed;
- sparse-aware baseline comparison.
