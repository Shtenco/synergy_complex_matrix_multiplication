# Q-CNO continuation report — V3 → V6

## Best result reached in this continuation

The strongest pure Q-CNO quality point is now:

- architecture: adaptive sum-of-diagonal-circulant / Fourier Q-CNO
- hidden size: 512
- rank sum across 24 square operator blocks: 768
- average rank: 32
- rank range: 27..46
- linear-weight compression: **8.0×**
- common deterministic evaluation:
  - dense teacher PPL: **6.3223**
  - Q-CNO PPL: **8.0046**
  - PPL ratio: **1.2661×**
  - teacher accuracy: **50.490%**
  - Q-CNO accuracy: **45.990%**

The extra CE-only annealing reduced the previous common-eval PPL from about 8.56
to **8.0046**. A final 1e-5 learning-rate pass only
marginally improved it, indicating a real plateau rather than simple under-training.

## Exact same-operator speed result

For the V5.3 Monarch + low-rank fast operator at N=4096, the optimized grouped-BMM
implementation was compared against a materialized dense matrix representing the
mathematically identical operator.

Measured speedups:

 T_vectors  dense_ms  implicit_ms  speedup_x  relative_error
         1  2.477651     0.262805   9.427716    1.270196e-06
         4  3.824044     0.577898   6.617161    1.102851e-06
         8  6.348746     0.601954  10.546896    9.891788e-07
        16  6.435961     1.521215   4.230803    8.880091e-07
        32  8.388280     2.062728   4.066595    8.252347e-07
        64 13.100077     2.143710   6.110937    8.202583e-07
       128 15.921765     2.576400   6.179850    8.917339e-07
       256 34.213680     6.125982   5.585012    9.375190e-07

The exact numerical relative error remained around 1e-6. Speedup exceeded 5× at
T=1, 4, 8, 64, 128 and 256 in this CPU runtime.

## Negative results retained

1. Multi-frame cyclic Q-CNO: no reconstruction gain at equal rank.
2. Pure product-Q-CNO (K=8): 32× compression but PPL plateau ~21.7.
3. 4-branch × depth-2 sum/product: worse than standard additive Q-CNO.
4. Butterfly: ~26.9× compression but remained near PPL 28.
5. Monarch: 10.45× compression; extended training around PPL 13.9.
6. Monarch + fixed low-rank residual: 8.13× compression; CE annealing ~11.62.
7. Adaptive/sensitivity allocation: only modest gains (~11.36 PPL).
8. Mixed dense first-layer Q/K: little improvement.
9. MPO chi=32: ~15.5× compression but mean relative weight error ~0.959.

## Current frontier

Separately demonstrated:
- quality: adaptive Q-CNO 8× compression, PPL ratio 1.266×;
- speed: >5× exact same-operator CPU acceleration at N=4096 in several T regimes;
- numerical parity around 1e-6 to 1e-7.

Not yet closed simultaneously: PPL ratio <=1.05, >=8× compression, >=5× measured speed.
