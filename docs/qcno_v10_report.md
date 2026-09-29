# Q-CNO V10 — Matrix-Free Sparse Complex State Machine

## Scope

This benchmark tests a qubit-inspired sparse complex state-transition representation on classical CPU hardware. It is not a quantum-hardware benchmark.

The exact transition operator uses a four-phase alphabet:

    {1, i, -1, -i}

and R=4 shared cyclic transition templates. The input state contains only k=8 active normalized complex amplitudes.

The custom kernel carries the state as (index, real amplitude, imaginary amplitude) and performs only phase rotations + index transitions. It never materializes N^2 weights or an N-dimensional dense intermediate state.

## Exact same-operator kernel benchmark

| N | active k | dense ms | CSR ms | state kernel ms | vs dense | vs CSR |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 8 | 0.4600 | 0.03037 | 0.001532 | 300.27× | 19.83× |
| 4096 | 8 | 2.3864 | 0.02638 | 0.001602 | 1489.66× | 16.47× |
| 8192 | 8 | 10.2617 | 0.02986 | 0.001592 | 6445.77× | 18.76× |

These are exact highly-sparse-kernel results, not Transformer end-to-end speedups.

## Real checkpoint sparsity probe

The existing Q-CNO checkpoint was evaluated while retaining only top-k residual hidden coordinates and renormalizing the state.

This was a quality diagnostic; the Q-CNO layers themselves still executed as dense PyTorch tensors.

## Important correction

Later stress testing added coalescing and constant active-fraction tests. At k=0.75N, N=8192, the custom state kernel achieved about 7.95× vs dense and was slower than CSR. Therefore the early 6445× number must not be quoted as quality-preserving Transformer acceleration.
