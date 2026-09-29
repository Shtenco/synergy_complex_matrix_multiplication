# Q-CNO V7 — 100× matrix-operation benchmark

## Scope

These are **classical CPU benchmarks of qubit-inspired representations**.
They are not quantum-hardware benchmarks.

## Complex-amplitude / Fourier operator

y = IFFT(FFT(x) * FFT(b)) * a

At N=12288:

| T | Dense ms | Implicit ms | Speedup | Relative error |
|---:|---:|---:|---:|---:|
| 1 | 16.503 | 0.228 | 72.26× | 7.08e-7 |
| 4 | 82.039 | 0.586 | 140.00× | 4.14e-7 |
| 8 | 107.511 | 1.644 | 65.38× | 4.12e-7 |
| 16 | 88.562 | 2.292 | 38.64× | 4.16e-7 |

## Normalization + weight deduplication

Each output column is represented as:

w_j = scale_j * state[id_j]

For ±1 normalized activations:

dot(x, state) = N - 2 * popcount(xor(bits(x), bits(state))).

Best measured point:

- N = 12288
- T = 8
- dictionary K = 64
- speedup = **199.02×**
- relative error = **2.91e-7**
- dense weight memory = 576 MiB
- structured weight memory = 0.1875 MiB
- memory reduction = **3072×**

This proves >100× matrix-operation acceleration for a compatible structured operator.
It does NOT prove arbitrary Transformer weight conversion at equal quality.
