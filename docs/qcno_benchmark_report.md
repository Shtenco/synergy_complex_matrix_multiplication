# Quantum Complex Neural Operator — classical proxy benchmark

This benchmark is intentionally **not** claimed as a quantum-hardware benchmark.
It compares a dense complex linear layer `Y=XW` against a structured unitary complex operator

`U = D_x F^-1 D_k F`

where `F` is an FFT and `D_x, D_k` are learned unit-modulus complex diagonal phase operators.

Rows/tokens T = 32.

Environment: Python 3.13.5, NumPy 2.3.5, Linux x86_64.

Key distinction:
- dense complex layer: O(T N^2) arithmetic and O(N^2) weight storage;
- structured Q-CNO proxy: O(T N log N) arithmetic and O(N) learned phase storage;
- a fault-tolerant quantum implementation could potentially apply the analogous QFT/phase circuit only when state preparation, phase oracles, precision, success probability, and readout assumptions are explicitly satisfied.
