# Q-CNO exact-operator benchmark

Date: 2026-09-29

This is a **classical CPU proxy**, not a quantum-hardware benchmark.

Operator tested:
U = D_x F^-1 D_k F

The dense baseline materializes this exact same U as an N×N complex64 matrix M.
The implicit path applies the mathematically same operator with FFT + learned phase
vectors and never materializes M during inference.

Input shape:
T=32 token rows, hidden dimension N.

Chain:
4 repeated applications of the exact same operator were used to test end-to-end
composition without N×N operator materialization.

Important:
This validates computational and memory savings for this structured operator class.
It does NOT prove that arbitrary dense Transformer weights can be replaced by this
operator without accuracy loss, and it does NOT prove a hardware quantum advantage.
