# Q-CNO V3.1 / V3.3 — Adaptive Rank and Speed Frontier

Dense teacher: 2-layer causal byte-level Transformer, hidden 512, 8 heads, FFN 2048.

Common deterministic evaluation before the final later CE annealing:
- Dense teacher: PPL 6.322
- Adaptive avgR=32: PPL 8.558
- Adaptive avgR=16: PPL 11.327
- Adaptive avgR=8: PPL 15.522

Later CE annealing improved avgR=32 further to PPL 8.0046; see qcno_v3_v6_continuation_report.md and qcno_v3_v6_frontier.csv.

## Adaptive rank

Adaptive allocation improved same-parameter uniform baselines:
- R=8: uniform 17.266 → adaptive 15.522
- R=16: uniform 14.114 → adaptive 11.327
- R=32: uniform 10.542 → adaptive 8.558

## Exact same-operator speed frontier at N=4096, T=32

- R=8: 9.70× speedup
- R=16: 3.36×
- R=32: 0.83×

Numerical errors remain around 1e-7.

## Activation-aware operator distillation

A teacher-forced local objective reduced normalized operator error but worsened end-to-end PPL before global recovery training. This falsified the assumption that minimizing every local linear activation error independently is sufficient to preserve a nonlinear Transformer.

## What V3 proves

1. Q-CNO Transformer training/distillation is viable.
2. Adaptive rank improves the quality/parameter frontier.
3. Dense materialization and implicit execution agree to floating-point precision.
4. Large-N low-rank Q-CNO can be faster than dense CPU matmul.
5. The simultaneous PPL<=1.05× teacher + >=5× speed + >=8× compression target was not closed.
