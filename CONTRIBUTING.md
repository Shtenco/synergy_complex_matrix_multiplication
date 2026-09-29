# Contributing

Новое утверждение о speedup или quality должно сопровождаться:

1. exact definition оператора;
2. корректным baseline;
3. hardware/runtime context;
4. warmup + repeated timings;
5. numerical error;
6. raw CSV/JSON;
7. указанием kernel-only / layer-only / end-to-end;
8. sparse-aware baseline для sparse методов;
9. одинаковым evaluation protocol для PPL/accuracy;
10. отсутствием скрытой N×N materialization, если заявляется matrix-free inference.

Operation-count reduction нельзя выдавать за wall-clock speedup без физического измерения.
