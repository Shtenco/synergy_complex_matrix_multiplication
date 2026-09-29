# SYNERGY Complex Matrix Multiplication

> **Исследовательский репозиторий по структурированным комплексным операторам, Q-CNO, квантово-вдохновлённым представлениям, дедупликации весов, Product-Quantized dictionaries и matrix-free sparse state machines.**

**Статус:** ACTIVE / EXPERIMENTAL  
**Главный принцип:** сильное утверждение считается закрытым только тогда, когда есть код, raw metrics, numerical check и корректный baseline.

---

## 1. Что исследует проект

Стартовая точка — обычный dense linear operator:

\[
Y=XW.
\]

Цель — понять, можно ли заменить произвольную плотную матрицу комбинацией структурированных операторов и shared states так, чтобы:

\[
O(TN^2)
\]

постепенно заменить на

\[
O(TN\log N),\quad O(RTN\log N),\quad O(KN)
\]

или на разреженную динамику состояния, вообще не материализующую \(N\times N\) матрицу.

Проект экспериментально проверяет:

1. exact same-operator implicit execution;
2. complex/Fourier diagonal–circulant operators;
3. native structured Transformer training;
4. adaptive ranks;
5. Butterfly / Householder / Monarch / MPO alternatives;
6. normalization + weight deduplication;
7. product-quantized shared dictionaries;
8. sparse complex state transitions;
9. hidden-state compressibility;
10. честные sparse-aware stress tests;
11. потенциальную quantum-compatible форму без заявления о quantum advantage.

---

# 2. Что уже подтверждено

- exact Q-CNO operator можно исполнять без материализации \(N\times N\);
- V1 exact same-operator benchmark дал **44.32×** при \(N=4096\);
- implicit weight storage в той точке был **2048×** меньше dense complex64 matrix;
- native Q-CNO Transformer обучается существенно лучше post-hoc проекции;
- лучший adaptive Q-CNO при **8× linear-weight compression** имеет PPL **8.0046** против teacher **6.3223**;
- V5.3 exact structured kernel при \(N=4096\) достиг **10.55×** в одном token-width режиме;
- exact complex Q-FFT operator при \(N=12288\) достиг **140.00×** против dense materialization того же оператора;
- normalized deduplicated bit-amplitude dictionary kernel достиг **199.02×**;
- PCA показывает, что hidden state можно сократить до \(128/512=25\%\) коэффициентов при PPL **8.1083**, то есть всего примерно +1.30% к PPL самого Q-CNO;
- highly sparse state-transition kernels действительно могут переходить 1000× против dense materialization;
- stress-test показал, что этот экстремальный выигрыш **не переносится автоматически** на quality-preserving high-active-fraction режим.

---

# 3. Что НЕ доказано

В этом проекте пока **не доказано**, что:

- end-to-end Transformer ускорен в 100×, 199×, 1000× или 6445×;
- существует hardware quantum advantage;
- arbitrary pretrained LLM можно заменить Q-CNO без существенной потери качества;
- student/teacher PPL ratio уже \(\le1.05\);
- generic sparse CSR проигрывает нашей state machine при высокой доле активных состояний;
- complex numbers или QFT сами по себе ускоряют произвольный dense GEMM.

Это принципиальная часть проекта: README сохраняет не только красивые числа, но и falsification tests.

---

# 4. V1 — exact same-operator Q-CNO

Базовый оператор:

\[
U=D_xF^{-1}D_kF.
\]

Сравнивались:

1. полная материализация \(M\in\mathbb C^{N\times N}\) и вычисление \(Y=XM\);
2. implicit FFT-phase execution без материализации \(M\).

| N | Dense 1 layer | Implicit | Speedup | Dense weight | Implicit | Memory reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.112 ms | 0.067 ms | 1.67× | 0.5 MiB | 0.0039 MiB | 128× |
| 512 | 0.381 ms | 0.094 ms | 4.06× | 2 MiB | 0.0078 MiB | 256× |
| 1024 | 1.381 ms | 0.224 ms | 6.17× | 8 MiB | 0.0156 MiB | 512× |
| 2048 | 5.321 ms | 0.395 ms | 13.47× | 32 MiB | 0.0313 MiB | 1024× |
| 4096 | 47.226 ms | 1.066 ms | **44.32×** | 128 MiB | 0.0625 MiB | **2048×** |

Numerical parity at \(N=4096\):

\[
\varepsilon_{1}\approx3.43\times10^{-7},
\qquad
\varepsilon_{4}\approx6.84\times10^{-7}.
\]

Primary evidence:

- [src/qcno_exact_operator_benchmark.py](src/qcno_exact_operator_benchmark.py)
- [results/qcno_exact_operator_results.csv](results/qcno_exact_operator_results.csv)
- [docs/qcno_exact_operator_report.md](docs/qcno_exact_operator_report.md)
- [figures_svg/v1_exact_operator_speed.svg](figures_svg/v1_exact_operator_speed.svg)

---

# 5. Математическая основа Q-CNO

## 5.1. Diagonal–Fourier operator

\[
U=D_xF^{-1}D_kF.
\]

Классическая implicit стоимость для \(T\) token-vectors:

\[
O(TN\log N)
\]

вместо dense:

\[
O(TN^2).
\]

## 5.2. Сумма diagonal–circulant блоков

Более выразительная форма:

\[
W\approx\sum_{r=1}^{R}C(b_r)D(a_r),
\]

где

\[
C(b_r)=F^{-1}\operatorname{diag}(Fb_r)F.
\]

Для row-vector convention:

\[
A_r[i,j]=a_r[j]b_r[(j-i)\bmod N].
\]

## 5.3. Cyclic unwrap

Вводится

\[
Z[j,s]=W[(j-s)\bmod N,j].
\]

Тогда \(R\) Q-CNO components соответствуют rank-\(R\) разложению:

\[
Z\approx\sum_{r=1}^{R}a_rb_r^T.
\]

Truncated SVD

\[
Z_R=U_R\Sigma_RV_R^T
\]

даёт Frobenius-optimal approximation внутри этого конкретного diagonal–circulant класса.

Важный результат поздних версий: минимальная Frobenius weight error **не гарантирует** минимальный language-model loss.

---

# 6. V2 — реальный Transformer

Эксперимент:

- causal byte-level Transformer;
- hidden size 128;
- 2 layers;
- 4 attention heads;
- FFN 512;
- dense baseline 435,456 parameters;
- Q/K/V/O/fc1/fc2 target linear parameters 393,216.

После 600 training steps:

| Model | Linear compression | Total model compression | PPL |
|---|---:|---:|---:|
| Dense | 1× | 1× | **6.5967** |
| Native Q-CNO R=32 | 2× | 1.82× | 10.3454 |
| Native Q-CNO R=16 | 4× | 3.10× | 11.3416 |
| Native Q-CNO R=8 | 8× | 4.76× | 12.3465 |

Главный вывод: **native structured training намного сильнее post-hoc compression**.

Например, post-hoc \(R=16\) давал PPL около 34, тогда как native \(R=16\) достиг около 11.34.

Files:

- [src/qcno_v2_train.py](src/qcno_v2_train.py)
- [src/qcno_v2_distill.py](src/qcno_v2_distill.py)
- [src/qcno_v2_native_train.py](src/qcno_v2_native_train.py)
- [src/qcno_v21_hybrid_train.py](src/qcno_v21_hybrid_train.py)
- [results/qcno_v2_native_summary.csv](results/qcno_v2_native_summary.csv)
- [docs/qcno_v2_final_report.md](docs/qcno_v2_final_report.md)
- [figures_svg/v2_quality_compression.svg](figures_svg/v2_quality_compression.svg)

---

# 7. V3 — teacher → adaptive Q-CNO

Common deterministic evaluation:

| Model | Linear compression | PPL | PPL/teacher | Accuracy |
|---|---:|---:|---:|---:|
| Dense teacher | 1× | **6.3223** | 1.0000 | 50.49% |
| Adaptive Q-CNO avgR=32 | **8×** | **8.0046** | **1.2661×** | 45.99% |
| Adaptive Q-CNO avgR=16 | 16× | 11.3267 | 1.7916× | 33.54% |
| Adaptive Q-CNO avgR=8 | 32× | 15.5218 | 2.4551× | 23.64% |

Для avgR=32:

- rank sum = 768 по 24 square blocks;
- average rank = 32;
- rank allocation неравномерен;
- linear compression = 8×;
- CE annealing улучшил PPL примерно с 8.56 до 8.0046;
- дальнейший low-LR pass почти не улучшил PPL — наблюдается архитектурное плато.

Evidence:

- [src/qcno_v31_adaptive.py](src/qcno_v31_adaptive.py)
- [src/qcno_v33_operator_distill.py](src/qcno_v33_operator_distill.py)
- [results/qcno_v3_v6_frontier.csv](results/qcno_v3_v6_frontier.csv)
- [results/qcno_v31_common_eval.json](results/qcno_v31_common_eval.json)
- [figures_svg/v3_v6_quality_frontier.svg](figures_svg/v3_v6_quality_frontier.svg)

---

# 8. V4–V6 — альтернативные fast operators

Проект сохраняет отрицательные результаты.

## Butterfly

Высокая compression, но качество недостаточно. Pure butterfly и несколько hybrid variants не обошли adaptive Q-CNO quality frontier.

## Householder

Householder reflections оказались полезным более выразительным fast basis. В operator-family V4 они не закрыли gap к teacher, но позже в V10 state compression они существенно обошли Hadamard и butterfly.

## Product / multi-basis / hybrid

Проверялись:

- product-Q-CNO;
- sum × product;
- multibasis;
- Q-CNO + low-rank residual.

Ни один из этих вариантов не обошёл лучший adaptive Q-CNO по quality при сопоставимом бюджете.

## Monarch + low-rank

V5.3 особенно важен как урок kernel engineering. Наивный einsum почти скрыл алгоритмический выигрыш; grouped batched matmul изменил картину.

При \(N=4096\):

| T | Speedup | Relative error |
|---:|---:|---:|
| 1 | 9.43× | 1.27e-6 |
| 4 | 6.62× | 1.10e-6 |
| 8 | **10.55×** | 9.89e-7 |
| 16 | 4.23× | 8.88e-7 |
| 32 | 4.07× | 8.25e-7 |
| 64 | 6.11× | 8.20e-7 |
| 128 | 6.18× | 8.92e-7 |
| 256 | 5.59× | 9.38e-7 |

Files:

- [src/qcno_v53_monarch_lowrank.py](src/qcno_v53_monarch_lowrank.py)
- [src/qcno_v53_speed_bmm.py](src/qcno_v53_speed_bmm.py)
- [src/qcno_v53_tokenwidth_speed.py](src/qcno_v53_tokenwidth_speed.py)
- [results/qcno_v53_tokenwidth_speed_results.csv](results/qcno_v53_tokenwidth_speed_results.csv)

## MPO / Tensor Train

Post-hoc MPO decomposition teacher weights также не дал достаточно сильной reconstruction quality при низком bond dimension.

- [src/qcno_v6_mpo_operator_eval.py](src/qcno_v6_mpo_operator_eval.py)
- [results/qcno_v6_mpo_operator_results.csv](results/qcno_v6_mpo_operator_results.csv)

---

# 9. V7 — 100× barrier на exact structured kernels

## 9.1. Complex amplitude / Q-FFT

Exact operator:

\[
y=\operatorname{IFFT}(\operatorname{FFT}(x)\odot\operatorname{FFT}(b))\odot a.
\]

Dense baseline — полная materialization того же operator.

Для \(N=12288\):

| T | Dense | Implicit | Speedup | Error |
|---:|---:|---:|---:|---:|
| 1 | 16.50 ms | 0.228 ms | 72.26× | 7.08e-7 |
| 4 | 82.04 ms | 0.586 ms | **140.00×** | 4.14e-7 |
| 8 | 107.51 ms | 1.644 ms | 65.38× | 4.12e-7 |
| 16 | 88.56 ms | 2.292 ms | 38.64× | 4.16e-7 |

Dense storage 576 MiB против 0.140625 MiB implicit: **4096× storage reduction**.

## 9.2. Normalized deduplicated bit-amplitude dictionary

\[
w_j=s_jq_{id_j}.
\]

Для normalized \(\pm1\) states:

\[
x\cdot q=N-2\,\operatorname{popcount}(x\oplus q).
\]

Лучший измеренный point:

- \(N=12288\);
- \(T=8\);
- dictionary \(K=64\);
- dense 82.72 ms;
- implicit 0.416 ms;
- **199.02×**;
- relative error \(2.91\times10^{-7}\);
- dense storage 576 MiB;
- structured storage 0.1875 MiB;
- **3072× memory reduction**.

Это **не arbitrary Transformer**. Это exact compatible dictionary-coded operator.

Evidence:

- [results/qcno_v7_100x_matrix_benchmark.csv](results/qcno_v7_100x_matrix_benchmark.csv)
- [docs/qcno_v7_100x_report.md](docs/qcno_v7_100x_report.md)
- [figures_svg/v7_100x_frontier.svg](figures_svg/v7_100x_frontier.svg)

---

# 10. V8 — дедупликация реального Q-CNO

| Model | Rank sum | Linear compression | PPL | Accuracy |
|---|---:|---:|---:|---:|
| Q-CNO base | 768 | 8.00× | **8.0046** | 45.99% |
| Greedy dedup ×2 | 391 | **15.71×** | 13.7607 | 27.07% |
| Dedup ×4 | 201 | **30.57×** | 27.1127 | 12.11% |

Вывод: whole-vector dedup быстро разрушает expressive capacity.

- [src/qcno_v8_greedy_dedup_50.py](src/qcno_v8_greedy_dedup_50.py)
- [src/qcno_v8_dedup_factor4_run.py](src/qcno_v8_dedup_factor4_run.py)
- [results/qcno_v8_real_checkpoint_dedup.csv](results/qcno_v8_real_checkpoint_dedup.csv)

---

# 11. V9 — Product-Quantized Qubit Dictionary

Вместо одного prototype на весь vector:

\[
w_j=[w_j^{(1)},\ldots,w_j^{(G)}],
\]

\[
w_j^{(g)}=s_{jg}q^{(g)}_{c_{jg}}.
\]

Для \(G=16,K=64\):

\[
64^{16}=2^{96}
\]

возможных комбинаций при хранении \(G\times K\) prototype states плюс IDs/scales.

Особенно важная идея — shared projection для Q/K/V:

\[
z_{g,k}=\langle x_g,q_k^{(g)}\rangle
\]

вычисляется один раз, а Q/K/V получают свои дешёвые decode/routing.

Подтверждённые промежуточные данные:

- грубый \(G=8\) оказался слабым;
- candidate \(G=32,K=48\);
- pre-training PPL около 59;
- первые ~30 recovery steps снизили PPL примерно до 18.4;
- полный V9 pipeline не был закрыт до финального checkpoint из-за медленного prototype forward/timeouts.

Raw evidence:

- [results/qcno_v9_target_pre.csv](results/qcno_v9_target_pre.csv)
- [results/qcno_v9_fast_target_pre.csv](results/qcno_v9_fast_target_pre.csv)
- [results/qcno_v9_G32K48_curve_part.csv](results/qcno_v9_G32K48_curve_part.csv)

---

# 12. V10 — Matrix-Free Sparse Complex State Machine

State representation:

\[
S=\{(i,\alpha_i)\}_{i\in\mathcal A},\qquad |\mathcal A|=k\ll N.
\]

Phase alphabet:

\[
\{1,i,-1,-i\}.
\]

Цель — хранить index + amplitude + phase и выполнять transitions без dense hidden vector и без \(N^2\) weights.

## 12.1. Почему ранние 1000× нельзя считать Transformer speedup

Extremely sparse exact kernels действительно дали 1000×+ против dense materialization.

После stress-test были добавлены:

- coalescing duplicate indices;
- несколько random seeds;
- \(k\), \(R\), \(N\) sweep;
- CSR sparse-aware baseline.

Для \(k=384,R=4\):

| N | Dense | CSR | State kernel | vs dense | vs CSR |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.379 ms | 0.0346 ms | 26.3 µs | 14.41× | 1.32× |
| 4096 | 3.173 ms | 0.0357 ms | 26.4 µs | 120.15× | 1.35× |
| 8192 | 12.927 ms | 0.0502 ms | 38.5 µs | **335.98×** | **1.30×** |

Но при constant active fraction \(k=0.75N\):

| N | k | vs dense | vs CSR |
|---:|---:|---:|---:|
| 1024 | 768 | 1.15× | 0.687× |
| 2048 | 1536 | 2.51× | 0.298× |
| 4096 | 3072 | 4.24× | 0.113× |
| 8192 | 6144 | **7.95×** | **0.092×** |

В последнем режиме CSR существенно быстрее custom state kernel.

**Falsification:** high active fraction + fanout создают fill-in и уничтожают экстремальный sparse advantage.

Evidence:

- [results/qcno_v101_stress_benchmark.csv](results/qcno_v101_stress_benchmark.csv)
- [results/qcno_v10_constant_fraction_speed.csv](results/qcno_v10_constant_fraction_speed.csv)
- [docs/qcno_v10_verified_report.md](docs/qcno_v10_verified_report.md)
- [figures_svg/v10_constant_fraction_speed.svg](figures_svg/v10_constant_fraction_speed.svg)

---

# 13. Hidden-state compressibility

## Coordinate top-k

Common 40-batch evaluation:

| k | Active | PPL | Accuracy |
|---:|---:|---:|---:|
| 512 | 100% | **8.0046** | 45.99% |
| 448 | 87.5% | 8.0126 | 45.95% |
| 384 | 75% | **8.0608** | 45.86% |
| 320 | 62.5% | 8.2122 | 45.55% |
| 256 | 50% | 8.5337 | 44.60% |
| 192 | 37.5% | 9.2917 | 42.33% |

75% coordinate support стоит лишь около +0.7% PPL относительно Q-CNO baseline.

## PCA basis

В learned PCA basis:

| k | Active | PPL | Accuracy |
|---:|---:|---:|---:|
| 512 | 100% | **8.0046** | 45.99% |
| 128 | 25% | **8.1083** | 45.45% |
| 96 | 18.75% | 8.1945 | 45.44% |
| 64 | 12.5% | 8.4727 | 44.24% |

Это доказывает, что hidden state **информационно** значительно более compressible, чем видно в raw coordinates.

Но PCA itself стоит \(O(N^2)\), поэтому это quality/compressibility result, а не speed result.

Files:

- [src/v102_pca_sparse.py](src/v102_pca_sparse.py)
- [src/v102_pca_common40.py](src/v102_pca_common40.py)
- [results/qcno_v102_pca_common40.csv](results/qcno_v102_pca_common40.csv)

---

# 14. Fast learned basis search

## Hadamard

Signed/permuted FHT, \(O(N\log N)\).

При \(k=128\):

\[
PPL\approx16.44.
\]

## Learned butterfly

9-stage butterfly:

\[
PPL\approx12.05\quad(k=128).
\]

## Householder-24

24 reflections, примерно \(O(24N)\):

| k | PPL |
|---:|---:|
| 256 | 8.2007 |
| 128 | **9.3596** |
| 96 | 10.2591 |
| 64 | 11.9386 |
| 32 | 16.3509 |

Householder значительно сильнее Hadamard/butterfly, но всё ещё не достигает PCA k=128 PPL 8.1083.

Files:

- [src/v103_hadamard_sparse.py](src/v103_hadamard_sparse.py)
- [src/v104_butterfly_basis.py](src/v104_butterfly_basis.py)
- [src/v105_householder_train.py](src/v105_householder_train.py)
- [src/v105_householder_eval.py](src/v105_householder_eval.py)
- [results/qcno_v105_householder_common40.csv](results/qcno_v105_householder_common40.csv)

---

# 15. Shared PCA и support churn

Один общий PCA basis:

| k | PPL | Accuracy |
|---:|---:|---:|
| 256 | **8.0489** | 45.64% |
| 128 | **8.2854** | 45.08% |
| 96 | 8.5038 | 44.28% |
| 64 | 9.0182 | 42.14% |

Support retention при \(k=128\):

- 0→1: 51.4%;
- 1→2: **83.4%**;
- 2→3: 68.4%.

То есть на переходе 1→2 churn около 16.6%, что поддерживает идею event-driven delta updates. Но support не является глобально постоянным.

- [src/v106_shared_pca_churn.py](src/v106_shared_pca_churn.py)
- [results/qcno_v106_shared_pca_common40.csv](results/qcno_v106_shared_pca_common40.csv)
- [results/qcno_v106_support_churn.csv](results/qcno_v106_support_churn.csv)

---

# 16. Quantum-inspired ≠ quantum hardware

В репозитории есть три уровня:

### Classical structured
FFT, diagonal phases, circulant operators, shared dictionaries.

### Qubit-inspired
Normalized amplitudes, phase alphabets, sparse states, bit-packed sign states, reversible-like local transitions.

### Potential quantum
QFT-compatible structures, phase oracles, block encoding / singular-value transforms.

Никакой CPU benchmark в этом репозитории **не является доказательством hardware quantum advantage**.

Настоящее quantum comparison обязано включать:

- state preparation;
- oracle construction;
- precision;
- success probability;
- error correction;
- logical → physical qubits;
- routing;
- measurement/readout.

См. [docs/qcno_quantum_target_spec.md](docs/qcno_quantum_target_spec.md).

---

# 17. Нелинейность — отдельная проблема

Closed quantum evolution линейна. Нельзя просто вставить классический ReLU/GELU над неизвестными amplitudes.

Будущая quantum-compatible версия должна использовать отдельный механизм нелинейности/state update:

- bounded polynomial transforms;
- singular-value transforms;
- measurement/re-preparation;
- parameterized circuits;
- data re-uploading;
- либо принципиально state-machine dynamics.

Поэтому проект разделяет fast structured linear core и nonlinear mechanism.

---

# 18. Что проект опроверг

1. Один FFT operator недостаточно выразителен для arbitrary Transformer weights.
2. Post-hoc compression слабее native structured training.
3. Frobenius-optimal weight approximation не гарантирует лучший LM loss.
4. Low-rank residual сам по себе не решает quality gap.
5. Butterfly не является универсальным ответом.
6. Low-bond MPO post-hoc также недостаточен.
7. Whole-vector dedup слишком быстро разрушает quality.
8. 6000× sparse-kernel timing нельзя автоматически переносить на Transformer.
9. High active fraction + fanout создаёт fill-in.
10. Hadamard слишком слаб как universal learned sparse basis.

---

# 19. Текущий главный разрыв

Отдельно подтверждены:

### Speed side
- V1: 44.32× exact structured operator;
- V7 Q-FFT: 140×;
- V7 dictionary: 199.02×;
- highly sparse V10 kernels: 1000×+ в специальном режиме.

### Quality side
- best adaptive Q-CNO: 8× compression, PPL 8.0046;
- teacher: PPL 6.3223;
- PCA state k=128: PPL 8.1083;
- shared PCA k=128: PPL 8.2854.

Но одна и та же архитектура пока не доказала одновременно:

\[
PPL_{student}/PPL_{teacher}\le1.05,
\]

\[
compression\ge8\times,
\]

\[
end\text{-}to\text{-}end\ wall\ clock\ge100\times.
\]

1000× остаётся отдельным будущим milestone.

---

# 20. V11 — наиболее обоснованный следующий шаг

\[
\boxed{\text{Fast Learned Sparse Basis + Support-Preserving Complex Gates + Improved Q-CNO}}
\]

## Fast learned sparse basis

Цель — приблизить PCA k=128 quality при subquadratic encode/decode.

Кандидаты:

- deeper / block Householder;
- richer butterfly stacks;
- structured sparse autoencoder;
- product quantization + local rotations;
- joint basis/operator training.

## Support-preserving gates

Не:

\[
k\to Rk\to dense,
\]

а:

\[
k\to k.
\]

Например local complex 2-state rotations:

\[
\begin{pmatrix}
\alpha_i'\\
\alpha_j'
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & e^{i\phi}\sin\theta\\
-e^{-i\phi}\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
\alpha_i\\
\alpha_j
\end{pmatrix}.
\]

Это сохраняет support и близко к gate-like/qubit-like dynamics.

## Joint teacher distillation

Учить одновременно:

- operator basis;
- sparse basis;
- routers;
- local gates;
- hidden-state targets;
- teacher logits.

## Acceptance gates

Новый сильный milestone закрывается только если **одна модель** даёт:

\[
PPL_{student}/PPL_{teacher}\le1.05,
\]

\[
linear/storage\ compression\ge8\times,
\]

\[
end\text{-}to\text{-}end\ wall\ clock\ge100\times,
\]

и fast path не materialize скрытую \(N^2\) matrix.

---

# 21. Репозиторная структура

~~~text
.
├── README.md
├── MANIFEST.csv
├── requirements.txt
├── CHECKPOINTS.md
├── ARTIFACTS.md
├── CONTRIBUTING.md
├── src/             # экспериментальные исходники
├── docs/            # полные отчёты и спецификации
├── results/         # raw CSV/JSON evidence
└── figures_svg/     # читаемые GitHub SVG-графики
~~~

---

# 22. Воспроизводимость

Минимальные зависимости:

- Python 3.11+;
- NumPy;
- SciPy;
- pandas;
- PyTorch;
- matplotlib;
- numba;
- threadpoolctl.

Установка:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

V1:

~~~bash
python src/qcno_exact_operator_benchmark.py
~~~

V2:

~~~bash
python src/qcno_v2_train.py
python src/qcno_v2_distill.py
python src/qcno_v2_native_train.py
~~~

Некоторые поздние scripts требуют checkpoint предыдущей стадии.

Raw CSV/JSON считаются первичными evidence files. README округляет числа только для читаемости.

---

# 23. Baseline discipline

Speed claims должны разделять:

1. dense materialization;
2. optimized BLAS;
3. sparse-aware CSR/CSC;
4. exact same-operator comparison;
5. end-to-end model latency.

Operation-count reduction **не равен** wall-clock speedup.

Если утверждается matrix-free inference, fast path не должен скрыто материализовывать \(N\times N\).

---

# 24. Метрики качества

Поздние fixed common evaluations используют одинаковый byte-level LM protocol.

Byte-level PPL подходит для **внутреннего teacher/student comparison**, но её нельзя напрямую сравнивать с GPT-2/Llama BPE/SentencePiece PPL.

---

# 25. Почему complex numbers полезны, но не магические

Complex representation полезно для:

- Fourier diagonalization;
- phase rotations;
- interference-like state coding;
- compact phase alphabets;
- unitary-like transforms;
- потенциальной QFT-compatible реализации.

Но complex64 arithmetic само по себе дороже real arithmetic. Выигрыш возникает только когда структура уменьшает число операций, memory traffic или повторное вычисление.

---

# 26. Дедупликация: compute states once

Главная идея V7–V9:

> Если веса разделяют маленький dictionary prototype states, response каждого unique state нужно считать один раз, а затем дешёво переиспользовать через IDs/scales.

Для Q/K/V особенно важно, что input shared:

\[
z=\Phi(X)
\]

можно вычислить один раз, после чего:

\[
Q=R_Q(z),\quad K=R_K(z),\quad V=R_V(z).
\]

Более радикальная версия вообще не materialize Q/K/V, а обновляет persistent state напрямую.

---

# 27. Ускорить matmul vs убрать matmul

Первый этап проекта ускорял structured matmul.

Поздний этап исследует более сильную цель:

\[
\boxed{\text{перестать выполнять dense matrix multiplication вообще}}
\]

через схему:

~~~text
encode → sparse state → structured/local transition → sparse state → decode
~~~

вместо:

~~~text
GEMM → activation → GEMM → attention → GEMM → ...
~~~

Это пока research target, не закрытый end-to-end result.

---

# 28. Классификация доказательности

### CLOSED / exact
Same-operator parity, measured numerical error, mathematically identical baseline.

### EMPIRICAL
Checkpoint реально обучен; PPL/accuracy/runtime измерены.

### PROXY
Classical implementation моделирует будущую quantum/neuromorphic structure.

### HYPOTHESIS
V9 shared QKV dictionary, V11 support-preserving gates, matrix-free Transformer, fault-tolerant quantum implementation.

---

# 29. Binary checkpoints и bundles

Локальный исследовательский архив содержит около 98 MB binary checkpoints/ZIP/PNG. Доступный GitHub connector не является Git LFS/release-assets uploader, поэтому основной commit публикует **все доступные source files, raw metrics, reports, manifest и SVG evidence**, а бинарные артефакты перечислены по именам и SHA-256 в [CHECKPOINTS.md](CHECKPOINTS.md), [ARTIFACTS.md](ARTIFACTS.md) и [MANIFEST.csv](MANIFEST.csv).

Ключевой best checkpoint локального архива: qcno_v31_adaptive_avg32_best_final.pt.

---

# 30. Главное резюме

Самый сильный подтверждённый quality point:

\[
\boxed{8\times\ linear\ compression,\ PPL=8.0046}
\]

против teacher:

\[
\boxed{PPL=6.3223}.
\]

Самый сильный exact compatible matrix-kernel point:

\[
\boxed{199.02\times}
\]

против dense materialization.

Самая сильная state-compressibility находка:

\[
\boxed{k=128/512\ в\ PCA\ basis,\ PPL=8.1083}.
\]

Самый важный falsification:

\[
\boxed{\text{1000× end-to-end Transformer speedup пока не доказан}}.
\]

Следовательно V11 должен закрыть разрыв между **PCA-level compressibility** и **fast support-preserving operator**, одновременно улучшая expressivity самого Q-CNO.

---

Repository: **Shtenco/synergy_complex_matrix_multiplication**  
Research status: **ACTIVE / EXPERIMENTAL**


---

# 31. Снапшот публикации в GitHub

На текущем подтверждённом снапшоте ветки `main` опубликовано **104 файла**:

| Раздел | Количество | Назначение |
|---|---:|---|
| корень репозитория | 7 | README, requirements, manifest, inventories, contributing |
| `src/` | **33** | полный набор уникальных текстовых экспериментальных исходников локального архива |
| `docs/` | **12** | version reports, quantum target spec, verified reports |
| `results/` | **46** | все уникальные raw CSV/JSON/TXT evidence files |
| `figures_svg/` | **6** | GitHub-readable visual summaries |

Локально исходно было 56 result-файлов. Десять из них были UI/display-копиями уже существующих CSV с альтернативными человекочитаемыми именами. Они намеренно **не продублированы** в GitHub. Все уникальные числовые результаты опубликованы.

Все **50 относительных ссылок** этого README автоматически сверены с recursive Git tree; на момент публикации:

[
oxed{	ext{broken README links}=0}
]

---

# 32. Канонический evidence index

Если нужно перепроверить конкретное утверждение, первичный источник следует выбирать так:

| Утверждение | Канонический evidence |
|---|---|
| V1 exact same-operator speed / numerical parity | `results/qcno_exact_operator_results.csv` |
| ранний complex structured proxy | `results/qcno_benchmark_results.csv` |
| idealized quantum logical scaling | `results/qcno_quantum_target_scaling.csv` |
| V2 post-hoc rank sweep | `results/qcno_v2_distillation_results.csv` |
| V2 native training | `results/qcno_v2_native_summary.csv` |
| V2 large-N scaling | `results/qcno_v2_scaling_results.csv` |
| V2 end-to-end dense-vs-implicit exactness | `results/qcno_v2_e2e_runtime.json` |
| selective attention/MLP compression | `results/qcno_v2_selective_results.csv` |
| V3 adaptive ranks | `results/qcno_v31_rank_allocations.json` |
| V3 common evaluation | `results/qcno_v31_common_eval.json` |
| V3/V6 global frontier | `results/qcno_v3_v6_frontier.csv` |
| V4 speed/quality frontier | `results/qcno_v4_frontier_summary.csv` |
| Householder speed | `results/qcno_v42_householder_speed_results.csv` |
| V5 Monarch pre-kernel-optimization speed | `results/qcno_v53_speed_results.csv` |
| V5 grouped-BMM speed | `results/qcno_v53_speed_bmm_results.csv` |
| V5 token-width speed | `results/qcno_v53_tokenwidth_speed_results.csv` |
| sensitivity allocation | `results/qcno_v56_sensitivity_alloc.json` |
| MPO raw operator errors | `results/qcno_v6_mpo_operator_results.csv` |
| V7 100× / 199× kernels | `results/qcno_v7_100x_matrix_benchmark.csv` |
| V8 real checkpoint dedup | `results/qcno_v8_real_checkpoint_dedup.csv` |
| V9 PQ initialization/recovery | `results/qcno_v9_fast_target_pre.csv`, `qcno_v9_G32K48_curve_part.csv` |
| V10 early exact sparse kernel | `results/qcno_v10_state_kernel_benchmark.csv` |
| V10 stress test | `results/qcno_v101_stress_benchmark.csv` |
| constant 75% active-state regime | `results/qcno_v10_constant_fraction_speed.csv` |
| coordinate hidden sparsity | `results/qcno_v101_common_eval_sparsity.csv` |
| PCA state compression | `results/qcno_v102_pca_common40.csv` |
| Hadamard | `results/qcno_v103_fast_basis_common40.csv` |
| Householder-24 state basis | `results/qcno_v105_householder_common40.csv` |
| shared PCA | `results/qcno_v106_shared_pca_common40.csv` |
| support churn | `results/qcno_v106_support_churn.csv` |
| итоговый verified state frontier | `results/qcno_v10_verified_frontier.csv` |

---

# 33. Какие числа можно цитировать корректно

### Можно цитировать

**44.32×**  
как measured CPU speedup exact same structured Q-CNO operator при (N=4096), 1 layer, из V1.

**140.00×**  
как measured speedup exact Q-FFT-compatible operator против его dense materialization при (N=12288,T=4).

**199.02×**  
как measured speedup exact normalized/deduplicated bit-amplitude dictionary operator при (N=12288,T=8,K=64).

**8× compression + PPL 8.0046**  
как лучший подтверждённый adaptive Q-CNO quality point на common evaluation.

**PCA k=128/512, PPL 8.1083**  
как доказательство высокой информационной compressibility hidden state.

**335.98× vs dense / 1.30× vs CSR**  
как stress-tested exact sparse state kernel для фиксированного (k=384,R=4,N=8192).

### Нельзя цитировать без квалификатора

**199× Transformer speedup** — не доказан.

**6445× Transformer speedup** — не доказан. Это ранний exact highly-sparse kernel point при (k=8).

**1000× end-to-end** — не доказан.

**quantum advantage** — не доказан.

**PPL parity with dense teacher** — не достигнута.

---

# 34. Binary reproducibility boundary

Текстовая часть исследования опубликована максимально полно в рамках текущего GitHub connector:

- 33/33 source files;
- 12/12 reports/specifications;
- 46/46 unique result files;
- visual summaries;
- full local-archive manifest.

Не опубликованы непосредственно только тяжёлые бинарные объекты:

- PyTorch `.pt` checkpoints;
- ZIP bundle snapshots;
- исходные PNG, поскольку эквивалентные SVG опубликованы для GitHub-навигации.

Причина техническая: доступный connector работает через GitHub Contents/Git Data API для текстовых payloads и не является Git LFS client или release-asset uploader.

При этом бинарные объекты **не потеряны как provenance**: `MANIFEST.csv` содержит их:

- точное имя;
- размер в bytes;
- SHA-256.

Особенно важный checkpoint:

`qcno_v31_adaptive_avg32_best_final.pt`

локальный SHA-256:

`049214001f67223dc878dc279bd06b7e38af51800a91d19ead7dd879d9c048ec`

Teacher:

`qcno_v3_teacher_d512.pt`

SHA-256:

`1fb70b4806fe19d90f998b6b500b0ea16cf134cf1a8ce14e56853f297b90fe06`

---

# 35. Research acceptance policy

Чтобы будущая версия получила статус сильного результата, она должна проходить не одну, а всю цепочку:

[
	ext{mathematical identity / approximation}
]

[
downarrow
]

[
	ext{numerical error}
]

[
downarrow
]

[
	ext{kernel benchmark}
]

[
downarrow
]

[
	ext{sparse-aware baseline}
]

[
downarrow
]

[
	ext{end-to-end Transformer quality}
]

[
downarrow
]

[
	ext{end-to-end wall-clock}
]

[
downarrow
]

[
	ext{memory / storage / energy accounting}
]

Только после этого допустимо повышать статус claim.

Текущий проект уже закрыл несколько отдельных звеньев этой цепочки, но **не всю цепочку одновременно**. Именно это является главным открытым вопросом V11+.

