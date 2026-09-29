# Q-CNO V1 — quantum-compatible target specification

## Coherent linear layer

For N = 2^n amplitude basis states:

U(theta_x, theta_k) = D_x QFT^\dagger D_k QFT

where each diagonal phase oracle is generated from a compact quadratic Boolean
phase polynomial over index bits b_i:

theta(b) = c0 + sum_i a_i b_i + sum_{i<j} a_ij b_i b_j.

This uses O(n^2)=O(log^2 N) trainable phase parameters and can be compiled from
single-qubit phase and controlled-phase gates without storing N independent phases.

## Complexity target

Ignoring fault-tolerance overhead, state preparation, success amplification and
measurement:

- QFT + inverse QFT: O(log^2 N) exact logical gates.
- Two quadratic phase oracles: O(log^2 N) logical phase gates.
- Parameter storage: O(log^2 N).
- No N x N weight matrix is materialized.

## Nonlinearity warning

A closed quantum evolution is linear in the state. Elementwise ReLU/GELU on
unknown amplitudes cannot simply be inserted as a unitary activation.

A quantum-native replacement must instead use one of:
1. QSVT/QSP polynomial transforms of a block-encoded operator's singular values;
2. ancilla + measurement + re-preparation;
3. data re-uploading / parameterized quantum circuit layers;
4. LCU/block-encoding constructions for bounded polynomial operator transforms.

Therefore the first falsifiable milestone is the coherent structured linear core,
followed by a separately costed quantum-native activation mechanism.

## What would count as genuine advantage

A quantum speedup claim is only meaningful if:
- input state preparation/oracle access is efficient;
- phase functions are compactly computable;
- required precision epsilon is included;
- success probability/amplification is included;
- output is a small set of observables/samples/decisions, not all N amplitudes;
- fault-tolerant logical-to-physical overhead is reported.
