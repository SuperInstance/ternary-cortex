# Ternary Cortex — Hierarchical Processing Layers for Ternary Intelligence

**Ternary Cortex** implements multi-tier ternary processing inspired by biological cortical architecture. It provides `CortexLayer` units that perform ternary-weighted transformations, threshold gating for attention-like filtering, and Hebbian-style weight adaptation — all using {-1, 0, +1} arithmetic. Layers stack into hierarchical pipelines where lower tiers extract features and higher tiers integrate decisions.

## Why It Matters

Deep neural networks achieve intelligence through hierarchical feature extraction, but traditional networks use full-precision weights that are expensive to store and compute. Ternary cortex layers use {-1, 0, +1} weights, reducing memory by 16× and replacing multiply-accumulate (MAC) operations with conditional add/subtract/skip — operations that are 3-5× faster on commodity hardware. The cortical architecture adds biological plausibility: threshold gating mimics how real cortical columns suppress weak signals, and the learning rate controls Hebbian adaptation. This crate provides the building blocks for constructing multi-layer ternary intelligence systems without GPU dependencies.

## How It Works

### CortexLayer

Each layer maintains a vector of ternary weights initialized to Zero (the neutral state). The `process()` method computes element-wise ternary multiplication between input and weights:

```
output[i] = input[i] × weight[i]   (ternary multiplication in Z₃)
```

Ternary multiplication follows the rules: (+1·+1)=+1, (-1·-1)=+1, (+1·-1)=-1, and anything × 0 = 0. This is equivalent to sign multiplication with a kill state. Processing is O(n) for n weights.

### Threshold Gating

After processing, the `threshold_gate()` suppresses all output unless enough positive values are present. If the count of positive outputs ≥ threshold, the full output vector passes through; otherwise, all values are zeroed. This implements a **k-of-n gate** — a form of attention that prevents weak signals from propagating up the hierarchy.

### Hebbian Adaptation

Weights adapt based on input and error signals using a ternary Hebbian rule:

```
(Pos, Pos) → Pos    (reinforce positive correlation)
(Neg, Pos) → Pos    (flip to match error direction)  
(Pos, Neg) → Neg    (flip to match error direction)
```

The learning rate modulates which weights update on each step. This is a discretized version of the delta rule: `Δw = η · error · input`, where the result is quantized to {-1, 0, +1}.

## Quick Start

```rust
use ternary_cortex::{Ternary, CortexLayer};

let mut layer = CortexLayer::new(0, 4, 2); // id=0, width=4, threshold=2
layer.weights = vec![Ternary::Pos, Ternary::Neg, Ternary::Zero, Ternary::Pos];

let input = vec![Ternary::Pos, Ternary::Pos, Ternary::Pos, Ternary::Pos];
let processed = layer.process(&input);
// [Pos, Neg, Zero, Pos]

let gated = layer.threshold_gate(&processed);
// Pos count = 2 >= threshold 2, so output passes through
```

```bash
cargo add ternary-cortex
```

## API

| Type / Function | Description |
|---|---|
| `Ternary` | Enum: `Neg(-1)`, `Zero(0)`, `Pos(1)` |
| `CortexLayer` | `{ id, width, weights, threshold, learning_rate }` |
| `CortexLayer::process(&[Ternary])` | Element-wise ternary multiply (O(n)) |
| `CortexLayer::threshold_gate(&[Ternary])` | k-of-n attention gate |
| `CortexLayer::adapt(&[Ternary], &[Ternary])` | Hebbian weight update |

## Architecture Notes

The cortex is the reasoning substrate in **SuperInstance**. Stacked cortical layers form the processing pipeline that transforms raw ternary inputs (sensor data, agent votes) into fleet-wide decisions. The γ + η = C conservation law manifests at each layer: the threshold gate ensures that only sufficiently strong growth signals (γ) propagate, while weak signals return to entropy (η). See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Hubel, David & Wiesel, Torsten. "Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex," *J. Physiology*, 160(1), 1962 — cortical columns.
- Hebb, Donald O. *The Organization of Behavior*, Wiley, 1949 — Hebbian learning.
- Li, Feng et al. "Ternary Weight Networks," *arXiv:1605.04711*, 2016 — ternary quantization.

## License

MIT
