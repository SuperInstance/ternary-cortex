# Ternary Cortex — Hierarchical Processing Layers for Ternary Intelligence

**Ternary Cortex** implements multi-tier processing structures inspired by biological cortical architecture — layers, columns, relay mechanisms, and topographic maps — all operating on the ternary value space **T = {−1, 0, +1}**. It provides the computational substrate for building hierarchical ternary intelligence pipelines where lower tiers extract features and higher tiers integrate decisions, using only conditional add/subtract/skip operations instead of full-precision multiply-accumulate (MAC).

## Why It Matters

Modern deep neural networks achieve intelligence through hierarchical feature extraction, but they rely on full-precision (FP32/FP16) weights that demand significant memory bandwidth and energy. Ternary Weight Networks (TWNs) quantize weights to {−1, 0, +1}, reducing model size by **16×** and replacing every MAC with a conditional add, subtract, or no-op — operations that are **3–5× cheaper** on commodity hardware and **10× cheaper** on custom ternary ALUs.

The cortex abstraction adds biological plausibility to this mathematical efficiency. Real cortical columns use **threshold gating** to suppress weak signals — only neurons receiving sufficient excitatory input fire. The Hebbian learning rule ("neurons that fire together, wire together") adapts weights based on local correlations, requiring no gradient computation or backpropagation. This makes ternary cortex layers ideal for **edge inference**, **neuromorphic hardware**, and **online learning** scenarios where power and compute budgets are tight.

## How It Works

### Ternary Multiplication (Sign Algebra)

Each `CortexLayer` stores a vector of ternary weights **w ∈ Tⁿ**. The `process()` method computes element-wise ternary multiplication between input **x ∈ Tⁿ** and weights:

```
y[i] = x[i] ⊙ w[i]
```

where ⊙ denotes ternary multiplication in the balanced ternary system:

| x | w | x ⊙ w |
|---|---|-------|
| +1 | +1 | +1 |
| +1 | −1 | −1 |
| −1 | +1 | −1 |
| −1 | −1 | +1 |
| 0  | * | 0 |
| *  | 0 | 0 |

This is equivalent to sign multiplication with an explicit kill state (multiply by zero). The operation is a **group homomorphism** from (T, ⊙) to the sign group {+1, −1} with zero as an absorbing element.

**Complexity:** O(n) per layer for n weights, with a constant factor of 1 comparison + 1 conditional add/subtract per element — no hardware multiplier required.

### Threshold Gating (k-of-n Attention Gate)

After processing, `threshold_gate()` implements a **k-of-n gate** — a binary decision that suppresses all output unless at least k positive values are present:

```
gate(y) = y     if |{i : y[i] = +1}| ≥ k
        = 0ⁿ    otherwise
```

This models the **firing threshold** of biological cortical columns: a neuron fires only when excitatory inputs exceed inhibitory inputs by a margin. Mathematically, it computes:

```
∑ᵢ max(0, y[i]) ≥ k
```

The gate is **not differentiable** (it is a step function), but in the ternary regime this is irrelevant — there are no gradients to compute. The discrete nature of ternary values means the threshold gate is exact, not an approximation.

**Complexity:** O(n) — single pass to count positives, one comparison.

### Hebbian Weight Adaptation

The `adapt()` method updates weights based on input **x** and error signal **e** using a ternary Hebbian rule:

```
if e[i] ≠ 0:
    w[i] ← sign(x[i] × e[i])    (quantized to T)
```

Specifically:

| x[i] | e[i] | New w[i] |
|------|------|----------|
| +1 | +1 | +1 |
| −1 | +1 | −1 |
| +1 | −1 | −1 |
| −1 | −1 | +1 |
| 0  | *  | unchanged |
| *  | 0  | unchanged |

This is a **discretized delta rule**: Δw = η · e · x, where the result is quantized to T. The learning rate η controls which weights update via a modular scheduling scheme.

**Complexity:** O(min(width, |input|, |error|)) per adaptation step.

### Cortical Columns

A `CorticalColumn` is a vertical slice through all layers — analogous to a **minicolumn** in biological cortex (~100 neurons deep). It maintains a shift register of activations across depth d:

```
activate(x): if x ≠ 0, shift all activations down by 1, insert x at
             position 0, mark column active; if x = 0, mark column
             inactive (no shift)
output(): return activation at depth d-1
```

**Complexity:** O(d) per activation (shift register update).

### Thalamic Gating

The `Thalamus` implements **thalamic relay gating** — the brain's attention mechanism. Each channel has a gate (open/closed) and an attention score. Channels with attention score below −3 are closed; channels with score above 0 are opened:

```
gate[i] = open   if attention[i] > 0
gate[i] = closed if attention[i] < −3
```

This implements a **hysteresis threshold** — once a channel closes, it requires positive evidence to reopen, preventing rapid on/off oscillation.

**Complexity:** O(channels) per relay cycle.

### Corpus Callosum (Inter-Hemispheric Bridge)

The `CorpusCallosum` bridges two cortical hemispheres with weighted fiber bundles. Transfer is bidirectional with ternary-weighted mixing:

```
transfer_L→R[i] = left_buffer[i] ⊙ weight[i]
```

Individual fibers can be severed (`sever_fiber()`), modeling **split-brain** scenarios. Active fiber count tracks connectivity health.

### Topographic Maps

`CorticalMap` provides a 2D grid representation of cortical surface activity. The `center_of_mass()` computes the centroid of all positive activations:

```
x̄ = (1/c) ∑ᵢ xᵢ · 𝟙[grid[xᵢ, yᵢ] = +1]
ȳ = (1/c) ∑ᵢ yᵢ · 𝟙[grid[xᵢ, yᵢ] = +1]
```

where c is the count of positive cells. This is used for **population coding** — decoding a continuous-valued signal from the spatial distribution of ternary activations.

**Complexity:** O(width × height) for center-of-mass computation.

## Quick Start

```rust
use ternary_cortex::{Ternary, CortexLayer, Thalamus, CorticalMap};

// Build a 3-layer cortical pipeline
let mut l1 = CortexLayer::new(0, 8, 3);   // width=8, threshold=3
let mut l2 = CortexLayer::new(1, 8, 5);   // width=8, threshold=5
l1.weights = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos, Ternary::Zero,
                  Ternary::Pos, Ternary::Neg, Ternary::Zero, Ternary::Pos];

let input = vec![Ternary::Pos, Ternary::Pos, Ternary::Neg, Ternary::Pos,
                 Ternary::Zero, Ternary::Pos, Ternary::Neg, Ternary::Pos];

// Forward pass
let processed = l1.process(&input);       // element-wise ternary multiply
let gated = l1.threshold_gate(&processed); // k-of-n attention gate

// Adapt with error signal
let error = vec![Ternary::Pos, Ternary::Zero, Ternary::Neg, Ternary::Pos,
                 Ternary::Zero, Ternary::Zero, Ternary::Neg, Ternary::Pos];
l1.adapt(&input, &error);                 // Hebbian weight update

// Thalamic gating
let mut thalamus = Thalamus::new(8);
thalamus.receive(&gated);
thalamus.set_gate(3, false);              // close channel 3
let relayed = thalamus.relay();
```

```bash
cargo add ternary-cortex
```

## API

| Type / Method | Complexity | Description |
|---|---|---|
| `Ternary` | — | Enum: `Neg(−1)`, `Zero(0)`, `Pos(+1)` |
| `CortexLayer::process(&[Ternary])` | O(n) | Element-wise ternary multiply |
| `CortexLayer::threshold_gate(&[Ternary])` | O(n) | k-of-n attention gate |
| `CortexLayer::adapt(&[Ternary], &[Ternary])` | O(n) | Hebbian weight update |
| `CorticalColumn::activate(Ternary)` | O(d) | Shift-register forward pass |
| `Thalamus::receive(&[Ternary])` | O(channels) | Gated input buffer |
| `Thalamus::modulate_attention(&[i32])` | O(channels) | Attention-driven gate control |
| `CorpusCallosum::transfer_left_to_right()` | O(fibers) | Cross-hemisphere signal transfer |
| `CorticalMap::center_of_mass()` | O(w×h) | Population-coded centroid |

## Architecture Notes

The cortex is the reasoning substrate in the **SuperInstance** ecosystem. Stacked cortical layers form the processing pipeline that transforms raw ternary inputs (sensor data, agent votes, fleet telemetry) into hierarchical decisions. The **γ + η = C** conservation law manifests at each layer: the threshold gate ensures that only sufficiently strong growth signals (γ) propagate upward, while weak signals return to entropy (η). The total information content C is conserved across layers — what doesn't propagate as structured signal dissipates as noise.

The thalamic gating mechanism controls the **information flow rate** between layers, analogous to how the biological thalamus regulates which sensory inputs reach conscious awareness. The corpus callosum enables **lateral integration** — left-hemisphere (analytical) and right-hemisphere (holistic) processing can share intermediate results.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for system-level design.

## References

1. Hubel, D. H. & Wiesel, T. N. (1962). "Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex." *Journal of Physiology*, 160(1), 106–154. — Cortical column architecture.
2. Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley. — Hebbian learning postulate.
3. Li, F., Zhang, B., & Liu, B. (2016). "Ternary Weight Networks." *arXiv:1605.04711*. — Ternary quantization for neural networks.
4. Douglas, R. J. & Martin, K. A. C. (2004). "Neuronal Circuits of the Neocortex." *Annual Review of Neuroscience*, 27, 419–451. — Canonical cortical microcircuit.
5. Sherman, S. M. & Guillery, R. W. (2002). "The Role of the Thalamus in the Flow of Information to the Cortex." *Philosophical Transactions of the Royal Society B*, 357(1428), 1695–1708. — Thalamic gating.
6. Rall, W. (1962). "Electrophysiology of a Dendritic Neuron Model." *Biophysical Journal*, 2(2), 145–167. — Cable theory and threshold firing.

## License

MIT
