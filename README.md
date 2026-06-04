# ternary-cortex — Hierarchical processing layers for ternary intelligence

Cortex struct, layers, columns, thalamic relay, corpus callosum bridge, and topographic maps for building multi-tier ternary processing pipelines. Inspired by biological cortical neuroarchitecture.

## Why This Exists

Ternary agents (operating on {-1, 0, +1} values) need structured processing hierarchies — not just flat arrays. Biological brains organize computation into layers, columns, and relay stations. This crate provides those same abstractions for ternary systems, enabling multi-stage signal processing with gating, attention, and inter-hemisphere communication.

## Core Concepts

- **Balanced ternary** — A number system using three values: -1 (Neg), 0 (Zero), +1 (Pos). Multiplication follows sign rules: Pos × Neg = Neg, Neg × Neg = Pos, anything × Zero = Zero.
- **Cortex** — The full hierarchical processor: multiple layers with relay and column support.
- **CortexLayer** — One processing tier. Applies ternary weights to input and gates output by threshold.
- **CorticalColumn** — A vertical slice through all layers. Activations propagate upward from input to output.
- **Thalamus** — A relay station between layers with per-channel gating and attention modulation. Closed gates block signal; negative attention scores auto-close channels.
- **CorpusCallosum** — A bridge between two hemispheres. Each fiber has a ternary weight; severed fibers (Zero weight) block transmission.
- **CorticalMap** — A 2D topographic map for spatial organization of ternary values, with neighbor queries and center-of-mass computation.

## Quick Start

```toml
[dependencies]
ternary-cortex = "0.1"
```

```rust
use ternary_cortex::*;

// Build a 3-layer cortex: layer widths and thresholds
let mut cortex = Cortex::new(&[(8, 2), (6, 1), (4, 1)]);

// Process a ternary input through all layers
let input = vec![Ternary::Pos, Ternary::Neg, Ternary::Zero, Ternary::Pos,
                 Ternary::Pos, Ternary::Neg, Ternary::Pos, Ternary::Zero];
let output = cortex.process(&input);

// Use thalamic relay for gated processing
let output = cortex.process_with_relay(&input);
```

## API Overview

| Type | Purpose |
|------|---------|
| `Ternary` | Core ternary value: Neg (-1), Zero (0), Pos (+1) |
| `Cortex` | Full multi-layer hierarchical processor |
| `CortexLayer` | Single processing tier with weights and threshold |
| `CorticalColumn` | Vertical slice propagating activation upward |
| `Thalamus` | Relay with per-channel gates and attention modulation |
| `CorpusCallosum` | Bridge between two hemispheres with weighted fibers |
| `CorticalMap` | 2D topographic map with spatial queries |

## How It Works

Feed-forward processing: input enters the bottom layer, each layer multiplies by its ternary weights, then applies a threshold gate. If too few positive values survive, the layer outputs all zeros — acting as an activity filter.

The thalamic relay sits between layers and provides attention-based gating. Channels with sustained negative attention close automatically, filtering out uninformative signal paths.

The corpus callosum enables bidirectional communication between hemispheres. Each fiber applies ternary multiplication, so severed (Zero-weight) fibers block cross-talk while Pos-weight fibers pass signal and Neg-weight fibers invert it.

## Known Limitations

- **No backpropagation** — Weight adaptation is heuristic (sign-based), not gradient-based. Learning is limited to simple correlation.
- **Fixed topology** — Layer sizes are set at construction; no dynamic layer creation or pruning.
- **No recurrent connections** — Processing is strictly feed-forward. Recurrent loops require external orchestration.
- **Threshold is global per layer** — No per-column or per-position thresholding.
- **Attention modulation is cumulative** — Attention scores grow without bound in one direction; only the gate threshold (-3) prevents runaway.

## Use Cases

1. **Multi-stage ternary classifier** — Route ternary signals through increasingly selective layers, with threshold gating filtering noise at each stage.
2. **Bilateral processing** — Run two cortexes in parallel (left/right hemispheres) with corpus callosum bridging, for contrastive analysis of ternary inputs.
3. **Spatial ternary attention** — Use CorticalMap to track where positive activations cluster, computing center-of-mass for attention targeting.

## Ecosystem Context

Part of the SuperInstance ternary crate family. `ternary-cortex` provides the processing hierarchy that `ternary-agent` and `ternary-cell` can use for structured reasoning. It maps to the cortical layers described in FLEET-NEUROARCHITECTURE.md. Related to `ternary-attention` for attention mechanisms and `ternary-network` for networked processing.

## License

MIT
