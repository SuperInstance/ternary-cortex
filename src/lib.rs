#![forbid(unsafe_code)]

//! Hierarchical processing layers for ternary intelligence.
//!
//! Provides cortical structures — layers, columns, relay mechanisms — for
//! building multi-tier ternary processing pipelines inspired by biological
//! neuroarchitecture.

/// Ternary value: -1, 0, or +1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }
}

/// A single processing tier in the cortex.
#[derive(Clone, Debug)]
pub struct CortexLayer {
    pub id: usize,
    pub width: usize,
    pub weights: Vec<Ternary>,
    pub threshold: usize,
    pub learning_rate: u8,
}

impl CortexLayer {
    pub fn new(id: usize, width: usize, threshold: usize) -> Self {
        Self {
            id,
            width,
            weights: vec![Ternary::Zero; width],
            threshold,
            learning_rate: 10,
        }
    }

    /// Ternary multiplication of input by weights.
    pub fn process(&self, input: &[Ternary]) -> Vec<Ternary> {
        let limit = self.width.min(input.len());
        input
            .iter()
            .zip(&self.weights)
            .take(limit)
            .map(|(&inp, &w)| ternary_mul(inp, w))
            .collect()
    }

    /// Threshold gate: suppress all output if not enough positives.
    pub fn threshold_gate(&self, values: &[Ternary]) -> Vec<Ternary> {
        let pos_count = values.iter().filter(|&&v| v == Ternary::Pos).count();
        if pos_count >= self.threshold {
            values.to_vec()
        } else {
            vec![Ternary::Zero; values.len()]
        }
    }

    /// Adapt weights based on input and error signal.
    pub fn adapt(&mut self, input: &[Ternary], error: &[Ternary]) {
        for i in 0..self.width.min(input.len()).min(error.len()) {
            if error[i] != Ternary::Zero && (self.learning_rate as usize) > i % 10 {
                self.weights[i] = match (input[i], error[i]) {
                    (Ternary::Pos, Ternary::Pos) => Ternary::Pos,
                    (Ternary::Neg, Ternary::Pos) => Ternary::Neg,
                    (Ternary::Pos, Ternary::Neg) => Ternary::Neg,
                    (Ternary::Neg, Ternary::Neg) => Ternary::Pos,
                    _ => self.weights[i],
                };
            }
        }
    }

    pub fn reset(&mut self) {
        self.weights = vec![Ternary::Zero; self.width];
    }
}

/// A vertical slice through all layers.
#[derive(Clone, Debug)]
pub struct CorticalColumn {
    pub index: usize,
    pub activations: Vec<Ternary>,
    pub active: bool,
}

impl CorticalColumn {
    pub fn new(index: usize, depth: usize) -> Self {
        Self {
            index,
            activations: vec![Ternary::Zero; depth],
            active: false,
        }
    }

    /// Feed activation upward through layers.
    pub fn activate(&mut self, input: Ternary) {
        if input != Ternary::Zero {
            self.active = true;
            for i in (1..self.activations.len()).rev() {
                self.activations[i] = self.activations[i - 1];
            }
            self.activations[0] = input;
        } else {
            self.active = false;
        }
    }

    pub fn output(&self) -> Ternary {
        self.activations.last().copied().unwrap_or(Ternary::Zero)
    }

    pub fn reset(&mut self) {
        self.activations = vec![Ternary::Zero; self.activations.len()];
        self.active = false;
    }
}

/// Relay between layers (thalamic gating).
#[derive(Clone, Debug)]
pub struct Thalamus {
    buffer: Vec<Ternary>,
    gates: Vec<bool>,
    attention: Vec<i32>,
}

impl Thalamus {
    pub fn new(channels: usize) -> Self {
        Self {
            buffer: vec![Ternary::Zero; channels],
            gates: vec![true; channels],
            attention: vec![0; channels],
        }
    }

    pub fn receive(&mut self, input: &[Ternary]) {
        let limit = self.buffer.len().min(input.len());
        for (i, &inp) in input.iter().enumerate().take(limit) {
            if self.gates[i] {
                self.buffer[i] = inp;
            }
        }
    }

    pub fn relay(&self) -> &[Ternary] {
        &self.buffer
    }

    pub fn set_gate(&mut self, channel: usize, open: bool) {
        if channel < self.gates.len() {
            self.gates[channel] = open;
        }
    }

    pub fn modulate_attention(&mut self, signal: &[i32]) {
        let limit = self.attention.len().min(signal.len());
        for (i, &sig) in signal.iter().enumerate().take(limit) {
            self.attention[i] += sig;
            if self.attention[i] < -3 {
                self.gates[i] = false;
            } else if self.attention[i] > 0 {
                self.gates[i] = true;
            }
        }
    }

    pub fn open_channel_count(&self) -> usize {
        self.gates.iter().filter(|&&g| g).count()
    }

    pub fn reset(&mut self) {
        self.buffer = vec![Ternary::Zero; self.buffer.len()];
        self.gates = vec![true; self.gates.len()];
        self.attention = vec![0; self.attention.len()];
    }
}

/// Bridge between two cortical hemispheres.
#[derive(Clone, Debug)]
pub struct CorpusCallosum {
    pub fiber_count: usize,
    left_buffer: Vec<Ternary>,
    right_buffer: Vec<Ternary>,
    pub weights: Vec<Ternary>,
}

impl CorpusCallosum {
    pub fn new(fiber_count: usize) -> Self {
        Self {
            fiber_count,
            left_buffer: vec![Ternary::Zero; fiber_count],
            right_buffer: vec![Ternary::Zero; fiber_count],
            weights: vec![Ternary::Pos; fiber_count],
        }
    }

    pub fn receive_left(&mut self, signals: &[Ternary]) {
        let limit = self.left_buffer.len().min(signals.len());
        self.left_buffer[..limit].copy_from_slice(&signals[..limit]);
    }

    pub fn receive_right(&mut self, signals: &[Ternary]) {
        let limit = self.right_buffer.len().min(signals.len());
        self.right_buffer[..limit].copy_from_slice(&signals[..limit]);
    }

    pub fn transfer_left_to_right(&self) -> Vec<Ternary> {
        self.left_buffer
            .iter()
            .zip(&self.weights)
            .map(|(&s, &w)| ternary_mul(s, w))
            .collect()
    }

    pub fn transfer_right_to_left(&self) -> Vec<Ternary> {
        self.right_buffer
            .iter()
            .zip(&self.weights)
            .map(|(&s, &w)| ternary_mul(s, w))
            .collect()
    }

    pub fn sever_fiber(&mut self, index: usize) {
        if index < self.weights.len() {
            self.weights[index] = Ternary::Zero;
        }
    }

    pub fn active_fiber_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w != Ternary::Zero).count()
    }
}

/// Topographic map of the cortical surface.
#[derive(Clone, Debug)]
pub struct CorticalMap {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Ternary>,
}

impl CorticalMap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            grid: vec![Ternary::Zero; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Option<Ternary> {
        if x < self.width && y < self.height {
            Some(self.grid[y * self.width + x])
        } else {
            None
        }
    }

    pub fn set(&mut self, x: usize, y: usize, value: Ternary) -> bool {
        if x < self.width && y < self.height {
            self.grid[y * self.width + x] = value;
            true
        } else {
            false
        }
    }

    /// 4-connected neighbors of (x, y).
    pub fn neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize, Ternary)> {
        let mut result = Vec::new();
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 {
                let (ux, uy) = (nx as usize, ny as usize);
                if ux < self.width && uy < self.height {
                    result.push((ux, uy, self.grid[uy * self.width + ux]));
                }
            }
        }
        result
    }

    /// Center of mass of all positive activations.
    pub fn center_of_mass(&self) -> (f64, f64) {
        let mut sx = 0i64;
        let mut sy = 0i64;
        let mut c = 0u64;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.grid[y * self.width + x] == Ternary::Pos {
                    sx += x as i64;
                    sy += y as i64;
                    c += 1;
                }
            }
        }
        if c == 0 {
            (self.width as f64 / 2.0, self.height as f64 / 2.0)
        } else {
            (sx as f64 / c as f64, sy as f64 / c as f64)
        }
    }

    pub fn clear(&mut self) {
        self.grid = vec![Ternary::Zero; self.width * self.height];
    }
}

/// Full hierarchical cortex: layers, relay, columns.
#[derive(Clone, Debug)]
pub struct Cortex {
    pub layers: Vec<CortexLayer>,
    pub thalamus: Thalamus,
    pub columns: Vec<CorticalColumn>,
}

impl Cortex {
    /// Create cortex with layer specs: (width, threshold) per layer.
    pub fn new(layer_sizes: &[(usize, usize)]) -> Self {
        let layers: Vec<CortexLayer> = layer_sizes
            .iter()
            .enumerate()
            .map(|(i, &(w, t))| CortexLayer::new(i, w, t))
            .collect();

        let max_w = layer_sizes.iter().map(|&(w, _)| w).max().unwrap_or(0);
        let depth = layers.len();

        let columns: Vec<CorticalColumn> =
            (0..max_w).map(|i| CorticalColumn::new(i, depth)).collect();

        Self {
            layers,
            thalamus: Thalamus::new(max_w),
            columns,
        }
    }

    /// Feed-forward through all layers.
    pub fn process(&mut self, input: &[Ternary]) -> Vec<Ternary> {
        let mut signal = input.to_vec();
        for layer in &self.layers {
            signal = layer.process(&signal);
            signal = layer.threshold_gate(&signal);
        }
        for (i, col) in self.columns.iter_mut().enumerate() {
            if i < signal.len() {
                col.activate(signal[i]);
            }
        }
        signal
    }

    /// Feed-forward with thalamic relay between each layer.
    pub fn process_with_relay(&mut self, input: &[Ternary]) -> Vec<Ternary> {
        let mut signal = input.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            // Relay through thalamus between layers
            if idx > 0 {
                self.thalamus.receive(&signal);
                signal = self.thalamus.relay().to_vec();
            }
            signal = layer.process(&signal);
            signal = layer.threshold_gate(&signal);
        }
        for (i, col) in self.columns.iter_mut().enumerate() {
            if i < signal.len() {
                col.activate(signal[i]);
            }
        }
        signal
    }

    /// Number of layers.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.columns.len()
    }
}

/// Ternary multiplication: sign rules.
fn ternary_mul(a: Ternary, b: Ternary) -> Ternary {
    match (a, b) {
        (Ternary::Neg, Ternary::Neg) => Ternary::Pos,
        (Ternary::Neg, Ternary::Pos) => Ternary::Neg,
        (Ternary::Pos, Ternary::Neg) => Ternary::Neg,
        (Ternary::Pos, Ternary::Pos) => Ternary::Pos,
        (_, Ternary::Zero) | (Ternary::Zero, _) => Ternary::Zero,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Pos));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn test_ternary_to_i8() {
        assert_eq!(Ternary::Neg.to_i8(), -1);
        assert_eq!(Ternary::Zero.to_i8(), 0);
        assert_eq!(Ternary::Pos.to_i8(), 1);
    }

    #[test]
    fn test_ternary_mul_table() {
        assert_eq!(ternary_mul(Ternary::Pos, Ternary::Pos), Ternary::Pos);
        assert_eq!(ternary_mul(Ternary::Neg, Ternary::Neg), Ternary::Pos);
        assert_eq!(ternary_mul(Ternary::Pos, Ternary::Neg), Ternary::Neg);
        assert_eq!(ternary_mul(Ternary::Neg, Ternary::Pos), Ternary::Neg);
        assert_eq!(ternary_mul(Ternary::Zero, Ternary::Pos), Ternary::Zero);
        assert_eq!(ternary_mul(Ternary::Pos, Ternary::Zero), Ternary::Zero);
        assert_eq!(ternary_mul(Ternary::Zero, Ternary::Zero), Ternary::Zero);
    }

    #[test]
    fn test_cortex_layer_process() {
        let mut layer = CortexLayer::new(0, 3, 1);
        layer.weights = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos];
        let input = vec![Ternary::Pos, Ternary::Pos, Ternary::Neg];
        let out = layer.process(&input);
        assert_eq!(out, vec![Ternary::Pos, Ternary::Neg, Ternary::Neg]);
    }

    #[test]
    fn test_cortex_layer_threshold_gate_pass() {
        let layer = CortexLayer::new(0, 4, 2);
        let vals = vec![Ternary::Pos, Ternary::Pos, Ternary::Zero, Ternary::Neg];
        let gated = layer.threshold_gate(&vals);
        assert_eq!(gated, vals);
    }

    #[test]
    fn test_cortex_layer_threshold_gate_block() {
        let layer = CortexLayer::new(0, 4, 3);
        let vals = vec![Ternary::Pos, Ternary::Pos, Ternary::Zero, Ternary::Neg];
        let gated = layer.threshold_gate(&vals);
        assert_eq!(gated, vec![Ternary::Zero; 4]);
    }

    #[test]
    fn test_cortex_layer_adapt() {
        let mut layer = CortexLayer::new(0, 3, 1);
        let input = vec![Ternary::Pos, Ternary::Neg, Ternary::Zero];
        let error = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos];
        layer.adapt(&input, &error);
        // Hebbian rule: w[i] = sign(x[i] × e[i])
        // i=0: sign(+1 × +1) = +1 → Pos
        assert_eq!(layer.weights[0], Ternary::Pos);
        // i=1: sign(-1 × -1) = +1 → Pos
        assert_eq!(layer.weights[1], Ternary::Pos);
        // i=2: input is Zero → weight unchanged (stays Zero)
        assert_eq!(layer.weights[2], Ternary::Zero);
    }

    #[test]
    fn test_cortex_layer_adapt_all_nonzero_combos() {
        // Verify the full Hebbian truth table: w[i] = sign(x[i] × e[i])
        let mut layer = CortexLayer::new(0, 4, 1);
        let input = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos, Ternary::Neg];
        let error = vec![Ternary::Pos, Ternary::Pos, Ternary::Neg, Ternary::Neg];
        layer.adapt(&input, &error);
        assert_eq!(layer.weights[0], Ternary::Pos); // +1 × +1 = +1
        assert_eq!(layer.weights[1], Ternary::Neg); // -1 × +1 = -1
        assert_eq!(layer.weights[2], Ternary::Neg); // +1 × -1 = -1
        assert_eq!(layer.weights[3], Ternary::Pos); // -1 × -1 = +1
    }

    #[test]
    fn test_cortex_layer_reset() {
        let mut layer = CortexLayer::new(0, 3, 1);
        layer.weights = vec![Ternary::Pos; 3];
        layer.reset();
        assert!(layer.weights.iter().all(|&w| w == Ternary::Zero));
    }

    #[test]
    fn test_cortical_column_activate() {
        let mut col = CorticalColumn::new(0, 3);
        col.activate(Ternary::Pos);
        assert!(col.active);
        assert_eq!(col.activations[0], Ternary::Pos);
    }

    #[test]
    fn test_cortical_column_shift() {
        let mut col = CorticalColumn::new(0, 3);
        col.activate(Ternary::Pos);
        col.activate(Ternary::Neg);
        assert_eq!(col.activations[0], Ternary::Neg);
        assert_eq!(col.activations[1], Ternary::Pos);
        assert_eq!(col.activations[2], Ternary::Zero);
    }

    #[test]
    fn test_cortical_column_output() {
        let mut col = CorticalColumn::new(0, 2);
        col.activate(Ternary::Pos);
        col.activate(Ternary::Neg);
        assert_eq!(col.output(), Ternary::Pos);
    }

    #[test]
    fn test_cortical_column_deactivate_on_zero() {
        let mut col = CorticalColumn::new(0, 2);
        col.activate(Ternary::Pos);
        col.activate(Ternary::Zero);
        assert!(!col.active);
    }

    #[test]
    fn test_thalamus_receive_and_relay() {
        let mut t = Thalamus::new(3);
        t.receive(&[Ternary::Pos, Ternary::Neg, Ternary::Zero]);
        assert_eq!(t.relay(), &[Ternary::Pos, Ternary::Neg, Ternary::Zero]);
    }

    #[test]
    fn test_thalamus_gate_blocks() {
        let mut t = Thalamus::new(3);
        t.set_gate(1, false);
        t.receive(&[Ternary::Pos, Ternary::Neg, Ternary::Pos]);
        assert_eq!(t.relay()[1], Ternary::Zero); // gate blocked
        assert_eq!(t.relay()[0], Ternary::Pos);
    }

    #[test]
    fn test_thalamus_attention_modulation() {
        let mut t = Thalamus::new(2);
        t.modulate_attention(&[-5, 2]);
        assert!(!t.gates[0]); // attention < -3
        assert!(t.gates[1]);
    }

    #[test]
    fn test_corpus_callosum_transfer() {
        let mut cc = CorpusCallosum::new(3);
        cc.receive_left(&[Ternary::Pos, Ternary::Neg, Ternary::Zero]);
        let right = cc.transfer_left_to_right();
        assert_eq!(right, vec![Ternary::Pos, Ternary::Neg, Ternary::Zero]);
    }

    #[test]
    fn test_corpus_callosum_sever() {
        let mut cc = CorpusCallosum::new(3);
        cc.sever_fiber(1);
        assert_eq!(cc.active_fiber_count(), 2);
        cc.receive_left(&[Ternary::Pos, Ternary::Pos, Ternary::Pos]);
        let right = cc.transfer_left_to_right();
        assert_eq!(right[1], Ternary::Zero); // severed
    }

    #[test]
    fn test_cortical_map_set_get() {
        let mut m = CorticalMap::new(3, 3);
        assert!(m.set(1, 2, Ternary::Pos));
        assert_eq!(m.get(1, 2), Some(Ternary::Pos));
        assert_eq!(m.get(0, 0), Some(Ternary::Zero));
        assert_eq!(m.get(3, 0), None); // out of bounds
    }

    #[test]
    fn test_cortical_map_neighbors() {
        let mut m = CorticalMap::new(3, 3);
        m.set(1, 1, Ternary::Pos);
        let n = m.neighbors(1, 1);
        assert_eq!(n.len(), 4); // all 4 neighbors in interior
    }

    #[test]
    fn test_cortical_map_neighbors_corner() {
        let m = CorticalMap::new(3, 3);
        let n = m.neighbors(0, 0);
        assert_eq!(n.len(), 2); // corner has 2 neighbors
    }

    #[test]
    fn test_cortical_map_center_of_mass() {
        let mut m = CorticalMap::new(4, 4);
        m.set(0, 0, Ternary::Pos);
        m.set(2, 2, Ternary::Pos);
        let (cx, cy) = m.center_of_mass();
        assert_eq!(cx, 1.0);
        assert_eq!(cy, 1.0);
    }

    #[test]
    fn test_cortical_map_center_of_mass_empty() {
        let m = CorticalMap::new(4, 4);
        let (cx, cy) = m.center_of_mass();
        assert_eq!(cx, 2.0); // center of 4-wide grid
        assert_eq!(cy, 2.0);
    }

    #[test]
    fn test_cortex_process() {
        let mut cortex = Cortex::new(&[(4, 1), (4, 1)]);
        // Set all weights to Pos on both layers
        for layer in &mut cortex.layers {
            for w in &mut layer.weights {
                *w = Ternary::Pos;
            }
        }
        let input = vec![Ternary::Pos, Ternary::Pos, Ternary::Pos, Ternary::Pos];
        let out = cortex.process(&input);
        // Layer 0: Pos⊙Pos=Pos for all 4, threshold 1: 4≥1 → [Pos;4]
        // Layer 1: Pos⊙Pos=Pos for all 4, threshold 1: 4≥1 → [Pos;4]
        assert_eq!(out, vec![Ternary::Pos; 4]);
    }

    #[test]
    fn test_cortex_process_threshold_blocks() {
        let mut cortex = Cortex::new(&[(4, 1), (4, 3)]);
        for layer in &mut cortex.layers {
            for w in &mut layer.weights {
                *w = Ternary::Pos;
            }
        }
        // Only 2 positives — layer 1 threshold is 3, so output is blocked
        let input = vec![Ternary::Pos, Ternary::Pos, Ternary::Zero, Ternary::Zero];
        let out = cortex.process(&input);
        // Layer 0: [Pos,Pos,Zero,Zero], threshold 1: 2≥1 → pass
        // Layer 1: [Pos,Pos,Zero,Zero], threshold 3: 2<3 → blocked → [Zero;4]
        assert_eq!(out, vec![Ternary::Zero; 4]);
    }

    #[test]
    fn test_cortex_depth_and_width() {
        let cortex = Cortex::new(&[(8, 1), (6, 2), (4, 1)]);
        assert_eq!(cortex.depth(), 3);
        assert_eq!(cortex.width(), 8); // max width
    }

    #[test]
    fn test_cortex_process_with_relay() {
        let mut cortex = Cortex::new(&[(3, 1), (3, 1)]);
        for layer in &mut cortex.layers {
            for w in &mut layer.weights {
                *w = Ternary::Pos;
            }
        }
        let input = vec![Ternary::Pos, Ternary::Pos, Ternary::Pos];
        let out = cortex.process_with_relay(&input);
        // Layer 0: Pos⊙Pos=Pos, threshold 1: 3≥1 → [Pos;3]
        // Relay (all gates open): [Pos;3]
        // Layer 1: Pos⊙Pos=Pos, threshold 1: 3≥1 → [Pos;3]
        assert_eq!(out, vec![Ternary::Pos; 3]);
    }

    #[test]
    fn test_thalamus_open_channel_count() {
        let mut t = Thalamus::new(4);
        assert_eq!(t.open_channel_count(), 4);
        t.set_gate(0, false);
        t.set_gate(2, false);
        assert_eq!(t.open_channel_count(), 2);
    }

    #[test]
    fn test_thalamus_reset() {
        let mut t = Thalamus::new(2);
        t.receive(&[Ternary::Pos, Ternary::Neg]);
        t.set_gate(0, false);
        t.reset();
        assert_eq!(t.relay(), &[Ternary::Zero, Ternary::Zero]);
        assert_eq!(t.open_channel_count(), 2);
    }
}
