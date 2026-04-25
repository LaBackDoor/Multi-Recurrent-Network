# Multi-Recurrent Neural Network (MRN)

A PyTorch implementation of the Multi-Recurrent Neural Network (MRN), a recurrent neural network architecture with a unique **sluggish state-based memory mechanism** that excels at time-series processing and sequential data modeling.

For 3-layer configurations this implementation is faithful to the canonical MRN of Ulbricht (1994) and the formulation in Orojo's PhD thesis (2022). For networks deeper than one hidden layer, the thesis lists "Deep MRN" as future work without specifying a memory topology; this implementation extends the architecture using a **chain topology** described in [Architecture Notes](#architecture-notes).

### Key Features

- **Sluggish State-Space Memory**: Memory banks store layer activations through a leaky-integrator update with bank-specific decay rates, ranging from flexible (recent-information-dominant) to rigid (historical-information-dominant)
- **Canonical 3-Layer MRN**: At depth 3 the implementation reproduces the thesis exactly. With `nn_structure=[10, 20, 1]` and `memory_structure=[4, 3, 4]` the model has 2,321 trainable parameters, matching Orojo (2022) Table 4.7 t+6 row to the parameter
- **Chain Memory Topology for Deep Networks**: Each memory bank projects to one specific hidden layer following the chain rule, giving every level of feature abstraction its own temporal context without the parameter blowup of broadcast or the front/back bias of feeding only one layer
- **Real Backpropagation Through Time**: Gradients flow through memory updates across the full sequence so the network learns proper temporal structure
- **Batch-Aware Memory**: Each batch element maintains its own memory state; running a sample alone produces identical results to its slice in a batched run
- **Parameter-Efficient**: At depth 3 with `[4, 3, 4]` memory, the MRN uses 2,321 parameters where the comparable stacked LSTM baselines in the thesis use 5,781 (2-layer) or 9,061 (3-layer) parameters for the same task
- **Four Ready-to-Use Applications**:
  - Time-series forecasting
  - Classification
  - Text generation
  - Ensemble methods

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi_recurrent_network.git
cd multi_recurrent_network

# Install dependencies (using uv)
uv sync
```

## Quick Start

```python
import torch
from src.model.mrn import MRN

# Canonical 3-layer MRN (matches the thesis exactly)
nn_structure = [10, 20, 1]      # input=10, hidden=20, output=1
memory_structure = [4, 3, 4]    # 4 input, 3 hidden, 4 output memory banks

model = MRN(
    nn_structure=nn_structure,
    memory_structure=memory_structure,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)

# Process a sequence
batch_size, seq_len, input_size = 32, 100, 10
inputs = torch.randn(batch_size, seq_len, input_size)

# Get outputs for all timesteps
outputs = model(inputs, return_sequences=True)        # [32, 100, 1]

# Get only the final output
final_output = model(inputs, return_sequences=False)  # [32, 1]
```

A deeper network with two hidden layers, using the chain topology:

```python
nn_structure = [10, 32, 16, 5]      # input, hidden_1, hidden_2, output
memory_structure = [4, 3, 2, 0]     # memory banks per layer
model = MRN(nn_structure=nn_structure, memory_structure=memory_structure)
```

## Applications

### 1. Time-Series Forecasting

Predict future values in temporal sequences (e.g., COVID-19 cases, stock prices).

```bash
python sample/timeseries.py \
    --csv_path data/time_series/c19.csv \
    --country Afghanistan \
    --window_size 14 \
    --horizon 7 \
    --epochs 200
```

### 2. Classification

Classify sequential data (e.g., handwritten digits as sequences).

```bash
python sample/classification.py \
    --hidden_sizes 32 16 \
    --memory_structure 3 2 1 0 \
    --epochs 100 \
    --batch_size 32
```

### 3. Text Generation

Generate text character-by-character using trained MRN models.

```bash
python sample/text_generation.py \
    --text_path data/text_gen/sample_text.txt \
    --seq_length 50 \
    --hidden_sizes 128 64 \
    --epochs 50 \
    --temperatures 0.5 1.0 1.5
```

### 4. Ensemble Methods

Combine multiple MRN models for improved forecasting accuracy.

```bash
python sample/ensemble.py \
    --num_models 5 \
    --diverse_architectures \
    --epochs 100
```

## Architecture Notes

The canonical MRN is a 3-layer feedforward network (input, hidden, output) augmented with three types of memory banks (input, hidden, output) that store layer activations through a leaky-integrator update. All memory banks project into the single hidden layer's pre-activation. See Orojo (2022), Chapter 3.

### Memory update

Each memory bank has `K` items. Item `i` (for `i` in `1..K`) updates per Orojo (2022) section 3.1.1.1:

```
M_t^{k,i} = (i/K) * L_{t-1}^k + (1 - i/K) * M_{t-1}^{k,i}
```

where `L_{t-1}^k` is the activation at source layer `k` (input, hidden, or output). Item 1 retains most history (decay rate `1 - 1/K`), item `K` is fully flexible (decay rate 0). The collection of `K` items per bank type captures information at different timescales simultaneously.

### Hidden pre-activation (eq. 3.4 of the thesis)

For the canonical 3-layer MRN:

```
Ĥ_t = W_ih I_t + Σ W_Mih M_t^i + Σ W_Mhh M_t^h + Σ W_Moh M_t^o + b_h
H_t = sigmoid(Ĥ_t)
O_t = W_ho H_t + b_o            # output is linear and memory-free
```

### Topology for networks with more than one hidden layer

The thesis specifies a single hidden layer. Section 8.2 lists deep MRN as future work and does not specify how memory should connect when there are multiple hidden layers. This implementation uses a **chain topology**: each memory bank projects to exactly one hidden layer following this rule:

| Source layer | Target hidden layer |
|---|---|
| Input (layer 0) | First hidden (layer 1) |
| Hidden at L, where 1 ≤ L ≤ N-3 | Layer L+1 (next hidden in the stack) |
| Last hidden (layer N-2) | Layer N-2 (self-loop) |
| Output (layer N-1) | First hidden (layer 1, long-range feedback) |

For 3-layer networks (N=2 layers between input and output, i.e., one hidden layer), all rules collapse to "target layer 1", recovering the canonical MRN exactly. For deeper networks the chain rule gives every level of feature abstraction its own temporal context, avoids concentrating memory only at the front (input-augmentation bias) or only at the back (late-fusion bias), and keeps the parameter count substantially below a broadcast topology that would project every memory bank to every hidden layer.

### Initialization

Following the thesis ("weight initialisations are very small") and Orojo's NumPy reference:

- All feedforward weights, biases, and memory-projection weights are sampled from `uniform(-w, w)` with `w = 0.01` by default. Configurable via `weight_init_range`.
- The first hidden layer's bias is initialized to a constant `0.5` by default, matching Orojo's NumPy reference. Pass `hidden_bias_init_value=None` to use small uniform initialization instead.
- Memory state is randomly initialized (`torch.rand`) by default, matching the thesis text ("memory randomly initialised"). Pass `init_memory_mode="constant"` for a constant `0.5` initialization matching the NumPy code.

### Backpropagation through time

Memory tensors are carried in `MRNState` without detachment, so PyTorch autograd builds the full temporal graph across the sequence and `loss.backward()` performs proper BPTT through the memory updates. For long sequences where this becomes expensive, call `state = state.detach()` between segments to truncate BPTT explicitly.

### Batch handling

Memory state has shape `[batch_size, num_items, layer_size]`. Each batch element updates its own memory from its own activations. There is no cross-batch contamination; running a single sample produces identical results to its slice in a batched run.

## Reproducing thesis results

The four MRN configurations from Orojo (2022), Table 4.7 (oil price forecasting):

```python
# t+1 horizon: 1,881 trainable parameters (Table 4.7 reports 1,640, memory weights only)
model = MRN(nn_structure=[10, 20, 1], memory_structure=[0, 4, 2],
            weight_init_range=0.01, hidden_bias_init_value=0.5,
            init_memory_mode="constant")

# t+3 horizon: 1,921 trainable parameters (Table 4.7 reports 1,680, memory weights only)
model = MRN(nn_structure=[10, 20, 1], memory_structure=[0, 4, 4],
            weight_init_range=0.01, hidden_bias_init_value=0.5,
            init_memory_mode="constant")

# t+6 horizon: 2,321 trainable parameters (matches Table 4.7 exactly)
model = MRN(nn_structure=[10, 20, 1], memory_structure=[4, 3, 4],
            weight_init_range=0.01, hidden_bias_init_value=0.5,
            init_memory_mode="constant")

# t+12 horizon: 301 trainable parameters (matches Table 4.7 exactly)
model = MRN(nn_structure=[10, 20, 1], memory_structure=[0, 0, 3],
            weight_init_range=0.01, hidden_bias_init_value=0.5,
            init_memory_mode="constant")
```

The two-of-four exact matches reproduce the thesis to the parameter. The other two row counts in Table 4.7 differ by exactly 241 (the count of feedforward weights and biases for these models), suggesting the thesis reports memory-weight-only counts for those rows. Including feedforward parameters everywhere gives the parenthetical numbers above.

## Repository Structure

```
multi_recurrent_network/
├── src/
│   ├── model/
│   │   ├── cell.py          # MRN cell, MemoryBank, MRNState
│   │   ├── mrn.py           # MRN sequence wrapper
│   │   ├── kan/             # MR-KAN (Multi-Recurrent KAN) variant
│   │   │   ├── kan_linear.py   # Vendored efficient-kan + LayerNorm coupling
│   │   │   ├── cell.py         # MRKANCell, KANMemoryBank, MRKANState
│   │   │   └── mrkan.py        # MRKAN sequence wrapper
│   │   └── __init__.py
│   └── utils.py             # Utility functions
├── sample/
│   ├── classification.py    # Digit classification example
│   ├── ensemble.py          # Ensemble forecasting
│   ├── text_generation.py   # Character-level text generation
│   ├── timeseries.py        # COVID-19 forecasting (MRN)
│   └── timeseries_kan.py    # COVID-19 forecasting (MR-KAN adapter)
├── test/
│   ├── mrn_test.py          # MRN unit tests
│   └── mrkan_test.py        # MR-KAN unit tests (7 cases)
├── data/                    # Output directories for results
│   ├── classification/
│   ├── ensemble/
│   ├── text_gen/
│   └── time_series/
├── pyproject.toml
├── uv.lock
└── README.md
```

## MR-KAN: Multi-Recurrent KAN variant

MR-KAN integrates Kolmogorov-Arnold Networks (KAN) into the MRN by replacing
the memory-to-hidden weight matrices with `KANLinear` modules (learnable
univariate B-splines on every edge). The input-to-hidden and hidden-to-output
projections stay as plain `nn.Linear`, so any attribution story is localised
to the temporal-integration path rather than the whole network. All other MRN
semantics (fixed per-item ratios `i/K`, sluggish memory update, sigmoid hidden,
linear memory-free output, chain topology for deeper networks) are preserved
verbatim.

### Quick start

```python
import torch
from src.model.kan import MRKAN

model = MRKAN(
    nn_structure=[10, 20, 1],      # same shape contract as MRN
    memory_structure=[4, 3, 4],    # same memory-bank counts as MRN
    kan_grid_size=3,               # spline resolution (default 3)
    kan_spline_order=3,            # cubic splines (default)
    kan_use_layernorm=True,        # default: LayerNorm before each spline
)

x = torch.randn(32, 100, 10)
y = model(x)                       # (32, 100, 1)
```

### What's actually different

1. **Memory projections are KAN, not linear.** Each memory item feeds a
   dedicated `KANLinear(source_size, target_size)` whose `base_weight` gives a
   linear residual path and whose `spline_weight` adds a learnable univariate
   correction per input dimension.
2. **LayerNorm before every spline, with grid widened to match.** Post-LN
   samples are ~`N(0, 1)`, so the default `grid_range` is `(-3, 3)` (vs.
   efficient-kan's `(-1, 1)`) — this keeps ~99.7% of samples inside the active
   grid. On 1-dim source layers (e.g. a regression output) LayerNorm is
   silently disabled, because LN over a single feature is deterministically
   zero and would block BPTT.
3. **`update_grid` is exposed but not wired into forward.** Call
   `model.update_grids(calibration_inputs)` between full forward+backward
   passes to adaptively refit the knot grid to training data. Never mid-
   sequence — the method is `@torch.no_grad()` and in-place.

### Parameter budget

With `grid_size=3, spline_order=3, enable_standalone_scale_spline=False,
use_layernorm=True`, each `KANLinear(in, out)` costs `out*in*7 + 2*in`
parameters (`out*in` base + `out*in*6` spline coefficients + `2*in` LN
gamma/beta). For the canonical `nn_structure=[10, 20, 1]` with
`memory_structure=[4, 3, 4]` (11 KANLinears), MR-KAN totals ~15k trainable
parameters, roughly 6-7x the MRN baseline. If that is too much, lower
`kan_grid_size` to 2 or disable LayerNorm (set `kan_use_layernorm=False`;
this also resets `grid_range` to `(-1, 1)`).

### Running the tests

```bash
PYTHONPATH=. python test/mrkan_test.py
```

All seven tests should pass. Test #4 (parity regression) confirms that MR-KAN
with splines zeroed, `base_activation=Identity`, and LN disabled produces
byte-identical outputs to the MRN baseline.

### Status: v1 (this release)

- Memory projections replaced with KAN; other paths remain linear
- LayerNorm + widened grid as the stability story
- Fixed `i/K` memory ratios (not learnable)
- PyTorch autograd for BPTT (no custom autograd.Function)

### v2 roadmap

- **SL-MR-KAN** (flagship): learnable ratios via `RatioControlUnit`s that are
  themselves small `KANLinear`s, extending thesis Chapter 6 (Self-Learning MRN)
  to the KAN setting. Enables spline-shape pruning (extends Chapter 7).
- **`kan_paths` flag**: optionally make `W_ih` and `W_ho` KAN too, for
  head-to-head comparison against a fully-KAN network.
- **Safe adaptive grid updates**: `model.calibrate_grids(loader)` helper that
  does the eval-mode hook + `update_grid` + re-init-state pattern correctly,
  gated to run only between epochs.
- **Spline-shape pruning** (stretch): extend Chapter 7 by pruning memory items
  whose KANLinears learn near-identical functions, measured by normalised L2
  distance of sampled spline curves.

## Citations

### Primary Reference

If you use this implementation, please cite the foundational thesis:

```bibtex
@phdthesis{orojo2021optimizing,
  title={Optimizing sluggish state-based neural networks for effective time-series processing},
  author={Orojo, Oluwatamilore O},
  year={2021},
  school={Nottingham Trent University}
}
```

### Recent Applications

**Software Vulnerability Prediction:**

```bibtex
@inproceedings{orojo2024predicting,
  title={Predicting Software Vulnerability Trends with Multi-Recurrent Neural Networks: A Time Series Forecasting Approach},
  author={Orojo, Abanisenioluwa and Elumelu, Webster and Orojo, Oluwatamilore and Donnahoo, Micheal and Hutton, Shaun},
  booktitle={The International Conference on Natural Language Processing and Artificial Intelligence for Cyber Security (NLPAICS'2024)},
  year={2024}
}
```

Related GitHub: [predicting-software-vulnerabilities](https://github.com/LaBackDoor/predicting-software-vulnerabilities)

**State-of-the-Art Time-Series Processing:**

```bibtex
@article{OROJO2023488,
  title={The Multi-Recurrent Neural Network for State-Of-The-Art Time-Series Processing},
  author={Orojo, Oluwatamilore and Tepper, Jonathan and McGinnity, T.M. and Mahmud, Mufti},
  journal={Procedia Computer Science},
  volume={222},
  pages={488--498},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.procs.2023.08.187}
}
```

## Contributing

Contributions are welcome. Areas for improvement include:

- Self-learning ratio mechanisms (RCUs from Orojo 2022, Chapter 6) where the layer-link and self-link ratios are learned rather than fixed
- Memory bank pruning algorithms based on similarity (Orojo 2022, Chapter 7)
- Knowledge extraction from learned representations
- Additional application domains (speech, video, etc.)
- Alternative deep MRN topologies (broadcast, first-hidden-only, last-hidden-only) exposed through a `memory_topology` parameter
- A custom autograd Function reproducing the simplified BPTT formula from Orojo's NumPy reference, for those who specifically need bit-for-bit reproducibility with the thesis's experiments

Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

This implementation builds upon the pioneering work of:

- **Claudia Ulbricht** (1994), original MRN architecture
- **Oluwatamilore Orojo**, extensive research on optimizing sluggish state-based neural networks for temporal processing
- **Nottingham Trent University**, supporting foundational research

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, additional optimization and testing may be required. The 3-layer MRN reproduces the thesis exactly; deeper configurations use the chain topology described above, which is one of several reasonable extensions of the canonical architecture and is not validated empirically by the thesis or any cited follow-up paper.