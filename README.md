# Multi-Recurrent Neural Network (MRN)

A PyTorch implementation of the Multi-Recurrent Neural Network (MRN), a recurrent neural network architecture with a unique **sluggish state-based memory mechanism** that excels at time-series processing and sequential data modeling.

### Key Features

- **Sluggish State-Space Memory**: Unique memory mechanism where information decays gradually across multiple memory banks
- **Broadcast Memory Design**: All memory banks project to all hidden layers for enhanced information flow
- **Variable Depth Architecture**: Support for arbitrary network depths (minimum 3 layers)
- **Lower Complexity**: Typically requires fewer parameters than LSTM while achieving competitive or superior performance
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

# Define network structure
nn_structure = [10, 32, 16, 5]  # input=10, hidden layers=32,16, output=5
memory_structure = [4, 3, 2, 0]  # memory banks per layer

# Create model
model = MRN(
    nn_structure=nn_structure,
    memory_structure=memory_structure,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Process a sequence
batch_size, seq_len, input_size = 32, 100, 10
inputs = torch.randn(batch_size, seq_len, input_size)

# Get outputs for all timesteps
outputs = model(inputs, return_sequences=True)  # [32, 100, 5]

# Get only final output
final_output = model(inputs, return_sequences=False)  # [32, 5]
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

## Repository Structure

```
multi_recurrent_network/
├── src/
│   ├── model/
│   │   ├── cell.py          # MRN cell implementation
│   │   ├── mrn.py           # MRN sequence processor
│   │   └── __init__.py
│   └── utils.py             # Utility functions
├── sample/
│   ├── classification.py    # Digit classification example
│   ├── ensemble.py          # Ensemble forecasting
│   ├── text_generation.py   # Character-level text generation
│   └── timeseries.py        # COVID-19 forecasting
├── test/
│   └── mrn_test.py          # Unit tests
├── data/                    # Output directories for results
│   ├── classification/
│   ├── ensemble/
│   ├── text_gen/
│   └── time_series/
├── pyproject.toml
├── uv.lock
└── README.md
```

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

Contributions are welcome! Areas for improvement include:
- Different activation functions for memory updates
- Memory bank pruning algorithms based on importance
- Deep MRN architectures for complex tasks
- Knowledge extraction from learned representations
- Additional application domains (speech, video, etc.)

Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

This implementation builds upon the pioneering work of:
- **Claudia Ulbricht** (1994) - Original MRN architecture
- **Oluwatamilore Orojo** - Extensive research on optimizing sluggish state-based neural networks for temporal processing
- **Nottingham Trent University** - Supporting foundational research

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, additional optimization and testing may be required.