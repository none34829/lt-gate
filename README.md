# LT-Gate: Local Timescale Gating for Spiking Neural Networks

This repository implements Local Timescale Gates (LT-Gate) for spiking neural networks, along with baseline algorithms HLOP and DSD-SNN. The codebase supports training and evaluation on temporally distinct datasets, with a focus on continual learning scenarios.

## Project Structure

```
lt-gates/
├── configs/           # Configuration files for different algorithms and experiments
│   ├── ltgate.yaml   # LT-Gate hyperparameters and training settings
│   ├── hlop.yaml     # HLOP baseline configuration
│   └── ablations/    # Ablation study configurations
├── data/             # Dataset storage
│   ├── fast/         # Fast temporal dataset (1kHz)
│   └── slow/         # Slow temporal dataset (50Hz)
├── src/              # Source code
│   ├── algorithms/   # Algorithm implementations
│   │   ├── ltgate.py # LT-Gate implementation
│   │   └── hlop.py   # HLOP baseline implementation
│   ├── layers/       # Neural network layers
│   │   ├── dual_lif_neuron.py    # Dual-pathway LIF neuron
│   │   ├── dual_conv_lif.py      # Dual convolutional LIF layer
│   │   ├── lt_conv.py            # LT-Gate convolutional layer
│   │   └── hlop_subspace.py      # HLOP subspace projection layer
│   ├── data_loader.py    # Dataset loading utilities
│   ├── model.py          # SNN backbone architecture
│   └── train.py          # Training pipeline entry point
└── tests/            # Unit and integration tests
    ├── test_ltgate.py    # LT-Gate unit tests
    ├── test_hlop.py      # HLOP unit tests
    └── test_backbone.py  # Backbone integration tests
```

## Features

- **LT-Gate Algorithm**: Implementation of Local Timescale Gates for adaptive temporal processing
- **Baseline Algorithms**: HLOP and DSD-SNN implementations for comparison
- **Flexible SNN Backbone**: Configurable backbone supporting all algorithms
- **Multi-Platform Support**: 
  - CPU/GPU training via PyTorch
  - Loihi 2 neuromorphic deployment
  - Akida runtime/energy measurements
- **Comprehensive Testing**: Unit tests and integration tests for all components
- **Experiment Tools**:
  - Dataset generation and validation
  - Training pipeline with checkpointing
  - Evaluation metrics and logging
  - Statistical analysis and plotting

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- h5py
- numpy
- tqdm
- pyyaml
- matplotlib
- seaborn
- pandas
- pingouin (for statistical analysis)

Optional dependencies:
- lava-dl (for Loihi 2 deployment)
- akida (for Akida runtime measurements)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ansschh/lt-gate.git
cd lt-gate
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Generate and validate dataset splits:
```bash
python src/generate_splits.py --fast-rate 1000 --slow-rate 50
```

### Training

Train LT-Gate model:
```bash
python src/train.py --config configs/ltgate.yaml --gpu
```

Train HLOP baseline:
```bash
python src/train.py --config configs/hlop.yaml --alg hlop --gpu
```

### Evaluation

Run evaluation metrics:
```bash
python src/evaluate.py --model-path ckpts/ltgate_best.pt
```

### Ablation Studies

Run ablation experiments:
```bash
python analysis/ablation_runner.py --config configs/ablations/
```

Generate ablation plots:
```bash
python analysis/plot_ablation.py --results-dir results/ablation/
```

## Configuration

Example LT-Gate configuration (configs/ltgate.yaml):
```yaml
model:
  backbone: "dual_conv"
  input_size: [1, 28, 28]
  hidden_size: 256
  num_classes: 10

training:
  batch_size: 32
  epochs: 100
  lr: 0.001
  optimizer: "adam"
  scheduler: "cosine"

ltgate:
  tau_fast: 1.0
  tau_slow: 20.0
  gamma_init: 0.5
  eta_gamma: 0.01
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
[Citation details will be added upon publication]
```
