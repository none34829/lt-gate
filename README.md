# LT-Gate: Local Timescale Gating for Spiking Neural Networks

This repository implements Local Timescale Gates (LT-Gate) for spiking neural networks, along with baseline algorithms HLOP and DSD-SNN. The codebase supports training and evaluation on temporally distinct datasets, with a focus on continual learning scenarios.

## Project Structure

```
lt-gate/
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
│   └── evaluate.py       # Evaluation script
├── analysis/         # Analysis and ablation scripts
│   ├── ablation_runner.py    # Automated ablation experiments
│   ├── plot_ablation.py      # Ablation result plotting
│   └── aggregate.py          # Result aggregation
├── tools/            # Utility scripts
│   ├── run_akida.py          # Akida runtime measurements
│   └── convert_akida.py      # Model conversion for Akida
└── tests/            # Unit and integration tests
    ├── test_ltgate.py    # LT-Gate unit tests
    ├── test_hlop.py      # HLOP unit tests
    └── test_model.py     # Model integration tests
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

### Training

Train LT-Gate model on fast dataset:
```bash
python train.py --config configs/ltgate.yaml --dataset fast --seed 42
```

Train LT-Gate model on slow dataset (continual learning):
```bash
python train.py --config configs/ltgate.yaml --dataset slow --seed 42
```

Train HLOP baseline:
```bash
python train.py --config configs/hlop.yaml --dataset fast --seed 42
```

### Evaluation

Evaluate a trained checkpoint:
```bash
python src/evaluate.py --checkpoint logs/ltgate/seed_42/ckpt_task2.pt
```

The evaluation script will:
- Load the checkpoint and its configuration
- Create a compatible model architecture
- Run evaluation on the test dataset
- Save results to a JSON file

### Ablation Studies

Run a specific ablation variant:
```bash
python analysis/ablation_runner.py --variant tau20
```

Run all ablation variants:
```bash
python analysis/ablation_runner.py --variant all
```

Available ablation variants:
- `tau10`, `tau20`, `tau100`: Different time constant settings
- `nofast`, `noslow`: Ablating fast/slow pathways
- `gamma_fixed`: Fixed gamma parameter
- `eta_half`: Reduced learning rate

The ablation runner will:
1. Train on fast dataset (Task 1)
2. Continue training on slow dataset (Task 2)
3. Run energy measurements on Akida hardware
4. Save results to `logs/ablations/{variant}/seed_{seed}/`

### Energy Measurement

Run energy measurements on Akida hardware:
```bash
python tools/run_akida.py logs/ablations/tau20/seed_0/ckpt_task2.pt data/slow/test.h5 logs/ablations/tau20/seed_0/energy.json
```

## Configuration

The system uses a hierarchical configuration system:

### Base Configuration (configs/ltgate.yaml)
```yaml
alg: ltgate
batch_size: 32
epochs: 100
eta: 0.01
eta_v: 0.001
variance_lambda: 0.001
pre_decay: 0.8
tau_fast: 0.005
tau_slow: 0.1
dt: 0.001
reset_mechanism: subtract

conv_layers:
  - in_channels: 1
    out_channels: 16
    kernel_size: 3
    stride: 1
    padding: 1
  - in_channels: 16
    out_channels: 32
    kernel_size: 3
    stride: 2
    padding: 1
  - in_channels: 32
    out_channels: 64
    kernel_size: 3
    stride: 2
    padding: 1

fc_layers:
  - in_features: 3136
    out_features: 256
  - in_features: 256
    out_features: 10

calibration:
  enabled: true
  target_rate: 0.02
  batches: 10
  iters: 3
  tolerance: 0.5
  min_threshold: 0.05
  max_threshold: 2.0
```

### Ablation Configuration (configs/ablations/tau20.yaml)
```yaml
base_config: configs/ltgate.yaml
variant_tag: tau20
tau_fast: 0.02
seeds: [0, 1, 2]
```

## Troubleshooting

### Common Issues

1. **Import Errors**: The codebase uses relative imports. Run scripts from the project root directory.

2. **Checkpoint Compatibility**: Older checkpoints may use different model architectures. The evaluation script includes compatibility layers.

3. **Memory Issues**: For large models, reduce batch size or use CPU training.

4. **Calibration Convergence**: If calibration doesn't converge, adjust `target_rate` or `tolerance` in config.

### Debug Mode

Enable debug mode for detailed logging:
```bash
python train.py --config configs/ltgate.yaml --dataset fast --seed 42 --debug
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
