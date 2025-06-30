# Makefile for LT-Gate experiments

.PHONY: all clean test train akida_env summary

# Directories
DATA_DIR = data
CKPT_DIR = ckpts
LOG_DIR = logs
TOOLS_DIR = tools
ANALYSIS_DIR = analysis

# Python environment
PYTHON = python
PIP = pip

# Default target
all: test train

# Clean build artifacts
clean:
	rm -rf $(CKPT_DIR)/*
	rm -rf $(LOG_DIR)/*

# Run tests
test:
	$(PYTHON) -m pytest tests/

# Train models
train: train_task1 train_task2

train_task1:
	$(PYTHON) train.py --config configs/ltgate.yaml \
		--dataset $(DATA_DIR)/fast/train.h5 \
		--phase task1 --gpu

train_task2:
	$(PYTHON) train.py --config configs/ltgate.yaml \
		--dataset $(DATA_DIR)/slow/train.h5 \
		--phase task2 --gpu

# Akida deployment and measurement
akida_env:
	conda create -n akida_env python=3.11 -y
	conda run -n akida_env pip install tensorflow==2.15.0
	conda run -n akida_env pip install akida==2.13.0
	conda run -n akida_env pip install cnn2snn==2.13.0 quantizeml==2.13.0 akida-models==1.7.0

prepare_calib:
	$(PYTHON) $(TOOLS_DIR)/prepare_calib.py \
		--dataset $(DATA_DIR)/fast/train.h5 \
		--output $(DATA_DIR)/calib.npz \
		--samples 1024

akida_ltgate: $(CKPT_DIR)/task2_ltgate.pt prepare_calib
	$(PYTHON) $(TOOLS_DIR)/torch2onnx.py $< ltgate.onnx
	$(PYTHON) $(TOOLS_DIR)/convert_akida.py ltgate.onnx ltgate.fbz $(DATA_DIR)/calib.npz
	$(PYTHON) $(TOOLS_DIR)/run_akida.py ltgate.fbz $(DATA_DIR)/slow/test.h5 $(LOG_DIR)/ltgate_energy_akida.json

# Loihi deployment and measurement (alternative)
loihi_ltgate: $(CKPT_DIR)/task2_ltgate.pt
	$(PYTHON) src/loihi_runner.py --ckpt $< \
		--dataset $(DATA_DIR)/slow/test.h5 \
		--energy_log $(LOG_DIR)/ltgate_energy_loihi.json

# Grid search
grid_search:
	$(PYTHON) grid_search.py \
		--base-config configs/ltgate.yaml \
		--output-dir $(LOG_DIR)/grid_search

# Analysis and summary
summary: $(LOG_DIR)/master_metrics.csv
	@echo "Generating summary plots and statistics..."
	$(PYTHON) $(ANALYSIS_DIR)/aggregate.py --output $(LOG_DIR)/master_metrics.csv
	$(PYTHON) $(ANALYSIS_DIR)/stats_tests.py --output $(LOG_DIR)/statistical_tests.json
	$(PYTHON) $(ANALYSIS_DIR)/plotting.py --metrics-csv $(LOG_DIR)/master_metrics.csv
	@echo "Opening summary notebook..."
	jupyter notebook $(ANALYSIS_DIR)/summary.ipynb

$(LOG_DIR)/master_metrics.csv:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) $(ANALYSIS_DIR)/aggregate.py --output $@
