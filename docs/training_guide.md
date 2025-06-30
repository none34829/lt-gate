# LT-Gate Training Pipeline Guide

## Command Cheat-Sheet

### Basic Training
```bash
# Train LT-Gate on Task 1 (fast task)
python train.py --config configs/ltgate.yaml --dataset data/fast/train.h5 --phase task1 --gpu

# Train LT-Gate on Task 2 (slow task)
python train.py --config configs/ltgate.yaml --dataset data/slow/train.h5 --phase task2 --gpu

# Enable mixed precision training
python train.py --config configs/ltgate.yaml --dataset data/fast/train.h5 --mixed-precision
```

### Checkpointing and Resume
```bash
# Save checkpoint every N epochs
python train.py --config configs/ltgate.yaml --ckpt-interval 5

# Resume from checkpoint
python train.py --config configs/ltgate.yaml --resume ckpts/task1_ltgate.pt
```

### Diagnostics and Monitoring
```bash
# Enable edge-case diagnostics
python train.py --config configs/ltgate.yaml --diagnostics --diag-interval 100

# Monitor weight ranges
python train.py --config configs/ltgate.yaml --weight-clip --max-weight 1.0 --min-weight -1.0

# Check γ stuck states
python train.py --config configs/ltgate.yaml --gamma-threshold 0.95
```

### Grid Search and Loihi Deployment
```bash
# Run hyperparameter grid search
python grid_search.py --base-config configs/ltgate.yaml

# Deploy and measure on Loihi-2
python src/loihi_runner.py --ckpt ckpts/best_model.pt --dataset data/test.h5
```

## Troubleshooting FAQ

### Q: Training is extremely slow or stuck
**A**: Check the following:
1. Verify GPU is being used: Look for "Using device: cuda" in output
2. Check batch size in config: Reduce if OOM errors occur
3. Monitor γ values: If stuck near 0/1, adjust learning rate
4. Enable mixed precision: Add --mixed-precision flag

### Q: Out of Memory (OOM) Errors
**A**: Try these solutions:
1. Reduce batch size in config
2. Enable gradient disabling: --disable-gradients
3. Use lazy data loading: Already enabled by default
4. Clear GPU cache between epochs: Automatic in training loop

### Q: Validation Accuracy Not Improving
**A**: Common fixes:
1. Check learning rate schedule in config
2. Monitor weight ranges with --weight-clip
3. Verify dataset splits are correct
4. Enable diagnostics to check γ behavior

### Q: Loihi Deployment Issues
**A**: Troubleshooting steps:
1. Verify weights are within 8-bit range
2. Check core utilization in output
3. Run with smaller test set first
4. Enable Loihi constraint checking during training

### Q: Grid Search Taking Too Long
**A**: Optimization options:
1. Reduce parameter combinations in grid_search.py
2. Use smaller validation set for quick iterations
3. Run multiple searches in parallel
4. Start with coarse grid, then refine

## Expected Resource Usage

### Training (per epoch)
- GPU Memory: ~4GB (RTX 4090)
- CPU Memory: ~2GB
- Disk Space: ~100MB for checkpoints
- Time: 2-5 minutes depending on dataset

### Loihi-2 Deployment
- Core Usage: Varies by model size
- Memory: ~100MB per core
- Latency: 1-5ms per sample
- Energy: 100-500 pJ per inference

## Best Practices

1. Always use version control for configs
2. Save checkpoints every 5 epochs
3. Enable diagnostics during initial runs
4. Validate Loihi constraints early
5. Document hyperparameter choices
