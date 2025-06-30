# Performance and Resource Usage Guide

## Hardware Requirements

### Minimum Requirements
- GPU: NVIDIA RTX 3060 or better
- CPU: 4 cores, 2.5GHz+
- RAM: 16GB
- Storage: 10GB free space

### Recommended Specifications (Used in Development)
- GPU: NVIDIA RTX 4090
- CPU: 8+ cores, 3.5GHz+
- RAM: 32GB
- Storage: 50GB SSD

## Runtime Expectations

### Training Phase
1. **Data Loading**
   - Initial load: 10-15 seconds
   - Batch loading: ~2ms per batch (batch size 32)
   - Memory usage: ~2GB RAM

2. **Forward Pass**
   - Time per batch: ~5ms
   - GPU memory: ~4GB
   - Spike processing overhead: ~1ms per 1000 spikes

3. **Learning Updates**
   - LT-Gate updates: ~3ms per batch
   - Weight updates: ~2ms per batch
   - Memory peaks: Up to 6GB GPU RAM during updates

4. **Full Epoch Statistics**
   - Fast task (1kHz): 3-4 minutes
   - Slow task (50Hz): 2-3 minutes
   - Checkpoint saving: ~5 seconds
   - Validation: ~30 seconds

### Loihi-2 Deployment

1. **Compilation**
   - Weight quantization: ~10 seconds
   - Network compilation: 1-2 minutes
   - Memory usage: ~4GB RAM peak

2. **Inference**
   - Latency per sample: 1-5ms
   - Energy per inference: 100-500 pJ
   - Core utilization: 20-30% of available cores
   - Memory per core: ~100MB

3. **Measurement Collection**
   - Setup time: ~30 seconds
   - Measurement per sample: ~10ms
   - Total time (1000 samples): ~3 minutes

## Resource Scaling

### Memory Scaling
- GPU memory scales linearly with batch size
- CPU memory increases ~500MB per worker process
- Disk usage grows ~20MB per checkpoint

### Time Scaling
- Training time scales linearly with:
  - Number of epochs
  - Dataset size
  - Batch size (up to GPU saturation)
- Grid search time scales with:
  - Number of hyperparameter combinations
  - Epochs per training run

### Network Size Impact
- Each additional layer adds:
  - ~100MB GPU memory
  - ~1ms to forward pass
  - ~2ms to learning updates
  - ~10% compilation time for Loihi

## Optimization Tips

1. **Memory Optimization**
   - Use mixed precision training
   - Enable lazy data loading
   - Adjust batch size based on GPU
   - Clean checkpoints regularly

2. **Speed Optimization**
   - Set optimal number of workers (CPU cores - 2)
   - Use async data loading
   - Enable CUDA graph optimization
   - Profile with small datasets first

3. **Loihi Deployment**
   - Pre-quantize weights during training
   - Batch Loihi measurements
   - Monitor core utilization
   - Use energy-aware compilation

## Monitoring Tools

1. **Training Metrics**
   - GPU utilization: nvidia-smi
   - Memory usage: torch.cuda.memory_summary()
   - Batch timing: built-in profiler
   - Spike statistics: per-layer counters

2. **Loihi Statistics**
   - Core usage: runtime monitor
   - Energy tracking: built-in counters
   - Latency profiling: hardware timers
   - Memory mapping: compiler output

## Known Limitations

1. **GPU Memory**
   - Peak usage during weight updates
   - Spikes in mixed precision mode
   - Temporary allocations during compilation

2. **CPU Bottlenecks**
   - Data loading with many workers
   - HDF5 file operations
   - Network compilation phase

3. **Loihi Constraints**
   - 8-bit weight precision
   - Core count limitations
   - Memory per core ceiling
   - Communication overhead
