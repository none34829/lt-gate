"""
H5 sequence dataset loader for LT-Gate training.
"""

import h5py  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset, IterableDataset  # type: ignore
import numpy as np


class H5SequenceDataset(IterableDataset):
    """
    Memory-efficient dataset for loading sequences from H5 files.
    Implements IterableDataset for streaming access to large datasets.
    """
    
    def __init__(self, h5_path, transform=None, limit_samples=None):
        """
        Initialize H5 sequence dataset.
        
        Args:
            h5_path (str): Path to H5 file containing sequences
            transform (callable, optional): Optional transform to be applied on a sample
            limit_samples (int, optional): Limit number of samples (for debugging)
        """
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.limit_samples = limit_samples
        
        # Get dataset info
        with h5py.File(h5_path, 'r') as f:
            # Get metadata
            self.frame_rate = f.attrs['frame_rate']
            self.duration_ms = f.attrs['duration_ms']
            self.classes = f.attrs['classes']
            
            # Get group name (train/val/test)
            self.group_name = list(f.keys())[0]
            
            # Count total samples
            self.total_samples = len(f[self.group_name].keys())
            if limit_samples:
                self.total_samples = min(self.total_samples, limit_samples)
    
    def __iter__(self):
        """
        Iterator for loading sequences.
        Yields (sequence, label) tuples.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # Open H5 file for this worker
        f = h5py.File(self.h5_path, 'r')
        group = f[self.group_name]
        
        # Get sample indices for this worker
        if worker_info is None:
            # Single-process loading
            start_idx = 0
            end_idx = self.total_samples
        else:
            # Multi-process loading - split workload
            per_worker = int(np.ceil(self.total_samples / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.total_samples)
        
        # Yield sequences for this worker's range
        try:
            for idx in range(start_idx, end_idx):
                # Get sequence and label
                seq_data = group[str(idx)][:]
                label = group[str(idx)].attrs['label']
                
                # Convert to torch tensors
                seq = torch.from_numpy(seq_data).float()
                label = torch.tensor(label, dtype=torch.long)
                
                # Apply transform if specified
                if self.transform is not None:
                    seq = self.transform(seq)
                
                yield seq, label
        
        finally:
            # Clean up
            f.close()
    
    def __len__(self):
        """Return total number of samples."""
        return self.total_samples
