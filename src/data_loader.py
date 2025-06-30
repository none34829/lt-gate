#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset loading utilities for LT-Gate experiments.

This module provides functions for loading and processing the analog MNIST
datasets with fast and slow time constants, along with utilities for creating
dataset splits and DataLoader objects.
"""

import os
import enum
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Define project root path
ROOT = Path(__file__).resolve().parents[1]


class DatasetSplit(enum.Enum):
    """
    Dataset split types.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AnalogMNISTDataset(Dataset):
    """
    Dataset for analog MNIST sequences.
    
    Args:
        dataset_name (str): Name of dataset to load ('fast' or 'slow')
        split (DatasetSplit): Which split to use (train, val, or test)
        transform (callable, optional): Optional transform to be applied on a sample
        max_samples (int, optional): Maximum number of samples to load (for debugging)
        verbose (bool): Whether to print verbose output
    """
    def __init__(self, dataset_name, split, transform=None, max_samples=None, verbose=False):
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.verbose = verbose
        
        # Load data from HDF5 file
        self.data_dir = ROOT / "data"
        self.split_name = split.value  # 'train', 'val', or 'test'
        
        # The file structure is data/{dataset_name}/{split}.h5
        self.file_path = self.data_dir / dataset_name / f"{self.split_name}.h5"
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file {self.file_path} not found. Make sure you've generated the dataset splits.")
        
        # Open the file but don't load all data yet - store file handle for lazy loading
        if self.verbose:
            print(f"Opening dataset file: {self.file_path}")
        
        self.h5_file = h5py.File(self.file_path, 'r')
        self.group = self.h5_file[self.split_name]
        
        # Get all sample keys and sort them for consistent ordering
        all_sample_keys = sorted(list(self.group.keys()), key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        # Limit samples if requested (for debugging)
        if max_samples is not None:
            if self.verbose:
                print(f"Limiting to {max_samples} samples out of {len(all_sample_keys)} total samples")
            self.sample_keys = all_sample_keys[:max_samples]
        else:
            self.sample_keys = all_sample_keys
        
        # Get shape information from first sample
        first_sample = self.group[self.sample_keys[0]]
        self.time_steps, self.channels, self.height, self.width = first_sample.shape
        
        if self.verbose:
            print(f"Dataset loaded: {self.split_name} split, {len(self.sample_keys)} samples")
            print(f"Sample shape: {self.time_steps} timesteps, {self.channels} channels, {self.height}x{self.width}")
    
    def __del__(self):
        # Close the file when the dataset is deleted
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
    
    def __len__(self):
        return len(self.sample_keys)
    
    def __getitem__(self, idx):
        # Get the sample key
        key = self.sample_keys[idx]
        
        # Load the sequence data on demand
        sequence = torch.from_numpy(self.group[key][:]).float()
        
        # Use the key as the label (assuming keys are digits)
        try:
            label = int(key)
        except ValueError:
            # If key can't be converted to int, use index as placeholder
            label = idx
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label


def load_dataset(dataset_name, split, batch_size=32, shuffle=True, seed=42, max_samples=None, debug=False):
    """
    Load dataset and return a DataLoader.
    
    Args:
        dataset_name (str): Name of dataset to load ('fast', 'slow', or 'both')
        split (DatasetSplit): Which split to use
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the dataset
        seed (int): Random seed for reproducibility
        max_samples (int, optional): Maximum number of samples to load (for debugging)
        debug (bool): Whether to enable debug mode with verbose output
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # For debug mode, use fewer samples and print verbose info
    if debug:
        # Override max_samples for debug mode if not explicitly set
        if max_samples is None:
            max_samples = 100  # Use only 100 samples in debug mode
        verbose = True
    else:
        verbose = False
    
    if dataset_name == "both":
        # TODO: Implement combined dataset loading
        raise NotImplementedError("Combined dataset loading not implemented yet")
    else:
        # Load single dataset (fast or slow)
        if verbose:
            print(f"Loading {dataset_name} dataset, {split.value} split")
            if max_samples:
                print(f"Debug mode: Using max {max_samples} samples")
        
        dataset = AnalogMNISTDataset(dataset_name, split, 
                                    max_samples=max_samples,
                                    verbose=verbose)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Use main process for loading data
        pin_memory=torch.cuda.is_available(),  # Pin memory if using GPU
    )
    
    if verbose:
        print(f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}")
    
    return loader
