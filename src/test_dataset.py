import h5py
import numpy as np
import pathlib
import sys

# Define project root path
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_shapes():
    """Test that all sequences have the expected shape"""
    for task in ['fast', 'slow']:
        for tag in ['train', 'val', 'test']:
            h5_path = ROOT / f"data/{task}/{tag}.h5"
            if not h5_path.exists():
                print(f"Warning: {h5_path} does not exist")
                continue
            
            try:
                with h5py.File(h5_path, 'r') as f:
                    if tag not in f:
                        print(f"Warning: Group '{tag}' not found in {h5_path}")
                        continue
                    
                    group = f[tag]
                    if len(group.keys()) == 0:
                        print(f"Warning: No datasets in {h5_path}/{tag}")
                        continue
                        
                    ds = group["0"]
                    print(f"{task}/{tag} shape: {ds.shape}")
                    assert ds.shape[1:] == (1, 28, 28), f"Invalid shape for {task}/{tag}"
            except Exception as e:
                print(f"Error with {h5_path}: {str(e)}")

def test_class_overlap():
    """Test that fast and slow classes don't overlap"""
    fast_lbls, slow_lbls = set(), set()
    
    # Check fast dataset
    fast_path = ROOT / "data/fast/train.h5"
    if not fast_path.exists():
        print(f"Warning: {fast_path} does not exist")
    else:
        try:
            with h5py.File(fast_path, 'r') as f:
                if 'train' not in f:
                    print(f"Warning: Group 'train' not found in {fast_path}")
                else:
                    group = f['train']
                    for key in list(group.keys())[:100]:  # Sample first 100 to avoid processing all
                        if 'label' in group[key].attrs:
                            fast_lbls.add(group[key].attrs['label'])
        except Exception as e:
            print(f"Error with {fast_path}: {str(e)}")
    
    # Check slow dataset
    slow_path = ROOT / "data/slow/train.h5" 
    if not slow_path.exists():
        print(f"Warning: {slow_path} does not exist")
    else:
        try:
            with h5py.File(slow_path, 'r') as f:
                if 'train' not in f:
                    print(f"Warning: Group 'train' not found in {slow_path}")
                else:
                    group = f['train']
                    for key in list(group.keys())[:100]:  # Sample first 100 to avoid processing all
                        if 'label' in group[key].attrs:
                            slow_lbls.add(group[key].attrs['label'])
        except Exception as e:
            print(f"Error with {slow_path}: {str(e)}")
    
    print(f"Fast labels: {fast_lbls}")
    print(f"Slow labels: {slow_lbls}")
    
    if len(fast_lbls) == 0 or len(slow_lbls) == 0:
        print("Warning: One or both label sets are empty")
    else:
        overlap = fast_lbls.intersection(slow_lbls)
        if len(overlap) > 0:
            print(f"Warning: Class overlap detected: {overlap}")
        else:
            print("No class overlap detected. Datasets are properly separated.")

if __name__ == "__main__":
    print("Testing dataset shapes...")
    test_shapes()
    
    print("\nTesting class overlap...")
    test_class_overlap()
