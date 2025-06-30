import torch
import torchvision
import pathlib
import numpy as np
import h5py
from tqdm import tqdm

# Set fixed random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Define project root path
ROOT = pathlib.Path(__file__).resolve().parents[1]

def analog_sequence(img: torch.Tensor, duration_ms: int, frame_rate: int, normalize=True):
    """Convert image to analog sequence with specific frame rate"""
    if normalize: 
        img = img.float() / 255.0
    T = int(round(duration_ms * frame_rate / 1000))
    seq = img.expand(T, 1, 28, 28).clone()
    return seq

def generate_dataset(task_name, digit_classes, frame_rate, duration_ms):
    print(f"Generating {task_name} dataset (classes {digit_classes}, {frame_rate}Hz, {duration_ms}ms)")
    
    # Load MNIST dataset
    mnist_train = torchvision.datasets.MNIST(root=ROOT / "data/raw_mnist", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=ROOT / "data/raw_mnist", train=False, download=True)
    
    # Filter by classes
    train_mask = torch.tensor([label in digit_classes for label in mnist_train.targets])
    test_mask = torch.tensor([label in digit_classes for label in mnist_test.targets])
    
    # Apply masks to get only the classes we want
    train_filtered_images = mnist_train.data[train_mask]
    train_filtered_labels = mnist_train.targets[train_mask]
    test_filtered_images = mnist_test.data[test_mask]
    test_filtered_labels = mnist_test.targets[test_mask]
    
    # Ensure output directories exist
    out_dir = ROOT / f"data/{task_name}"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate expected sequence length
    frame_count = int(round(duration_ms * frame_rate / 1000))
    print(f"Sequence length: {frame_count} frames")
    
    # Split training data into train and validation
    total_train = len(train_filtered_images)
    indices = torch.randperm(total_train)
    train_size = int(total_train * 0.8)  # 80% for training
    val_size = total_train - train_size   # 20% for validation
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Save train set
    print(f"Creating train set ({len(train_indices)} samples)...")
    with h5py.File(out_dir / "train.h5", "w") as f:
        f.attrs['frame_rate'] = frame_rate
        f.attrs['duration_ms'] = duration_ms
        f.attrs['generator'] = "lt-gates v1.0"
        f.attrs['classes'] = digit_classes
        
        train_grp = f.create_group("train")
        for i, idx in enumerate(tqdm(train_indices)):
            img = train_filtered_images[idx]
            label = train_filtered_labels[idx].item()
            seq = analog_sequence(img, duration_ms, frame_rate)
            train_grp.create_dataset(str(i), data=seq, compression="gzip")
            train_grp[str(i)].attrs['label'] = label
    
    # Save validation set
    print(f"Creating validation set ({len(val_indices)} samples)...")
    with h5py.File(out_dir / "val.h5", "w") as f:
        f.attrs['frame_rate'] = frame_rate
        f.attrs['duration_ms'] = duration_ms
        f.attrs['generator'] = "lt-gates v1.0"
        f.attrs['classes'] = digit_classes
        
        val_grp = f.create_group("val")
        for i, idx in enumerate(tqdm(val_indices)):
            img = train_filtered_images[idx]
            label = train_filtered_labels[idx].item()
            seq = analog_sequence(img, duration_ms, frame_rate)
            val_grp.create_dataset(str(i), data=seq, compression="gzip")
            val_grp[str(i)].attrs['label'] = label
    
    # Save test set
    print(f"Creating test set ({len(test_filtered_images)} samples)...")
    with h5py.File(out_dir / "test.h5", "w") as f:
        f.attrs['frame_rate'] = frame_rate
        f.attrs['duration_ms'] = duration_ms
        f.attrs['generator'] = "lt-gates v1.0"
        f.attrs['classes'] = digit_classes
        
        test_grp = f.create_group("test")
        for i in tqdm(range(len(test_filtered_images))):
            img = test_filtered_images[i]
            label = test_filtered_labels[i].item()
            seq = analog_sequence(img, duration_ms, frame_rate)
            test_grp.create_dataset(str(i), data=seq, compression="gzip")
            test_grp[str(i)].attrs['label'] = label
    
    print(f"Finished generating {task_name} dataset")

def main():
    # Generate fast dataset (classes 0-4, 1000Hz, 200ms)
    generate_dataset("fast", [0, 1, 2, 3, 4], 1000, 200)
    
    # Generate slow dataset (classes 5-9, 50Hz, 400ms)
    generate_dataset("slow", [5, 6, 7, 8, 9], 50, 400)
    
    # Create manifest file
    import json
    manifest = {
        "fast": {"classes": [0, 1, 2, 3, 4], "rate_hz": 1000, "frames": 200},
        "slow": {"classes": [5, 6, 7, 8, 9], "rate_hz": 50, "frames": 20},
        "seed": 1234,
        "split": "80% train, 20% val per task + original test split"
    }
    
    with open(ROOT / "data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("Created manifest.json")

if __name__ == "__main__":
    main()
