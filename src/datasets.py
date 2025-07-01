import torch  # type: ignore
import torchvision  # type: ignore
import pathlib
import numpy as np
import h5py  # type: ignore
import argparse
import json
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore

# Set fixed random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Define project root path
ROOT = pathlib.Path(__file__).resolve().parents[1]

def download_mnist():
    """Download original MNIST dataset"""
    print("Downloading MNIST training set...")
    torchvision.datasets.MNIST(
        root=ROOT / "data/raw_mnist",
        train=True, download=True)
    
    print("Downloading MNIST test set...")
    torchvision.datasets.MNIST(
        root=ROOT / "data/raw_mnist",
        train=False, download=True)

def analog_sequence(img: torch.Tensor,
                    duration_ms: int,
                    frame_rate: int,
                    normalize=True):
    """
    Returns tensor shape [T, 1, 28, 28] where
    T = duration_ms * frame_rate / 1000
    """
    if normalize: 
        img = img.float() / 255.0          # 0–1 current
    T = int(round(duration_ms * frame_rate / 1000))
    seq = img.expand(T, 1, 28, 28).clone()           # copy by view
    return seq

def split_indices(n_total, n_train, n_val):
    """Create deterministic train/val/test splits"""
    idx = np.arange(n_total)
    rng = np.random.default_rng(1234)
    rng.shuffle(idx)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def build_split(imgs, labels, classes, frame_rate, dur, out_h5):
    """Generate sequences for a specific task and save to H5 files"""
    # Filter images for specific classes
    keep = [i for i, l in enumerate(labels) if l in classes]
    imgs = imgs[keep]
    labels = labels[keep]
    
    # Create train/val/test splits
    tr, val, ts = split_indices(len(imgs), 50000, 10000)
    
    # Create output directory if it doesn't exist
    out_h5.mkdir(parents=True, exist_ok=True)
    
    for tag, subset in zip(['train', 'val', 'test'], [tr, val, ts]):
        print(f"Generating {tag} set...")
        with h5py.File(out_h5 / f"{tag}.h5", "w") as f:
            # Add metadata
            f.attrs['frame_rate'] = frame_rate
            f.attrs['duration_ms'] = dur
            f.attrs['generator'] = "lt-gates v1.0"
            f.attrs['classes'] = classes
            
            grp = f.create_group(tag)
            for k, i in tqdm(enumerate(subset[:min(len(subset), 70000)])):
                seq = analog_sequence(imgs[i], dur, frame_rate)
                grp.create_dataset(str(k), data=seq, compression="gzip")
                grp[str(k)].attrs['label'] = int(labels[i])

def create_manifest(fast_classes, fast_rate, fast_frames,
                   slow_classes, slow_rate, slow_frames):
    """Create a JSON manifest file with dataset metadata"""
    manifest = {
        "fast": {"classes": fast_classes, "rate_hz": fast_rate, "frames": fast_frames},
        "slow": {"classes": slow_classes, "rate_hz": slow_rate, "frames": slow_frames},
        "seed": 1234,
        "split": "50k/10k/10k per task"
    }
    
    with open(ROOT / "data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("Created manifest.json")

def main():
    parser = argparse.ArgumentParser(description="Temporal-Split Analog-MNIST dataset generation")
    parser.add_argument("--download_only", action="store_true", help="Only download MNIST dataset")
    parser.add_argument("--build_fast", action="store_true", help="Build fast dataset")
    parser.add_argument("--build_slow", action="store_true", help="Build slow dataset")
    parser.add_argument("--fast_classes", nargs="+", type=int, default=[0, 1, 2, 3, 4], 
                        help="Classes for fast dataset")
    parser.add_argument("--slow_classes", nargs="+", type=int, default=[5, 6, 7, 8, 9], 
                        help="Classes for slow dataset")
    parser.add_argument("--fast_rate", type=int, default=1000, help="Frame rate for fast dataset (Hz)")
    parser.add_argument("--slow_rate", type=int, default=50, help="Frame rate for slow dataset (Hz)")
    parser.add_argument("--fast_dur", type=int, default=200, help="Duration for fast dataset (ms)")
    parser.add_argument("--slow_dur", type=int, default=400, help="Duration for slow dataset (ms)")
    parser.add_argument("--out", type=str, default=str(ROOT / "data"), 
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Handle download_only flag
    if args.download_only:
        download_mnist()
        return
    
    # Load MNIST dataset
    mnist_train = torchvision.datasets.MNIST(root=ROOT / "data/raw_mnist", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=ROOT / "data/raw_mnist", train=False, download=True)
    
    # Combine train and test for our custom splitting
    all_images = torch.cat([torch.tensor(mnist_train.data), torch.tensor(mnist_test.data)])
    all_labels = torch.cat([torch.tensor(mnist_train.targets), torch.tensor(mnist_test.targets)])
    
    # Build fast dataset
    if args.build_fast:
        print(f"Building fast dataset with classes {args.fast_classes}, {args.fast_rate}Hz")
        fast_out_dir = pathlib.Path(args.out) / "fast"
        build_split(all_images, all_labels, args.fast_classes, 
                   args.fast_rate, args.fast_dur, fast_out_dir)
    
    # Build slow dataset
    if args.build_slow:
        print(f"Building slow dataset with classes {args.slow_classes}, {args.slow_rate}Hz")
        slow_out_dir = pathlib.Path(args.out) / "slow"
        build_split(all_images, all_labels, args.slow_classes, 
                   args.slow_rate, args.slow_dur, slow_out_dir)
    
    # Create manifest if both datasets are built
    if args.build_fast and args.build_slow:
        fast_frames = int(round(args.fast_dur * args.fast_rate / 1000))
        slow_frames = int(round(args.slow_dur * args.slow_rate / 1000))
        create_manifest(args.fast_classes, args.fast_rate, fast_frames,
                       args.slow_classes, args.slow_rate, slow_frames)

# Simple test functions for data integrity
def test_shapes():
    """Test that all sequences have the expected shape"""
    for task in ['fast', 'slow']:
        for tag in ['train', 'val', 'test']:
            h5_path = ROOT / f"data/{task}/{tag}.h5"
            if not h5_path.exists():
                continue
            
            with h5py.File(h5_path, 'r') as f:
                ds = f[f"{tag}/0"]
                print(f"{task}/{tag} shape: {ds.shape}")
                assert ds.shape[1:] == (1, 28, 28), f"Invalid shape for {task}/{tag}"

def test_class_overlap():
    """Test that fast and slow classes don't overlap"""
    fast_lbls, slow_lbls = set(), set()
    
    # Check if files exist first
    if not (ROOT / "data/fast/train.h5").exists() or not (ROOT / "data/slow/train.h5").exists():
        print("Both fast and slow datasets must exist to test class overlap")
        return
    
    # Get labels from fast dataset
    with h5py.File(ROOT / "data/fast/train.h5", 'r') as f:
        group = f['train']
        for key in group.keys():
            fast_lbls.add(group[key].attrs['label'])
    
    # Get labels from slow dataset
    with h5py.File(ROOT / "data/slow/train.h5", 'r') as f:
        group = f['train']
        for key in group.keys():
            slow_lbls.add(group[key].attrs['label'])
    
    print(f"Fast labels: {fast_lbls}")
    print(f"Slow labels: {slow_lbls}")
    assert fast_lbls.isdisjoint(slow_lbls), "Class overlap detected between fast and slow datasets!"
    print("No class overlap detected. Datasets are properly separated.")

def visual_sanity_check():
    """Visual sanity check to ensure data looks correct"""
    try:
        # Try to load first sample from fast dataset
        with h5py.File(ROOT / "data/fast/train.h5", 'r') as f:
            seq = torch.tensor(f['train/0'][()])
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"Fast Dataset: {f['train/0'].attrs['label']}")
            plt.imshow(seq[0, 0], cmap='gray')
            
            # Try to load first sample from slow dataset
            with h5py.File(ROOT / "data/slow/train.h5", 'r') as f2:
                seq2 = torch.tensor(f2['train/0'][()])
                plt.subplot(1, 2, 2)
                plt.title(f"Slow Dataset: {f2['train/0'].attrs['label']}")
                plt.imshow(seq2[0, 0], cmap='gray')
            
            plt.tight_layout()
            plt.savefig(ROOT / "data/sample_visualization.png")
            print(f"Visualization saved to {ROOT / 'data/sample_visualization.png'}")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
