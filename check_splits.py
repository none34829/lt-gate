# check_splits.py
from pathlib import Path
import json
import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("Missing dependency: pip install h5py")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def summarize_h5(path: Path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"Available keys in {path}: {keys}")
        
        # The files seem to have a nested structure with 'train', 'val', 'test' as top-level keys
        # Let's explore the first key to see what's inside
        if keys:
            first_key = keys[0]
            print(f"Exploring key '{first_key}' in {path}")
            
            # Check if it's a group (has subgroups) or dataset
            if isinstance(f[first_key], h5py.Group):
                subgroup_keys = list(f[first_key].keys())
                print(f"  Subgroup keys: {subgroup_keys}")
                
                # Look for data and labels in subgroups
                data_key = None
                label_key = None
                
                for subkey in subgroup_keys:
                    if subkey in ['x', 'data', 'images', 'features', 'input']:
                        data_key = f"{first_key}/{subkey}"
                    elif subkey in ['y', 'labels', 'targets', 'classes']:
                        label_key = f"{first_key}/{subkey}"
                
                if data_key and label_key:
                    x = f[data_key]
                    y = f[label_key]
                else:
                    print(f"  Could not find data/label keys in subgroups: {subgroup_keys}")
                    return {
                        "path": str(path),
                        "keys": keys,
                        "subgroup_keys": subgroup_keys,
                        "error": "No data/label keys found in subgroups"
                    }
            else:
                # It's a dataset, let's see what it contains
                dataset = f[first_key]
                print(f"  Dataset shape: {dataset.shape}")
                print(f"  Dataset dtype: {dataset.dtype}")
                
                # If it's a single dataset, it might contain both data and labels
                # Let's assume it's data for now
                x = dataset
                y = None  # No separate labels
                
                return {
                    "path": str(path),
                    "keys": keys,
                    "dataset_shape": tuple(x.shape),
                    "dataset_dtype": str(x.dtype),
                    "num_samples": int(x.shape[0]),
                    "seq_len": int(x.shape[1]) if len(x.shape) > 1 else None,
                    "channels": int(x.shape[2]) if len(x.shape) > 2 else None,
                    "height": int(x.shape[3]) if len(x.shape) > 3 else None,
                    "width": int(x.shape[4]) if len(x.shape) > 4 else None,
                    "x_min": float(np.min(x[:10])),
                    "x_max": float(np.max(x[:10])),
                    "has_nan": bool(np.isnan(x[:10]).any()),
                    "note": "Single dataset, no separate labels"
                }
        
        if y is not None:
            x_shape = tuple(x.shape)
            y_shape = tuple(y.shape)
            labels = np.unique(y[:]).tolist()

            stats = {
                "path": str(path),
                "keys": keys,
                "data_key": data_key,
                "label_key": label_key,
                "x_shape": x_shape,
                "y_shape": y_shape,
                "label_set": labels,
                "num_samples": int(x_shape[0]),
                "seq_len": int(x_shape[1]) if len(x_shape) > 1 else None,
                "channels": int(x_shape[2]) if len(x_shape) > 2 else None,
                "height": int(x_shape[3]) if len(x_shape) > 3 else None,
                "width": int(x_shape[4]) if len(x_shape) > 4 else None,
                "x_min": float(np.min(x[:10])),  # sample a few to be quick
                "x_max": float(np.max(x[:10])),
                "has_nan": bool(np.isnan(x[:10]).any()),
            }
            return stats
        else:
            return {
                "path": str(path),
                "keys": keys,
                "error": "Could not determine data structure"
            }

def pretty(d):
    return json.dumps(d, indent=2)

def main():
    manifest = {}
    man_path = DATA / "manifest.json"
    if man_path.exists():
        try:
            manifest = json.load(open(man_path, "r"))
        except Exception:
            pass

    results = {}
    for split in ["fast", "slow"]:
        split_dir = DATA / split
        for part in ["train", "val", "test"]:
            h5 = split_dir / f"{part}.h5"
            if not h5.exists():
                print(f"[MISSING] {h5}")
                continue
            stats = summarize_h5(h5)
            results[f"{split}/{part}"] = stats

    # Print summary
    print("\n=== DATA SUMMARY ===")
    for k, v in results.items():
        print(f"\n[{k}]")
        print(pretty(v))

    # Quick sanity checks
    print("\n=== QUICK CHECKS ===")
    fast_labels = set(results.get("fast/train", {}).get("label_set", []))
    slow_labels = set(results.get("slow/train", {}).get("label_set", []))
    ok_fast = fast_labels.issubset(set(range(0,5)))
    ok_slow = slow_labels.issubset(set(range(5,10)))

    print(f"Fast labels subset of {{0..4}}: {ok_fast} -> {sorted(fast_labels)}")
    print(f"Slow labels subset of {{5..9}}: {ok_slow} -> {sorted(slow_labels)}")

    # Compare counts to manifest if present
    if manifest:
        print("\n=== MANIFEST CHECK ===")
        for k in ["fast", "slow"]:
            for part in ["train", "val", "test"]:
                key = f"{k}/{part}"
                cnt = results.get(key, {}).get("num_samples")
                man_cnt = manifest.get(k, {}).get(part)
                if cnt is not None and man_cnt is not None:
                    print(f"{key}: data={cnt}, manifest={man_cnt}, match={cnt==man_cnt}")

if __name__ == "__main__":
    main()
