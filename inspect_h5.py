import h5py
import sys

def inspect_h5(filename):
    print(f"\nInspecting file: {filename}")
    with h5py.File(filename, 'r') as f:
        print("Root keys:", list(f.keys()))
        
        # Inspect the 'train' group
        if 'train' in f:
            train_group = f['train']
            print("\nTrain group keys count:", len(list(train_group.keys())))
            print("First 5 keys in train group:", list(train_group.keys())[:5])
            
            # Inspect the first item
            first_key = list(train_group.keys())[0]
            first_item = train_group[first_key]
            print(f"\nFirst item (key={first_key}):\n  Type: {type(first_item)}")
            
            if isinstance(first_item, h5py.Dataset):
                print(f"  Shape: {first_item.shape}")
                print(f"  Dtype: {first_item.dtype}")
                try:
                    print(f"  First few values: {first_item[...]}")
                except Exception as e:
                    print(f"  Could not display values: {e}")
            else:
                print(f"  Content: {list(first_item.keys()) if hasattr(first_item, 'keys') else 'Cannot display'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_h5.py <h5_file_path>")
        sys.exit(1)
        
    inspect_h5(sys.argv[1])
