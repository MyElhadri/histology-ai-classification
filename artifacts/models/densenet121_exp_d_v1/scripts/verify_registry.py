import os
import sys
import hashlib
from pathlib import Path

# Setup root automatically relative to this script
REGISTRY_ROOT = Path(__file__).resolve().parent.parent

def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def main():
    print(f"INFO: Registry root located at {REGISTRY_ROOT}")
    
    # 1. Read and verify checksums
    checksum_file = REGISTRY_ROOT / "metadata" / "checksums_sha256.txt"
    if not checksum_file.is_file():
        print(f"ERROR: Checksums file not found at {checksum_file}")
        sys.exit(1)
        
    print("INFO: Verifying SHA-256 checksums...")
    with open(checksum_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            print(f"ERROR: Invalid checksum line format: {line}")
            sys.exit(1)
        expected_sha, rel_path = parts
        
        file_path = REGISTRY_ROOT / rel_path
        if not file_path.is_file():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
            
        actual_sha = sha256sum(file_path)
        if actual_sha != expected_sha:
            print(f"ERROR: Checksum mismatch for {rel_path}!")
            print(f"  Expected: {expected_sha}")
            print(f"  Actual:   {actual_sha}")
            sys.exit(1)
        else:
            print(f"  [OK] {rel_path} (size: {file_path.stat().st_size} bytes)")

    # 2. Check the checkpoints presence
    for fold in range(5):
        ckpt_path = REGISTRY_ROOT / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        if not ckpt_path.is_file():
            print(f"ERROR: Checkpoint for fold {fold} is missing at {ckpt_path}")
            sys.exit(1)

    # 3. Load each model and check output shape
    # Importing TensorFlow locally inside main to keep script fast if TensorFlow is slow to import
    print("INFO: Importing TensorFlow...")
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is not installed in the current environment!")
        sys.exit(1)
        
    for fold in range(5):
        ckpt_path = REGISTRY_ROOT / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        print(f"INFO: Loading checkpoint for fold {fold} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)...")
        try:
            model = tf.keras.models.load_model(ckpt_path, compile=False)
            output_shape = model.output_shape
            print(f"  [OK] Loaded. Output shape: {output_shape}")
            if output_shape != (None, 22):
                print(f"ERROR: Expected output shape (None, 22), got {output_shape}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint for fold {fold}: {e}")
            sys.exit(1)
        finally:
            tf.keras.backend.clear_session()
            
    print("\nDENSENET121 EXP D REGISTRY VALID")
    sys.exit(0)

if __name__ == "__main__":
    main()
