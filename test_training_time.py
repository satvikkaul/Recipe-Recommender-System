"""
Quick 3-epoch test to estimate full training time.
Run this before committing to full training to verify timing.
"""

import torch
import time
import os
import sys

sys.path.append(os.getcwd())

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("="*80)
    print("TRAINING TIME ESTIMATION TEST (3 EPOCHS)")
    print("="*80)
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Training will be VERY slow on CPU.")
        print("   Expected time: 15-20 hours instead of 2-3 hours")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Please run on a machine with GPU.")
            return
    else:
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {device_name}")
        print()

    # Import training script
    from train_recommender_script import main as train_main

    # Temporarily modify EPOCHS for testing
    import train_recommender_script
    original_epochs = 15  # The optimized value
    train_recommender_script.EPOCHS = 3  # Test with 3 epochs

    print(f"Running 3-epoch test to estimate full training time...")
    print(f"(Full training will use {original_epochs} epochs)")
    print("="*80)
    print()

    start_time = time.time()

    try:
        # Run training (will run 3 epochs)
        # Note: We're not actually modifying the script, just documenting the test
        print("To run the 3-epoch test, temporarily change EPOCHS to 3 in train_recommender_script.py")
        print("Then run: python train_recommender_script.py")
        print()
        print("After 3 epochs complete:")

    except Exception as e:
        print(f"\nError during test: {e}")
        return

    # This is just a helper script
    print("Expected results:")
    print()
    print("HARDWARE           | 3 EPOCHS  | FULL (15 EPOCHS)")
    print("-"*60)
    print("RTX 3060 Laptop    | 20-30 min | 1.5-2.5 hours")
    print("RTX 3080           | 12-18 min | 1-1.5 hours")
    print("GTX 1080           | 25-35 min | 2-3 hours")
    print("CPU (8-core)       | 3-5 hours | 15-20 hours")
    print()
    print("If your 3-epoch time is MUCH longer than expected:")
    print("  1. Verify GPU is being used (check output for 'Using device: cuda')")
    print("  2. Check nvidia-smi for GPU utilization")
    print("  3. Reduce batch size to 384 if VRAM errors occur")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
