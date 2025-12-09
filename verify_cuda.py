import torch
import sys

def verify():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\nSUCCESS: CUDA is available!")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test allocation
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
            print(f"\nTest Tensor allocated on: {x.device}")
            print("GPU is working correctly.")
        except Exception as e:
            print(f"\nError allocating tensor on GPU: {e}")
    else:
        print("\nFAILURE: CUDA is NOT available.")
        print("Please ensure you installed the correct version of PyTorch.")

if __name__ == "__main__":
    verify()
