import os
import sys

def main():
    print("=== Training Ensemble Model (MobileNetV3-Large) ===")
    ret1 = os.system("python train.py --model-type large")
    if ret1 != 0:
        print("Large model training failed.")
        sys.exit(ret1)
        
    print("\n=== Training Ensemble Model (MobileNetV3-Small) ===")
    ret2 = os.system("python train.py --model-type small")
    if ret2 != 0:
        print("Small model training failed.")
        sys.exit(ret2)
        
    print("\n[OK] Ensemble training successfully completed!")

if __name__ == "__main__":
    main()
