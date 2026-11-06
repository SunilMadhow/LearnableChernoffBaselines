import os
import glob
import torch
# Import the function from your file
from nets_language import train_critic

def run_smoke_test():
    print("=== CRITIC TRAINING SMOKE TEST ===")

    # 1. Device Setup (Force CPU/MPS for Mac safety)
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # 2. Auto-discover the latest run from the Collector
    base_dir = "data_runs"
    if not os.path.exists(base_dir):
        print(f"❌ Error: '{base_dir}' does not exist. Run the collector smoke test first!")
        return

    # Sort by time, newest first
    all_runs = sorted(glob.glob(os.path.join(base_dir, "*")), key=os.path.getmtime, reverse=True)
    
    # Filter for directories that actually contain .npy files
    valid_runs = [r for r in all_runs if os.path.exists(os.path.join(r, "meta.json")) or len(glob.glob(os.path.join(r, "*.npy"))) > 0]

    if not valid_runs:
        print("❌ Error: No valid run folders found in 'data_runs'.")
        return
    
    latest_run_path = valid_runs[0]
    run_id = os.path.basename(latest_run_path)
    print(f"Targeting Run ID: {run_id}")

    # 3. Execute train_critic with minimal params
    try:
        print("--- Starting Training ---")
        train_critic(
            run_id=run_id,
            data_dir=base_dir,
            epochs=1,          # Smoke test: 1 epoch is enough
            batch_size=2,      # Smoke test: Tiny batch to prevent OOM
            lr=1e-4,
            device=device
        )
        print("\n✅ SMOKE TEST PASSED")
        print(f"Checkpoint saved in: {latest_run_path}")

    except Exception as e:
        print("\n❌ SMOKE TEST FAILED")
        print(e)
        
        # specific debug hint for the tensor mismatch we solved earlier
        if "size mismatch" in str(e) or "index" in str(e):
             print("\n[Hint] Check if nets_language.py is correctly translating LLaDA tokens to RoBERTa tokens.")

if __name__ == "__main__":
    run_smoke_test()