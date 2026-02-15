import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
import numpy as np

import os

# --- HARDCODED FIX FOR WINDOWS CUDA ---
# We force the script to look in your v13.1 folder
os.environ["PATH"] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
# --------------------------------------

# ... (rest of the file continues below)

# Ensure we can import your model
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from model import ChessNet
    from config import Config
except ImportError:
    print("❌ Could not import ChessNet. Run this from the folder containing 'src'.")
    sys.exit()

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # Small batch to ensure it fits in both modes
STEPS = 50       # How many training steps to simulate
WARMUP = 5       # Steps to ignore (compilation time, cuda initialization)

def get_dummy_batch():
    # Create fake inputs (Batch, 18, 8, 8)
    data = torch.randn(BATCH_SIZE, 18, 8, 8, device=DEVICE)
    # Fake targets
    target_p = torch.randn(BATCH_SIZE, 4096, device=DEVICE) # Policy target
    target_v = torch.randn(BATCH_SIZE, 1, device=DEVICE)    # Value target
    return data, target_p, target_v

def run_benchmark(enable_amp, enable_jit, name):
    print(f"\n--- Running: {name} ---")
    
    # 1. Reset Model
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if enable_amp else None
    
    # 2. Compile if requested
    if enable_jit:
        print("   Compiling model (this takes a moment)...")
        # 'reduce-overhead' is the best mode for small batches/training loops
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"   ⚠️ JIT Compilation Warning: {e}")
            print("   Falling back to default compilation...")
            model = torch.compile(model)

    model.train()
    
    # 3. Warmup (don't measure this)
    # The first few steps are always slow due to memory allocation/compilation
    print("   Warming up...")
    for _ in range(WARMUP):
        data, tp, tv = get_dummy_batch()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=enable_amp):
            p, v = model(data)
            loss = F.mse_loss(p, tp) + F.mse_loss(v, tv)
        
        if enable_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    # 4. The Real Test
    torch.cuda.synchronize() # Wait for GPU to finish warmup
    start_time = time.time()
    
    for i in range(STEPS):
        data, tp, tv = get_dummy_batch()
        optimizer.zero_grad()
        
        # --- FORWARD PASS ---
        with torch.amp.autocast('cuda', enabled=enable_amp):
            p, v = model(data)
            loss = F.mse_loss(p, tp) + F.mse_loss(v, tv)
        
        # --- BACKWARD PASS ---
        if enable_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
    torch.cuda.synchronize() # Wait for GPU to finish all tasks
    end_time = time.time()
    
    total_time = end_time - start_time
    sps = STEPS / total_time
    print(f"   Time: {total_time:.2f}s | Speed: {sps:.2f} steps/sec")
    
    # Peak Memory Check
    mem_bytes = torch.cuda.max_memory_allocated()
    mem_mb = mem_bytes / 1024 / 1024
    print(f"   Peak Memory: {mem_mb:.2f} MB")
    
    return sps, mem_mb

if __name__ == "__main__":
    print(f"🔥 BENCHMARKING ON {DEVICE} 🔥")
    print(f"Model: {Config.num_residual_blocks} Blocks | {Config.num_filters} Filters")
    
    # Test 1: Baseline (Slow)
    torch.cuda.reset_peak_memory_stats()
    sps_base, mem_base = run_benchmark(enable_amp=False, enable_jit=False, name="BASELINE (No AMP, No JIT)")
    
    # Test 2: Fully Optimized (Fast)
    torch.cuda.reset_peak_memory_stats()
    # We set JIT to True here to test your configuration
    sps_opt, mem_opt = run_benchmark(enable_amp=True, enable_jit=True, name="OPTIMIZED (AMP + JIT)")
    
    # Results
    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    
    speedup = sps_opt / sps_base
    mem_saved = (1 - (mem_opt / mem_base)) * 100
    
    print(f"Baseline Speed:  {sps_base:.2f} steps/sec")
    print(f"Optimized Speed: {sps_opt:.2f} steps/sec")
    print(f"🚀 SPEEDUP:      {speedup:.2f}x FASTER")
    print("-" * 30)
    print(f"Baseline Mem:    {mem_base:.2f} MB")
    print(f"Optimized Mem:   {mem_opt:.2f} MB")
    print(f"💾 MEMORY SAVED: {mem_saved:.1f}%")