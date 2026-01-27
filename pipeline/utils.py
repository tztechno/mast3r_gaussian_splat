# utils.py
import gc
import torch
import psutil

def clear_memory():
    """Aggressively clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    cpu_mem = psutil.virtual_memory().percent
    print(f"CPU Memory Usage: {cpu_mem:.1f}%")
