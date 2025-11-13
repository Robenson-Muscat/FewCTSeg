import gc
import torch

def free_gpu():
    """After deleting huge variables, free collect garbage and CUDA cache"""
    
    torch.cuda.synchronize()
    
    gc.collect()  
    torch.cuda.empty_cache() 


